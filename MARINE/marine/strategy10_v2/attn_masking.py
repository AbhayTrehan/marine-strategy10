"""
Attention-level occlusion for LLaVA-style LVLMs.

WHAT THIS REPLACES, AND WHY
---------------------------
The v2.1 build realised the causal-occlusion intervention by INPAINTING: the
object's patches were overwritten with the image's mean pixel value, and that
altered image was re-encoded. `masking.py`'s own docstring records the two leaks
that forces us to fight (partial patches, resize interpolation) and the pixel-
value re-assertion added to fight them.

That whole battle is against a symptom. Painting pixels grey does not remove the
object from the computation; it replaces the object with a *different* visual
stimulus (a flat grey rectangle) and hopes the encoder treats that as "nothing".
It does not: a grey rectangle is itself a feature, mean-fill is out of the vision
encoder's training distribution, and -- the point of this file -- the ViT's own
self-attention has *already* copied the object's content into neighbouring patch
tokens before any masking downstream can act, so even a perfectly grey object
region leaves the object's information alive in the tokens *around* it, which the
LVLM still reads.

The intervention the method actually wants is: *the model does not attend to the
object's patches at all*. That is what this file implements. The image is left
untouched; instead, for the duration of one scored forward pass, attention to the
object's patch positions is removed in two places:

  (1) ViT SELF-ATTENTION. In every layer of the vision tower, the object's patch
      tokens are removed as attention *keys/values*, so no surviving patch token
      (and no CLS token) can route information out of the object region. The
      surviving patch tokens the LVLM reads are therefore computed as if the
      object patches were never there -- not as if they were grey.

  (2) LVLM ATTENTION TO IMAGE TOKENS. In the language model, the image tokens
      corresponding to the object's patches are removed as attention keys, so the
      generated/teacher-forced text tokens cannot read them directly either.

Both are exact: after occlusion, perturbing the pixels *inside* the object region
provably cannot change the model's output (there is no attention path from those
pixels to any read position). `attn_masking_selftest.py` checks exactly that
invariant on a tiny CPU model. Pixel inpainting cannot make that guarantee, which
is the whole reason `masking.py` had to *measure* its residual leak rather than
eliminate it.

MECHANISM
---------
* Locating the vision tower is layout-dependent and this is handled defensively:
  pre-refactor transformers exposes `LlavaForConditionalGeneration.vision_tower`
  directly; transformers>=5 (and some 4.5x point releases) moved the whole
  multimodal stack under an inner `LlavaModel`, so it is instead
  `LlavaForConditionalGeneration.model.vision_tower`. Both paths are tried, then
  we fall back to scanning for a submodule literally named `vision_tower` so a
  third layout fails loudly with a diagnostic rather than silently.

* The vision tower's `CLIPVisionTransformer.forward` (or, post-refactor, the
  flattened `CLIPVisionModel.forward`) calls its encoder with
  `attention_mask=None` (hard-coded; it never exposes a mask argument). We
  therefore wrap each vision-encoder self-attention module's `forward` and,
  while occlusion is armed, substitute an additive mask of shape (bsz, 1, S, S)
  whose columns for the object's ViT token indices are -inf. This additive-bias
  shape is honoured by every attention-kernel variant transformers has shipped
  for CLIP (eager, sdpa, flash, and the newer unified `ALL_ATTENTION_FUNCTIONS`
  dispatch that replaced the separate per-kernel classes), so the same wrapper
  works unmodified across that whole range; `inspect.signature` on the installed
  version's `forward` decides which of `causal_attention_mask` /
  `output_attentions` are still valid kwargs to pass through, since newer CLIP
  refactors dropped both in favour of a bare `**kwargs`.

* The language model needs no wrapping: a standard 2D `attention_mask` with 0 at
  the object image-token positions removes those positions as keys for every
  query. Crucially, LLaMA derives RoPE positions from `cache_position`
  (`arange`), NOT from the attention mask, so zeroing interior positions removes
  them as keys WITHOUT shifting any position id. The image tokens keep their
  places; they are merely unreadable.

TOKEN INDEXING
--------------
Patch index p in {0 .. g^2-1} (row-major on the g x g grid, exactly as
attribution.py produces) maps to:
  * ViT token index  p + 1   (CLS is token 0 and is never masked), and
  * the (p+1)-th image token in the LVLM sequence, i.e. the p-th position where
    input_ids == image_token_index (CLS is dropped by the "default" feature
    strategy, so the first *image token* is patch 0).
Both mappings are the identity up to the CLS offset, and both are asserted at run
time rather than assumed.
"""

from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence

import torch


# Attention-module class names we know how to drive. Membership is checked by
# name so we do not have to import every transformers version's private classes.
_VISION_ATTN_CLASSES = (
    "CLIPAttention",
    "CLIPSdpaAttention",
    "CLIPFlashAttention2",
)


def _resolve_image_token_index(config) -> int:
    """The id of the <image> placeholder token, across transformers renames.

    Older configs call it `image_token_index`; some newer ones `image_token_id`.
    """
    for attr in ("image_token_index", "image_token_id"):
        v = getattr(config, attr, None)
        if v is not None:
            return int(v)
    raise AttributeError(
        "LlavaConfig exposes neither image_token_index nor image_token_id; "
        "cannot locate the <image> placeholder token."
    )


class AttentionOcclusion:
    """Installs (once) the vision-tower attention wrappers and drives occlusion.

    Usage:

        occ = AttentionOcclusion(model, grid=24)
        mask = occ.llm_attention_mask(input_ids, attention_mask, patches)
        with occ.occlude(patches):
            out = model(input_ids=..., pixel_values=..., attention_mask=mask, ...)

    `occlude(patches)` arms the ViT wrappers for the *object's* patch set;
    `llm_attention_mask(...)` produces the matching language-model key mask. Use
    them together: one governs (1), the other governs (2). When `patches` is empty
    the wrappers are inert and the model runs bit-identically to the unwrapped one.
    """

    def __init__(self, model, grid: int, image_token_index: Optional[int] = None):
        self.model = model
        self.grid = int(grid)
        self.n_patches = self.grid * self.grid
        self.image_token_index = (
            int(image_token_index) if image_token_index is not None
            else int(_resolve_image_token_index(model.config))
        )

        # Runtime occlusion state, read by the wrappers.
        self._armed = False
        self._vit_key_mask: Optional[frozenset] = None   # ViT token indices to block

        self._wrapped = []
        self._install_vision_wrappers()

    # ------------------------------------------------------------------ #
    # installation
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_path(root, path: str):
        obj = root
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                return None
        return obj

    def _find_vision_tower(self):
        """Locate the vision tower across transformers layouts.

        Pre-refactor (<=~4.5x): `LlavaForConditionalGeneration.vision_tower`.
        Post-refactor (transformers>=5, and some 4.5x point releases): the whole
        multimodal model was moved under an inner `LlavaModel`, so it is instead
        `LlavaForConditionalGeneration.model.vision_tower`. Both are tried, then we
        fall back to a scan for a submodule literally named `vision_tower` so a
        third layout does not fail silently.
        """
        for path in ("vision_tower", "model.vision_tower"):
            vt = self._get_path(self.model, path)
            if vt is not None:
                return vt
        for name, module in self.model.named_modules():
            if name.rsplit(".", 1)[-1] == "vision_tower":
                return module
        return None

    def _vision_self_attn_modules(self):
        """Every self-attention module in the vision tower's encoder layers.

        Navigated defensively across layouts: pre-refactor, `vision_tower` is a
        `CLIPVisionModel` wrapping a `.vision_model.encoder.layers`; post-refactor,
        `CLIPVisionModel` was flattened so the encoder sits directly at
        `.encoder.layers`. Both are tried, then we fall back to a class-name scan.
        """
        vt = self._find_vision_tower()
        if vt is None:
            available = sorted({n.split(".")[0] for n, _ in self.model.named_modules() if n})
            raise RuntimeError(
                "could not locate the vision tower on this model. Tried "
                "`model.vision_tower`, `model.model.vision_tower`, and a scan for a "
                "submodule literally named 'vision_tower'. Top-level submodules found: "
                f"{available}. Run `for n, _ in model.named_modules(): print(n)` and "
                "tell me the path to the vision tower so I can add it."
            )

        layers = None
        for path in ("vision_model.encoder.layers", "encoder.layers"):
            obj = self._get_path(vt, path)
            if obj is not None:
                layers = obj
                break

        mods = []
        if layers is not None:
            for layer in layers:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    mods.append(attn)
        else:
            # last-resort scan
            for m in vt.modules():
                if m.__class__.__name__ in _VISION_ATTN_CLASSES:
                    mods.append(m)

        if not mods:
            raise RuntimeError("found no vision self-attention modules to wrap")
        return mods

    def _install_vision_wrappers(self):
        import inspect
        controller = self
        for attn in self._vision_self_attn_modules():
            if getattr(attn, "_occ_wrapped", False):
                continue
            orig_forward = attn.forward
            # Which keyword args does this version's attention forward accept?
            # (Newer CLIP refactors dropped `causal_attention_mask` and
            #  `output_attentions` as named params in favour of a bare **kwargs;
            #  passing an unaccepted kwarg would raise. So we only forward what
            #  the installed version's forward actually takes.)
            try:
                params = set(inspect.signature(orig_forward).parameters)
            except (TypeError, ValueError):
                params = {"hidden_states", "attention_mask",
                          "causal_attention_mask", "output_attentions"}

            def make_wrapped(_orig, _params):
                def wrapped(hidden_states,
                            attention_mask=None,
                            causal_attention_mask=None,
                            output_attentions=False,
                            **kw):
                    if controller._armed and controller._vit_key_mask:
                        # Override the (normally None) vision attention mask with
                        # our additive key-block bias. Vision self-attention has
                        # no causal or padding mask, so there is nothing to merge.
                        attention_mask = controller._build_vit_bias(
                            hidden_states.shape[0],   # bsz
                            hidden_states.shape[1],   # S (tokens)
                            hidden_states.dtype,
                            hidden_states.device,
                        )
                    call = {"attention_mask": attention_mask}
                    if "causal_attention_mask" in _params:
                        call["causal_attention_mask"] = causal_attention_mask
                    if "output_attentions" in _params:
                        call["output_attentions"] = output_attentions
                    call.update(kw)
                    return _orig(hidden_states, **call)
                return wrapped

            attn.forward = make_wrapped(orig_forward, params)
            attn._occ_wrapped = True
            attn._occ_orig_forward = orig_forward
            self._wrapped.append(attn)

    def remove(self):
        """Restore original forwards. Optional; mainly for tests."""
        for attn in self._wrapped:
            if getattr(attn, "_occ_wrapped", False):
                attn.forward = attn._occ_orig_forward
                attn._occ_wrapped = False
        self._wrapped = []

    # ------------------------------------------------------------------ #
    # ViT additive bias
    # ------------------------------------------------------------------ #
    def _build_vit_bias(self, bsz: int, S: int, dtype, device) -> torch.Tensor:
        """(bsz, 1, S, S) additive mask; masked KEY columns set to dtype-min.

        Only key columns are set, so every query attends to the surviving
        keys. CLS (token 0) is never in the mask, so no row is ever fully
        masked -> softmax is always well defined (no NaN), even when every
        patch is occluded (the full-occlusion / language-prior case).
        """
        min_val = torch.finfo(dtype).min
        bias = torch.zeros((bsz, 1, S, S), dtype=dtype, device=device)
        cols = [t for t in self._vit_key_mask if 0 <= t < S]
        if cols:
            idx = torch.tensor(cols, dtype=torch.long, device=device)
            bias[:, :, :, idx] = min_val
        return bias

    # ------------------------------------------------------------------ #
    # arming
    # ------------------------------------------------------------------ #
    @contextmanager
    def occlude(self, patches: Optional[Sequence[int]]):
        """Arm ViT occlusion for `patches` for the duration of the block."""
        prev_armed, prev_mask = self._armed, self._vit_key_mask
        if patches:
            self._vit_key_mask = frozenset(int(p) + 1 for p in patches)  # +1: CLS offset
            self._armed = True
        else:
            self._vit_key_mask = None
            self._armed = False
        try:
            yield
        finally:
            self._armed, self._vit_key_mask = prev_armed, prev_mask

    # ------------------------------------------------------------------ #
    # language-model key mask
    # ------------------------------------------------------------------ #
    def image_token_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Sequence positions (in the EXPANDED ids) that are image tokens."""
        if input_ids.dim() == 2:
            row = input_ids[0]
        else:
            row = input_ids
        return (row == self.image_token_index).nonzero(as_tuple=True)[0]

    def llm_attention_mask(self,
                           input_ids: torch.Tensor,
                           base_mask: Optional[torch.Tensor],
                           patches: Optional[Sequence[int]]) -> torch.Tensor:
        """2D attention mask with 0 at the object's image-token positions.

        Requires the processor to have EXPANDED <image> into one token per patch
        (so len(image tokens) == g^2). If that expansion did not happen -- e.g. an
        old transformers whose LlavaProcessor leaves <image> as a single token and
        lets the model expand internally -- the per-patch mapping does not exist at
        the input_ids level and this raises, rather than silently masking the wrong
        thing.
        """
        if base_mask is None:
            base_mask = torch.ones_like(input_ids)
        mask = base_mask.clone()
        if not patches:
            return mask

        positions = self.image_token_positions(input_ids)
        n = int(positions.numel())
        if n != self.n_patches:
            raise RuntimeError(
                f"expected {self.n_patches} image tokens in input_ids but found {n}. "
                "Attention occlusion needs the processor to expand <image> into one "
                "token per patch (num_additional_image_tokens set, modern LlavaProcessor). "
                "Without that expansion there is no per-patch position to mask."
            )

        for p in patches:
            p = int(p)
            if 0 <= p < self.n_patches:
                mask[0, positions[p]] = 0
        return mask