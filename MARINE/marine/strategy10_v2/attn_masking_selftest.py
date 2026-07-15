"""
Self-test for attn_masking.AttentionOcclusion.

Runs on CPU in a few seconds against a TINY randomly-initialised LLaVA -- no
weights, no GPU, no dataset. It checks the one property the whole attention-
occlusion idea rests on, and the properties that guard its edges:

  [1] a wrapped-but-disarmed model is bit-identical to the pristine model;
  [2] occluding the empty set is a no-op;
  [3] THE INVARIANT: with a patch set occluded, perturbing the pixels INSIDE that
      set does not change the logits at any position the scorer reads (every text
      position). i.e. the object provably cannot influence the output;
  [4] perturbing pixels OUTSIDE the set DOES change them (we did not hide the
      whole image by accident);
  [5] occlusion has a non-zero effect versus no occlusion;
  [6] full occlusion is finite (no NaN) and invariant to every image pixel;
  [7] the language-model key mask lands on exactly the right positions;
  [8] a sequence whose <image> was not expanded per-patch raises, loudly.

Run:  python -m marine.strategy10_v2.attn_masking_selftest
"""

import copy

import torch

try:  # allow both `python -m ...` and direct execution
    from .attn_masking import AttentionOcclusion
except ImportError:  # pragma: no cover
    from attn_masking import AttentionOcclusion

from transformers import (CLIPVisionConfig, LlamaConfig, LlavaConfig,
                          LlavaForConditionalGeneration)

GRID, PATCH = 4, 8
IMG = GRID * PATCH
NPATCH = GRID * GRID
IMG_TOK = 1


def _build(attn_impl):
    vision = CLIPVisionConfig(hidden_size=32, intermediate_size=64,
                              num_hidden_layers=2, num_attention_heads=4,
                              image_size=IMG, patch_size=PATCH)
    text = LlamaConfig(hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=4,
                       vocab_size=100, max_position_embeddings=512)
    cfg = LlavaConfig(vision_config=vision, text_config=text,
                      image_token_index=IMG_TOK, vision_feature_layer=-2,
                      vision_feature_select_strategy="default")
    torch.manual_seed(1)
    model = LlavaForConditionalGeneration(cfg).eval()
    try:
        model.vision_tower.config._attn_implementation = attn_impl
    except Exception:
        pass
    return model


def _inputs():
    ids = [5, 6] + [IMG_TOK] * NPATCH + [7, 8, 9, 10]
    input_ids = torch.tensor([ids], dtype=torch.long)
    attn = torch.ones_like(input_ids)
    torch.manual_seed(2)
    pix = torch.randn(1, 3, IMG, IMG)
    return input_ids, attn, pix


def _logits(model, input_ids, attn, pix):
    with torch.inference_mode():
        return model(input_ids=input_ids, pixel_values=pix, attention_mask=attn,
                     use_cache=False, return_dict=True).logits[0].float()


def _perturb(pix, patches):
    p = pix.clone()
    ppx = IMG // GRID
    torch.manual_seed(999)
    for idx in patches:
        r, c = divmod(idx, GRID)
        p[0, :, r*ppx:(r+1)*ppx, c*ppx:(c+1)*ppx] += 5.0 * torch.randn(3, ppx, ppx)
    return p


def _text_rows(occ, input_ids):
    img = set(int(x) for x in occ.image_token_positions(input_ids))
    return [t for t in range(input_ids.shape[1]) if t not in img]


def run(attn_impl="sdpa", verbose=True):
    def say(*a):
        if verbose:
            print(*a)

    model = _build(attn_impl)
    input_ids, attn, pix = _inputs()
    base = _logits(copy.deepcopy(model), input_ids, attn, pix)

    occ = AttentionOcclusion(model, grid=GRID, image_token_index=IMG_TOK)
    rows = _text_rows(occ, input_ids)

    d = (_logits(model, input_ids, attn, pix) - base).abs().max().item()
    say(f"[1] disarmed == pristine:            {d:.2e}"); assert d < 1e-5

    with occ.occlude([]):
        d = (_logits(model, input_ids, attn, pix) - base).abs().max().item()
    say(f"[2] empty occlusion == baseline:     {d:.2e}"); assert d < 1e-5

    patches = [5, 6, 9, 10]
    m = occ.llm_attention_mask(input_ids, attn, patches)
    with occ.occlude(patches):
        occ_l = _logits(model, input_ids, m, pix)
        occ_in = _logits(model, input_ids, m, _perturb(pix, patches))
    d_in = (occ_l[rows] - occ_in[rows]).abs().max().item()
    say(f"[3] INVARIANT (perturb inside):      {d_in:.2e}"); assert d_in < 1e-4

    outside = [p for p in range(NPATCH) if p not in patches]
    with occ.occlude(patches):
        occ_out = _logits(model, input_ids, m, _perturb(pix, outside))
    d_out = (occ_l[rows] - occ_out[rows]).abs().max().item()
    say(f"[4] perturb outside changes output:  {d_out:.2e}"); assert d_out > 1e-3

    d_eff = (occ_l[rows] - base[rows]).abs().max().item()
    say(f"[5] occlusion has an effect:         {d_eff:.2e}"); assert d_eff > 1e-3

    allp = list(range(NPATCH))
    mf = occ.llm_attention_mask(input_ids, attn, allp)
    with occ.occlude(allp):
        full = _logits(model, input_ids, mf, pix)
        full_p = _logits(model, input_ids, mf, _perturb(pix, allp))
    assert torch.isfinite(full).all()
    d_full = (full[rows] - full_p[rows]).abs().max().item()
    say(f"[6] full occlusion finite & blind:   {d_full:.2e}"); assert d_full < 1e-4

    positions = occ.image_token_positions(input_ids)
    zeros = (m[0] == 0).nonzero(as_tuple=True)[0].tolist()
    want = sorted(int(positions[p]) for p in patches)
    say(f"[7] llm mask positions {zeros} == {want}"); assert zeros == want

    try:
        occ.llm_attention_mask(torch.tensor([[5, 6, IMG_TOK, 7]]), None, [0])
        raised = False
    except RuntimeError:
        raised = True
    say(f"[8] unexpanded <image> raises:       {raised}"); assert raised

    say(f"OK ({attn_impl})")
    return True


def main():
    for impl in ("eager", "sdpa"):
        print(f"--- attn_implementation = {impl} ---")
        run(impl)
    print("all attention-occlusion invariants hold.")


if __name__ == "__main__":
    main()