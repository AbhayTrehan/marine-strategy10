"""
Sec 3.2 / 3.4 -- the causal occlusion evidence score.

THE FORMULAE ARE EXACTLY AS SPECIFIED. Nothing here is reinterpreted.

    ell(w)        = (1/L) * sum_k log p_theta(w_k | c_elicit, w_<k, I)          (Eq. 4)
    ell_masked(w) = (1/L) * sum_k log p_theta(w_k | c_elicit, w_<k, I_masked)   (Eq. 6)
    Delta(w)      = ell(w) - ell_masked(w)                                       (Eq. 7)

The ONLY thing added relative to a naive implementation is `mask_patches`: the
set of ViT patches to overwrite on the normalised pixel_values tensor before it
reaches the vision tower (see masking.py for why that is necessary). It changes
what the model SEES; it does not change how the score is COMPUTED.

Two implementation details that decide whether Eq. (4) is actually what gets
computed, rather than something that merely resembles it:

* ALIGNMENT. LLaVA expands <image> into 576 image tokens. Depending on the
  transformers version that expansion happens either in the processor (input_ids
  already expanded) or inside the model (logits LONGER than input_ids). Both are
  handled by aligning from the END of the sequence: expansion only ever inserts
  tokens at the <image> position, which is strictly before the forced word
  tokens, so the last L logits/tokens are the word tokens under either regime.

* L. The subword count of w is obtained by diffing the tokenisation of c_elicit
  against c_elicit + " " + w at their first divergence, NOT by tokenising w on
  its own. SentencePiece attaches a leading-space marker to a word-initial
  subword, so tokenising w in isolation yields a different (wrong) token
  sequence than w-as-a-continuation, and Eq. (4) would then be summing log-probs
  of tokens the model was never actually asked to produce.

  Length-normalising by L is what stops multi-token words ("refrigerator",
  "traffic light") from mechanically accruing more negative log-likelihood than
  single-token ones -- without it Delta's scale is word-dependent and candidates
  are not comparable to probes at all.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import masking, prompts


def continuation_length(tokenizer, prefix: str, full: str) -> int:
    """Number of tokens w occupies when appended to `prefix`."""
    ids_p = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    ids_f = tokenizer(full, add_special_tokens=True)["input_ids"]

    i = 0
    while i < len(ids_p) and i < len(ids_f) and ids_p[i] == ids_f[i]:
        i += 1
    return max(1, len(ids_f) - i)


class OcclusionScorer:
    def __init__(self, model, tokenizer, processor, device: str, dtype: torch.dtype,
                 grid: int):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.grid = grid

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _forward(self, text: str, image,
                 mask_patches: Optional[Sequence[int]] = None,
                 fill_norm: Optional[Sequence[float]] = None,
                 measure: bool = False):
        """One forward pass, with the mask asserted on pixel_values before the ViT.

        Shared by BOTH scoring heads (likelihood and existence) so that a masked
        image is masked identically no matter which score is being computed.
        """
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        leak = None
        if mask_patches is not None and len(mask_patches) and fill_norm is not None:
            if measure:
                leak = masking.measure_leak(
                    inputs["pixel_values"].float(), mask_patches, self.grid, fill_norm
                )
            inputs["pixel_values"] = masking.enforce_on_pixel_values(
                inputs["pixel_values"], mask_patches, self.grid, fill_norm
            )
            enforced_ok = masking.verify_enforced(
                inputs["pixel_values"], mask_patches, self.grid, fill_norm
            )
        else:
            enforced_ok = True

        outputs = self.model(**inputs, use_cache=False, return_dict=True)
        return outputs.logits[0].float(), inputs["input_ids"][0], leak, enforced_ok

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def score_existence(self, word: str, image,
                        mask_patches: Optional[Sequence[int]] = None,
                        fill_norm: Optional[Sequence[float]] = None,
                        measure: bool = False) -> Dict:
        """LO(w) = log p("Yes") - log p("No") for "Is there a {w} in this image?".

        A log ODDS-RATIO of the model's belief that w EXISTS -- not a likelihood of
        the token w. See prompts.existence_prefix for why that difference matters.
        Read off the single next-token distribution after "ASSISTANT:", so it costs
        exactly one forward pass, same as Eq. (4).
        """
        prefix = prompts.existence_prefix(word)
        logits, _ids, leak, ok = self._forward(
            prefix, image, mask_patches, fill_norm, measure
        )
        lp = torch.log_softmax(logits[-1, :], dim=-1)   # next-token distribution

        yes_id, no_id = self._yes_no_ids(prefix)
        return {
            "lo": float(lp[yes_id].item() - lp[no_id].item()),
            "p_yes": float(lp[yes_id].exp().item()),
            "leak": leak,
            "mask_enforced_ok": bool(ok),
        }

    def _yes_no_ids(self, prefix: str):
        """Token ids for "Yes"/"No" AS CONTINUATIONS of the prompt.

        Tokenised in isolation, SentencePiece gives a different (word-initial) piece
        than it gives mid-sentence, so we diff prefix vs prefix+" Yes" and take the
        first divergent token -- the same trick continuation_length uses, and for the
        same reason.
        """
        if getattr(self, "_yn", None) is None:
            def first_new(word):
                a = self.tokenizer(prefix, add_special_tokens=True)["input_ids"]
                b = self.tokenizer(f"{prefix} {word}", add_special_tokens=True)["input_ids"]
                i = 0
                while i < len(a) and i < len(b) and a[i] == b[i]:
                    i += 1
                return b[i]
            self._yn = (first_new("Yes"), first_new("No"))
        return self._yn

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def score(self, prefix: str, word: str, image,
              mask_patches: Optional[Sequence[int]] = None,
              fill_norm: Optional[Sequence[float]] = None,
              measure: bool = False) -> Dict:
        """One teacher-forced elicitation pass. Returns ell(w) and diagnostics.

        If `mask_patches` is given, those patches are overwritten on pixel_values
        with `fill_norm` BEFORE the model runs -- i.e. before the vision tower can
        encode them. `measure=True` additionally reports how much object signal
        was still present in those patches BEFORE that overwrite (i.e. how much
        the PIL-level mask alone had left behind); see masking.measure_leak.
        """
        full = f"{prefix} {word}"
        L = continuation_length(self.tokenizer, prefix, full)

        logits, input_ids_row, leak, enforced_ok = self._forward(
            full, image, mask_patches, fill_norm, measure
        )
        seq_len = int(logits.shape[0])

        target_ids = input_ids_row[-L:]                        # the word's tokens
        pred_logits = logits[seq_len - L - 1: seq_len - 1, :]  # shift by one
        logprobs = torch.log_softmax(pred_logits, dim=-1)
        token_lp = logprobs.gather(-1, target_ids.view(-1, 1)).squeeze(-1)

        return {
            "ell": float(token_lp.mean().item()),   # Eq. (4) / Eq. (6), VERBATIM
            "L": L,
            "leak": leak,
            "mask_enforced_ok": bool(enforced_ok),
        }


def delta(ell_unmasked: float, ell_masked: float) -> float:
    """Eq. (7). Verbatim."""
    return ell_unmasked - ell_masked


def confidence_drop_pct(d: float) -> float:
    """A readability-only restatement of Delta; NOT used in any decision.

    Delta is a difference of length-normalised log-likelihoods, i.e. the log-ratio
    of the geometric-mean per-token probability of w with and without its
    purported visual support. Exponentiating turns it into a percentage:

        100 * (p - p_masked) / p = 100 * (1 - exp(-Delta))

    Positive -> occluding R(w) COST the model confidence in w (visual grounding).
    ~Zero/negative -> the commitment survives, or strengthens, without the pixels.
    Every verify/flag decision is made on Delta itself, per Eq. (11)/(12).
    """
    try:
        return 100.0 * (1.0 - math.exp(-d))
    except OverflowError:
        return float("-inf") if d < 0 else 100.0


def probability(ell: float) -> float:
    """exp(ell): the geometric-mean per-token probability. Display only."""
    try:
        return math.exp(ell)
    except OverflowError:
        return 0.0
