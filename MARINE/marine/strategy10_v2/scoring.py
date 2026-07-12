"""
Sec 3.2 / 3.4 -- the causal occlusion evidence score.

    ell(w)        = (1/L) sum_k log p_theta(w_k | c_elicit, w_<k, I)          (Eq. 4)
    ell_masked(w) = (1/L) sum_k log p_theta(w_k | c_elicit, w_<k, I_masked)   (Eq. 6)
    Delta(w)      = ell(w) - ell_masked(w)                                     (Eq. 7)

Implementation notes that matter for correctness:

* ALIGNMENT. LLaVA expands <image> into 576 image tokens. Depending on the
  transformers version that expansion happens either in the processor (so
  input_ids is already expanded) or inside the model (so logits are LONGER than
  input_ids). Both regimes are handled by aligning from the END of the sequence:
  the expansion only ever inserts tokens at the <image> position, which is
  strictly before the forced word tokens, so the last L logits/tokens are the
  word tokens under either regime.

* L. The number of subword tokens of w is obtained by diffing the tokenisation
  of (c_elicit) against (c_elicit + " " + w) at their first divergence, rather
  than tokenising w in isolation. SentencePiece attaches a leading-space marker
  to the first subword, so tokenising w alone gives a different (and wrong)
  token sequence than w-as-a-continuation.

* Length normalisation by L is what stops multi-token words ("refrigerator",
  "traffic light") from mechanically accruing more negative log-likelihood than
  single-token words -- which would otherwise make Delta's scale word-dependent
  and destroy the comparability of candidates with probes.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch


def continuation_length(tokenizer, prefix: str, full: str) -> int:
    """Number of tokens the word occupies when appended to `prefix`."""
    ids_p = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    ids_f = tokenizer(full, add_special_tokens=True)["input_ids"]

    i = 0
    while i < len(ids_p) and i < len(ids_f) and ids_p[i] == ids_f[i]:
        i += 1

    L = len(ids_f) - i
    return max(1, L)


class OcclusionScorer:
    """Runs the teacher-forced elicitation passes for one LVLM."""

    def __init__(self, model, tokenizer, processor, device: str, dtype: torch.dtype,
                 grid: int, n_image_tokens: int, skip_cls: bool):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.dtype = dtype
        self.grid = grid
        self.n_image_tokens = n_image_tokens
        self.skip_cls = skip_cls
        self.image_token_index = int(getattr(model.config, "image_token_index", 32000))

    def _prepare(self, text: str, image):
        inputs = self.processor(text=text, images=image, return_tensors="pt")
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                v = v.to(self.device)
                if k == "pixel_values":
                    v = v.to(self.dtype)
                out[k] = v
        return out

    @torch.inference_mode()
    def score(self, prefix: str, word: str, image, want_attentions: bool = False
              ) -> Tuple[float, int, Optional[Sequence[torch.Tensor]], int, torch.Tensor]:
        """Returns (ell, L, attentions_or_None, row_idx_of_w1, input_ids_row).

        `row_idx_of_w1` is the position of w_1 in the *model's* (expanded)
        sequence, which is where the attention fallback of Sec 3.3 must read.
        """
        full = f"{prefix} {word}"
        L = continuation_length(self.tokenizer, prefix, full)

        inputs = self._prepare(full, image)
        outputs = self.model(
            **inputs,
            output_attentions=want_attentions,
            use_cache=False,
            return_dict=True,
        )

        logits = outputs.logits[0].float()          # [S, V]
        seq_len = int(logits.shape[0])

        input_ids_row = inputs["input_ids"][0]
        target_ids = input_ids_row[-L:]             # last L input tokens == the word

        # logits at position t predict token t+1  ->  shift by one
        pred_logits = logits[seq_len - L - 1: seq_len - 1, :]
        logprobs = torch.log_softmax(pred_logits, dim=-1)
        token_lp = logprobs.gather(-1, target_ids.view(-1, 1)).squeeze(-1)

        ell = float(token_lp.mean().item())
        row_idx = seq_len - L                       # position of w_1

        attentions = outputs.attentions if want_attentions else None
        return ell, L, attentions, row_idx, input_ids_row


def delta(ell_unmasked: float, ell_masked: float) -> float:
    """Eq. (7)."""
    return ell_unmasked - ell_masked


def confidence_drop_pct(d: float) -> float:
    """Delta is a difference of length-normalised log-likelihoods, so it is the
    log-ratio of the geometric-mean per-token probability of w with and without
    its purported visual support. Exponentiating turns it into the quantity the
    sanity check actually wants to look at:

        % change in confidence = 100 * (p - p_masked) / p = 100 * (1 - exp(-Delta))

    Positive  -> occluding R(w) COST the model confidence in w (visual grounding).
    ~Zero/neg -> the commitment survives (or strengthens) without the pixels.
    """
    import math

    try:
        return 100.0 * (1.0 - math.exp(-d))
    except OverflowError:
        return float("-inf") if d < 0 else 100.0
