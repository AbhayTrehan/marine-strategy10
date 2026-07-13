"""
Sec 3.3 -- visual attribution, via GroundingDINO, for EVERY word.

WHY GROUNDINGDINO SERVES BOTH CANDIDATES AND PROBES
---------------------------------------------------
The spec's two-tier scheme (detector if s_det >= tau_box, else LVLM-attention
fallback) is not used here. It routed candidates and probes down different code
paths depending on where their detector score fell, which quietly destroys the
"procedural symmetry between candidates and probes" that Sec 4.2 says the null
calibration depends on -- a probe scored through the attention branch and a
candidate scored through the detector branch are simply not measuring the same
thing, so their Deltas are not comparable, so mu_hat/sigma_hat are meaningless.

GroundingDINO removes the need for the fallback: it is a *phrase*-grounded
detector, and it emits its best-guess box for ANY phrase you give it, including
one that is not in the image. That best-guess box is exactly what Sec 3.3 asks
for -- "a region R(w) PURPORTING to support w visually" -- and for an absent
probe it is precisely the honest null: "where would this object be, if it were
here?" One detector, one procedure, both groups.

ALL INSTANCES, NOT JUST THE BEST ONE
------------------------------------
R(w) is "the region purporting to support w". If the image contains three people,
all three support the word "person" -- masking only GroundingDINO's single
best-scoring box leaves the other two in plain sight, the LVLM still sees people,
l_masked barely moves, Delta collapses to ~0, and a perfectly REAL object gets
flagged as hallucinated. So R(w) is the UNION of every instance the detector
finds for w.

Which boxes count as "an instance" has to be decided by a rule that is applied
IDENTICALLY to candidates and probes, or the null stops being comparable. The rule:

    always take the argmax box                        (so every word, including an
                                                       absent probe, always has a
                                                       region -- no region, no Delta)
    plus every box scoring >= max(inst_floor, inst_ratio * s_top(w))
    then NMS, then cap at max_instances

Note what this does NOT do: it does not give probes a bigger region. A probe's top
score sits far below inst_floor, so the "plus" clause fires for nothing and the
probe gets exactly one box -- as before. The same is true of a genuinely
hallucinated candidate, which is the point: hallucinated candidates and probes go
through the same procedure and land in the same null. Only a word the detector
actually finds, repeatedly and confidently, gets a bigger region -- and for that
word, a bigger region is simply the correct region.

ONE PHRASE PER FORWARD PASS
---------------------------
GroundingDINO's text encoder is a BERT that attends across the whole prompt. If
several phrases are packed into one prompt ("dog. cat. sofa."), they condition
one another through text self-attention, and each phrase's grounding is then
contaminated by the others -- which would leak the candidate set into the probe
scores and vice versa. Every word therefore gets its OWN prompt containing only
that word. Throughput is recovered by batching along the BATCH dimension (the
same image repeated N times against N single-phrase prompts), never by packing
phrases into one prompt.
"""

from typing import Dict, List, Sequence

import torch


class GroundingDinoDetector:
    def __init__(self, model_path: str = "IDEA-Research/grounding-dino-base",
                 device: str = "cuda", fp16: bool = False,
                 prompt_template: str = "{word}.", batch_size: int = 8,
                 inst_ratio: float = 0.5, inst_floor: float = 0.25,
                 max_instances: int = 10, nms_iou: float = 0.5):
        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        self.prompt_template = prompt_template
        self.batch_size = max(1, int(batch_size))
        self.inst_ratio = float(inst_ratio)
        self.inst_floor = float(inst_floor)
        self.max_instances = max(1, int(max_instances))
        self.nms_iou = float(nms_iou)

        self.processor = self._load_processor(model_path)
        self.model = self._load_model(model_path)
        self.model.eval()

        tok = self.processor.tokenizer
        self._special_ids = {
            i for i in (
                tok.cls_token_id, tok.sep_token_id, tok.pad_token_id,
                tok.convert_tokens_to_ids("."),
            ) if i is not None
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_processor(model_path: str):
        from transformers import AutoProcessor

        try:
            return AutoProcessor.from_pretrained(model_path)
        except Exception as exc:
            # AutoProcessor pulls the FAST (Rust) BERT tokenizer, which reads
            # tokenizer.json -- a format that is not forward-compatible across
            # `tokenizers` releases. Fall back to the slow tokenizer, which reads
            # vocab.txt and cannot hit that incompatibility.
            print(f"[detector] AutoProcessor failed ({type(exc).__name__}: {exc}); "
                  f"retrying with the slow tokenizer.")
            from transformers import (AutoImageProcessor, AutoTokenizer,
                                      GroundingDinoProcessor)
            return GroundingDinoProcessor(
                image_processor=AutoImageProcessor.from_pretrained(model_path),
                tokenizer=AutoTokenizer.from_pretrained(model_path, use_fast=False),
            )

    def _load_model(self, model_path: str):
        try:
            from transformers import AutoModelForZeroShotObjectDetection
            model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_path, torch_dtype=self.dtype
            )
        except Exception:
            from transformers import GroundingDinoForObjectDetection
            model = GroundingDinoForObjectDetection.from_pretrained(
                model_path, torch_dtype=self.dtype
            )
        return model.to(self.device)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cxcywh_to_xyxy(box: torch.Tensor, width: int, height: int) -> List[float]:
        cx, cy, w, h = [float(v) for v in box]
        x0 = (cx - 0.5 * w) * width
        y0 = (cy - 0.5 * h) * height
        x1 = (cx + 0.5 * w) * width
        y1 = (cy + 0.5 * h) * height
        # clamp into the image; guarantee non-degenerate
        x0, x1 = max(0.0, min(x0, width)), max(0.0, min(x1, width))
        y0, y1 = max(0.0, min(y0, height)), max(0.0, min(y1, height))
        if x1 <= x0:
            x0, x1 = max(0.0, x0 - 1.0), min(float(width), x0 + 1.0)
        if y1 <= y0:
            y0, y1 = max(0.0, y0 - 1.0), min(float(height), y0 + 1.0)
        return [x0, y0, x1, y1]

    @staticmethod
    def _iou(a: Sequence[float], b: Sequence[float]) -> float:
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _select_instances(self, scores: torch.Tensor, boxes_xyxy: List[List[float]]):
        """argmax + every box clearing max(floor, ratio*top), NMS'd and capped.

        The argmax is kept unconditionally so that EVERY word -- including an
        absent probe, whose scores all sit near zero -- always has a region. A word
        with no region has no Delta, and a probe with no Delta is not in the null.
        """
        top = float(scores.max())
        cut = max(self.inst_floor, self.inst_ratio * top)

        order = torch.argsort(scores, descending=True).tolist()
        best = order[0]
        cands = [best] + [i for i in order[1:] if float(scores[i]) >= cut]

        kept: List[int] = []
        for i in cands:
            if all(self._iou(boxes_xyxy[i], boxes_xyxy[j]) < self.nms_iou for j in kept):
                kept.append(i)
            if len(kept) >= self.max_instances:
                break
        return kept, top

    @torch.inference_mode()
    def score_words(self, image, words: Sequence[str]) -> Dict[str, Dict]:
        """{word: {"score", "boxes", "scores", "n_instances"}} for EVERY word.

        `score` is the top score for the phrase; `boxes` is EVERY instance the
        detector found (see _select_instances). R(w) downstream is the union of
        those boxes -- masking only the best one leaves the other instances of a
        multi-instance object visible, which is how a REAL object ends up with
        Delta ~ 0 and gets flagged.
        """
        width, height = image.size
        out: Dict[str, Dict] = {}
        words = list(words)

        for start in range(0, len(words), self.batch_size):
            chunk = words[start:start + self.batch_size]
            texts = [self.prompt_template.format(word=w.lower()) for w in chunk]

            inputs = self.processor(
                images=[image] * len(chunk), text=texts,
                return_tensors="pt", padding=True,
            )
            inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                      for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

            outputs = self.model(**inputs)

            # Exactly HF's own post_process_grounded_object_detection maths:
            #   scores = sigmoid(logits).max(over text positions)
            #   boxes  = center_to_corners(pred_boxes) * [W, H, W, H]
            # ...with one necessary addition: we mask out PAD and special tokens
            # before the max. HF does not, because HF assumes a single un-padded
            # prompt; we batch, so padded positions carry arbitrary logits that
            # would otherwise win the max and produce a garbage score.
            probs = outputs.logits.sigmoid().float()          # [B, n_query, n_text]
            input_ids = inputs["input_ids"]                    # [B, T]
            T = min(int(input_ids.shape[1]), int(probs.shape[-1]))
            probs = probs[:, :, :T]

            valid = inputs["attention_mask"][:, :T].bool()
            for sid in self._special_ids:
                valid &= (input_ids[:, :T] != sid)
            # if a phrase somehow tokenised to nothing but specials, fall back to
            # the attention mask so we never take a max over an empty set
            empty = ~valid.any(dim=1)
            if bool(empty.any()):
                valid[empty] = inputs["attention_mask"][:, :T].bool()[empty]

            probs = probs.masked_fill(~valid[:, None, :], 0.0)
            query_scores = probs.max(dim=-1).values            # [B, n_query]

            for i, word in enumerate(chunk):
                sc = query_scores[i].float().cpu()             # [n_query]
                bx = [self._cxcywh_to_xyxy(b, width, height)
                      for b in outputs.pred_boxes[i].float().cpu()]
                kept, top = self._select_instances(sc, bx)
                out[word] = {
                    "score": top,
                    "scores": [float(sc[k]) for k in kept],
                    "boxes": [bx[k] for k in kept],
                    "n_instances": len(kept),
                }

        return out
