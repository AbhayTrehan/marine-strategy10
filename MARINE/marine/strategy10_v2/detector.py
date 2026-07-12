"""
Sec 3.3 step 1 + Sec 4.1: the "lightweight zero-shot detector".

The spec names OWL-ViT explicitly, and it has to be open-vocabulary: probes are
sampled from a vocabulary and must be scored with the *same* detector as
candidates (Sec 4.2, procedural symmetry). The DETR wrapper already in this repo
(marine/grounding_models/detr_detect.py) is closed-vocabulary over COCO-91 and
outputs a fixed class distribution, so it cannot serve this role for arbitrary
probe words. Hence OWL-ViT.

Efficiency note: OWL-ViT scores every text query against ONE shared image
embedding, so we score the entire 80-word COCO vocabulary in a single forward
pass per image. That gives us, for free and at once:
    * s_det(w) + top box for each candidate  (Sec 3.3)
    * s_det(v) for every probe candidate, needed for the "exclude words with
      s_det(v) > tau_low" filter                                   (Sec 4.1)
"""

from typing import Dict, List

import torch


class OwlViTDetector:
    def __init__(self, model_path: str = "google/owlvit-base-patch32",
                 device: str = "cuda", fp16: bool = True):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        self.processor = OwlViTProcessor.from_pretrained(model_path)
        self.model = OwlViTForObjectDetection.from_pretrained(
            model_path, torch_dtype=self.dtype
        ).to(device)
        self.model.eval()

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor, width: int, height: int) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        x0 = (cx - 0.5 * w) * width
        y0 = (cy - 0.5 * h) * height
        x1 = (cx + 0.5 * w) * width
        y1 = (cy + 0.5 * h) * height
        return torch.stack([x0, y0, x1, y1], dim=-1)

    @torch.inference_mode()
    def score_vocabulary(self, image, vocabulary: List[str]) -> Dict[str, Dict]:
        """One forward pass; returns {word: {"score": float, "box": [x0,y0,x1,y1]}}.

        `score` is the max sigmoid confidence over all predicted boxes for the
        query "a photo of a {word}" -- i.e. s_det(w) in the spec. `box` is that
        argmax box, in ORIGINAL image pixel coordinates.

        We compute this from raw outputs rather than via
        post_process_object_detection(threshold=...) because we need the argmax
        box for *every* query, including ones scoring near zero (a probe still
        needs a region if it is ever routed to the detector branch).
        """
        width, height = image.size
        queries = [f"a photo of a {w}." for w in vocabulary]

        inputs = self.processor(
            text=[queries], images=image, return_tensors="pt",
            padding=True, truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        outputs = self.model(**inputs)

        # logits: [1, n_boxes, n_queries]; pred_boxes: [1, n_boxes, 4] (cxcywh, normalised)
        probs = torch.sigmoid(outputs.logits[0].float())      # [n_boxes, n_queries]
        boxes = outputs.pred_boxes[0].float()                 # [n_boxes, 4]

        best_score, best_idx = probs.max(dim=0)               # [n_queries] each
        best_boxes = self._cxcywh_to_xyxy(boxes[best_idx], width, height)

        result = {}
        for i, word in enumerate(vocabulary):
            x0, y0, x1, y1 = best_boxes[i].tolist()
            x0 = max(0.0, min(x0, width))
            x1 = max(0.0, min(x1, width))
            y0 = max(0.0, min(y0, height))
            y1 = max(0.0, min(y1, height))
            if x1 <= x0 or y1 <= y0:
                box = None  # degenerate -> caller must use the attention fallback
            else:
                box = [x0, y0, x1, y1]
            result[word] = {"score": float(best_score[i].item()), "box": box}

        return result
