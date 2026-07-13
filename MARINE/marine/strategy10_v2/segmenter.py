"""
Grounded-SAM: GroundingDINO finds WHERE, SAM finds exactly WHICH PIXELS.

WHY THIS MATTERS
----------------
A bounding box is the wrong shape for an occlusion experiment. In the first real
run, "laptop" masked 441 of 576 patches -- 77% of the image -- because
GroundingDINO's laptop box is a rectangle that also swallows the desk, the cables
and the wall behind it. Delta then measures "what happens when I delete most of the
picture", which is a question about masked AREA, not about whether the laptop
supports the word "laptop".

SAM turns that box into the object's silhouette. Same object, a fraction of the
pixels, almost no collateral damage. This is the single change most likely to lift
the signal, because it attacks the confound directly rather than trying to
threshold around it.

WHY NOT "SAM INSTEAD OF GROUNDINGDINO"
--------------------------------------
SAM has no text input. It cannot be given the word "laptop" and asked to find the
laptop; it is a *promptable* segmenter and needs a box or a point to start from. So
SAM does not replace GroundingDINO -- it refines it. GDINO stays exactly where it
was (localisation, one phrase per forward, all instances); SAM converts each of its
boxes into a mask.

WHY NOT DINOv2
--------------
DINOv2 patch-feature clustering inside the box approximates the same thing, but it
is a heuristic: it needs a cluster count, it has no notion of "instance", and it
degrades on low-contrast or textured objects. SAM is trained for precisely this
task. If SAM is too heavy, --seg_backend box reverts to the previous behaviour, and
the report will tell you what that cost you.
"""

from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image


class SamSegmenter:
    def __init__(self, model_path: str = "facebook/sam-vit-base",
                 device: str = "cuda", fp16: bool = False):
        from transformers import SamModel, SamProcessor

        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32
        self.processor = SamProcessor.from_pretrained(model_path)
        self.model = SamModel.from_pretrained(
            model_path, torch_dtype=self.dtype
        ).to(device).eval()

    @torch.inference_mode()
    def mask_for_boxes(self, image: Image.Image,
                       boxes: Sequence[Sequence[float]]) -> Optional[np.ndarray]:
        """Union silhouette of every box, as a bool array of the ORIGINAL image size.

        All of a word's instance boxes go in as one batch of prompts, and their masks
        are OR-ed: R(w) is still "every instance of w", just at pixel precision
        instead of rectangle precision.
        """
        if not boxes:
            return None

        W, H = image.size
        inputs = self.processor(
            image, input_boxes=[[list(map(float, b)) for b in boxes]],
            return_tensors="pt",
        )
        inputs = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                  for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        outputs = self.model(**inputs, multimask_output=False)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.float().cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]                                   # [n_boxes, 1, H, W] bool

        m = masks.numpy().astype(bool)
        if m.ndim == 4:
            m = m[:, 0]                        # [n_boxes, H, W]
        union = m.any(axis=0)                  # [H, W]

        if union.shape != (H, W):              # defensive: post_process should give (H, W)
            im = Image.fromarray((union * 255).astype(np.uint8)).resize((W, H), Image.NEAREST)
            union = np.array(im) > 127

        # A SAM mask can come back empty on a degenerate prompt. Falling through with
        # an empty mask would silently mask nothing and hand the word Delta == 0, so
        # the caller is told (None) and falls back to the box.
        return union if union.any() else None
