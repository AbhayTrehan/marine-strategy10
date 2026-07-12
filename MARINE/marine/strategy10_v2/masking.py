"""
Sec 3.3, final paragraph:

    "the masking operator I_masked(w) = Mask(I, R(w)) replaces the pixels in
     R(w) with the per-channel mean pixel value of I, and the modified image is
     re-encoded in full through the vision encoder."

So we mask in ORIGINAL pixel space and hand a fresh PIL image back to the
processor -- we do NOT poke at the normalised pixel_values tensor. That keeps
the masked image on the same preprocessing path as the unmasked one (identical
resize/crop/normalise), which is what "re-encoded in full" means.
"""

from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image


def mean_pixel(image: Image.Image) -> Tuple[int, int, int]:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    mean = arr.reshape(-1, 3).mean(axis=0)
    return tuple(int(round(float(c))) for c in mean)


def apply_mask(image: Image.Image, boxes: Sequence[Sequence[float]],
               fill: Tuple[int, int, int]) -> Tuple[Image.Image, float]:
    """Fill the union of `boxes` with `fill`.

    Returns (masked_image, masked_area_fraction). The area fraction is the
    fraction of the image's pixels actually overwritten -- reported per word,
    because a systematic candidate-vs-probe difference in masked area would be a
    confound in Delta (a bigger occlusion removes more evidence for *any* word).
    """
    img = image.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    touched = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x0, y0, x1, y1 = box
        c0 = max(0, int(np.floor(x0)))
        r0 = max(0, int(np.floor(y0)))
        c1 = min(w, int(np.ceil(x1)))
        r1 = min(h, int(np.ceil(y1)))
        # guarantee at least one pixel for degenerate-but-valid boxes
        c1 = max(c1, min(c0 + 1, w))
        r1 = max(r1, min(r0 + 1, h))
        if c1 <= c0 or r1 <= r0:
            continue
        touched[r0:r1, c0:c1] = True

    arr[touched] = np.array(fill, dtype=arr.dtype)
    area = float(touched.sum()) / float(h * w) if h * w else 0.0
    return Image.fromarray(arr), area


def boxes_area_fraction(boxes: List[Sequence[float]], width: int, height: int) -> float:
    """Union area of boxes / image area (without materialising the image)."""
    if not boxes or width <= 0 or height <= 0:
        return 0.0
    touched = np.zeros((height, width), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        c0, r0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        c1, r1 = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
        if c1 > c0 and r1 > r0:
            touched[r0:r1, c0:c1] = True
    return float(touched.sum()) / float(width * height)
