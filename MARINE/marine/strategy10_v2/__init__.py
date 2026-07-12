"""
Strategy 10 (v2): Single-Pass Causal Occlusion Verification with Probe-Derived
Threshold Calibration.

Self-contained. The only things it borrows from the surrounding MARINE repo are:
  * eval/eval_chair.py's CHAIR class (object extraction + COCO ground truth)
  * data/org_qa/chair/coco_chair.json (the image list)
  * data/org_qa/pope/coco/coco_co_occur.json (co-occurrence table for probe bias)
  * data/coco/{val2014,annotations} (the dataset itself)

Nothing outside marine/strategy10_v2/ and scripts/ is modified.
"""

from .config import Strategy10V2Config

__all__ = ["Strategy10V2Config"]
