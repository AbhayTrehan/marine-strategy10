"""
Hyperparameters for Strategy 10 (v2)  --  GroundingDINO / patch-aligned revision.

WHAT CHANGED, AND WHY
---------------------
1. Localiser is GroundingDINO for EVERY word (candidates AND probes). The
   two-tier scheme of Sec 3.3 (detector, else LVLM-attention fallback) is gone.
   Rationale: the fallback made candidates and probes travel different code paths
   whenever s_det straddled tau_box, which silently breaks the "procedural
   symmetry" that Sec 4.2 says the entire null calibration rests on.
   GroundingDINO always emits a best-guess box for any phrase -- including a
   phrase that is not in the image -- so it can serve BOTH groups through one
   identical procedure. That best-guess box for an absent probe IS the honest
   null: "where would this object be, if it were here?"
   => tau_box, rho, attn_layers, max_patch_frac are deleted. They no longer exist
      as knobs because the branch they controlled no longer exists.

2. Masking is PATCH-ALIGNED and ENFORCED IN pixel_values. See masking.py. The old
   code masked the box's exact pixels in the PIL image; LLaVA's ViT reads 14x14
   patches, so a partially-covered patch still carried the object into the
   encoder, and bicubic resize smeared masked edges into neighbouring patches.
   Object information therefore leaked past the mask. Now every patch the box
   touches is masked in full, and the mask is re-asserted on the normalised
   pixel_values tensor immediately before the vision tower runs. Leakage is
   MEASURED and reported per object, not assumed.

3. The causal evidence score itself is UNTOUCHED. scoring.py still computes
   exactly Eq. (4), (6), (7); calibration.py still computes exactly Eq. (8), (9),
   (11), (12).
"""

from dataclasses import dataclass, field
from typing import List


VICUNA_V1_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

TASK_PROMPT = "Generate a short caption of the image."

# The fixed, task-agnostic elicitation template c_elicit (Sec 3.1). Unchanged.
ELICIT_QUESTION = "What objects are visible in this image?"
ELICIT_ANSWER_PREFIX = "This image contains a"


@dataclass
class Strategy10V2Config:
    # ---- models -------------------------------------------------------------
    model_path: str = "llava-hf/llava-1.5-7b-hf"
    detector_path: str = "IDEA-Research/grounding-dino-base"
    fp16: bool = True            # LVLM in fp16
    detector_fp16: bool = False  # GroundingDINO in fp32: DETR-style heads are
                                 # numerically twitchy in fp16 and the model is
                                 # only ~230M params, so there is nothing to gain.
    device: str = "cuda"

    # ---- data ---------------------------------------------------------------
    image_folder: str = "./data/coco/val2014"
    coco_annotations: str = "./data/coco/annotations"
    chair_cache: str = "./data/coco/chair_cache_s10v2.pkl"
    question_file: str = "./data/org_qa/chair/coco_chair.json"
    cooccur_file: str = "./data/org_qa/pope/coco/coco_co_occur.json"
    num_images: int = 50

    # ---- Stage 1 generation -------------------------------------------------
    max_new_tokens: int = 64
    seed: int = 242

    # ---- Sec 3.3: visual attribution (GroundingDINO) ------------------------
    # GroundingDINO is phrase-grounded, not caption-grounded like OWL-ViT, so the
    # spec's OWL-ViT-flavoured query ("a photo of a {w}.") is replaced by GD's
    # native phrase form. This is a detector-API convention, not a change to the
    # method: the identical string form is used for candidates and probes alike.
    det_prompt_template: str = "{word}."
    det_batch_size: int = 8   # same image repeated; ONE phrase per batch element

    # ---- Sec 4.1: probe sampling -------------------------------------------
    K: int = 20
    tau_low: float = 0.15   # probes must have s_det <= tau_low ("believed absent").
                            # GroundingDINO's scores live on a different scale to
                            # OWL-ViT's, so this is NOT the old 0.05. The report
                            # prints the full s_det distribution so this can be
                            # re-tuned against real data rather than guessed.
    cooccur_bias: float = 3.0

    # ---- Sec 4.3: calibration ----------------------------------------------
    kappa: float = 1.0
    kappa_sweep: List[float] = field(
        default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    )

    # ---- masking integrity --------------------------------------------------
    # Re-assert the mask on the normalised pixel_values tensor right before the
    # vision tower. Guarantees a masked patch carries literally zero object
    # signal, including signal that resize interpolation would otherwise smear in
    # from just outside the box. Leave this on.
    enforce_pixel_mask: bool = True

    # ---- output -------------------------------------------------------------
    output_dir: str = "./output/strategy10_v2"
    html_max_width: int = 460    # px, embedded original / boxed images
    html_thumb_width: int = 190  # px, per-object masked-image thumbnails
