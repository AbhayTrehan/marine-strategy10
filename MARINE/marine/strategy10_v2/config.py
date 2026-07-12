"""
Hyperparameters for Strategy 10 (v2).

Symbol map to the spec (strategy10_v2.pdf):

    tau_box   : detector confidence above which R(w) := top detector box   (Sec 3.3, step 1)
    tau_low   : detector confidence above which a word is DISQUALIFIED as
                a "guaranteed-absent" probe                                 (Sec 4.1)
    rho       : cumulative attention mass for the attention fallback region (Sec 3.3, step 2)
    K         : number of probes per image                                  (Sec 4.1)
    kappa     : threshold multiplier, tau = mu_hat + kappa * sigma_hat      (Sec 4.3.1, Eq. 9)

NOTE ON kappa (Sec 4.3.2 / Sec 8): v2 gives up v1's exact finite-sample
false-verification guarantee. kappa is therefore treated here EXACTLY as the
spec instructs -- as a plain tunable sensitivity knob, on the same footing as
tau_box / rho / tau_low -- with no probability claim attached. The sanity-check
report sweeps kappa and prints the resulting precision/recall trade-off, which
is the empirical selection procedure the spec prescribes.
"""

from dataclasses import dataclass, field
from typing import List


# LLaVA-1.5 (vicuna_v1) system prompt. Kept here verbatim so this package can run
# even if the `llava` repo is not on PYTHONPATH; when it *is* importable we build
# the prompt through llava.conversation instead and this is used only as a fallback.
VICUNA_V1_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Stage 1 task prompt x. Same string the MARINE paper / this repo use for CHAIR.
TASK_PROMPT = "Generate a short caption of the image."

# The fixed, task-agnostic elicitation template c_elicit (Sec 3.1).
ELICIT_QUESTION = "What objects are visible in this image?"
ELICIT_ANSWER_PREFIX = "This image contains a"


@dataclass
class Strategy10V2Config:
    # ---- models -------------------------------------------------------------
    model_path: str = "llava-hf/llava-1.5-7b-hf"
    detector_path: str = "google/owlvit-base-patch32"
    fp16: bool = True
    device: str = "cuda"

    # ---- data ---------------------------------------------------------------
    image_folder: str = "./data/coco/val2014"
    coco_annotations: str = "./data/coco/annotations"
    chair_cache: str = "./data/coco/chair_cache.pkl"
    question_file: str = "./data/org_qa/chair/coco_chair.json"
    cooccur_file: str = "./data/org_qa/pope/coco/coco_co_occur.json"
    num_images: int = 50

    # ---- Stage 1 generation -------------------------------------------------
    max_new_tokens: int = 64
    seed: int = 242  # same seed the repo uses

    # ---- Sec 3.3: visual attribution ---------------------------------------
    tau_box: float = 0.10   # OWL-ViT sigmoid confidence; 0.1 is the standard OWL-ViT operating point
    rho: float = 0.25       # fraction of *image-normalised* attention mass
    attn_layers: List[int] = field(default_factory=lambda: [14, 15, 16, 17, 18, 19])
    attn_heads: List[int] = field(default_factory=list)  # empty == all heads
    max_patch_frac: float = 0.50  # safety valve: never mask more than half the patch grid

    # ---- Sec 4.1: probe sampling -------------------------------------------
    K: int = 20
    tau_low: float = 0.05   # probes must have s_det <= tau_low ("believed absent")
    cooccur_bias: float = 3.0  # >0 up-weights language-prior distractors; 0 == uniform

    # ---- Sec 4.3: calibration ----------------------------------------------
    kappa: float = 1.0
    kappa_sweep: List[float] = field(
        default_factory=lambda: [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    )

    # ---- output -------------------------------------------------------------
    output_dir: str = "./output/strategy10_v2"
