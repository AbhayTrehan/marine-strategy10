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

# Stage 1 task prompt x. The spec (Sec 2) allows any task prompt -- it gives
# "Generate a short caption of the image." and "a VQA question" as EXAMPLES, not as
# a constraint -- and an object-listing prompt is a far cleaner source of mentions
# for a verification sanity check: free-form captions produce compound noun phrases
# ("a CD/DVD player", "a baby carriage") whose head nouns collide with CHAIR's
# word-level synonym table and invent objects nobody mentioned. Listing asks the
# model for the object names directly.
# Use --task_prompt to switch back to captioning.
TASK_PROMPT = "List all the objects visible in this image."
CAPTION_PROMPT = "Generate a short caption of the image."

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
    task_prompt: str = TASK_PROMPT
    max_new_tokens: int = 64
    seed: int = 242

    # ---- Sec 3.3: visual attribution (GroundingDINO) ------------------------
    # GroundingDINO is phrase-grounded, not caption-grounded like OWL-ViT, so the
    # spec's OWL-ViT-flavoured query ("a photo of a {w}.") is replaced by GD's
    # native phrase form. This is a detector-API convention, not a change to the
    # method: the identical string form is used for candidates and probes alike.
    det_prompt_template: str = "{word}."
    det_batch_size: int = 8   # same image repeated; ONE phrase per batch element

    # R(w) is the UNION of every instance of w that the detector finds, not just its
    # best box: with three people in the image, masking one leaves two visible, the
    # LVLM still sees people, and a REAL object collapses to Delta ~ 0 and gets
    # flagged. See detector._select_instances for why this does NOT hand probes a
    # bigger region (their scores sit far below det_inst_floor, so they keep getting
    # exactly one box -- as does a genuinely hallucinated candidate, which is the
    # point: hallucinated candidates and probes stay in the same null).
    det_inst_ratio: float = 0.5      # keep boxes scoring >= ratio * this word's top score
    det_inst_floor: float = 0.25     # ...but never below this absolute score
    det_max_instances: int = 10      # cap, applied identically to both groups
    det_nms_iou: float = 0.5         # GD emits ~900 queries; NMS dedupes them

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

    # ---- segmentation (Grounded-SAM) ----------------------------------------
    # SAM cannot replace GroundingDINO -- it has no text input and cannot be asked to
    # "find the laptop". GDINO localises (box), SAM refines (silhouette). A box around
    # a table contains the laptop, the cup and the wall; a silhouette does not. In the
    # first real run one object's box masked 77% of the image, which made Delta a
    # measure of masked AREA rather than of grounding.
    seg_backend: str = "sam"                       # "sam" | "box"
    sam_path: str = "facebook/sam-vit-base"
    mask_dilate_patches: int = 0                   # grow R(w) by n patch rings

    # ---- scoring heads -------------------------------------------------------
    # All are computed from the SAME masked images and calibrated against the SAME
    # probes, so the report can print AUROC for each and the data can pick the winner.
    #   delta         : Eq. (4)/(6)/(7), the spec's score, VERBATIM
    #   delta_lo      : existence log-odds -- cancels the word's unigram prior
    #   delta_ctrl    : area-controlled  (needs --control_mask)
    #   delta_lo_ctrl : both             (needs --control_mask)
    #   delta_ins     : SUFFICIENCY -- mask everything EXCEPT R(w)
    #   delta_lo_ins  : the same, on the existence head
    #
    # WHY delta ALONE WAS NEVER GOING TO BE ENOUGH
    # -------------------------------------------
    # delta is a DELETION (necessity) test: remove R(w), see if belief in w drops.
    # But the spec itself defines a hallucination as "a mention driven by language
    # priors or SCENE-LEVEL PLAUSIBILITY rather than the pixels". Deletion cannot see
    # that: deleting a hallucinated object's region leaves the entire scene that
    # invented it untouched, so the model keeps believing and delta ~ 0. You learn
    # that the model did not need those pixels; you never learn that it was leaning on
    # context, because you never removed the context.
    #
    # INSERTION does. Mask everything EXCEPT R(w) and the model can see ONLY the
    # region that allegedly supports w:
    #     real          -> its pixels are still there  -> belief survives
    #     hallucinated  -> nothing in R(w), AND the context is gone -> belief collapses
    #
    # Also note: delta_ctrl and delta_ins are contrasts between two MASKED conditions
    # and never reference l_full. That is not incidental. l_full is precisely where
    # candidates and probes stop being exchangeable -- candidates are words the model
    # CHOSE to say (l_full high by construction), probes are words it did not (l_full
    # at the floor, no room to fall). Any score shaped l_full - l_masked inherits that
    # asymmetry; a masked-vs-masked contrast cancels it, along with the word's unigram
    # prior and the masked-AREA effect.
    scores: List[str] = field(default_factory=lambda: [
        "delta", "delta_lo", "delta_ins", "delta_lo_ins",
        "delta_ctrl", "delta_lo_ctrl",
    ])
    primary_score: str = "delta"       # DECISION score. Default = the spec's own, so the
                                       # headline stays the spec's number; every other
                                       # score is measured alongside and its AUROC printed.
    control_mask: bool = True          # mask an equal-size, equal-shape region ELSEWHERE
    insertion: bool = True             # mask everything EXCEPT R(w)  (sufficiency)

    # THE LANGUAGE-PRIOR BASELINE. The model's belief in w with the image ENTIRELY
    # masked. Without it, every score is a linear combination of just three numbers
    # (l_full, l_del, l_keep), which is why they all kept collapsing onto each other.
    # With it, existence log-odds DECOMPOSE:
    #     PRIOR = LO_blank                 pure language prior, no image
    #     SUF   = LO_keep - LO_blank       evidence from the REGION
    #     CTX   = LO_del  - LO_blank       evidence from the SCENE     <- never measured
    #     NEC   = LO_full - LO_del         necessity
    # CTX is the new signal, and it points the OTHER WAY: a hallucination survives the
    # deletion of its own region (LO_del stays high) while its prior is only a prior,
    # so CTX is LARGE for hallucinations and small/negative for real objects. Sec 3.4
    # literally defines a hallucination as one "driven by language priors or
    # scene-level plausibility" -- CTX is that quantity, measured.
    language_prior: bool = True

    # sigma_hat from K=20 probes has ~16% relative standard error, so tau is itself a
    # noisy random variable. Shrink the per-image sigma_hat toward the pooled sigma
    # across images (empirical-Bayes / James-Stein). 0.0 = pure Eq. (8), spec-faithful.
    sigma_shrink: float = 0.0

    # ---- vocabulary ----------------------------------------------------------
    # Ground truth is COCO-80 and nothing else, so only COCO objects can be SCORED.
    # Non-COCO mentions ("headphones", "doll") are extracted and shown with
    # gt_label=UNKNOWN, and excluded from every metric -- visible, not measured.
    ram_vocab_file: str = "./data/marine_qa/guidance/coco_ram_th0.68.json"
    extract_non_coco: bool = True      # surface non-COCO mentions (not scored)
    probe_vocab: str = "coco"          # "coco" | "ram"
    dup_iou: float = 0.5               # flag candidates whose regions collide

    # ---- occlusion mechanism ------------------------------------------------
    # HOW a patch region R(w) is removed from the model for the causal test.
    #
    #   "attention" (default): R(w) is removed from ATTENTION -- its patch tokens
    #       are dropped as keys in every ViT self-attention layer, and its image
    #       tokens are dropped as keys in the LVLM's attention -- while the image
    #       itself is left untouched. This is the intervention the method actually
    #       wants: the model does not attend to the object's patches at all. After
    #       occlusion the object's pixels have no attention path to any read
    #       position, so perturbing them cannot change l_masked (verified by
    #       attn_masking_selftest.py). This eliminates -- rather than measures --
    #       the two leaks the pixel path fought (partial patches, resize bleed) and
    #       the third the pixel path could not touch at all: the ViT's own self-
    #       attention having already copied the object into neighbouring tokens.
    #
    #   "pixel": the legacy inpainting path. R(w)'s patches are overwritten with
    #       the per-channel mean pixel value on the normalised pixel_values tensor
    #       (see masking.py). Retained for ablation -- pixel-vs-attention is a
    #       first-class comparison for Section 8 -- and NOT the default.
    occlusion: str = "attention"

    # ---- masking integrity (occlusion="pixel" only) -------------------------
    # Re-assert the mask on the normalised pixel_values tensor right before the
    # vision tower. Guarantees a masked patch carries literally zero object
    # signal, including signal that resize interpolation would otherwise smear in
    # from just outside the box. Only consulted by the legacy pixel path;
    # attention occlusion needs no such enforcement because it never inpaints.
    enforce_pixel_mask: bool = True

    # ---- output -------------------------------------------------------------
    output_dir: str = "./output/strategy10_v2"
    html_max_width: int = 460    # px, embedded original / boxed images
    html_thumb_width: int = 190  # px, per-object masked-image thumbnails