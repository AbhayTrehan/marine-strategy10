"""
Algorithm 1 -- per image.

    Stage 1  y <- M(.|x, I);  O <- ExtractCanonicalObjects(y)
    Stage 2  GroundingDINO localises EVERY word in V (one phrase per forward)
             P <- SampleAbsentProbes(V, O, K, tau_low)      [uses those s_det]
             for w in O U P:
                 R(w) <- patches touched by GD's box for w
                 I_masked(w) <- Mask(I, R(w))               [patch-aligned + enforced]
                 Delta(w) <- ell(w) - ell_masked(w)         [Eq. 4/6/7, verbatim]
             mu_hat, sigma_hat <- moments of {Delta(p)}      [Eq. 8]
             tau <- mu_hat + kappa*sigma_hat                 [Eq. 9]
             O_pos <- {Delta >= tau};  H <- O \\ O_pos        [Eq. 11/12]
    Stage 3  (rewriting -- NOT run; this is the verification sanity check)

Cost note: GroundingDINO is run once over the whole 80-word COCO vocabulary per
image (batched, one phrase per batch element). That is what buys us s_det for the
probe filter AND a box for every probe in a single pass -- and every probe NEEDS a
box, because without R(p) there is no Delta(p) and hence no null distribution.
"""

import random
from typing import Dict, List, Optional

import torch
from PIL import Image

from . import attribution, calibration, extraction, masking, probes, prompts, scoring
from .config import TASK_PROMPT


class Strategy10V2Pipeline:
    def __init__(self, cfg, model, tokenizer, processor, detector, evaluator,
                 vocabulary, cooccur, grid):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.detector = detector
        self.evaluator = evaluator
        self.vocabulary = vocabulary
        self.cooccur = cooccur
        self.grid = grid

        self.dtype = torch.float16 if cfg.fp16 else torch.float32
        self.scorer = scoring.OcclusionScorer(
            model, tokenizer, processor, cfg.device, self.dtype, grid
        )
        self.image_processor = processor.image_processor
        self.elicit_prefix = prompts.elicitation_prefix()
        self.task_prompt = prompts.task_prompt(TASK_PROMPT)

    # ------------------------------------------------------------------ #
    # Stage 1
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.processor(text=self.task_prompt, images=image, return_tensors="pt")
        inputs = {
            k: (v.to(self.cfg.device).to(self.dtype) if k == "pixel_values"
                else v.to(self.cfg.device))
            for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        n_in = int(inputs["input_ids"].shape[1])
        out_ids = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=self.cfg.max_new_tokens,
            use_cache=True,
        )
        return self.tokenizer.batch_decode(
            out_ids[:, n_in:], skip_special_tokens=True
        )[0].strip()

    # ------------------------------------------------------------------ #
    # Sec 3.3 + 3.4 -- identical code path for candidates and probes
    # ------------------------------------------------------------------ #
    def score_word(self, word: str, image: Image.Image, det: Dict,
                   fill: tuple, fill_norm: List[float]) -> Dict:
        W, H = image.size
        entry = det.get(word, {})
        s_det = float(entry.get("score", 0.0))
        box = entry.get("box")

        # R(w): every ViT patch GroundingDINO's box touches, in full.
        if box is None:
            patches, geo = [], {"outside_crop": True, "clipped_by_crop": False,
                                "n_patches": 0, "patch_frac": 0.0}
        else:
            patches, geo = attribution.box_to_patches(
                box, self.grid, W, H, self.image_processor
            )

        mask_boxes = attribution.patches_to_orig_boxes(
            patches, self.grid, W, H, self.image_processor
        )
        masked_img, area = masking.apply_mask(image, mask_boxes, fill)

        # ell(w) on the clean image  -- Eq. (4)
        r_clean = self.scorer.score(self.elicit_prefix, word, image)

        # ell_masked(w) on the occluded image -- Eq. (6).
        # `measure=True` reports how much object signal the PIL-level mask alone
        # had left inside the masked patches; the enforcement then removes it.
        r_masked = self.scorer.score(
            self.elicit_prefix, word, masked_img,
            mask_patches=patches if self.cfg.enforce_pixel_mask else None,
            fill_norm=fill_norm,
            measure=True,
        )

        d = scoring.delta(r_clean["ell"], r_masked["ell"])   # Eq. (7)
        leak = r_masked.get("leak") or {"max_abs_dev": 0.0, "mean_abs_dev": 0.0}

        return {
            "word": word,
            "s_det": s_det,
            "box": box,
            "mask_bbox": attribution.patches_bounding_box(
                patches, self.grid, W, H, self.image_processor
            ),
            "n_patches": geo["n_patches"],
            "patch_frac": geo["patch_frac"],
            "outside_crop": bool(geo["outside_crop"]),
            "clipped_by_crop": bool(geo["clipped_by_crop"]),
            "masked_area_frac": area,
            "n_subword_tokens": r_clean["L"],
            "ell": r_clean["ell"],
            "ell_masked": r_masked["ell"],
            "p": scoring.probability(r_clean["ell"]),
            "p_masked": scoring.probability(r_masked["ell"]),
            "delta": d,
            "conf_drop_pct": scoring.confidence_drop_pct(d),
            "leak_max": leak["max_abs_dev"],
            "leak_mean": leak["mean_abs_dev"],
            "mask_enforced_ok": bool(r_masked["mask_enforced_ok"]),
            "_masked_img": masked_img,   # stripped before JSON; used by the HTML report
        }

    # ------------------------------------------------------------------ #
    def run_image(self, image_file: str, image_id: int, rng: random.Random) -> Dict:
        cfg = self.cfg
        path = f"{cfg.image_folder.rstrip('/')}/{image_file}"
        image = Image.open(path).convert("RGB")

        fill = masking.mean_pixel(image)
        fill_norm = masking.normalised_fill(fill, self.image_processor)

        caption = self.generate_caption(image)
        objects = extraction.extract_objects(self.evaluator, caption)
        gt = extraction.ground_truth_objects(self.evaluator, image_id)

        record = {
            "image_file": image_file,
            "image_id": image_id,
            "caption": caption,
            "gt_objects": sorted(gt),
            "mean_pixel": list(fill),
            "objects": [],
            "probes": [],
            "skipped": None,
            "_image": image,
        }

        if not objects:
            record["skipped"] = "no COCO objects mentioned in the caption"
            return record

        mentioned = {o["object"] for o in objects}

        # GroundingDINO over the whole vocabulary: gives s_det for the probe
        # filter AND a box for every probe, in one batched pass.
        det = self.detector.score_words(image, self.vocabulary)

        probe_words, probe_info = probes.sample_probes(
            vocabulary=self.vocabulary,
            mentioned=mentioned,
            det_scores={k: v["score"] for k, v in det.items()},
            cooccur=self.cooccur,
            K=cfg.K,
            tau_low=cfg.tau_low,
            cooccur_bias=cfg.cooccur_bias,
            rng=rng,
        )
        record["probe_info"] = probe_info

        if len(probe_words) < 2:
            record["skipped"] = f"only {len(probe_words)} probes (need >=2 for sigma_hat)"
            return record

        for o in objects:
            row = self.score_word(o["object"], image, det, fill, fill_norm)
            row["surface"] = o["surface"]
            row["span_idx"] = o["span_idx"]
            row["gt_label"] = extraction.label_object(o["object"], gt)
            record["objects"].append(row)

        for p in probe_words:
            row = self.score_word(p, image, det, fill, fill_norm)
            row["gt_present"] = bool(p in gt)   # diagnostic ONLY; never used by the method
            row.pop("_masked_img", None)        # probes don't get thumbnails
            record["probes"].append(row)

        # Sec 4.3 -- calibration
        mu, sigma = calibration.probe_moments([r["delta"] for r in record["probes"]])
        record["mu_hat"] = mu
        record["sigma_hat"] = sigma
        record["tau"] = calibration.threshold(mu, sigma, cfg.kappa)
        record["kappa"] = cfg.kappa

        for row in record["objects"]:
            row["decision"] = "VERIFIED" if row["delta"] >= record["tau"] else "HALLUCINATED"

        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return record
