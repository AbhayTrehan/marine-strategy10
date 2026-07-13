"""
Algorithm 1, extended to compute SEVERAL candidate scores in one run so the data can
pick the winner instead of us guessing.

Per word w, per image I:

    R(w)  = GroundingDINO boxes  ->  SAM silhouette  ->  ViT patches
            (--seg_backend box reverts to the rectangle)

    C(w)  = a same-shape, same-size patch set placed ELSEWHERE   [--control_mask]

    scores, all from the same masked images:
      delta        = l(w|I) - l(w|I_masked)                 Eq. (4)/(6)/(7), VERBATIM
      delta_lo     = LO(w|I) - LO(w|I_masked)               existence log-odds
      delta_ctrl   = l(w|I_control) - l(w|I_masked)         area-controlled
      delta_lo_ctrl= LO(w|I_control) - LO(w|I_masked)       both
      s_det        = GroundingDINO confidence               THE BASELINE TO BEAT

Every score is calibrated against the SAME probe set with its OWN mu_hat/sigma_hat/tau
(Eq. 8/9), and the report prints AUROC for each. The verify/flag decision is taken on
--primary_score (default `delta`, i.e. the spec's own score) so the headline numbers
remain the spec's numbers; the others are measured alongside.

Forward passes per word: 2 (delta) + 2 (delta_lo) + 2 (control, if on) = 4-6.
"""

import random
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from . import attribution, calibration, extraction, masking, probes, prompts, scoring

SCORES = ["delta", "delta_lo", "delta_ctrl", "delta_lo_ctrl"]


class Strategy10V2Pipeline:
    def __init__(self, cfg, model, tokenizer, processor, detector, evaluator,
                 vocabulary, cooccur, grid, segmenter=None, extra_vocab=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.detector = detector
        self.segmenter = segmenter
        self.evaluator = evaluator
        self.vocabulary = vocabulary
        self.extra_vocab = extra_vocab or []
        self.cooccur = cooccur
        self.grid = grid

        self.dtype = torch.float16 if cfg.fp16 else torch.float32
        self.scorer = scoring.OcclusionScorer(
            model, tokenizer, processor, cfg.device, self.dtype, grid
        )
        self.image_processor = processor.image_processor
        self.elicit_prefix = prompts.elicitation_prefix()
        self.task_prompt = prompts.task_prompt(cfg.task_prompt)

        self.want_lo = "delta_lo" in cfg.scores or "delta_lo_ctrl" in cfg.scores
        self.want_ctrl = cfg.control_mask

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
        out = self.model.generate(**inputs, do_sample=False,
                                  max_new_tokens=self.cfg.max_new_tokens, use_cache=True)
        return self.tokenizer.batch_decode(out[:, n_in:], skip_special_tokens=True)[0].strip()

    # ------------------------------------------------------------------ #
    def _region(self, word: str, image: Image.Image, det: Dict):
        """R(w): GroundingDINO boxes -> SAM silhouette -> ViT patches."""
        W, H = image.size
        entry = det.get(word, {})
        boxes = list(entry.get("boxes") or [])

        seg_used = "box"
        sam_mask = None
        if self.segmenter is not None and boxes:
            sam_mask = self.segmenter.mask_for_boxes(image, boxes)
            if sam_mask is not None:
                seg_used = "sam"

        if sam_mask is not None:
            patches, geo = attribution.mask_to_patches(
                sam_mask, self.grid, W, H, self.image_processor
            )
        else:
            # box fallback: union over all instance boxes
            patches, any_clip, all_out = set(), False, True
            for b in boxes:
                p, g = attribution.box_to_patches(b, self.grid, W, H, self.image_processor)
                patches.update(p)
                any_clip |= bool(g["clipped_by_crop"])
                all_out &= bool(g["outside_crop"])
            patches = sorted(patches)
            geo = {"outside_crop": bool(all_out) if boxes else True,
                   "clipped_by_crop": any_clip,
                   "n_patches": len(patches),
                   "patch_frac": len(patches) / float(self.grid * self.grid)}

        if self.cfg.mask_dilate_patches:
            patches = attribution.dilate_patches(
                patches, self.grid, self.cfg.mask_dilate_patches)
            geo["n_patches"] = len(patches)
            geo["patch_frac"] = len(patches) / float(self.grid * self.grid)

        return patches, geo, boxes, entry, seg_used, sam_mask

    # ------------------------------------------------------------------ #
    def score_word(self, word: str, image: Image.Image, det: Dict,
                   fill: tuple, fill_norm: List[float], rng: random.Random) -> Dict:
        cfg = self.cfg
        W, H = image.size
        patches, geo, boxes, entry, seg_used, sam_mask = self._region(word, image, det)
        s_det = float(entry.get("score", 0.0))

        mask_boxes = attribution.patches_to_orig_boxes(
            patches, self.grid, W, H, self.image_processor)
        masked_img, area = masking.apply_mask(image, mask_boxes, fill)

        ctrl_patches, ctrl_img = [], None
        if self.want_ctrl and patches:
            ctrl_patches = masking.sample_control_patches(patches, self.grid, rng)
            if ctrl_patches:
                cb = attribution.patches_to_orig_boxes(
                    ctrl_patches, self.grid, W, H, self.image_processor)
                ctrl_img, _ = masking.apply_mask(image, cb, fill)

        row = {
            "word": word,
            "s_det": s_det,
            "boxes": boxes,
            "inst_scores": list(entry.get("scores") or []),
            "n_instances": len(boxes),
            "seg": seg_used,
            "mask_bbox": attribution.patches_bounding_box(
                patches, self.grid, W, H, self.image_processor),
            "n_patches": geo["n_patches"],
            "patch_frac": geo["patch_frac"],
            "outside_crop": bool(geo["outside_crop"]),
            "clipped_by_crop": bool(geo.get("clipped_by_crop", False)),
            "masked_area_frac": area,
            "n_ctrl_patches": len(ctrl_patches),
            "_patches": patches,
            "_masked_img": masked_img,
            "scores": {},
        }

        # ---- likelihood head: Eq. (4) / (6) / (7), verbatim -----------------
        clean = self.scorer.score(self.elicit_prefix, word, image)
        msk = self.scorer.score(self.elicit_prefix, word, masked_img,
                                mask_patches=patches, fill_norm=fill_norm, measure=True)
        row.update({
            "ell": clean["ell"], "ell_masked": msk["ell"],
            "p": scoring.probability(clean["ell"]),
            "p_masked": scoring.probability(msk["ell"]),
            "n_subword_tokens": clean["L"],
            "leak_max": (msk.get("leak") or {}).get("max_abs_dev", 0.0),
            "mask_enforced_ok": bool(msk["mask_enforced_ok"]),
        })
        row["scores"]["delta"] = scoring.delta(clean["ell"], msk["ell"])
        row["conf_drop_pct"] = scoring.confidence_drop_pct(row["scores"]["delta"])

        if ctrl_img is not None:
            c = self.scorer.score(self.elicit_prefix, word, ctrl_img,
                                  mask_patches=ctrl_patches, fill_norm=fill_norm)
            row["ell_ctrl"] = c["ell"]
            # "does deleting w's OWN region hurt w more than deleting an equally
            # large irrelevant region?" -- the masked-AREA effect cancels.
            row["scores"]["delta_ctrl"] = c["ell"] - msk["ell"]

        # ---- existence head: log-odds of "is there a w?" --------------------
        if self.want_lo:
            lo_clean = self.scorer.score_existence(word, image)
            lo_msk = self.scorer.score_existence(word, masked_img,
                                                 mask_patches=patches, fill_norm=fill_norm)
            row.update({"lo": lo_clean["lo"], "lo_masked": lo_msk["lo"],
                        "p_yes": lo_clean["p_yes"], "p_yes_masked": lo_msk["p_yes"]})
            row["scores"]["delta_lo"] = lo_clean["lo"] - lo_msk["lo"]

            if ctrl_img is not None:
                lo_c = self.scorer.score_existence(word, ctrl_img,
                                                   mask_patches=ctrl_patches,
                                                   fill_norm=fill_norm)
                row["lo_ctrl"] = lo_c["lo"]
                row["scores"]["delta_lo_ctrl"] = lo_c["lo"] - lo_msk["lo"]

        return row

    # ------------------------------------------------------------------ #
    def run_image(self, image_file: str, image_id: int, rng: random.Random) -> Dict:
        cfg = self.cfg
        image = Image.open(f"{cfg.image_folder.rstrip('/')}/{image_file}").convert("RGB")
        fill = masking.mean_pixel(image)
        fill_norm = masking.normalised_fill(fill, self.image_processor)

        caption = self.generate_caption(image)
        objects = extraction.extract_objects_extended(
            self.evaluator, caption, self.extra_vocab if cfg.extract_non_coco else None)
        gt = extraction.ground_truth_objects(self.evaluator, image_id)

        rec = {
            "image_file": image_file, "image_id": image_id, "caption": caption,
            "gt_objects": sorted(gt), "mean_pixel": list(fill),
            "objects": [], "probes": [], "skipped": None, "_image": image,
        }
        if not objects:
            rec["skipped"] = "no objects extracted from the response"
            return rec

        mentioned = {o["object"] for o in objects if o.get("in_coco", True)}

        # every word we will need a region for, in ONE detector pass
        need = sorted(set(self.vocabulary) | {o["object"] for o in objects})
        det = self.detector.score_words(image, need)

        probe_words, probe_info = probes.sample_probes(
            vocabulary=self.vocabulary, mentioned=mentioned,
            det_scores={k: v["score"] for k, v in det.items()},
            cooccur=self.cooccur, K=cfg.K, tau_low=cfg.tau_low,
            cooccur_bias=cfg.cooccur_bias, rng=rng,
        )
        rec["probe_info"] = probe_info
        if len(probe_words) < 2:
            rec["skipped"] = f"only {len(probe_words)} probes (need >=2 for sigma_hat)"
            return rec

        for o in objects:
            row = self.score_word(o["object"], image, det, fill, fill_norm, rng)
            row["surface"] = o["surface"]
            row["span_idx"] = o["span_idx"]
            row["in_coco"] = bool(o.get("in_coco", True))
            # An object with no COCO class has NO ground truth -- COCO does not
            # annotate "headphones". It is scored and shown, but it cannot enter any
            # metric, because we would have to invent its label to do so.
            row["gt_label"] = (extraction.label_object(o["object"], gt)
                               if row["in_coco"] else "UNKNOWN")
            rec["objects"].append(row)

        for p in probe_words:
            row = self.score_word(p, image, det, fill, fill_norm, rng)
            row["gt_present"] = bool(p in gt)   # diagnostic only; never used
            row.pop("_masked_img", None)
            rec["probes"].append(row)

        # ---- Sec 4.3: calibrate EVERY score against the SAME probes ---------
        rec["null"] = {}
        for s in SCORES:
            vals = [r["scores"][s] for r in rec["probes"] if s in r["scores"]]
            if len(vals) < 2:
                continue
            mu, sd = calibration.probe_moments(vals)
            rec["null"][s] = {"mu": mu, "sigma": sd,
                              "tau": calibration.threshold(mu, sd, cfg.kappa)}

        prim = cfg.primary_score
        rec["mu_hat"] = rec["null"].get(prim, {}).get("mu", float("nan"))
        rec["sigma_hat"] = rec["null"].get(prim, {}).get("sigma", float("nan"))
        rec["tau"] = rec["null"].get(prim, {}).get("tau", float("nan"))
        rec["kappa"] = cfg.kappa
        rec["primary_score"] = prim

        for row in rec["objects"]:
            v = row["scores"].get(prim)
            row["delta"] = row["scores"].get("delta")   # kept for the console table
            row["decision"] = ("VERIFIED" if (v is not None and v >= rec["tau"])
                               else "HALLUCINATED")
        for row in rec["probes"]:
            row["delta"] = row["scores"].get("delta")

        # ---- duplicate / colliding regions ----------------------------------
        # If two mentions mask the same patches, their Deltas are not independent
        # measurements of two different objects -- they are one measurement counted
        # twice. Flag it rather than let it quietly correlate the results.
        objs = rec["objects"]
        for i, a in enumerate(objs):
            dup = []
            for j, b in enumerate(objs):
                if i == j:
                    continue
                iou = attribution.patch_iou(a["_patches"], b["_patches"])
                if iou >= cfg.dup_iou:
                    dup.append({"word": b["word"], "iou": round(iou, 3)})
            a["region_collisions"] = dup
        for a in objs:
            a.pop("_patches", None)
        for p in rec["probes"]:
            p.pop("_patches", None)

        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return rec
