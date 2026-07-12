"""
Algorithm 1 -- per-image orchestration.

    Stage 1  y <- M(.|x, I);  O <- ExtractCanonicalObjects(y)
    Stage 2  P <- SampleAbsentProbes(V, O, K, tau_low)
             for w in O U P:  R(w) -> I_masked(w) -> ell, ell_masked -> Delta(w)
             mu_hat, sigma_hat from {Delta(p)};  tau = mu_hat + kappa*sigma_hat
             O_pos = {Delta >= tau};  H = O \\ O_pos
    Stage 3  (rewriting -- NOT run: this is the sanity check only)

Everything downstream of Delta is cheap and cached, so the report can re-derive
the verify/flag decision for any kappa without re-running the model.
"""

import random
from typing import Dict, List

import torch
from PIL import Image

from . import attribution, calibration, extraction, masking, probes, prompts, scoring
from .config import TASK_PROMPT


class Strategy10V2Pipeline:
    def __init__(self, cfg, model, tokenizer, processor, detector, evaluator,
                 vocabulary, cooccur, grid, n_image_tokens, skip_cls):
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
            model, tokenizer, processor, cfg.device, self.dtype,
            grid, n_image_tokens, skip_cls,
        )
        self.n_image_tokens = n_image_tokens
        self.skip_cls = skip_cls
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
            **inputs,
            do_sample=False,                 # greedy, matching the repo's CHAIR setting
            max_new_tokens=self.cfg.max_new_tokens,
            use_cache=True,
        )
        text = self.tokenizer.batch_decode(
            out_ids[:, n_in:], skip_special_tokens=True
        )[0]
        return text.strip()

    # ------------------------------------------------------------------ #
    # Sec 3.3 + 3.4: R(w) -> Delta(w).  Identical code path for candidate & probe.
    # ------------------------------------------------------------------ #
    def score_word(self, word: str, image: Image.Image, det: Dict) -> Dict:
        cfg = self.cfg
        W, H = image.size

        s_det = float(det.get(word, {}).get("score", 0.0))
        det_box = det.get(word, {}).get("box")
        use_detector = (s_det >= cfg.tau_box) and (det_box is not None)

        # --- unmasked pass (Eq. 4); ask for attention only if we'll need it ---
        ell, L, attns, row_idx, input_ids_row = self.scorer.score(
            self.elicit_prefix, word, image, want_attentions=not use_detector
        )

        capped = False
        if use_detector:
            boxes = [det_box]
            source = "det"
        else:
            # --- attention fallback (Eq. 5) ---------------------------------
            seq_len = int(attns[0].shape[-1])
            img_pos = attribution.image_token_positions(
                input_ids_row, seq_len, self.scorer.image_token_index, self.n_image_tokens
            )
            a = attribution.aggregate_attention(
                attns, row_idx, img_pos, cfg.attn_layers, cfg.attn_heads, self.skip_cls
            )
            patch_idx = attribution.top_mass_patches(a, cfg.rho, cfg.max_patch_frac)
            boxes = attribution.patches_to_boxes(
                patch_idx.tolist(), self.grid, W, H, self.processor.image_processor
            )
            source = "attn"
            # If the max_patch_frac safety valve is binding, every attention-routed
            # word gets an identically-sized region and the attention signal has
            # effectively been thrown away. Surface it rather than hiding it.
            n_patches = int(a.numel())
            capped = len(patch_idx) >= max(1, int(cfg.max_patch_frac * n_patches))

        del attns

        # --- mask and re-encode in full (Eq. 6) -----------------------------
        masked_img, area = masking.apply_mask(image, boxes, self._mean_pixel)
        ell_masked, L2, _, _, _ = self.scorer.score(
            self.elicit_prefix, word, masked_img, want_attentions=False
        )

        d = scoring.delta(ell, ell_masked)
        return {
            "word": word,
            "s_det": s_det,
            "region_source": source,
            "region_capped": bool(capped),
            "n_boxes": len(boxes),
            "masked_area_frac": area,
            "n_subword_tokens": L,
            "ell": ell,
            "ell_masked": ell_masked,
            "delta": d,
            "conf_drop_pct": scoring.confidence_drop_pct(d),
        }

    # ------------------------------------------------------------------ #
    def run_image(self, image_file: str, image_id: int, rng: random.Random) -> Dict:
        cfg = self.cfg
        path = f"{cfg.image_folder.rstrip('/')}/{image_file}"
        image = Image.open(path).convert("RGB")
        self._mean_pixel = masking.mean_pixel(image)

        # -------- Stage 1 ------------------------------------------------
        caption = self.generate_caption(image)
        objects = extraction.extract_objects(self.evaluator, caption)
        gt = extraction.ground_truth_objects(self.evaluator, image_id)

        record = {
            "image_file": image_file,
            "image_id": image_id,
            "caption": caption,
            "gt_objects": sorted(gt),
            "objects": [],
            "probes": [],
            "skipped": None,
        }

        if not objects:
            record["skipped"] = "no COCO objects mentioned in caption"
            return record

        mentioned = {o["object"] for o in objects}

        # -------- one OWL-ViT pass scores the whole vocabulary ------------
        det = self.detector.score_vocabulary(image, self.vocabulary)

        # -------- Sec 4.1: probes ----------------------------------------
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
            record["skipped"] = f"only {len(probe_words)} probes available (need >=2 for sigma)"
            return record

        # -------- Sec 3/4.2: Delta for candidates AND probes, same code path
        for o in objects:
            row = self.score_word(o["object"], image, det)
            row["surface"] = o["surface"]
            row["span_idx"] = o["span_idx"]
            row["gt_label"] = extraction.label_object(o["object"], gt)
            record["objects"].append(row)

        for p in probe_words:
            row = self.score_word(p, image, det)
            # diagnostic ONLY -- never used by the method
            row["gt_present"] = bool(p in gt)
            record["probes"].append(row)

        # -------- Sec 4.3: calibration -----------------------------------
        probe_deltas = [r["delta"] for r in record["probes"]]
        mu, sigma = calibration.probe_moments(probe_deltas)
        record["mu_hat"] = mu
        record["sigma_hat"] = sigma
        record["tau"] = calibration.threshold(mu, sigma, cfg.kappa)
        record["kappa"] = cfg.kappa

        for row in record["objects"]:
            row["decision"] = (
                "VERIFIED" if row["delta"] >= record["tau"] else "HALLUCINATED"
            )

        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return record
