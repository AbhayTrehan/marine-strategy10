#!/usr/bin/env python
"""
Strategy 10 (v2) -- SANITY CHECK  (GroundingDINO / patch-aligned masking).

Runs Stages 1-2 on N COCO images and reports, for EVERY object the LVLM mentioned
in its own unguided caption:

    * where GroundingDINO localised it
    * the model's confidence in the word before occlusion   -- ell(w),  Eq. (4)
    * ...and after its region is masked out                 -- ell_masked(w), Eq. (6)
    * the causal evidence score                             -- Delta = ell - ell_masked, Eq. (7)
    * whether Strategy 10 (v2) calls it HALLUCINATED or VERIFIED  -- Eq. (11)/(12)
    * the COCO ground-truth label (CHAIR's own definition)

and in aggregate: % of actual hallucinations CAUGHT, % of actual real objects
FALSE-FLAGGED, and the kappa-free AUROC ceiling of Delta.

Stage 3 (the LLM rewriter, Sec 5) is deliberately NOT run.

Usage, from the MARINE repo root:

    python ./scripts/eval_strategy10_v2_sanity.py --num_images 50

Outputs land in ./output/strategy10_v2/ :
    report.html       <- the readable one: every image, every box, every number
    records.json      <- every Delta, so kappa can be re-swept with no GPU
    per_object.csv
    sanity_report.txt
"""

import argparse
import json
import os
import random
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from marine.strategy10_v2 import report, report_html  # noqa: E402
from marine.strategy10_v2.config import Strategy10V2Config  # noqa: E402
from marine.strategy10_v2.detector import GroundingDinoDetector  # noqa: E402
from marine.strategy10_v2.extraction import (  # noqa: E402
    build_chair_for_images, coco_categories, load_cooccurrence, load_questions,
)
from marine.strategy10_v2.model_loader import describe_visual_grid, load_lvlm  # noqa: E402
from marine.strategy10_v2.pipeline import Strategy10V2Pipeline  # noqa: E402


def parse_args():
    d = Strategy10V2Config()
    p = argparse.ArgumentParser(description="Strategy 10 (v2) sanity check")

    p.add_argument("--model_path", type=str, default=d.model_path)
    p.add_argument("--detector_path", type=str, default=d.detector_path)
    p.add_argument("--fp32", action="store_true", help="load the LVLM in fp32 (default fp16)")
    p.add_argument("--detector_fp16", action="store_true",
                   help="run GroundingDINO in fp16 (default fp32; DETR heads are fp16-twitchy)")
    p.add_argument("--device", type=str, default=d.device)

    p.add_argument("--image_folder", type=str, default=d.image_folder)
    p.add_argument("--coco_annotations", type=str, default=d.coco_annotations)
    p.add_argument("--chair_cache", type=str, default=d.chair_cache)
    p.add_argument("--question_file", type=str, default=d.question_file)
    p.add_argument("--cooccur_file", type=str, default=d.cooccur_file)
    p.add_argument("--num_images", type=int, default=d.num_images)

    p.add_argument("--max_new_tokens", type=int, default=d.max_new_tokens)
    p.add_argument("--seed", type=int, default=d.seed)

    p.add_argument("--det_prompt_template", type=str, default=d.det_prompt_template)
    p.add_argument("--det_batch_size", type=int, default=d.det_batch_size)

    p.add_argument("--K", type=int, default=d.K)
    p.add_argument("--tau_low", type=float, default=d.tau_low)
    p.add_argument("--cooccur_bias", type=float, default=d.cooccur_bias)

    p.add_argument("--kappa", type=float, default=d.kappa)
    p.add_argument("--kappa_sweep", type=str, default=",".join(map(str, d.kappa_sweep)))

    p.add_argument("--no_enforce_pixel_mask", action="store_true",
                   help="DIAGNOSTIC ONLY: skip re-asserting the mask on pixel_values. "
                        "Leaves resize-interpolation leakage in the masked patches; "
                        "use it to see how much that leakage was worth.")

    p.add_argument("--output_dir", type=str, default=d.output_dir)
    p.add_argument("--html_max_width", type=int, default=d.html_max_width)
    p.add_argument("--from_cache", type=str, default=None,
                   help="path to a previous records.json; re-reports without touching a GPU")

    a = p.parse_args()

    cfg = Strategy10V2Config(
        model_path=a.model_path, detector_path=a.detector_path,
        fp16=not a.fp32, detector_fp16=a.detector_fp16, device=a.device,
        image_folder=a.image_folder, coco_annotations=a.coco_annotations,
        chair_cache=a.chair_cache, question_file=a.question_file,
        cooccur_file=a.cooccur_file, num_images=a.num_images,
        max_new_tokens=a.max_new_tokens, seed=a.seed,
        det_prompt_template=a.det_prompt_template, det_batch_size=a.det_batch_size,
        K=a.K, tau_low=a.tau_low, cooccur_bias=a.cooccur_bias,
        kappa=a.kappa,
        kappa_sweep=[float(x) for x in a.kappa_sweep.split(",") if x.strip()],
        enforce_pixel_mask=not a.no_enforce_pixel_mask,
        output_dir=a.output_dir, html_max_width=a.html_max_width,
    )
    return cfg, a


def strip_assets(rec):
    """Drop PIL images (not JSON-serialisable, and huge) but keep every number."""
    out = {k: v for k, v in rec.items() if not k.startswith("_")}
    out["objects"] = [{k: v for k, v in o.items() if not k.startswith("_")}
                      for o in rec.get("objects", [])]
    out["probes"] = [{k: v for k, v in p.items() if not k.startswith("_")}
                     for p in rec.get("probes", [])]
    return out


def write_csv(records, path):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image_file", "image_id", "role", "word", "surface", "s_det",
            "n_patches", "patch_frac", "outside_crop", "leak_max",
            "ell", "ell_masked", "delta", "conf_drop_pct",
            "mu_hat", "sigma_hat", "tau", "decision", "gt_label", "outcome",
        ])
        for rec in records:
            if rec.get("skipped"):
                continue
            for role, rows in (("candidate", rec["objects"]), ("probe", rec["probes"])):
                for r in rows:
                    w.writerow([
                        rec["image_file"], rec["image_id"], role, r["word"],
                        r.get("surface", ""), f"{r['s_det']:.4f}",
                        r["n_patches"], f"{r['patch_frac']:.4f}",
                        int(bool(r.get("outside_crop"))), f"{r.get('leak_max', 0.0):.3e}",
                        f"{r['ell']:.5f}", f"{r['ell_masked']:.5f}",
                        f"{r['delta']:.5f}", f"{r['conf_drop_pct']:.3f}",
                        f"{rec['mu_hat']:.5f}", f"{rec['sigma_hat']:.5f}", f"{rec['tau']:.5f}",
                        r.get("decision", ""),
                        r.get("gt_label", "GT_PRESENT" if r.get("gt_present") else "GT_ABSENT"),
                        report._outcome(r["decision"], r["gt_label"]).split()[0]
                        if role == "candidate" else "",
                    ])


def main():
    cfg, args = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---- re-report from cache (no GPU, no models) --------------------------
    if args.from_cache:
        with open(args.from_cache) as f:
            blob = json.load(f)
        records = blob["records"]
        saved = blob.get("config") or {}
        fields = set(Strategy10V2Config().__dict__)
        merged = {k: v for k, v in saved.items() if k in fields}
        merged["kappa"] = cfg.kappa
        merged["kappa_sweep"] = cfg.kappa_sweep
        cached_cfg = Strategy10V2Config(**merged) if merged else cfg
        print(report.format_summary(records, cached_cfg))
        html_path = os.path.join(cfg.output_dir, "report.html")
        with open(html_path, "w") as f:
            f.write(report_html.build_html(records, cached_cfg))
        print(f"\n[saved] {html_path}  (images unavailable from cache; numbers are complete)")
        return

    from transformers import set_seed

    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    # ---- assets ------------------------------------------------------------
    questions = load_questions(cfg.question_file, cfg.num_images)
    print(f"[setup] {len(questions)} images from {cfg.question_file}")

    print("[setup] building ground truth for just these images...")
    evaluator = build_chair_for_images(REPO_ROOT, cfg.coco_annotations, questions, cfg.chair_cache)
    vocabulary = coco_categories(evaluator)
    print(f"[setup] probe vocabulary V: {len(vocabulary)} canonical COCO categories")

    cooccur = load_cooccurrence(cfg.cooccur_file)

    print(f"[setup] loading LVLM {cfg.model_path} ({'fp16' if cfg.fp16 else 'fp32'})...")
    model, tokenizer, processor = load_lvlm(cfg.model_path, cfg.fp16, cfg.device)
    grid, n_image_tokens = describe_visual_grid(model)
    print(f"[setup] vision grid {grid}x{grid} -> {n_image_tokens} patch tokens")

    print(f"[setup] loading localiser {cfg.detector_path} "
          f"({'fp16' if cfg.detector_fp16 else 'fp32'})...")
    detector = GroundingDinoDetector(
        cfg.detector_path, cfg.device, cfg.detector_fp16,
        prompt_template=cfg.det_prompt_template, batch_size=cfg.det_batch_size,
    )

    pipe = Strategy10V2Pipeline(
        cfg, model, tokenizer, processor, detector, evaluator, vocabulary, cooccur, grid
    )
    print(f"[setup] c_elicit = {pipe.elicit_prefix!r}")
    print(f"[setup] mask: patch-aligned; pixel_values enforcement = {cfg.enforce_pixel_mask}")
    print()

    # ---- run ---------------------------------------------------------------
    records = []
    t0 = time.time()
    for i, (image_file, image_id) in enumerate(questions, start=1):
        if i == 1:
            # No try/except on the first image: a systemic problem should fail
            # loudly and immediately, not repeat silently 50 times.
            rec = pipe.run_image(image_file, image_id, rng)
        else:
            try:
                rec = pipe.run_image(image_file, image_id, rng)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                rec = {"image_file": image_file, "image_id": image_id, "caption": "",
                       "gt_objects": [], "objects": [], "probes": [],
                       "skipped": f"ERROR: {type(exc).__name__}: {exc}"}
        records.append(rec)
        print(report.format_image_block(rec, i, len(questions)), flush=True)

    elapsed = time.time() - t0

    # ---- report ------------------------------------------------------------
    summary = report.format_summary(records, cfg)
    print(summary)
    print(f"\n[timing] {elapsed / 60:.1f} min total, "
          f"{elapsed / max(len(questions), 1):.1f} s/image\n")

    html_path = os.path.join(cfg.output_dir, "report.html")
    with open(html_path, "w") as f:
        f.write(report_html.build_html(records, cfg))

    clean = [strip_assets(r) for r in records]

    json_path = os.path.join(cfg.output_dir, "records.json")
    with open(json_path, "w") as f:
        json.dump({"config": cfg.__dict__, "records": clean}, f, indent=1)

    csv_path = os.path.join(cfg.output_dir, "per_object.csv")
    write_csv(clean, csv_path)

    txt_path = os.path.join(cfg.output_dir, "sanity_report.txt")
    with open(txt_path, "w") as f:
        for i, rec in enumerate(records, start=1):
            f.write(report.format_image_block(rec, i, len(records)) + "\n")
        f.write(summary + "\n")

    print(f"[saved] {html_path}   <-- open this one")
    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {txt_path}")
    print(f"\nRe-sweep kappa with no GPU:  "
          f"python ./scripts/eval_strategy10_v2_sanity.py --from_cache {json_path} --kappa 1.5")


if __name__ == "__main__":
    main()
