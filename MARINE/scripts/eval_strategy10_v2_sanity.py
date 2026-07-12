#!/usr/bin/env python
"""
Strategy 10 (v2) -- SANITY CHECK.

Runs the first two stages of Strategy 10 (v2) on N COCO images and reports, per
mentioned object:

    * the objects the LVLM mentioned in its own unguided caption
    * the % change in the model's confidence in that object after its attributed
      visual region is occluded  (== 100 * (1 - exp(-Delta)))
    * whether Strategy 10 (v2) marks it HALLUCINATED or VERIFIED
    * the COCO ground-truth label (REAL / HALLUCINATED, per CHAIR's definition)

and in aggregate:

    * % of actually-hallucinated objects the strategy CAUGHT
    * % of actually-real objects the strategy FALSE-FLAGGED

Stage 3 (the LLM rewriter, Sec 5) is deliberately NOT run -- this is a
verification-stage sanity check only.

Usage (from the MARINE repo root):

    python ./scripts/eval_strategy10_v2_sanity.py --num_images 50

Everything is cached to JSON, so kappa can be re-swept afterwards with
--from_cache without touching the GPU.
"""

import argparse
import json
import os
import random
import sys
import time

# make `marine` importable when run as ./scripts/eval_strategy10_v2_sanity.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from marine.strategy10_v2 import report  # noqa: E402
from marine.strategy10_v2.config import Strategy10V2Config  # noqa: E402
from marine.strategy10_v2.detector import OwlViTDetector  # noqa: E402
from marine.strategy10_v2.extraction import (  # noqa: E402
    build_chair_for_images,
    coco_categories,
    load_cooccurrence,
    load_questions,
)
from marine.strategy10_v2.model_loader import describe_visual_grid, load_lvlm  # noqa: E402
from marine.strategy10_v2.pipeline import Strategy10V2Pipeline  # noqa: E402


def parse_args():
    d = Strategy10V2Config()
    p = argparse.ArgumentParser(description="Strategy 10 (v2) sanity check")

    p.add_argument("--model_path", type=str, default=d.model_path)
    p.add_argument("--detector_path", type=str, default=d.detector_path)
    p.add_argument("--fp32", action="store_true", help="load the LVLM in fp32 (default: fp16)")
    p.add_argument("--device", type=str, default=d.device)

    p.add_argument("--image_folder", type=str, default=d.image_folder)
    p.add_argument("--coco_annotations", type=str, default=d.coco_annotations)
    p.add_argument("--chair_cache", type=str, default=d.chair_cache)
    p.add_argument("--question_file", type=str, default=d.question_file)
    p.add_argument("--cooccur_file", type=str, default=d.cooccur_file)
    p.add_argument("--num_images", type=int, default=d.num_images)

    p.add_argument("--max_new_tokens", type=int, default=d.max_new_tokens)
    p.add_argument("--seed", type=int, default=d.seed)

    p.add_argument("--tau_box", type=float, default=d.tau_box)
    p.add_argument("--tau_low", type=float, default=d.tau_low)
    p.add_argument("--rho", type=float, default=d.rho)
    p.add_argument("--attn_layers", type=str, default=",".join(map(str, d.attn_layers)))
    p.add_argument("--max_patch_frac", type=float, default=d.max_patch_frac)

    p.add_argument("--K", type=int, default=d.K)
    p.add_argument("--cooccur_bias", type=float, default=d.cooccur_bias)

    p.add_argument("--kappa", type=float, default=d.kappa)
    p.add_argument("--kappa_sweep", type=str,
                   default=",".join(map(str, d.kappa_sweep)))

    p.add_argument("--output_dir", type=str, default=d.output_dir)
    p.add_argument("--from_cache", type=str, default=None,
                   help="path to a previous records.json; re-reports without running the models")

    a = p.parse_args()

    cfg = Strategy10V2Config(
        model_path=a.model_path,
        detector_path=a.detector_path,
        fp16=not a.fp32,
        device=a.device,
        image_folder=a.image_folder,
        coco_annotations=a.coco_annotations,
        chair_cache=a.chair_cache,
        question_file=a.question_file,
        cooccur_file=a.cooccur_file,
        num_images=a.num_images,
        max_new_tokens=a.max_new_tokens,
        seed=a.seed,
        tau_box=a.tau_box,
        tau_low=a.tau_low,
        rho=a.rho,
        attn_layers=[int(x) for x in a.attn_layers.split(",") if x.strip() != ""],
        max_patch_frac=a.max_patch_frac,
        K=a.K,
        cooccur_bias=a.cooccur_bias,
        kappa=a.kappa,
        kappa_sweep=[float(x) for x in a.kappa_sweep.split(",") if x.strip() != ""],
        output_dir=a.output_dir,
    )
    return cfg, a


def write_csv(records, path):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "image_file", "image_id", "role", "word", "surface", "s_det",
            "region_source", "masked_area_frac", "n_subword_tokens",
            "ell", "ell_masked", "delta", "conf_drop_pct",
            "mu_hat", "sigma_hat", "tau", "decision", "gt_label", "outcome",
        ])
        for rec in records:
            if rec.get("skipped"):
                continue
            for r in rec["objects"]:
                w.writerow([
                    rec["image_file"], rec["image_id"], "candidate", r["word"], r["surface"],
                    f"{r['s_det']:.4f}", r["region_source"], f"{r['masked_area_frac']:.4f}",
                    r["n_subword_tokens"], f"{r['ell']:.5f}", f"{r['ell_masked']:.5f}",
                    f"{r['delta']:.5f}", f"{r['conf_drop_pct']:.3f}",
                    f"{rec['mu_hat']:.5f}", f"{rec['sigma_hat']:.5f}", f"{rec['tau']:.5f}",
                    r["decision"], r["gt_label"],
                    report._outcome(r["decision"], r["gt_label"]).split()[0],
                ])
            for r in rec["probes"]:
                w.writerow([
                    rec["image_file"], rec["image_id"], "probe", r["word"], "",
                    f"{r['s_det']:.4f}", r["region_source"], f"{r['masked_area_frac']:.4f}",
                    r["n_subword_tokens"], f"{r['ell']:.5f}", f"{r['ell_masked']:.5f}",
                    f"{r['delta']:.5f}", f"{r['conf_drop_pct']:.3f}",
                    f"{rec['mu_hat']:.5f}", f"{rec['sigma_hat']:.5f}", f"{rec['tau']:.5f}",
                    "", "GT_PRESENT" if r.get("gt_present") else "GT_ABSENT", "",
                ])


def main():
    cfg, args = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ---------------- re-report from cache (no GPU) -------------------------
    if args.from_cache:
        with open(args.from_cache) as f:
            blob = json.load(f)
        records = blob["records"]

        # Report against the hyperparameters the run ACTUALLY used, not argparse
        # defaults -- otherwise the Setup block would silently lie. kappa (and the
        # sweep) are the only things re-thresholding can legitimately change, so
        # those alone are taken from the CLI.
        saved = blob.get("config") or {}
        fields = set(Strategy10V2Config().__dict__)
        merged = {k: v for k, v in saved.items() if k in fields}
        merged["kappa"] = cfg.kappa
        merged["kappa_sweep"] = cfg.kappa_sweep
        cached_cfg = Strategy10V2Config(**merged) if merged else cfg

        print(report.format_summary(records, cached_cfg))
        return

    from transformers import set_seed

    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    # ---------------- assets ------------------------------------------------
    questions = load_questions(cfg.question_file, cfg.num_images)
    print(f"[setup] {len(questions)} images from {cfg.question_file}")

    print("[setup] building ground truth (object extraction + COCO GT) "
          "for just these images...")
    evaluator = build_chair_for_images(REPO_ROOT, cfg.coco_annotations, questions, cfg.chair_cache)
    vocabulary = coco_categories(evaluator)
    print(f"[setup] probe vocabulary V: {len(vocabulary)} canonical COCO categories")

    cooccur = load_cooccurrence(cfg.cooccur_file)

    print(f"[setup] loading LVLM {cfg.model_path} ({'fp16' if cfg.fp16 else 'fp32'}, eager attn)...")
    model, tokenizer, processor = load_lvlm(cfg.model_path, cfg.fp16, cfg.device)
    grid, n_image_tokens, skip_cls = describe_visual_grid(model)
    print(f"[setup] vision grid {grid}x{grid} -> {n_image_tokens} image tokens (skip_cls={skip_cls})")

    print(f"[setup] loading zero-shot detector {cfg.detector_path}...")
    detector = OwlViTDetector(cfg.detector_path, cfg.device, cfg.fp16)

    pipe = Strategy10V2Pipeline(
        cfg, model, tokenizer, processor, detector, evaluator,
        vocabulary, cooccur, grid, n_image_tokens, skip_cls,
    )
    print(f"[setup] c_elicit = {pipe.elicit_prefix!r}")
    print()

    # ---------------- run ---------------------------------------------------
    records = []
    t0 = time.time()
    for i, (image_file, image_id) in enumerate(questions, start=1):
        if i == 1:
            # No try/except on the first image. If something about the
            # environment is wrong (wrong transformers version, misaligned
            # image-token handling, OOM, etc.) this fails immediately and
            # clearly with one full traceback -- instead of silently
            # repeating the same failure across all N images and burying the
            # real cause under 50 near-identical stack traces.
            rec = pipe.run_image(image_file, image_id, rng)
        else:
            try:
                rec = pipe.run_image(image_file, image_id, rng)
            except Exception as exc:  # a single bad image shouldn't kill the run
                import traceback

                traceback.print_exc()
                rec = {
                    "image_file": image_file, "image_id": image_id, "caption": "",
                    "gt_objects": [], "objects": [], "probes": [],
                    "skipped": f"ERROR: {type(exc).__name__}: {exc}",
                }
        records.append(rec)
        print(report.format_image_block(rec, i, len(questions)), flush=True)

    elapsed = time.time() - t0

    # ---------------- report -----------------------------------------------
    summary = report.format_summary(records, cfg)
    print(summary)
    print(f"\n[timing] {elapsed / 60:.1f} min total, "
          f"{elapsed / max(len(questions), 1):.1f} s/image\n")

    json_path = os.path.join(cfg.output_dir, "records.json")
    with open(json_path, "w") as f:
        json.dump({"config": cfg.__dict__, "records": records}, f, indent=1)

    csv_path = os.path.join(cfg.output_dir, "per_object.csv")
    write_csv(records, csv_path)

    txt_path = os.path.join(cfg.output_dir, "sanity_report.txt")
    with open(txt_path, "w") as f:
        for i, rec in enumerate(records, start=1):
            f.write(report.format_image_block(rec, i, len(records)) + "\n")
        f.write(summary + "\n")

    print(f"[saved] {json_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {txt_path}")
    print(f"\nRe-sweep kappa with no GPU:  "
          f"python ./scripts/eval_strategy10_v2_sanity.py --from_cache {json_path} --kappa 1.5")


if __name__ == "__main__":
    main()
