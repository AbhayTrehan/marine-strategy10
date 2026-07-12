"""
Sanity-check reporting.

The two headline numbers the sanity check exists to produce:

  * catch rate  = % of ACTUALLY hallucinated objects that Strategy 10 (v2) flags
                = TP / (TP + FN)                     [recall on hallucinations]
  * false-flag  = % of ACTUALLY real objects that it wrongly flags
                = FP / (FP + TN)                     [FPR]

"Actually hallucinated" is CHAIR's own definition: the canonical object is not
in the image's COCO ground-truth object set.

Because every Delta is cached, the kappa sweep is free -- no extra forward
passes. That sweep IS the empirical selection procedure Sec 4.3.2 prescribes
("selected empirically ... by sweeping kappa and inspecting the resulting
precision/recall trade-off").
"""

from typing import Dict, List

from . import calibration


def _fmt(x, w=8, p=3):
    if x is None:
        return "-".rjust(w)
    try:
        return f"{x:{w}.{p}f}"
    except (TypeError, ValueError):
        return str(x).rjust(w)


def _pct(num, den):
    return 100.0 * num / den if den else float("nan")


# --------------------------------------------------------------------------- #
# per-image
# --------------------------------------------------------------------------- #

def format_image_block(rec: Dict, idx: int, total: int) -> str:
    lines = []
    lines.append("=" * 118)
    lines.append(f"[{idx:>3}/{total}] {rec['image_file']}   (image_id={rec['image_id']})")
    lines.append(f"  caption : {rec['caption']}")
    lines.append(f"  GT COCO : {', '.join(rec['gt_objects']) if rec['gt_objects'] else '(none)'}")

    if rec.get("skipped"):
        lines.append(f"  SKIPPED : {rec['skipped']}")
        return "\n".join(lines)

    n_contam = sum(1 for p in rec["probes"] if p.get("gt_present"))
    lines.append(
        f"  probes  : K={len(rec['probes'])}  "
        f"(GT-present probes: {n_contam} = {_pct(n_contam, len(rec['probes'])):.1f}%"
        f"{'  [tau_low relaxed]' if rec.get('probe_info', {}).get('relaxed_tau_low') else ''})"
    )
    lines.append(
        f"  null    : mu_hat={rec['mu_hat']:+.4f}  sigma_hat={rec['sigma_hat']:.4f}  "
        f"-> tau(kappa={rec['kappa']}) = {rec['tau']:+.4f}"
    )
    lines.append("")
    lines.append(
        "  {:<14} {:<12} {:>6} {:>6} {:>7} {:>8} {:>8} {:>8} {:>9}  {:<13} {:<13} {}".format(
            "object", "surface", "s_det", "region", "area%",
            "ell", "ell_msk", "Delta", "conf_drop", "decision", "ground truth", "outcome",
        )
    )
    lines.append("  " + "-" * 114)

    for r in rec["objects"]:
        outcome = _outcome(r["decision"], r["gt_label"])
        lines.append(
            "  {:<14} {:<12} {:>6.3f} {:>6} {:>6.1f}% {:>8.3f} {:>8.3f} {:>+8.3f} {:>+8.1f}%  {:<13} {:<13} {}".format(
                r["word"][:14], str(r["surface"])[:12], r["s_det"], r["region_source"],
                100.0 * r["masked_area_frac"], r["ell"], r["ell_masked"],
                r["delta"], r["conf_drop_pct"], r["decision"], r["gt_label"], outcome,
            )
        )
    return "\n".join(lines)


def _outcome(decision: str, gt_label: str) -> str:
    if decision == "HALLUCINATED" and gt_label == "HALLUCINATED":
        return "TP  (caught)"
    if decision == "HALLUCINATED" and gt_label == "REAL":
        return "FP  (false flag)"
    if decision == "VERIFIED" and gt_label == "HALLUCINATED":
        return "FN  (missed)"
    return "TN  (ok)"


# --------------------------------------------------------------------------- #
# aggregate
# --------------------------------------------------------------------------- #

def confusion_at_kappa(records: List[Dict], kappa: float) -> Dict:
    """Re-derive the verify/flag decision at an arbitrary kappa from cached Deltas."""
    TP = FP = FN = TN = 0
    for rec in records:
        if rec.get("skipped"):
            continue
        tau = calibration.threshold(rec["mu_hat"], rec["sigma_hat"], kappa)
        for r in rec["objects"]:
            flagged = r["delta"] < tau
            hallucinated = r["gt_label"] == "HALLUCINATED"
            if flagged and hallucinated:
                TP += 1
            elif flagged and not hallucinated:
                FP += 1
            elif (not flagged) and hallucinated:
                FN += 1
            else:
                TN += 1

    return {
        "kappa": kappa,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "catch_rate": _pct(TP, TP + FN),          # % of real hallucinations caught
        "false_flag_rate": _pct(FP, FP + TN),     # % of real objects wrongly flagged
        "precision": _pct(TP, TP + FP),
        "flag_rate": _pct(TP + FP, TP + FP + FN + TN),
        "balanced_acc": 0.5 * (_pct(TP, TP + FN) + _pct(TN, TN + FP)),
    }


def format_summary(records: List[Dict], cfg) -> str:
    L = []
    n_total = len(records)
    used = [r for r in records if not r.get("skipped")]
    skipped = [r for r in records if r.get("skipped")]

    n_obj = sum(len(r["objects"]) for r in used)
    n_hall = sum(1 for r in used for o in r["objects"] if o["gt_label"] == "HALLUCINATED")
    n_real = n_obj - n_hall
    n_probes = sum(len(r["probes"]) for r in used)
    n_contam = sum(1 for r in used for p in r["probes"] if p.get("gt_present"))

    L.append("")
    L.append("#" * 118)
    L.append("# STRATEGY 10 (v2) -- SANITY CHECK SUMMARY")
    L.append("#" * 118)
    L.append("")
    L.append("Setup")
    L.append("-----")
    L.append(f"  LVLM                 : {cfg.model_path}   (greedy, max_new_tokens={cfg.max_new_tokens}, seed={cfg.seed})")
    L.append(f"  zero-shot detector   : {cfg.detector_path}")
    L.append(f"  images               : {n_total} from {cfg.question_file}")
    L.append(f"  hyperparameters      : kappa={cfg.kappa}  K={cfg.K}  tau_box={cfg.tau_box}  "
             f"tau_low={cfg.tau_low}  rho={cfg.rho}")
    L.append(f"                         attn layers A={cfg.attn_layers}  max_patch_frac={cfg.max_patch_frac}")
    L.append("")

    L.append("Data")
    L.append("----")
    L.append(f"  images scored        : {len(used)}   (skipped: {len(skipped)})")
    for r in skipped:
        L.append(f"      - {r['image_file']}: {r['skipped']}")
    L.append(f"  objects mentioned    : {n_obj}")
    L.append(f"      actually REAL         : {n_real}  ({_pct(n_real, n_obj):.1f}%)")
    L.append(f"      actually HALLUCINATED : {n_hall}  ({_pct(n_hall, n_obj):.1f}%)   <- base rate")
    L.append(f"  probes scored        : {n_probes}")
    L.append(f"      GT-present (contaminated null) : {n_contam}  ({_pct(n_contam, n_probes):.1f}%)")
    L.append(f"  LVLM forward passes  : {2 * (n_obj + n_probes)}")
    L.append("")

    if n_obj == 0:
        L.append("  No objects to score -- nothing further to report.")
        return "\n".join(L)

    # ---- headline numbers at the configured kappa -------------------------
    c = confusion_at_kappa(records, cfg.kappa)
    L.append(f"HEADLINE  (kappa = {cfg.kappa})")
    L.append("-" * 40)
    L.append(f"  % of ACTUAL hallucinations CAUGHT      : {c['catch_rate']:.1f}%   "
             f"({c['TP']}/{c['TP'] + c['FN']})")
    L.append(f"  % of ACTUAL real objects FALSE-FLAGGED : {c['false_flag_rate']:.1f}%   "
             f"({c['FP']}/{c['FP'] + c['TN']})")
    L.append("")
    L.append(f"  precision of the HALLUCINATED flag     : {c['precision']:.1f}%   "
             f"({c['TP']}/{c['TP'] + c['FP']})")
    L.append(f"  overall flag rate                      : {c['flag_rate']:.1f}%")
    L.append(f"  balanced accuracy                      : {c['balanced_acc']:.1f}%")
    L.append("")
    L.append("            predicted HALLUC   predicted VERIFIED")
    L.append(f"  GT HALLUC       {c['TP']:>6}             {c['FN']:>6}")
    L.append(f"  GT REAL         {c['FP']:>6}             {c['TN']:>6}")
    L.append("")

    # ---- kappa sweep (free -- Deltas are cached) --------------------------
    L.append("KAPPA SWEEP   (zero extra compute: re-thresholds the cached Deltas)")
    L.append("-" * 100)
    L.append("  {:>6} {:>10} {:>12} {:>12} {:>11} {:>10} {:>10}".format(
        "kappa", "tau shift", "catch %", "false-flag %", "precision %", "flag %", "bal.acc %"))
    L.append("  " + "-" * 96)
    for k in cfg.kappa_sweep:
        cc = confusion_at_kappa(records, k)
        mark = "  <- default" if abs(k - cfg.kappa) < 1e-9 else ""
        L.append("  {:>6.2f} {:>10} {:>12.1f} {:>12.1f} {:>11.1f} {:>10.1f} {:>10.1f}{}".format(
            k, f"mu{k:+g}s", cc["catch_rate"], cc["false_flag_rate"],
            cc["precision"], cc["flag_rate"], cc["balanced_acc"], mark))
    L.append("")
    L.append("  (kappa UP  -> tau UP -> fewer objects clear the bar -> MORE flagged:")
    L.append("   catch rate rises, false-flag rate rises. kappa is a pure sensitivity knob.)")
    L.append("")

    # ---- diagnostics ------------------------------------------------------
    L.append("DIAGNOSTICS")
    L.append("-" * 60)

    obj_rows = [o for r in used for o in r["objects"]]
    probe_rows = [p for r in used for p in r["probes"]]

    def _mean(rows, key):
        vals = [x[key] for x in rows]
        return sum(vals) / len(vals) if vals else float("nan")

    def _frac_det(rows):
        return _pct(sum(1 for x in rows if x["region_source"] == "det"), len(rows))

    L.append("  Region attribution (Sec 3.3) -- candidates vs probes:")
    L.append("      {:<26} {:>12} {:>12}".format("", "candidates", "probes"))
    L.append("      {:<26} {:>12.1f} {:>12.1f}".format("routed to detector (%)",
                                                       _frac_det(obj_rows), _frac_det(probe_rows)))
    L.append("      {:<26} {:>12.1f} {:>12.1f}".format("mean masked area (%)",
                                                       100 * _mean(obj_rows, "masked_area_frac"),
                                                       100 * _mean(probe_rows, "masked_area_frac")))
    L.append("      {:<26} {:>12.3f} {:>12.3f}".format("mean s_det",
                                                       _mean(obj_rows, "s_det"),
                                                       _mean(probe_rows, "s_det")))
    L.append("      {:<26} {:>+12.3f} {:>+12.3f}".format("mean ell (unmasked)",
                                                         _mean(obj_rows, "ell"),
                                                         _mean(probe_rows, "ell")))
    L.append("      {:<26} {:>+12.3f} {:>+12.3f}".format("mean Delta",
                                                         _mean(obj_rows, "delta"),
                                                         _mean(probe_rows, "delta")))

    attn_obj = [x for x in obj_rows if x["region_source"] == "attn"]
    attn_pr = [x for x in probe_rows if x["region_source"] == "attn"]
    L.append("      {:<26} {:>12.1f} {:>12.1f}".format(
        "attn region hit cap (%)",
        _pct(sum(1 for x in attn_obj if x.get("region_capped")), len(attn_obj)),
        _pct(sum(1 for x in attn_pr if x.get("region_capped")), len(attn_pr))))
    L.append("")
    L.append("      ^ If candidates and probes differ systematically in masked area or in")
    L.append("        detector-routing rate, Delta is not measuring the same thing for the two")
    L.append("        groups, and the 'procedural symmetry' Sec 4.2 relies on is only nominal.")
    L.append("      ^ If 'attn region hit cap' is high, max_patch_frac is binding, every")
    L.append("        attention-routed word is getting an identically-sized region, and the")
    L.append("        attention signal has effectively been discarded -- lower rho.")
    L.append("")

    hall_rows = [o for o in obj_rows if o["gt_label"] == "HALLUCINATED"]
    real_rows = [o for o in obj_rows if o["gt_label"] == "REAL"]
    L.append("  Separation of the signal itself (does Delta even carry the information?):")
    L.append("      mean Delta | GT REAL          : {:+.4f}".format(_mean(real_rows, "delta")))
    L.append("      mean Delta | GT HALLUCINATED  : {:+.4f}".format(_mean(hall_rows, "delta")))
    L.append("      mean Delta | probes (null)    : {:+.4f}".format(_mean(probe_rows, "delta")))
    L.append("      AUROC (Delta as a REAL-vs-HALLUC score) : {:.3f}".format(
        _auroc([o["delta"] for o in real_rows], [o["delta"] for o in hall_rows])))
    L.append("      ^ AUROC is kappa-free: it is the ceiling ANY threshold rule on this Delta")
    L.append("        could reach. 0.50 == the score is uninformative.")
    L.append("")

    # ---- Sec 4.3.2: is the Gaussian reading of kappa even licensed? -------
    std_probe = []
    for r in used:
        s = r["sigma_hat"]
        if s and s > 0:
            std_probe.extend([(p["delta"] - r["mu_hat"]) / s for p in r["probes"]])
    norm = calibration.probe_normality(std_probe)
    L.append("  Sec 4.3.2 -- is the Gaussian quantile reading of kappa licensed?")
    L.append(f"      pooled standardised probe Deltas: n={norm['n']}, "
             f"skew={norm['skew']:+.3f}, excess kurtosis={norm['excess_kurtosis']:+.3f}")
    L.append("      (a normal reference gives skew ~ 0 and excess kurtosis ~ 0)")
    L.append("      For reference, at the default kappa the two OPTIONAL readings would say:")
    L.append(f"          Gaussian tail  1-Phi({cfg.kappa})       = {calibration.gaussian_tail(cfg.kappa):.3f}"
             "   [valid ONLY if the above looks normal]")
    L.append(f"          Cantelli bound 1/(1+kappa^2)         = {calibration.cantelli_bound(cfg.kappa):.3f}"
             "   [distribution-free, but very loose]")
    L.append(f"      Observed false-verification proxy (false-flag rate) = "
             f"{c['false_flag_rate']:.1f}% = {c['false_flag_rate'] / 100:.3f}")
    L.append("      Neither reading is invoked by the method; kappa is reported as an")
    L.append("      empirically-tuned sensitivity parameter, per Sec 8.")
    L.append("")
    L.append("#" * 118)
    return "\n".join(L)


def _auroc(pos: List[float], neg: List[float]) -> float:
    """AUROC of 'Delta is high' as a predictor of GT REAL. Rank-based (handles ties)."""
    if not pos or not neg:
        return float("nan")
    allv = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    allv.sort(key=lambda t: t[0])

    # average ranks for ties
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    rank_sum_pos = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)
