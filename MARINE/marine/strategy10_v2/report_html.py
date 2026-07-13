"""
Self-contained HTML report.

Design intent: every number the method acts on should be visible next to the
picture it came from, and every mask should be shown as the MODEL saw it -- not
as a box drawn on top of an unmasked image. If the masking is wrong, you should
be able to SEE that it is wrong, not have to trust a docstring.

Per image:
  * the original
  * the same image with GroundingDINO's box for every mentioned object
  * for every mentioned object: the actual masked image fed to the LVLM, next to
    ell(w), ell_masked(w), Delta, the confidence drop, the decision, and the COCO
    ground-truth label
  * the probe table (the null distribution), collapsed by default
"""

import base64
import html as _html
import io
from typing import Dict, List, Optional

from PIL import Image, ImageDraw

from . import calibration, report

# colour-blind-safe-ish palette
C_REAL = "#1a7f37"      # green
C_HALLUC = "#cf222e"    # red
C_TP = "#1a7f37"
C_FP = "#bc4c00"
C_FN = "#9a6700"
C_TN = "#57606a"


def _b64(img: Image.Image, max_w: int, quality: int = 82) -> str:
    im = img.convert("RGB")
    if im.width > max_w:
        h = max(1, int(im.height * max_w / im.width))
        im = im.resize((max_w, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _colour_for(row: Dict) -> str:
    return C_HALLUC if row.get("gt_label") == "HALLUCINATED" else C_REAL


def draw_boxes(image: Image.Image, rows: List[Dict]) -> Image.Image:
    """GroundingDINO's raw box (solid) and the patch-aligned masked region
    actually used (dashed) for every mentioned object."""
    img = image.convert("RGB").copy()
    d = ImageDraw.Draw(img)
    w = max(2, int(min(img.size) / 300))

    for row in rows:
        col = _colour_for(row)
        boxes = row.get("boxes") or ([row["box"]] if row.get("box") else [])
        scores = row.get("inst_scores") or []
        # EVERY instance the detector found -- all of them get masked, so all of
        # them get drawn. If only one box appears for an object you can see twice
        # in the picture, that is the bug you are looking for.
        for j, box in enumerate(boxes):
            d.rectangle([box[0], box[1], box[2], box[3]], outline=col, width=w)
            s = scores[j] if j < len(scores) else row["s_det"]
            label = f"{row['word']} {s:.2f}"
            tx, ty = box[0] + 2, max(0, box[1] - 14)
            d.rectangle([tx - 2, ty - 1, tx + 7 * len(label), ty + 13], fill=col)
            d.text((tx, ty), label, fill="white")
        mb = row.get("mask_bbox")
        if mb:
            # bounding rect of the region actually masked (whole ViT patches)
            d.rectangle([mb[0], mb[1], mb[2], mb[3]], outline="#ffffff", width=1)
    return img


def _fmt(v, spec="{:+.4f}"):
    if v is None:
        return "&mdash;"
    try:
        return spec.format(v)
    except (TypeError, ValueError):
        return _html.escape(str(v))


def _outcome_badge(row: Dict) -> str:
    if row.get("gt_label") not in ("REAL", "HALLUCINATED"):
        return '<span class="badge" style="background:#8250df">n/a</span>'
    o = report._outcome(row.get("decision", ""), row.get("gt_label", ""))
    tag = o.split()[0]
    col = {"TP": C_TP, "FP": C_FP, "FN": C_FN, "TN": C_TN}.get(tag, C_TN)
    return f'<span class="badge" style="background:{col}">{tag}</span>'


def _flags(row: Dict) -> str:
    out = []
    if row.get("outside_crop"):
        out.append('<span class="flag warn">box outside centre-crop &rarr; nothing masked</span>')
    if row.get("clipped_by_crop"):
        out.append('<span class="flag">box clipped by centre-crop</span>')
    if not row.get("mask_enforced_ok", True):
        out.append('<span class="flag bad">MASK NOT ENFORCED</span>')
    if row.get("n_patches", 0) >= 0.9 * 576:
        out.append('<span class="flag warn">masks ~whole image</span>')
    return " ".join(out)


# --------------------------------------------------------------------------- #

def _object_rows_html(rec: Dict, cfg) -> str:
    parts = []
    for row in rec["objects"]:
        thumb = ""
        img = row.get("_masked_img")
        if img is not None:
            thumb = f'<img class="thumb" src="{_b64(img, cfg.html_thumb_width)}">'

        dec = row.get("decision", "")
        dec_col = C_HALLUC if dec == "HALLUCINATED" else C_REAL
        gt = row.get("gt_label", "")
        gt_col = {"HALLUCINATED": C_HALLUC, "REAL": C_REAL}.get(gt, "#8250df")

        sc = row.get("scores", {})
        extra = "".join(
            f'<div class="sub"><code>{k}</code> {v:+.4f}</div>'
            for k, v in sc.items() if k != "delta")

        lo = ""
        if "lo" in row:
            lo = (f'<div class="sub">LO {row["lo"]:+.2f} &rarr; {row["lo_masked"]:+.2f}'
                  f' (p(yes) {row["p_yes"]:.2f} &rarr; {row["p_yes_masked"]:.2f})</div>')

        coll = ""
        if row.get("region_collisions"):
            c = ", ".join(f"{d['word']} (IoU {d['iou']})" for d in row["region_collisions"])
            coll = f'<span class="flag warn">region collides with {_html.escape(c)}</span>'

        parts.append(f"""
        <tr>
          <td>{thumb}</td>
          <td>
            <div class="obj">{_html.escape(row['word'])}</div>
            <div class="sub">said as &ldquo;{_html.escape(str(row.get('surface','')))}&rdquo;
                {'' if row.get('surface','').lower() == row['word'] else '&rarr; ' + _html.escape(row['word'])}</div>
            <div class="sub">s_det {row['s_det']:.3f} &middot;
                {row.get('n_instances', 1)} inst &middot; {row.get('seg','box')} &middot;
                {row['n_patches']}/576 patches ({100*row['patch_frac']:.1f}%)</div>
            <div>{_flags(row)} {coll}</div>
          </td>
          <td class="num">{_fmt(row['ell'], '{:.4f}')}<div class="sub">p={row['p']:.4f}</div>{lo}</td>
          <td class="num">{_fmt(row['ell_masked'], '{:.4f}')}<div class="sub">p={row['p_masked']:.4f}</div></td>
          <td class="num strong">{_fmt(sc.get('delta'))}{extra}</td>
          <td class="num">{row['conf_drop_pct']:+.1f}%</td>
          <td><span class="pill" style="background:{dec_col}">{dec}</span></td>
          <td><span class="pill" style="background:{gt_col}">{gt}</span></td>
          <td>{_outcome_badge(row)}</td>
        </tr>""")
    return "".join(parts)


def _probe_rows_html(rec: Dict) -> str:
    parts = []
    for row in sorted(rec["probes"], key=lambda r: r["delta"]):
        present = ' <span class="flag warn">GT-PRESENT</span>' if row.get("gt_present") else ""
        parts.append(f"""
        <tr>
          <td>{_html.escape(row['word'])}{present}</td>
          <td class="num">{row['s_det']:.3f}</td>
          <td class="num">{row.get('n_instances', 1)}</td>
          <td class="num">{row['n_patches']}</td>
          <td class="num">{_fmt(row['ell'], '{:.4f}')}</td>
          <td class="num">{_fmt(row['ell_masked'], '{:.4f}')}</td>
          <td class="num strong">{_fmt(row['delta'])}</td>
        </tr>""")
    return "".join(parts)


def _image_card(rec: Dict, cfg, idx: int, total: int) -> str:
    head = (f'<div class="cardhead"><span class="idx">{idx}/{total}</span> '
            f'{_html.escape(rec["image_file"])} '
            f'<span class="sub">image_id {rec["image_id"]}</span></div>')

    if rec.get("skipped"):
        return f'<div class="card">{head}<div class="skip">SKIPPED &mdash; ' \
               f'{_html.escape(str(rec["skipped"]))}</div></div>'

    image = rec.get("_image")
    orig = f'<img src="{_b64(image, cfg.html_max_width)}">' if image is not None else ""
    boxed = ""
    if image is not None:
        boxed = f'<img src="{_b64(draw_boxes(image, rec["objects"]), cfg.html_max_width)}">'

    n_contam = sum(1 for p in rec["probes"] if p.get("gt_present"))
    leak_max = max([r["leak_max"] for r in rec["objects"]] or [0.0])

    gt_txt = ", ".join(rec["gt_objects"]) if rec["gt_objects"] else "(none)"

    return f"""
    <div class="card">
      {head}
      <div class="caption"><b>LVLM caption:</b> &ldquo;{_html.escape(rec['caption'])}&rdquo;</div>
      <div class="sub"><b>COCO ground truth:</b> {_html.escape(gt_txt)}</div>

      <div class="imgrow">
        <figure>{orig}<figcaption>original</figcaption></figure>
        <figure>{boxed}<figcaption>GroundingDINO box per mentioned object
            (white = patch-aligned region actually masked)</figcaption></figure>
      </div>

      <div class="null">
        null model from {len(rec['probes'])} probes &mdash;
        <b>&mu;&#770;<sub>&Delta;</sub></b> = {rec['mu_hat']:+.4f} &nbsp;
        <b>&sigma;&#770;<sub>&Delta;</sub></b> = {rec['sigma_hat']:.4f} &nbsp;
        &rarr; <b>&tau;</b> = &mu;&#770; + {rec['kappa']}&middot;&sigma;&#770; = {rec['tau']:+.4f}
        &nbsp;&middot;&nbsp; GT-present probes: {n_contam}
        &nbsp;&middot;&nbsp; residual mask leak (max): {leak_max:.2e}
      </div>

      <table class="objs">
        <thead><tr>
          <th>masked image fed to the LVLM</th>
          <th>object</th>
          <th>&#8467;(w)</th><th>&#8467;<sub>masked</sub>(w)</th>
          <th>&Delta;(w)</th><th>conf&nbsp;drop</th>
          <th>decision</th><th>ground&nbsp;truth</th><th></th>
        </tr></thead>
        <tbody>{_object_rows_html(rec, cfg)}</tbody>
      </table>

      <details>
        <summary>probe set (the null distribution) &mdash; {len(rec['probes'])} words</summary>
        <table class="probes">
          <thead><tr><th>probe</th><th>s_det</th><th>inst</th><th>patches</th>
            <th>&#8467;(p)</th><th>&#8467;<sub>masked</sub>(p)</th><th>&Delta;(p)</th></tr></thead>
          <tbody>{_probe_rows_html(rec)}</tbody>
        </table>
      </details>
    </div>"""


# --------------------------------------------------------------------------- #

def _auroc_rows(records) -> str:
    rows = report.auroc_table(records)
    if not rows:
        return '<tr><td colspan="4">(no scored objects)</td></tr>'
    best = max(rows, key=lambda r: r["auroc"])
    out = []
    for r in rows:
        cls = " class=cur" if r is best else ""
        out.append(
            f"<tr{cls}><td><code>{_html.escape(r['score'])}</code>"
            f"<div class='sub'>{_html.escape(r['label'])}</div></td>"
            f"<td class='num strong'>{r['auroc']:.3f}</td>"
            f"<td class='num'>{r['mean_real']:+.3f}</td>"
            f"<td class='num'>{r['mean_hall']:+.3f}</td></tr>")
    return "".join(out)


def _summary_html(records: List[Dict], cfg) -> str:
    used = [r for r in records if not r.get("skipped")]
    obj_rows = [o for r in used for o in r["objects"]]
    probe_rows = [p for r in used for p in r["probes"]]

    n_obj = len(obj_rows)
    n_hall = sum(1 for o in obj_rows if o["gt_label"] == "HALLUCINATED")
    n_real = n_obj - n_hall

    c = report.confusion_at_kappa(records, cfg.kappa)

    def _mean(rows, key):
        vals = [x[key] for x in rows if x.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    hall = [o for o in obj_rows if o["gt_label"] == "HALLUCINATED"]
    real = [o for o in obj_rows if o["gt_label"] == "REAL"]
    auroc = report._auroc([o["delta"] for o in real], [o["delta"] for o in hall])

    sweep = "".join(
        f"<tr{' class=cur' if abs(k - cfg.kappa) < 1e-9 else ''}>"
        f"<td>{k:+.2f}</td><td>{cc['catch_rate']:.1f}%</td>"
        f"<td>{cc['false_flag_rate']:.1f}%</td><td>{cc['precision']:.1f}%</td>"
        f"<td>{cc['flag_rate']:.1f}%</td><td>{cc['balanced_acc']:.1f}%</td></tr>"
        for k, cc in ((k, report.confusion_at_kappa(records, k)) for k in cfg.kappa_sweep)
    )

    n_contam = sum(1 for p in probe_rows if p.get("gt_present"))
    leak_max = max([o["leak_max"] for o in obj_rows] or [0.0])
    not_enforced = sum(1 for o in obj_rows if not o.get("mask_enforced_ok", True))
    outside = sum(1 for o in obj_rows if o.get("outside_crop"))

    return f"""
    <div class="summary">
      <h1>Strategy 10 (v2) &mdash; causal occlusion verification</h1>
      <div class="sub">
        {cfg.model_path} &nbsp;&middot;&nbsp; localiser: {cfg.detector_path}
        &nbsp;&middot;&nbsp; {len(used)} images scored ({len(records) - len(used)} skipped)
        &nbsp;&middot;&nbsp; &kappa;={cfg.kappa}, K={cfg.K}, &tau;<sub>low</sub>={cfg.tau_low}
      </div>

      <div class="headline">
        <div class="metric">
          <div class="mv">{c['catch_rate']:.1f}%</div>
          <div class="ml">of ACTUAL hallucinations caught</div>
          <div class="sub">{c['TP']} / {c['TP'] + c['FN']}</div>
        </div>
        <div class="metric">
          <div class="mv">{c['false_flag_rate']:.1f}%</div>
          <div class="ml">of ACTUAL real objects false-flagged</div>
          <div class="sub">{c['FP']} / {c['FP'] + c['TN']}</div>
        </div>
        <div class="metric">
          <div class="mv">{auroc:.3f}</div>
          <div class="ml">AUROC of &Delta;  (&kappa;-free ceiling)</div>
          <div class="sub">0.50 = &Delta; carries no signal at all</div>
        </div>
      </div>

      <h2>AUROC &mdash; can each score tell REAL from HALLUCINATED at all?</h2>
      <table class="sweep">
        <thead><tr><th>score</th><th>AUROC</th><th>mean | REAL</th><th>mean | HALLUC</th></tr></thead>
        <tbody>{_auroc_rows(records)}</tbody>
      </table>
      <div class="note">
        AUROC = P(a random REAL object scores higher than a random HALLUCINATED one).
        Threshold-free and &kappa;-free: it is the <b>ceiling</b> any threshold rule on
        that score could reach. 0.50 = the score carries no information.
        <b>s_det is the baseline to beat</b> &mdash; it is GroundingDINO's raw confidence,
        with no masking, no occlusion and no LVLM at all. If the occlusion scores do not
        clearly beat it, the occlusion machinery is not earning its keep, and that matters
        more than any catch-rate below.
      </div>

      <div class="grid2">
        <div>
          <h2>Where the objects came from</h2>
          <table class="kv">
            <tr><td>objects mentioned by the LVLM</td><td>{n_obj}</td></tr>
            <tr><td>&nbsp;&nbsp;actually REAL</td><td>{n_real} ({100*n_real/max(n_obj,1):.1f}%)</td></tr>
            <tr><td>&nbsp;&nbsp;actually HALLUCINATED</td>
                <td>{n_hall} ({100*n_hall/max(n_obj,1):.1f}%) &larr; base rate</td></tr>
            <tr><td>probes scored</td><td>{len(probe_rows)}</td></tr>
            <tr><td>&nbsp;&nbsp;GT-present (contaminated null)</td>
                <td>{n_contam} ({100*n_contam/max(len(probe_rows),1):.1f}%)</td></tr>
          </table>

          <h2>Is &Delta; separating anything?</h2>
          <table class="kv">
            <tr><td>mean &Delta; | GT REAL</td><td>{_mean(real,'delta'):+.4f}</td></tr>
            <tr><td>mean &Delta; | GT HALLUCINATED</td><td>{_mean(hall,'delta'):+.4f}</td></tr>
            <tr><td>mean &Delta; | probes (the null)</td><td>{_mean(probe_rows,'delta'):+.4f}</td></tr>
          </table>
        </div>

        <div>
          <h2>&kappa; sweep <span class="sub">(free &mdash; re-thresholds cached &Delta;s)</span></h2>
          <table class="sweep">
            <thead><tr><th>&kappa;</th><th>catch</th><th>false-flag</th>
              <th>precision</th><th>flag rate</th><th>bal. acc</th></tr></thead>
            <tbody>{sweep}</tbody>
          </table>

          <h2>Masking integrity</h2>
          <table class="kv">
            <tr><td>residual leak into masked patches (max |dev|)</td>
                <td>{leak_max:.2e} <span class="sub">before enforcement</span></td></tr>
            <tr><td>objects where enforcement failed</td>
                <td>{'<b style="color:%s">%d</b>' % (C_HALLUC, not_enforced) if not_enforced else '0'}</td></tr>
            <tr><td>boxes falling outside the centre-crop</td><td>{outside}</td></tr>
            <tr><td>mean instances masked | candidates</td><td>{_mean(obj_rows,'n_instances'):.2f}</td></tr>
            <tr><td>mean instances masked | probes</td><td>{_mean(probe_rows,'n_instances'):.2f}</td></tr>
            <tr><td>mean masked patches | candidates</td><td>{_mean(obj_rows,'n_patches'):.0f} / 576</td></tr>
            <tr><td>mean masked patches | probes</td><td>{_mean(probe_rows,'n_patches'):.0f} / 576</td></tr>
            <tr><td>mean s_det | candidates</td><td>{_mean(obj_rows,'s_det'):.3f}</td></tr>
            <tr><td>mean s_det | probes</td><td>{_mean(probe_rows,'s_det'):.3f}</td></tr>
          </table>
          <div class="note">
            If candidates and probes differ systematically in masked-patch count,
            &Delta; is not measuring the same thing for the two groups and the null
            calibration is only nominally symmetric &mdash; read this before
            reading the headline numbers.
          </div>
        </div>
      </div>
    </div>"""


CSS = """
:root{--bd:#d0d7de;--bg:#f6f8fa;--fg:#1f2328;--mut:#57606a}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
     color:var(--fg);margin:0;padding:24px;background:#fff}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:14px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);
   margin:20px 0 8px;border-bottom:1px solid var(--bd);padding-bottom:4px}
.sub{color:var(--mut);font-size:12px}
.summary{border:1px solid var(--bd);border-radius:8px;padding:20px;margin-bottom:28px;background:var(--bg)}
.headline{display:flex;gap:16px;margin:18px 0}
.metric{flex:1;background:#fff;border:1px solid var(--bd);border-radius:8px;padding:14px;text-align:center}
.mv{font-size:30px;font-weight:650;letter-spacing:-.02em}
.ml{font-size:12px;color:var(--mut);margin-top:2px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:24px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid var(--bd);padding:6px 8px;text-align:left;vertical-align:top}
th{background:var(--bg);font-weight:600;font-size:12px}
.kv td:first-child{color:var(--mut)}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.strong{font-weight:650}
.sweep tr.cur{background:#fff8c5;font-weight:600}
.note{font-size:12px;color:var(--mut);margin-top:8px;padding:8px;border-left:3px solid var(--bd)}
.card{border:1px solid var(--bd);border-radius:8px;padding:18px;margin-bottom:22px}
.cardhead{font-weight:600;margin-bottom:6px}
.idx{display:inline-block;background:var(--fg);color:#fff;border-radius:4px;
     padding:1px 6px;font-size:12px;margin-right:6px}
.caption{margin:6px 0}
.imgrow{display:flex;gap:14px;margin:12px 0;flex-wrap:wrap}
.imgrow figure{margin:0}
.imgrow img{border:1px solid var(--bd);border-radius:6px;display:block}
figcaption{font-size:11px;color:var(--mut);margin-top:4px;max-width:460px}
.null{background:var(--bg);border:1px solid var(--bd);border-radius:6px;
      padding:8px 10px;margin:10px 0;font-size:13px}
.thumb{border:1px solid var(--bd);border-radius:4px;display:block}
.obj{font-weight:600}
.pill,.badge{color:#fff;border-radius:10px;padding:1px 8px;font-size:11px;
             font-weight:600;white-space:nowrap;display:inline-block}
.badge{border-radius:4px}
.flag{display:inline-block;font-size:10px;padding:1px 5px;border-radius:3px;
      background:#eaeef2;color:var(--mut);margin-top:2px}
.flag.warn{background:#fff1e5;color:#bc4c00}
.flag.bad{background:#ffebe9;color:#cf222e}
.skip{color:var(--mut);font-style:italic}
details{margin-top:10px}
summary{cursor:pointer;font-size:12px;color:var(--mut)}
"""


def build_html(records: List[Dict], cfg) -> str:
    cards = "".join(
        _image_card(rec, cfg, i, len(records)) for i, rec in enumerate(records, start=1)
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Strategy 10 (v2) sanity check</title>
<style>{CSS}</style></head>
<body>
{_summary_html(records, cfg)}
{cards}
</body></html>"""
