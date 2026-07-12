# Strategy 10 (v2) — sanity check

Training-free object-hallucination **verification** for LVLMs, per `strategy10_v2.pdf`.
Implements Stages 1–2 (generate → causally verify). **Stage 3 (the LLM rewriter,
Sec 5) is not run** — this is a sanity check on the verification signal.

## Run

```bash
bash scripts/run_strategy10_v2_sanity.sh
# or
python ./scripts/eval_strategy10_v2_sanity.py --num_images 50 --kappa 1.0
```

Then open **`output/strategy10_v2/report.html`**. It shows, for every image: the
original, the GroundingDINO box for every mentioned object, and — for each object —
**the actual masked image that was fed to the LVLM**, next to ℓ(w), ℓ_masked(w), Δ,
the confidence drop, the decision, and the COCO ground-truth label.

κ can be re-swept with no GPU (every Δ is cached):

```bash
python ./scripts/eval_strategy10_v2_sanity.py --from_cache output/strategy10_v2/records.json --kappa 1.5
```

## The two things that changed, and why

### 1. GroundingDINO localises *every* word. There is no attention fallback.

The spec's two-tier scheme (detector if `s_det ≥ τ_box`, else LVLM cross-attention)
routed candidates and probes down **different code paths** depending on where their
detector score happened to land. That quietly destroys the "procedural symmetry
between candidates and probes" that Sec 4.2 says the entire null calibration rests
on: a probe scored through the attention branch and a candidate scored through the
detector branch are not measuring the same thing, so their Δs are not comparable, so
μ̂/σ̂ are meaningless.

GroundingDINO removes the need for the fallback. It is *phrase*-grounded and emits a
best-guess box for **any** phrase — including one that isn't in the image. That
best-guess box is exactly what Sec 3.3 asks for ("a region *purporting* to support w
visually"), and for an absent probe it is precisely the honest null: *where would
this object be, if it were here?* One detector, one procedure, both groups.

`τ_box`, `ρ`, `attn_layers`, `max_patch_frac` are **deleted** — the branch they
controlled no longer exists.

> **One phrase per forward pass.** GroundingDINO's text encoder is a BERT that
> attends across the whole prompt. Packing several phrases into one prompt
> (`"dog. cat. sofa."`) lets them condition each other, contaminating each phrase's
> grounding with the others — which would leak the candidate set into the probe
> scores. Every word gets its own single-phrase prompt. Throughput is recovered by
> batching along the *batch* dimension (same image, N single-phrase prompts), never
> by packing phrases.

### 2. Masking is patch-aligned and enforced on `pixel_values`. This was leaking.

LLaVA's ViT does not see pixels — it sees **576 patch tokens**, one per 14×14 cell of
the preprocessed 336×336 crop. The old code painted the box's *exact pixels* grey,
which left two channels open:

- **partial patches** — a patch the box only half-covers is still one token, and half
  of it is still the object;
- **resize bleed** — the mask is painted at original resolution, then bicubically
  resized to 336; interpolation drags unmasked neighbours *into* the masked area.

Measured, on a planted high-contrast object (max |deviation from fill| inside the
patches we claim to have masked, in normalised `pixel_values` units):

| what the ViT actually received | residual object signal |
|---|---|
| no mask (control) | 1.8978 |
| **exact-box PIL mask (the old behaviour)** | **0.3752** |
| patch-aligned PIL mask | 0.0300 |
| **+ enforced on `pixel_values` (now)** | **0.0000** |

So roughly **20 % of the object's signal was still reaching the encoder** through a
mask that looked fine. That inflates ℓ_masked, which compresses Δ toward zero, which
flattens the separation between real and hallucinated objects.

Now: the region is defined **on the patch grid** (every patch the box touches, in
full), and after the processor has resized/cropped/normalised, the masked patches are
overwritten **again**, directly on the `pixel_values` tensor, with the normalised fill
colour — before the vision tower runs. The object provably cannot reach the encoder.

The leak is **measured, not asserted**: `leak_max` (reported per object, in the HTML
and the CSV) is how much signal the PIL mask alone left behind. If the patch geometry
were wrong, that number would be large — so it's a live check on the mapping, not a
comment claiming the mapping is right.

## What is unchanged

**The confidence measurement is exactly the spec's, untouched:**

```
ℓ(w)        = (1/L) Σ_k log p_θ(w_k | c_elicit, w_<k, I)          Eq. (4)
ℓ_masked(w) = (1/L) Σ_k log p_θ(w_k | c_elicit, w_<k, I_masked)   Eq. (6)
Δ(w)        = ℓ(w) − ℓ_masked(w)                                  Eq. (7)
μ̂, σ̂       from {Δ(p)}, Bessel-corrected                         Eq. (8)
τ           = μ̂ + κ·σ̂                                             Eq. (9)
O_pos       = {Δ ≥ τ} ;  H = O \ O_pos                            Eq. (11)/(12)
```

Every verify/flag decision is made on Δ itself. `conf_drop%` = `100·(1 − e^{−Δ})` and
`p = e^{ℓ}` appear in the report for readability only and feed no decision.

Also unchanged: the elicitation template `c_elicit` (Sec 3.1), probe sampling
(Sec 4.1, co-occurrence-biased, GT-blind), and object extraction + ground truth, which
come straight from the repo's own `eval/eval_chair.py` — so REAL/HALLUCINATED here is
*by construction* the same label CHAIR scores against.

## Hyperparameters

| symbol | flag | default | note |
|---|---|---|---|
| `κ` | `--kappa` | `1.0` | Sec 4.3.2/Sec 8 are explicit that v2 has **no** finite-sample guarantee; κ is a plain sensitivity knob. Read the sweep, not the default. |
| `K` | `--K` | `20` | probes per image |
| `τ_low` | `--tau_low` | `0.15` | probes must be *believed absent*. **Not** the old 0.05 — GroundingDINO's scores live on a different scale to OWL-ViT's. The report prints the s_det distribution so this can be re-tuned against real data. |
| — | `--det_batch_size` | `8` | same image, N single-phrase prompts |
| — | `--no_enforce_pixel_mask` | off | **diagnostic**: disable the `pixel_values` enforcement to see what the interpolation leak was worth |

## Read the report in this order

1. **AUROC of Δ** — κ-free. It is the ceiling *any* threshold rule on this Δ could
   reach. If it's ≈ 0.50, no κ helps and the problem is upstream of calibration.
2. **Masking integrity** — residual leak, enforcement failures, boxes falling outside
   the centre-crop (those get *nothing* masked, so Δ = 0 by construction).
3. **Candidate-vs-probe symmetry** — mean masked-patch count per group. If these
   diverge, Δ is not measuring the same thing for the two groups and the null is only
   nominally symmetric, whatever the headline numbers say.
4. **Probe contamination** — GT-present probes inflate μ̂.
5. Only then, the catch / false-flag rates.

## Files

```
config.py       hyperparameters (+ what changed and why)
prompts.py      vicuna_v1 prompt construction; the elicitation template c_elicit
model_loader.py LLaVA-1.5-7B (fp16, slow tokenizer, correct <image> expansion)
detector.py     GroundingDINO: one phrase per forward, a box for EVERY word
extraction.py   CHAIR reuse: ExtractCanonicalObjects, targeted COCO GT, vocabulary V
probes.py       Sec 4.1 probe sampling, co-occurrence-biased, GT-blind
attribution.py  box -> ViT patch grid -> back (exact inverse of resize+centre-crop)
masking.py      patch-aligned mask, pixel_values enforcement, LEAK MEASUREMENT
scoring.py      Eq. (4)/(6)/(7), verbatim
calibration.py  Eq. (8)/(9)/(11)/(12), verbatim
pipeline.py     Algorithm 1, per image
report.py       console metrics, kappa sweep, diagnostics
report_html.py  the readable report
```

Nothing outside `marine/strategy10_v2/` and `scripts/` is modified.
