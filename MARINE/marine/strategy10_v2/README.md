# Strategy 10 (v2) — sanity check

Training-free, single-pass object-hallucination **verification** for LVLMs, per
`strategy10_v2.pdf`. This directory implements Stages 1–2 (generate → causally
verify) and a diagnostic report. **Stage 3 (the LLM rewriter, Sec 5) is not run** —
this is a sanity check on the verification signal, not the full pipeline.

## What it does

For each of N COCO images:

1. **Generate** (Sec 2). Run LLaVA-1.5-7B *once, unguided*, on `"Generate a short
   caption of the image."` Extract every canonical MSCOCO object it mentions.
2. **Verify** (Sec 3–4). For each mentioned object *and* for K guaranteed-absent
   probes, re-elicit the word under a fixed prompt, find its purported visual
   region `R(w)`, black that region out, and measure how much the model's own
   log-likelihood for the word falls:  `Δ(w) = ℓ(w) − ℓ_masked(w)`.
   Calibrate per-image against the probes: `τ = μ̂_Δ + κ·σ̂_Δ`.
   Flag every object with `Δ < τ`.
3. **Report**. Compare each flag against COCO ground truth and print the catch
   rate and the false-flag rate.

## Run it

From the MARINE repo root:

```bash
bash scripts/run_strategy10_v2_sanity.sh
# or
python ./scripts/eval_strategy10_v2_sanity.py --num_images 50 --kappa 1.0
```

Outputs land in `./output/strategy10_v2/`: `sanity_report.txt`, `records.json`
(every Δ), `per_object.csv`.

Because every Δ is cached, **κ can be re-swept with no GPU**:

```bash
python ./scripts/eval_strategy10_v2_sanity.py \
    --from_cache ./output/strategy10_v2/records.json --kappa 1.5
```

The report also prints a full κ sweep by default, which *is* the empirical
selection procedure Sec 4.3.2 prescribes.

## Why this is fast (and the repo's own CHAIR eval is slow)

`eval/eval_chair.py`'s `CHAIR.__init__()` builds ground truth for the **entire**
COCO corpus — both train and val splits, ~900K segmentation annotations, ~617K
captions — and runs NLTK tokenisation + TextBlob singularisation on *every one*
of those ~617K captions. That per-caption NLP call, not JSON parsing, is what
actually costs the many minutes you'll see if you use it directly, and it reruns
in full on every invocation because it doesn't know in advance which images
you'll ask about.

We always know in advance: every image in `data/org_qa/chair/coco_chair.json` is
a `COCO_val2014_*` image, and we only ever score the N images actually being
tested. So `extraction.build_chair_for_images()`:

1. never opens `*_train2014.json` at all,
2. filters `val2014`'s annotations/captions down to the requested images
   *before* calling `caption_to_words()`, so NLP runs on `~5*N` captions
   instead of ~617,000.

It reuses the real `CHAIR.caption_to_words()` and the same
`inverse_synonym_dict` the full build uses, so the resulting ground-truth labels
are **identical** to what the slow, full-corpus build would produce for the same
images (verified by a parity test) — this is purely a speed optimisation. For
50 images this drops setup from several/many minutes to a few seconds.

Results are cached to `cfg.chair_cache`, keyed on the exact set of image IDs
requested — a cache built for 50 images is never silently reused for a
different 200; it just rebuilds (which, again, is now fast).

If you ever need ground truth for non-`val2014` images, `build_chair_full_corpus()`
is still available (unused by default) and behaves exactly like the repo's own
`eval/eval_chair.py` — slow, but general-purpose.

## Hyperparameters and why these values

| symbol | flag | default | rationale |
|---|---|---|---|
| `κ` | `--kappa` | `1.0` | Sec 4.3.2/Sec 8 are explicit that v2 has **no** finite-sample guarantee and that κ is a plain sensitivity knob. 1.0 is a neutral starting point; the sweep is what you should actually read. |
| `K` | `--K` | `20` | Probes per image. Enough for a stable σ̂ without doubling runtime (cost is `2(K+|O|)` forward passes/image). |
| `τ_box` | `--tau_box` | `0.10` | OWL-ViT's standard operating threshold on its sigmoid confidence. |
| `τ_low` | `--tau_low` | `0.05` | Probes must be *believed absent*: strictly below the detector's accept threshold. |
| `ρ` | `--rho` | `0.25` | Fraction of **image-normalised** attention mass in the fallback region (see note below). |
| `A` | `--attn_layers` | `14–19` | Mid-to-late LLaMA layers, where cross-modal grounding is strongest; layer 0–5 attention is near-uninformative. |
| — | `--max_patch_frac` | `0.50` | Safety valve so a diffuse attention row can't mask the whole image. The report tells you if it is binding. |

## Design decisions worth knowing

**OWL-ViT, not DETR.** The repo ships DETR + RAM++, but DETR is closed-vocabulary
over COCO-91 and cannot score an arbitrary probe word. Sec 4.2 requires probes and
candidates to pass through an *identical* measurement procedure, so the detector
must be open-vocabulary. The spec names OWL-ViT; we use
`google/owlvit-base-patch32`. One forward pass scores the entire 80-word vocabulary
against a shared image embedding, so `s_det` for every candidate *and* every probe
candidate costs one detector call per image.

**Extraction and ground truth are the repo's own.** `ExtractCanonicalObjects` and
the REAL/HALLUCINATED labels come from `eval/eval_chair.py`'s `CHAIR` class,
loaded unmodified by file path. So the labels here are *by construction* the same
ones the CHAIR benchmark scores against — no second, divergent definition of
"hallucinated".

**ρ is a threshold on renormalised attention.** A raw LLaMA attention row spends
most of its mass on BOS/system tokens (the attention-sink effect), so thresholding
raw mass would measure "how visual is this token", not "which patches support w".
We renormalise over the image patches first. This is a deliberate deviation from a
literal reading of Sec 3.3 and is the only one.

**Probes never look at ground truth.** Absence is established by the detector
(`s_det ≤ τ_low`), never by COCO annotations — using GT would make the null model
unobtainable at inference time. We do *measure* how often a probe turns out to be
GT-present and report it as "contaminated null", because probe contamination
inflates μ̂ and is the first thing to check if calibration misbehaves.

**Masking happens in original pixel space.** Sec 3.3 says the modified image is
"re-encoded in full", so we mask the PIL image and send it back through the
processor rather than editing the normalised `pixel_values` tensor. That requires
inverting CLIP's resize-shortest-edge→336 + centre-crop-336 to map the 24×24 patch
grid back to original pixels; `attribution.compute_geometry` replicates HF's
integer arithmetic exactly.

## What the report will tell you beyond the two headline numbers

- **AUROC of Δ** as a REAL-vs-HALLUCINATED score. This is κ-free: it is the ceiling
  *any* threshold rule on this Δ could reach. If AUROC ≈ 0.5, no choice of κ will
  help and the problem is upstream of calibration.
- **Candidate-vs-probe symmetry**: detector-routing rate and mean masked area for
  each group. If these differ systematically, Δ is not measuring the same thing for
  the two groups and the procedural symmetry of Sec 4.2 is only nominal — which
  would undermine the calibration regardless of κ.
- **Probe normality** (skew, excess kurtosis of pooled standardised probe Δ). Sec
  4.3.2 says the Gaussian reading of κ "should be checked ... before being relied
  upon"; this checks it. The Cantelli bound is printed alongside for contrast.

## Files

```
config.py         hyperparameters (+ symbol map to the spec)
prompts.py        vicuna_v1 prompt construction; the elicitation template c_elicit
model_loader.py   LLaVA-1.5-7B in fp16 with eager attention (needed for the fallback)
detector.py       OWL-ViT: s_det(w) + top box for the whole vocabulary in one pass
extraction.py     CHAIR reuse: ExtractCanonicalObjects, TARGETED GT lookup, vocabulary V
probes.py         Sec 4.1 probe sampling, co-occurrence-biased
attribution.py    Sec 3.3: two-tier region R(w); patch-grid → original-pixel inverse
masking.py        Sec 3.3: Mask(I, R(w)) with the per-channel mean pixel
scoring.py        Sec 3.2/3.4: teacher-forced ℓ, ℓ_masked, Δ
calibration.py    Sec 4.3: μ̂, σ̂, τ, the verified/flagged split, and the optional readings
pipeline.py       Algorithm 1, per image
report.py         tables, confusion metrics, κ sweep, diagnostics
```

Nothing outside `marine/strategy10_v2/` and `scripts/` is modified.
