#!/usr/bin/env bash
# Strategy 10 (v2) -- sanity check on 50 COCO images.
#   GroundingDINO localises -> SAM segments -> patch-aligned mask, enforced pre-ViT.
#   Scores: deletion (delta, delta_lo) AND insertion (delta_ins, delta_lo_ins) AND
#   area-controlled (delta_ctrl, delta_lo_ctrl) -- all six, calibrated against the
#   same probes. AUROC is reported for each, plus the s_det baseline.
# Run from the MARINE repo root:  bash scripts/run_strategy10_v2_sanity.sh
set -e

export PYTHONPATH=$PYTHONPATH:/path/to/your/llava2

OUTPUT_DIR=./output/strategy10_v2
mkdir -p $OUTPUT_DIR

python ./scripts/eval_strategy10_v2_sanity.py \
    --model_path       llava-hf/llava-1.5-7b-hf \
    --detector_path    IDEA-Research/grounding-dino-base \
    --seg_backend      sam \
    --sam_path         facebook/sam-vit-base \
    --scores           delta,delta_lo,delta_ins,delta_lo_ins,delta_ctrl,delta_lo_ctrl \
    --primary_score    delta \
    --num_images       50 \
    --kappa            1.0 \
    --K                20 \
    --tau_low          0.15 \
    --seed             242 \
    --output_dir       $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/console.log

echo
echo "==> open $OUTPUT_DIR/report.html"
echo "==> the AUROC table is the thing to read first."
echo
echo "control_mask and insertion are ON by default now. Other variants worth trying:"
echo "  --no_insertion                 # isolate what insertion alone bought you"
echo "  --no_control_mask              # isolate what the control mask alone bought you"
echo "  --seg_backend box              # what SAM bought you (A/B)"
echo "  --primary_score delta_ins      # decide on sufficiency instead of deletion"
echo "  --sigma_shrink 0.5             # shrink tau's noise at K=20"
echo "  --no_language_prior            # drop the blank pass (disables CES fusions)"
echo "  --primary_score ces            # decide on the fused causal-existence score"
echo "  --probe_vocab ram              # richer null (probes need no ground truth)"
