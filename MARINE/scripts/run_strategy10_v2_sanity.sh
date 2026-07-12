#!/usr/bin/env bash
# Strategy 10 (v2) -- sanity check on 50 COCO images.
#   GroundingDINO localises every word; masking is patch-aligned and enforced.
# Run from the MARINE repo root:  bash scripts/run_strategy10_v2_sanity.sh
set -e

# Same PYTHONPATH line as scripts/eval_llava2.sh. Optional: strategy10_v2 falls
# back to a hard-coded vicuna_v1 template if the llava repo is not on the path.
export PYTHONPATH=$PYTHONPATH:/path/to/your/llava2

OUTPUT_DIR=./output/strategy10_v2
mkdir -p $OUTPUT_DIR

python ./scripts/eval_strategy10_v2_sanity.py \
    --model_path       llava-hf/llava-1.5-7b-hf \
    --detector_path    IDEA-Research/grounding-dino-base \
    --image_folder     ./data/coco/val2014 \
    --coco_annotations ./data/coco/annotations \
    --question_file    ./data/org_qa/chair/coco_chair.json \
    --num_images       50 \
    --kappa            1.0 \
    --K                20 \
    --tau_low          0.15 \
    --det_batch_size   8 \
    --seed             242 \
    --output_dir       $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/console.log

echo
echo "==> open $OUTPUT_DIR/report.html"
