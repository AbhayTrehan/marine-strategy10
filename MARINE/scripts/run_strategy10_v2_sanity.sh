#!/usr/bin/env bash
# Strategy 10 (v2) -- sanity check on 50 COCO images.
# Run from the MARINE repo root:   bash scripts/run_strategy10_v2_sanity.sh
set -e

# Same PYTHONPATH line as scripts/eval_llava2.sh. Optional here: strategy10_v2
# falls back to a hard-coded vicuna_v1 template if the llava repo is absent.
export PYTHONPATH=$PYTHONPATH:/path/to/your/llava2

MODEL_VERSION="llava-hf/llava-1.5-7b-hf"
NUM_IMAGES=50
KAPPA=1.0
K=20
SEED=242

OUTPUT_DIR=./output/strategy10_v2
mkdir -p $OUTPUT_DIR

python ./scripts/eval_strategy10_v2_sanity.py \
    --model_path       $MODEL_VERSION \
    --detector_path    google/owlvit-base-patch32 \
    --image_folder     ./data/coco/val2014 \
    --coco_annotations ./data/coco/annotations \
    --question_file    ./data/org_qa/chair/coco_chair.json \
    --num_images       $NUM_IMAGES \
    --kappa            $KAPPA \
    --K                $K \
    --tau_box          0.10 \
    --tau_low          0.05 \
    --rho              0.25 \
    --seed             $SEED \
    --output_dir       $OUTPUT_DIR \
    2>&1 | tee $OUTPUT_DIR/console.log
