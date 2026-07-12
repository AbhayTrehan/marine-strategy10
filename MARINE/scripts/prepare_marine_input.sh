#/bin/bash

METRIC="pope" # chair, pope, qa90
DATASET="coco" # coco, aokvqa, gqa, qa90

THRESHOLD_DETR=0.95
THRESHOLD_RAM=0.68

OUTPUT_DIR="./data/marine_qa/guidance"
SAVE_DIR="./data/marine_qa/"

########## Visual grounding ##########
###### DETR
echo "Running DETR on ${METRIC} with threshold ${THRESHOLD_DETR}"
python ./marine/grounding_models/detr_detect.py \
    --th ${THRESHOLD_DETR} \
    --metric ${METRIC} \
    --dataset ${DATASET} \
    --save_path ${OUTPUT_DIR}/${DATASET}_detr_th${THRESHOLD_DETR}.json



###### RAM++
echo "Running RAM++ on ${METRIC} with threshold ${THRESHOLD_RAM}"
python ./marine/grounding_models/ram_detect.py \
    --th ${THRESHOLD_RAM} \
    --metric ${METRIC} \
    --dataset ${DATASET} \
    --save_path ${OUTPUT_DIR}/${DATASET}_ram_th${THRESHOLD_RAM}.json



########## QA generation ##########
echo "Generating QA on ${METRIC} of Dataset ${DATASET} with guidance from DETR and RAM"
python ./eval/create_qa.py \
    --metric ${METRIC} \
    --dataset ${DATASET} \
    --th_detr ${THRESHOLD_DETR} \
    --th_ram ${THRESHOLD_RAM} \
    --guidance_dir ${OUTPUT_DIR} \
    --save_dir ${SAVE_DIR} \
    --save_name ${METRIC}_${DATASET}_detr_th${THRESHOLD_DETR}_ram_th${THRESHOLD_RAM}.json
