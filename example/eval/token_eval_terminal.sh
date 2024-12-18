#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

source /mnt/workspace/xulifeng_work/bin/activate ai_search
NUM_SERVICES=1

EVAL_FOLDER_PATH=${WORK_DIR}/data/metrics_rlts/model_api
EVAL_FILE_NAME=(
    "deepseek-chat.jsonl" \
)

for FILE_NAME in "${EVAL_FILE_NAME[@]}"
do
    echo "Processing evaluation file: $FILE_NAME"
    
    python ${WORK_DIR}/src/metrics/token_eval.py \
        --eval_folder_path $EVAL_FOLDER_PATH \
        --eval_name $FILE_NAME
done