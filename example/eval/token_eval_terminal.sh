#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

NUM_SERVICES=1

EVAL_FOLDER_PATH=${WORK_DIR}/data/metrics_rlts/model_api
# 手动选择
# EVAL_FILE_NAME=(
#     "moonshot-v1-128k.jsonl" \
# )
# 自动选择
EVAL_FILE_NAME=($(find "$EVAL_FOLDER_PATH" -maxdepth 1 -type f -name "*.jsonl" -exec basename {} \;))

for FILE_NAME in "${EVAL_FILE_NAME[@]}"
do
    echo "Processing evaluation file: $FILE_NAME"
    
    python ${WORK_DIR}/src/metrics/token_eval.py \
        --eval_folder_path $EVAL_FOLDER_PATH \
        --eval_name $FILE_NAME
done
