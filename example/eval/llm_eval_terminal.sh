#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"
ENV_PATH=$WORK_DIR/config/.env

# 检查并加载 .env 文件
if [ -f "$ENV_PATH" ]; then
  export $(grep -v '^#' "$ENV_PATH" | xargs)
  echo ".env file loaded from $ENV_PATH"
else
  echo "Error: .env file not found at $ENV_PATH"
  exit 1
fi

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

NUM_SERVICES=8

SERVER_MODEL=$EVALUATOR_NAME

API_BASES=()
for ((i=0; i<NUM_SERVICES; i++)); do
    API_BASES+=($API_BASES)
done
echo "SERVER_MODEL: $SERVER_MODEL"

EMBEDDING_NAME_OR_PATH=BAAI/bge-large-zh-v1.5
EMBEDDING_MODEL=$(basename "$EMBEDDING_NAME_OR_PATH")
echo "EMBEDDING_MODEL: $EMBEDDING_MODEL"

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
    
    python ${WORK_DIR}/src/metrics/llm_eval.py \
        --num_processes $NUM_SERVICES \
        --model_name $SERVER_MODEL \
        --api_key $API_KEY \
        --api_base ${API_BASES[@]} \
        --embedding_path $EMBEDDING_NAME_OR_PATH \
        --eval_folder_path $EVAL_FOLDER_PATH \
        --eval_name $FILE_NAME \
        --debug
done
