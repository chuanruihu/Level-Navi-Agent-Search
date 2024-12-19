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

# 使用 API_KEY 变量
echo "API_KEY is $API_KEY"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

NUM_SERVICES=4
# 定义模型相关参数
SERVER_MODEL=$MODEL_NAME
API_BASES=()
for ((i=0; i<NUM_SERVICES; i++)); do
    API_BASES+=($API_BASE)
done

echo "SERVER_MODEL: $SERVER_MODEL"
INPUT_PATH=${WORK_DIR}/data/ai_search_benchmark.json
SAVE_PATH=${WORK_DIR}/data/metrics_rlts/$SERVER_MODEL.jsonl

python ${WORK_DIR}/src/ai_search/search.py \
    --num_processes $NUM_SERVICES\
    --model_name $SERVER_MODEL \
    --api_key $API_KEY \
    --api_base ${API_BASES[@]} \
    --input_path $INPUT_PATH \
    --save_path $SAVE_PATH \
    --debug \
