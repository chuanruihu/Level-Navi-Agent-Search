#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/../..")"

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

source /mnt/workspace/xulifeng_work/bin/activate langgraph
NUM_SERVICES=4
# 定义模型相关参数
SERVER_MODEL=moonshot-v1-128k
API_KEY=fk251639865.xTU0vtdqU1QB7d9zni9bV6KDk8P_hzI-82b761ed
API_BASES=()
for ((i=0; i<NUM_SERVICES; i++)); do
    API_BASES+=("http://api.360.cn/v1")
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
