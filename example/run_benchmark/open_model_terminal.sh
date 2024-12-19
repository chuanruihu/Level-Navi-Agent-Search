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

# 定义模型相关参数
MODEL_NAME_OR_PATH=Qwen/Qwen2.5-7B-Instruct
SERVER_MODEL=$(basename "$MODEL_NAME_OR_PATH")

echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "SERVER_MODEL: $SERVER_MODEL"

# 定义服务端口范围
START_PORT=8001
NUM_SERVICES=6
# GPU 设置
GPUS_PER_SERVICE=1
ALL_GPUS=(2 3 4 5 6 7)
TOTAL_GPUS=${#ALL_GPUS[@]}

echo "Starting $NUM_SERVICES services with $GPUS_PER_SERVICE GPUs each..."

LOG_DIR=$SCRIPT_DIR/logs
mkdir -p "$LOG_DIR"


for ((i=0; i<NUM_SERVICES; i++)); do
    START_INDEX=$((i * GPUS_PER_SERVICE))
    END_INDEX=$((START_INDEX + GPUS_PER_SERVICE - 1))
    GPU_LIST=("${ALL_GPUS[@]:$START_INDEX:$GPUS_PER_SERVICE}")

    CUDA_VISIBLE_DEVICES=$(IFS=','; echo "${GPU_LIST[*]}")

    PORT=$((START_PORT + i))

    echo "Starting service on port $PORT using GPUs $CUDA_VISIBLE_DEVICES"

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    nohup \
    python -m vllm.entrypoints.openai.api_server \
        --port $PORT \
        --served-model-name $SERVER_MODEL \
        --model $MODEL_NAME_OR_PATH \
        --tensor-parallel-size $GPUS_PER_SERVICE \
        --gpu-memory-utilization 0.9 \
        --max-model-len 131072 \
        --max-num-batched-tokens 131072 \
        --trust-remote-code \
    > "$LOG_DIR/vllm_server_${PORT}.log" 2>&1 &
done

echo "Waiting for all vllm servers to start..."

for ((i=0; i<NUM_SERVICES; i++)); do
    PORT=$((START_PORT + i))
    while ! curl -s http://localhost:$PORT/health > /dev/null; do
        sleep 1
    done
    echo "vllm server on port $PORT has started."
done

echo "All vllm servers have started, proceeding with the next steps..."

# Run the evaluation script
API_KEY=EMPTY
API_BASES=()
for ((i=0; i<NUM_SERVICES; i++)); do
    PORT=$((START_PORT + i))
    API_BASES+=("http://localhost:$PORT/v1")
done

INPUT_PATH=${WORK_DIR}/data/ai_search_benchmark.json
SAVE_PATH=${WORK_DIR}/data/metrics_rlts/$SERVER_MODEL.jsonl

echo "Generated API_BASES: ${API_BASES[@]}"

python ${WORK_DIR}/src/ai_search/search.py \
    --num_processes $NUM_SERVICES\
    --model_name $SERVER_MODEL \
    --api_key $API_KEY \
    --api_base ${API_BASES[@]} \
    --input_path $INPUT_PATH \
    --save_path $SAVE_PATH \
    --debug \

pkill -f "vllm.entrypoints.openai.api_server"
echo "vllm server has been stopped."
