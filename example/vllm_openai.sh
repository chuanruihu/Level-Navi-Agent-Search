#!/bin/bash
source /mnt/workspace/envs/wbx_llama/bin/activate
export CUDA_VISIBLE_DEVICES=7

MODEL_NAME_OR_PATH=/mnt/workspace/pretrain_model/qwen2/qwen/Qwen2.5-7B-Instruct
# MODEL_NAME_OR_PATH=/mnt/workspace/wangbaoxin_wkspace/LLaMA-Factory/models/qwen2_5_full_mindsearch_template_v4_only_graph
SERVER_MODEL=$(basename "$MODEL_NAME_OR_PATH")

echo "MODEL_NAME_OR_PATH: $MODEL_NAME_OR_PATH"
echo "SERVER_MODEL: $SERVER_MODEL"

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME_OR_PATH \
    --served-model-name $SERVER_MODEL \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \