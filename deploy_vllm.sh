#!/bin/bash
# 使用 vLLM 部署合并后的模型（多卡版本）并处理数据

# ==================== 配置 ====================
export CUDA_VISIBLE_DEVICES=5,6  # 使用 GPU 5,6
MERGED_MODEL_PATH="/home/user150/models/Qwen3-14B-Financial-Sentiment-Merged"
LOG_FILE="vllm_server.log"

# ==================== Step 1: 合并 LoRA ====================
if [ ! -d "$MERGED_MODEL_PATH" ]; then
    echo "Merged model not found at $MERGED_MODEL_PATH. Exporting..."
    # CUDA_VISIBLE_DEVICES=5 llamafactory-cli export examples/merge_lora/qwen3_14b_financial_sentiment_merge.yaml
    # Assuming the user might need to run this manually or uncomment if needed.
    # Since I saw the files, I'll assume it's done.
    echo "Please ensure the model is merged. If not, uncomment the export command above."
else
    echo "Merged model found."
fi

# ==================== Step 2: 启动 vLLM 服务 (后台运行) ====================
echo "Starting vLLM server..."
nohup python -m vllm.entrypoints.openai.api_server \
    --model $MERGED_MODEL_PATH \
    --served-model-name financial-sentiment \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 512 \
    --gpu-memory-utilization 0.5 \
    --dtype bfloat16 > $LOG_FILE 2>&1 &

VLLM_PID=$!
echo "vLLM server started with PID $VLLM_PID. Logs in $LOG_FILE"

# ==================== Step 3: 运行数据处理脚本 ====================
echo "Running data processing script..."
python scripts/process_dfcf.py

# ==================== Step 4: 清理 ====================
echo "Processing complete. Stopping vLLM server..."
kill $VLLM_PID



