#!/bin/bash
# 使用 vLLM 部署合并后的模型（多卡版本）

# ==================== 配置 ====================
export CUDA_VISIBLE_DEVICES=5,6  # 使用 GPU 5,6

# ==================== Step 1: 合并 LoRA ====================
# 如果还没合并，先执行合并（单卡即可，主要用 CPU 内存）
# CUDA_VISIBLE_DEVICES=5 llamafactory-cli export examples/merge_lora/qwen3_14b_financial_sentiment_merge.yaml

# ==================== Step 2: 启动 vLLM 服务 (3卡并行) ====================
python -m vllm.entrypoints.openai.api_server \
    --model /home/user150/models/Qwen3-14B-Financial-Sentiment-Merged \
    --served-model-name financial-sentiment \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16

# ==================== 测试 API ====================
# curl http://localhost:8000/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "financial-sentiment",
#     "messages": [{"role": "user", "content": "Analyze the sentiment of this financial text and classify it as positive, negative, or neutral.\n\nText: Apple stock surges 5%"}]
#   }'
\
