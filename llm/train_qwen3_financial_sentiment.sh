#!/bin/bash
#############################################################
# Qwen3-32B 金融情感分析微调 - 训练启动脚本
# 使用 LLaMA-Factory 框架
#############################################################

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # 根据实际GPU数量调整
export WANDB_DISABLED=true  # 如果不使用wandb，禁用它

# 进入 LLaMA-Factory 目录
cd /home/user150/LLaMA-Factory

#############################################################
# Step 1: 准备数据集
#############################################################
echo "Step 1: Preparing datasets..."

# 安装必要依赖
pip install datasets

# 运行数据准备脚本
python /home/user150/model_train/train_code/llm/prepare_financial_sentiment_data.py \
    --output_dir /home/user150/LLaMA-Factory/data \
    --fpb_subset sentences_allagree \
    --train_ratio 0.9 \
    --seed 42

#############################################################
# Step 2: 更新 dataset_info.json
#############################################################
echo "Step 2: Updating dataset_info.json..."

# 备份原始文件
cp /home/user150/LLaMA-Factory/data/dataset_info.json \
   /home/user150/LLaMA-Factory/data/dataset_info.json.bak

# 使用 Python 合并配置
python << 'EOF'
import json

# 读取原始 dataset_info.json
with open("/home/user150/LLaMA-Factory/data/dataset_info.json", "r") as f:
    dataset_info = json.load(f)

# 添加新的数据集配置
new_datasets = {
    "financial_sentiment_train": {
        "file_name": "financial_sentiment_train.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system"
        }
    },
    "financial_sentiment_eval": {
        "file_name": "financial_sentiment_eval.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system"
        }
    },
    "financial_sentiment_all": {
        "file_name": "financial_sentiment_all.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system"
        }
    }
}

dataset_info.update(new_datasets)

# 保存更新后的文件
with open("/home/user150/LLaMA-Factory/data/dataset_info.json", "w") as f:
    json.dump(dataset_info, f, indent=2, ensure_ascii=False)

print("dataset_info.json updated successfully!")
EOF

#############################################################
# Step 3: 开始训练
#############################################################
echo "Step 3: Starting training..."

# 获取GPU数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPUs"

# 选择训练方式：
# 方式1: 单GPU或少量GPU使用普通配置
# 方式2: 多GPU使用DeepSpeed ZeRO-3

if [ $NUM_GPUS -le 2 ]; then
    echo "Using single/dual GPU mode..."
    llamafactory-cli train examples/train_lora/qwen3_32b_financial_sentiment_lora_sft.yaml
else
    echo "Using multi-GPU mode with DeepSpeed ZeRO-3..."
    llamafactory-cli train examples/train_lora/qwen3_32b_financial_sentiment_lora_sft_ds3.yaml
fi

#############################################################
# Step 4: 合并LoRA权重 (可选)
#############################################################
# 训练完成后，如果需要合并LoRA权重，取消下面的注释：

# echo "Step 4: Merging LoRA weights..."
# llamafactory-cli export \
#     --model_name_or_path /home/user150/models/Qwen3-32B \
#     --adapter_name_or_path saves/qwen3-32b/lora/financial_sentiment \
#     --template qwen3 \
#     --finetuning_type lora \
#     --export_dir saves/qwen3-32b/merged/financial_sentiment \
#     --export_size 4 \
#     --export_device auto

echo "Training completed!"
echo "Model saved to: saves/qwen3-32b/lora/financial_sentiment"
