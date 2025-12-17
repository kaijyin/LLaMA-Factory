# Qwen3-32B 金融情感分析微调指南

> 使用 LLaMA-Factory 框架微调 Qwen3-32B 模型进行金融文本情感分析

---

## 📚 目录

1. [项目概述](#1-项目概述)
2. [数据集介绍](#2-数据集介绍)
3. [环境准备](#3-环境准备)
4. [数据处理](#4-数据处理)
5. [训练配置](#5-训练配置)
6. [开始训练](#6-开始训练)
7. [模型导出与合并](#7-模型导出与合并)
8. [模型推理](#8-模型推理)
9. [常见问题](#9-常见问题)

---

## 1. 项目概述

### 1.1 任务描述

将金融文本（新闻、推文等）分类为三种情感：
- **positive** (正面): 看涨、利好、增长
- **neutral** (中性): 客观陈述、无明显倾向
- **negative** (负面): 看跌、利空、下跌

### 1.2 文件结构

```
model_train/train_code/llm/
├── prepare_financial_sentiment_data.py    # 数据处理脚本
├── train_qwen3_financial_sentiment.sh     # 一键训练脚本
├── financial_sentiment_inference.py       # 推理脚本
└── README_金融情感分析微调指南.md          # 本文档

LLaMA-Factory/
├── data/
│   ├── dataset_info.json                  # 数据集配置（需更新）
│   ├── financial_sentiment_train.json     # 训练集（生成）
│   ├── financial_sentiment_eval.json      # 验证集（生成）
│   └── financial_sentiment_all.json       # 完整数据集（生成）
└── examples/train_lora/
    ├── qwen3_32b_financial_sentiment_lora_sft.yaml      # 基础配置
    └── qwen3_32b_financial_sentiment_lora_sft_ds3.yaml  # DeepSpeed配置
```

---

## 2. 数据集介绍

### 2.1 数据来源

| 数据集 | 来源 | 样本数 | 原始标签 | 说明 |
|--------|------|--------|----------|------|
| **FPB** | [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) | ~2,264 | positive/neutral/negative | 金融新闻短句，专家标注 |
| **TFNS** | [Twitter Financial News](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | ~11,930 | Bullish/Bearish/Neutral | 金融推文 |
| **NWGI** | [News with GPT Instructions](https://huggingface.co/datasets/oliverwang15/news_with_gpt_instructions) | ~16,200 | 7类细粒度标签 | GPT标注金融新闻 |

### 2.2 标签统一映射

```
FPB:   positive → positive,  neutral → neutral,  negative → negative
TFNS:  Bullish → positive,   Neutral → neutral,  Bearish → negative
NWGI:  *positive* → positive, neutral → neutral, *negative* → negative
```

### 2.3 数据格式 (LLaMA-Factory Alpaca格式)

```json
{
  "instruction": "Analyze the sentiment of the following financial text...\n\nText: Apple reported record quarterly revenue...",
  "input": "",
  "output": "positive",
  "system": "You are an expert financial analyst..."
}
```

---

## 3. 环境准备

### 3.1 安装 LLaMA-Factory

```bash
cd /home/user150/LLaMA-Factory

# 安装基础依赖
pip install -e ".[torch,metrics]"

# 安装 Flash Attention 2 (可选，加速训练)
pip install flash-attn --no-build-isolation
```

### 3.2 安装数据处理依赖

```bash
pip install datasets
```

### 3.3 验证安装

```bash
llamafactory-cli version
```

### 3.4 硬件要求

| 配置 | GPU | 显存要求 | 推荐配置 |
|------|-----|----------|----------|
| 基础 | 4×A100 80G | ~60GB/GPU | 单机多卡 |
| 推荐 | 8×A100 80G | ~35GB/GPU | DeepSpeed ZeRO-3 |
| 最低 | 4×A100 40G | 需要 offload | ZeRO-3 + CPU offload |

---

## 4. 数据处理

### 4.1 运行数据处理脚本

```bash
python /home/user150/model_train/train_code/llm/prepare_financial_sentiment_data.py \
    --output_dir /home/user150/LLaMA-Factory/data \
    --fpb_subset sentences_allagree \
    --train_ratio 0.9 \
    --seed 42
```

### 4.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `/home/user150/LLaMA-Factory/data` | 输出目录 |
| `--fpb_subset` | `sentences_allagree` | FPB子集选择 |
| `--train_ratio` | `0.9` | 训练集比例 |
| `--seed` | `42` | 随机种子 |

### 4.3 FPB 子集选择

| 子集名 | 样本数 | 说明 |
|--------|--------|------|
| `sentences_allagree` | ~2,264 | 所有标注者一致（质量最高） |
| `sentences_75agree` | ~3,453 | 75%以上一致 |
| `sentences_66agree` | ~4,217 | 66%以上一致 |
| `sentences_50agree` | ~4,846 | 50%以上一致（数量最多） |

### 4.4 更新 dataset_info.json

数据处理完成后，需要将以下配置添加到 `LLaMA-Factory/data/dataset_info.json`：

```json
{
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
```

**快捷命令（自动合并配置）：**

```bash
python << 'EOF'
import json

# 读取原始配置
with open("/home/user150/LLaMA-Factory/data/dataset_info.json", "r") as f:
    config = json.load(f)

# 添加新数据集
config.update({
    "financial_sentiment_train": {
        "file_name": "financial_sentiment_train.json",
        "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
    },
    "financial_sentiment_eval": {
        "file_name": "financial_sentiment_eval.json",
        "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
    },
    "financial_sentiment_all": {
        "file_name": "financial_sentiment_all.json",
        "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
    }
})

# 保存
with open("/home/user150/LLaMA-Factory/data/dataset_info.json", "w") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("✅ dataset_info.json 更新成功!")
EOF
```

---

## 5. 训练配置

### 5.1 基础配置 (qwen3_32b_financial_sentiment_lora_sft.yaml)

```yaml
### model
model_name_or_path: /home/user150/models/Qwen3-32B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target: all

### dataset
dataset: financial_sentiment_train
template: qwen3
cutoff_len: 1024
max_samples: 50000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen3-32b/lora/financial_sentiment
logging_steps: 10
save_steps: 500
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
report_to: tensorboard

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
gradient_checkpointing: true
flash_attn: fa2

### eval
eval_dataset: financial_sentiment_eval
val_size: 0.1
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 500
```

### 5.2 关键参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **lora_rank** | 64 | LoRA秩，越大表达能力越强，显存占用越大 |
| **lora_alpha** | 128 | 缩放因子，通常设为 2×rank |
| **lora_dropout** | 0.05 | 防止过拟合 |
| **lora_target** | all | 对所有线性层应用LoRA |
| **learning_rate** | 2e-5 | 大模型建议使用较小学习率 |
| **per_device_train_batch_size** | 1 | 32B模型显存有限 |
| **gradient_accumulation_steps** | 16 | 等效batch_size = 1×16×GPU数 |
| **num_train_epochs** | 3 | 分类任务2-3轮通常足够 |
| **cutoff_len** | 1024 | 情感分析文本较短 |
| **gradient_checkpointing** | true | 节省显存 |
| **flash_attn** | fa2 | Flash Attention 2 加速 |

### 5.3 DeepSpeed ZeRO-3 配置

适用于多GPU分布式训练，在基础配置上添加：

```yaml
deepspeed: examples/deepspeed/ds_z3_config.json
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

### 5.4 显存不足时的调整

如果遇到 OOM (Out of Memory)，按以下顺序调整：

1. 减小 `per_device_train_batch_size` → 1
2. 增大 `gradient_accumulation_steps` 保持等效batch_size
3. 减小 `lora_rank` → 32 或 16
4. 减小 `cutoff_len` → 512
5. 使用 DeepSpeed ZeRO-3 + CPU offload

---

## 6. 开始训练

### 6.1 方式一：使用一键脚本

```bash
# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 运行脚本（包含数据处理+训练）
bash /home/user150/model_train/train_code/llm/train_qwen3_financial_sentiment.sh
```

### 6.2 方式二：分步执行

```bash
cd /home/user150/LLaMA-Factory
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Step 1: 处理数据
python /home/user150/model_train/train_code/llm/prepare_financial_sentiment_data.py \
    --output_dir /home/user150/LLaMA-Factory/data

# Step 2: 更新 dataset_info.json (参考 4.4 节)

# Step 3: 开始训练
# 单机训练
llamafactory-cli train examples/train_lora/qwen3_32b_financial_sentiment_lora_sft.yaml

# 或 DeepSpeed 多卡训练
llamafactory-cli train examples/train_lora/qwen3_32b_financial_sentiment_lora_sft_ds3.yaml
```

### 6.3 使用 WebUI 训练（可选）

```bash
llamafactory-cli webui
```

然后在浏览器中配置参数并启动训练。

### 6.4 监控训练

```bash
# 查看 TensorBoard
tensorboard --logdir saves/qwen3-32b/lora/financial_sentiment

# 查看训练日志
tail -f saves/qwen3-32b/lora/financial_sentiment/trainer_log.jsonl
```

### 6.5 预计训练时间

| 配置 | 数据量 | 预计时间 |
|------|--------|----------|
| 8×A100 80G | ~30,000条 × 3 epochs | 2-4 小时 |
| 4×A100 80G | ~30,000条 × 3 epochs | 4-8 小时 |

---

## 7. 模型导出与合并

### 7.1 合并 LoRA 权重

训练完成后，将 LoRA 权重合并到基础模型中：

```bash
llamafactory-cli export \
    --model_name_or_path /home/user150/models/Qwen3-32B \
    --adapter_name_or_path saves/qwen3-32b/lora/financial_sentiment \
    --template qwen3 \
    --finetuning_type lora \
    --export_dir saves/qwen3-32b/merged/financial_sentiment \
    --export_size 4 \
    --export_device auto
```

### 7.2 参数说明

| 参数 | 说明 |
|------|------|
| `--adapter_name_or_path` | LoRA 权重路径 |
| `--export_dir` | 合并后模型保存路径 |
| `--export_size` | 分片数量 |
| `--export_device` | 使用的设备 |

### 7.3 只使用 LoRA 权重（不合并）

如果不想合并，也可以直接加载 LoRA 权重：

```python
from llamafactory.chat import ChatModel

model = ChatModel({
    "model_name_or_path": "/home/user150/models/Qwen3-32B",
    "adapter_name_or_path": "saves/qwen3-32b/lora/financial_sentiment",
    "template": "qwen3",
    "finetuning_type": "lora",
})
```

---

## 8. 模型推理

### 8.1 使用 vLLM 推理（推荐）

```bash
python /home/user150/model_train/train_code/llm/financial_sentiment_inference.py
```

### 8.2 使用 LLaMA-Factory CLI 推理

```bash
llamafactory-cli chat \
    --model_name_or_path saves/qwen3-32b/merged/financial_sentiment \
    --template qwen3
```

### 8.3 Python API 示例

```python
from vllm import LLM, SamplingParams

# 加载模型
llm = LLM(
    model="saves/qwen3-32b/merged/financial_sentiment",
    tensor_parallel_size=4,
    trust_remote_code=True,
)

# 构造提示词
system = "You are an expert financial analyst..."
text = "Apple reported record quarterly revenue of $123.9 billion."
prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nAnalyze the sentiment: {text}<|im_end|>\n<|im_start|>assistant\n"

# 推理
outputs = llm.generate([prompt], SamplingParams(temperature=0.1, max_tokens=10))
print(outputs[0].outputs[0].text)  # positive
```

### 8.4 批量推理

```python
from financial_sentiment_inference import load_financial_sentiment_model, analyze_sentiment

# 加载模型
llm = load_financial_sentiment_model("saves/qwen3-32b/merged/financial_sentiment")

# 批量分析
texts = [
    "Tesla shares plunged 12% after disappointing delivery numbers.",
    "The Federal Reserve announced it will maintain current interest rates.",
    "Amazon's cloud computing division AWS continues to show strong growth.",
]

results = analyze_sentiment(llm, texts)
for r in results:
    print(f"{r['sentiment']}: {r['text'][:50]}...")
```

---

## 9. 常见问题

### Q1: 训练时 OOM 怎么办？

**解决方案：**
1. 减小 `per_device_train_batch_size` 到 1
2. 增大 `gradient_accumulation_steps`
3. 启用 `gradient_checkpointing: true`
4. 减小 `lora_rank` 到 32 或 16
5. 使用 DeepSpeed ZeRO-3 + offload

### Q2: 数据集下载失败？

**解决方案：**
```bash
# 设置 Hugging Face 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后放到本地
```

### Q3: Flash Attention 安装失败？

**解决方案：**
```bash
# 确保 CUDA 版本匹配
pip install flash-attn --no-build-isolation

# 如果仍失败，可在配置中禁用
# flash_attn: disabled
```

### Q4: 如何断点续训？

**解决方案：**
```yaml
# 在配置文件中设置
resume_from_checkpoint: saves/qwen3-32b/lora/financial_sentiment/checkpoint-1000
```

### Q5: 如何调整学习率策略？

**解决方案：**
```yaml
lr_scheduler_type: cosine  # 可选: linear, cosine, constant
warmup_ratio: 0.1          # 预热比例
# 或使用 warmup_steps: 100
```

### Q6: 如何添加更多数据集？

**解决方案：**
1. 将数据转换为 alpaca 格式
2. 在 `dataset_info.json` 中添加配置
3. 在训练配置的 `dataset` 字段中用逗号分隔多个数据集名

```yaml
dataset: financial_sentiment_train,your_new_dataset
```

---

## 📞 参考资源

- [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)
- [LLaMA-Factory 文档](https://llamafactory.readthedocs.io/)
- [Qwen3 模型](https://huggingface.co/Qwen)
- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
- [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
- [News with GPT Instructions](https://huggingface.co/datasets/oliverwang15/news_with_gpt_instructions)

---

*文档创建日期: 2025年12月15日*
