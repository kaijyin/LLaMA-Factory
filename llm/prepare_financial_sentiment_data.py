"""
金融情感分析数据集处理脚本
将 Financial PhraseBank (FPB), Twitter Financial News Sentiment (TFNS),
和 GPT-labeled Financial News (NWGI) 三个数据集转换为 LLaMA-Factory 所需的格式

使用方法:
    python prepare_financial_sentiment_data.py --output_dir /path/to/output

    # 如果无法访问 HuggingFace，使用镜像站:
    HF_ENDPOINT=https://hf-mirror.com python prepare_financial_sentiment_data.py --output_dir /path/to/output
"""

import json
import os
import argparse
import sys

# 检查是否需要使用镜像（必须在导入 datasets 之前设置）
if "--use_mirror" in sys.argv or os.environ.get("HF_ENDPOINT"):
    mirror_url = "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = mirror_url
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = mirror_url
    print(f"Using HuggingFace mirror: {mirror_url}")

from datasets import load_dataset
from typing import List, Dict
import random


# 情感标签统一映射
SENTIMENT_MAP = {
    # FPB 标签
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    # TFNS 标签
    "Bullish": "positive",
    "Bearish": "negative",
    "Neutral": "neutral",
    # NWGI 7类标签 -> 3类
    "strong positive": "positive",
    "moderately positive": "positive",
    "mildly positive": "positive",
    "strong negative": "negative",
    "moderately negative": "negative",
    "mildly negative": "negative",
}


def get_system_prompt():
    """获取金融情感分析系统提示词"""
    return """You are an expert financial analyst specializing in sentiment analysis. Your task is to analyze the sentiment of financial texts and classify them into one of three categories: positive, neutral, or negative.

Guidelines:
- positive: Indicates optimistic outlook, growth potential, favorable market conditions, or good financial performance.
- neutral: Indicates factual information without clear positive or negative implications.
- negative: Indicates pessimistic outlook, decline, unfavorable market conditions, or poor financial performance.

Respond with only one word: positive, neutral, or negative."""


def get_instruction_template():
    """获取指令模板"""
    return "Analyze the sentiment of the following financial text and classify it as positive, neutral, or negative.\n\nText: {text}"


def normalize_label(label, dataset_name: str) -> str:
    """统一不同数据集的标签"""
    if isinstance(label, int):
        # FPB 使用数字标签: 0=negative, 1=neutral, 2=positive
        if dataset_name == "fpb":
            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            return label_map.get(label, "neutral")
        # TFNS 使用数字标签: 0=Bearish, 1=Bullish, 2=Neutral
        elif dataset_name == "tfns":
            label_map = {0: "negative", 1: "positive", 2: "neutral"}
            return label_map.get(label, "neutral")

    # 字符串标签
    label_str = str(label).lower().strip()
    return SENTIMENT_MAP.get(label_str, SENTIMENT_MAP.get(label, "neutral"))


def save_raw_dataset(dataset, name: str, output_dir: str):
    """保存原始数据集到 raw_data 子目录"""
    raw_dir = os.path.join(output_dir, "raw_data")
    os.makedirs(raw_dir, exist_ok=True)

    raw_data = []
    for split in dataset.keys():
        for item in dataset[split]:
            raw_item = dict(item)
            raw_item["_split"] = split
            raw_data.append(raw_item)

    filepath = os.path.join(raw_dir, f"{name}_raw.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, ensure_ascii=False, indent=2)

    print(f"  📁 Saved raw data: {filepath} ({len(raw_data)} samples)")


def process_fpb(subset: str = "sentences_allagree", output_dir: str = None) -> List[Dict]:
    """
    处理 Financial PhraseBank 数据集

    Args:
        subset: 数据子集，可选值:
            - sentences_allagree: 所有标注者一致 (~2,264条)
            - sentences_75agree: 75%以上一致 (~3,453条)
            - sentences_66agree: 66%以上一致 (~4,217条)
            - sentences_50agree: 50%以上一致 (~4,846条)
        output_dir: 输出目录，用于保存原始数据
    """
    print(f"Loading Financial PhraseBank ({subset})...")

    try:
        # 尝试不同的加载方式
        try:
            dataset = load_dataset("takala/financial_phrasebank", subset)
        except Exception as e1:
            print(f"  First attempt failed: {e1}")
            # 尝试使用 FinancialPhraseBank 的替代仓库
            try:
                dataset = load_dataset("financial_phrasebank", subset)
            except Exception as e2:
                print(f"  Second attempt failed: {e2}")
                raise e1

        data = dataset["train"]

        # 保存原始数据
        if output_dir:
            save_raw_dataset(dataset, f"fpb_{subset}", output_dir)

    except Exception as e:
        print(f"Error loading FPB: {e}")
        print("FPB dataset failed to load. Skipping...")
        return []

    processed = []
    for item in data:
        text = item.get("sentence", item.get("text", ""))
        label = item.get("label")

        if not text:
            continue

        sentiment = normalize_label(label, "fpb")

        processed.append(
            {
                "instruction": get_instruction_template().format(text=text),
                "input": "",
                "output": sentiment,
                "system": get_system_prompt(),
                "source": "fpb",
            }
        )

    print(f"Processed {len(processed)} samples from FPB")
    return processed


def process_tfns(output_dir: str = None) -> List[Dict]:
    """处理 Twitter Financial News Sentiment 数据集"""
    print("Loading Twitter Financial News Sentiment...")

    try:
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

        # 保存原始数据
        if output_dir:
            save_raw_dataset(dataset, "tfns", output_dir)

    except Exception as e:
        print(f"Error loading TFNS: {e}")
        print("TFNS dataset failed to load. Skipping...")
        return []

    processed = []

    # 处理训练集和验证集
    for split in ["train", "validation"]:
        if split in dataset:
            for item in dataset[split]:
                text = item.get("text", "")
                label = item.get("label")

                if not text:
                    continue

                sentiment = normalize_label(label, "tfns")

                processed.append(
                    {
                        "instruction": get_instruction_template().format(text=text),
                        "input": "",
                        "output": sentiment,
                        "system": get_system_prompt(),
                        "source": "tfns",
                    }
                )

    print(f"Processed {len(processed)} samples from TFNS")
    return processed


def process_nwgi(output_dir: str = None) -> List[Dict]:
    """处理 GPT-labeled Financial News (NWGI) 数据集"""
    print("Loading GPT-labeled Financial News (NWGI)...")

    try:
        dataset = load_dataset("oliverwang15/news_with_gpt_instructions")

        # 保存原始数据
        if output_dir:
            save_raw_dataset(dataset, "nwgi", output_dir)

    except Exception as e:
        print(f"Error loading NWGI: {e}")
        print("NWGI dataset failed to load. Skipping...")
        return []

    processed = []

    # NWGI 可能有多个split
    for split in dataset.keys():
        for item in dataset[split]:
            # NWGI 的字段可能是 news/text/content 和 label/sentiment
            text = item.get("news", item.get("text", item.get("content", "")))
            label = item.get("label", item.get("sentiment", ""))

            if not text:
                continue

            # 将7类标签映射到3类
            label_str = str(label).lower().strip()

            if "positive" in label_str:
                sentiment = "positive"
            elif "negative" in label_str:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            processed.append(
                {
                    "instruction": get_instruction_template().format(text=text),
                    "input": "",
                    "output": sentiment,
                    "system": get_system_prompt(),
                    "source": "nwgi",
                }
            )

    print(f"Processed {len(processed)} samples from NWGI")
    return processed


def split_dataset(data: List[Dict], train_ratio: float = 0.9, seed: int = 42) -> tuple:
    """将数据集划分为训练集和验证集"""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    eval_data = shuffled[split_idx:]

    return train_data, eval_data


def save_dataset(data: List[Dict], filepath: str):
    """保存数据集为JSON格式"""
    # 移除 source 字段（仅用于内部追踪）
    clean_data = [{k: v for k, v in item.items() if k != "source"} for item in data]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(clean_data)} samples to {filepath}")


def generate_dataset_info(output_dir: str) -> Dict:
    """生成 dataset_info.json 配置"""
    return {
        "financial_sentiment_train": {
            "file_name": "financial_sentiment_train.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
        "financial_sentiment_eval": {
            "file_name": "financial_sentiment_eval.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
        "financial_sentiment_all": {
            "file_name": "financial_sentiment_all.json",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        },
    }


def print_statistics(data: List[Dict], name: str):
    """打印数据集统计信息"""
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"{'='*50}")
    print(f"Total samples: {len(data)}")

    # 统计标签分布
    label_counts = {}
    source_counts = {}

    for item in data:
        label = item["output"]
        source = item.get("source", "unknown")

        label_counts[label] = label_counts.get(label, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} ({count/len(data)*100:.1f}%)")

    print("\nSource distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count} ({count/len(data)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare financial sentiment datasets for LLaMA-Factory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/user150/LLaMA-Factory/data",
        help="Output directory for processed datasets",
    )
    parser.add_argument(
        "--fpb_subset",
        type=str,
        default="sentences_allagree",
        choices=[
            "sentences_allagree",
            "sentences_75agree",
            "sentences_66agree",
            "sentences_50agree",
        ],
        help="FPB subset to use",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.9, help="Ratio of training data"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for data splitting"
    )
    parser.add_argument(
        "--use_mirror",
        action="store_true",
        help="Use HuggingFace mirror (hf-mirror.com) for users in China",
    )
    parser.add_argument(
        "--save_raw",
        action="store_true",
        default=True,
        help="Save raw datasets to raw_data/ subdirectory (default: True)",
    )
    parser.add_argument(
        "--no_save_raw",
        action="store_true",
        help="Do not save raw datasets",
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 是否保存原始数据
    save_raw = args.save_raw and not args.no_save_raw
    raw_output_dir = args.output_dir if save_raw else None

    # 处理各数据集
    all_data = []

    print("\n" + "="*50)
    print("Step 1: Loading and processing datasets...")
    print("="*50)

    # 1. Financial PhraseBank
    fpb_data = process_fpb(args.fpb_subset, raw_output_dir)
    all_data.extend(fpb_data)

    # 2. Twitter Financial News Sentiment
    tfns_data = process_tfns(raw_output_dir)
    all_data.extend(tfns_data)

    # 3. GPT-labeled Financial News
    nwgi_data = process_nwgi(raw_output_dir)
    all_data.extend(nwgi_data)

    # 检查是否有数据
    if len(all_data) == 0:
        print("\n" + "=" * 50)
        print("ERROR: No data loaded!")
        print("=" * 50)
        print("Please check your network connection or try using --use_mirror flag:")
        print("  python prepare_financial_sentiment_data.py --use_mirror")
        print("\nOr manually download datasets and place them in the output directory.")
        return

    # 打印统计信息
    print_statistics(all_data, "All Combined")

    # 划分训练集和验证集
    train_data, eval_data = split_dataset(all_data, args.train_ratio, args.seed)

    print_statistics(train_data, "Training Set")
    print_statistics(eval_data, "Evaluation Set")

    print("\n" + "="*50)
    print("Step 2: Saving processed datasets...")
    print("="*50)

    # 保存数据集
    save_dataset(
        train_data, os.path.join(args.output_dir, "financial_sentiment_train.json")
    )
    save_dataset(
        eval_data, os.path.join(args.output_dir, "financial_sentiment_eval.json")
    )
    save_dataset(
        all_data, os.path.join(args.output_dir, "financial_sentiment_all.json")
    )

    # 生成 dataset_info 配置
    dataset_info = generate_dataset_info(args.output_dir)

    print("\n" + "=" * 50)
    print("Dataset Info Configuration (add to dataset_info.json):")
    print("=" * 50)
    print(json.dumps(dataset_info, indent=2, ensure_ascii=False))

    # 保存配置到单独文件
    config_path = os.path.join(args.output_dir, "financial_sentiment_dataset_info.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"\nDataset info saved to: {config_path}")

    print("\n" + "=" * 50)
    print("📂 OUTPUT FILES:")
    print("=" * 50)
    print(f"  📄 {args.output_dir}/financial_sentiment_train.json  (训练集)")
    print(f"  📄 {args.output_dir}/financial_sentiment_eval.json   (验证集)")
    print(f"  📄 {args.output_dir}/financial_sentiment_all.json    (完整数据)")
    print(f"  📄 {args.output_dir}/financial_sentiment_dataset_info.json (配置)")
    if save_raw:
        print(f"\n  📁 {args.output_dir}/raw_data/  (原始数据目录)")
        print(f"     ├── fpb_{args.fpb_subset}_raw.json")
        print(f"     ├── tfns_raw.json")
        print(f"     └── nwgi_raw.json")

    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("=" * 50)
    print("1. Merge the dataset_info config into LLaMA-Factory/data/dataset_info.json")
    print("2. Use the training config file to start training")
    print(f"\nTotal samples prepared: {len(all_data)}")
    print(f"  - Training: {len(train_data)}")
    print(f"  - Evaluation: {len(eval_data)}")


if __name__ == "__main__":
    main()
