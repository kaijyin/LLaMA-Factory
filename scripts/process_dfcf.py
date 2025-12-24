import os
import time
import pandas as pd
import requests
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
INPUT_DIR = "/home/text_data/dfcf/dfcf_raw"
OUTPUT_DIR = "/home/user150/LLaMA-Factory/data/dfcf"
MODEL_NAME = "qwen3-32b"
SYSTEM_PROMPT = """你是一位专业的金融分析师，擅长情感分析。你的任务是分析金融文本的情感，并将其分类为以下三类之一：positive（积极）、neutral（中性）或 negative（消极）。

分类指南：
- positive：表示乐观的前景、增长潜力、有利的市场条件或良好的财务表现。
- neutral：表示事实性信息，没有明显的积极或消极含义。
- negative：表示悲观的前景、下滑、不利的市场条件或糟糕的财务表现。

只回复一个词：positive、neutral 或 negative。"""


def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0  # Default to neutral for empty text

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"分析以下金融文本的情感，将其分类为 positive（积极）、neutral（中性）或 negative（消极）。\n\n文本: {text}",
            },
        ],
        "temperature": 0.0,  # Deterministic
        "max_tokens": 1024,
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip().lower()

        if "positive" in content:
            print(f"Positive detected in {content}: {text}")
            return 1
        elif "negative" in content:
            print(f"Negative detected in {content}: {text}")
            return -1
        else:
            print(f"Neutral detected in {content}: {text}")
            return 0
    except Exception as e:
        print(f"Error processing text: {text[:50]}... Error: {e}")
        return 0


def process_file(file_path):
    filename = os.path.basename(file_path)
    output_path = os.path.join(OUTPUT_DIR, filename.replace(".csv", ".parquet"))

    if os.path.exists(output_path):
        print(f"Skipping {filename}, already processed.")
        return

    print(f"Processing {filename}...")
    try:
        # Read CSV, handling potential encoding issues or bad lines
        try:
            df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding="gbk", on_bad_lines="skip")

        if "post_content" not in df.columns:
            print(f"Warning: 'post_content' column not found in {filename}. Skipping.")
            return

        # Use ThreadPoolExecutor for parallel API calls
        # Adjust max_workers based on vLLM capacity
        results = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            results = list(
                tqdm(
                    executor.map(get_sentiment, df["post_content"]),
                    total=len(df),
                    desc=f"Analyzing {filename}",
                )
            )

        df["emotion"] = results

        # Save as Parquet
        df.to_parquet(output_path, index=False)
        print(f"Saved {output_path}")

    except Exception as e:
        print(f"Failed to process {filename}: {e}")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Found {len(files)} files to process.")

    for file in files:
        process_file(file)


if __name__ == "__main__":
    main()
