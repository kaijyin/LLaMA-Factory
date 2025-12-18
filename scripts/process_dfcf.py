import os
import time
import pandas as pd
import requests
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configuration
API_URL = "http://localhost:8000/v1/chat/completions"
INPUT_DIR = "/home/text_data/dfcf"
OUTPUT_DIR = "/home/user150/LLaMA-Factory/data/dfcf"
MODEL_NAME = "financial-sentiment"

SYSTEM_PROMPT = """You are an expert financial analyst specializing in sentiment analysis. Your task is to analyze the sentiment of financial texts and classify them into one of three categories: positive, neutral, or negative.

Guidelines:
- positive: Indicates optimistic outlook, growth potential, favorable market conditions, or good financial performance.
- neutral: Indicates factual information without clear positive or negative implications.
- negative: Indicates pessimistic outlook, decline, unfavorable market conditions, or poor financial performance.

Respond with only one word: positive, neutral, or negative."""


def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0  # Default to neutral for empty text

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Analyze the sentiment of the following financial text and classify it as positive, neutral, or negative.\n\nText: {text}",
            },
        ],
        "temperature": 0.0,  # Deterministic
        "max_tokens": 10,
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip().lower()

        if "positive" in content:
            return 1
        elif "negative" in content:
            return -1
        else:
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
        with ThreadPoolExecutor(max_workers=20) as executor:
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


def wait_for_server():
    print("Waiting for vLLM server to be ready...")
    while True:
        try:
            response = requests.get("http://localhost:8000/v1/models")
            if response.status_code == 200:
                print("Server is ready!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    wait_for_server()

    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Found {len(files)} files to process.")

    for file in files:
        process_file(file)


if __name__ == "__main__":
    main()
