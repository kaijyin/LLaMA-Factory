#!/usr/bin/env python3
"""
对比微调前后模型在金融情感分析测试集上的表现 (vLLM API 版本)
"""

import json
import requests
import time
from collections import Counter
from typing import List, Dict, Tuple
import re

# 配置
FINETUNED_URL = "http://localhost:8001/v1/chat/completions"
BASE_URL = "http://localhost:8002/v1/chat/completions"
TEST_DATA_PATH = "/home/user150/LLaMA-Factory/data/financial_sentiment_test.json"

# 模型名称
FINETUNED_MODEL = "financial-sentiment-finetuned"
BASE_MODEL = "financial-sentiment-base"


def load_test_data(path: str, max_samples: int = None) -> List[Dict]:
    """加载测试数据"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if max_samples:
        data = data[:max_samples]
    print(f"Loaded {len(data)} test samples")
    return data


def extract_sentiment(response: str) -> str:
    """从响应中提取情感标签"""
    # 去除 <think>...</think> 部分
    response_clean = re.sub(
        r"<think>.*?</think>", "", response, flags=re.DOTALL
    ).strip()

    if not response_clean:
        response_clean = response

    response_lower = response_clean.lower().strip()

    # 首先尝试直接匹配
    if response_lower in ["positive", "neutral", "negative"]:
        return response_lower

    # 尝试从响应中提取
    for sentiment in ["positive", "neutral", "negative"]:
        if sentiment in response_lower:
            return sentiment

    # 从原始响应中查找最后出现的情感词
    response_lower_full = response.lower()
    last_pos = -1
    result = "unknown"
    for sentiment in ["positive", "negative", "neutral"]:
        pos = response_lower_full.rfind(sentiment)
        if pos > last_pos:
            last_pos = pos
            result = sentiment

    return result


def call_model(
    url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> str:
    """调用模型API"""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2048,
        "temperature": 0,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"Error calling {model_name}: {e}")
                return "error"
    return "error"


def wait_for_service(url: str, max_wait: int = 180) -> bool:
    """等待服务启动"""
    health_url = url.replace("/v1/chat/completions", "/health")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(5)
        print(f"Waiting for service at {health_url}...")

    return False


def evaluate_model(
    url: str, model_name: str, test_data: List[Dict]
) -> Tuple[float, Dict]:
    """评估单个模型"""
    correct = 0
    total = 0
    predictions = []
    labels = []

    confusion = {
        "positive": {"positive": 0, "neutral": 0, "negative": 0, "other": 0},
        "neutral": {"positive": 0, "neutral": 0, "negative": 0, "other": 0},
        "negative": {"positive": 0, "neutral": 0, "negative": 0, "other": 0},
    }

    print(f"\nEvaluating {model_name}...")

    for i, sample in enumerate(test_data):
        system_prompt = sample.get("system", "")
        user_prompt = sample["instruction"]
        if sample.get("input"):
            user_prompt += f"\n{sample['input']}"

        expected = sample["output"].lower().strip()

        # 调用模型
        response = call_model(url, model_name, system_prompt, user_prompt)
        predicted = extract_sentiment(response)

        predictions.append(predicted)
        labels.append(expected)

        if predicted == expected:
            correct += 1

        # 更新混淆矩阵
        if expected in confusion:
            if predicted in confusion[expected]:
                confusion[expected][predicted] += 1
            else:
                confusion[expected]["other"] += 1

        total += 1

        if (i + 1) % 50 == 0:
            print(
                f"  Progress: {i + 1}/{len(test_data)}, Current Accuracy: {correct/total:.4f}"
            )

    accuracy = correct / total if total > 0 else 0

    # 计算每个类别的精确率和召回率
    metrics = {}
    for sentiment in ["positive", "neutral", "negative"]:
        tp = confusion[sentiment][sentiment]
        fp = sum(
            confusion[other][sentiment]
            for other in ["positive", "neutral", "negative"]
            if other != sentiment
        )
        fn = sum(
            confusion[sentiment][other]
            for other in ["positive", "neutral", "negative", "other"]
            if other != sentiment
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        metrics[sentiment] = {"precision": precision, "recall": recall, "f1": f1}

    return accuracy, {
        "confusion": confusion,
        "metrics": metrics,
        "predictions": predictions,
        "labels": labels,
    }


def print_results(model_name: str, accuracy: float, details: Dict):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print(f"\nPer-class Metrics:")
    print(f"{'Sentiment':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 48)

    for sentiment in ["positive", "neutral", "negative"]:
        m = details["metrics"][sentiment]
        print(
            f"{sentiment:<12} {m['precision']:<12.4f} {m['recall']:<12.4f} {m['f1']:<12.4f}"
        )

    print(f"\nConfusion Matrix:")
    print(
        f"{'True/Pred':<12} {'positive':<12} {'neutral':<12} {'negative':<12} {'other':<12}"
    )
    print("-" * 60)

    for true_label in ["positive", "neutral", "negative"]:
        row = details["confusion"][true_label]
        print(
            f"{true_label:<12} {row['positive']:<12} {row['neutral']:<12} {row['negative']:<12} {row['other']:<12}"
        )


def main():
    print("=" * 60)
    print("Financial Sentiment Analysis Model Comparison")
    print("=" * 60)

    # 等待服务启动
    print("\nWaiting for services to be ready...")

    finetuned_ready = wait_for_service(FINETUNED_URL, max_wait=180)
    if not finetuned_ready:
        print("WARNING: Finetuned model service not ready!")
    else:
        print("✓ Finetuned model service is ready")

    base_ready = wait_for_service(BASE_URL, max_wait=180)
    if not base_ready:
        print("WARNING: Base model service not ready!")
    else:
        print("✓ Base model service is ready")

    # 加载测试数据
    test_data = load_test_data(TEST_DATA_PATH)

    results = {}

    # 评估微调后的模型
    if finetuned_ready:
        accuracy, details = evaluate_model(FINETUNED_URL, FINETUNED_MODEL, test_data)
        results["finetuned"] = {"accuracy": accuracy, "details": details}
        print_results("Finetuned Model (DS3)", accuracy, details)

    # 评估基础模型
    if base_ready:
        accuracy, details = evaluate_model(BASE_URL, BASE_MODEL, test_data)
        results["base"] = {"accuracy": accuracy, "details": details}
        print_results("Base Model (Qwen3-14B)", accuracy, details)

    # 对比总结
    if "finetuned" in results and "base" in results:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")

        finetuned_acc = results["finetuned"]["accuracy"]
        base_acc = results["base"]["accuracy"]
        improvement = finetuned_acc - base_acc

        print(f"Base Model Accuracy:      {base_acc:.4f} ({base_acc*100:.2f}%)")
        print(
            f"Finetuned Model Accuracy: {finetuned_acc:.4f} ({finetuned_acc*100:.2f}%)"
        )
        print(f"Improvement:              {improvement:+.4f} ({improvement*100:+.2f}%)")

        # 保存结果
        output_path = "/home/user150/LLaMA-Factory/model_comparison_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            # 不保存predictions和labels以节省空间
            save_results = {
                "finetuned": {
                    "accuracy": results["finetuned"]["accuracy"],
                    "confusion": results["finetuned"]["details"]["confusion"],
                    "metrics": results["finetuned"]["details"]["metrics"],
                },
                "base": {
                    "accuracy": results["base"]["accuracy"],
                    "confusion": results["base"]["details"]["confusion"],
                    "metrics": results["base"]["details"]["metrics"],
                },
                "improvement": improvement,
            }
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
