"""
金融情感分析模型推理脚本
使用微调后的 Qwen3-32B 模型进行情感分析
"""

from vllm import LLM, SamplingParams
import json
import os
from typing import List, Dict

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"


def load_financial_sentiment_model(
    model_path: str = "/home/user150/LLaMA-Factory/saves/qwen3-32b/merged/financial_sentiment",
    tensor_parallel_size: int = 4,
) -> LLM:
    """
    加载微调后的金融情感分析模型

    如果使用未合并的 LoRA 权重，需要先用 LLaMA-Factory 导出合并后的模型
    """
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=2048,
    )
    return llm


def get_system_prompt() -> str:
    """获取系统提示词（与训练时一致）"""
    return """You are an expert financial analyst specializing in sentiment analysis. Your task is to analyze the sentiment of financial texts and classify them into one of three categories: positive, neutral, or negative.

Guidelines:
- positive: Indicates optimistic outlook, growth potential, favorable market conditions, or good financial performance.
- neutral: Indicates factual information without clear positive or negative implications.
- negative: Indicates pessimistic outlook, decline, unfavorable market conditions, or poor financial performance.

Respond with only one word: positive, neutral, or negative."""


def format_prompt(text: str) -> str:
    """格式化输入提示词"""
    system = get_system_prompt()
    instruction = f"Analyze the sentiment of the following financial text and classify it as positive, neutral, or negative.\n\nText: {text}"

    # Qwen3 对话模板格式
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def analyze_sentiment(
    llm: LLM,
    texts: List[str],
    temperature: float = 0.1,
    max_tokens: int = 10,
) -> List[Dict]:
    """
    批量分析文本情感

    Args:
        llm: 加载的模型
        texts: 待分析文本列表
        temperature: 采样温度（情感分类任务建议使用较低温度）
        max_tokens: 最大生成token数

    Returns:
        包含文本和情感标签的字典列表
    """
    # 格式化所有提示词
    prompts = [format_prompt(text) for text in texts]

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.9,
        max_tokens=max_tokens,
    )

    # 批量推理
    outputs = llm.generate(prompts, sampling_params)

    # 解析结果
    results = []
    for text, output in zip(texts, outputs):
        response = output.outputs[0].text.strip().lower()

        # 标准化输出
        if "positive" in response:
            sentiment = "positive"
        elif "negative" in response:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        results.append(
            {
                "text": text,
                "sentiment": sentiment,
                "raw_response": output.outputs[0].text,
            }
        )

    return results


def demo():
    """演示金融情感分析"""

    # 示例金融文本
    sample_texts = [
        "Apple reported record quarterly revenue of $123.9 billion, up 11% year over year.",
        "The stock market experienced significant volatility today amid recession fears.",
        "Tesla shares plunged 12% after disappointing delivery numbers.",
        "The Federal Reserve announced it will maintain current interest rates.",
        "Amazon's cloud computing division AWS continues to show strong growth momentum.",
        "Oil prices fell sharply as OPEC+ members disagreed on production cuts.",
        "Microsoft's quarterly earnings exceeded analyst expectations.",
        "The company announced a dividend increase of 10% for shareholders.",
        "Due to supply chain disruptions, the company expects lower profits next quarter.",
        "The merger between the two tech giants is expected to create significant synergies.",
    ]

    print("Loading model...")
    # 注意：如果模型尚未合并，使用原始模型路径
    # 实际部署时，建议使用合并后的模型路径
    llm = load_financial_sentiment_model(
        model_path="/home/user150/models/Qwen3-32B",  # 或合并后的路径
        tensor_parallel_size=4,
    )

    print("\nAnalyzing sentiments...")
    results = analyze_sentiment(llm, sample_texts)

    print("\n" + "=" * 80)
    print("Financial Sentiment Analysis Results")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Text: {result['text'][:100]}...")
        print(f"    Sentiment: {result['sentiment'].upper()}")

    # 统计结果
    sentiment_counts = {}
    for r in results:
        s = r["sentiment"]
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    for sentiment, count in sorted(sentiment_counts.items()):
        print(f"  {sentiment}: {count} ({count/len(results)*100:.1f}%)")


def batch_analyze_file(input_file: str, output_file: str):
    """
    批量分析文件中的文本

    Args:
        input_file: 输入文件路径（每行一条文本，或JSON格式）
        output_file: 输出文件路径
    """
    # 读取输入文件
    texts = []
    if input_file.endswith(".json"):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            texts = [item.get("text", item.get("content", "")) for item in data]
    else:
        with open(input_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(texts)} texts from {input_file}")

    # 加载模型
    llm = load_financial_sentiment_model()

    # 批量分析
    results = analyze_sentiment(llm, texts)

    # 保存结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    demo()
