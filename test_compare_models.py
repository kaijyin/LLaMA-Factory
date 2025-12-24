"""
金融情感分析模型测试脚本
对比 LoRA 微调模型 vs 原始模型的准确率
"""

import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ==================== 配置 ====================
BASE_MODEL_PATH = "/home/user150/models/Qwen3-14B"
LORA_PATH = "/home/user150/LLaMA-Factory/saves/qwen3-14b/qlora/financial_sentiment/checkpoint-2000"

# ==================== 加载模型 ====================
print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

# QLoRA: 4-bit 量化加载
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("正在加载基础模型 (4-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("正在创建微调模型 (加载 LoRA 权重)...")
finetuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)
finetuned_model.eval()

print("模型加载完成！\n")


# ==================== 工具函数 ====================
def extract_sentiment(text: str) -> str:
    """
    从模型输出中提取情感标签
    处理 <think>...</think> 格式，提取最终答案
    """
    # 去除 <think>...</think> 部分
    text_clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 如果清理后为空，尝试从原文提取
    if not text_clean:
        text_clean = text

    # 转小写便于匹配
    text_lower = text_clean.lower()

    # 按优先级匹配情感词
    if "positive" in text_lower:
        return "positive"
    elif "negative" in text_lower:
        return "negative"
    elif "neutral" in text_lower:
        return "neutral"
    else:
        # 尝试从原始文本（包括 think 部分）提取
        text_lower_full = text.lower()
        # 查找最后出现的情感词（通常是结论）
        last_pos = -1
        result = "unknown"
        for sentiment in ["positive", "negative", "neutral"]:
            pos = text_lower_full.rfind(sentiment)
            if pos > last_pos:
                last_pos = pos
                result = sentiment
        return result


# ==================== 推理函数 ====================
def predict_sentiment(
    model, text: str, use_lora: bool = True, max_tokens: int = 500
) -> str:
    """
    预测金融文本的情感
    use_lora: True 使用微调模型，False 使用原始模型
    """
    prompt = f"Analyze the sentiment of this financial text and classify it as positive, negative, or neutral.\n\nText: {text}"

    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        if use_lora:
            model.enable_adapter_layers()
        else:
            model.disable_adapter_layers()

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return response.strip()


# ==================== 测试数据（带标签） ====================
test_cases = [
    # (文本, 正确标签)
    ("今天不乐观", "negative"),
    (
        "Apple stock surges 5% after strong quarterly earnings beat expectations",
        "positive",
    ),
    ("The company announced massive layoffs affecting 10,000 employees", "negative"),
    ("Federal Reserve keeps interest rates unchanged as expected", "neutral"),
    ("Tesla shares plummet amid concerns over declining demand in China", "negative"),
    ("Microsoft reports record cloud revenue growth in Q3", "positive"),
    ("Oil prices remain stable despite geopolitical tensions", "neutral"),
    ("Amazon shares jump 8% on strong holiday shopping forecast", "positive"),
    ("Bank of America reports $2 billion loss in trading division", "negative"),
    ("The S&P 500 closed flat as investors await inflation data", "neutral"),
    ("Netflix subscriber growth exceeds analyst expectations", "positive"),
    ("Company faces massive lawsuit over environmental violations", "negative"),
]

print("=" * 100)
print("金融情感分析准确率对比：🎯 微调模型 vs 📦 原始模型")
print("=" * 100)

finetuned_correct = 0
original_correct = 0
total = len(test_cases)

results = []

for i, (text, label) in enumerate(test_cases, 1):
    print(f"\n【测试 {i}/{total}】")
    print(f"📰 文本: {text}")
    print(f"✅ 正确标签: {label}")
    print("-" * 100)

    # 微调模型
    finetuned_raw = predict_sentiment(
        finetuned_model, text, use_lora=True, max_tokens=50
    )
    finetuned_pred = extract_sentiment(finetuned_raw)
    finetuned_match = "✓" if finetuned_pred == label else "✗"
    if finetuned_pred == label:
        finetuned_correct += 1

    # 原始模型（需要更多 token 让它完成推理）
    original_raw = predict_sentiment(
        finetuned_model, text, use_lora=False, max_tokens=500
    )
    original_pred = extract_sentiment(original_raw)
    original_match = "✓" if original_pred == label else "✗"
    if original_pred == label:
        original_correct += 1

    print(
        f"🎯 微调模型: {finetuned_pred:10s} {finetuned_match}  (原始输出: {finetuned_raw[:50]})"
    )
    print(
        f"📦 原始模型: {original_pred:10s} {original_match}  (提取自 {len(original_raw)} 字符)"
    )

    results.append(
        {
            "text": text[:50],
            "label": label,
            "finetuned": finetuned_pred,
            "original": original_pred,
        }
    )

# ==================== 汇总统计 ====================
print("\n" + "=" * 100)
print("📊 准确率统计")
print("=" * 100)
print(
    f"🎯 微调模型准确率: {finetuned_correct}/{total} = {finetuned_correct/total*100:.1f}%"
)
print(
    f"📦 原始模型准确率: {original_correct}/{total} = {original_correct/total*100:.1f}%"
)
print(f"📈 提升: {(finetuned_correct - original_correct)/total*100:+.1f}%")
print("=" * 100)

# 详细结果表格
print("\n详细结果:")
print(
    f"{'序号':<4} {'标签':<10} {'微调':<10} {'原始':<10} {'微调正确':<8} {'原始正确':<8}"
)
print("-" * 60)
for i, r in enumerate(results, 1):
    ft_ok = "✓" if r["finetuned"] == r["label"] else "✗"
    og_ok = "✓" if r["original"] == r["label"] else "✗"
    print(
        f"{i:<4} {r['label']:<10} {r['finetuned']:<10} {r['original']:<10} {ft_ok:<8} {og_ok:<8}"
    )
