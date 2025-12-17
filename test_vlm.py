import requests


def analyze_sentiment(text):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "financial-sentiment",
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze the sentiment of this financial text and classify it as positive, negative, or neutral.\n\nText: {text}",
                }
            ],
            "max_tokens": 10,
            "temperature": 0,
        },
    )
    return response.json()["choices"][0]["message"]["content"]


# 测试
print(analyze_sentiment("this is not good"))  # negative
