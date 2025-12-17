import pandas as pd

# 读取 Excel 文件
df = pd.read_excel(
    "/home/user150/model_train/train_code/llm/data/artificially tagging data.xlsx"
)
print(df.head())
