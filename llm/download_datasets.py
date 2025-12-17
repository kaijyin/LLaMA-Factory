"""
手动下载金融情感数据集
支持使用镜像站
"""
import os
import requests
from tqdm import tqdm

HF_MIRROR = "https://hf-mirror.com"

def download_file(url, save_path):
    """下载文件"""
    print(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"✅ Saved to: {save_path}")

def main():
    output_dir = "/home/user150/LLaMA-Factory/data/raw_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 更新后的正确URL
    datasets = {
        # Financial PhraseBank - sentences_allagree 子集
        "fpb": f"{HF_MIRROR}/datasets/takala/financial_phrasebank/resolve/main/sentences_allagree/train-00000-of-00001.parquet",
        
        # NWGI - News with GPT Instructions (尝试不同的文件路径)
        "nwgi_train": f"{HF_MIRROR}/datasets/oliverwang15/news_with_gpt_instructions/resolve/main/data/train-00000-of-00001.parquet",
    }
    
    for name, url in datasets.items():
        ext = url.split('.')[-1]
        save_path = os.path.join(output_dir, f"{name}.{ext}")
        
        # 跳过已存在的文件
        if os.path.exists(save_path):
            print(f"⏭️  {name} already exists, skipping...")
            continue
            
        try:
            download_file(url, save_path)
        except Exception as e:
            print(f"❌ Failed to download {name}: {e}")
            
    print("\n" + "="*50)
    print("Download Summary:")
    print("="*50)
    for f in os.listdir(output_dir):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  📄 {f} ({size:.1f} KB)")

if __name__ == "__main__":
    main()
