from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import run_server

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"  # 必须在导入 vllm 之前设置


# 方式一：使用 Python API
def deploy_qwen3_32b():
    # 初始化模型
    llm = LLM(
        model="/home/user150/models/Qwen3-32B",  # 或本地路径
        tensor_parallel_size=4,  # 根据 GPU 数量调整
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=8192,
    )

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2048,
    )

    # 推理示例
    prompts = ["你好，请介绍一下你自己"]
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        print(output.outputs[0].text)


if __name__ == "__main__":
    deploy_qwen3_32b()
