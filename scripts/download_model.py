import os
from huggingface_hub import snapshot_download

# 设置模型保存路径
model_path = "./models/Qwen1.5-14B-Chat"
os.makedirs(model_path, exist_ok=True)

# 下载模型
print(f"开始下载Qwen1.5-14B-Chat模型到 {model_path}")
snapshot_download(
    repo_id="Qwen/Qwen1.5-14B-Chat",
    local_dir=model_path,
    token=None,  # 如果需要访问私有模型，请提供你的HF token
    ignore_patterns=["*.bin", "*.pt"],  # 如果只需要safetensors格式，可以忽略其他格式
)
print("模型下载完成！") 