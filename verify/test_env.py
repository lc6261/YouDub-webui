# test_env.py
import os
from dotenv import load_dotenv

# 打印当前工作目录
print(f"当前目录: {os.getcwd()}")

# 尝试加载 .env 文件
env_path = os.path.join(os.getcwd(), '.env')
print(f".env 路径: {env_path}")
print(f".env 文件存在: {os.path.exists(env_path)}")

load_dotenv()

# 检查环境变量
print(f"HF_TOKEN: {os.getenv('HF_TOKEN')}")
print(f"BYTEDANCE_APPID: {os.getenv('BYTEDANCE_APPID')}")
print(f"BYTEDANCE_ACCESS_TOKEN: {os.getenv('BYTEDANCE_ACCESS_TOKEN')}")

import torch
import whisperx
import ctranslate2
import pyannote.audio
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU(s):", torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")


print("✅ PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
print("✅ faster-whisper + CTranslate2:", ctranslate2.__version__)
print("✅ pyannote.audio:", pyannote.audio.__version__)
print("✅ WhisperX ready!")


import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件

print("=== 环境变量测试 ===")
print(f"HF_HUB_CACHE: {os.getenv('HF_HUB_CACHE')}")
print(f"Type: {type(os.getenv('HF_HUB_CACHE'))}")
print(f"Length: {len(os.getenv('HF_HUB_CACHE'))}")

# 检查是否有引号
value = os.getenv('HF_HUB_CACHE')
if value:
    print(f"First char: '{value[0]}'")
    print(f"Last char: '{value[-1]}'")
    if value[0] == '"' or value[0] == "'":
        print("⚠️ WARNING: Value starts with a quote!")