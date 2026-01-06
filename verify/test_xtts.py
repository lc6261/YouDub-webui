# test_xtts.py
import os

# 👇 设置 TTS 模型存储目录为 C:\model
os.environ["TTS_HOME"] = r"C:\model"

from TTS.api import TTS
print("Downloading XTTS model (one-time)...")
print(f"Model will be saved to: {os.environ['TTS_HOME']}")

# 加载模型（会自动下载到 C:\model\tts_models\...）
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
print("✅ Done! Model saved locally.")