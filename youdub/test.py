'''

from kokoro_onnx import Kokoro
import soundfile as sf

# ✅ 正确初始化（按 thewh1teagle 版 API）
tts = Kokoro(r"models\kokoro-onnx\kokoro-v1.0.fp16-gpu.onnx", r"models\kokoro-onnx\voices-v1.0.bin")

# 合成（注意：返回 samples 和 sample_rate）
samples, sample_rate = tts.create(
    text="你好，你是煞笔吗？This is a test of the Kokoro TTS system",
    voice="af_sky",
    speed=1.0
)

# 保存
sf.write("output.wav", samples, sample_rate)
'''



'''
import ChatTTS
import soundfile as sf

chat = ChatTTS.Chat()
chat.load(compile=False)  # CPU 模式

text = "你好，Hello World！今天是2026年1月3日。"
wavs = chat.infer(
    text,
    params=ChatTTS.TextParams(
        speed=1.2  # 语速 1.2x
    )
)

sf.write("output.wav", wavs[0], 24000)

'''



import soundfile as sf
from voxcpm import VoxCPM

model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")

wav = model.generate(
    text="你好啊，这个是什么系统？VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly expressive speech.",
    prompt_wav_path=None,      # optional: path to a prompt speech for voice cloning
    prompt_text=None,          # optional: reference text
    cfg_value=2.0,             # LM guidance on LocDiT, higher for better adherence to the prompt, but maybe worse
    inference_timesteps=10,   # LocDiT inference timesteps, higher for better result, lower for fast speed
    normalize=True,           # enable external TN tool
    denoise=True,             # enable external Denoise tool
    retry_badcase=True,        # enable retrying mode for some bad cases (unstoppable)
    retry_badcase_max_times=3,  # maximum retrying times
    retry_badcase_ratio_threshold=6.0, # maximum length restriction for bad case detection (simple but effective), it could be adjusted for slow pace speech
)

sf.write("output.wav", wav, 16000)
print("saved: output.wav")




# test_voxcpm.py
import os
import soundfile as sf
from voxcpm import VoxCPM

print("🔊 正在加载 VoxCPM-0.5B 模型（首次运行会自动下载，约2.1GB）...")
model = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")

# 中英混合测试文本（验证多语言能力）
text = "你好，Hello World！欢迎使用 VoxCPM 语音合成系统，这是2025年发布的开源TTS模型。"

print(f"📝 输入文本: {text}")
print("⏳ 正在合成语音...")

wav = model.generate(
    text=text,
    normalize=True,          # 启用文本规范化（处理数字、标点）
    inference_timesteps=10,  # 质量/速度平衡（6~20）
    cfg_value=2.0,           # 遵循文本强度
    denoise=False,           # 关闭降噪（避免依赖额外模型）
    retry_badcase=False      # 关闭重试（加速测试）
)

# 保存音频
output_path = "test_voxcpm_output.wav"
sf.write(output_path, wav, 16000)

print(f"✅ 合成完成！音频已保存到: {os.path.abspath(output_path)}")
print("🎧 请用音频播放器打开试听。")