
# GPT-SoVITS API 使用示例

import requests
import json

# 1. 准备数据
data = {
    "refer_wav_path": "videos\\卢克文工作室\\20251229 美国斩杀线真相 政府资本合谋 底层毫无反抗之力\\audio.wav",
    "prompt_text": "这是一个参考语音的文本",
    "prompt_language": "zh",
    "text": "要合成的文本内容",
    "text_language": "zh",
    "cut_punc": "，。！？；",
    "top_k": 5,
    "top_p": 0.8,
    "temperature": 0.8,
    "batch_size": 1,
    "speed_factor": 1.0,
    "split_bucket": True,
}

# 2. 发送请求
response = requests.post(
    "http://localhost:9880/tts",
    json=data,
    timeout=30
)

# 3. 处理响应
if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print("✅ 语音合成成功")
else:
    print(f"❌ 合成失败: {response.text}")
