# test_volcano_tts_x_api_key.py
import os
import uuid
import base64
import json
import requests
from dotenv import load_dotenv

# 加载 .env
load_dotenv()

# 从 .env 读取 API Key（不是 access_token！）
API_KEY = os.getenv("BYTEDANCE_API_KEY")  # 注意：现在叫这个！

if not API_KEY:
    raise ValueError("❌ 请在 .env 文件中设置 BYTEDANCE_API_KEY")

API_URL = "https://openspeech.bytedance.com/api/v1/tts"
HEADERS = {
    "x-api-key": API_KEY,  # 👈 关键：使用 x-api-key
    "Content-Type": "application/json"
}

def test_volcano_tts_x_api_key(
    text="你好，火山引擎 TTS 测试成功！",
    voice_type="BV701_streaming",
    output_path="test_volcano_output.wav"
):
    # 注意：JSON 中不需要 appid/token！
    payload = {
        "app": {
            "cluster": "volcano_tts"
        },
        "user": {
            "uid": "youdub_test"
        },
        "audio": {
            "voice_type": voice_type,
            "encoding": "wav",  # 建议用 wav，避免 mp3 兼容问题
            "speed_ratio": 1.0,
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()).replace("-", ""),  # 确保无横杠（可选）
            "text": text,
            "text_type": "plain",
            "operation": "query",
        }
    }

    print(f"📤 使用 x-api-key 调用火山引擎 TTS...\n文本: {text}\n音色: {voice_type}")
    
    try:
        resp = requests.post(API_URL, json=payload, headers=HEADERS, timeout=30)
        print(f"📡 状态码: {resp.status_code}")
        
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data:
                audio_data = base64.b64decode(data["data"])
                with open(output_path, "wb") as f:
                    f.write(audio_data)
                print(f"✅ 音频已保存: {output_path}")
                print("🎉 火山引擎 TTS 调用成功！")
                return True
            else:
                print(f"❌ 响应错误: {data}")
        else:
            print(f"❌ 请求失败: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"💥 异常: {e}")
        import traceback
        traceback.print_exc()
    
    return False

if __name__ == "__main__":
    test_volcano_tts_x_api_key()