# coding=utf-8
'''
字节跳动 TTS 模块（支持 x-api-key 认证 + 动态 speed_ratio）
适用于火山引擎 TTS 最新 API（2026）
requires: requests, librosa, numpy, loguru, python-dotenv
'''
import base64
import json
import os
import time
import uuid
import librosa
import numpy as np
import requests
from loguru import logger
from dotenv import load_dotenv
import traceback

# 尝试导入 pyannote，如果失败则提供回退方案
try:
    from pyannote.audio import Model, Inference
    from scipy.spatial.distance import cosine
    HAS_PYANNOTE = True
except ImportError:
    HAS_PYANNOTE = False
    logger.warning("pyannote.audio 未安装，将使用简化的语音类型匹配")

load_dotenv()

# === 新认证方式：使用 x-api-key ===
API_KEY = os.getenv('BYTEDANCE_API_KEY')
BYTEDANCE_AVAILABLE = bool(API_KEY)

if not BYTEDANCE_AVAILABLE:
    logger.warning("字节跳动 TTS 环境变量未设置，将仅使用本地 XTTS")
    logger.info("请在 .env 文件中设置: BYTEDANCE_API_KEY=your_api_key")

# API 配置（注意：URL 末尾不能有空格！）
API_URL = "https://openspeech.bytedance.com/api/v1/tts"  # 🔥 修复：移除末尾空格！

# 初始化 embedding 模型（如果可用）
embedding_model = None
embedding_inference = None
hf_token = os.getenv('HF_TOKEN')

if HAS_PYANNOTE and BYTEDANCE_AVAILABLE:
    try:
        logger.info("正在加载 pyannote/embedding 模型...")
        os.environ['PYANNOTE_CACHE'] = os.path.expanduser('~/.cache/pyannote')
        embedding_model = Model.from_pretrained(
            "pyannote/embedding",
            use_auth_token=hf_token,
            cache_dir=os.environ.get('PYANNOTE_CACHE')
        )
        embedding_inference = Inference(embedding_model, window="whole")
        logger.info("✅ pyannote/embedding 模型加载成功")
    except Exception as e:
        logger.warning(f"pyannote/embedding 模型加载失败: {e}")
        HAS_PYANNOTE = False


def generate_embedding_simple(wav_path):
    try:
        wav, sr = librosa.load(wav_path, sr=24000, duration=3.0)
        mfccs = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])
    except Exception as e:
        logger.warning(f"简化特征提取失败: {e}")
        return np.zeros(26)


def generate_embedding(wav_path):
    if HAS_PYANNOTE and embedding_inference is not None:
        try:
            return embedding_inference(wav_path)
        except Exception as e:
            logger.warning(f"pyannote 嵌入生成失败: {e}")
    return generate_embedding_simple(wav_path)


def cosine_similarity(vec1, vec2):
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return 1 - cosine(vec1, vec2)


def generate_speaker_to_voice_type(folder):
    speaker_to_voice_type_path = os.path.join(folder, 'speaker_to_voice_type.json')
    if os.path.exists(speaker_to_voice_type_path):
        try:
            with open(speaker_to_voice_type_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载语音类型映射失败: {e}")

    speaker_folder = os.path.join(folder, 'SPEAKER')
    if not os.path.exists(speaker_folder):
        logger.warning(f"SPEAKER 文件夹不存在: {speaker_folder}")
        return {"SPEAKER_00": "BV701_streaming", "SPEAKER_01": "BV701_streaming"}

    # 尝试加载预定义 voice_type
    voice_types = {}
    voice_type_dir = 'voice_type'
    if os.path.exists(voice_type_dir):
        for file in os.listdir(voice_type_dir):
            if file.endswith('.npy'):
                vt = file.replace('.npy', '')
                try:
                    voice_types[vt] = np.load(os.path.join(voice_type_dir, file))
                except Exception as e:
                    logger.warning(f"加载 {vt} 失败: {e}")

    speaker_to_voice_type = {}
    if not voice_types:
        # 默认映射
        for f in os.listdir(speaker_folder):
            if f.endswith('.wav'):
                speaker = f.replace('.wav', '')
                speaker_to_voice_type[speaker] = "BV701_streaming"
        try:
            with open(speaker_to_voice_type_path, 'w', encoding='utf-8') as f:
                json.dump(speaker_to_voice_type, f, indent=2, ensure_ascii=False)
        except:
            pass
        return speaker_to_voice_type

    # 基于 embedding 匹配
    for f in os.listdir(speaker_folder):
        if not f.endswith('.wav'):
            continue
        speaker = f.replace('.wav', '')
        wav_path = os.path.join(speaker_folder, f)
        try:
            emb = generate_embedding(wav_path)
            np.save(wav_path.replace('.wav', '.npy'), emb)
            best_vt, best_sim = None, -1
            for vt, vt_emb in voice_types.items():
                sim = cosine_similarity(emb, vt_emb)
                if sim > best_sim:
                    best_sim, best_vt = sim, vt
            speaker_to_voice_type[speaker] = best_vt or "BV701_streaming"
            logger.info(f'{speaker}: {best_vt} (相似度: {best_sim:.3f})')
        except Exception as e:
            logger.error(f"处理 {speaker} 失败: {e}")
            speaker_to_voice_type[speaker] = "BV701_streaming"

    try:
        with open(speaker_to_voice_type_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_to_voice_type, f, indent=2, ensure_ascii=False)
    except:
        pass
    return speaker_to_voice_type


# ========================
# ✅ 升级：支持 target_duration
# ========================
def tts(text, output_path, speaker_wav=None, voice_type=None, target_duration=None):
    if not BYTEDANCE_AVAILABLE:
        logger.warning("字节跳动 TTS 不可用，请检查 .env 中的 BYTEDANCE_API_KEY")
        return False

    if os.path.exists(output_path):
        logger.info(f'火山TTS 音频已存在: {output_path}')
        return True

    # 确定 voice_type
    if voice_type is None and speaker_wav:
        folder = os.path.dirname(os.path.dirname(output_path))
        mapping = generate_speaker_to_voice_type(folder)
        speaker = os.path.basename(speaker_wav).replace('.wav', '')
        voice_type = mapping.get(speaker, "BV701_streaming")
    if voice_type is None:
        voice_type = "BV701_streaming"

    # ✅ 动态计算 speed_ratio（关键升级！）
    speed_ratio = 1.0
    if target_duration is not None and target_duration > 0:
        # 保守估计：5 字/秒
        expected_duration = max(0.8, len(text) / 5.0)
        speed_ratio = expected_duration / target_duration
        # 火山引擎安全范围（实测）
        speed_ratio = np.clip(speed_ratio, 0.7, 1.8)
    
    logger.info(f"使用语音类型: {voice_type}, speed_ratio: {speed_ratio:.2f}")

    # 构造请求
    payload = {
        "app": {
            "cluster": "volcano_tts"
        },
        "user": {
            "uid": "youdub"
        },
        "audio": {
            "voice_type": voice_type,
            "encoding": "wav",
            "speed_ratio": speed_ratio,   # ← 动态值！
            "volume_ratio": 1.0,
            "pitch_ratio": 1.0,
        },
        "request": {
            "reqid": str(uuid.uuid4()).replace("-", "")[:32],
            "text": text,
            "text_type": "plain",
            "operation": "query",
        }
    }

    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    for retry in range(3):
        try:
            logger.debug(f"发送 TTS 请求: {text[:50]}... (目标={target_duration:.2f}s, speed_ratio={speed_ratio:.2f})")
            resp = requests.post(API_URL, json=payload, headers=headers, timeout=30)

            if resp.status_code == 200:
                data = resp.json()
                if "data" in data:
                    audio_data = base64.b64decode(data["data"])
                    with open(output_path, "wb") as f:
                        f.write(audio_data)
                    # 验证音频
                    wav, sr = librosa.load(output_path, sr=24000)
                    if len(wav) > 0:
                        logger.info(f'✅ 火山TTS 成功: {output_path}')
                        return True
                    else:
                        logger.warning("生成的音频为空")
                else:
                    logger.warning(f"响应无 data: {data}")
            else:
                logger.warning(f"TTS 失败 {resp.status_code}: {resp.text}")

        except Exception as e:
            logger.warning(f"TTS 异常 (重试 {retry+1}/3): {e}")
            logger.debug(traceback.format_exc())

        if retry < 2:
            time.sleep(1 * (retry + 1))

    logger.error("火山TTS 多次重试失败")
    return False


def get_available_speakers():
    if not BYTEDANCE_AVAILABLE:
        return False

    os.makedirs('voice_type', exist_ok=True)
    voice_types = [
        'BV001_streaming', 'BV002_streaming', 'BV700_streaming', 'BV701_streaming',
        'BV119_streaming', 'BV115_streaming', 'BV033_streaming'
    ]
    success = 0
    test_text = "测试音色。"

    for vt in voice_types:
        wav_path = f'voice_type/{vt}.wav'
        if os.path.exists(wav_path) and os.path.exists(wav_path.replace('.wav', '.npy')):
            continue
        if tts(test_text, wav_path, voice_type=vt):
            try:
                emb = generate_embedding(wav_path)
                np.save(wav_path.replace('.wav', '.npy'), emb)
                success += 1
                logger.info(f"✅ 获取音色: {vt}")
            except:
                pass
        time.sleep(0.5)

    logger.info(f"语音类型获取完成: {success}/{len(voice_types)}")
    return success > 0


def create_default_voice_mapping(folder):
    speaker_folder = os.path.join(folder, 'SPEAKER')
    mapping = {}
    if os.path.exists(speaker_folder):
        speakers = sorted([f.replace('.wav', '') for f in os.listdir(speaker_folder) if f.endswith('.wav')])
        voices = ['BV701_streaming', 'BV700_streaming', 'BV119_streaming']
        for i, spk in enumerate(speakers):
            mapping[spk] = voices[i % len(voices)]
    return mapping or {"SPEAKER_00": "BV701_streaming"}


if __name__ == '__main__':
    if BYTEDANCE_AVAILABLE:
        logger.info("🔥 使用 x-api-key 测试火山引擎 TTS")
        test_file = f"test_bytedance_{uuid.uuid4().hex[:8]}.wav"
        # 测试不同 target_duration
        if tts("你好，火山引擎 TTS 已成功接入 YouDub！", test_file, voice_type="BV701_streaming", target_duration=2.0):
            logger.info(f"🎉 测试成功！音频: {test_file}")
        else:
            logger.error("❌ 测试失败")
    else:
        logger.warning("⚠️ 请设置 BYTEDANCE_API_KEY")
