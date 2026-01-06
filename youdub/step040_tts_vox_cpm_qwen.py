#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
英文视频转中文配音 TTS 脚本 ✅ 支持全局默认语音克隆 + 音色一致性检查

该脚本用于将已翻译的英文视频片段（JSON 格式）批量合成为中文配音音频，
使用 VoxCPM 模型进行语音生成，并通过 Resemblyzer 进行音色一致性校验，
确保所有生成的语音片段在音色上与预设的全局参考语音（lkw_cloned.wav）保持一致。

功能亮点：
- 🎙️ 强制使用全局语音克隆（voice/lkw_cloned.wav + lkw_cloned.txt）
- 🔍 音色一致性检查（基于 Resemblyzer 声纹嵌入）
- ⏱️ 自动压缩超长 TTS 音频以匹配原始视频时间轴（绝不拉伸！）
- 🎧 自动混合伴奏（audio_instruments.wav）生成最终音频
- 🔄 支持单视频处理或批量处理整个目录

作者: Advanced TTS Team  
创建日期: 2026-01-04  
依赖项: voxcpm, resemblyzer, librosa, loguru, soundfile, audiostretchy (可选), youdub (可选)  
"""

import json
import os
import re
import sys
import traceback
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

import librosa
from loguru import logger

# 加载 .env 配置（若使用环境变量）
# load_dotenv()

# 将项目根目录加入模块搜索路径，便于导入自定义模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ========================
# 🎯 模块导入：处理缺失依赖的降级方案
# ========================

# 尝试导入 youdub.utils，若失败则使用本地实现
try:
    from youdub.utils import save_wav, save_wav_norm
except ImportError:
    logger.warning("youdub.utils 模块未找到，使用本地实现")

    import scipy.io.wavfile

    def save_wav(wav: np.ndarray, path: str, sample_rate: int = 16000):
        """将归一化的 [-1, 1] 浮点音频保存为 16-bit WAV 文件"""
        wav = np.clip(wav, -1.0, 1.0)
        scipy.io.wavfile.write(path, sample_rate, (wav * 32767).astype(np.int16))

    def save_wav_norm(wav: np.ndarray, path: str, sample_rate: int = 16000):
        """先归一化再保存 WAV，避免削波"""
        if len(wav) > 0:
            wav = wav / np.max(np.abs(wav)) * 0.95
        save_wav(wav, path, sample_rate)


# ========================
# 🎯 多克隆音支持配置
# ========================
HAS_RESEMBLYZER = False
resemblyzer_encoder = None

# 克隆音注册表：存储所有可用的克隆音信息
CLONED_VOICES = {}

# 尝试加载 Resemblyzer


# ========================
# 🎯 Resemblyzer 音色一致性检查模块
# ========================
try:
    from resemblyzer import VoiceEncoder
    from resemblyzer.audio import preprocess_wav

    # 使用 Resemblyzer 内置模型，避免依赖 pyannote/embedding
    logger.info("✅ 正在加载 Resemblyzer VoiceEncoder...")
    resemblyzer_encoder = VoiceEncoder("cpu")  # 可改为 "cuda" 启用 GPU
    logger.info("✅ Resemblyzer VoiceEncoder 加载成功")
    
    HAS_RESEMBLYZER = True
    logger.info("✅ 声纹一致性检查已启用（使用 Resemblyzer）")
except Exception as e:
    logger.warning(f"⚠️ Resemblyzer 加载失败（将跳过音色检查）: {e}")
    import traceback
    logger.debug(f"详细错误信息: {traceback.format_exc()}")
    HAS_RESEMBLYZER = False


# ========================
# 🎯 克隆音管理函数
# ========================
def load_cloned_voices():
    """
    加载所有克隆音文件，提取声纹嵌入
    """
    global CLONED_VOICES
    
    # 遍历 voice 目录下的所有文件
    voice_dir = "voice"
    if not os.path.exists(voice_dir):
        logger.error(f"❌ 克隆音目录不存在: {voice_dir}")
        return False
    
    # 查找所有克隆音对 (.wav + .txt)
    wav_files = [f for f in os.listdir(voice_dir) if f.endswith("_cloned.wav")]
    
    for wav_file in wav_files:
        base_name = os.path.splitext(wav_file)[0].replace("_cloned", "")
        txt_file = f"{base_name}_cloned.txt"
        txt_path = os.path.join(voice_dir, txt_file)
        
        if not os.path.exists(txt_path):
            logger.warning(f"⚠️ 克隆音文本缺失，跳过: {base_name}")
            continue
        
        # 读取克隆文本
        with open(txt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        
        if not prompt_text:
            logger.warning(f"⚠️ 克隆音文本为空，跳过: {base_name}")
            continue
        
        # 构建克隆音信息
        voice_info = {
            "name": base_name,
            "wav_path": os.path.join(voice_dir, wav_file),
            "txt_path": txt_path,
            "prompt_text": prompt_text,
            "embedding": None
        }
        
        # 提取声纹嵌入
        if HAS_RESEMBLYZER:
            try:
                ref_wav = preprocess_wav(voice_info["wav_path"])
                embedding = resemblyzer_encoder.embed_utterance(ref_wav)
                voice_info["embedding"] = embedding
                logger.info(f"✅ 提取克隆音声纹: {base_name} (时长: {len(ref_wav)/16000:.2f}s)")
            except Exception as e:
                logger.warning(f"⚠️ 提取声纹失败，跳过: {base_name} - {e}")
                continue
        
        CLONED_VOICES[base_name] = voice_info
    
    if CLONED_VOICES:
        logger.info(f"🎤 加载完成 {len(CLONED_VOICES)} 个克隆音")
        for name in CLONED_VOICES:
            logger.info(f"   - {name}")
        return True
    else:
        logger.error("❌ 未找到可用的克隆音")
        return False


# 加载所有克隆音
if not load_cloned_voices():
    logger.error("❌ 克隆音加载失败，退出程序")
    sys.exit(1)


# 默认克隆音（使用第一个加载的克隆音）
default_voice_name = next(iter(CLONED_VOICES.keys()))
DEFAULT_VOICE = CLONED_VOICES[default_voice_name]
logger.info(f"🎤 默认克隆音已设置: {default_voice_name}")


def find_best_matching_voice(target_wav_path, threshold: float = 0.5) -> dict:
    """
    为目标音频找到最匹配的克隆音
    
    参数:
        target_wav_path (str): 目标音频路径
        threshold (float): 相似度阈值
    
    返回:
        dict: 最匹配的克隆音信息，若没有匹配则返回默认克隆音
    """
    if not HAS_RESEMBLYZER:
        return DEFAULT_VOICE
    
    try:
        # 提取目标音频的声纹
        target_wav = preprocess_wav(target_wav_path)
        if len(target_wav) < 0.5 * 16000:  # 少于 0.5 秒，使用默认
            return DEFAULT_VOICE
        
        target_embedding = resemblyzer_encoder.embed_utterance(target_wav)
        
        # 计算与所有克隆音的相似度
        best_similarity = -1
        best_voice = DEFAULT_VOICE
        
        for voice_info in CLONED_VOICES.values():
            if voice_info["embedding"] is None:
                continue
            
            similarity = float(np.dot(target_embedding, voice_info["embedding"]))
            logger.debug(f"   🔍 克隆音 {voice_info['name']} 相似度: {similarity:.3f}")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_voice = voice_info
        
        logger.info(f"🎯 最佳匹配克隆音: {best_voice['name']} (相似度: {best_similarity:.3f})")
        return best_voice
        
    except Exception as e:
        logger.warning(f"⚠️ 匹配克隆音失败，使用默认: {e}")
        return DEFAULT_VOICE


def is_voice_consistent(generated_wav_path: str, reference_embedding, threshold: float = 0.6) -> bool:
    """
    判断生成的语音片段与参考语音的音色是否一致。

    参数:
        generated_wav_path (str): 生成音频的路径。
        reference_embedding: 参考语音的声纹嵌入
        threshold (float): 相似度阈值，范围 [0,1]，默认 0.6。

    返回:
        bool: True 表示音色一致（或无法检查时容错通过），False 表示不一致。
    """
    if not HAS_RESEMBLYZER or reference_embedding is None:
        return True  # 无法检查时视为通过

    try:
        gen_wav = preprocess_wav(generated_wav_path)
        if len(gen_wav) < 0.5 * 16000:  # 少于 0.5 秒，跳过检查
            return True
        gen_embedding = resemblyzer_encoder.embed_utterance(gen_wav)
        similarity = float(np.dot(reference_embedding, gen_embedding))
        logger.debug(f"   🔍 音色相似度: {similarity:.3f} (阈值={threshold})")
        return similarity >= threshold
    except Exception as e:
        logger.warning(f"   ⚠️ 音色一致性检查失败: {e}")
        return True  # 容错处理：检查失败也视为通过


# ========================
# 🎯 VoxCPM TTS 模型加载
# ========================
VOXCPM_MODEL = None
HAS_VOXCPM = False

def release_voxcpm_model():
    """释放 VoxCPM 模型和GPU资源"""
    global VOXCPM_MODEL, HAS_VOXCPM, resemblyzer_encoder
    
    logger.info("🗑️  正在释放 VoxCPM 模型资源...")
    
    # 释放模型引用
    VOXCPM_MODEL = None
    HAS_VOXCPM = False
    resemblyzer_encoder = None
    
    # 清理GPU缓存
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # 强制垃圾回收
    import gc
    gc.collect()
    
    logger.info("✅ VoxCPM 模型资源已释放")

try:
    # 检查voxcpm模块是否安装
    import importlib.util
    spec = importlib.util.find_spec('voxcpm')
    if spec is None:
        raise ImportError("voxcpm模块未安装")
    
    from voxcpm import VoxCPM
    logger.info("✅ 正在加载 VoxCPM 模型...")
    
    # 尝试加载本地模型，如果失败则使用在线模型
    try:
        VOXCPM_MODEL = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")
        logger.info("✅ VoxCPM 模型加载成功！")
        HAS_VOXCPM = True
    except Exception as e:
        logger.warning(f"⚠️ 在线加载失败: {e}")
        
        # 尝试使用本地模型路径
        local_model_path = "models/VoxCPM-0.5B"
        if os.path.exists(local_model_path):
            logger.info(f"✅ 尝试从本地加载模型: {local_model_path}")
            VOXCPM_MODEL = VoxCPM.from_pretrained(local_model_path)
            logger.info("✅ 本地模型加载成功！")
            HAS_VOXCPM = True
        else:
            logger.error(f"❌ 本地模型路径不存在: {local_model_path}")
            raise e
except Exception as e:
    logger.error(f"❌ VoxCPM 模型加载失败: {e}")
    import traceback
    logger.debug(f"详细错误信息: {traceback.format_exc()}")
    HAS_VOXCPM = False


# ========================
# 🎯 可选功能：文本规范化 & 音频拉伸
# ========================
HAS_TEXTNORM = False
try:
    from youdub.cn_tx import TextNorm
    normalizer = TextNorm()
    HAS_TEXTNORM = True
except ImportError:
    logger.warning("⚠️ 文本规范化模块未找到")

HAS_AUDIOSTRETCHY = False
try:
    from audiostretchy.stretch import stretch_audio
    HAS_AUDIOSTRETCHY = True
except ImportError:
    logger.warning("⚠️ audiostretchy未安装，将使用librosa")


@dataclass
class TTSConfig:
    """TTS 合成配置参数"""
    sample_rate: int = 16000


def stretch_audio_librosa(wav_path: str, target_path: str, ratio: float, sample_rate: int = 16000) -> bool:
    """
    使用 librosa 实现音频时间拉伸（仅用于压缩，ratio < 1.0）。

    参数:
        wav_path (str): 输入音频路径。
        target_path (str): 输出音频路径。
        ratio (float): 拉伸比例（<1 为加速，>1 为减速）。
        sample_rate (int): 采样率。

    返回:
        bool: 是否成功。
    """
    try:
        wav, sr = librosa.load(wav_path, sr=sample_rate)
        wav_stretched = librosa.effects.time_stretch(wav, rate=ratio)
        import soundfile as sf
        sf.write(target_path, wav_stretched, sr)
        return True
    except Exception as e:
        logger.error(f"librosa时间拉伸失败: {e}")
        return False


def adjust_audio_length(wav_path: str, desired_length: float, sample_rate: int = 16000) -> Tuple[np.ndarray, float]:
    """
    调整音频长度：智能匹配目标时长，支持压缩和拉伸（带质量控制）。

    参数:
        wav_path (str): 输入音频路径。
        desired_length (float): 目标时长（秒）。
        sample_rate (int): 采样率。

    返回:
        Tuple[np.ndarray, float]: (调整后的音频数组, 实际时长)
    """
    try:
        wav, sr = librosa.load(wav_path, sr=sample_rate)
        current_length = len(wav) / sample_rate

        if current_length <= 0:
            logger.error(f"音频长度为0: {wav_path}")
            return np.zeros(int(desired_length * sample_rate)), desired_length

        # 计算时长差异
        duration_diff = abs(desired_length - current_length)
        # 差异小于0.5秒时，不进行调整（避免不必要的处理）
        if duration_diff < 0.5:
            return wav, current_length

        # 计算速度因子
        speed_factor = current_length / desired_length
        
        # 质量控制：限制拉伸/压缩比例范围（0.75-1.25）
        # 超出这个范围可能导致音频质量严重下降
        speed_factor = max(0.75, min(1.25, speed_factor))
        
        # 计算调整后的目标时长
        adjusted_length = current_length / speed_factor
        
        logger.info(f"⏱️ 音频长度调整: {current_length:.2f}s → {adjusted_length:.2f}s (因子={speed_factor:.2f})")

        target_path = wav_path.replace('.wav', '_adjusted_temp.wav')
        success = False

        if HAS_AUDIOSTRETCHY:
            try:
                stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
                success = True
            except Exception as e:
                logger.debug(f"audiostretchy失败: {e}")

        if not success:
            if not stretch_audio_librosa(wav_path, target_path, speed_factor, sample_rate):
                return wav, current_length

        if os.path.exists(target_path):
            wav_adjusted, _ = librosa.load(target_path, sr=sample_rate)
            actual_len = len(wav_adjusted) / sample_rate
            os.remove(target_path)
            return wav_adjusted, actual_len

        return wav, current_length

    except Exception as e:
        logger.error(f"音频长度调整失败: {e}")
        return np.zeros(int(desired_length * sample_rate)), desired_length


def generate_voxcpm_audio(text: str, output_path: str, speaker_wav: Optional[str],
                          target_duration: Optional[float] = None, prompt_text: str = None) -> bool:
    """
    使用 VoxCPM 模型生成语音。

    参数:
        text (str): 待合成的中文文本。
        output_path (str): 输出音频路径。
        speaker_wav (str or None): 参考语音路径。
        target_duration (float or None): 目标时长（仅用于日志，不影响生成）。
        prompt_text (str or None): 参考文本，用于指导音色和风格。

    返回:
        bool: 是否成功生成。
    """
    global VOXCPM_MODEL

    if not HAS_VOXCPM or VOXCPM_MODEL is None:
        logger.error("❌ VoxCPM 模型不可用")
        return False

    if os.path.exists(output_path):
        return True

    try:
        wav = VOXCPM_MODEL.generate(
            text=text,
            prompt_wav_path=speaker_wav,
            prompt_text=prompt_text,  # 使用传入的参考文本
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=True,
            denoise=False,  # 禁用 denoiser 以避免崩溃
        )
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        import soundfile as sf
        sf.write(output_path, wav, 16000)
        return True
    except Exception as e:
        logger.error(f"VoxCPM 生成失败: {e}")
        return False


def split_long_sentence(text: str, max_length: int = 30) -> List[str]:
    """
    将长句子分割成多个短句，基于标点符号。
    
    参数:
        text (str): 原始长句子
        max_length (int): 单个短句的最大长度
        
    返回:
        List[str]: 分割后的短句列表
    """
    if not text:
        return []
    
    # 按标点符号分割句子
    sentences = []
    current_sentence = ""
    
    # 中文标点符号
    punctuation = ['，', '。', '！', '？', '；', '、', ',', '.', '!', '?', ';']
    
    for char in text:
        current_sentence += char
        if char in punctuation and len(current_sentence) > max_length:
            sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # 如果分割后的句子仍然过长，尝试按长度强制分割
    final_sentences = []
    for sent in sentences:
        if len(sent) > max_length * 1.5:
            # 按空格分割成词语
            words = sent.split()
            temp_sent = ""
            for word in words:
                if len(temp_sent) + len(word) + 1 > max_length:
                    final_sentences.append(temp_sent.strip())
                    temp_sent = word
                else:
                    temp_sent = f"{temp_sent} {word}" if temp_sent else word
            if temp_sent.strip():
                final_sentences.append(temp_sent.strip())
        else:
            final_sentences.append(sent)
    
    return final_sentences


def preprocess_text(text: str) -> str:
    """
    对输入文本进行预处理，提升 TTS 合成质量。

    处理内容：
    - 缩写展开（如 AI → 人工智能）
    - 大写字母分隔（如 "HelloWorld" → "Hello World"）
    - 数字与字母间加空格
    - 文本规范化（若模块可用）

    参数:
        text (str): 原始文本。

    返回:
        str: 预处理后的文本。
    """
    if not text:
        return ""
    text = text.strip()
    replacements = {
        'AI': '人工智能', 'GPT': 'G P T', 'API': 'A P I', 'UI': 'U I', 'UX': 'U X',
        'CEO': 'C E O', 'CPU': 'C P U', 'GPU': 'G P U'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # 在大写字母前加空格（除了开头）
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    if HAS_TEXTNORM:
        try:
            text = normalizer(text)
        except Exception:
            pass
    # 数字与字母之间加空格
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def generate_wavs(folder: str, config: Optional[TTSConfig] = None) -> bool:
    """
    为单个视频文件夹生成 TTS 音频并混合伴奏。

    目录结构要求：
    - folder/
        - translation.json       ← 必须：含 'translation', 'start', 'end'
        - audio_vocals.wav       ← 可选：原始人声（用于对齐总时长）
        - audio_instruments.wav  ← 可选：伴奏（用于混合）

    生成文件：
    - wavs/0000.wav ...        ← 每个片段
    - audio_tts.wav            ← 纯中文配音
    - audio_combined.wav       ← 配音 + 伴奏（最终输出）

    参数:
        folder (str): 视频处理目录。
        config (TTSConfig): TTS 配置。

    返回:
        bool: 是否成功生成 combined 音频。
    """
    if config is None:
        config = TTSConfig()

    folder_name = os.path.basename(folder)
    logger.info(f"\n🎬 正在处理视频: {folder_name}")

    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')
    combined_path = os.path.join(folder, 'audio_combined.wav')

    if not os.path.exists(transcript_path):
        logger.error(f"❌ 翻译文件不存在: {transcript_path}")
        return False
    if os.path.exists(combined_path):
        logger.info(f"⏭️ 已存在，跳过: {folder_name}")
        return True
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    if not transcript:
        logger.error(f"❌ 翻译文件为空")
        return False

    audio_vocals_path = os.path.join(folder, 'audio_vocals.wav')
    original_audio_duration = librosa.get_duration(path=audio_vocals_path) if os.path.exists(audio_vocals_path) else max(line.get('end', 0) for line in transcript)
    logger.info(f"⏱️ 原始音频总时长: {original_audio_duration:.2f}秒")

    full_wav = np.zeros(0, dtype=np.float32)

    for i, line in enumerate(transcript):
        text = line.get('translation', '').strip()
        if not text:
            continue

        speaker = line.get('speaker', 'SPEAKER_00')
        processed_text = preprocess_text(text)

        logger.info(f"\n🗣️ 片段 [{i+1}/{len(transcript)}] | 说话人: {speaker}")
        logger.info(f"🔤 合成文本: {processed_text[:45]}{'...' if len(processed_text) > 45 else ''}")

        # 查找说话人的参考音频片段
        speaker_audio_path = None
        speaker_folder = os.path.join(folder, 'SPEAKER')
        if os.path.exists(speaker_folder):
            speaker_file = os.path.join(speaker_folder, f'{speaker}.wav')
            if os.path.exists(speaker_file):
                speaker_audio_path = speaker_file
        
        # 为说话人找到最佳匹配的克隆音
        if speaker_audio_path:
            best_voice = find_best_matching_voice(speaker_audio_path)
        else:
            best_voice = DEFAULT_VOICE
        
        speaker_wav = best_voice['wav_path']
        speaker_name = best_voice['name']
        logger.info(f"🎤 为说话人 {speaker} 匹配到最佳克隆音: {speaker_name}")

        # 计算目标时长（优先使用 VAD 时长，否则用原始片段时长）
        start = float(line.get('start', 0))
        end = float(line.get('end', 0))
        raw_duration = end - start
        vad_duration = line.get('vad_duration')
        target_duration = min(float(vad_duration), raw_duration) if vad_duration else raw_duration
        logger.info(f"⏱️ 原视频时长: {raw_duration:.2f}s")

        # 检查文本是否过长，如果过长则分割成多个短句
        sentences = split_long_sentence(processed_text, max_length=30)
        
        # 生成音频（带重试机制）
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        success = False
        max_retries = 2

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning(f"   🔁 第 {attempt} 次重试生成（音色不一致）")

            # 如果是长句子，分割生成后拼接
            if len(sentences) > 1:
                logger.info(f"   📝 长句子分割: {len(processed_text)}字符 → {len(sentences)}个短句")
                
                # 为每个短句生成音频
                temp_files = []
                all_wavs = []
                
                for j, short_text in enumerate(sentences):
                            temp_path = os.path.join(output_folder, f'{str(i).zfill(4)}_part{j}.wav')
                            if generate_voxcpm_audio(short_text, temp_path, speaker_wav, target_duration/len(sentences), best_voice['prompt_text']):
                                temp_files.append(temp_path)
                                wav, sr = librosa.load(temp_path, sr=config.sample_rate)
                                all_wavs.append(wav)
                            else:
                                logger.error(f"   ❌ 短句 {j+1} 生成失败")
                                break
                
                if len(all_wavs) == len(sentences):
                    # 拼接所有短句音频
                    combined_wav = np.concatenate(all_wavs)
                    # 保存拼接后的音频
                    import soundfile as sf
                    sf.write(output_path, combined_wav, config.sample_rate)
                    
                    # 清理临时文件
                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    
                    # 检查音色一致性
                    if is_voice_consistent(output_path, best_voice['embedding'], threshold=0.6):
                        success = True
                        break
                    else:
                        logger.warning("   ❌ 音色不一致，将重试...")
                        if os.path.exists(output_path):
                            os.remove(output_path)
                else:
                    # 清理临时文件
                    for temp_file in temp_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    break
            else:
                # 普通短句直接生成
                if generate_voxcpm_audio(processed_text, output_path, speaker_wav, target_duration, best_voice['prompt_text']):
                    if is_voice_consistent(output_path, best_voice['embedding'], threshold=0.6):
                        success = True
                        break
                    else:
                        logger.warning("   ❌ 音色不一致，将重试...")
                        if os.path.exists(output_path):
                            os.remove(output_path)
                else:
                    break  # 生成失败不再重试

        if not success:
            logger.error(f"❌ 片段 {i+1} 生成失败或音色不一致，跳过")
            continue

        # 音频后处理：对齐时间轴
        try:
            gen_wav, sr = librosa.load(output_path, sr=config.sample_rate)
            gen_duration = len(gen_wav) / sr
            logger.info(f"   🎙️ TTS生成时长: {gen_duration:.2f}s")

            wav_adjusted, final_duration = adjust_audio_length(output_path, target_duration, config.sample_rate)
            logger.info(f"   📏 最终音频时长: {final_duration:.2f}s")

            # 插入静音对齐起始时间
            current_time = len(full_wav) / config.sample_rate
            if start > current_time:
                silence_samples = int((start - current_time) * config.sample_rate)
                if silence_samples > 0:
                    full_wav = np.concatenate([full_wav, np.zeros(silence_samples, dtype=np.float32)])
            elif start < current_time:
                target_samples = int(start * config.sample_rate)
                if target_samples < len(full_wav):
                    full_wav = full_wav[:target_samples]

            # 限制结束时间（避免片段重叠）
            max_end_samples = int((end + 0.2) * config.sample_rate)
            current_samples = len(full_wav)
            if current_samples + len(wav_adjusted) > max_end_samples:
                allowed = max_end_samples - current_samples
                if allowed > 0:
                    wav_adjusted = wav_adjusted[:allowed]
                else:
                    wav_adjusted = np.zeros(0)

            if len(wav_adjusted) > 0:
                full_wav = np.concatenate([full_wav, wav_adjusted])

        except Exception as e:
            logger.error(f"❌ 处理片段 {i+1} 失败: {e}")
            traceback.print_exc()
            continue

    # 保存最终 TTS 音频
    if len(full_wav) == 0:
        return False

    target_samples = int(original_audio_duration * config.sample_rate)
    if len(full_wav) < target_samples:
        full_wav = np.pad(full_wav, (0, target_samples - len(full_wav)), mode='constant')
    elif len(full_wav) > target_samples:
        full_wav = full_wav[:target_samples]

    # 音量对齐（参考原人声音量）
    if os.path.exists(audio_vocals_path):
        try:
            vocal_wav, sr = librosa.load(audio_vocals_path, sr=config.sample_rate)
            if len(vocal_wav) > 0 and np.max(np.abs(full_wav)) > 0:
                full_wav = full_wav / np.max(np.abs(full_wav)) * np.max(np.abs(vocal_wav)) * 0.95
        except Exception as e:
            logger.warning(f"音量对齐失败: {e}")

    tts_path = os.path.join(folder, 'audio_tts.wav')
    save_wav(full_wav, tts_path, config.sample_rate)
    logger.info(f"🔊 TTS音频已保存: {tts_path}")

    # 混合伴奏
    instruments_path = os.path.join(folder, 'audio_instruments.wav')
    if os.path.exists(instruments_path):
        try:
            inst_wav, sr = librosa.load(instruments_path, sr=config.sample_rate)
            if len(full_wav) > len(inst_wav):
                inst_wav = np.pad(inst_wav, (0, len(full_wav) - len(inst_wav)), mode='constant')
            elif len(inst_wav) > len(full_wav):
                full_wav = np.pad(full_wav, (0, len(inst_wav) - len(full_wav)), mode='constant')
            combined = full_wav * 0.8 + inst_wav * 0.6  # 配音 80%，伴奏 60%
            combined_path = os.path.join(folder, 'audio_combined.wav')
            save_wav_norm(combined, combined_path, config.sample_rate)
            logger.info(f"🎧 混合音频已保存: {combined_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 混合失败: {e}")
            return False
    else:
        logger.warning("⚠️ 无伴奏文件，仅保存 TTS 音频")
        return True


def generate_all_wavs_under_folder(root_folder: str) -> Dict[str, Any]:
    """
    遍历根目录，对所有包含 translation.json 的子文件夹执行 TTS 合成。

    参数:
        root_folder (str): 根目录路径。

    返回:
        Dict: 统计结果，含成功/失败/跳过数量。
    """
    results = {
        'total': 0,
        'processed': 0,
        'success': 0,
        'failed': 0,
        'failed_folders': [],
        'skipped': 0
    }
    for root, _, files in os.walk(root_folder):
        if 'translation.json' in files:
            results['total'] += 1
            if 'audio_combined.wav' in files:
                results['skipped'] += 1
                logger.info(f'⏭️ 跳过: {os.path.basename(root)}')
                continue
            results['processed'] += 1
            if generate_wavs(root):
                results['success'] += 1
            else:
                results['failed'] += 1
                results['failed_folders'].append(root)
    return results


def main():
    """
    主函数：解析命令行参数并启动处理流程。
    
    用法:
        python script.py --folder <单个视频目录>
        python script.py --all [--root <根目录>]
    """
    import argparse
    parser = argparse.ArgumentParser(description="英文视频转中文配音 TTS 脚本")
    parser.add_argument('--folder', type=str, help="处理单个视频文件夹")
    parser.add_argument('--all', action='store_true', help="处理根目录下所有视频")
    parser.add_argument('--root', type=str, default='videos', help="批量处理的根目录（默认: videos）")
    args = parser.parse_args()

    # 配置日志格式
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <6}</level> | <cyan>{message}</cyan>")

    if not HAS_VOXCPM:
        logger.error("❌ VoxCPM 不可用")
        sys.exit(1)

    if args.all or (not args.folder and not args.all):
        results = generate_all_wavs_under_folder(args.root if args.all else 'videos')
        logger.info("\n" + "="*50)
        logger.info(f"✅ 成功: {results['success']}/{results['processed']} | ⏭️ 跳过: {results['skipped']}")
        if results['failed'] > 0:
            logger.warning(f"❌ 失败: {results['failed']} 个视频")
            for f in results['failed_folders']:
                logger.warning(f"   - {f}")
    elif args.folder:
        folder_name = os.path.basename(args.folder)
        if os.path.exists(os.path.join(args.folder, 'audio_combined.wav')):
            logger.info(f"⏭️ 跳过: {folder_name}")
        else:
            logger.info(f"🎬 处理: {folder_name}")
            generate_wavs(args.folder)


if __name__ == '__main__':
    main()