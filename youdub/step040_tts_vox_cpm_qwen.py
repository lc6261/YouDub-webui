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
# 🎯 全局默认语音克隆配置
# ========================
DEFAULT_VOICE_WAV = os.path.join("voice", "lkw_cloned.wav")
DEFAULT_VOICE_TXT = os.path.join("voice", "lkw_cloned.txt")

# 校验全局参考音频和文本是否存在
if not os.path.exists(DEFAULT_VOICE_WAV):
    logger.error(f"❌ 全局克隆音频文件缺失: {DEFAULT_VOICE_WAV}")
    sys.exit(1)
if not os.path.exists(DEFAULT_VOICE_TXT):
    logger.error(f"❌ 全局克隆文本文件缺失: {DEFAULT_VOICE_TXT}")
    sys.exit(1)

with open(DEFAULT_VOICE_TXT, 'r', encoding='utf-8') as f:
    GLOBAL_PROMPT_TEXT = f.read().strip()

if not GLOBAL_PROMPT_TEXT:
    logger.error(f"❌ 全局克隆文本为空: {DEFAULT_VOICE_TXT}")
    sys.exit(1)

logger.info(f"🎤 全局默认语音克隆已启用: {DEFAULT_VOICE_WAV}")


# ========================
# 🎯 Resemblyzer 音色一致性检查模块
# ========================
HAS_RESEMBLYZER = False
resemblyzer_encoder = None
GLOBAL_REFERENCE_EMBEDDING = None

try:
    from resemblyzer import VoiceEncoder
    from resemblyzer.audio import preprocess_wav

    resemblyzer_encoder = VoiceEncoder("cpu")  # 可改为 "cuda" 启用 GPU
    ref_wav = preprocess_wav(DEFAULT_VOICE_WAV)
    GLOBAL_REFERENCE_EMBEDDING = resemblyzer_encoder.embed_utterance(ref_wav)
    HAS_RESEMBLYZER = True
    logger.info("✅ 声纹一致性检查已启用（使用 Resemblyzer）")
except Exception as e:
    logger.warning(f"⚠️ Resemblyzer 加载失败（将跳过音色检查）: {e}")
    HAS_RESEMBLYZER = False


def is_voice_consistent(generated_wav_path: str, threshold: float = 0.6) -> bool:
    """
    判断生成的语音片段与全局参考语音的音色是否一致。

    参数:
        generated_wav_path (str): 生成音频的路径。
        threshold (float): 相似度阈值，范围 [0,1]，默认 0.6。

    返回:
        bool: True 表示音色一致（或无法检查时容错通过），False 表示不一致。
    """
    if not HAS_RESEMBLYZER or GLOBAL_REFERENCE_EMBEDDING is None:
        return True  # 无法检查时视为通过

    try:
        gen_wav = preprocess_wav(generated_wav_path)
        if len(gen_wav) < 0.5 * 16000:  # 少于 0.5 秒，跳过检查
            return True
        gen_embedding = resemblyzer_encoder.embed_utterance(gen_wav)
        similarity = float(np.dot(GLOBAL_REFERENCE_EMBEDDING, gen_embedding))
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

try:
    from voxcpm import VoxCPM
    logger.info("✅ 正在加载 VoxCPM 模型...")
    VOXCPM_MODEL = VoxCPM.from_pretrained("openbmb/VoxCPM-0.5B")
    HAS_VOXCPM = True
    logger.info("✅ VoxCPM 模型加载成功！")
except Exception as e:
    logger.error(f"❌ VoxCPM 模型加载失败: {e}")
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
    调整音频长度：仅在 TTS 音频超长时进行压缩（绝不拉伸！）。

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

        if current_length <= desired_length:
            return wav, current_length

        # 限制最小压缩比例为 0.85（避免过度失真）
        speed_factor = max(desired_length / current_length, 0.85)
        logger.warning(f"⚠️ 超时压缩: {current_length:.2f}s → {desired_length:.2f}s (因子={speed_factor:.2f})")

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
                          target_duration: Optional[float] = None) -> bool:
    """
    使用 VoxCPM 模型生成语音。

    参数:
        text (str): 待合成的中文文本。
        output_path (str): 输出音频路径。
        speaker_wav (str or None): 参考语音路径。
        target_duration (float or None): 目标时长（仅用于日志，不影响生成）。

    返回:
        bool: 是否成功生成。
    """
    global VOXCPM_MODEL

    if not HAS_VOXCPM or VOXCPM_MODEL is None:
        logger.error("❌ VoxCPM 模型不可用")
        return False

    if os.path.exists(output_path):
        return True

    logger.debug(f"🗣️ VoxCPM 生成: \"{text[:50]}...\" (目标时长: {target_duration:.2f}s)")

    # === 关键：安全加载参考文本（必须与音频语言一致）===
    prompt_text = None
    if speaker_wav and os.path.exists(speaker_wav):
        base = os.path.splitext(speaker_wav)[0]
        # 优先使用 _CLONE.txt（标准输出）
        candidate_txts = [
            base + "_CLONE.txt",
            base + ".txt"
        ]
        for txt_candidate in candidate_txts:
            if os.path.exists(txt_candidate) and os.path.getsize(txt_candidate) > 0:
                try:
                    with open(txt_candidate, 'r', encoding='utf-8') as f:
                        prompt_text = f.read().strip()
                    if prompt_text:
                        logger.debug(f"📜 使用参考文本: {prompt_text[:40]}...")
                        break  # 找到就停止
                except Exception as e:
                    logger.warning(f"⚠️ 参考文本加载失败 ({txt_candidate}): {e}")
        
        if not prompt_text:
            logger.warning(f"⚠️ 有参考音频但无有效文本，将使用全局默认文本: {speaker_wav}")
            # 使用全局默认文本作为备用方案
            prompt_text = GLOBAL_PROMPT_TEXT
            logger.info(f"📜 回退到全局默认文本")
    else:
        speaker_wav = None
        prompt_text = None

    try:
        wav = VOXCPM_MODEL.generate(
            text=text,                    # ← 中文（目标语言）
            prompt_wav_path=speaker_wav,  # ← 参考音频（音色来源）
            prompt_text=prompt_text,      # ← 参考文本（与音频匹配）
            cfg_value=2.0,
            inference_timesteps=10,
            normalize=True,
            denoise=True,
            retry_badcase=True,
            retry_badcase_max_times=3,
            retry_badcase_ratio_threshold=6.0,
        )
        if isinstance(wav, list):
            wav = np.array(wav, dtype=np.float32)
        elif not isinstance(wav, np.ndarray):
            wav = np.array(wav)
        import soundfile as sf
        sf.write(output_path, wav, 16000)
        return True
    except Exception as e:
        logger.error(f"VoxCPM 生成失败: {e}")
        return False


def unload_voxcpm_model():
    """
    卸载 VoxCPM 和 Resemblyzer 模型，释放内存和显存资源。
    """
    global VOXCPM_MODEL, resemblyzer_encoder, GLOBAL_REFERENCE_EMBEDDING
    import gc
    import torch
    
    logger.info("✅ 正在卸载 VoxCPM 相关资源...")
    
    # 卸载 VoxCPM 模型
    if VOXCPM_MODEL is not None:
        logger.info("   🗣️ 卸载 VoxCPM 模型...")
        # 移到CPU释放GPU资源
        if hasattr(VOXCPM_MODEL, 'to'):
            VOXCPM_MODEL.to('cpu')
        del VOXCPM_MODEL
        VOXCPM_MODEL = None
    
    # 卸载 Resemblyzer 编码器
    if resemblyzer_encoder is not None:
        logger.info("   🎤 卸载 Resemblyzer 编码器...")
        del resemblyzer_encoder
        resemblyzer_encoder = None
    
    # 释放全局参考嵌入
    if GLOBAL_REFERENCE_EMBEDDING is not None:
        logger.info("   📊 释放全局参考嵌入...")
        del GLOBAL_REFERENCE_EMBEDDING
        GLOBAL_REFERENCE_EMBEDDING = None
    
    # 清理PyTorch缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 强制垃圾回收
    gc.collect()
    
    logger.info("✅ VoxCPM 相关资源已全部卸载")


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

        # 强制使用全局默认语音克隆
        speaker_wav = DEFAULT_VOICE_WAV
        logger.info("🎤 使用全局默认音色: lkw_cloned")

        # 计算目标时长（优先使用 VAD 时长，否则用原始片段时长）
        start = float(line.get('start', 0))
        end = float(line.get('end', 0))
        raw_duration = end - start
        vad_duration = line.get('vad_duration')
        target_duration = min(float(vad_duration), raw_duration) if vad_duration else raw_duration
        logger.info(f"⏱️ 原视频时长: {raw_duration:.2f}s")

        # 生成音频（带重试机制）
        output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')
        success = False
        max_retries = 2

        for attempt in range(max_retries + 1):
            if attempt > 0:
                logger.warning(f"   🔁 第 {attempt} 次重试生成（音色不一致）")

            if generate_voxcpm_audio(processed_text, output_path, speaker_wav, target_duration):
                if is_voice_consistent(output_path, threshold=0.6):
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