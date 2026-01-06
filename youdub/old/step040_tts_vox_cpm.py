#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音合成与音频处理脚本 - VAD 时长感知版（VoxCPM 集成）✅ 最终版
核心特性：
  - ✅ 英文音色 + 中文语音合成（跨语言克隆）
  - ✅ 自动优先使用 SPEAKER_XX_CLONE.wav / .txt
  - ✅ 自动跳过已存在 audio_combined.wav 的视频
  - ✅ 修复 VoxCPM prompt_wav + prompt_text 语言一致性
  - ✅ 无参考时降级为默认声音
  - ✅ 保留 VAD 时长感知 + 音频拉伸
  - ✅ 中文文本预处理 + 伴奏混合

作者: Advanced TTS Team
日期: 2026-01-04
版本: 1.8
"""

import json
import os
import re
import librosa
import sys
import traceback
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块（带错误处理）
try:
    from youdub.utils import save_wav, save_wav_norm
except ImportError:
    logger.warning("youdub.utils 模块导入失败，将使用本地实现")
    import scipy.io.wavfile

    def save_wav(wav: np.ndarray, path: str, sample_rate: int = 16000):
        wav = np.clip(wav, -1.0, 1.0)
        scipy.io.wavfile.write(path, sample_rate, (wav * 32767).astype(np.int16))

    def save_wav_norm(wav: np.ndarray, path: str, sample_rate: int = 16000):
        if len(wav) > 0:
            wav = wav / np.max(np.abs(wav)) * 0.95
        save_wav(wav, path, sample_rate)


# ========================
# 🎯 VoxCPM 模型加载
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


# 文本规范化
try:
    from youdub.cn_tx import TextNorm
    normalizer = TextNorm()
    HAS_TEXTNORM = True
except ImportError:
    HAS_TEXTNORM = False
    logger.warning("⚠️ 文本规范化模块未找到")


# 音频拉伸
try:
    from audiostretchy.stretch import stretch_audio
    HAS_AUDIOSTRETCHY = True
except ImportError:
    HAS_AUDIOSTRETCHY = False
    logger.warning("⚠️ audiostretchy未安装，将使用librosa")


@dataclass
class TTSConfig:
    sample_rate: int = 16000
    use_voxcpm: bool = True
    min_speed_factor: float = 0.95
    max_speed_factor: float = 1.05


def preprocess_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    
    text = text.strip()
    
    replacements = {
        'AI': '人工智能',
        'GPT': 'G P T',
        'API': 'A P I',
        'UI': 'U I',
        'UX': 'U X',
        'CEO': 'C E O',
        'CPU': 'C P U',
        'GPU': 'G P U',
    }
    
    for key, value in replacements.items():
        text = text.replace(key, value)
    
    text = re.sub(r'(?<!^)([A-Z])', r' \1', text)
    
    if HAS_TEXTNORM:
        try:
            text = normalizer(text)
        except Exception:
            logger.warning("文本规范化失败")
    
    text = re.sub(r'(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def stretch_audio_librosa(wav_path: str, target_path: str, ratio: float, sample_rate: int = 16000):
    try:
        wav, sr = librosa.load(wav_path, sr=sample_rate)
        wav_stretched = librosa.effects.time_stretch(wav, rate=ratio)
        import soundfile as sf
        sf.write(target_path, wav_stretched, sr)
        return True
    except Exception as e:
        logger.error(f"librosa时间拉伸失败: {e}")
        return False


def adjust_audio_length(wav_path: str, desired_length: float,
                        sample_rate: int = 16000,
                        min_speed_factor: float = 0.95,
                        max_speed_factor: float = 1.05) -> Tuple[np.ndarray, float]:
    try:
        wav, sr = librosa.load(wav_path, sr=sample_rate)
        current_length = len(wav) / sample_rate

        if current_length <= 0:
            logger.error(f"音频长度为0: {wav_path}")
            return np.zeros(int(desired_length * sample_rate)), desired_length

        speed_factor = max(
            min(desired_length / current_length, max_speed_factor),
            min_speed_factor
        )

        logger.debug(f"音频长度调整: {current_length:.2f}s -> {desired_length:.2f}s, 因子: {speed_factor:.3f}")

        target_path = wav_path.replace('.wav', f'_adjusted.wav')

        if HAS_AUDIOSTRETCHY:
            try:
                stretch_audio(wav_path, target_path, ratio=speed_factor, sample_rate=sample_rate)
            except Exception as e:
                logger.warning(f"audiostretchy失败，使用librosa: {e}")
                if not stretch_audio_librosa(wav_path, target_path, speed_factor, sample_rate):
                    target_path = wav_path
        else:
            if not stretch_audio_librosa(wav_path, target_path, speed_factor, sample_rate):
                target_path = wav_path

        wav_adjusted, sr = librosa.load(target_path, sr=sample_rate)

        if target_path != wav_path and os.path.exists(target_path):
            try:
                os.remove(target_path)
            except:
                pass

        return wav_adjusted, current_length * speed_factor

    except Exception as e:
        logger.error(f"音频长度调整失败: {e}")
        return np.zeros(int(desired_length * sample_rate)), desired_length


def generate_voxcpm_audio(text: str, output_path: str, speaker_wav: Optional[str],
                          target_duration: Optional[float] = None) -> bool:
    global VOXCPM_MODEL

    if not HAS_VOXCPM or VOXCPM_MODEL is None:
        logger.error("❌ VoxCPM 模型不可用")
        return False

    if os.path.exists(output_path):
        logger.info(f"✅ 音频已存在: {output_path}")
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
            logger.warning(f"⚠️ 有参考音频但无有效文本，禁用语音克隆: {speaker_wav}")
            speaker_wav = None
            prompt_text = None
    else:
        speaker_wav = None
        prompt_text = None

    try:
        wav = VOXCPM_MODEL.generate(
            text=text,                    # ← 中文（目标语言）
            prompt_wav_path=speaker_wav,  # ← 英文音频（音色来源）
            prompt_text=prompt_text,      # ← 英文原文（必须匹配音频）
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
        logger.error(f"❌ VoxCPM 生成失败: {e}")
        logger.error(traceback.format_exc())
        return False


def generate_tts_audio(text: str, output_path: str, speaker_wav: Optional[str],
                       engine: str, config: TTSConfig, target_duration: float = None) -> bool:
    return generate_voxcpm_audio(text, output_path, speaker_wav, target_duration)


def choose_tts_engine(num_speakers: int, config: TTSConfig) -> str:
    if HAS_VOXCPM:
        return 'voxcpm'
    logger.error("❌ 没有可用的TTS引擎")
    return 'none'


def generate_wavs(folder: str, config: Optional[TTSConfig] = None) -> bool:
    if config is None:
        config = TTSConfig()

    transcript_path = os.path.join(folder, 'translation.json')
    output_folder = os.path.join(folder, 'wavs')

    if not os.path.exists(transcript_path):
        logger.error(f"❌ 翻译文件不存在: {transcript_path}")
        return False

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # === 跳过已存在最终输出的视频 ===
    combined_path = os.path.join(folder, 'audio_combined.wav')
    if os.path.exists(combined_path):
        logger.info(f"⏭️ 最终音频已存在，跳过: {folder}")
        return True

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)

        if not transcript:
            logger.error(f"❌ 翻译文件为空: {transcript_path}")
            return False

        audio_vocals_path = os.path.join(folder, 'audio_vocals.wav')
        if os.path.exists(audio_vocals_path):
            original_audio_duration = librosa.get_duration(path=audio_vocals_path)
        else:
            original_audio_duration = max(line.get('end', 0) for line in transcript)
        logger.info(f"⏱️ 原始音频总时长: {original_audio_duration:.2f}秒")

        speakers = {line.get('speaker', 'SPEAKER_00') for line in transcript}
        num_speakers = len(speakers)
        logger.info(f'👥 发现 {num_speakers} 个说话人: {sorted(speakers)}')

        engine = choose_tts_engine(num_speakers, config)
        if engine == 'none':
            return False
        logger.info(f'🤖 使用TTS引擎: {engine}')

        full_wav = np.zeros(0, dtype=np.float32)

        for i, line in enumerate(transcript):
            speaker = line.get('speaker', 'SPEAKER_00')
            original_text = line.get('translation', '').strip()  # ← 中文！

            if not original_text:
                logger.warning(f"⚠️ 第{i}行文本为空，跳过")
                continue

            text = preprocess_text(original_text)
            logger.debug(f"🔤 处理片段 {i}: {text[:50]}...")

            output_path = os.path.join(output_folder, f'{str(i).zfill(4)}.wav')

            # === 查找参考音频（优先 _CLONE.wav）===
            speaker_wav = None
            speaker_dir = os.path.join(folder, 'SPEAKER')
            if os.path.exists(speaker_dir):
                candidates = [
                    os.path.join(speaker_dir, f'{speaker}_CLONE.wav'),
                    os.path.join(speaker_dir, f'{speaker}.wav')
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        speaker_wav = cand
                        break

            original_start = float(line.get('start', 0))
            original_end = float(line.get('end', 0))
            raw_duration = original_end - original_start
            vad_duration = line.get('vad_duration')

            if vad_duration is not None:
                target_duration = min(float(vad_duration), raw_duration)
                logger.debug(f"🎯 片段 {i}: VAD 时长 = {target_duration:.2f}s (原始 {raw_duration:.2f}s)")
            else:
                target_duration = raw_duration
                logger.debug(f"🎯 片段 {i}: 原始时长 = {target_duration:.2f}s (无 VAD 数据)")

            success = generate_tts_audio(
                text,
                output_path,
                speaker_wav,
                engine,
                config,
                target_duration=target_duration
            )

            if not success:
                logger.error(f"❌ 片段 {i} TTS生成失败")
                silence_duration = line.get('end', 0) - line.get('start', 0)
                if silence_duration > 0:
                    silence_samples = int(silence_duration * config.sample_rate)
                    silence_wav = np.zeros(silence_samples, dtype=np.float32)
                    save_wav(silence_wav, output_path, config.sample_rate)
                else:
                    continue

            current_time = len(full_wav) / config.sample_rate
            if original_start > current_time:
                silence_samples = int((original_start - current_time) * config.sample_rate)
                if silence_samples > 0:
                    full_wav = np.concatenate((full_wav, np.zeros(silence_samples, dtype=np.float32)))
            elif original_start < current_time:
                target_samples = int(original_start * config.sample_rate)
                if target_samples < len(full_wav):
                    full_wav = full_wav[:target_samples]
                current_time = original_start

            wav_adjusted, _ = adjust_audio_length(
                output_path,
                target_duration,
                sample_rate=config.sample_rate,
                min_speed_factor=config.min_speed_factor,
                max_speed_factor=config.max_speed_factor
            )

            overlap_tolerance = 0.15
            max_allowed_samples = int((original_end + overlap_tolerance) * config.sample_rate)
            current_samples = len(full_wav)
            if current_samples + len(wav_adjusted) > max_allowed_samples:
                allowed_length = max_allowed_samples - current_samples
                if allowed_length > 0:
                    wav_adjusted = wav_adjusted[:allowed_length]
                else:
                    wav_adjusted = np.zeros(0)

            if len(wav_adjusted) > 0:
                full_wav = np.concatenate((full_wav, wav_adjusted))
            else:
                logger.warning(f"⚠️ 片段 {i} 音频为空，跳过")

        if len(full_wav) == 0:
            logger.error("❌ 没有生成任何音频")
            return False

        target_final_samples = int(original_audio_duration * config.sample_rate)
        if len(full_wav) < target_final_samples:
            full_wav = np.pad(full_wav, (0, target_final_samples - len(full_wav)), mode='constant')
        elif len(full_wav) > target_final_samples:
            full_wav = full_wav[:target_final_samples]

        if os.path.exists(audio_vocals_path):
            try:
                vocal_wav, sr = librosa.load(audio_vocals_path, sr=config.sample_rate)
                if len(vocal_wav) > 0:
                    vocal_max = np.max(np.abs(vocal_wav))
                    if vocal_max > 0 and np.max(np.abs(full_wav)) > 0:
                        full_wav = full_wav / np.max(np.abs(full_wav)) * vocal_max * 0.95
            except Exception as e:
                logger.warning(f"音量参考失败: {e}")

        tts_output_path = os.path.join(folder, 'audio_tts.wav')
        save_wav(full_wav, tts_output_path, config.sample_rate)
        logger.info(f"🔊 TTS音频已保存: {tts_output_path}")

        instruments_path = os.path.join(folder, 'audio_instruments.wav')
        if os.path.exists(instruments_path):
            try:
                instruments_wav, sr = librosa.load(instruments_path, sr=config.sample_rate)
                if len(full_wav) > len(instruments_wav):
                    instruments_wav = np.pad(instruments_wav, (0, len(full_wav) - len(instruments_wav)), mode='constant')
                elif len(instruments_wav) > len(full_wav):
                    full_wav = np.pad(full_wav, (0, len(instruments_wav) - len(full_wav)), mode='constant')
                combined_wav = full_wav * 0.8 + instruments_wav * 0.6
                combined_output_path = os.path.join(folder, 'audio_combined.wav')
                save_wav_norm(combined_wav, combined_output_path, config.sample_rate)
                logger.info(f"🎧 混合音频已保存: {combined_output_path}")
            except Exception as e:
                logger.error(f"❌ 音频混合失败: {e}")
                return False
        else:
            logger.warning(f"⚠️ 伴奏文件不存在: {instruments_path}")

        return True

    except Exception as e:
        logger.error(f"💥 音频生成失败: {e}")
        logger.error(traceback.format_exc())
        return False


def generate_all_wavs_under_folder(root_folder: str,
                                   config: Optional[TTSConfig] = None,
                                   skip_existing: bool = True) -> Dict[str, Any]:
    if config is None:
        config = TTSConfig()

    results = {
        'total_folders': 0,
        'processed': 0,
        'success': 0,
        'failed': 0,
        'failed_folders': [],
        'skipped': 0
    }

    for root, dirs, files in os.walk(root_folder):
        if 'translation.json' in files:
            results['total_folders'] += 1

            if 'audio_combined.wav' in files:
                logger.info(f'⏭️ 跳过已处理: {root}')
                results['skipped'] += 1
                continue

            logger.info(f'📁 处理: {root}')
            results['processed'] += 1

            success = generate_wavs(root, config)
            if success:
                results['success'] += 1
                logger.info(f'✅ 完成: {root}')
            else:
                results['failed'] += 1
                results['failed_folders'].append(root)
                logger.error(f'❌ 失败: {root}')

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='使用 VoxCPM 的语音合成脚本（✅ 跨语言克隆版）')
    parser.add_argument('--folder', type=str, help='处理单个文件夹')
    parser.add_argument('--all', action='store_true', help='批量处理')
    parser.add_argument('--root', type=str, default='videos', help='根目录')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已存在')

    args = parser.parse_args()

    config = TTSConfig()

    logger.remove()
    logger.add(sys.stdout, level="INFO",
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>")

    if not HAS_VOXCPM:
        logger.error("❌ VoxCPM 模型不可用，退出")
        sys.exit(1)

    if args.all or (not args.folder and not args.all):
        root_dir = args.root if args.all else 'videos'
        logger.info(f"🔄 批量处理所有待处理视频（根目录: {root_dir})")
        results = generate_all_wavs_under_folder(root_dir, config)

        logger.info("\n" + "=" * 50)
        logger.info("📊 处理完成！")
        logger.info(f"总计: {results['total_folders']}")
        logger.info(f"处理: {results['processed']}")
        logger.info(f"成功: {results['success']}")
        logger.info(f"失败: {results['failed']}")
        logger.info(f"跳过: {results['skipped']}")

        if results['failed'] > 0:
            logger.warning(f"失败列表: {results['failed_folders']}")
    elif args.folder:
        combined_path = os.path.join(args.folder, 'audio_combined.wav')
        if os.path.exists(combined_path):
            logger.info(f"⏭️ 单个文件夹已处理，跳过: {args.folder}")
            return
        logger.info(f"🎬 处理单个: {args.folder}")
        success = generate_wavs(args.folder, config)
        if success:
            logger.info("🎉 完成！")
        else:
            logger.error("💥 失败！")
    else:
        logger.error("❌ 请指定 --folder 或使用默认批量模式")


if __name__ == '__main__':
    main()
