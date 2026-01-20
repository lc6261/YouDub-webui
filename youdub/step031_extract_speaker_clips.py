#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立工具：从已有的翻译和人声音频中，自动提取说话人克隆参考片段
生成: SPEAKER/SPEAKER_XX_CLONE.wav + .txt

要求:
- videos/ 下每个视频文件夹包含:
  - translation.json
  - audio_vocals.wav

输出:
- SPEAKER/SPEAKER_00_CLONE.wav
- SPEAKER/SPEAKER_00_CLONE.txt
- ...

作者: Advanced TTS Team
日期: 2026-01-04
"""

import os
import json
import argparse
from typing import List, Dict
import sys

try:
    import librosa
    import numpy as np
    import soundfile as sf
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    print("❌ 错误: 请安装依赖: pip install librosa soundfile")
    sys.exit(1)

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <6}</level> | <cyan>{message}</cyan>"
)


def extract_speaker_clips_for_folder(folder: str, max_duration: float = 60.0):
    """
    为单个视频文件夹提取克隆音频
    """
    transcript_path = os.path.join(folder, 'translation.json')
    vocals_path = os.path.join(folder, 'audio_vocals.wav')
    speaker_dir = os.path.join(folder, 'SPEAKER')

    if not os.path.exists(transcript_path):
        logger.warning(f"⚠️ 跳过 {folder}: 缺少 translation.json")
        return False
    if not os.path.exists(vocals_path):
        logger.warning(f"⚠️ 跳过 {folder}: 缺少 audio_vocals.wav")
        return False

    # 创建 SPEAKER 目录
    os.makedirs(speaker_dir, exist_ok=True)

    # 加载人声音频 (16kHz)
    try:
        vocals, sr = librosa.load(vocals_path, sr=16000)
    except Exception as e:
        logger.error(f"❌ 无法加载 {vocals_path}: {e}")
        return False

    # 加载翻译
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = json.load(f)
    except Exception as e:
        logger.error(f"❌ 无法加载 {transcript_path}: {e}")
        return False

    # 按说话人分组，选最长且 <= max_duration 的非空片段
    speaker_best = {}
    for line in transcript:
        speaker = line.get('speaker', 'SPEAKER_00')
        start = float(line.get('start', 0))
        end = float(line.get('end', 0))
        text = line.get('text', '').strip()
        duration = end - start

        if not text or duration < 0.8 or duration > max_duration:
            continue  # 忽略太短、太长或空文本

        if speaker not in speaker_best or duration > speaker_best[speaker]['duration']:
            speaker_best[speaker] = {
                'start': start,
                'end': end,
                'text': text,
                'duration': duration
            }

    # 保存每个说话人的最佳片段
    extracted = 0
    for speaker, seg in speaker_best.items():
        start_samp = int(seg['start'] * sr)
        end_samp = int(seg['end'] * sr)
        clip = vocals[start_samp:end_samp]

        if len(clip) == 0:
            continue

        wav_path = os.path.join(speaker_dir, f"{speaker}_CLONE.wav")
        txt_path = os.path.join(speaker_dir, f"{speaker}_CLONE.txt")

        try:
            sf.write(wav_path, clip, sr)
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(seg['text'])
            logger.info(f"🔊 {wav_path} ({seg['duration']:.1f}s)")
            extracted += 1
        except Exception as e:
            logger.error(f"❌ 保存失败 {speaker}: {e}")

    if extracted == 0:
        logger.warning(f"⚠️ 未提取任何克隆音频: {folder}")
        return False

    logger.success(f"✅ 成功提取 {extracted} 个说话人克隆片段: {folder}")
    return True


def main():
    parser = argparse.ArgumentParser(description="从已有翻译中提取说话人克隆音频")
    parser.add_argument('--folder', type=str, help='处理单个视频文件夹')
    parser.add_argument('--all', action='store_true', help='处理 videos/ 下所有视频')
    parser.add_argument('--root', type=str, default='videos', help='根目录 (默认: videos)')
    parser.add_argument('--max-duration', type=float, default=60.0, help='最大片段时长 (秒, 默认 60.0)')

    args = parser.parse_args()

    if args.folder:
        extract_speaker_clips_for_folder(args.folder, args.max_duration)
    elif args.all:
        root = args.root
        folders = []
        for item in os.listdir(root):
            folder_path = os.path.join(root, item)
            if os.path.isdir(folder_path) and os.path.exists(os.path.join(folder_path, 'translation.json')):
                folders.append(folder_path)
        logger.info(f"🎯 发现 {len(folders)} 个视频文件夹")
        success = 0
        for folder in folders:
            if extract_speaker_clips_for_folder(folder, args.max_duration):
                success += 1
        logger.info(f"🏁 完成! 成功处理 {success}/{len(folders)} 个文件夹")
    else:
        logger.error("❌ 请指定 --folder 或 --all")
        sys.exit(1)


if __name__ == '__main__':
    main()