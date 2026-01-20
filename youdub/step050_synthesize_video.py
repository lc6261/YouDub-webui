# -*- coding: utf-8 -*-
"""
视频合成与字幕嵌入工具

本脚本用于将原始视频、合成语音和翻译文本自动合成为一个带内嵌字幕的最终视频。
主要功能包括：
1. 将长翻译文本按中文标点智能切分为短句（用于字幕分段）
2. 生成 SRT 字幕文件
3. 使用 FFmpeg 将字幕、加速后的视频、加速后的音频合成为最终 MP4 视频
4. 支持分辨率适配（如 1080p）、字幕样式自适应、播放速度调整

依赖：
- Python 3.7+
- 第三方库：loguru
- 系统工具：FFmpeg, ffprobe（需在 PATH 中）

作者：[你的名字]
最后更新：2026-01-02
"""
import sys
import json
import os
import subprocess
import time
from loguru import logger


def split_text(input_data, punctuations=['，', '；', '：', '。', '？', '！', '\n', '”']):
    """
    将输入的翻译文本按中文标点符号切分为多个短句，并为每句分配对应的时间区间。

    切分规则：
    - 遇到指定标点时尝试切分
    - 避免过短句子（<5 字符，除非是最后一句）
    - 避免连续标点导致的空句（如 "。！"）

    参数:
        input_data (list): 包含字典的列表，每个字典包含：
            - "start": 起始时间（秒）
            - "end": 结束时间（秒）
            - "text": 原始英文/原文文本（用于调试或对齐）
            - "translation": 中文翻译文本
            - "speaker" (可选): 说话人标识
        punctuations (list): 用于切分的标点符号列表

    返回:
        list: 切分后的字幕项列表，每项包含：
            - "start", "end": 新的时间区间
            - "text": 原始文本（不变）
            - "translation": 切分后的短句
            - "speaker": 说话人
    """
    def is_punctuation(char):
        """判断字符是否为指定的中文标点"""
        return char in punctuations

    output_data = []
    for item in input_data:
        start = item["start"]
        text = item["translation"]
        speaker = item.get("speaker", "SPEAKER_00")
        original_text = item["text"]
        sentence_start = 0

        # 若文本为空，跳过
        if not text:
            continue

        # 假设字符均匀分布，计算每个字符的持续时间
        duration_per_char = (item["end"] - item["start"]) / len(text)

        for i, char in enumerate(text):
            # 不是标点且不是最后一个字符 → 继续
            if not is_punctuation(char) and i != len(text) - 1:
                continue

            # 避免过短句子（少于5字），除非是最后一句
            if i - sentence_start < 5 and i != len(text) - 1:
                continue

            # 避免在连续标点处分割（如 "！？"），跳过后一个标点
            if i < len(text) - 1 and is_punctuation(text[i + 1]):
                continue

            # 提取当前句子
            sentence = text[sentence_start:i + 1]
            sentence_end = start + duration_per_char * len(sentence)

            # 保存分段结果
            output_data.append({
                "start": round(start, 3),
                "end": round(sentence_end, 3),
                "text": original_text,
                "translation": sentence,
                "speaker": speaker
            })

            # 更新下一句的起始时间与字符位置
            start = sentence_end
            sentence_start = i + 1

    return output_data


def format_timestamp(seconds):
    """
    将秒数转换为 SRT 字幕标准时间格式。

    示例: 3661.123 → "01:01:01,123"

    参数:
        seconds (float): 时间（秒）

    返回:
        str: SRT 时间戳字符串
    """
    millisec = int((seconds - int(seconds)) * 1000)
    hours, seconds = divmod(int(seconds), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"


def generate_srt(translation, srt_path, speed_up=1, max_line_char=30):
    """
    生成 SRT 字幕文件。

    功能：
    - 对翻译文本进行切分（调用 split_text）
    - 根据播放速度（speed_up）调整时间戳
    - 自动换行（每行不超过 max_line_char 字符）

    参数:
        translation (list): 原始翻译数据（未切分）
        srt_path (str): 输出 SRT 文件路径
        speed_up (float): 播放加速倍数（>1 表示加速，时间戳需除以此值）
        max_line_char (int): 每行最大字符数（用于自动换行）
    """
    translation = split_text(translation)
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(translation):
            # 应用速度调整：原始时间 / speed_up
            start = format_timestamp(line['start'] / speed_up)
            end = format_timestamp(line['end'] / speed_up)
            text = line['translation']

            # 自动换行：尽量均分字符到多行，每行不超过 max_line_char
            line_count = len(text) // (max_line_char + 1) + 1
            avg_chars_per_line = min(round(len(text) / line_count), max_line_char)
            wrapped_text = '\n'.join([
                text[j * avg_chars_per_line:(j + 1) * avg_chars_per_line]
                for j in range(line_count)
            ])

            # 写入 SRT 格式
            f.write(f'{i + 1}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{wrapped_text}\n\n')


def get_aspect_ratio(video_path):
    """
    使用 ffprobe 获取视频的宽高比（width / height）。

    参数:
        video_path (str): 视频文件路径

    返回:
        float: 宽高比（如 16/9 ≈ 1.777）
    """
    command = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    dimensions = json.loads(result.stdout)['streams'][0]
    return dimensions['width'] / dimensions['height']


def convert_resolution(aspect_ratio, resolution='1080p'):
    """
    根据原始视频宽高比和目标分辨率（如 '1080p'），计算目标宽高。

    规则：
    - 若视频为竖屏（aspect_ratio < 1），以 width=1080 为基准
    - 否则以 height=1080 为基准
    - 宽高需为偶数（H.264 编码要求）

    参数:
        aspect_ratio (float): 原始宽高比
        resolution (str): 目标分辨率，如 '1080p', '720p'

    返回:
        tuple: (width, height)，均为偶数
    """
    target_size = int(resolution[:-1])  # '1080p' → 1080
    if aspect_ratio < 1:
        # 竖屏：固定宽度
        width = target_size
        height = int(width / aspect_ratio)
    else:
        # 横屏：固定高度
        height = target_size
        width = int(height * aspect_ratio)

    # 确保宽高为偶数（FFmpeg x264 要求）
    width = width - (width % 2)
    height = height - (height % 2)
    return width, height


def synthesize_video(folder, subtitles=True, speed_up=1.05, fps=30, resolution='1080p'):
    """
    合成单个视频：将原始视频、合成音频、翻译文本合成为带字幕的最终视频。

    输入要求：
    - folder 下必须存在：
        - download.mp4（原始视频）
        - audio_combined.wav（合成语音）
        - translation.json（翻译文本）
    - 若 video.mp4 已存在，则跳过

    输出：
    - video.mp4（最终合成视频）

    参数:
        folder (str): 视频素材所在目录
        subtitles (bool): 是否嵌入字幕
        speed_up (float): 播放加速倍数（同时加速视频和音频）
        fps (int): 输出视频帧率
        resolution (str): 输出分辨率（如 '1080p'）
    """
    video_output_path = os.path.join(folder, 'video.mp4')
    if os.path.exists(video_output_path):
        logger.info(f'Video already synthesized in {folder}')
        return

    translation_path = os.path.join(folder, 'translation.json')
    input_audio = os.path.join(folder, 'audio_combined.wav')
    input_video = os.path.join(folder, 'download.mp4')

    # 检查必要输入文件
    if not (os.path.exists(translation_path) and os.path.exists(input_audio)):
        logger.warning(f"Missing input files in {folder}. Skipping.")
        return

    # 加载翻译数据
    with open(translation_path, 'r', encoding='utf-8') as f:
        translation = json.load(f)

    # 生成 SRT 字幕
    srt_path = os.path.join(folder, 'subtitles.srt')
    generate_srt(translation, srt_path, speed_up=speed_up)

    # 处理路径分隔符（FFmpeg 在 Windows 需要 /）
    srt_path = srt_path.replace('\\', '/')

    # 获取视频宽高比并计算目标分辨率
    aspect_ratio = get_aspect_ratio(input_video)
    width, height = convert_resolution(aspect_ratio, resolution)
    resolution_str = f'{width}x{height}'

    # 计算字幕字体大小（自适应）
    font_size = int(width / 128)
    outline = max(1, int(round(font_size / 8)))  # 确保至少为1

    # FFmpeg 滤镜：加速 + 字幕
    video_speed_filter = f"setpts=PTS/{speed_up}"  # 视频加速
    audio_speed_filter = f"atempo={speed_up}"      # 音频加速（1.0~100.0，>2 需级联）

    # 注意：atempo 只支持 0.5~2.0，若 speed_up > 2 需拆分为多个 atempo
    # 本脚本假设 speed_up <= 2（如 1.05）

    subtitle_filter = (
        f"subtitles={srt_path}:"
        f"force_style='FontName=Arial,FontSize={font_size},"
        f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
        f"Outline={outline},WrapStyle=2'"
    )

    if subtitles:
        filter_complex = f"[0:v]{video_speed_filter},{subtitle_filter}[v];[1:a]{audio_speed_filter}[a]"
    else:
        filter_complex = f"[0:v]{video_speed_filter}[v];[1:a]{audio_speed_filter}[a]"

    # 构建 FFmpeg 命令
    ffmpeg_command = [
        'ffmpeg',
        '-i', input_video,
        '-i', input_audio,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '[a]',
        '-r', str(fps),
        '-s', resolution_str,
        '-c:v', 'libx264',      # 视频编码
        '-c:a', 'aac',          # 音频编码
        '-y',                   # 覆盖输出
        video_output_path
    ]

    logger.info(f"Running FFmpeg in {folder}")
    subprocess.run(ffmpeg_command, check=True)
    time.sleep(1)  # 避免文件系统延迟


def synthesize_all_video_under_folder(folder, subtitles=True, speed_up=1.05, fps=30, resolution='1080p'):
    """
    递归遍历指定目录，对所有包含 'download.mp4' 但无 'video.mp4' 的子目录执行视频合成。

    参数:
        folder (str): 根目录路径
        其他参数同 synthesize_video

    返回:
        str: 完成提示信息
    """
    for root, dirs, files in os.walk(folder):
        if 'download.mp4' in files and 'video.mp4' not in files:
            logger.info(f"Synthesizing video in: {root}")
            synthesize_video(
                root,
                subtitles=subtitles,
                speed_up=speed_up,
                fps=fps,
                resolution=resolution
            )
    return f'Synthesized all videos under {folder}'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='视频合成与字幕嵌入工具')
    parser.add_argument('--folder', type=str, 
                       help='要处理的单个视频文件夹路径（若未指定，则批量处理所有）')
    parser.add_argument('--root', type=str, default='videos',
                       help='批量处理时的根目录（默认: videos）')
    parser.add_argument('--no-subtitles', action='store_true',
                       help='不嵌入字幕')
    parser.add_argument('--speed-up', type=float, default=1.00,
                       help='播放加速倍数（默认: 1.00）')
    parser.add_argument('--fps', type=int, default=30,
                       help='输出视频帧率（默认: 30）')
    parser.add_argument('--resolution', type=str, default='1080p',
                       help='输出分辨率（如 1080p, 720p，默认: 1080p）')

    args = parser.parse_args()

    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )

    # 核心逻辑：指定 folder vs 批量处理
    if args.folder:
        # 处理单个指定文件夹
        if not os.path.isdir(args.folder):
            logger.error(f"指定路径不是目录: {args.folder}")
            sys.exit(1)
        logger.info(f"🎬 合成单个视频: {args.folder}")
        synthesize_video(
            args.folder,
            subtitles=not args.no_subtitles,
            speed_up=args.speed_up,
            fps=args.fps,
            resolution=args.resolution
        )
        logger.success("✅ 单个视频合成完成！")
    
    else:
        # 默认行为：批量处理 root 下所有待处理视频
        root_dir = args.root
        if not os.path.isdir(root_dir):
            logger.error(f"根目录不存在: {root_dir}")
            sys.exit(1)
        logger.info(f"🔄 批量合成所有待处理视频（根目录: {root_dir}）")
        result = synthesize_all_video_under_folder(
            root_dir,
            subtitles=not args.no_subtitles,
            speed_up=args.speed_up,
            fps=args.fps,
            resolution=args.resolution
        )
        logger.success(result)