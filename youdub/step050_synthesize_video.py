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


def split_text(input_data, punctuations=['，', '；', '：', '。', '？', '！', '\n', '”', ',', ';', ':', '.', '?', '!', '"']):
    """
    将输入的翻译文本按中文标点符号切分为多个短句，并为每句分配对应的时间区间。
    同时根据中文切分比例，同步切分英文原文，确保字幕与音频同步。

    切分规则：
    - 遇到指定标点时尝试切分
    - 避免过短句子（<5 字符，除非是最后一句）
    - 避免连续标点导致的空句（如 "。！"）
    - 中英文文本同步切分，确保时间对齐

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
            - "text": 切分后的英文原文
            - "translation": 切分后的中文短句
            - "speaker": 说话人
    """
    def is_punctuation(char):
        """判断字符是否为指定的中文标点"""
        return char in punctuations

    output_data = []
    for item in input_data:
        # 原始片段信息
        seg_start = item["start"]
        seg_end = item["end"]
        seg_duration = seg_end - seg_start
        seg_text = item["text"]  # 英文原文
        seg_translation = item["translation"]  # 中文翻译
        speaker = item.get("speaker", "SPEAKER_00")

        # 若文本为空，跳过
        if not seg_translation:
            continue

        # 计算中英文文本的总长度
        total_en_chars = len(seg_text)
        total_zh_chars = len(seg_translation)

        # 中文文本切分
        zh_sentences = []
        zh_sentence_start = 0
        
        for i, char in enumerate(seg_translation):
            # 不是标点且不是最后一个字符 → 继续
            if not is_punctuation(char) and i != len(seg_translation) - 1:
                continue

            # 避免过短句子（少于5字），除非是最后一句
            if i - zh_sentence_start < 5 and i != len(seg_translation) - 1:
                continue

            # 避免在连续标点处分割（如 "！？"），跳过后一个标点
            if i < len(seg_translation) - 1 and is_punctuation(seg_translation[i + 1]):
                continue

            # 提取当前中文句子
            zh_sentence = seg_translation[zh_sentence_start:i + 1]
            zh_sentences.append({
                "text": zh_sentence,
                "char_count": len(zh_sentence),
                "start_pos": zh_sentence_start,
                "end_pos": i + 1
            })
            
            # 更新下一句的起始位置
            zh_sentence_start = i + 1
        
        # 处理剩余文本
        if zh_sentence_start < len(seg_translation):
            zh_sentence = seg_translation[zh_sentence_start:]
            zh_sentences.append({
                "text": zh_sentence,
                "char_count": len(zh_sentence),
                "start_pos": zh_sentence_start,
                "end_pos": len(seg_translation)
            })
        
        # 根据中文句子的字符比例，切分英文原文
        def find_english_sentence_boundary(text, start_pos, target_pos, max_search_range=100):
            """
            在目标位置附近寻找合适的英文句子边界
            优先考虑：句号、问号、感叹号、分号、冒号、换行符
            其次考虑：逗号、空格
            确保不截断单词
            """
            if target_pos >= len(text):
                return len(text)
            
            # 定义英文句子结束标点的优先级
            primary_punct = ['.', '?', '!', ';', ':', '\n']
            secondary_punct = [',', ' ']
            
            # 确保不会在单词中间截断
            def is_word_char(c):
                return c.isalnum() or c == "'"
            
            # 从目标位置向前搜索，寻找最近的句子边界
            for i in range(target_pos, max(start_pos, target_pos - max_search_range), -1):
                if text[i:i+1] in primary_punct:
                    return i + 1
            
            # 如果没有找到主要标点，尝试次要标点
            for i in range(target_pos, max(start_pos, target_pos - max_search_range), -1):
                if text[i:i+1] in secondary_punct:
                    return i + 1
            
            # 如果还是没有找到，确保不截断单词
            # 检查目标位置是否在单词中间
            if target_pos < len(text) - 1 and is_word_char(text[target_pos]) and is_word_char(text[target_pos+1]):
                # 从目标位置向后搜索，寻找单词结束
                for i in range(target_pos, min(len(text), target_pos + max_search_range)):
                    if not is_word_char(text[i:i+1]):
                        return i
            
            # 如果还是没有找到，返回目标位置
            return target_pos
        
        current_time = seg_start
        
        for i, zh_sent in enumerate(zh_sentences):
            # 计算中文句子在整个片段中的字符比例
            zh_char_ratio = zh_sent["char_count"] / total_zh_chars
            
            # 根据比例计算英文句子的字符范围
            en_start_pos = int(total_en_chars * (zh_sent["start_pos"] / total_zh_chars))
            en_end_pos = int(total_en_chars * (zh_sent["end_pos"] / total_zh_chars))
            
            # 确保英文句子至少有一个字符
            if en_start_pos == en_end_pos:
                en_end_pos = min(en_start_pos + 1, total_en_chars)
            
            # 寻找合适的英文句子边界
            if i < len(zh_sentences) - 1:  # 不是最后一句
                en_end_pos = find_english_sentence_boundary(seg_text, en_start_pos, en_end_pos)
            else:  # 最后一句，包含所有剩余文本
                en_end_pos = total_en_chars
            
            # 提取英文句子
            en_sentence = seg_text[en_start_pos:en_end_pos].strip()
            
            # 特殊处理：如果是最后一句，确保包含所有剩余文本
            if i == len(zh_sentences) - 1:
                en_sentence = seg_text[en_start_pos:].strip()
                sentence_end = seg_end
            else:
                # 计算当前句子的实际字符比例
                actual_char_ratio = (en_end_pos - en_start_pos) / total_en_chars
                # 根据实际字符比例调整时间
                sentence_end = current_time + (seg_duration * actual_char_ratio)
            
            # 保存分段结果
            output_data.append({
                "start": round(current_time, 3),
                "end": round(sentence_end, 3),
                "text": en_sentence,
                "translation": zh_sent["text"].strip(),
                "speaker": speaker
            })
            
            # 更新下一句的起始时间和位置
            current_time = sentence_end

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
    - 生成中英文双语字幕，英文在上，中文在下

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
            
            # 获取英文原文和中文翻译
            english_text = line['text']
            chinese_text = line['translation']
            
            # 改进的英文自动换行算法，确保不会在单词中间截断
            def wrap_text_english(text, max_line_length):
                """
                英文文本自动换行，确保单词完整性
                """
                if not text:
                    return ""
                
                words = text.split(' ')
                lines = []
                current_line = []
                current_length = 0
                
                for word in words:
                    # 计算单词长度（考虑空格）
                    word_length = len(word)
                    if current_length + word_length + (1 if current_line else 0) <= max_line_length:
                        # 单词可以加入当前行
                        current_line.append(word)
                        current_length += word_length + (1 if current_line else 0)
                    else:
                        # 单词不能加入当前行，开始新行
                        lines.append(' '.join(current_line))
                        current_line = [word]
                        current_length = word_length
                
                # 处理最后一行
                if current_line:
                    lines.append(' '.join(current_line))
                
                return '\n'.join(lines)
            
            # 中文自动换行（按字符）
            def wrap_text_chinese(text, max_line_length):
                """
                中文文本自动换行，按字符拆分
                """
                if not text:
                    return ""
                
                lines = []
                for i in range(0, len(text), max_line_length):
                    lines.append(text[i:i+max_line_length])
                
                return '\n'.join(lines)
            
            # 处理英文原文的自动换行
            if english_text:
                wrapped_english = wrap_text_english(english_text, max_line_char)
            else:
                wrapped_english = ''
            
            # 处理中文翻译的自动换行
            if chinese_text:
                wrapped_chinese = wrap_text_chinese(chinese_text, max_line_char)
            else:
                wrapped_chinese = ''
            
            # 组合双语字幕，英文在上，中文在下
            if wrapped_english and wrapped_chinese:
                # 英文和中文都有
                bilingual_text = f'{wrapped_english}\n{wrapped_chinese}'
            elif wrapped_english:
                # 只有英文
                bilingual_text = wrapped_english
            elif wrapped_chinese:
                # 只有中文
                bilingual_text = wrapped_chinese
            else:
                # 都没有，跳过
                continue

            # 写入 SRT 格式
            f.write(f'{i + 1}\n')
            f.write(f'{start} --> {end}\n')
            f.write(f'{bilingual_text}\n\n')


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


def detect_gpu_encoder():
    """
    检测系统中可用的GPU视频编码器。
    
    返回:
        str: 可用的GPU编码器名称，如 'h264_nvenc'（NVIDIA）、'h264_amf'（AMD）、'h264_qsv'（Intel），
             若没有可用GPU编码器则返回 'libx264'（CPU编码）
    """
    # 支持的GPU编码器列表，按优先级排序
    gpu_encoders = [
        'h264_nvenc',   # NVIDIA H.264
        'hevc_nvenc',   # NVIDIA HEVC
        'h264_amf',     # AMD H.264
        'hevc_amf',     # AMD HEVC
        'h264_qsv',     # Intel H.264
        'hevc_qsv',     # Intel HEVC
    ]
    
    for encoder in gpu_encoders:
        try:
            # 测试编码器是否可用
            cmd = ['ffmpeg', '-hide_banner', '-encoders', '|', 'findstr', encoder]
            result = subprocess.run(
                f'ffmpeg -hide_banner -encoders | findstr {encoder}',
                shell=True, 
                capture_output=True,
                text=True
            )
            if encoder in result.stdout:
                return encoder
        except Exception:
            continue
    
    logger.info("未检测到可用的GPU编码器，将使用CPU编码")
    return 'libx264'


def synthesize_video(folder, subtitles=True, speed_up=1.00, fps=30, resolution='1080p', use_gpu=True):
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
        use_gpu (bool): 是否使用GPU加速编码
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

    # 计算字幕字体大小（自适应）- 为双语字幕调整字体大小
    font_size = int(width / 140)  # 双语字幕需要更小的字体以适应垂直空间
    outline = max(1, int(round(font_size / 8)))  # 确保至少为1

    # FFmpeg 滤镜：加速 + 字幕
    video_speed_filter = f"setpts=PTS/{speed_up}"  # 视频加速
    audio_speed_filter = f"atempo={speed_up}"      # 音频加速（1.0~100.0，>2 需级联）

    # 处理 speed_up > 2 的情况，拆分为多个 atempo
    if speed_up > 2:
        # 计算需要多少个 atempo 滤镜（每个最多 2.0）
        atempo_filters = []
        remaining_speed = speed_up
        while remaining_speed > 2.0:
            atempo_filters.append("2.0")
            remaining_speed /= 2.0
        atempo_filters.append(f"{remaining_speed:.2f}")
        audio_speed_filter = ",".join([f"atempo={f}" for f in atempo_filters])

    subtitle_filter = (
        f"subtitles={srt_path}:" 
        f"force_style='FontName=Arial,FontSize={font_size}," 
        f"PrimaryColour=&HFFFFFF,OutlineColour=&H000000," 
        f"Outline={outline},WrapStyle=2,MarginV={int(height * 0.05)}," 
        f"Alignment=2,Bold=1'"
    )

    if subtitles:
        filter_complex = f"[0:v]{video_speed_filter},{subtitle_filter}[v];[1:a]{audio_speed_filter}[a]"
    else:
        filter_complex = f"[0:v]{video_speed_filter}[v];[1:a]{audio_speed_filter}[a]"

    # 选择编码器
    video_encoder = detect_gpu_encoder() if use_gpu else 'libx264'
    logger.info(f"使用编码器: {video_encoder}")

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
        '-c:v', video_encoder,      # 视频编码
        '-c:a', 'aac',               # 音频编码
    ]
    
    # 添加GPU编码优化参数
    if video_encoder in ['h264_nvenc', 'hevc_nvenc']:  # NVIDIA 特定优化
        ffmpeg_command.extend([
            '-preset', 'p4',          # 编码预设：p0（最快）- p7（最高质量）
            '-cq', '23',              # 质量控制，类似CRF，范围0-51
            '-rc', 'vbr_hq',          # 高质量可变比特率
        ])
    elif video_encoder in ['h264_amf', 'hevc_amf']:  # AMD 特定优化
        ffmpeg_command.extend([
            '-preset', 'balanced',     # 编码预设：speed, balanced, quality
            '-quality', 'quality',     # 质量模式
        ])
    elif video_encoder in ['h264_qsv', 'hevc_qsv']:  # Intel 特定优化
        ffmpeg_command.extend([
            '-preset', 'balanced',     # 编码预设：veryfast, fast, balanced, quality
        ])
    else:  # CPU 编码优化
        ffmpeg_command.extend([
            '-preset', 'medium',       # CPU编码预设
            '-crf', '23',              # CPU编码质量控制
        ])
    
    # 添加输出文件
    ffmpeg_command.extend([
        '-y',                   # 覆盖输出
        video_output_path
    ])

    logger.info(f"Running FFmpeg in {folder}，使用 {'GPU' if video_encoder != 'libx264' else 'CPU'} 编码")
    subprocess.run(ffmpeg_command, check=True)
    time.sleep(1)  # 避免文件系统延迟


def synthesize_all_video_under_folder(folder, subtitles=True, speed_up=1.00, fps=30, resolution='1080p', use_gpu=True):
    """
    递归遍历指定目录，对所有包含 'download.mp4' 但无 'video.mp4' 的子目录执行视频合成。

    参数:
        folder (str): 根目录路径
        其他参数同 synthesize_video
        use_gpu (bool): 是否使用GPU加速编码

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
                resolution=resolution,
                use_gpu=use_gpu
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
    parser.add_argument('--use-gpu', action='store_true',
                       help='使用GPU加速编码（默认: CPU编码）')

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
            resolution=args.resolution,
            use_gpu=args.use_gpu
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
            resolution=args.resolution,
            use_gpu=args.use_gpu
        )
        logger.success(result)