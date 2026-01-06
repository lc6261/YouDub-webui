"""
whisper_asr_stdout_parser.py - 直接从stdout解析whisper输出
"""

import os
import subprocess
import json
import time
import re
import sys

# 配置
WHISPER_DIR = r"C:\whisper-cublas-12.4.0-bin-x64\Release"
WHISPER_EXE = os.path.join(WHISPER_DIR, "whisper-cli.exe")
MODEL = os.path.join(WHISPER_DIR, "ggml-large-v3-q5_0.bin")

def parse_stdout_timestamps(stdout):
    """解析stdout中的时间戳和文本"""
    segments = []
    
    if not stdout:
        return segments
    
    # 查找所有时间戳行
    lines = stdout.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 匹配时间戳模式: [HH:MM:SS.mmm --> HH:MM:SS.mmm]   text
        pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.+)'
        match = re.match(pattern, line)
        
        if match:
            try:
                start_time = match.group(1)
                end_time = match.group(2)
                text = match.group(3).strip()
                
                # 转换时间为秒
                start_seconds = time_str_to_seconds(start_time)
                end_seconds = time_str_to_seconds(end_time)
                
                segments.append({
                    'start': start_seconds,
                    'end': end_seconds,
                    'text': text
                })
                
            except Exception as e:
                print(f"解析时间戳失败: {e}, 行: {line[:50]}")
                continue
    
    return segments

def time_str_to_seconds(time_str):
    """将时间字符串转换为秒数"""
    # 格式: HH:MM:SS.mmm
    parts = time_str.split(':')
    
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        return 0

def transcribe_audio_direct(audio_path, language="en"):
    """直接转录音频文件，从stdout获取结果"""
    print(f"转录音频: {os.path.basename(audio_path)}")
    
    if not os.path.exists(audio_path):
        print(f"❌ 文件不存在: {audio_path}")
        return []
    
    # 检查是否需要转换音频格式
    converted_path = convert_audio_if_needed(audio_path)
    
    # 构建命令
    cmd = [
        WHISPER_EXE,
        "--model", MODEL,
        "--file", os.path.abspath(converted_path),
        "--language", language,
        "--threads", "4",
        "--beam-size", "5",
        "--best-of", "5",
        "--temperature", "0.0,0.2,0.4",
        #"--suppress-blank",
        #"--no-speech-threshold", "0.4",
        "--output-txt",
        "--print-progress"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # 运行命令，直接捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            cwd=WHISPER_DIR,
            encoding='utf-8',
            errors='replace'
        )
        
        stdout, stderr = process.communicate(timeout=1800)  # 30分钟超时
        
        elapsed = time.time() - start_time
        print(f"转录耗时: {elapsed:.1f} 秒")
        print(f"返回码: {process.returncode}")
        
        if process.returncode != 0:
            print(f"❌ 转录失败")
            if stderr:
                print(f"错误: {stderr[:500]}")
            return []
        
        # 从stdout解析结果
        segments = parse_stdout_timestamps(stdout)
        
        print(f"✅ 转录成功，共 {len(segments)} 个段落")
        
        # 显示前几个段落
        if segments:
            print("\n前5个段落:")
            for i, seg in enumerate(segments[:5]):
                print(f"  [{i+1}] [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text'][:60]}...")
        
        return segments
        
    except subprocess.TimeoutExpired:
        print(f"❌ 命令执行超时 (1800秒)")
        return []
    except Exception as e:
        print(f"❌ 转录过程出错: {e}")
        import traceback
        traceback.print_exc()
        return []

def convert_audio_if_needed(audio_path):
    """如果需要，转换音频格式为16kHz单声道"""
    try:
        # 检查音频格式
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        codec, sample_rate, channels = result.stdout.strip().split('\n')[:3]
        sample_rate = int(sample_rate)
        channels = int(channels)
        
        print(f"音频信息: {codec}, {sample_rate}Hz, {channels}声道")
        
        # 如果已经是合适的格式，无需转换
        if sample_rate == 16000 and channels == 1 and codec == 'pcm_s16le':
            print("音频格式已符合要求，无需转换")
            return audio_path
        else:
            print("需要转换音频格式...")
            
            # 创建转换文件
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            converted_path = f"converted_{base_name}.wav"
            
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                converted_path,
                "-y",
                "-loglevel", "error"
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if os.path.exists(converted_path):
                print(f"✅ 音频转换完成: {converted_path}")
                return converted_path
            else:
                print("⚠ 音频转换失败，使用原文件")
                return audio_path
                
    except Exception as e:
        print(f"⚠ 无法检查/转换音频格式: {e}")
        return audio_path

def save_transcription_results(segments, output_base="transcription"):
    """保存转录结果"""
    if not segments:
        print("❌ 无结果可保存")
        return False
    
    try:
        # 按时间排序
        segments.sort(key=lambda x: x['start'])
        
        # 1. 保存为JSON
        json_file = f"{output_base}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON已保存: {json_file}")
        
        # 2. 保存为分钟格式文本
        txt_minutes = f"{output_base}_minutes.txt"
        with open(txt_minutes, 'w', encoding='utf-8') as f:
            for seg in segments:
                start_min = seg['start'] / 60
                end_min = seg['end'] / 60
                f.write(f"[{start_min:.2f}m - {end_min:.2f}m] {seg['text']}\n")
        print(f"✅ 分钟格式文本已保存: {txt_minutes}")
        
        # 3. 保存为分钟:秒格式文本
        txt_minsec = f"{output_base}_minsec.txt"
        with open(txt_minsec, 'w', encoding='utf-8') as f:
            for seg in segments:
                start_min = int(seg['start'] // 60)
                start_sec = seg['start'] % 60
                end_min = int(seg['end'] // 60)
                end_sec = seg['end'] % 60
                f.write(f"[{start_min:02d}:{start_sec:05.2f} - {end_min:02d}:{end_sec:05.2f}] {seg['text']}\n")
        print(f"✅ 分钟:秒格式文本已保存: {txt_minsec}")
        
        # 4. 保存为SRT格式
        srt_file = f"{output_base}.srt"
        with open(srt_file, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                start_h = int(seg['start'] // 3600)
                start_m = int((seg['start'] % 3600) // 60)
                start_s = seg['start'] % 60
                end_h = int(seg['end'] // 3600)
                end_m = int((seg['end'] % 3600) // 60)
                end_s = seg['end'] % 60
                
                f.write(f"{i}\n")
                f.write(f"{start_h:02d}:{start_m:02d}:{start_s:06.3f} --> {end_h:02d}:{end_m:02d}:{end_s:06.3f}\n")
                f.write(f"{seg['text']}\n\n")
        print(f"✅ SRT格式已保存: {srt_file}")
        
        # 5. 保存为合并文本
        merged_file = f"{output_base}_merged.txt"
        with open(merged_file, 'w', encoding='utf-8') as f:
            merged_text = " ".join(seg['text'] for seg in segments)
            f.write(merged_text)
        print(f"✅ 合并文本已保存: {merged_file}")
        
        # 统计信息
        total_chars = sum(len(seg['text']) for seg in segments)
        if segments and segments[-1]['end'] > 0:
            total_duration = segments[-1]['end']
            print(f"\n📊 统计信息:")
            print(f"  段落数: {len(segments)}")
            print(f"  总字符数: {total_chars}")
            print(f"  音频时长: {total_duration:.1f} 秒 ({total_duration/60:.1f} 分钟)")
            if total_duration > 0:
                print(f"  平均语速: {total_chars/(total_duration/60):.0f} 字/分钟")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def split_large_audio(audio_path, segment_duration=600):
    """分割大音频文件"""
    if not os.path.exists(audio_path):
        return None, []
    
    print(f"分割大音频文件: {os.path.basename(audio_path)}")
    
    # 获取音频时长
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        duration = float(result.stdout.strip())
    except:
        print("⚠ 无法获取音频时长，使用估算值")
        file_size = os.path.getsize(audio_path)
        duration = file_size / (16000 * 2)
    
    print(f"音频总时长: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
    
    if duration <= segment_duration * 2:  # 小于20分钟不分割
        print("音频较短，无需分割")
        return None, []
    
    # 创建分割文件
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    segment_files = []
    
    # 计算需要分割多少段
    num_segments = int(duration // segment_duration) + 1
    
    print(f"将分割为 {num_segments} 个片段，每个约 {segment_duration} 秒")
    
    for i in range(num_segments):
        start_time = i * segment_duration
        segment_file = f"{base_name}_part{i:03d}.wav"
        
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(segment_duration),
            "-c", "copy",
            segment_file,
            "-y",
            "-loglevel", "error"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(segment_file):
            segment_files.append(segment_file)
            print(f"  ✅ 片段 {i+1}/{num_segments}: {segment_file}")
        else:
            print(f"  ❌ 片段 {i+1} 分割失败")
    
    print(f"分割完成: {len(segment_files)} 个片段")
    return segment_files

def transcribe_with_split(audio_path, language="en", segment_duration=600):
    """分割大文件并转录"""
    print(f"\n开始处理大文件: {os.path.basename(audio_path)}")
    
    # 分割音频
    segment_files = split_large_audio(audio_path, segment_duration)
    
    if not segment_files:
        print("无需分割，直接转录")
        return transcribe_audio_direct(audio_path, language)
    
    all_segments = []
    
    # 转录每个片段
    for i, segment_file in enumerate(segment_files):
        print(f"\n{'='*60}")
        print(f"处理片段 {i+1}/{len(segment_files)}: {os.path.basename(segment_file)}")
        print(f"{'='*60}")
        
        segments = transcribe_audio_direct(segment_file, language)
        
        if segments:
            # 调整时间戳
            time_offset = i * segment_duration
            for seg in segments:
                seg['start'] += time_offset
                seg['end'] += time_offset
            
            all_segments.extend(segments)
            print(f"✅ 片段转录完成: {len(segments)} 个段落")
        
        # 清理临时文件
        try:
            os.remove(segment_file)
        except:
            pass
    
    # 合并所有段落
    all_segments.sort(key=lambda x: x['start'])
    
    print(f"\n✅ 所有片段转录完成")
    print(f"总计段落数: {len(all_segments)}")
    
    return all_segments

def main():
    print("="*60)
    print("Whisper.cpp 音频转录工具 (直接解析stdout版)")
    print("="*60)
    
    # 检查环境
    if not os.path.exists(WHISPER_EXE):
        print(f"❌ whisper可执行文件不存在: {WHISPER_EXE}")
        return
    
    if not os.path.exists(MODEL):
        print(f"❌ 模型文件不存在: {MODEL}")
        return
    
    print("✅ 环境检查通过")
    print(f"   可执行文件: {WHISPER_EXE}")
    print(f"   模型文件: {MODEL}")
    
    # 获取音频文件
    audio_files = [f for f in os.listdir() if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
    
    if not audio_files:
        print("❌ 当前目录没有音频文件")
        return
    
    print("\n找到的音频文件:")
    for i, f in enumerate(audio_files):
        size = os.path.getsize(f) / 1024 / 1024
        print(f"  [{i+1}] {f} ({size:.1f} MB)")
    
    choice = input("\n选择要转录的文件 (输入编号): ").strip()
    
    try:
        audio_index = int(choice) - 1
        if 0 <= audio_index < len(audio_files):
            audio_file = audio_files[audio_index]
        else:
            print("❌ 无效选择")
            return
    except:
        print("❌ 无效输入")
        return
    
    print(f"\n选择转录文件: {audio_file}")
    file_size = os.path.getsize(audio_file) / 1024 / 1024
    print(f"文件大小: {file_size:.1f} MB")
    
    # 选择语言
    print("\n选择语言:")
    print("  1. 英语 (en)")
    print("  2. 中文 (zh)")
    print("  3. 自动检测")
    
    lang_choice = input("选择 (1/2/3): ").strip()
    
    if lang_choice == "1":
        language = "en"
    elif lang_choice == "2":
        language = "zh"
    else:
        language = "auto"
    
    # 选择转录方式
    print("\n选择转录方式:")
    print("  1. 直接转录 (适合小文件)")
    print("  2. 分割后转录 (适合大文件)")
    
    method_choice = input("选择 (1/2): ").strip()
    
    start_time = time.time()
    
    if method_choice == "1":
        segments = transcribe_audio_direct(audio_file, language)
    else:
        segments = transcribe_with_split(audio_file, language)
    
    total_time = time.time() - start_time
    
    if segments:
        print(f"\n✅ 转录成功!")
        print(f"总耗时: {total_time/60:.1f} 分钟")
        print(f"共 {len(segments)} 个段落")
        
        # 显示前几个段落
        print("\n前5个段落:")
        for i, seg in enumerate(segments[:5]):
            start_min = seg['start'] / 60
            end_min = seg['end'] / 60
            text_preview = seg['text'][:60] + "..." if len(seg['text']) > 60 else seg['text']
            print(f"  [{i+1}] [{start_min:.2f}m - {end_min:.2f}m]: {text_preview}")
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"transcription_{timestamp}_{os.path.splitext(audio_file)[0]}"
        
        print(f"\n保存转录结果...")
        save_transcription_results(segments, base_name)
    else:
        print("\n❌ 转录失败")
    
    print("\n" + "="*60)
    print("完成")
    print("="*60)

if __name__ == "__main__":
    main()