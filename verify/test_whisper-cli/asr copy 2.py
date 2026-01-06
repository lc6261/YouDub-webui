"""
whisper_asr_reliable_local.py - 本地临时文件版本
"""

import os
import subprocess
import json
import tempfile
import time
import sys
import threading
from typing import List, Dict, Optional
import queue
import shutil

# 配置
WHISPER_DIR = r"C:\whisper-cublas-12.4.0-bin-x64\Release"
WHISPER_EXE = os.path.join(WHISPER_DIR, "whisper-cli.exe")
MODEL = os.path.join(WHISPER_DIR, "ggml-large-v3-q5_0.bin")

# 测试文件
test_file = "audio_vocals.wav"

# 创建本地临时目录
LOCAL_TEMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
print(f"本地临时目录: {LOCAL_TEMP_DIR}")

class ProcessTimeout(Exception):
    """进程超时异常"""
    pass

def run_command_with_timeout(cmd, timeout=3600):
    """运行命令并支持超时和实时输出"""
    def target(queue, cmd, cwd):
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                cwd=cwd,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            queue.put({
                'returncode': process.returncode,
                'stdout': stdout,
                'stderr': stderr,
                'process': process
            })
        except Exception as e:
            queue.put({'error': str(e)})
    
    q = queue.Queue()
    thread = threading.Thread(target=target, args=(q, cmd, WHISPER_DIR))
    thread.daemon = True
    thread.start()
    
    try:
        result = q.get(timeout=timeout)
        return result
    except queue.Empty:
        raise ProcessTimeout(f"命令执行超时 ({timeout}秒)")
    except Exception as e:
        return {'error': str(e)}

def split_audio_file(input_path, segment_duration=600):
    """分割大音频文件为小段"""
    if not os.path.exists(input_path):
        return None, []
    
    # 获取音频时长
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        duration = float(result.stdout.strip())
    except:
        print("⚠ 无法获取音频时长，使用估算值")
        # 根据文件大小估算：假设16000Hz 16-bit 单声道
        file_size = os.path.getsize(input_path)
        duration = file_size / (16000 * 2)  # 16-bit = 2 bytes
        
    print(f"音频总时长: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
    
    if duration <= segment_duration * 2:  # 小于20分钟不分割
        print("音频较短，无需分割")
        return input_path, []
    
    # 创建本地分割目录
    split_dir = os.path.join(LOCAL_TEMP_DIR, f"split_{int(time.time())}")
    os.makedirs(split_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 分割音频
    print(f"开始分割音频 (每段 {segment_duration} 秒)...")
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-c", "copy",
        "-map", "0:a",
        os.path.join(split_dir, f"{base_name}_%03d.wav"),
        "-y",
        "-loglevel", "error"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        print(f"⚠ 分割失败: {result.stderr[:200]}")
        return input_path, []
    
    # 获取分割后的文件列表
    segment_files = []
    for f in sorted(os.listdir(split_dir)):
        if f.endswith('.wav'):
            segment_files.append(os.path.join(split_dir, f))
    
    print(f"✅ 分割完成: {len(segment_files)} 个片段")
    print(f"片段保存在: {split_dir}")
    return split_dir, segment_files

def convert_audio(input_path):
    """转换音频为16kHz单声道"""
    if not os.path.exists(input_path):
        return input_path
    
    # 检查音频格式
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        codec, sample_rate, channels = result.stdout.strip().split('\n')[:3]
        sample_rate = int(sample_rate)
        channels = int(channels)
        
        print(f"音频信息: {codec}, {sample_rate}Hz, {channels}声道")
        
        # 如果已经是合适的格式，无需转换
        if sample_rate == 16000 and channels == 1 and codec == 'pcm_s16le':
            print("音频格式已符合要求，无需转换")
            return input_path
    except:
        print("⚠ 无法获取音频信息，执行转换")
    
    # 在本地临时目录创建转换文件
    timestamp = int(time.time())
    output_path = os.path.join(LOCAL_TEMP_DIR, f"converted_{timestamp}_{os.path.basename(input_path)}")
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-acodec", "pcm_s16le",
        output_path,
        "-y",
        "-loglevel", "error"
    ]
    
    print("转换音频格式...")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(output_path):
            elapsed = time.time() - start_time
            print(f"✅ 音频转换成功 ({elapsed:.1f} 秒)")
            print(f"转换文件: {output_path}")
            return output_path
        else:
            print(f"⚠ 音频转换失败: {result.stderr[:200]}")
            return input_path
    except Exception as e:
        print(f"⚠ 转换出错: {e}")
        return input_path

def debug_json_file(json_file):
    """调试JSON文件内容"""
    try:
        print(f"\n调试JSON文件: {json_file}")
        print(f"文件大小: {os.path.getsize(json_file)} bytes")
        
        # 查看文件头部
        with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
            first_500 = f.read(500)
            print(f"文件头部 (前500字符):\n{first_500}")
        
        # 尝试解析
        with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        data = json.loads(content)
        print(f"JSON结构类型: {type(data)}")
        
        if isinstance(data, dict):
            print(f"字典键: {list(data.keys())}")
            if 'segments' in data:
                segments = data['segments']
                print(f"segments 类型: {type(segments)}, 长度: {len(segments) if hasattr(segments, '__len__') else 'N/A'}")
                
                # 打印第一个segment
                if segments and len(segments) > 0:
                    print(f"第一个segment: {segments[0]}")
        
        return data
    except Exception as e:
        print(f"调试出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def transcribe_segments(segment_files, language="en"):
    """转录多个音频片段"""
    all_segments = []
    total_files = len(segment_files)
    
    for i, segment_file in enumerate(segment_files, 1):
        print(f"\n{'='*60}")
        print(f"处理片段 {i}/{total_files}: {os.path.basename(segment_file)}")
        print(f"{'='*60}")
        
        file_size = os.path.getsize(segment_file) / 1024 / 1024
        print(f"片段大小: {file_size:.1f} MB")
        print(f"片段路径: {segment_file}")
        
        try:
            segments = transcribe_with_json_single(segment_file, language)
            if segments:
                # 调整时间戳
                time_offset = (i-1) * 600  # 假设每个片段600秒
                for seg in segments:
                    seg['start'] += time_offset
                    seg['end'] += time_offset
                all_segments.extend(segments)
                print(f"✅ 片段转录完成: {len(segments)} 个段落")
                print(f"示例段落: {segments[0]['text'][:100] if segments else '无'}")
            else:
                print(f"⚠ 片段转录失败或无内容")
        except Exception as e:
            print(f"❌ 片段转录出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 返回分割目录路径，以便后续清理
    split_dir = os.path.dirname(segment_files[0]) if segment_files else None
    return all_segments, split_dir

def transcribe_with_json_single(audio_path, language="en"):
    """转录单个音频文件"""
    print(f"开始转写: {os.path.basename(audio_path)}")
    
    # 转换音频
    audio_to_use = convert_audio(audio_path)
    print(f"使用音频文件: {audio_to_use}")
    
    # 在本地临时目录创建输出文件
    timestamp = int(time.time())
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_base = os.path.join(LOCAL_TEMP_DIR, f"whisper_out_{timestamp}_{base_name}")
    
    # 构建命令 - 使用更可靠的参数
    cmd = [
        WHISPER_EXE,
        "--model", MODEL,
        "--file", os.path.abspath(audio_to_use),
        "--language", language,
        "--output-json",
        "--output-file", output_base,
        "--output-txt",  # 同时输出txt
        "--threads", "4",
        "--beam-size", "1",  # 使用更快的设置
        "--print-progress",
        "--no-timestamps"  # 有些版本需要这个
    ]
    
    print(f"执行whisper命令...")
    print(f"输出基名: {output_base}")
    
    start_time = time.time()
    
    try:
        # 直接运行并捕获输出
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
        
        # 读取输出
        stdout, stderr = process.communicate(timeout=1800)  # 30分钟超时
        
        elapsed = time.time() - start_time
        print(f"转写耗时: {elapsed:.1f} 秒")
        print(f"返回码: {process.returncode}")
        
        if stdout:
            print(f"标准输出预览: {stdout[:200]}")
        
        if stderr:
            print(f"错误输出预览: {stderr[:200]}")
        
        if process.returncode != 0:
            print(f"❌ 转写失败")
            return []
        
        # 查找JSON输出文件
        json_file = output_base + ".json"
        txt_file = output_base + ".txt"
        
        print(f"查找输出文件:")
        print(f"  JSON文件: {json_file} - 存在: {os.path.exists(json_file)}")
        print(f"  TXT文件: {txt_file} - 存在: {os.path.exists(txt_file)}")
        
        # 检查本地目录
        print(f"\n检查本地临时目录 {LOCAL_TEMP_DIR}:")
        for f in os.listdir(LOCAL_TEMP_DIR):
            if f".json" in f or f".txt" in f:
                print(f"  - {f}")
        
        # 检查whisper目录
        print(f"\n检查whisper目录 {WHISPER_DIR}:")
        json_files_in_dir = []
        for f in os.listdir(WHISPER_DIR):
            if f".json" in f:
                full_path = os.path.join(WHISPER_DIR, f)
                json_files_in_dir.append(full_path)
                print(f"  - {f}")
        
        # 尝试不同的JSON文件位置
        possible_json_files = []
        
        if os.path.exists(json_file):
            possible_json_files.append(json_file)
        
        # 在whisper目录中查找
        for f in json_files_in_dir:
            if base_name in f or "whisper_out" in f:
                possible_json_files.append(f)
        
        # 如果没有找到，尝试任何JSON文件
        if not possible_json_files and json_files_in_dir:
            possible_json_files = json_files_in_dir
        
        if not possible_json_files:
            print("❌ 未找到任何JSON输出文件")
            
            # 尝试从stdout提取
            if stdout and len(stdout.strip()) > 10:
                print("尝试从标准输出提取文本...")
                # 简单的文本提取
                lines = stdout.strip().split('\n')
                meaningful_lines = [line.strip() for line in lines if len(line.strip()) > 10]
                if meaningful_lines:
                    text = ' '.join(meaningful_lines)
                    return [{'start': 0, 'end': 0, 'text': text}]
            
            return []
        
        # 调试第一个JSON文件
        json_to_use = possible_json_files[0]
        print(f"使用JSON文件: {json_to_use}")
        
        data = debug_json_file(json_to_use)
        
        segments = []
        
        if data is not None:
            # 尝试不同的JSON结构
            if isinstance(data, dict):
                if 'segments' in data and isinstance(data['segments'], list):
                    for seg in data['segments']:
                        if isinstance(seg, dict):
                            text = seg.get('text', '').strip()
                            if text:
                                segments.append({
                                    'start': float(seg.get('start', 0)),
                                    'end': float(seg.get('end', 0)),
                                    'text': text
                                })
                elif 'text' in data:
                    # 直接包含text字段
                    text = data['text'].strip()
                    if text:
                        segments.append({
                            'start': 0,
                            'end': 0,
                            'text': text
                        })
            elif isinstance(data, list):
                # 直接是段落列表
                for item in data:
                    if isinstance(item, dict):
                        text = item.get('text', '').strip()
                        if text:
                            segments.append({
                                'start': float(item.get('start', 0)),
                                'end': float(item.get('end', 0)),
                                'text': text
                            })
        
        # 如果JSON没有内容，尝试读取txt文件
        if not segments and os.path.exists(txt_file):
            print(f"从TXT文件读取: {txt_file}")
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    segments.append({'start': 0, 'end': 0, 'text': text})
        
        print(f"✅ 解析成功: {len(segments)} 个段落")
        return segments
        
    except subprocess.TimeoutExpired:
        print(f"❌ 命令执行超时 (1800秒)")
        return []
    except Exception as e:
        print(f"❌ 转写过程出错: {e}")
        import traceback
        traceback.print_exc()
        return []

def transcribe_with_json(audio_path, language="en", enable_split=True):
    """转录音频文件，支持大文件分割"""
    print(f"\n开始带时间戳转写: {os.path.basename(audio_path)}")
    
    # 检查文件大小
    file_size = os.path.getsize(audio_path) / 1024 / 1024
    print(f"文件大小: {file_size:.1f} MB")
    
    if file_size > 200 and enable_split:  # 大于200MB时分割
        print("文件较大，启用分割模式...")
        split_dir, segments = split_audio_file(audio_path, segment_duration=600)
        
        if segments:
            all_segments, _ = transcribe_segments(segments, language)
            
            # 清理分割目录
            if split_dir and os.path.exists(split_dir):
                try:
                    shutil.rmtree(split_dir)
                    print(f"清理分割目录: {split_dir}")
                except:
                    print(f"⚠ 无法清理分割目录: {split_dir}")
            
            return all_segments
        else:
            print("分割失败，尝试直接处理...")
    
    # 直接处理
    return transcribe_with_json_single(audio_path, language)

def save_results(segments, base_name="transcription"):
    """保存转写结果"""
    if not segments:
        print("❌ 无结果可保存")
        return False
    
    try:
        # 按时间排序
        segments.sort(key=lambda x: x['start'])
        
        # 保存为JSON
        json_file = f"{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON已保存: {json_file}")
        
        # 保存为带时间戳的文本
        txt_file = f"{base_name}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            for seg in segments:
                if seg['start'] > 0 or seg['end'] > 0:
                    start_min = seg['start'] / 60
                    end_min = seg['end'] / 60
                    f.write(f"[{start_min:.2f}m - {end_min:.2f}m] {seg['text']}\n")
                else:
                    f.write(f"{seg['text']}\n")
        print(f"✅ 时间戳文本已保存: {txt_file}")
        
        # 保存为纯合并文本
        merged_file = f"{base_name}_merged.txt"
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
            print(f"  音频时长: {total_duration/60:.1f} 分钟")
            if total_duration > 0:
                print(f"  平均语速: {total_chars/(total_duration/60):.0f} 字/分钟")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_temp_files():
    """清理临时文件"""
    print(f"\n清理临时目录: {LOCAL_TEMP_DIR}")
    
    if os.path.exists(LOCAL_TEMP_DIR):
        try:
            # 只删除临时文件，保留目录
            for filename in os.listdir(LOCAL_TEMP_DIR):
                file_path = os.path.join(LOCAL_TEMP_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"  删除: {filename}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"  删除目录: {filename}")
                except Exception as e:
                    print(f"  无法删除 {filename}: {e}")
            
            print("✅ 临时文件清理完成")
        except Exception as e:
            print(f"⚠ 清理失败: {e}")

def main():
    print("="*60)
    print("whisper.cpp 大文件转录工具 (本地临时文件版)")
    print("="*60)
    print(f"本地临时目录: {LOCAL_TEMP_DIR}")
    
    # 检查环境
    if not os.path.exists(WHISPER_EXE):
        print(f"❌ whisper可执行文件不存在: {WHISPER_EXE}")
        return
    
    if not os.path.exists(MODEL):
        print(f"❌ 模型文件不存在: {MODEL}")
        return
    
    # 检查FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        print("✅ FFmpeg 可用")
    except:
        print("⚠ FFmpeg 可能未安装或不在PATH中")
    
    print(f"✅ 环境检查通过")
    print(f"   可执行文件: {WHISPER_EXE}")
    print(f"   模型文件: {MODEL}")
    
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    # 显示文件信息
    file_size = os.path.getsize(test_file) / 1024 / 1024
    print(f"\n转录文件: {test_file}")
    print(f"文件大小: {file_size:.1f} MB")
    
    if file_size > 1000:  # 大于1GB
        print("⚠ 警告：文件非常大，转录可能需要很长时间（数小时）")
        print("建议先分割音频或使用更强大的硬件")
        
        confirm = input("是否继续？ (y/n): ").strip().lower()
        if confirm != 'y':
            print("取消转录")
            return
    
    print("\n请选择转录方式:")
    print("  1. 带时间戳转录（JSON输出，支持大文件分割）")
    print("  2. 带时间戳转录（不分割，适合小文件）")
    print("  3. 测试单个小片段")
    
    choice = input("选择 (1/2/3): ").strip()
    
    start_time = time.time()
    segments = []
    
    if choice == "1":
        segments = transcribe_with_json(test_file, "zh", enable_split=True)
    elif choice == "2":
        segments = transcribe_with_json(test_file, "zh", enable_split=False)
    elif choice == "3":
        # 测试模式：只处理第一个片段
        print("\n测试模式：只处理第一个10分钟片段")
        split_dir, segment_files = split_audio_file(test_file, segment_duration=600)
        if segment_files and len(segment_files) > 0:
            segments, _ = transcribe_segments([segment_files[0]], "zh")
            
            # 清理其他片段
            for i in range(1, len(segment_files)):
                try:
                    os.remove(segment_files[i])
                except:
                    pass
            
            # 清理分割目录
            if split_dir and os.path.exists(split_dir):
                try:
                    shutil.rmtree(split_dir)
                except:
                    pass
    else:
        print("无效选择，使用默认方式（带分割）")
        segments = transcribe_with_json(test_file, "zh", enable_split=True)
    
    total_time = time.time() - start_time
    
    if segments:
        print(f"\n✅ 转录成功!")
        print(f"总耗时: {total_time/60:.1f} 分钟")
        print(f"共 {len(segments)} 个段落")
        
        print("\n前10个段落:")
        for i, seg in enumerate(segments[:10]):
            start_min = seg['start'] / 60
            end_min = seg['end'] / 60
            text_preview = seg['text'][:50] + "..." if len(seg['text']) > 50 else seg['text']
            print(f"  [{i+1:2d}] [{start_min:6.2f}m - {end_min:6.2f}m]: {text_preview}")
        
        # 保存结果
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"transcription_{timestamp}"
        
        print(f"\n保存转录结果...")
        save_results(segments, base_name)
    else:
        print("\n❌ 转录失败，请检查:")
        print("   1. 文件是否损坏")
        print("   2. 音频格式是否支持")
        print("   3. whisper模型是否完整")
        print("   4. 是否有足够内存（大文件需要大量RAM）")
    
    # 清理临时文件
    cleanup_temp_files()
    
    print("\n" + "="*60)
    print("完成")
    print("="*60)

# 兼容原接口的函数
def transcribe_with_whisper_cpp(wav_path: str, language: str = "en") -> List[Dict[str, any]]:
    """原接口兼容函数"""
    return transcribe_with_json(wav_path, language, enable_split=True)

if __name__ == "__main__":
    main()