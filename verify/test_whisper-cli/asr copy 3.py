"""
whisper_asr_reliable.py - 可靠版本，直接获取输出
"""

import os
import subprocess
import json
import tempfile
import time
from typing import List, Dict
import sys

# 配置
WHISPER_DIR = r"C:\whisper-cublas-12.4.0-bin-x64\Release"
WHISPER_EXE = os.path.join(WHISPER_DIR, "whisper-cli.exe")
MODEL = os.path.join(WHISPER_DIR, "ggml-large-v3-q5_0.bin")

# 测试文件
test_file = "audio_vocals2.wav"

def run_command(cmd, timeout=600):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
            cwd=WHISPER_DIR,
            encoding='utf-8',
            errors='replace'
        )
        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
    except subprocess.TimeoutExpired:
        print("❌ 命令执行超时")
        return None
    except Exception as e:
        print(f"❌ 命令执行出错: {e}")
        return None


def convert_audio(input_path):
    """转换音频为16kHz单声道"""
    if not os.path.exists(input_path):
        return input_path
    
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, f"converted_{int(time.time())}.wav")
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
        "-y",
        "-loglevel", "error"
    ]
    
    print("转换音频格式...")
    result = run_command(cmd, timeout=300)
    
    if result and result['returncode'] == 0 and os.path.exists(output_path):
        print("✅ 音频转换成功")
        return output_path
    else:
        print("⚠ 使用原始音频文件")
        return input_path


def transcribe_text_only(audio_path, language="zh"):
    """
    最简单的方法：直接获取文本输出
    返回: 转写文本
    """
    print(f"\n开始转写: {os.path.basename(audio_path)}")
    
    # 转换音频
    audio_to_use = convert_audio(audio_path)
    
    # 构建命令 - 直接输出文本到stdout
    cmd = [
        WHISPER_EXE,
        "--model", MODEL,
        "--file", os.path.abspath(audio_to_use),
        "--language", language,
        "--threads", "4",
        "--output-txt"  # 关键：直接输出文本
    ]
    
    print("执行whisper命令...")
    start_time = time.time()
    
    result = run_command(cmd, timeout=600)
    
    if not result:
        return ""
    
    elapsed = time.time() - start_time
    print(f"转写耗时: {elapsed:.1f} 秒")
    print(f"返回码: {result['returncode']}")
    
    if result['returncode'] != 0:
        print(f"❌ 转写失败")
        if result['stderr']:
            print(f"错误: {result['stderr'][:200]}")
        return ""
    
    # 提取转写文本
    if result['stdout']:
        text = result['stdout'].strip()
        print(f"✅ 转写成功: {len(text)} 字符")
        return text
    else:
        print("⚠ 无转写输出")
        return ""


def transcribe_with_json(audio_path, language="zh"):
    """
    获取带时间戳的JSON输出
    返回: 段落列表
    """
    print(f"\n开始带时间戳转写: {os.path.basename(audio_path)}")
    
    # 转换音频
    audio_to_use = convert_audio(audio_path)
    
    # 创建临时输出文件
    temp_dir = tempfile.gettempdir()
    output_base = os.path.join(temp_dir, f"whisper_out_{int(time.time())}")
    
    # 构建命令 - 输出JSON
    cmd = [
        WHISPER_EXE,
        "--model", MODEL,
        "--file", os.path.abspath(audio_to_use),
        "--language", language,
        "--output-json",
        "--output-file", output_base,
        "--threads", "4",
        "--print-progress"
    ]
    
    print("执行whisper命令...")
    start_time = time.time()
    
    result = run_command(cmd, timeout=600)
    
    if not result:
        return []
    
    elapsed = time.time() - start_time
    print(f"转写耗时: {elapsed:.1f} 秒")
    print(f"返回码: {result['returncode']}")
    
    if result['returncode'] != 0:
        print(f"❌ 转写失败")
        if result['stderr']:
            print(f"错误: {result['stderr'][:200]}")
        return []
    
    # 查找JSON输出文件
    json_file = output_base + ".json"
    
    if not os.path.exists(json_file):
        print(f"❌ 未找到JSON输出文件: {json_file}")
        
        # 检查可能的其他位置
        for f in os.listdir(WHISPER_DIR):
            if f.endswith(".json"):
                json_file = os.path.join(WHISPER_DIR, f)
                print(f"找到JSON文件: {json_file}")
                break
        else:
            print("❌ 没有找到任何JSON文件")
            return []
    
    # 读取并解析JSON
    try:
        print(f"读取JSON文件: {json_file}")
        
        # 尝试多种编码
        content = None
        for encoding in ['utf-8', 'gbk', 'latin-1']:
            try:
                with open(json_file, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"使用 {encoding} 编码读取成功")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            # 使用二进制读取
            with open(json_file, 'rb') as f:
                raw_bytes = f.read()
            content = raw_bytes.decode('utf-8', errors='ignore')
            print("使用二进制读取")
        
        # 解析JSON
        data = json.loads(content)
        
        # 提取段落
        segments = []
        
        # 尝试不同的JSON结构
        if isinstance(data, list):
            # 直接是段落列表
            for item in data:
                if isinstance(item, dict):
                    text = item.get('text', item.get('content', ''))
                    if text and text.strip():
                        segments.append({
                            'start': float(item.get('start', 0)),
                            'end': float(item.get('end', 0)),
                            'text': text.strip()
                        })
        
        elif isinstance(data, dict):
            # 包含segments字段
            if 'segments' in data and isinstance(data['segments'], list):
                for seg in data['segments']:
                    if isinstance(seg, dict):
                        text = seg.get('text', seg.get('content', ''))
                        if text and text.strip():
                            segments.append({
                                'start': float(seg.get('start', 0)),
                                'end': float(seg.get('end', 0)),
                                'text': text.strip()
                            })
            
            # 或者直接是transcription
            elif 'transcription' in data and isinstance(data['transcription'], list):
                for seg in data['transcription']:
                    if isinstance(seg, dict):
                        text = seg.get('text', seg.get('content', ''))
                        if text and text.strip():
                            segments.append({
                                'start': float(seg.get('start', 0)),
                                'end': float(seg.get('end', 0)),
                                'text': text.strip()
                            })
        
        print(f"✅ 解析成功: {len(segments)} 个段落")
        return segments
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析失败: {e}")
        # 查看文件内容
        try:
            with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
                preview = f.read()[:500]
            print(f"文件内容预览: {preview}")
        except:
            pass
        return []
    except Exception as e:
        print(f"❌ 处理JSON时出错: {e}")
        return []
    finally:
        # 清理文件
        try:
            os.remove(json_file)
        except:
            pass
        
        # 清理临时音频文件
        if audio_to_use != audio_path and os.path.exists(audio_to_use):
            try:
                os.remove(audio_to_use)
            except:
                pass


def save_results(segments, base_name="transcription"):
    """保存转写结果"""
    if not segments:
        print("❌ 无结果可保存")
        return False
    
    try:
        # 保存为JSON
        json_file = f"{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON已保存: {json_file}")
        
        # 保存为文本
        txt_file = f"{base_name}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            if 'start' in segments[0] and segments[0]['start'] > 0:
                # 带时间戳
                for seg in segments:
                    f.write(f"[{seg['start']:.2f}-{seg['end']:.2f}s] {seg['text']}\n")
            else:
                # 纯文本
                for seg in segments:
                    f.write(f"{seg['text']}\n")
        print(f"✅ 文本已保存: {txt_file}")
        
        # 保存为纯合并文本
        merged_file = f"{base_name}_merged.txt"
        with open(merged_file, 'w', encoding='utf-8') as f:
            merged_text = " ".join(seg['text'] for seg in segments)
            f.write(merged_text)
        print(f"✅ 合并文本已保存: {merged_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False


def main():
    print("="*60)
    print("whisper.cpp 可靠转写工具")
    print("="*60)
    
    # 检查环境
    if not os.path.exists(WHISPER_EXE):
        print(f"❌ whisper可执行文件不存在: {WHISPER_EXE}")
        return
    
    if not os.path.exists(MODEL):
        print(f"❌ 模型文件不存在: {MODEL}")
        return
    
    print(f"✅ 环境检查通过")
    print(f"   可执行文件: {WHISPER_EXE}")
    print(f"   模型文件: {MODEL}")
    
    
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    file_size = os.path.getsize(test_file) / 1024 / 1024
    print(f"\n测试文件: {test_file}")
    print(f"文件大小: {file_size:.1f} MB")
    
    # 选项
    print("\n请选择转写方式:")
    print("  1. 简单文本转写（推荐）")
    print("  2. 带时间戳转写（JSON）")
    print("  3. 两者都试")
    
    choice = input("选择 (1/2/3): ").strip()
    
    results = []
    
    if choice == "1" or choice == "3":
        # 简单文本转写
        text = transcribe_text_only(test_file, "zh")
        
        if text:
            print(f"\n✅ 文本转写成功!")
            print(f"转写内容 ({len(text)} 字符):")
            print("-" * 60)
            print(text[:500] + "..." if len(text) > 500 else text)
            print("-" * 60)
            
            # 转换为段落格式
            results.append({"start": 0, "end": 0, "text": text})
        else:
            print("❌ 文本转写失败")
    
    if choice == "2" or choice == "3":
        # 带时间戳转写
        segments = transcribe_with_json(test_file, "zh")
        
        if segments:
            print(f"\n✅ 带时间戳转写成功!")
            print(f"共 {len(segments)} 个段落")
            
            print("\n前5个段落:")
            for i, seg in enumerate(segments[:5]):
                time_str = f"{seg['start']:.1f}-{seg['end']:.1f}s"
                text_preview = seg['text'][:50] + "..." if len(seg['text']) > 50 else seg['text']
                print(f"  [{i+1}] [{time_str:>10}]: {text_preview}")
            
            results.extend(segments)
        else:
            print("❌ 带时间戳转写失败")
    
    # 保存结果
    if results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"transcription_{timestamp}"
        
        print(f"\n保存转写结果...")
        save_results(results, base_name)
        
        # 统计信息
        if len(results) > 1 and 'start' in results[0]:
            total_chars = sum(len(seg['text']) for seg in results)
            total_duration = results[-1]['end'] if results else 0
            
            print(f"\n📊 统计信息:")
            print(f"  段落数: {len(results)}")
            print(f"  总字符数: {total_chars}")
            print(f"  音频时长: {total_duration/60:.1f} 分钟")
            if total_duration > 0:
                print(f"  平均语速: {total_chars/(total_duration/60):.0f} 字/分钟")
    
    print("\n" + "="*60)
    print("完成")
    print("="*60)


# 兼容原接口的函数
def transcribe_with_whisper_cpp(wav_path: str, language: str = "zh") -> List[Dict[str, any]]:
    """原接口兼容函数"""
    return transcribe_with_json(wav_path, language)


def transcribe_audio_file(audio_path: str, language: str = "zh") -> str:
    """原接口兼容函数"""
    text = transcribe_text_only(audio_path, language)
    return text if text else ""


if __name__ == "__main__":
    main()