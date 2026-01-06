"""
whisper_debug.py - 调试whisper.cpp输出问题
"""

import os
import subprocess
import json
import time
import re
from datetime import datetime

# 配置
WHISPER_DIR = r"C:\whisper-cublas-12.4.0-bin-x64\Release"
WHISPER_EXE = os.path.join(WHISPER_DIR, "whisper-cli.exe")
MODEL = os.path.join(WHISPER_DIR, "ggml-large-v3-q5_0.bin")

# 测试文件
test_file = "converted_audio_vocals_part000.wav"

def vtt_time_to_seconds(time_str):
    """VTT时间转秒"""
    time_str = time_str.strip()
    
    # 移除可能的时间戳格式修饰符
    if time_str.startswith('[') or time_str.startswith('('):
        time_str = time_str[1:]
    if time_str.endswith(']') or time_str.endswith(')'):
        time_str = time_str[:-1]
    
    try:
        # 处理毫秒分隔符（可能是.或,）
        if '.' in time_str:
            parts = time_str.split('.')
        elif ',' in time_str:
            parts = time_str.split(',')
        else:
            parts = [time_str]
        
        time_part = parts[0]
        ms_part = parts[1] if len(parts) > 1 else "000"
        
        # 解析时:分:秒
        time_components = time_part.split(':')
        
        if len(time_components) == 3:  # HH:MM:SS
            hours = float(time_components[0])
            minutes = float(time_components[1])
            seconds = float(time_components[2])
        elif len(time_components) == 2:  # MM:SS
            hours = 0
            minutes = float(time_components[0])
            seconds = float(time_components[1])
        else:
            hours = 0
            minutes = 0
            seconds = float(time_part)
        
        # 添加毫秒
        total_seconds = hours * 3600 + minutes * 60 + seconds + float(f"0.{ms_part[:3]}") if ms_part else 0
        return total_seconds
    except Exception as e:
        print(f"  警告: 无法解析时间 '{time_str}': {e}")
        return 0

def debug_whisper_output():
    """调试whisper输出"""
    print("=" * 80)
    print("🎯 WHISPER.CPP 输出调试工具")
    print("=" * 80)
    
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return
    
    print(f"📁 音频文件: {os.path.abspath(test_file)}")
    print(f"📊 大小: {os.path.getsize(test_file) / 1024 / 1024:.1f} MB")
    
    # 创建临时输出文件
    timestamp = int(time.time())
    output_base = f"debug_output_{timestamp}"
    
    # 步骤1: 直接运行whisper命令
    print("\n" + "=" * 80)
    print("🚀 步骤1: 运行whisper命令")
    print("=" * 80)
    
    cmd = [
        WHISPER_EXE,
        "--model", MODEL,
        "--file", os.path.abspath(test_file),
        "--language", "en",
        "--output-vtt",
        "--output-srt",
        "--output-txt",
        "--output-file", output_base,
        "--threads", "4",
        "--print-progress"
    ]
    
    print(f"💻 命令: {' '.join(cmd[:4])} ... {' '.join(cmd[4:])}")
    print(f"⏱️  开始时间: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        print("\n🔄 正在运行whisper...")
        start_time = time.time()
        
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
        
        # 实时显示进度
        progress_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                line = line.strip()
                if line:
                    print(f"   {line}")
                    progress_lines.append(line)
        
        stdout, stderr = process.communicate(timeout=1800)
        elapsed = time.time() - start_time
        
        print(f"\n✅ 完成! 耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        print(f"📊 返回码: {process.returncode}")
        
        # 保存原始输出
        with open(f"{output_base}_stdout.txt", 'w', encoding='utf-8') as f:
            f.write(stdout)
        print(f"💾 stdout已保存到: {output_base}_stdout.txt")
        
        with open(f"{output_base}_stderr.txt", 'w', encoding='utf-8') as f:
            f.write(stderr)
        print(f"💾 stderr已保存到: {output_base}_stderr.txt")
        
        # 步骤2: 分析输出文件
        print("\n" + "=" * 80)
        print("📊 步骤2: 分析输出文件")
        print("=" * 80)
        
        files_to_check = [
            f"{output_base}.vtt",
            f"{output_base}.srt",
            f"{output_base}.txt",
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"\n📁 分析文件: {file_path}")
                print(f"  📏 大小: {os.path.getsize(file_path)} 字节")
                
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # 显示文件类型特定的信息
                if file_path.endswith('.vtt'):
                    analyze_vtt_file(content)
                elif file_path.endswith('.srt'):
                    analyze_srt_file(content)
                elif file_path.endswith('.txt'):
                    analyze_txt_file(content)
            else:
                print(f"\n❌ 文件不存在: {file_path}")
        
        # 步骤3: 检查前10分钟的重复内容
        print("\n" + "=" * 80)
        print("🔍 步骤3: 专项检查前10分钟重复内容")
        print("=" * 80)
        
        vtt_file = f"{output_base}.vtt"
        if os.path.exists(vtt_file):
            check_first_10_minutes(vtt_file)
        else:
            print("❌ VTT文件不存在，无法检查前10分钟内容")
        
        # 步骤4: 详细检查所有重复内容
        print("\n" + "=" * 80)
        print("🔍 步骤4: 详细检查所有重复内容")
        print("=" * 80)
        
        if os.path.exists(vtt_file):
            check_all_duplicates(vtt_file)
        
    except subprocess.TimeoutExpired:
        print("❌ 命令执行超时 (30分钟)")
    except KeyboardInterrupt:
        print("\n⚠️  用户中断")
    except Exception as e:
        print(f"❌ 调试过程出错: {e}")
        import traceback
        traceback.print_exc()

def analyze_vtt_file(content):
    """分析VTT文件"""
    lines = [line.rstrip() for line in content.strip().split('\n')]
    
    # 统计段落数
    timestamp_lines = [line for line in lines if '-->' in line]
    print(f"  📝 时间戳行数: {len(timestamp_lines)}")
    
    if timestamp_lines:
        # 显示前5个段落
        print(f"  📋 前5个段落:")
        segment_count = 0
        i = 0
        
        while i < len(lines) and segment_count < 5:
            line = lines[i]
            if '-->' in line:
                segment_count += 1
                
                # 解析时间
                times = line.split('-->')
                start_time = times[0].strip()
                end_time = times[1].strip().split()[0] if ' ' in times[1] else times[1].strip()
                
                # 收集文本
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                
                text = ' '.join(text_lines).strip()
                text_preview = text[:60] + "..." if len(text) > 60 else text
                
                print(f"    段落 {segment_count}:")
                print(f"      ⏰ {start_time} --> {end_time}")
                print(f"      💬 {text_preview}")
                
                # 计算持续时间
                start_sec = vtt_time_to_seconds(start_time)
                end_sec = vtt_time_to_seconds(end_time)
                if end_sec > start_sec:
                    print(f"      ⏱️  时长: {end_sec - start_sec:.1f}秒")
                print()
            else:
                i += 1
        
        # 检查总时长
        if timestamp_lines:
            last_line = timestamp_lines[-1]
            end_time = last_line.split('-->')[1].strip().split()[0] if ' ' in last_line else last_line.split('-->')[1].strip()
            total_seconds = vtt_time_to_seconds(end_time)
            print(f"  ⏱️  估计总时长: {total_seconds:.1f}秒 ({total_seconds/60:.1f}分钟)")

def analyze_srt_file(content):
    """分析SRT文件"""
    blocks = content.strip().split('\n\n')
    print(f"  📝 段落块数: {len(blocks)}")
    
    if blocks:
        print(f"  📋 前3个段落:")
        for i, block in enumerate(blocks[:3]):
            print(f"    段落 {i+1}:")
            lines = [line.strip() for line in block.split('\n') if line.strip()]
            for j, line in enumerate(lines):
                prefix = "      "
                if j == 0:
                    prefix = "      #️⃣  "
                elif '-->' in line:
                    prefix = "      ⏰  "
                else:
                    prefix = "      💬  "
                print(f"{prefix}{line[:70]}{'...' if len(line) > 70 else ''}")

def analyze_txt_file(content):
    """分析TXT文件"""
    lines = content.strip().split('\n')
    print(f"  📝 行数: {len(lines)}")
    
    if lines:
        print(f"  📋 内容预览:")
        print(f"    第一行: {lines[0][:80]}{'...' if len(lines[0]) > 80 else ''}")
        if len(lines) > 1:
            print(f"    第二行: {lines[1][:80]}{'...' if len(lines[1]) > 80 else ''}")
        if len(lines) > 2:
            print(f"    第三行: {lines[2][:80]}{'...' if len(lines[2]) > 80 else ''}")

def check_first_10_minutes(vtt_path):
    """专项检查前10分钟的重复内容"""
    print("🔍 正在检查前10分钟(600秒)的重复内容...")
    
    with open(vtt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    # 收集前10分钟的段落
    segments_first_10min = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            try:
                times = line.split('-->')
                start_str = times[0].strip()
                end_str = times[1].strip().split()[0] if ' ' in times[1] else times[1].strip()
                
                start_seconds = vtt_time_to_seconds(start_str)
                end_seconds = vtt_time_to_seconds(end_str)
                
                # 只收集前10分钟的段落
                if start_seconds < 600:
                    # 收集文本
                    text_lines = []
                    i += 1
                    while i < len(lines) and lines[i].strip():
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    text = ' '.join(text_lines).strip()
                    
                    if text:
                        segments_first_10min.append({
                            'start': start_seconds,
                            'end': end_seconds,
                            'text': text,
                            'text_short': text[:40] + "..." if len(text) > 40 else text,
                            'time_str': f"{start_str} --> {end_str}",
                            'duration': end_seconds - start_seconds
                        })
                    else:
                        i += 1
                else:
                    i += 1
                    # 跳过文本行
                    while i < len(lines) and lines[i].strip():
                        i += 1
            except Exception as e:
                print(f"  警告: 解析行时出错 '{line}': {e}")
                i += 1
        else:
            i += 1
    
    print(f"  在前10分钟找到 {len(segments_first_10min)} 个段落")
    
    if segments_first_10min:
        # 显示前10分钟的时间范围
        first_start = segments_first_10min[0]['start']
        last_end = segments_first_10min[-1]['end']
        print(f"  时间范围: {first_start:.1f}s - {last_end:.1f}s")
        print(f"  覆盖时长: {last_end - first_start:.1f}秒")
        
        # 检查重复
        text_dict = {}
        for seg in segments_first_10min:
            text_key = seg['text'].lower().strip()
            if len(text_key) > 5:  # 只检查有意义的文本（长度>5字符）
                if text_key not in text_dict:
                    text_dict[text_key] = []
                text_dict[text_key].append(seg)
        
        # 找出重复
        duplicates = {k: v for k, v in text_dict.items() if len(v) > 1}
        
        if duplicates:
            print(f"\n  ⚠️  在前10分钟发现 {len(duplicates)} 种重复文本:")
            print("  " + "-" * 70)
            
            # 按重复次数排序
            sorted_duplicates = sorted(duplicates.items(), key=lambda x: len(x[1]), reverse=True)
            
            for text_key, segs in sorted_duplicates[:10]:  # 只显示前10种最重复的
                if len(text_key) > 50:
                    display_text = text_key[:47] + "..."
                else:
                    display_text = text_key
                
                print(f"  重复文本: '{display_text}'")
                print(f"    重复次数: {len(segs)}")
                
                # 显示重复的时间点
                time_points = []
                for seg in segs[:5]:  # 最多显示5个时间点
                    time_points.append(f"{seg['start']:.1f}s")
                
                if len(segs) > 5:
                    time_points.append(f"... 还有 {len(segs)-5} 处")
                
                print(f"    出现时间: {', '.join(time_points)}")
                
                # 显示具体的段落信息
                for idx, seg in enumerate(segs[:3]):  # 最多显示前3个具体段落
                    print(f"      [{idx+1}] {seg['time_str']} - {seg['text_short']}")
                
                print("  " + "-" * 70)
            
            if len(duplicates) > 10:
                print(f"  ... 还有 {len(duplicates) - 10} 种重复文本未显示")
            
            # 统计重复段落的数量
            total_duplicate_segments = sum(len(segs) for segs in duplicates.values())
            print(f"\n  📊 重复统计:")
            print(f"    唯一文本数: {len(text_dict)}")
            print(f"    重复文本类型数: {len(duplicates)}")
            print(f"    重复段落总数: {total_duplicate_segments}")
            print(f"    重复率: {total_duplicate_segments/len(segments_first_10min)*100:.1f}%")
        else:
            print("  ✅ 前10分钟未发现重复文本")
        
        # 显示前10分钟的段落统计
        print(f"\n  📊 前10分钟段落统计:")
        print(f"    段落总数: {len(segments_first_10min)}")
        
        if segments_first_10min:
            avg_duration = sum(seg['duration'] for seg in segments_first_10min) / len(segments_first_10min)
            print(f"    平均段落时长: {avg_duration:.1f}秒")
            print(f"    最短段落: {min(seg['duration'] for seg in segments_first_10min):.1f}秒")
            print(f"    最长段落: {max(seg['duration'] for seg in segments_first_10min):.1f}秒")
            
            # 文本长度统计
            text_lengths = [len(seg['text']) for seg in segments_first_10min]
            avg_text_len = sum(text_lengths) / len(text_lengths)
            print(f"    平均文本长度: {avg_text_len:.0f}字符")
    else:
        print("  ⚠️  前10分钟没有找到任何段落")

def check_all_duplicates(vtt_path):
    """检查所有重复内容"""
    print("🔍 正在检查所有重复内容...")
    
    with open(vtt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.strip().split('\n')
    
    # 收集所有段落
    all_segments = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            try:
                times = line.split('-->')
                start_str = times[0].strip()
                end_str = times[1].strip().split()[0] if ' ' in times[1] else times[1].strip()
                
                start_seconds = vtt_time_to_seconds(start_str)
                end_seconds = vtt_time_to_seconds(end_str)
                
                # 收集文本
                text_lines = []
                i += 1
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                
                text = ' '.join(text_lines).strip()
                
                if text:
                    all_segments.append({
                        'start': start_seconds,
                        'end': end_seconds,
                        'text': text,
                        'text_short': text[:30] + "..." if len(text) > 30 else text
                    })
                else:
                    i += 1
            except Exception as e:
                print(f"  警告: 解析行时出错 '{line}': {e}")
                i += 1
        else:
            i += 1
    
    print(f"  总共找到 {len(all_segments)} 个段落")
    
    if all_segments:
        # 检查所有重复
        text_dict = {}
        for seg in all_segments:
            text_key = seg['text'].lower().strip()
            if len(text_key) > 3:  # 只检查有意义的文本
                if text_key not in text_dict:
                    text_dict[text_key] = []
                text_dict[text_key].append(seg)
        
        # 找出所有重复
        all_duplicates = {k: v for k, v in text_dict.items() if len(v) > 1}
        
        if all_duplicates:
            print(f"  ⚠️  总共发现 {len(all_duplicates)} 种重复文本")
            
            # 按重复次数排序
            sorted_duplicates = sorted(all_duplicates.items(), key=lambda x: len(x[1]), reverse=True)
            
            # 显示最重复的几种
            print(f"\n  🏆 重复最频繁的前5种文本:")
            for i, (text_key, segs) in enumerate(sorted_duplicates[:5]):
                if len(text_key) > 40:
                    display_text = text_key[:37] + "..."
                else:
                    display_text = text_key
                
                print(f"    {i+1}. '{display_text}' - 重复 {len(segs)} 次")
            
            # 统计
            total_segments = len(all_segments)
            duplicate_segments = sum(len(segs) for segs in all_duplicates.values())
            unique_texts = len(text_dict)
            
            print(f"\n  📊 总体统计:")
            print(f"    段落总数: {total_segments}")
            print(f"    唯一文本数: {unique_texts}")
            print(f"    重复文本类型数: {len(all_duplicates)}")
            print(f"    重复段落总数: {duplicate_segments}")
            print(f"    重复率: {duplicate_segments/total_segments*100:.1f}%")
            
            # 检查是否有明显的循环模式
            print(f"\n  🔄 检查重复模式:")
            if len(all_duplicates) > 0:
                # 检查最常见的重复是否在相近的时间出现
                most_common_text, most_common_segs = sorted_duplicates[0]
                times = [seg['start'] for seg in most_common_segs]
                times.sort()
                
                # 计算时间间隔
                intervals = []
                for j in range(1, len(times)):
                    intervals.append(times[j] - times[j-1])
                
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    print(f"    最常见的重复文本平均间隔: {avg_interval:.1f}秒")
                    
                    # 检查是否有规律的时间间隔
                    if len(intervals) >= 3:
                        variance = max(intervals) - min(intervals)
                        if variance < 10:  # 如果间隔变化很小
                            print(f"    ⚠️  发现可能的规律性重复，间隔约 {avg_interval:.1f} 秒")
        else:
            print("  ✅ 未发现任何重复文本")
    else:
        print("  ⚠️  没有找到任何段落")

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🎯 WHISPER.CPP 输出调试工具")
    print("=" * 80)
    
    # 检查环境
    if not os.path.exists(WHISPER_EXE):
        print(f"❌ whisper可执行文件不存在: {WHISPER_EXE}")
        return
    
    if not os.path.exists(MODEL):
        print(f"❌ 模型文件不存在: {MODEL}")
        return
    
    print("✅ 环境检查通过")
    print(f"  📍 Whisper路径: {WHISPER_EXE}")
    print(f"  📍 模型文件: {MODEL}")
    print(f"  📍 音频文件: {test_file}")
    
    # 运行调试
    debug_whisper_output()
    
    print("\n" + "=" * 80)
    print("✅ 调试完成")
    print("=" * 80)

if __name__ == "__main__":
    main()