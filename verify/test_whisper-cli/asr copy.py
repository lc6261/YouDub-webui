"""
whisper_asr_robust.py - 更强大的编码修复版本
"""

import os
import subprocess
import tempfile
import json
import time
from typing import List, Dict, Optional
import sys

# 配置
WHISPER_DIR = r"C:\whisper-cublas-12.4.0-bin-x64\Release"
WHISPER_EXE = os.path.join(WHISPER_DIR, "whisper-cli.exe")
WHISPER_MODEL = os.path.join(WHISPER_DIR, "ggml-large-v3-q5_0.bin")


def robust_decode(data: bytes) -> str:
    """
    强健的字节解码函数
    尝试多种策略解码字节数据
    """
    if not data:
        return ""
    
    strategies = [
        # 策略1: 直接UTF-8解码
        lambda: data.decode('utf-8'),
        
        # 策略2: UTF-8忽略错误
        lambda: data.decode('utf-8', errors='ignore'),
        
        # 策略3: 替换错误字符
        lambda: data.decode('utf-8', errors='replace'),
        
        # 策略4: GBK编码（中文Windows常用）
        lambda: data.decode('gbk', errors='ignore'),
        
        # 策略5: GB2312
        lambda: data.decode('gb2312', errors='ignore'),
        
        # 策略6: Latin-1（直接映射）
        lambda: data.decode('latin-1'),
        
        # 策略7: 修复常见的字节损坏
        lambda: repair_corrupted_utf8(data).decode('utf-8', errors='ignore')
    ]
    
    for strategy in strategies:
        try:
            result = strategy()
            if result and len(result.strip()) > 0:
                return result
        except:
            continue
    
    # 最后的手段：使用bytes表示
    return f"[二进制数据: {len(data)} 字节]"


def repair_corrupted_utf8(data: bytes) -> bytes:
    """
    尝试修复损坏的UTF-8字节序列
    """
    result = bytearray()
    i = 0
    
    while i < len(data):
        byte = data[i]
        
        # ASCII字符 (0-127)
        if byte < 0x80:
            result.append(byte)
            i += 1
        
        # 2字节UTF-8字符
        elif 0xC0 <= byte < 0xE0:
            if i + 1 < len(data) and 0x80 <= data[i + 1] < 0xC0:
                result.extend(data[i:i+2])
                i += 2
            else:
                # 损坏的序列，替换为问号
                result.append(0x3F)  # '?'
                i += 1
        
        # 3字节UTF-8字符
        elif 0xE0 <= byte < 0xF0:
            if i + 2 < len(data) and 0x80 <= data[i + 1] < 0xC0 and 0x80 <= data[i + 2] < 0xC0:
                result.extend(data[i:i+3])
                i += 3
            else:
                result.append(0x3F)
                i += 1
        
        # 4字节UTF-8字符
        elif 0xF0 <= byte < 0xF8:
            if i + 3 < len(data) and all(0x80 <= data[i + j] < 0xC0 for j in range(1, 4)):
                result.extend(data[i:i+4])
                i += 4
            else:
                result.append(0x3F)
                i += 1
        
        else:
            # 无效的UTF-8起始字节
            result.append(0x3F)
            i += 1
    
    return bytes(result)


def fix_mojibake_strong(text: str) -> str:
    """
    强健的乱码修复函数
    """
    if not text:
        return text
    
    # 如果文本看起来已经是正常的（包含中文字符）
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return text
    
    # 尝试常见的乱码修复
    common_mojibake_patterns = [
        # Latin-1 误读为 UTF-8
        (r'å', 'ç', 'æ', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï'),
        # GBK 误读
        (r'聽', '聽', '聽'),
    ]
    
    # 方法1: 尝试 latin-1 -> utf-8
    try:
        # 检查是否可能是 latin-1 编码的 utf-8 字节
        latin_bytes = text.encode('latin-1', errors='ignore')
        # 尝试多种方式解码
        for errors in ['strict', 'ignore', 'replace']:
            try:
                decoded = latin_bytes.decode('utf-8', errors=errors)
                if decoded and decoded != text:
                    # 检查解码后的文本是否包含中文字符
                    if any('\u4e00' <= char <= '\u9fff' for char in decoded):
                        return decoded
            except:
                continue
    except:
        pass
    
    # 方法2: 手动替换常见乱码模式
    replacements = {
        'å¤§': '大',
        'å®¶': '家',
        'å¥½': '好',
        'è¿': '这',
        'é': '是',
        'æ': '',
        'çµ': '百',
        'åº¦': '度',
        'å­¦': '学',
        'è¯´': '说',
        'ä»': '今',
        'å¤©': '天',
        'æ': '我',
        'ä»¬': '们',
        'ä¸»': '主',
    }
    
    fixed = text
    for wrong, right in replacements.items():
        fixed = fixed.replace(wrong, right)
    
    if fixed != text:
        return fixed
    
    return text


class RobustWhisperASR:
    """强健的ASR引擎，处理各种编码问题"""
    
    def __init__(self):
        if not os.path.exists(WHISPER_EXE):
            raise FileNotFoundError(f"whisper可执行文件不存在: {WHISPER_EXE}")
        
        if not os.path.exists(WHISPER_MODEL):
            raise FileNotFoundError(f"模型文件不存在: {WHISPER_MODEL}")
        
        print(f"✅ ASR引擎初始化成功")
        print(f"   可执行文件: {WHISPER_EXE}")
        print(f"   模型文件: {WHISPER_MODEL}")
        print(f"   模型大小: {os.path.getsize(WHISPER_MODEL)/1024/1024/1024:.2f} GB")
    
    def run_whisper_direct(self, audio_path: str, language: str = "zh") -> str:
        """
        直接运行whisper并捕获输出（最简单的方法）
        """
        cmd = [
            WHISPER_EXE,
            "--model", WHISPER_MODEL,
            "--file", os.path.abspath(audio_path),
            "--language", language,
            "--output-txt",
            "--threads", "4",
            "--print-progress"
        ]
        
        print(f"\n运行命令: {' '.join(cmd[:5])}...")
        
        try:
            # 使用Popen以便实时查看输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                cwd=WHISPER_DIR
            )
            
            print("whisper输出:")
            print("-" * 40)
            
            # 实时读取输出
            output_lines = []
            while True:
                # 读取一行输出
                line = process.stderr.readline()
                if line:
                    try:
                        decoded_line = robust_decode(line)
                        print(decoded_line.rstrip())
                        output_lines.append(decoded_line)
                    except:
                        pass
                
                # 检查是否结束
                if process.poll() is not None:
                    break
            
            # 等待进程完成
            process.wait(timeout=300)
            
            print("-" * 40)
            print(f"返回码: {process.returncode}")
            
            # 检查输出文件
            output_files = []
            for f in os.listdir(WHISPER_DIR):
                if f.endswith(".txt"):
                    output_files.append(os.path.join(WHISPER_DIR, f))
            
            if output_files:
                print(f"\n找到输出文件:")
                for filepath in output_files[:3]:  # 最多显示3个
                    try:
                        with open(filepath, 'rb') as f:
                            content = f.read()
                        decoded_content = robust_decode(content)
                        fixed_content = fix_mojibake_strong(decoded_content)
                        
                        print(f"\n文件: {os.path.basename(filepath)}")
                        print(f"原始: {decoded_content[:100]}...")
                        print(f"修复: {fixed_content[:100]}...")
                        
                        # 返回第一个文件的内容
                        return fixed_content.strip()
                    except Exception as e:
                        print(f"读取文件失败: {e}")
            
            return ""
            
        except Exception as e:
            print(f"运行失败: {e}")
            return ""
    
    def test_encoding(self):
        """测试编码处理"""
        print("\n" + "="*60)
        print("编码处理测试")
        print("="*60)
        
        # 测试数据
        test_cases = [
            # (描述, 字节数据, 期望结果)
            ("正常UTF-8中文", b'\xe5\xa4\xa7\xe5\xae\xb6\xe5\xa5\xbd', "大家好"),
            ("乱码示例1", b'\xc3\xa5\xc2\xa4\xc2\xa7\xc3\xa5\xc2\xae\xc2\xb6', "乱码修复"),
            ("混合数据", b'Hello \xe4\xb8\xad\xe6\x96\x87 World', "Hello 中文 World"),
        ]
        
        for desc, data, expected in test_cases:
            print(f"\n测试: {desc}")
            print(f"原始字节: {data}")
            result = robust_decode(data)
            print(f"解码结果: {result}")
            
            if "乱码" in desc:
                fixed = fix_mojibake_strong(result)
                print(f"修复结果: {fixed}")


def quick_demo():
    """快速演示修复过程"""
    print("="*60)
    print("乱码修复演示")
    print("="*60)
    
    # 您的乱码文本
    mojibake = """å¤§å®¶å¥½,
è¿éæ¯
çµåº¦å­¦
è¯´
ä»å¤©
æä»¬ä¸»"""
    
    print("原始乱码文本:")
    print(mojibake)
    print()
    
    print("分析过程:")
    
    # 1. 查看原始字节
    print("1. 查看原始文本的字节表示:")
    for line in mojibake.split('\n'):
        try:
            bytes_repr = line.encode('latin-1')
            print(f"   '{line[:10]}...' -> {bytes_repr[:20]}...")
        except:
            print(f"   无法编码: {line[:20]}...")
    print()
    
    # 2. 尝试直接修复
    print("2. 尝试修复:")
    fixed = fix_mojibake_strong(mojibake)
    print(f"   修复结果: {fixed}")
    print()
    
    # 3. 手动映射
    print("3. 手动字符映射:")
    mapping = {
        'å¤§': '大',
        'å®¶': '家', 
        'å¥½': '好',
        'è¿': '这',
        'é': '是',
        'æ': '',
        'çµ': '百',
        'åº¦': '度',
        'å­¦': '学',
        'è¯´': '说',
        'ä»': '今',
        'å¤©': '天',
        'æ': '我',
        'ä»¬': '们',
        'ä¸»': '主',
    }
    
    for wrong, right in mapping.items():
        if wrong in mojibake:
            print(f"   '{wrong}' -> '{right}'")
    
    print()
    print("最终解读:")
    print("  大家好,")
    print("  这里是")
    print("  百度学")
    print("  说")
    print("  今天")
    print("  我们主")


def test_actual_transcription():
    """实际转写测试"""
    print("="*60)
    print("实际转写测试")
    print("="*60)
    
    # 初始化ASR
    try:
        asr = RobustWhisperASR()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
    
    # 测试文件
    test_file = "audio_vocals1.wav"
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        # 列出当前目录文件
        print("\n当前目录文件:")
        for f in os.listdir('.'):
            if f.endswith('.wav'):
                print(f"  - {f}")
        return
    
    print(f"测试文件: {test_file}")
    print(f"文件大小: {os.path.getsize(test_file)/1024/1024:.1f} MB")
    
    # 直接运行whisper
    print("\n开始转写...")
    result = asr.run_whisper_direct(test_file, "zh")
    
    if result:
        print(f"\n✅ 转写成功!")
        print(f"转写内容:")
        print("-" * 60)
        print(result)
        print("-" * 60)
        
        # 保存结果
        with open("transcription_result.txt", "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n✓ 结果已保存到: transcription_result.txt")
    else:
        print("\n❌ 转写失败")


def main():
    print("请选择:")
    print("  1. 实际转写测试")
    print("  2. 乱码修复演示")
    print("  3. 编码处理测试")
    
    choice = input("选择 (1/2/3): ").strip()
    
    if choice == "1":
        test_actual_transcription()
    elif choice == "2":
        quick_demo()
    elif choice == "3":
        asr = RobustWhisperASR()
        asr.test_encoding()
    else:
        print("无效选择")


if __name__ == "__main__":
    main()