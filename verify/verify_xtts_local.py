# verify_xtts_local_fixed.py
import os
import sys
import torch
import warnings
import time
import librosa
import numpy as np
from datetime import datetime

# 创建验证目录
VERIFY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "验证结果")
os.makedirs(VERIFY_DIR, exist_ok=True)

# 设置日志文件
log_file = os.path.join(VERIFY_DIR, f"验证日志_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
sys.stdout = open(log_file, 'w', encoding='utf-8')
sys.stderr = sys.stdout

# 过滤警告
warnings.filterwarnings("ignore")

print("=" * 80)
print("🎯 XTTS v2 本地验证测试（修复版）")
print(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# === 设置模型存储路径 ===
MODEL_HOME = r"C:\model"
os.makedirs(MODEL_HOME, exist_ok=True)
os.environ["TTS_HOME"] = MODEL_HOME
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"\n📂 TTS 模型目录: {MODEL_HOME}")
print(f"📂 验证结果目录: {VERIFY_DIR}")
print(f"🧠 CUDA 可用: {torch.cuda.is_available()}")
print(f"🏋️‍♂️ GPU 型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# === 创建测试结果目录 ===
TEST_RESULTS_DIR = os.path.join(VERIFY_DIR, "测试音频")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# === 音频保存函数（兼容所有版本） ===
def save_audio(output_path, wav, sr, normalize=True):
    """通用的音频保存函数"""
    try:
        # 1. 首先尝试 soundfile（最佳选择）
        try:
            import soundfile as sf
            if normalize:
                # 确保音频在 [-1, 1] 范围内
                if np.max(np.abs(wav)) > 1.0:
                    wav = wav / np.max(np.abs(wav))
            sf.write(output_path, wav, sr, subtype='PCM_16')
            return True, "soundfile"
        except ImportError:
            pass
        
        # 2. 尝试 scipy
        try:
            from scipy.io import wavfile
            # 标准化到16位整数范围
            wav_normalized = np.int16(wav * 32767)
            wavfile.write(output_path, sr, wav_normalized)
            return True, "scipy"
        except ImportError:
            pass
        
        # 3. 使用 wave 库（内置，无需安装）
        import wave
        # 确保音频是单声道
        if len(wav.shape) > 1:
            wav = wav[:, 0]  # 如果是立体声，取左声道
        
        # 标准化到16位
        wav_normalized = np.int16(wav * 32767)
        
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16位 = 2字节
            wf.setframerate(sr)
            wf.writeframes(wav_normalized.tobytes())
        
        return True, "wave"
        
    except Exception as e:
        print(f"    保存音频失败: {e}")
        return False, str(e)

# === 优化语速的后处理函数 ===
def speed_up_audio(wav_path, speed_factor=1.3):
    """加速音频文件"""
    try:
        print(f"    正在加速音频 {speed_factor}x...")
        
        # 读取音频
        wav, sr = librosa.load(wav_path, sr=24000)
        original_duration = len(wav) / sr
        
        # 选择加速方法
        wav_fast = None
        method = ""
        
        # 方法1：尝试使用 pyrubberband（质量最好）
        try:
            import pyrubberband as pyrb
            wav_fast = pyrb.time_stretch(wav, sr, speed_factor)
            method = "pyrubberband"
            print(f"      使用 pyrubberband 时间拉伸")
        except ImportError:
            print(f"      pyrubberband 未安装，尝试其他方法")
        except Exception as e:
            print(f"      pyrubberband 错误: {e}")
        
        # 方法2：使用 librosa 的时间拉伸
        if wav_fast is None:
            try:
                wav_fast = librosa.effects.time_stretch(wav, rate=speed_factor)
                method = "librosa"
                print(f"      使用 librosa 时间拉伸")
            except Exception as e:
                print(f"      librosa 时间拉伸错误: {e}")
        
        # 方法3：简单的重采样（最后的选择）
        if wav_fast is None:
            try:
                # 通过改变采样率来模拟加速
                new_length = int(len(wav) / speed_factor)
                wav_fast = signal.resample(wav, new_length)
                method = "resample"
                print(f"      使用重采样方法")
            except Exception as e:
                print(f"      重采样错误: {e}")
                return wav_path, None, None
        
        # 保存加速后的音频
        base_name = os.path.basename(wav_path).replace('.wav', '')
        output_path = os.path.join(TEST_RESULTS_DIR, f"{base_name}_加速{speed_factor}x.wav")
        
        success, save_method = save_audio(output_path, wav_fast, sr)
        
        if success:
            fast_duration = len(wav_fast) / sr
            print(f"      ✅ 加速成功！")
            print(f"      原时长: {original_duration:.2f}s → 加速后: {fast_duration:.2f}s")
            print(f"      加速方法: {method}, 保存方法: {save_method}")
            return output_path, wav_fast, sr
        else:
            print(f"      ❌ 保存失败")
            return wav_path, None, None
        
    except Exception as e:
        print(f"    ⚠️ 音频加速失败: {e}")
        import traceback
        traceback.print_exc()
        return wav_path, None, None

# === 检查音频处理库 ===
def check_audio_libraries():
    """检查所需的音频处理库"""
    print("\n📦 检查音频处理库:")
    
    libraries = {
        'librosa': False,
        'soundfile': False,
        'pyrubberband': False,
        'scipy': False,
    }
    
    try:
        import librosa
        libraries['librosa'] = True
        print(f"  ✅ librosa {librosa.__version__}")
    except:
        print(f"  ❌ librosa 未安装")
    
    try:
        import soundfile as sf
        libraries['soundfile'] = True
        print(f"  ✅ soundfile")
    except:
        print(f"  ❌ soundfile 未安装")
    
    try:
        import pyrubberband
        libraries['pyrubberband'] = True
        print(f"  ✅ pyrubberband")
    except:
        print(f"  ❌ pyrubberband 未安装")
    
    try:
        import scipy
        libraries['scipy'] = True
        print(f"  ✅ scipy {scipy.__version__}")
    except:
        print(f"  ❌ scipy 未安装")
    
    return libraries

# 检查库
libs = check_audio_libraries()

# 安装建议
if not libs['soundfile']:
    print(f"\n💡 建议安装 soundfile: pip install soundfile")
if not libs['pyrubberband']:
    print(f"💡 建议安装 pyrubberband: pip install pyrubberband")

# === 加载 TTS ===
print("\n" + "=" * 80)
print("🚀 正在加载 XTTS v2 模型...")
print("=" * 80)

try:
    from TTS.api import TTS
    
    load_start = time.time()
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=True,
        gpu=torch.cuda.is_available()
    )
    load_time = time.time() - load_start
    
    # 确保使用 GPU
    if torch.cuda.is_available():
        tts = tts.to("cuda:0")
    
    print(f"✅ 模型加载成功！耗时: {load_time:.2f}秒")
    
    print("\n📋 模型信息:")
    print(f"  - 加载时间: {load_time:.2f}秒")
    print(f"  - 设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  - 支持语言: {tts.languages}")
    
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# === 检查参考音频 ===
TEST_REF_WAV = r"videos\卢克文工作室\20251229 美国斩杀线真相 政府资本合谋 底层毫无反抗之力\audio.wav"

if os.path.exists(TEST_REF_WAV):
    print(f"\n✅ 找到参考音频: {TEST_REF_WAV}")
    # 分析参考音频
    try:
        ref_wav, ref_sr = librosa.load(TEST_REF_WAV, sr=24000)
        ref_duration = len(ref_wav) / ref_sr
        print(f"   📊 参考音频时长: {ref_duration:.2f}秒")
        print(f"   🎵 采样率: {ref_sr} Hz")
    except Exception as e:
        print(f"   ⚠️ 无法读取参考音频: {e}")
else:
    print(f"\n⚠️ 未找到参考音频: {TEST_REF_WAV}")
    TEST_REF_WAV = None
    print("   ℹ️ 将使用模型默认声音")

# === 测试不同参数组合 ===
test_cases = [
    {"text": "你好！这是本地语音合成测试，完全离线运行。", "speed": 1.0, "name": "默认语速"},
    {"text": "你好！这是本地语音合成测试，完全离线运行。", "speed": 1.2, "name": "1.2倍语速"},
    {"text": "你好！这是本地语音合成测试，完全离线运行。", "speed": 1.5, "name": "1.5倍语速"},
    {"text": "语音合成速度已经优化，听起来更自然了。", "speed": 1.3, "temperature": 0.7, "name": "优化参数"},
    {"text": "这是一个关于人工智能和语音技术的演示。", "speed": 1.4, "temperature": 0.8, "name": "快速模式"},
]

print("\n" + "=" * 80)
print("🔬 测试不同参数配置")
print("=" * 80)

results = []

for i, case in enumerate(test_cases):
    print(f"\n🧪 测试 {i+1}/{len(test_cases)}: {case['name']}")
    print(f"📝 文本: {case['text']}")
    
    # 准备输出文件路径
    output_file = os.path.join(TEST_RESULTS_DIR, f"test_case_{i+1}.wav")
    params = {
        "text": case["text"],
        "file_path": output_file,
        "language": "zh",
        "split_sentences": True,
    }
    
    # 添加可选参数
    if TEST_REF_WAV:
        params["speaker_wav"] = TEST_REF_WAV
    
    if "speed" in case:
        params["speed"] = case["speed"]
    
    if "temperature" in case:
        params["temperature"] = case["temperature"]
    
    # 生成语音
    start_time = time.time()
    try:
        print(f"    🎙️ 正在生成语音...")
        tts.tts_to_file(**params)
        gen_time = time.time() - start_time
        
        # 读取并分析音频
        wav, sr = librosa.load(output_file, sr=24000)
        duration = len(wav) / sr
        
        # 计算指标
        char_count = len(case["text"].replace(' ', ''))  # 中文字符数
        char_per_sec = char_count / duration
        real_time_factor = gen_time / duration
        
        print(f"    ✅ 生成成功！")
        print(f"    ⏱️  生成时间: {gen_time:.2f}秒")
        print(f"    🎵 音频时长: {duration:.2f}秒")
        print(f"    📏 文字数量: {char_count}字")
        print(f"    🚀 语速: {char_per_sec:.2f}字/秒")
        print(f"    ⚡ 实时因子: {real_time_factor:.2f}")
        
        # 保存结果
        result = {
            "case_name": case["name"],
            "text": case["text"],
            "speed_param": case.get("speed", 1.0),
            "gen_time": gen_time,
            "duration": duration,
            "char_count": char_count,
            "char_per_sec": char_per_sec,
            "real_time_factor": real_time_factor,
            "output_file": output_file
        }
        
        # 检查是否真的加速了
        if case.get('speed', 1.0) > 1.1 and char_per_sec < 5:
            print(f"    ⚠️  语速偏慢 ({char_per_sec:.2f}字/秒)，尝试后处理加速...")
            fast_path, fast_wav, fast_sr = speed_up_audio(output_file, case['speed'])
            if fast_wav is not None:
                fast_duration = len(fast_wav) / fast_sr
                fast_char_per_sec = char_count / fast_duration
                result["fast_file"] = fast_path
                result["fast_duration"] = fast_duration
                result["fast_char_per_sec"] = fast_char_per_sec
                print(f"    ⚡ 加速后: {fast_char_per_sec:.2f}字/秒")
        
        results.append(result)
        
    except Exception as e:
        print(f"    ❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()

# === 批量测试 ===
print("\n" + "=" * 80)
print("📊 批量性能测试")
print("=" * 80)

batch_texts = [
    "你好",
    "你好世界",
    "欢迎使用本地语音合成系统",
    "这是一个测试语音合成的示例文本，用于评估系统性能。",
]

batch_results = []

for i, text in enumerate(batch_texts):
    output_file = os.path.join(TEST_RESULTS_DIR, f"batch_{i+1}.wav")
    
    print(f"\n📝 文本长度 {len(text)}: {text}")
    
    start_time = time.time()
    try:
        tts.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=TEST_REF_WAV,
            language="zh",
            speed=1.5,
            temperature=0.7,
            split_sentences=False,
        )
        gen_time = time.time() - start_time
        
        wav, sr = librosa.load(output_file, sr=24000)
        duration = len(wav) / sr
        
        char_count = len(text.replace(' ', ''))
        char_per_sec = char_count / duration
        real_time_factor = gen_time / duration
        
        print(f"    ✅ 生成成功")
        print(f"    ⏱️  处理: {gen_time:.2f}秒, 播放: {duration:.2f}秒")
        print(f"    🚀 实时因子: {real_time_factor:.2f}")
        print(f"    🎯 语速: {char_per_sec:.2f}字/秒")
        
        batch_results.append({
            "text": text,
            "length": len(text),
            "gen_time": gen_time,
            "duration": duration,
            "real_time_factor": real_time_factor,
            "char_per_sec": char_per_sec
        })
        
    except Exception as e:
        print(f"    ❌ 失败: {e}")

# === 生成总结报告 ===
print("\n" + "=" * 80)
print("📋 验证测试总结报告")
print("=" * 80)

# 保存详细报告
report_file = os.path.join(VERIFY_DIR, "详细验证报告.txt")
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("🎯 XTTS v2 本地验证测试报告（修复版）\n")
    f.write(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("📊 系统配置:\n")
    f.write(f"  - 模型目录: {MODEL_HOME}\n")
    f.write(f"  - CUDA 可用: {torch.cuda.is_available()}\n")
    f.write(f"  - GPU 型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}\n")
    f.write(f"  - 模型加载时间: {load_time:.2f}秒\n\n")
    
    f.write("📦 音频库状态:\n")
    for lib_name, status in libs.items():
        f.write(f"  - {lib_name}: {'✅' if status else '❌'}\n")
    f.write("\n")
    
    f.write("📊 参数测试结果:\n")
    for result in results:
        f.write(f"\n【{result['case_name']}】\n")
        f.write(f"  文本: {result['text']}\n")
        f.write(f"  参数语速: {result['speed_param']}x\n")
        f.write(f"  生成时间: {result['gen_time']:.2f}秒\n")
        f.write(f"  音频时长: {result['duration']:.2f}秒\n")
        f.write(f"  语速: {result['char_per_sec']:.2f}字/秒\n")
        f.write(f"  实时因子: {result['real_time_factor']:.2f}\n")
        if 'fast_char_per_sec' in result:
            f.write(f"  ⚡ 加速后语速: {result['fast_char_per_sec']:.2f}字/秒\n")
    
    f.write("\n📊 批量测试结果:\n")
    for br in batch_results:
        f.write(f"\n  文本长度 {br['length']}: {br['text']}\n")
        f.write(f"  处理时间: {br['gen_time']:.2f}秒\n")
        f.write(f"  实时因子: {br['real_time_factor']:.2f}\n")
        f.write(f"  语速: {br['char_per_sec']:.2f}字/秒\n")

# === 安装所需库的指令 ===
print("\n" + "=" * 80)
print("📦 所需库安装指令")
print("=" * 80)

print("""
为了正常使用音频加速功能，请安装以下库：

1. 基础音频处理（必需）:
   pip install soundfile numpy

2. 高质量时间拉伸（推荐）:
   pip install pyrubberband
   
   注意: pyrubberband 在Windows上可能需要额外步骤:
   - 先安装: pip install numpy
   - 再安装: pip install pyrubberband
   
3. 科学计算支持:
   pip install scipy
   
4. 完整安装指令:
   pip install librosa soundfile pyrubberband scipy numpy
""")

# === 结论和建议 ===
print("\n📋 **测试结果统计**:")
print(f"   参数测试用例: {len(results)} 个")
print(f"   批量测试用例: {len(batch_results)} 个")
print(f"   生成音频文件: {len([f for f in os.listdir(TEST_RESULTS_DIR) if f.endswith('.wav')])} 个")

print("\n🎯 **关键发现**:")
if results:
    avg_char_per_sec = np.mean([r['char_per_sec'] for r in results])
    avg_real_time_factor = np.mean([r['real_time_factor'] for r in results])
    print(f"   1. 平均语速: {avg_char_per_sec:.2f} 字/秒")
    print(f"   2. 平均实时因子: {avg_real_time_factor:.2f}")
    print(f"   3. speed参数效果: {'有限' if avg_char_per_sec < 5 else '明显'}")
    
    # 音频库状态
    print(f"   4. 音频库状态:")
    for lib_name, status in libs.items():
        print(f"      - {lib_name}: {'✅ 已安装' if status else '❌ 未安装'}")

print("\n💡 **优化建议**:")
print("   1. ✅ 安装 soundfile 和 pyrubberband")
print("   2. ✅ 使用后处理加速（修复版已可用）")
print("   3. ✅ 设置 speed=1.5 + temperature=0.7")
print("   4. ✅ 批量处理提高效率")

print("\n🔧 **优化后的配置代码**:")
print("""
import soundfile as sf
import pyrubberband as pyrb

def optimized_tts_generation(tts, text, output_path, ref_wav=None, target_speed=1.3):
    \"\"\"优化后的TTS生成函数\"\"\"
    # 1. 生成参数
    params = {
        'text': text,
        'file_path': output_path,
        'language': 'zh',
        'split_sentences': len(text) > 20,
        'speed': 1.5,
        'temperature': 0.7,
        'speaker_wav': ref_wav
    }
    
    # 2. 生成语音
    tts.tts_to_file(**params)
    
    # 3. 后处理加速
    if len(text) > 10:
        wav, sr = librosa.load(output_path, sr=24000)
        
        # 使用 pyrubberband 高质量时间拉伸
        wav_fast = pyrb.time_stretch(wav, sr, target_speed)
        
        # 使用 soundfile 保存
        sf.write(output_path, wav_fast, sr)
""")

print("\n" + "=" * 80)
print("✅ **验证完成**:")
print(f"   验证日志: {log_file}")
print(f"   详细报告: {report_file}")
print(f"   测试音频: {TEST_RESULTS_DIR}")
print(f"   总计用时: {time.time() - load_start:.2f}秒")
print("=" * 80)

# 恢复标准输出
sys.stdout.close()
sys.stdout = sys.__stdout__

print(f"\n🎉 验证测试已完成！")
print(f"📁 所有结果已保存到: {VERIFY_DIR}")
print(f"📝 详细报告: {report_file}")
print(f"🔊 测试音频: {TEST_RESULTS_DIR}")
print(f"\n💡 请先安装所需库:")
print(f"   pip install soundfile pyrubberband")