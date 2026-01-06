#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XTTS v2 本地 TTS 封装模块（适配验证测试结论）

功能：
- 自动加载 XTTS v2 模型（仅一次，避免重复加载）
- 支持参考音频克隆（音色迁移）
- 使用验证推荐参数：speed=1.5, temperature=0.7
- 后处理加速（使用 librosa，兼容 Windows 无需 rubberband-cli）
- 自动跳过已存在的输出文件
- GPU 自动检测（优先使用 CUDA）
- 支持本地模型路径指定

作者：根据验证日志优化
日期：2026-01-02
"""

import os
import time
import shutil
import numpy as np
import torch
from loguru import logger
from pathlib import Path

# 可选：仅在需要后处理加速时导入
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("librosa 未安装，将无法使用后处理加速功能")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    logger.warning("soundfile 未安装，将使用备用保存方式")

# 全局模型变量
_model = None

# 可能的模型存储路径（按优先级排序）
POSSIBLE_MODEL_PATHS = [
    r"C:\model\tts\tts_models--multilingual--multi-dataset--xtts_v2",  # 你实际下载的位置
    os.path.expanduser(r"~\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2"),
    os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"),
    os.path.expanduser("~/.tts/tts_models--multilingual--multi-dataset--xtts_v2"),
]


def find_model_path():
    """查找本地已存在的模型路径"""
    for path in POSSIBLE_MODEL_PATHS:
        if os.path.exists(path):
            logger.info(f"Found model at: {path}")
            return path
    return None


def copy_model_to_cache(src_path, dst_path=None):
    """将模型复制到 TTS 默认缓存位置"""
    if dst_path is None:
        dst_path = os.path.expanduser(r"~\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2")
    
    if os.path.exists(dst_path):
        logger.info(f"Cache path already exists: {dst_path}")
        return dst_path
    
    try:
        logger.info(f"Copying model from {src_path} to {dst_path}")
        shutil.copytree(src_path, dst_path)
        logger.info(f"Model copied successfully to {dst_path}")
        return dst_path
    except Exception as e:
        logger.error(f"Failed to copy model: {e}")
        return src_path


def init_TTS(model_path=None):
    """初始化 TTS 模型（懒加载）"""
    load_model(model_path)


def load_model(model_path=None, device='auto'):
    """
    加载 XTTS v2 模型（单例模式）
    
    优先使用本地模型，避免网络下载

    Args:
        model_path (str): 模型标识或本地路径，如果为None则自动查找
        device (str): 设备类型 ('auto', 'cuda', 'cpu')
    """
    global _model
    if _model is not None:
        return

    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        device = device

    # 如果未指定模型路径，尝试查找本地模型
    if model_path is None:
        local_path = find_model_path()
        if local_path:
            # 尝试复制到默认缓存位置，避免下载
            cache_path = copy_model_to_cache(local_path)
            model_path = cache_path
        else:
            model_path = "tts_models/multilingual/multi-dataset/xtts_v2"
            logger.warning("No local model found, will try to download (may fail due to network)")

    logger.info(f"Loading TTS model from {model_path} on {device}")

    try:
        from TTS.api import TTS
        
        # 如果是本地路径，确保路径存在
        if os.path.exists(model_path):
            logger.info(f"Using local model at: {model_path}")
            t_start = time.time()
            
            # 方法1：直接使用本地路径
            try:
                _model = TTS(model_path=model_path, progress_bar=False).to(device)
            except:
                # 方法2：尝试使用模型名称但设置本地缓存
                os.environ['TTS_HOME'] = os.path.dirname(os.path.dirname(model_path))
                _model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", 
                            progress_bar=False).to(device)
        else:
            # 尝试下载模型（可能会失败）
            t_start = time.time()
            _model = TTS(model_name=model_path, progress_bar=False).to(device)
        
        t_end = time.time()
        logger.info(f"✅ TTS model loaded in {t_end - t_start:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        
        # 提供详细的错误解决建议
        if "ConnectionError" in str(e) or "下载" in str(e):
            logger.error("\n" + "="*60)
            logger.error("网络连接问题！请按以下步骤操作：")
            logger.error("1. 确认模型文件已下载到以下位置之一：")
            for path in POSSIBLE_MODEL_PATHS:
                logger.error(f"   - {path}")
            logger.error("2. 或者手动指定模型路径：")
            logger.error("   tts(text, output_path, model_path='C:/model/tts/tts_models--multilingual--multi-dataset--xtts_v2')")
            logger.error("3. 确保模型目录包含以下文件：")
            logger.error("   - config.json")
            logger.error("   - model.pth")
            logger.error("   - vocab.json (如果有的话)")
            logger.error("="*60)
        raise


def _apply_post_speedup(wav_path: str, speed_factor: float = 1.3) -> None:
    """
    对已生成的音频进行后处理加速（使用 librosa）

    Args:
        wav_path (str): 音频文件路径
        speed_factor (float): 加速倍数（>1 加速，<1 减速）
    """
    if not HAS_LIBROSA:
        logger.warning("librosa 未安装，跳过后处理加速")
        return

    try:
        logger.debug(f"正在对 {wav_path} 应用后处理加速 x{speed_factor}...")
        wav, sr = librosa.load(wav_path, sr=24000)

        # 使用 librosa 时间拉伸（无需 rubberband-cli）
        wav_stretched = librosa.effects.time_stretch(wav, rate=speed_factor)

        # 优先使用 soundfile 保存（兼容性好）
        if HAS_SOUNDFILE:
            sf.write(wav_path, wav_stretched, sr)
        else:
            # 转为 int16
            wav_int16 = np.clip(wav_stretched, -1.0, 1.0)  # 防溢出
            wav_int16 = (wav_int16 * 32767).astype(np.int16)
            import scipy.io.wavfile
            scipy.io.wavfile.write(wav_path, sr, wav_int16)

        logger.debug("✅ 后处理加速完成")
    except Exception as e:
        logger.error(f"后处理加速失败: {e}")


def tts(
    text: str,
    output_path: str,
    speaker_wav: str = None,
    model_path: str = None,  # 改为 model_path，优先使用本地路径
    device: str = 'auto',
    language: str = 'zh',  # 注意：XTTS v2 使用 'zh'，非 'zh-cn'
    speed: float = 1.5,
    temperature: float = 0.7,
    split_sentences: bool = True,
    enable_post_speedup: bool = False,
    post_speed_factor: float = 1.3,
):
    """
    文本转语音主函数

    Args:
        text (str): 待合成文本
        output_path (str): 输出音频路径（.wav）
        speaker_wav (str, optional): 参考音频路径，用于音色克隆
        model_path (str): 本地模型路径（优先使用），如果为None则自动查找
        device (str): 设备 ('auto', 'cuda', 'cpu')
        language (str): 语言代码（'zh' 表示中文）
        speed (float): XTTS 内置语速（1.0~2.0，效果有限）
        temperature (float): 生成随机性（0.5~0.8，越低越稳定）
        split_sentences (bool): 是否自动分句（推荐 True）
        enable_post_speedup (bool): 是否启用后处理加速
        post_speed_factor (float): 后处理加速倍数（1.3~1.5 推荐）
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 跳过已存在文件
    if os.path.exists(output_path):
        logger.info(f"TTS output already exists: {output_path}")
        return

    # 懒加载模型
    global _model
    if _model is None:
        load_model(model_path, device)

    # 检查参考音频
    if speaker_wav and not os.path.exists(speaker_wav):
        logger.warning(f"Reference audio not found: {speaker_wav}, using default voice")
        speaker_wav = None

    # 构建参数
    params = {
        "text": text,
        "file_path": output_path,
        "language": language,
        "speed": speed,
        "temperature": temperature,
        "split_sentences": split_sentences,
    }
    if speaker_wav:
        params["speaker_wav"] = speaker_wav

    # 生成语音
    for retry in range(3):
        try:
            logger.info(f"Generating TTS for: {text[:50]}...")
            _model.tts_to_file(**params)
            logger.info(f"✅ TTS saved to: {output_path}")

            # 后处理加速（长文本更有效）
            if enable_post_speedup and len(text) > 10:
                _apply_post_speedup(output_path, post_speed_factor)
            break
        except Exception as e:
            logger.warning(f"TTS failed (attempt {retry+1}/3): {e}")
            if retry == 2:
                raise e
            time.sleep(1)


# ========================
# 交互式测试入口
# ========================
if __name__ == '__main__':
    # 示例参考音频路径
    speaker_wav = r'videos\WorldofAI\20251229 Gemini 3 Deep Think Enhancing Gemini to Become a Super Agent Build Automate ANYTHING\audio_vocals.wav'
    
    # 直接指定模型路径（避免下载）
    MODEL_PATH = r"C:\model\tts\tts_models--multilingual--multi-dataset--xtts_v2"
    
    # 验证模型路径
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Model path not found: {MODEL_PATH}")
        logger.info("Searching for alternative model locations...")
        found_path = find_model_path()
        if found_path:
            MODEL_PATH = found_path
            logger.info(f"Using alternative path: {MODEL_PATH}")
        else:
            logger.warning("No model found, will try to download (may fail)")
    
    os.makedirs('playground', exist_ok=True)

    while True:
        text = input('请输入要合成的文本：').strip()
        if not text:
            continue
        # 安全文件名
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in text[:30])
        output_path = f'playground/{safe_name}.wav'

        try:
            tts(
                text=text,
                output_path=output_path,
                speaker_wav=speaker_wav,
                model_path=MODEL_PATH,  # 明确指定本地模型路径
                speed=1.5,
                temperature=0.7,
                enable_post_speedup=True,
                post_speed_factor=1.4,
            )
            print(f"✅ 合成完成！音频保存到: {output_path}")
        except Exception as e:
            print(f"❌ 合成失败: {e}")
            print("提示：请确认模型文件是否完整，包含 config.json 和 model.pth 等文件")