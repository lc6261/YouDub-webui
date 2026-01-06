#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频分离工具：使用 Demucs 模型从视频或音频中提取人声与伴奏。

功能：
- 从 download.mp4 提取 audio.wav
- 使用 Demucs (htdemucs/MDX) 分离 vocals / instruments
- 支持 API 调用与 CLI 回退方案

依赖：
- Python ≥ 3.8
- torch, torchaudio, demucs, ffmpeg, loguru

作者：Your Name
创建时间：2025-xx-xx
最后修改：2025-12-29
版本: 1.0
"""
import sys         # 系统相关功能（如标准输入输出、退出等）
import os          # 操作系统接口（路径、环境变量等）
# 将项目根目录（youdub 的父目录）加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入标准库
import shutil      # 用于文件/目录操作（如复制、删除）
import time        # 时间相关操作（如计时）
import subprocess  # 用于调用外部命令（如命令行工具）
import tempfile    # 创建临时文件/目录
from pathlib import Path  # 面向对象的路径操作（比 os.path 更现代）

# 第三方库
import torch                    # 深度学习框架
import torchaudio               # PyTorch 的音频处理库
import numpy as np              # 数值计算
from loguru import logger       # 灵活的日志记录工具

# 本地模块（需确保 utils.py 存在）
from youdub.utils import save_wav, normalize_wav

# 全局变量设置
# 自动选择设备：CUDA（GPU）可用则用，否则用 CPU
auto_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 全局分离器实例（避免重复加载模型）
separator = None


# ==============================================================================
# Demucs 4.0.1 兼容层：封装新版本 Demucs 以模拟旧 API
# ==============================================================================
class Separator:
    """
    Demucs 音频分离模型的封装类，兼容 Demucs 4.0.1 的 API。
    支持从文件或内存中的音频数据进行分离。
    """

    def __init__(self, model_name="htdemucs_ft", device='auto', progress=True, shifts=5):
        """
        初始化分离器。
        
        参数:
            model_name (str): 模型名称（支持别名映射）
            device (str): 'auto', 'cpu', 或 'cuda'
            progress (bool): 是否显示进度条
            shifts (int): 音频时移增强次数（提高分离质量，但更慢）
        """
        self.model_name = model_name
        # 自动选择设备
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.shifts = shifts
        self.progress = progress

        # 导入 Demucs 核心组件
        from demucs.pretrained import get_model
        from demucs.audio import convert_audio

        # 模型名称映射：将旧名称映射到 Demucs 4.0.1 的实际模型名
        model_map = {
            "htdemucs_ft": "htdemucs",     # fine-tuned 版本实际对应 htdemucs
            "htdemucs": "htdemucs",
            "mdx_extra": "mdx_extra_q",    # 高质量 MDX 模型
            "mdx": "mdx_extra_q",
        }

        actual_model = model_map.get(model_name, model_name)
        logger.info(f"Loading Demucs model: {actual_model}")

        try:
            self.model = get_model(name=actual_model)
        except Exception as e:
            logger.warning(f"Model {actual_model} not found: {e}. Falling back to 'htdemucs'.")
            self.model = get_model(name="htdemucs")  # 降级使用默认模型

        # 将模型加载到指定设备并设为评估模式
        self.model.to(self.device)
        self.model.eval()

        # 获取模型的采样率和音频通道数
        self.sample_rate = self.model.samplerate
        self.convert_audio = convert_audio  # 用于重采样和通道转换

    def separate_audio_file(self, audio_path):
        """
        从音频文件中分离音轨。
        
        参数:
            audio_path (str): 输入音频文件路径
        
        返回:
            tuple: (原始音频 tensor, 分离结果字典)
                - origin: 重采样后的原始音频 (channels, samples)
                - separated: {'drums': ..., 'bass': ..., 'other': ..., 'vocals': ...}
        """
        import torch
        from demucs.apply import apply_model

        # 1. 加载原始音频（使用 torchaudio）
        wav, sr = torchaudio.load(audio_path)  # shape: [channels, samples]

        # 2. 转换音频格式以匹配模型输入要求（采样率、通道数）
        wav = self.convert_audio(wav, sr, self.sample_rate, self.model.audio_channels)

        # 3. 执行分离（添加 batch 维度，移除后再返回）
        with torch.no_grad():
            sources = apply_model(
                self.model,
                wav[None],  # 添加 batch 维度: [1, channels, samples]
                device=self.device,
                shifts=self.shifts if self.shifts > 1 else 0,  # shifts=0 表示不启用
                split=True,        # 自动分块处理长音频
                progress=self.progress
            )[0]  # 移除 batch 维度: [sources, channels, samples]

        # 4. 构建分离结果字典（Demucs 4.0 固定顺序）
        track_names = ['drums', 'bass', 'other', 'vocals']
        separated = {}
        for i, name in enumerate(track_names):
            separated[name] = sources[i]  # 每个音轨: [channels, samples]

        return wav, separated

    def separate_audio(self, wav, sr):
        """
        从内存中的音频张量进行分离（不依赖文件）。
        
        参数:
            wav (Tensor): 音频张量，shape [channels, samples]
            sr (int): 原始采样率
        
        返回:
            tuple: (转换后的原始音频, 分离结果字典)
        """
        from demucs.apply import apply_model

        # 转换格式
        wav = self.convert_audio(wav, sr, self.sample_rate, self.model.audio_channels)

        # 分离
        with torch.no_grad():
            sources = apply_model(
                self.model,
                wav[None],
                device=self.device,
                shifts=self.shifts if self.shifts > 1 else 0,
                split=True,
                progress=self.progress
            )[0]

        # 构建结果
        track_names = ['drums', 'bass', 'other', 'vocals']
        separated = {}
        for i, name in enumerate(track_names):
            separated[name] = sources[i]

        return wav, separated


# ==============================================================================
# 全局模型管理函数（单例模式）
# ==============================================================================
def init_demucs():
    """初始化分离器（兼容旧接口）"""
    global separator
    separator = load_model()

def load_model(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int = 5):
    """
    加载 Demucs 模型（如果尚未加载）。
    避免重复加载以节省时间和显存。
    """
    global separator
    if separator is not None:
        logger.info(f'Demucs model already loaded')
        return

    logger.info(f'Loading Demucs model: {model_name}')
    t_start = time.time()
    separator = Separator(model_name, device=device, progress=progress, shifts=shifts)
    t_end = time.time()
    logger.info(f'Demucs model loaded in {t_end - t_start:.2f} seconds')

def reload_model(model_name: str = "htdemucs_ft", device: str = 'auto', progress: bool = True, shifts: int = 5):
    """
    强制重新加载模型（用于切换模型或设备）。
    """
    global separator
    logger.info(f'Reloading Demucs model: {model_name}')
    t_start = time.time()
    separator = Separator(model_name, device=device, progress=progress, shifts=shifts)
    t_end = time.time()
    logger.info(f'Demucs model reloaded in {t_end - t_start:.2f} seconds')


# ==============================================================================
# 主分离功能
# ==============================================================================
def separate_audio(folder: str, model_name: str = "htdemucs_ft", device: str = 'auto',
                   progress: bool = True, shifts: int = 5) -> None:
    """
    对指定文件夹内的 audio.wav 进行音效分离。
    
    输出:
        - audio_vocals.wav: 人声音轨
        - audio_instruments.wav: 伴奏（drums + bass + other）
        - audio_drums.wav: 鼓声音轨
        - audio_bass.wav: 贝斯音轨
        - audio_other.wav: 其他乐器音轨
    """
    global separator
    audio_path = os.path.join(folder, 'audio.wav')
    if not os.path.exists(audio_path):
        return  # 无音频文件，跳过

    # 输出文件路径
    vocal_output_path = os.path.join(folder, 'audio_vocals.wav')
    instruments_output_path = os.path.join(folder, 'audio_instruments.wav')
    drums_output_path = os.path.join(folder, 'audio_drums.wav')
    bass_output_path = os.path.join(folder, 'audio_bass.wav')
    other_output_path = os.path.join(folder, 'audio_other.wav')

    # 检查是否已分离所有音效组
    all_files = [vocal_output_path, instruments_output_path, drums_output_path, bass_output_path, other_output_path]
    if all(os.path.exists(f) for f in all_files):
        logger.info(f'All audio tracks already separated in {folder}')
        return

    logger.info(f'Separating audio from {folder}')
    load_model(model_name, device, progress, shifts)  # 确保模型已加载
    t_start = time.time()

    try:
        origin, separated = separator.separate_audio_file(audio_path)
    except Exception as e:
        logger.error(f'Error separating audio from {folder}: {str(e)}')
        time.sleep(5)

        # 备用方案：尝试调用命令行版 demucs
        try:
            logger.info("Trying alternative separation method...")
            success = separate_with_cli(audio_path, folder, model_name)
            if success:
                logger.info(f"Audio separated using CLI method for {folder}")
                return
        except Exception as cli_e:
            logger.error(f"CLI fallback also failed: {cli_e}")

        # 如果备选也失败，抛出异常
        raise Exception(f'Error separating audio from {folder}: {str(e)}')

    t_end = time.time()
    logger.info(f'Audio separated in {t_end - t_start:.2f} seconds')

    # 保存各个音效组
    for track_name in ['drums', 'bass', 'other', 'vocals']:
        track_data = separated[track_name].cpu().numpy().T
        output_path = os.path.join(folder, f'audio_{track_name}.wav')
        save_wav(track_data, output_path, sample_rate=separator.sample_rate)
        logger.info(f'{track_name.title()} saved to {output_path}')

    # 合并所有非人声音轨为伴奏
    instruments = None
    for k, v in separated.items():
        if k == 'vocals':
            continue
        if instruments is None:
            instruments = v.cpu()
        else:
            instruments += v.cpu()  # 直接相加（Demucs 输出已归一化）

    instruments = instruments.numpy().T

    # 保存合并后的伴奏
    save_wav(instruments, instruments_output_path, sample_rate=separator.sample_rate)
    logger.info(f'Combined instruments saved to {instruments_output_path}')


# ==============================================================================
# 备用方案：调用命令行版 demucs
# ==============================================================================
def separate_with_cli(audio_path, output_folder, model_name="htdemucs"):
    """
    作为备选方案，调用系统命令行的 `demucs` 工具进行分离。
    适用于 Python API 崩溃或兼容性问题时。
    """
    try:
        temp_dir = tempfile.mkdtemp()  # 创建临时目录存放输出

        # 模型名称映射（同上）
        model_map = {
            "htdemucs_ft": "htdemucs",
            "htdemucs": "htdemucs",
            "mdx_extra": "mdx_extra_q",
            "mdx": "mdx_extra_q",
        }
        actual_model = model_map.get(model_name, model_name)

        # 构建命令
        cmd = [
            "demucs",
            "--name", actual_model,
            "--device", "cuda" if torch.cuda.is_available() else "cpu",
            "--out", temp_dir,
            audio_path
        ]

        logger.info(f"Running CLI command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"CLI separation failed: {result.stderr}")
            return False

        # Demucs CLI 输出路径为: <temp_dir>/<model_name>/<filename>/
        output_base = Path(temp_dir) / actual_model / Path(audio_path).stem

        # 寻找人声文件（通常命名为 <filename>_vocals.wav）
        vocal_file = output_base.with_name(output_base.name + "_vocals.wav")
        if vocal_file.exists():
            shutil.copy(vocal_file, os.path.join(output_folder, 'audio_vocals.wav'))
            logger.info("Vocals copied from CLI output.")
        else:
            logger.warning("Vocals file not found in CLI output.")

        # 【注意】：此函数目前仅处理人声，未实现伴奏合并！
        # 若需完整功能，应读取 drums/bass/other 并混合——但 CLI 方式较复杂，建议优先使用 API。

        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)
        return True

    except Exception as e:
        logger.error(f"CLI separation error: {str(e)}")
        return False


# ==============================================================================
# 视频转音频
# ==============================================================================
def extract_audio_from_video(folder: str) -> bool:
    """
    从 download.mp4 中提取音频为 audio.wav（44.1kHz, 16-bit, stereo）。
    使用 ffmpeg 命令行工具。
    """
    video_path = os.path.join(folder, 'download.mp4')
    if not os.path.exists(video_path):
        return False

    audio_path = os.path.join(folder, 'audio.wav')
    if os.path.exists(audio_path):
        logger.info(f'Audio already extracted in {folder}')
        return True

    logger.info(f'Extracting audio from {folder}')

    # 调用 ffmpeg（需确保系统已安装）
    cmd = f'ffmpeg -loglevel error -i "{video_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{audio_path}"'
    os.system(cmd)

    # 可选：后续可取消注释以启用音频归一化
    # normalize_wav(audio_path)

    time.sleep(1)  # 确保文件写入完成
    logger.info(f'Audio extracted from {folder}')
    return True


# ==============================================================================
# 批量处理：遍历目录，自动提取+分离
# ==============================================================================
def separate_all_audio_under_folder(root_folder: str, model_name: str = "htdemucs_ft",
                                    device: str = 'auto', progress: bool = True, shifts: int = 5):
    """
    遍历 root_folder 下所有子目录：
        - 若有 download.mp4 → 提取 audio.wav
        - 若无 audio_vocals.wav → 执行分离
    """
    global separator
    for subdir, dirs, files in os.walk(root_folder):
        if 'download.mp4' not in files:
            continue  # 跳过无视频的目录

        # 步骤1：提取音频（如未存在）
        if 'audio.wav' not in files:
            extract_audio_from_video(subdir)

        # 步骤2：分离音频（如未完成）
        if 'audio_vocals.wav' not in files:
            try:
                separate_audio(subdir, model_name, device, progress, shifts)
            except Exception as e:
                logger.error(f"Failed to separate audio in {subdir}: {str(e)}")
                continue  # 跳过出错目录，继续处理其他

    logger.info(f'All audio separated under {root_folder}')
    return f'All audio separated under {root_folder}'


# ==============================================================================
# 主程序入口（用于测试）
# ==============================================================================
if __name__ == '__main__':
    # 示例：处理 videos/ 目录下所有视频
    folder = r"videos"
    separate_all_audio_under_folder(folder, shifts=0)  # shifts=0 可加快速度（但质量略低）