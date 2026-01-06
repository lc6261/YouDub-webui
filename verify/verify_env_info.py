#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境信息诊断脚本
作者: Assistant
日期: 2026-01-01
用途: 打印当前 Python 环境、PyTorch/CUDA 状态、导出 requirements.txt 内容
"""

import sys
import platform
import subprocess
import os

def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        return result.stdout.strip(), result.stderr.strip()
    except Exception as e:
        return "", str(e)

def main():
    print("=" * 60)
    print("🔍 当前环境诊断报告")
    print("=" * 60)

    # 1. Python & 系统信息
    print("\n[1] Python 与系统信息")
    print(f"Python 版本    : {sys.version}")
    print(f"Python 路径    : {sys.executable}")
    print(f"平台           : {platform.platform()}")
    print(f"架构           : {platform.machine()}")
    print(f"当前工作目录   : {os.getcwd()}")
    print(f"虚拟环境       : {sys.prefix}")

    # 2. pip 版本
    print("\n[2] pip 信息")
    pip_out, pip_err = run_cmd("pip --version")
    if pip_out:
        print(pip_out)
    else:
        print(f"⚠️  pip 错误: {pip_err}")

    # 3. PyTorch & CUDA
    print("\n[3] PyTorch 与 CUDA 信息")
    try:
        import torch
        print(f"PyTorch 版本     : {torch.__version__}")
        print(f"CUDA 可用        : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 设备数量    : {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i} 名称   : {torch.cuda.get_device_name(i)}")
            print(f"CUDA 版本        : {torch.version.cuda}")
            print(f"cuDNN 版本       : {torch.backends.cudnn.version()}")
        else:
            print("⚠️  CUDA 不可用，请检查驱动或 PyTorch 安装")
    except ImportError:
        print("❌ PyTorch 未安装")
    except Exception as e:
        print(f"⚠️  PyTorch 检测异常: {e}")

    # 4. NVIDIA 驱动（仅 Windows / Linux）
    print("\n[4] NVIDIA 驱动信息（如可获取）")
    if platform.system() == "Windows":
        nvidia_out, _ = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
        if nvidia_out:
            print(f"NVIDIA 驱动版本  : {nvidia_out}")
        else:
            print("⚠️  无法获取 NVIDIA 驱动信息（请确认 nvidia-smi 是否在 PATH 中）")
    elif platform.system() == "Linux":
        nvidia_out, _ = run_cmd("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null")
        if nvidia_out:
            print(f"NVIDIA 驱动版本  : {nvidia_out}")
        else:
            print("⚠️  无法获取 NVIDIA 驱动信息（nvidia-smi 未找到）")
    else:
        print("ℹ️  非 Windows/Linux 系统，跳过 nvidia-smi 检测")

    # 5. 导出 requirements.txt（精确版本）
    print("\n[5] 当前环境的 requirements.txt 内容（可直接复制使用）")
    print("-" * 60)
    pip_list_out, pip_list_err = run_cmd("pip list --format=freeze")
    if pip_list_out:
        # 过滤掉以 -e 开头的本地开发包（避免路径泄漏）
        lines = pip_list_out.splitlines()
        clean_lines = [line for line in lines if not line.startswith("-e ")]
        for line in sorted(clean_lines):
            print(line)
    else:
        print(f"❌ 无法获取 pip list: {pip_list_err}")
    print("-" * 60)

    print("\n✅ 诊断完成！你可以将 [5] 的内容保存为 requirements.txt 使用。")

if __name__ == "__main__":
    main()