#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试不同克隆音的TTS生成效果
"""

import os
import sys
import time
from loguru import logger

# 导入TTS相关功能
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from youdub.step040_tts_vox_cpm_qwen import generate_voxcpm_audio

# 测试文本
test_text = "这是一个测试文本，用于验证不同克隆音的TTS生成效果。"

# 测试克隆音列表
test_voices = [
    "亲切女声",
    "稳重男",
    "青年女性",
    "解说小帅"
]

def main():
    logger.info("🎤 开始测试不同克隆音的TTS生成效果")
    logger.info(f"📝 测试文本: {test_text}")
    
    # 创建输出目录
    output_dir = "test_tts_output"
    os.makedirs(output_dir, exist_ok=True)
    
    for voice_name in test_voices:
        logger.info(f"\n🎬 测试克隆音: {voice_name}")
        
        # 构建克隆音文件路径
        voice_wav = os.path.join("voice", f"{voice_name}_cloned.wav")
        
        if not os.path.exists(voice_wav):
            logger.error(f"❌ 克隆音文件不存在: {voice_wav}")
            continue
        
        # 生成输出文件名
        output_path = os.path.join(output_dir, f"tts_{voice_name}.wav")
        
        # 读取克隆音文本
        voice_txt = os.path.join("voice", f"{voice_name}_cloned.txt")
        if not os.path.exists(voice_txt):
            logger.error(f"❌ 克隆音文本不存在: {voice_txt}")
            continue
        
        with open(voice_txt, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        
        if not prompt_text:
            logger.error(f"❌ 克隆音文本为空: {voice_txt}")
            continue
        
        # 生成TTS语音
        start_time = time.time()
        success = generate_voxcpm_audio(
            text=test_text,
            output_path=output_path,
            speaker_wav=voice_wav,
            target_duration=None,
            prompt_text=prompt_text
        )
        end_time = time.time()
        
        if success:
            logger.success(f"✅ 克隆音 {voice_name} 生成成功")
            logger.info(f"⏱️ 生成时间: {end_time - start_time:.2f} 秒")
            logger.info(f"📦 输出文件: {output_path}")
        else:
            logger.error(f"❌ 克隆音 {voice_name} 生成失败")
    
    logger.info("\n🎉 所有测试完成！")
    logger.info(f"📁 输出目录: {output_dir}")

if __name__ == "__main__":
    main()
