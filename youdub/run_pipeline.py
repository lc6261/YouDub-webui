#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube 视频自动翻译配音 - 一键执行脚本
支持断点续传，自动跳过已完成步骤

使用方法:
  python run_pipeline.py                    # 处理 tasks.csv 中的所有视频
  python run_pipeline.py --url "视频URL"    # 处理单个视频
  python run_pipeline.py --step 3          # 从第3步开始执行

作者: Pipeline Integration
日期: 2026-01-03
"""

import os
import sys

# 添加当前目录到Python路径，确保能正确导入youdub模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from pathlib import Path
from loguru import logger
from typing import List, Optional

# ===== 配置区域 =====
DEFAULT_CONFIG = {
    "video_folder": "videos",           # 视频存储根目录
    "tasks_file": "tasks.csv",          # 任务列表文件
    "cookies_file": "cookies.txt",      # YouTube cookies
    "whisper_model": "large-v3",        # 语音识别模型
    "translation_lang": "简体中文",      # 目标语言
    "force_bytedance_tts": True,        # 强制使用字节跳动TTS
    "video_speed_up": 1.05,             # 视频加速倍数
    "video_resolution": "1080p",        # 输出分辨率
    "enable_subtitles": True,           # 嵌入字幕
    "use_gpu": True,                   # 使用GPU加速编码
}

# ===== 步骤定义 =====
PIPELINE_STEPS = [
    {
        "id": 0,
        "name": "视频下载",
        "module": "step000_video_downloader_csv",
        "function": "main",
        "check_file": "download.mp4",
        "description": "从 YouTube 下载视频"
    },
    {
        "id": 1,
        "name": "音频分离",
        "module": "step010_demucs_vr",
        "function": "separate_all_audio_under_folder",
        "check_file": "audio_vocals.wav",
        "description": "分离人声和伴奏"
    },
    {
        "id": 2,
        "name": "语音识别",
        "module": "step020_whisperx_silero_vad",
        "function": "transcribe_all_audio_under_folder",
        "check_file": "transcript.json",
        "description": "生成字幕 + 说话人分离"
    },
    {
        "id": 3,
        "name": "字幕翻译",
        "module": "step030_translation_vad_qwen",
        "function": "translate_all_advanced",
        "check_file": "translation.json",
        "description": "翻译字幕为目标语言"
    },
    {
        "id": 4,
        "name": "语音合成",
        "module": "step040_tts_vox_cpm_qwen",
        "function": "generate_all_wavs_under_folder",
        "check_file": "audio_combined.wav",
        "description": "生成翻译配音"
    },
    {
        "id": 5,
        "name": "视频合成",
        "module": "step050_synthesize_video",
        "function": "synthesize_all_video_under_folder",
        "check_file": "video.mp4",
        "description": "合成最终视频"
    },
    {
        "id": 6,
        "name": "生成视频信息",
        "module": "step060_genrate_info",
        "function": "generate_all_info_under_folder",
        "check_file": "video.png",
        "description": "生成视频摘要和调整缩略图尺寸"
    },
    {
        "id": 7,
        "name": "上传B站",
        "module": "step070_upload_bilibili",
        "function": "upload_all_videos_under_folder",
        "check_file": "bilibili.json",
        "description": "将视频上传到B站"
    }
]


class VideoPipeline:
    """视频处理流水线管理器"""
    
    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG
        self.video_folder = self.config["video_folder"]
        
        # 配置日志
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level="INFO"
        )
        logger.add(
            "pipeline.log",
            rotation="10 MB",
            retention="7 days",
            encoding="utf-8",
            level="DEBUG"
        )
    
    def check_dependencies(self) -> bool:
        """检查必要的依赖和文件"""
        logger.info("🔍 检查依赖环境...")
        
        # 检查 Python 模块
        required_modules = [
            "torch", "librosa", "whisperx", "demucs", 
            "openai", "loguru", "numpy"
        ]
        missing = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            logger.error(f"❌ 缺少 Python 模块: {', '.join(missing)}")
            logger.info("请运行: pip install " + " ".join(missing))
            return False
        
        # 检查 ffmpeg
        import shutil
        if not shutil.which("ffmpeg"):
            logger.error("❌ 未找到 ffmpeg，请安装后添加到 PATH")
            return False
        
        logger.info("✅ 依赖检查通过")
        return True
    
    def get_video_folders(self, use_task_steps: bool = False) -> List[Path]:
        """获取所有待处理的视频文件夹"""
        all_folders = []
        for root, dirs, files in os.walk(self.video_folder):
            if "download.mp4" in files:
                all_folders.append(Path(root))
        
        # 如果使用任务步骤，只返回需要完整处理的文件夹
        if use_task_steps:
            import csv
            try:
                with open(self.config["tasks_file"], 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # 获取所有需要完整处理的任务的output_path
                    full_process_paths = set()
                    for task in reader:
                        if task.get('task_type') == 'full_process' and task.get('status') in ['pending', 'processing']:
                            output_path = task.get('output_path', '')
                            if output_path:
                                full_process_paths.add(output_path.replace('/', '\\'))
                    
                # 过滤出需要完整处理的文件夹
                filtered_folders = []
                for folder in all_folders:
                    folder_str = str(folder)
                    if any(full_path in folder_str for full_path in full_process_paths):
                        filtered_folders.append(folder)
                
                logger.info(f"🔍 过滤后，待处理文件夹数量: {len(filtered_folders)}/{len(all_folders)}")
                return sorted(filtered_folders)
            except Exception as e:
                logger.warning(f"⚠️  过滤视频文件夹失败: {e}")
                return sorted(all_folders)
        
        return sorted(all_folders)
    
    def check_step_completion(self, folder: Path, step: dict) -> bool:
        """检查某个步骤是否已完成"""
        check_file = folder / step["check_file"]
        return check_file.exists()
    
    def get_step_module(self, step: dict):
        """动态导入步骤模块"""
        try:
            # 对于 youdub 包内的模块，使用完整的包路径
            module_name = f"youdub.{step['module']}"
            module = __import__(module_name, fromlist=[step["function"]])
            return getattr(module, step["function"])
        except (ImportError, AttributeError) as e:
            logger.error(f"❌ 无法导入模块 {step['module']}: {e}")
            logger.debug(f"尝试导入完整路径: youdub.{step['module']}")
            return None
    
    def run_step(self, step: dict, start_from_step: int = 0, skip_translation: bool = False, use_task_steps: bool = False) -> bool:
        """执行单个处理步骤"""
        # 跳过翻译步骤
        if skip_translation and step["id"] == 3:
            logger.info(f"⏭️  跳过翻译步骤 {step['id']}: {step['name']}")
            return True
        
        if step["id"] < start_from_step:
            logger.info(f"⏭️  跳过步骤 {step['id']}: {step['name']}")
            return True
        
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 步骤 {step['id']}: {step['name']}")
        logger.info(f"📝 {step['description']}")
        logger.info(f"{'='*60}\n")
        
        # 检查已完成的文件夹数量
        folders = self.get_video_folders(use_task_steps)
        completed = sum(1 for f in folders if self.check_step_completion(f, step))
        total = len(folders)
        
        if completed == total:
            logger.info(f"✅ 所有视频已完成此步骤 ({completed}/{total})")
            return True
        
        logger.info(f"📊 进度: {completed}/{total} 已完成，{total - completed} 待处理")
        
        # 获取执行函数
        func = self.get_step_module(step)
        if not func:
            return False
        
        # 执行步骤
        try:
            start_time = time.time()
            
            # 根据步骤传递不同参数
            if step["id"] == 0:  # 视频下载
                result = func()
            elif step["id"] == 1:  # 音频分离
                result = func(self.video_folder, shifts=0)
            elif step["id"] == 2:  # 语音识别
                result = func(
                    self.video_folder,
                    model_name=self.config["whisper_model"],
                    diarization=True
                )
            elif step["id"] == 3:  # 翻译
                result = func(
                    self.video_folder,
                    target_language=self.config["translation_lang"]
                )
            elif step["id"] == 4:  # TTS
                result = func(self.video_folder)
            elif step["id"] == 5:  # 视频合成
                result = func(
                    self.video_folder,
                    subtitles=self.config["enable_subtitles"],
                    speed_up=self.config["video_speed_up"],
                    resolution=self.config["video_resolution"],
                    use_gpu=self.config["use_gpu"]
                )
            elif step["id"] == 6:  # 生成视频信息
                result = func(self.video_folder)
            elif step["id"] == 7:  # 上传B站
                result = func(self.video_folder)
            
            elapsed = time.time() - start_time
            logger.success(f"✅ 步骤 {step['id']} 完成！用时: {elapsed/60:.1f} 分钟")
            return True
            
        except Exception as e:
            logger.error(f"❌ 步骤 {step['id']} 失败: {e}")
            logger.exception("详细错误:")
            return False
    
    def run_pipeline(self, start_from_step: int = 0, end_at_step: int = None, skip_translation: bool = False, use_task_steps: bool = False):
        """执行完整流水线"""
        logger.info("\n" + "🎬"*30)
        logger.info("YouTube 视频翻译配音流水线启动")
        logger.info("🎬"*30 + "\n")
        
        # 检查依赖
        if not self.check_dependencies():
            return False
        
        # 读取任务配置
        task_steps = None
        full_process_tasks = []
        if use_task_steps:
            import csv
            try:
                with open(self.config["tasks_file"], 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    task_steps = [row for row in reader]
                logger.info(f"✅ 读取到 {len(task_steps)} 个任务配置")
                
                # 筛选出需要完整处理的任务
                full_process_tasks = [
                    task for task in task_steps 
                    if task.get('status') in ['pending', 'processing'] and 
                    task.get('task_type') == 'full_process'
                ]
                logger.info(f"✅ 筛选出 {len(full_process_tasks)} 个需要完整处理的任务")
            except Exception as e:
                logger.warning(f"⚠️  读取任务配置失败: {e}")
                use_task_steps = False
        
        # 执行步骤
        end_step = end_at_step if end_at_step is not None else len(PIPELINE_STEPS)
        pipeline_start = time.time()
        
        # 先执行所有步骤，包括视频下载
        for step in PIPELINE_STEPS:
            if step["id"] >= end_step:
                break
            
            # 检查是否需要执行当前步骤
            if use_task_steps:
                # 检查所有任务是否都需要执行该步骤
                # 如果有任何一个任务需要执行，就执行该步骤
                should_run = False
                for task in full_process_tasks:
                    # 安全处理steps字段，确保即使为空也能正常处理
                    steps_str = task.get('steps', '0,1,2,3,4,5,6,7')
                    if not steps_str:
                        steps_str = '0,1,2,3,4,5,6,7'
                    # 移除引号并分割
                    steps_str = steps_str.strip('"').strip("'")
                    task_step_list = [int(s) for s in steps_str.split(',') if s.strip().isdigit()]
                    if step["id"] in task_step_list:
                        should_run = True
                        break
                if not should_run:
                    logger.info(f"⏭️  所有待处理任务都不需要执行步骤 {step['id']}: {step['name']}")
                    continue
            
            success = self.run_step(step, start_from_step, skip_translation, use_task_steps)
            if not success:
                logger.error(f"💥 流水线在步骤 {step['id']} 中断")
                return False
            
            # 步骤间短暂休息
            time.sleep(2)
        
        # 获取视频文件夹
        folders = self.get_video_folders()
        
        # 完成统计
        total_time = time.time() - pipeline_start
        logger.info("\n" + "🎉"*30)
        logger.success("所有步骤执行完成！")
        logger.info(f"⏱️  总用时: {total_time/60:.1f} 分钟")
        logger.info("🎉"*30 + "\n")
        
        # 输出结果位置
        logger.info("📂 输出文件位置:")
        for folder in folders:
            video_file = folder / "video.mp4"
            if video_file.exists():
                logger.info(f"  ✅ {video_file}")
        
        # 释放所有模型资源，避免GPU内存泄漏
        logger.info("🗑️  正在释放所有模型资源...")
        
        try:
            # 释放 WhisperX 模型
            from youdub.step020_whisperx_silero_vad import release_models
            release_models()
        except Exception as e:
            logger.warning(f"⚠️  释放 WhisperX 模型资源失败: {e}")
        
        try:
            # 释放 VoxCPM 模型
            from youdub.step040_tts_vox_cpm_qwen import release_voxcpm_model
            release_voxcpm_model()
        except Exception as e:
            logger.warning(f"⚠️  释放 VoxCPM 模型资源失败: {e}")
        
        # 清理GPU缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception as e:
            logger.warning(f"⚠️  清理GPU缓存失败: {e}")
        
        # 强制垃圾回收
        try:
            import gc
            gc.collect()
        except Exception as e:
            logger.warning(f"⚠️  垃圾回收失败: {e}")
        
        logger.info("✅ 所有资源已释放完成")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="YouTube 视频自动翻译配音流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理 tasks.csv 中的所有视频（完整流程）
  python run_pipeline.py
  
  # 从第3步（翻译）开始执行
  python run_pipeline.py --step 3
  
  # 只执行前3步（下载+分离+识别）
  python run_pipeline.py --end 3
  
  # 使用自定义配置
  python run_pipeline.py --model medium --lang 英语
        """
    )
    
    parser.add_argument(
        "--step", type=int, default=0,
        help="从指定步骤开始执行（0-5，默认: 0）"
    )
    parser.add_argument(
        "--end", type=int, default=None,
        help="执行到指定步骤结束（不含）"
    )
    parser.add_argument(
        "--folder", type=str, default="videos",
        help="视频根目录（默认: videos）"
    )
    parser.add_argument(
        "--model", type=str, default="large-v3",
        choices=["large-v3", "medium", "small"],
        help="Whisper 模型（默认: large-v3）"
    )
    parser.add_argument(
        "--lang", type=str, default="简体中文",
        help="目标语言（默认: 简体中文）"
    )
    parser.add_argument(
        "--no-subtitles", action="store_true",
        help="不嵌入字幕"
    )
    parser.add_argument(
        "--speed", type=float, default=1.05,
        help="视频加速倍数（默认: 1.05）"
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="不使用GPU加速编码（默认: 使用GPU）"
    )
    parser.add_argument(
        "--skip-translation", action="store_true",
        help="跳过字幕翻译步骤（默认: 不跳过）"
    )
    parser.add_argument(
        "--use-task-steps", action="store_true",
        help="根据任务的steps字段执行特定步骤（默认: 不使用）"
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = DEFAULT_CONFIG.copy()
    config.update({
        "video_folder": args.folder,
        "whisper_model": args.model,
        "translation_lang": args.lang,
        "enable_subtitles": not args.no_subtitles,
        "video_speed_up": args.speed,
        "use_gpu": not args.no_gpu,
    })
    
    # 执行流水线
    pipeline = VideoPipeline(config)
    success = pipeline.run_pipeline(
        start_from_step=args.step,
        end_at_step=args.end,
        skip_translation=args.skip_translation,
        use_task_steps=args.use_task_steps
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()