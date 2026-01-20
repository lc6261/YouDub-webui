import json
import os
import sys
import time
import argparse
from loguru import logger
from .step000_video_downloader_csv import get_video_infos, download_and_merge, _get_output_path, sanitize_title, update_csv_file, initialize_csv_file
from .step010_demucs_vr import separate_all_audio_under_folder, init_demucs, separate_audio, unload_demucs_model
from .step020_whisperx_silero_vad import transcribe_all_audio_under_folder, init_whisperx, transcribe_audio, unload_whisper_model
from .step030_translation_vad_qwen import translate_advanced
from .step031_extract_speaker_clips import extract_speaker_clips_for_folder
from .step040_tts_vox_cpm_qwen import generate_all_wavs_under_folder, generate_wavs, unload_voxcpm_model
from .step050_synthesize_video import synthesize_video
from .step060_genrate_info import generate_info, generate_all_info_under_folder
# from .step070_upload_bilibili import upload_all_videos_under_folder, upload_video
from concurrent.futures import ThreadPoolExecutor, as_completed
import re


def get_info_list_from_url(urls, num_videos=5, status_mask='1111111', csv_path=None):
    """获取视频信息列表"""
    # 初始化CSV文件，如果提供了路径
    if csv_path:
        initialize_csv_file(csv_path)
    
    video_infos = list(get_video_infos(urls, num_videos=num_videos))
    
    # 为每个视频信息添加status_mask和csv_path
    for info in video_infos:
        info['status_mask'] = status_mask
        info['csv_path'] = csv_path
    
    return video_infos



def download_single_video(info, root_folder, resolution='1080p'):
    """下载单个视频"""
    try:
        folder = download_and_merge(info, root_folder, resolution=resolution)
        
        # 收集视频信息
        video_info = {
            'video_title': info.get('title', ''),
            'video_uploader': info.get('uploader', ''),
            'video_duration': info.get('duration', 0),
            'video_upload_date': info.get('upload_date', '')
        }
        
        # 更新CSV文件
        csv_path = info.get('csv_path')
        if csv_path:
            update_csv_file(csv_path, info['webpage_url'], video_info)
        
        return folder
    except Exception as e:
        logger.error(f"下载视频失败: {e}")
        return None



def get_target_folder(info, root_folder):
    """获取目标文件夹路径"""
    try:
        output_path = _get_output_path(info, root_folder)
        return os.path.dirname(output_path)
    except Exception as e:
        logger.error(f"获取目标文件夹失败: {e}")
        return None



def process_video(info, root_folder, resolution, demucs_model, device, shifts, whisper_model, whisper_download_root, whisper_batch_size, whisper_diarization, whisper_min_speakers, whisper_max_speakers, translation_target_language, subtitles, speed_up, fps, target_resolution, max_retries):
    # 获取状态掩码，默认全执行
    status_mask = info.get('status_mask', '1111111')
    # 确保掩码长度为7位
    if len(status_mask) < 7:
        status_mask = status_mask.ljust(7, '1')
    elif len(status_mask) > 7:
        status_mask = status_mask[:7]
    
    # 获取CSV路径
    csv_path = info.get('csv_path')
    
    for retry in range(max_retries):
        try:
            folder = get_target_folder(info, root_folder)
            if folder is None:
                logger.warning(f'Failed to get target folder for video {info["title"]}')
                return False
            
            # 跳过已处理的视频
            if os.path.exists(os.path.join(folder, 'info.json')):
                logger.info(f'Video already processed in {folder}')
                return True
            
            # 记录启动时间
            start_time = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # 更新CSV文件的启动时间
            if csv_path:
                update_csv_file(csv_path, info['webpage_url'], {'start_time': start_time})
                
            folder = download_single_video(info, root_folder, resolution)
            if folder is None:
                logger.warning(f'Failed to download video {info["title"]}')
                # 设置end_time表示处理结束（虽然失败了）
                if csv_path:
                    end_time = time.strftime('%Y-%m-%d %H:%M:%S')
                    update_csv_file(csv_path, info['webpage_url'], {'end_time': end_time})
                return True
                
            logger.info(f'Process video in {folder}')
            
            # 1. 分离音频
            if status_mask[0] == '1':
                logger.info(f'Step 1: Separating audio in {folder}')
                separate_audio(folder, model_name=demucs_model, device=device, progress=True, shifts=shifts)
                # 更新CSV状态
                if csv_path:
                    update_csv_file(csv_path, info['webpage_url'], {'step1_status': 'completed'})
                # 清理资源
                try:
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.debug('✅ 音频分离后资源清理完成')
                except Exception as e:
                    logger.warning(f'音频分离后清理资源出错: {e}')
            else:
                logger.info(f'Step 1: Skipped audio separation (mask: {status_mask[0]})')
            
            # 2. 语音识别
            if status_mask[1] == '1':
                logger.info(f'Step 2: Transcribing audio in {folder}')
                transcribe_audio(folder, model_name=whisper_model, download_root=whisper_download_root, device=device, batch_size=whisper_batch_size, diarization=whisper_diarization, min_speakers=whisper_min_speakers, max_speakers=whisper_max_speakers)
                # 更新CSV状态
                if csv_path:
                    update_csv_file(csv_path, info['webpage_url'], {'step2_status': 'completed'})
                # 清理资源
                try:
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.debug('✅ 语音识别后资源清理完成')
                except Exception as e:
                    logger.warning(f'语音识别后清理资源出错: {e}')
            else:
                logger.info(f'Step 2: Skipped audio transcription (mask: {status_mask[1]})')
            
            # 3. 翻译文本
            if status_mask[2] == '1':
                logger.info(f'Step 3: Translating transcript in {folder}')
                translate_advanced(folder, target_language=translation_target_language)
                # 更新CSV状态
                if csv_path:
                    update_csv_file(csv_path, info['webpage_url'], {'step3_status': 'completed'})
            else:
                logger.info(f'Step 3: Skipped transcript translation (mask: {status_mask[2]})')
            
            # 4. 提取说话人克隆音频
            if status_mask[3] == '1':
                logger.info(f'Step 4: Extracting speaker clips in {folder}')
                extract_speaker_clips_for_folder(folder)
                # 更新CSV状态
                if csv_path:
                    update_csv_file(csv_path, info['webpage_url'], {'step4_status': 'completed'})
            else:
                logger.info(f'Step 4: Skipped speaker clip extraction (mask: {status_mask[3]})')
            
            # 5. 生成TTS音频
            if status_mask[4] == '1':
                logger.info(f'Step 5: Generating TTS audio in {folder}')
                generate_wavs(folder)
                # 更新CSV状态
                if csv_path:
                    update_csv_file(csv_path, info['webpage_url'], {'step5_status': 'completed'})
                # 清理资源
                try:
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    logger.debug('✅ TTS生成后资源清理完成')
                except Exception as e:
                    logger.warning(f'TTS生成后清理资源出错: {e}')
            else:
                logger.info(f'Step 5: Skipped TTS generation (mask: {status_mask[4]})')
            
            # 6. 合成视频
            if status_mask[5] == '1':
                logger.info(f'Step 6: Synthesizing video in {folder}')
                synthesize_video(folder, subtitles=subtitles, speed_up=speed_up, fps=fps, resolution=target_resolution)
                # 更新CSV状态
                if csv_path:
                    update_csv_file(csv_path, info['webpage_url'], {'step6_status': 'completed'})
            else:
                logger.info(f'Step 6: Skipped video synthesis (mask: {status_mask[5]})')
            
            # 7. 生成信息
            if status_mask[6] == '1':
                logger.info(f'Step 7: Generating info in {folder}')
                generate_info(folder)
                # 更新CSV状态
                if csv_path:
                    update_csv_file(csv_path, info['webpage_url'], {'step7_status': 'completed'})
            else:
                logger.info(f'Step 7: Skipped info generation (mask: {status_mask[6]})')
            
            # 记录结束时间
            end_time = time.strftime('%Y-%m-%d %H:%M:%S')
            if csv_path:
                update_csv_file(csv_path, info['webpage_url'], {'end_time': end_time})
            
            logger.success(f'✅ Video processing completed: {folder}')
            return True
        except Exception as e:
            logger.error(f'Error processing video {info["title"]}: {e}')
            # 记录错误信息
            if csv_path:
                update_csv_file(csv_path, info['webpage_url'], {'status': f'error: {str(e)[:100]}'})
        finally:
            # 在每个视频处理完成后清理资源
            try:
                logger.info('🔄 清理资源，准备处理下一个视频...')
                import torch
                import gc
                
                # 清理PyTorch缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 强制垃圾回收
                gc.collect()
                logger.debug('✅ 资源清理完成')
            except Exception as e:
                logger.warning(f'清理资源时出错: {e}')
    return False



def do_everything(root_folder, url, num_videos=5, resolution='1080p', demucs_model='htdemucs_ft', device='auto', shifts=5, whisper_model='large-v3', whisper_download_root='models/ASR/whisper', whisper_batch_size=32, whisper_diarization=True, whisper_min_speakers=None, whisper_max_speakers=None, translation_target_language='简体中文', subtitles=True, speed_up=1.05, fps=30, target_resolution='1080p', max_workers=3, max_retries=5):
    success_list = []
    fail_list = []

    # 检查是否是CSV文件路径
    is_csv = False
    csv_path = None
    if isinstance(url, str) and url.endswith('.csv'):
        is_csv = True
        csv_path = url
        logger.info(f'检测到CSV文件输入: {csv_path}')
        
        # 初始化CSV文件
        initialize_csv_file(csv_path)
        
        # 从CSV文件中获取所有URL
        import csv
        urls = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_url = row.get('url')
                if row_url:
                    urls.append(row_url)
    else:
        # 处理普通URL输入
        url = url.replace(' ', '').replace('，', '\n').replace(',', '\n')
        urls = [_ for _ in url.split('\n') if _]
    
    # 使用线程池执行任务
    with ThreadPoolExecutor() as executor:
        # 初始化模型
        executor.submit(init_demucs)
        executor.submit(init_whisperx)

    # 获取视频信息列表，传递CSV路径
    video_info_list = get_info_list_from_url(urls, num_videos, csv_path=csv_path)
    
    # 处理每个视频
    for info in video_info_list:
        # 如果是CSV输入，从CSV中获取该URL的状态掩码
        if is_csv:
            import csv
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('url') == info.get('webpage_url'):
                        # 优先使用CSV中的状态掩码
                        status_mask = row.get('status_mask')
                        if status_mask:
                            info['status_mask'] = status_mask
                        # 优先使用CSV中的分辨率
                        csv_resolution = row.get('resolution')
                        if csv_resolution:
                            resolution = csv_resolution
                        break
        
        success = process_video(info, root_folder, resolution, demucs_model, device, shifts, whisper_model, whisper_download_root, whisper_batch_size,
                                whisper_diarization, whisper_min_speakers, whisper_max_speakers, translation_target_language, subtitles, speed_up, fps, target_resolution, max_retries)
        if success:
            success_list.append(info)
        else:
            fail_list.append(info)

    # 在所有视频处理完成后卸载所有模型
    try:
        logger.info('🔄 所有视频处理完成，正在卸载所有模型...')
        
        # 卸载所有模型
        from .step010_demucs_vr import unload_demucs_model
        from .step020_whisperx_silero_vad import unload_whisper_model
        from .step040_tts_vox_cpm_qwen import unload_voxcpm_model
        
        unload_demucs_model()
        unload_whisper_model()
        unload_voxcpm_model()
        
        # 最后一次清理
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        logger.success('✅ 所有模型已卸载，资源已彻底释放')
    except Exception as e:
        logger.warning(f'卸载模型时出错: {e}')

    return f'Success: {len(success_list)}\nFail: {len(fail_list)}'



def main():
    parser = argparse.ArgumentParser(
        description="YouTube 视频自动翻译配音流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理 tasks.csv 中的所有视频（完整流程）
  python do_everything.py
  
  # 从第3步（翻译）开始执行
  python do_everything.py --step 3
  
  # 只执行前3步（下载+分离+识别）
  python do_everything.py --end 3
  
  # 使用自定义配置
  python do_everything.py --model medium --lang 英语
  
  # 处理单个视频
  python do_everything.py --url "https://www.youtube.com/watch?v=xxx"
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
        "--url", type=str, default=None,
        help="处理单个视频URL"
    )
    
    args = parser.parse_args()
    
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
    
    try:
        logger.info("\n" + "🎬"*30)
        logger.info("YouTube 视频翻译配音流水线启动")
        logger.info("🎬"*30 + "\n")
        
        # 根据命令行参数设置 status_mask
        status_mask = "1111111"
        if args.end is not None:
            # 如果指定了结束步骤，则将该步骤之后的步骤标记为不执行
            status_mask = "1" * args.end + "0" * (7 - args.end)
        if args.step > 0:
            # 如果指定了开始步骤，则将该步骤之前的步骤标记为不执行
            status_mask = "0" * args.step + status_mask[args.step:]
        
        # 准备参数
        url = args.url or "tasks.csv"  # 如果没有指定URL，则使用tasks.csv
        
        # 调用 do_everything 函数处理视频
        result = do_everything(
            root_folder=args.folder,
            url=url,
            num_videos=5,  # 默认处理5个视频
            resolution="1080p",
            whisper_model=args.model,
            translation_target_language=args.lang,
            subtitles=not args.no_subtitles,
            speed_up=args.speed,
            target_resolution="1080p"
        )
        
        logger.info("\n" + "🎉"*30)
        logger.success("所有步骤执行完成！")
        logger.info(f"📋 处理结果: {result}")
        logger.info("🎉"*30 + "\n")
        
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"💥 流水线执行失败: {e}")
        logger.exception("详细错误:")
        sys.exit(1)


if __name__ == "__main__":
    main()