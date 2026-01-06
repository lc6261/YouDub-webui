"""
YouTube 视频下载工具（增强版）
支持从文件批量导入URL下载

✅ 支持的文件格式：
- JSON (.json): [{"url": "https://...", "resolution": "1080p", "mode": "all"}]
- TXT (.txt): 每行一个URL
- CSV (.csv): url,resolution,mode 格式（无需output_folder）
- 直接URL字符串

✅ 自动文件夹命名：videos/上传者/上传日期 标题/

作者: [Your Name]
创建时间: 2025-12-29
更新: 2026-01-01 - 简化CSV格式，移除output_folder
版本: 1.0
"""

import os
import re
import json
import time
import csv
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import yt_dlp


@dataclass
class DownloadTask:
    """下载任务配置"""
    url: str
    resolution: str = "1080p"
    mode: str = "all"  # all, video, audio, merge, separate


def sanitize_title(title: str) -> str:
    """清理标题为安全文件名（兼容中文、英文、数字）"""
    if not title:
        return "Unknown"
    title = re.sub(r'[^\w\u4e00-\u9fff \d\-_]', ' ', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title[:100]


def _resolve_cookies(cookies_file: str | None) -> str | None:
    """自动检测 cookies.txt"""
    if cookies_file and os.path.isfile(cookies_file):
        return cookies_file
    if os.path.isfile('cookies.txt'):
        return 'cookies.txt'
    return None


def _get_output_path(info: dict, folder_path: str, suffix: str = '') -> str:
    """生成输出路径"""
    upload_date = info.get('upload_date') or '00000000'
    title = sanitize_title(info.get('title', 'Unknown'))
    uploader = sanitize_title(info.get('uploader', 'Unknown'))
    output_dir = os.path.join(folder_path, uploader, f"{upload_date} {title}")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f'download{suffix}')


def _common_ydl_opts(cookies_file: str | None, sleep_secs: int = 5) -> dict:
    """返回所有下载共用的基础配置"""
    opts = {
        'extractor_args': {'youtube': {'player_skip': ['webpage', 'configs']}},
        'sleep_interval': sleep_secs,
        'quiet': False,
        'ignoreerrors': True,
        'retries': 3,
        'fragment_retries': 3,
        'no_warnings': False,
    }
    cookies = _resolve_cookies(cookies_file)
    if cookies:
        opts['cookiefile'] = cookies
        logger.debug(f"Using cookies from: {cookies}")
    else:
        logger.warning("No cookies.txt found. May fail on bot-protected videos.")
    return opts


def parse_input_source(input_source: str) -> List[DownloadTask]:
    """
    解析输入源，支持多种格式
    
    Args:
        input_source: 可以是：
            - URL字符串
            - JSON文件路径 (.json)
            - TXT文件路径 (.txt)
            - CSV文件路径 (.csv)
    
    Returns:
        List[DownloadTask]: 下载任务列表
    """
    tasks = []
    
    # 如果输入的是URL（直接字符串）
    if input_source.startswith(('http://', 'https://')):
        return [DownloadTask(url=input_source)]
    
    # 如果是文件路径
    path = Path(input_source)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_source}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.json':
        # JSON格式: [{"url": "...", "resolution": "1080p", "mode": "all"}]
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]  # 单个对象转为列表
            for item in data:
                task = DownloadTask(
                    url=item.get('url'),
                    resolution=item.get('resolution', '1080p'),
                    mode=item.get('mode', 'all')
                )
                if task.url:
                    tasks.append(task)
    
    elif suffix == '.txt':
        # TXT格式: 每行一个URL，可选附加参数
        # 格式: url [resolution] [mode]
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 1:
                    task = DownloadTask(url=parts[0])
                    if len(parts) >= 2:
                        task.resolution = parts[1]
                    if len(parts) >= 3:
                        task.mode = parts[2]
                    tasks.append(task)
    
    elif suffix == '.csv':
        # CSV格式: url,resolution,mode
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = DownloadTask(
                    url=row.get('url'),
                    resolution=row.get('resolution', '1080p'),
                    mode=row.get('mode', 'all')
                )
                if task.url:
                    tasks.append(task)
    
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，支持: .json, .txt, .csv")
    
    logger.info(f"从 {input_source} 解析到 {len(tasks)} 个下载任务")
    return tasks


def download_video_only(info: dict, folder_path: str, resolution: str = '1080p', cookies_file: str = None) -> str | None:
    """仅下载视频"""
    output_path = _get_output_path(info, folder_path, '_video')
    if any(os.path.isfile(output_path + ext) for ext in ['.mp4', '.webm']):
        logger.info(f"视频已存在: {output_path}")
        return os.path.dirname(output_path)

    try:
        max_h = int(resolution.lower().rstrip('p'))
    except (ValueError, AttributeError):
        max_h = 1080

    ydl_opts = _common_ydl_opts(cookies_file)
    ydl_opts.update({
        'format': f'bestvideo[height<={max_h}][ext=webm]/bestvideo[height<={max_h}][ext=mp4]/bestvideo',
        'outtmpl': output_path,
        'writethumbnail': False,
        'writeinfojson': True,
    })

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([info['webpage_url']])
        logger.info(f"仅视频保存: {output_path}")
        return os.path.dirname(output_path)
    except Exception as e:
        logger.error(f"视频下载失败: {e}")
        return None


def download_audio_only(info: dict, folder_path: str, cookies_file: str = None) -> str | None:
    """仅下载音频"""
    output_path = _get_output_path(info, folder_path, '_audio')
    if any(os.path.isfile(output_path + ext) for ext in ['.m4a', '.webm']):
        logger.info(f"音频已存在: {output_path}")
        return os.path.dirname(output_path)

    ydl_opts = _common_ydl_opts(cookies_file)
    ydl_opts.update({
        'format': 'bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio',
        'outtmpl': output_path,
        'writethumbnail': False,
        'writeinfojson': True,
    })

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([info['webpage_url']])
        logger.info(f"仅音频保存: {output_path}")
        return os.path.dirname(output_path)
    except Exception as e:
        logger.error(f"音频下载失败: {e}")
        return None


def download_and_merge(info: dict, folder_path: str, resolution: str = '1080p', cookies_file: str = None) -> str | None:
    """下载并合并"""
    output_path = _get_output_path(info, folder_path)
    if os.path.isfile(output_path + '.mp4'):
        logger.info(f"合并视频已存在: {output_path}.mp4")
        return os.path.dirname(output_path)

    try:
        max_h = int(resolution.lower().rstrip('p'))
    except (ValueError, AttributeError):
        max_h = 1080

    ydl_opts = _common_ydl_opts(cookies_file)
    ydl_opts.update({
        'format': (
            f'bestvideo[height<={max_h}][ext=webm]+bestaudio[ext=webm]/'
            f'bestvideo[height<={max_h}][ext=mp4]+bestaudio[ext=m4a]/'
            f'best[height<={max_h}]/best'
        ),
        'outtmpl': output_path + '.mp4',
        'merge_output_format': 'mp4',
        'writethumbnail': True,
        'writeinfojson': True,
    })

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([info['webpage_url']])
        logger.info(f"合并视频保存: {output_path}.mp4")
        return os.path.dirname(output_path)
    except Exception as e:
        logger.error(f"合并下载失败: {e}")
        return None


def download_separate_files(info: dict, folder_path: str, resolution: str = '1080p', cookies_file: str = None) -> str | None:
    """下载分离文件"""
    v_dir = download_video_only(info, folder_path, resolution, cookies_file)
    time.sleep(2)
    a_dir = download_audio_only(info, folder_path, cookies_file)
    if v_dir and a_dir:
        logger.info(f"分离文件就绪: {v_dir}")
        return v_dir
    return None


def get_video_infos(urls, num_videos: int = 5, cookies_file: str = None):
    """获取视频信息"""
    if isinstance(urls, str):
        urls = [urls]
    ydl_opts = _common_ydl_opts(cookies_file)
    ydl_opts.update({
        'dumpjson': True,
        'playlistend': num_videos,
        'quiet': True,
    })

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            info = ydl.extract_info(url, download=False)
            if not info:
                continue
            entries = info.get('entries') or []
            if entries:
                yield from (e for e in entries if e)
            else:
                yield info


def process_task(task: DownloadTask, base_folder: str, cookies_file: str = None):
    """处理单个下载任务"""
    logger.info(f"开始处理: {task.url} (模式: {task.mode}, 分辨率: {task.resolution})")
    
    for info in get_video_infos(task.url, cookies_file=cookies_file):
        if not info or not info.get('title'):
            continue
            
        logger.info(f"下载: {info['title']}")
        
        # 提取视频元数据
        video_id = info.get('id', '')
        title = info.get('title', '')
        duration = info.get('duration', 0)
        publish_date = info.get('upload_date', '')
        uploader = info.get('uploader', '')
        
        # 更新任务状态和元数据
        update_task_status(task, 'processing', title=title, duration=duration, 
                         video_id=video_id, publish_date=publish_date, uploader=uploader)
        
        try:
            if task.mode == 'audio':
                download_audio_only(info, base_folder, cookies_file)
            
            elif task.mode == 'video':
                download_video_only(info, base_folder, task.resolution, cookies_file)
            
            elif task.mode == 'merge':
                download_and_merge(info, base_folder, task.resolution, cookies_file)
            
            elif task.mode == 'separate':
                download_separate_files(info, base_folder, task.resolution, cookies_file)
            
            elif task.mode == 'all':
                # 执行全部模式
                download_audio_only(info, base_folder, cookies_file)
                time.sleep(2)
                download_video_only(info, base_folder, task.resolution, cookies_file)
                time.sleep(2)
                download_and_merge(info, base_folder, task.resolution, cookies_file)
            
            else:
                logger.warning(f"未知模式: {task.mode}, 跳过")
            
            # 生成输出路径
            upload_date = info.get('upload_date') or '00000000'
            title = sanitize_title(info.get('title', 'Unknown'))
            uploader = sanitize_title(info.get('uploader', 'Unknown'))
            output_path = os.path.join(base_folder, uploader, f"{upload_date} {title}")
            
            # 更新任务状态为completed
            update_task_status(task, 'completed', title=title, duration=duration, 
                             video_id=video_id, publish_date=publish_date, uploader=uploader, 
                             output_path=output_path)
        
        except Exception as e:
            error_msg = str(e)
            logger.error(f"处理视频失败 {info.get('title', 'Unknown')}: {error_msg}")
            update_task_status(task, 'failed', title=title, duration=duration, 
                             video_id=video_id, publish_date=publish_date, uploader=uploader, 
                             error_message=error_msg)
        
        # 任务间延迟
        time.sleep(10)


def update_task_status(task_info, status, title=None, duration=None, video_id=None, publish_date=None, uploader=None, 
                     progress=None, start_time=None, end_time=None, output_path=None, error_message=None):
    """更新任务状态并保存到CSV文件"""
    tasks_file = 'tasks.csv'
    if not os.path.exists(tasks_file):
        return
    
    # 读取现有任务
    with open(tasks_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    # 更新对应任务
    for row in rows:
        if row['url'] == task_info.url:
            row['status'] = status
            if title:
                row['title'] = title
            if duration:
                row['duration'] = duration
            if video_id:
                row['video_id'] = video_id
            if publish_date:
                row['publish_date'] = publish_date
            if uploader:
                row['uploader'] = uploader
            if progress:
                row['progress'] = progress
            if start_time:
                row['start_time'] = start_time
            if end_time:
                row['end_time'] = end_time
            if output_path:
                row['output_path'] = output_path
            if error_message:
                row['error_message'] = error_message
            break
    
    # 写入更新后的任务
    with open(tasks_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """主程序 - 支持文件批量导入和默认值"""
    
    # 设置默认值
    DEFAULT_INPUT = 'tasks.csv'
    DEFAULT_FOLDER = 'videos'
    DEFAULT_COOKIES = 'cookies.txt'
    DEFAULT_LOG = 'download.log'
    
    parser = argparse.ArgumentParser(description='YouTube视频批量下载工具')
    parser.add_argument('input', nargs='?', default=DEFAULT_INPUT,
                       help=f'输入源: URL、JSON/TXT/CSV文件路径 (默认: {DEFAULT_INPUT})')
    parser.add_argument('--folder', default=DEFAULT_FOLDER,
                       help=f'输出文件夹 (默认: {DEFAULT_FOLDER})')
    parser.add_argument('--cookies', default=DEFAULT_COOKIES,
                       help=f'cookies文件路径 (默认: {DEFAULT_COOKIES})')
    parser.add_argument('--log', default=DEFAULT_LOG,
                       help=f'日志文件路径 (默认: {DEFAULT_LOG})')
    parser.add_argument('--limit', type=int, default=0,
                       help='限制下载数量 (0表示无限制)')
    
    args = parser.parse_args()
    
    # 配置日志
    logger.add(args.log, rotation="10 MB", level="INFO")
    
    # 创建输出文件夹
    os.makedirs(args.folder, exist_ok=True)
    
    try:
        # 解析输入源
        tasks = parse_input_source(args.input)
        
        # 如果输入源是tasks.csv，初始化增强字段
        if args.input == DEFAULT_INPUT:
            tasks_file = DEFAULT_INPUT
            with open(tasks_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            # 如果没有增强字段，添加这些字段
                if 'status' not in reader.fieldnames:
                    fieldnames = ['url', 'resolution', 'mode', 'status', 'title', 'duration', 'video_id', 'publish_date', 'uploader', 'progress', 'start_time', 'end_time', 'output_path', 'error_message', 'task_type', 'steps']
                    with open(tasks_file, 'w', encoding='utf-8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for row in rows:
                            # 初始化新字段
                            row.setdefault('status', 'pending')
                            row.setdefault('title', '')
                            row.setdefault('duration', '')
                            row.setdefault('video_id', '')
                            row.setdefault('publish_date', '')
                            row.setdefault('uploader', '')
                            row.setdefault('progress', '')
                            row.setdefault('start_time', '')
                            row.setdefault('end_time', '')
                            row.setdefault('output_path', '')
                            row.setdefault('error_message', '')
                            row.setdefault('task_type', 'full_process')  # 默认为完整处理
                            row.setdefault('steps', '0,1,2,3,4,5,6,7')  # 默认执行所有步骤
                            writer.writerow(row)
                # 如果有status字段但缺少新字段，添加新字段
                else:
                    fieldnames = reader.fieldnames
                    if 'task_type' not in fieldnames:
                        fieldnames.extend(['task_type', 'steps'])
                        with open(tasks_file, 'w', encoding='utf-8', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            for row in rows:
                                row.setdefault('task_type', 'full_process')
                                row.setdefault('steps', '0,1,2,3,4,5,6,7')
                                writer.writerow(row)
        
        if args.limit > 0:
            tasks = tasks[:args.limit]
            logger.info(f"限制下载前 {args.limit} 个任务")
        
        # 只处理状态为pending的任务
        pending_tasks = []
        with open(args.input, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('status') == 'pending':
                    pending_tasks.append(row['url'])
        
        # 处理每个任务
        total = len(pending_tasks)
        processed_count = 0
        for i, task_url in enumerate(pending_tasks, 1):
            # 找到对应的任务
            task = next((t for t in tasks if t.url == task_url), None)
            if not task:
                continue
            
            logger.info(f"进度: {i}/{total}")
            try:
                # 更新状态为processing
                update_task_status(task, 'processing')
                process_task(task, args.folder, args.cookies)
                processed_count += 1
            except Exception as e:
                logger.error(f"任务失败 {task.url}: {e}")
                update_task_status(task, 'failed', error_message=str(e))
            
            # 任务间延迟
            if i < total:
                time.sleep(30)
        
        logger.success(f"所有任务完成! 共处理 {processed_count} 个视频")
        
    except FileNotFoundError as e:
        # 如果是默认tasks.csv不存在，提示用户
        if args.input == DEFAULT_INPUT and not Path(DEFAULT_INPUT).exists():
            logger.error(f"默认任务文件 {DEFAULT_INPUT} 不存在。请创建该文件或指定其他输入源。")
            print(f"\n请创建 {DEFAULT_INPUT} 文件，格式如下：")
            print("========================================")
            print("url,resolution,mode,status,title,duration,video_id,publish_date,uploader")
            print("https://youtube.com/watch?v=xxx,1080p,all,pending,,,,,")
            print("https://youtube.com/watch?v=yyy,720p,audio,pending,,,,,")
            print("https://youtube.com/watch?v=zzz,1080p,separate,pending,,,,,")
            print("========================================")
        else:
            logger.error(f"文件未找到: {e}")
        return 1
    
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    # 现在可以直接运行，使用默认值：
    # python step000_video_downloader.py
    
    # 或者指定参数：
    # python step000_video_downloader.py my_tasks.json --folder my_videos --cookies my_cookies.txt
    
    exit(main())