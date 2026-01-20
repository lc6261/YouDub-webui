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
    status_mask: str = "1111111"  # 7位掩码，每位对应一个步骤（1=执行，0=跳过）
    csv_path: str = None  # CSV文件路径，用于更新状态


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
        # CSV格式: url,resolution,mode,status_mask,step1_status,...
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = DownloadTask(
                    url=row.get('url'),
                    resolution=row.get('resolution', '1080p'),
                    mode=row.get('mode', 'all'),
                    status_mask=row.get('status_mask', '1111111'),
                    csv_path=str(path)
                )
                if task.url:
                    tasks.append(task)
    
    else:
        raise ValueError(f"不支持的文件格式: {suffix}，支持: .json, .txt, .csv")
    
    logger.info(f"从 {input_source} 解析到 {len(tasks)} 个下载任务")
    return tasks


def update_csv_file(csv_path, url, updates):
    """
    更新CSV文件中指定URL的行
    
    Args:
        csv_path: CSV文件路径
        url: 要更新的视频URL
        updates: 要更新的字段字典
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV文件不存在: {csv_path}")
        return
    
    # 读取所有行
    rows = []
    headers = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        for row in reader:
            if row.get('url') == url:
                # 更新匹配的行
                row.update(updates)
            rows.append(row)
    
    # 确保所有新字段都在表头中
    for field in updates.keys():
        if field not in headers:
            headers.append(field)
    
    # 写回CSV文件
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"已更新CSV文件: {csv_path} 中的URL: {url}")



def initialize_csv_file(csv_path):
    """
    初始化CSV文件，添加所有必要的表头
    
    Args:
        csv_path: CSV文件路径
    """
    if not os.path.exists(csv_path):
        logger.error(f"CSV文件不存在: {csv_path}")
        return
    
    # 读取所有行
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return  # 文件为空
    
    # 定义所有必要的表头
    required_headers = [
        'url', 'resolution', 'mode', 'status_mask',
        'step1_status', 'step2_status', 'step3_status', 'step4_status',
        'step5_status', 'step6_status', 'step7_status',
        'start_time', 'end_time',
        'video_title', 'video_uploader', 'video_duration', 'video_upload_date'
    ]
    
    # 获取当前表头
    current_headers = reader.fieldnames or []
    
    # 添加缺失的表头
    new_headers = current_headers.copy()
    for header in required_headers:
        if header not in new_headers:
            new_headers.append(header)
    
    # 如果表头没有变化，直接返回
    if new_headers == current_headers:
        return
    
    # 写回CSV文件
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_headers)
        writer.writeheader()
        
        # 确保每行都有所有新字段
        for row in rows:
            for header in new_headers:
                if header not in row:
                    row[header] = '' if header != 'status_mask' else '1111111'
            writer.writerow(row)
    
    logger.info(f"已初始化CSV文件表头: {csv_path}")


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
        
        except Exception as e:
            logger.error(f"处理视频失败 {info.get('title', 'Unknown')}: {e}")
        
        # 任务间延迟，防止限流
        time.sleep(10)


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
        
        if args.limit > 0:
            tasks = tasks[:args.limit]
            logger.info(f"限制下载前 {args.limit} 个任务")
        
        # 处理每个任务
        total = len(tasks)
        for i, task in enumerate(tasks, 1):
            logger.info(f"进度: {i}/{total}")
            try:
                process_task(task, args.folder, args.cookies)
            except Exception as e:
                logger.error(f"任务失败 {task.url}: {e}")
            
            # 任务间延迟
            if i < total:
                time.sleep(30)
        
        logger.success(f"所有任务完成! 共处理 {total} 个视频")
        
    except FileNotFoundError as e:
        # 如果是默认tasks.csv不存在，提示用户
        if args.input == DEFAULT_INPUT and not Path(DEFAULT_INPUT).exists():
            logger.error(f"默认任务文件 {DEFAULT_INPUT} 不存在。请创建该文件或指定其他输入源。")
            print(f"\n请创建 {DEFAULT_INPUT} 文件，格式如下：")
            print("========================================")
            print("url,resolution,mode")
            print("https://youtube.com/watch?v=xxx,1080p,all")
            print("https://youtube.com/watch?v=yyy,720p,audio")
            print("https://youtube.com/watch?v=zzz,1080p,separate")
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