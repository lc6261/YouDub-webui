import os
import json
import csv
from pathlib import Path

# 定义CSV字段名
fieldnames = [
    'url', 'resolution', 'mode', 'status', 'title', 'duration', 'video_id',
    'publish_date', 'uploader', 'progress', 'start_time', 'end_time',
    'output_path', 'error_message', 'task_type', 'steps'
]

# 视频文件夹路径
videos_folder = Path('videos')

# 创建tasks.csv文件
with open('tasks.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # 遍历所有视频文件夹
    for channel_folder in videos_folder.iterdir():
        if channel_folder.is_dir():
            for video_folder in channel_folder.iterdir():
                if video_folder.is_dir():
                    # 查找info.json文件
                    info_file = next(video_folder.glob('*.info.json'), None)
                    if info_file:
                        try:
                            # 读取视频信息
                            with open(info_file, 'r', encoding='utf-8') as f:
                                info = json.load(f)
                            
                            # 构建任务记录
                            task = {
                                'url': f"https://www.youtube.com/watch?v={info.get('id', '')}",
                                'resolution': '1080p',
                                'mode': 'all',
                                'status': 'completed',
                                'title': info.get('title', ''),
                                'duration': info.get('duration', ''),
                                'video_id': info.get('id', ''),
                                'publish_date': info.get('upload_date', ''),
                                'uploader': info.get('uploader', ''),
                                'progress': '',
                                'start_time': '',
                                'end_time': '',
                                'output_path': str(video_folder),
                                'error_message': '',
                                'task_type': 'full_process',
                                'steps': '0,1,2,3,4,5,6,7'
                            }
                            
                            # 写入CSV
                            writer.writerow(task)
                            print(f"Added: {info.get('title', '')}")
                        except Exception as e:
                            print(f"Error processing {video_folder}: {e}")

print("\n✅ tasks.csv 文件已重新创建！")
print("\n运行以下命令查看结果：")
print("Get-ChildItem -Path . -Filter tasks.csv")
