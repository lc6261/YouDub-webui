#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测所有任务的完成情况，不仅检查status字段，还验证实际文件是否存在
"""

import os
import csv
from pathlib import Path

def check_task_completion():
    """
    检查所有任务的完成情况
    """
    # 获取当前目录
    current_dir = Path(__file__).resolve().parent
    tasks_csv = current_dir / "tasks.csv"
    
    # 检查tasks.csv是否存在
    if not tasks_csv.exists():
        print("❌ 错误: tasks.csv 文件不存在")
        return
    
    print("🎬 开始检测所有任务的完成情况...")
    print("=" * 80)
    
    # 统计信息
    total_tasks = 0
    completed_tasks = 0
    status_mismatch_tasks = 0
    
    with open(tasks_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_tasks += 1
            
            url = row.get('url', '')
            status = row.get('status', '')
            task_type = row.get('task_type', 'download_only')
            output_path = row.get('output_path', '')
            video_id = row.get('video_id', '')
            
            # 构建视频文件夹路径
            if output_path:
                video_folder = current_dir / output_path
            else:
                # 如果没有output_path，尝试从其他字段构建
                title = row.get('title', '')
                uploader = row.get('uploader', '')
                publish_date = row.get('publish_date', '')
                if uploader and title and publish_date:
                    video_folder = current_dir / "videos" / uploader / f"{publish_date} {title}"
                else:
                    video_folder = current_dir / "videos" / video_id
            
            # 修复：正确处理CSV列，防止错位
            # 重新解析行，确保正确获取task_type
            # 检查task_type是否是有效的值
            valid_task_types = ['download_only', 'full_process']
            actual_task_type = task_type
            
            # 如果task_type不是有效的值，可能是列错位了，尝试检查steps字段
            if actual_task_type not in valid_task_types:
                # 尝试从steps字段获取task_type（处理列错位情况）
                steps_value = row.get('steps', '')
                if steps_value in valid_task_types:
                    actual_task_type = steps_value
                else:
                    # 检查URL是否已经下载
                    download_mp4 = video_folder / "download.mp4"
                    if download_mp4.exists():
                        # 检查是否有完整处理的标记
                        video_mp4 = video_folder / "video.mp4"
                        if video_mp4.exists():
                            actual_task_type = 'full_process'
                        else:
                            actual_task_type = 'download_only'
                    else:
                        # 默认值
                        actual_task_type = 'download_only'
            
            # 检查文件是否存在
            if actual_task_type == 'download_only':
                # 对于download_only任务，只需要检查download.mp4是否存在
                required_file = video_folder / "download.mp4"
                file_exists = required_file.exists()
                expected_status = "completed" if file_exists else "pending"
            else:  # full_process
                # 对于full_process任务，检查最终的video.mp4是否存在
                required_file = video_folder / "video.mp4"
                file_exists = required_file.exists()
                expected_status = "completed" if file_exists else "pending"
            
            # 检查状态是否匹配
            status_match = (status == expected_status)
            if not status_match:
                status_mismatch_tasks += 1
            
            # 确定任务是否真正完成
            is_completed = file_exists
            if is_completed:
                completed_tasks += 1
            
            # 输出任务信息
            print(f"📋 任务 {total_tasks}:")
            print(f"   URL: {url}")
            print(f"   任务类型: {actual_task_type}")
            print(f"   状态字段: {status}")
            print(f"   实际状态: {'已完成' if is_completed else '未完成'}")
            print(f"   视频文件夹: {video_folder}")
            print(f"   检查文件: {required_file}")
            print(f"   文件存在: {'✅' if file_exists else '❌'}")
            if not status_match:
                print(f"   ⚠️  状态不匹配: 应该是 '{expected_status}'")
            print()
    
    # 输出汇总信息
    print("=" * 80)
    print("📊 任务完成情况汇总:")
    print(f"   总任务数: {total_tasks}")
    print(f"   实际完成数: {completed_tasks}")
    print(f"   状态不匹配数: {status_mismatch_tasks}")
    
    if completed_tasks == total_tasks:
        print("🎉 所有任务都已完成！")
    else:
        print(f"⚠️  还有 {total_tasks - completed_tasks} 个任务未完成")
    
    if status_mismatch_tasks > 0:
        print(f"⚠️  有 {status_mismatch_tasks} 个任务状态字段与实际情况不符")
    
    print("=" * 80)

if __name__ == "__main__":
    check_task_completion()
