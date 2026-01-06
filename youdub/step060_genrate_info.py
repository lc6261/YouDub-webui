"""
视频信息生成工具
功能：批量处理视频文件夹，生成摘要文本和调整缩略图尺寸
作者：[Your Name]
日期：2023-10-27
版本：1.0
"""

import json
import os
from PIL import Image


def resize_thumbnail(folder, size=(1280, 960)):
    """
    调整缩略图尺寸并居中放置到指定大小的黑色背景上
    
    参数：
        folder: 包含图片的文件夹路径
        size: 目标尺寸，默认为(1280, 960)
    
    返回：
        new_img_path: 生成的新图片路径
    
    处理流程：
        1. 查找文件夹中的图片文件
        2. 计算保持宽高比的缩放尺寸
        3. 将缩放后的图片居中放置在黑色背景上
        4. 保存为新图片文件
    """
    print(f"开始处理缩略图，文件夹: {folder}")
    
    # 支持的图片格式
    image_suffix = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_path = None
    
    # 查找文件夹中的图片文件
    for suffix in image_suffix:
        temp_path = os.path.join(folder, f'download{suffix}')
        if os.path.exists(temp_path):
            image_path = temp_path
            print(f"找到图片文件: {image_path}")
            break
    
    if image_path is None:
        print(f"错误: 在文件夹 {folder} 中未找到图片文件")
        return None
    
    try:
        # 打开图片文件
        with Image.open(image_path) as img:
            print(f"原始图片尺寸: {img.width} x {img.height}")
            
            # 计算原始图片和目标尺寸的宽高比
            img_ratio = img.width / img.height
            target_ratio = size[0] / size[1]
            print(f"图片宽高比: {img_ratio:.2f}, 目标宽高比: {target_ratio:.2f}")
            
            # 根据宽高比计算缩放后的尺寸
            if img_ratio < target_ratio:
                # 图片比目标比例更宽，固定高度
                new_height = size[1]
                new_width = int(new_height * img_ratio)
                print(f"固定高度缩放: {new_width} x {new_height}")
            else:
                # 图片比目标比例更高，固定宽度
                new_width = size[0]
                new_height = int(new_width / img_ratio)
                print(f"固定宽度缩放: {new_width} x {new_height}")
            
            # 使用高质量算法缩放图片
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"图片缩放完成")
            
            # 创建黑色背景图片
            new_img = Image.new('RGB', size, "black")
            
            # 计算居中位置
            x_offset = (size[0] - new_width) // 2
            y_offset = (size[1] - new_height) // 2
            print(f"居中偏移: x={x_offset}, y={y_offset}")
            
            # 将缩放后的图片粘贴到黑色背景上
            new_img.paste(img, (x_offset, y_offset))
            
            # 保存新图片
            new_img_path = os.path.join(folder, 'video.png')
            new_img.save(new_img_path)
            print(f"缩略图已保存: {new_img_path}")
            
            return new_img_path
            
    except Exception as e:
        print(f"处理图片时出错: {e}")
        return None


def generate_summary_txt(folder):
    """
    生成视频摘要文本文件，包含中文翻译内容
    
    参数：
        folder: 视频文件夹路径
    
    处理流程：
        1. 读取download.info.json获取视频元数据
        2. 读取translation.json获取中文翻译
        3. 生成格式化文本
        4. 保存为video.txt文件
    """
    print(f"开始生成摘要文本，文件夹: {folder}")
    
    try:
        # 1. 读取download.info.json获取视频元数据
        info_json_path = os.path.join(folder, 'download.info.json')
        title = "未知标题"
        author = "未知作者"
        duration = "0:00"
        
        if os.path.exists(info_json_path):
            with open(info_json_path, 'r', encoding='utf-8') as f:
                info = json.load(f)
            title = info.get("fulltitle", "未知标题")
            author = info.get("uploader", "未知作者")
            duration = info.get("duration_string", "0:00")
            
        print(f"标题: {title}")
        print(f"作者: {author}")
        print(f"时长: {duration}")
        
        # 2. 读取translation.json获取中文翻译
        translation_json_path = os.path.join(folder, 'translation.json')
        chinese_translation = ""
        
        if os.path.exists(translation_json_path):
            with open(translation_json_path, 'r', encoding='utf-8') as f:
                translation_data = json.load(f)
            
            # 提取所有中文翻译内容
            for segment in translation_data:
                translation = segment.get("translation", "")
                if translation:
                    chinese_translation += translation + "\n"
            
            # 去除末尾多余换行
            chinese_translation = chinese_translation.strip()
            
        print(f"中文翻译长度: {len(chinese_translation)} 字符")
        
        # 3. 生成格式化文本
        formatted_title = f"{title} - {author}"
        txt_content = f"{formatted_title}\n"
        txt_content += f"时长: {duration}\n\n"
        txt_content += "=== 中文翻译 ===\n"
        txt_content += chinese_translation
        
        # 4. 保存文本文件（使用UTF-8编码，确保中文正确显示）
        txt_path = os.path.join(folder, 'video.txt')
        with open(txt_path, 'w', encoding='utf-8-sig') as f:
            f.write(txt_content)
        
        print(f"摘要文本已保存: {txt_path}")
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"生成摘要文本时出错: {e}")


def generate_info(folder):
    """
    生成视频信息：包括摘要文本和缩略图
    
    参数：
        folder: 视频文件夹路径
    """
    print(f"\n{'='*50}")
    print(f"开始处理文件夹: {folder}")
    print(f"{'='*50}")
    
    # 生成摘要文本
    generate_summary_txt(folder)
    
    # 调整缩略图尺寸
    resize_thumbnail(folder)
    
    print(f"文件夹处理完成: {folder}")
    print(f"{'='*50}\n")


def generate_all_info_under_folder(root_folder):
    """
    遍历指定根目录下的所有文件夹，批量生成视频信息
    
    参数：
        root_folder: 根目录路径
    
    返回：
        str: 处理结果信息
    
    处理流程：
        1. 递归遍历根目录
        2. 查找包含download.info.json的文件夹
        3. 对每个符合条件的文件夹调用generate_info函数
    """
    print(f"开始批量处理根目录: {root_folder}")
    
    if not os.path.exists(root_folder):
        print(f"错误: 根目录不存在: {root_folder}")
        return f"错误: 根目录不存在: {root_folder}"
    
    processed_count = 0
    error_count = 0
    
    # 遍历根目录下的所有文件夹
    for root, dirs, files in os.walk(root_folder):
        # 检查是否包含download.info.json文件
        if 'download.info.json' in files:
            try:
                generate_info(root)
                processed_count += 1
                print(f"已处理数量: {processed_count}")
            except Exception as e:
                error_count += 1
                print(f"处理文件夹时出错 {root}: {e}")
    
    result_message = (
        f"批量处理完成！\n"
        f"根目录: {root_folder}\n"
        f"成功处理: {processed_count} 个文件夹\n"
        f"处理失败: {error_count} 个文件夹"
    )
    
    print(result_message)
    return result_message


if __name__ == '__main__':
    """
    主程序入口
    当直接运行此脚本时，自动处理'videos'目录下的所有视频文件夹
    """
    print("视频信息生成工具启动")
    print("版本：1.0")
    print("-" * 30)
    
    # 指定要处理的根目录
    root_directory = 'videos'
    
    # 执行批量处理
    result = generate_all_info_under_folder(root_directory)
    
    print("\n程序执行完成")
    print("-" * 30)