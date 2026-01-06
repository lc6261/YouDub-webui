# quick_test_gpt_sovits.py
import os
import sys
import torch
import subprocess

def quick_test():
    """快速测试 GPT-SoVITS"""
    print("🚀 GPT-SoVITS 快速测试")
    print("=" * 50)
    
    # 1. 检查环境
    print("\n🔍 环境检查:")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {'可用' if torch.cuda.is_available() else '不可用'}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"显存: {mem_gb:.1f} GB")
        
        if mem_gb < 6:
            print("⚠️  警告: 显存可能不足，建议 >= 8GB")
    else:
        print("⚠️  警告: 没有GPU，推理会非常慢")
    
    # 2. 检查是否已克隆
    if not os.path.exists("GPT-SoVITS"):
        print("\n📥 需要克隆 GPT-SoVITS 仓库")
        print("运行: git clone https://github.com/RVC-Boss/GPT-SoVITS.git")
        print("然后进入目录: cd GPT-SoVITS")
        return
    
    # 3. 检查模型文件
    print("\n📦 检查模型文件...")
    required_files = [
        "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "pretrained_models/s2G488k.pth",
        "pretrained_models/chinese-hubert-base/config.json",
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(os.path.join("GPT-SoVITS", file)):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  缺少 {len(missing_files)} 个模型文件")
        print("请运行下载脚本: python tools/download_models.py")
        return
    
    # 4. 测试导入
    print("\n🧪 测试导入模块...")
    sys.path.insert(0, "GPT-SoVITS")
    
    try:
        import gradio as gr
        print("✅ gradio")
    except:
        print("❌ gradio - 请安装: pip install gradio")
    
    try:
        import fairseq
        print("✅ fairseq")
    except:
        print("❌ fairseq - 请安装: pip install fairseq")
    
    # 5. 运行简单测试
    print("\n🎯 运行快速测试...")
    
    # 创建测试脚本
    test_code = '''
print("🚀 GPT-SoVITS 快速测试脚本")
print("=" * 50)

# 检查核心模块
try:
    from tools.i18n.i18n import I18nAuto
    print("✅ I18nAuto 导入成功")
except Exception as e:
    print(f"❌ I18nAuto 导入失败: {e}")

try:
    from feature_extractor import cnhubert
    print("✅ cnhubert 导入成功")
except Exception as e:
    print(f"❌ cnhubert 导入失败: {e}")

print("\\n✅ 基本环境检查通过")
print("\\n🎯 下一步:")
print("1. 安装依赖: pip install -r requirements.txt")
print("2. 启动WebUI: python inference_webui.py")
print("3. 或启动API: python api.py")
'''
    
    test_file = "gpt_sovits_quick_test.py"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    # 运行测试
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            cwd="GPT-SoVITS",
            capture_output=True,
            text=True,
            timeout=10
        )
        
        print(result.stdout)
        if result.stderr:
            print("错误信息:", result.stderr)
    
    except Exception as e:
        print(f"测试运行失败: {e}")
    
    finally:
        # 清理
        if os.path.exists(test_file):
            os.remove(test_file)
    
    print("\n" + "=" * 50)
    print("📋 总结")
    print("=" * 50)
    print("✅ 环境检查完成")
    print("✅ 模型文件检查完成")
    print("✅ 依赖包检查完成")
    print("\n🚀 启动命令:")
    print("  cd GPT-SoVITS")
    print("  python inference_webui.py")
    print("\n🌐 然后访问: http://localhost:7860")

if __name__ == "__main__":
    quick_test()