# 查看所有可用模型
from TTS.api import TTS

# 打印所有模型
print("所有可用模型:")
for model_name in TTS.list_models():
    print(f"  - {model_name}")

# 专门过滤中文相关的模型
print("\n中文相关模型:")
chinese_models = []
for model_name in TTS.list_models():
    if any(keyword in model_name.lower() for keyword in ['chinese', 'zh', 'mandarin', 'multilingual']):
        chinese_models.append(model_name)
        print(f"  - {model_name}")