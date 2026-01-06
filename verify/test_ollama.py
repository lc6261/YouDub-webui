# -*- coding: utf-8 -*-
"""
模型翻译能力验证脚本
用于测试 Ollama / LM Studio / OpenAI 兼容模型是否能正确响应翻译请求
"""

import os
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 配置
MODEL_NAME = os.getenv('MODEL_NAME', 'mistral:7b-instruct').strip()
API_BASE = os.getenv('OPENAI_API_BASE', '').strip()
API_KEY = os.getenv('OPENAI_API_KEY', '').strip()

# 自动设置（Ollama 专用）
if not API_BASE:
    API_BASE = 'http://127.0.0.1:11434/v1'
if not API_KEY:
    API_KEY = 'ollama'

print("🔧 配置信息:")
print(f"  模型: {MODEL_NAME}")
print(f"  API 地址: {API_BASE}")
print(f"  API Key: {'✅ 已设置' if API_KEY else '❌ 未设置'}")

# 创建客户端
client = OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=60)

# 测试句子
test_sentence = "Hello, how are you today?"
expected_lang = "简体中文"

print(f"\n📤 发送翻译请求: \"{test_sentence}\" → {expected_lang}")

try:
    start = time.time()
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": f"你是一位专业翻译，请将英文翻译成{expected_lang}。只输出译文，不要任何解释、前缀、后缀或标点。"
            },
            {
                "role": "user",
                "content": test_sentence
            }
        ],
        max_tokens=50,
        temperature=0.1,
        timeout=30
    )
    
    elapsed = time.time() - start
    output = response.choices[0].message.content.strip()
    
    print(f"\n⏱️  响应时间: {elapsed:.2f} 秒")
    print(f"📥 原始输出: [{repr(output)}]")
    
    if not output:
        print("❌ 失败: 模型返回空内容")
    elif len(output) < 2:
        print("⚠️ 警告: 输出过短，可能无效")
    elif "hello" in output.lower() or "how are you" in output.lower():
        print("❌ 失败: 模型未翻译，直接返回原文或英文")
    elif any(word in output for word in ["翻译", "translate", "Translate", "输出", "结果", "assistant"]):
        print("❌ 失败: 模型输出了多余解释")
    else:
        print(f"✅ 成功! 翻译结果: {output}")
        
    # 可选：手动判断
    print("\n❓ 请人工判断翻译是否合理（应为中文且自然）")

except Exception as e:
    print(f"💥 请求失败: {e}")
    
    # 尝试直接访问 Ollama tag 列表判断服务是否运行
    try:
        tags_url = API_BASE.replace('/v1', '/api/tags')
        r = requests.get(tags_url, timeout=5)
        if r.status_code == 200:
            print("✅ Ollama 服务正在运行")
        else:
            print(f"⚠️ Ollama 服务响应异常: {r.status_code}")
    except:
        print("❌ 无法连接到 Ollama 服务")