#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级视频字幕翻译模块（Qwen2.5 + 多轮校对 + 术语库）

改进点：
1. 使用 Qwen2.5-14B（显著优于 Llama3.1-8B）
2. 领域术语自动提取与一致性控制
3. 两遍翻译 + 质量校对
4. Few-shot 示例学习
5. 口语化优化

安装：
ollama pull qwen2.5:14b
# 或使用 Qwen2.5-32B: ollama pull qwen2.5:32b

作者: Advanced Translation Team
日期: 2026-01-03
版本: 1.0
"""

import json
import os
import re
import sys
import time
from typing import List, Dict, Tuple, Optional
from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ===== 配置 =====
MODEL_NAME = os.getenv('MODEL_NAME', 'qwen2.5:14b')
API_BASE = os.getenv('OPENAI_API_BASE', 'http://127.0.0.1:11434/v1')
API_KEY = os.getenv('OPENAI_API_KEY', 'ollama')

logger.info(f"🤖 使用翻译模型: {MODEL_NAME}")
logger.info(f"🌐 API地址: {API_BASE}")

_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=API_BASE, api_key=API_KEY, timeout=120.0)
    return _client


# ===== 术语提取与管理 =====
class TerminologyManager:
    """术语库管理器"""
    
    def __init__(self):
        self.terms = {}
        self.domain_keywords = {
            "AI": ["transformer", "attention", "neural network", "GPT", "LLM", 
                   "embedding", "tokenizer", "fine-tuning", "reinforcement learning"],
            "Math": ["derivative", "integral", "matrix", "vector", "function",
                    "equation", "theorem", "proof", "optimization"],
            "Programming": ["API", "function", "variable", "algorithm", "database",
                          "framework", "compiler", "debugger", "deployment"]
        }
    
    def extract_terms(self, text: str, target_language: str = '简体中文') -> Dict[str, str]:
        """
        从文本中提取关键术语并生成翻译映射
        
        Args:
            text: 输入文本
            target_language: 目标语言
        
        Returns:
            术语映射字典 {英文: 中文}
        """
        # 检测领域
        detected_domains = []
        text_lower = text.lower()
        
        for domain, keywords in self.domain_keywords.items():
            if any(kw.lower() in text_lower for kw in keywords):
                detected_domains.append(domain)
        
        logger.info(f"🔍 检测到领域: {detected_domains or ['通用']}")
        
        # 使用 LLM 提取术语
        prompt = f"""你是专业术语提取专家。请从以下文本中提取需要特别注意翻译的术语。

要求：
1. 提取技术术语、专有名词、关键概念
2. 优先提取：{', '.join(detected_domains)} 领域的术语
3. 每个术语提供准确的{target_language}翻译
4. 输出标准JSON格式

文本（节选）:
{text[:1500]}

输出格式（严格JSON）：
{{
  "transformer": "Transformer模型",
  "attention mechanism": "注意力机制",
  "gradient descent": "梯度下降"
}}"""

        try:
            client = get_client()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是专业术语提取专家，精通技术领域翻译。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            raw = response.choices[0].message.content.strip()
            logger.debug(f"术语提取结果: {raw[:200]}...")
            
            # 解析JSON
            try:
                terms = json.loads(raw)
            except:
                # 正则提取
                terms = {}
                matches = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', raw)
                for en, zh in matches:
                    terms[en] = zh
            
            self.terms.update(terms)
            logger.info(f"✅ 提取到 {len(terms)} 个关键术语")
            return terms
            
        except Exception as e:
            logger.warning(f"⚠️ 术语提取失败: {e}")
            return {}
    
    def apply_terms(self, text: str) -> str:
        """应用术语替换（后处理）"""
        for en, zh in self.terms.items():
            # 精确匹配（考虑大小写）
            pattern = re.compile(re.escape(en), re.IGNORECASE)
            text = pattern.sub(zh, text)
        return text


# ===== 高级翻译器 =====
class AdvancedTranslator:
    """高级翻译器（两遍翻译 + 质量校对）"""
    
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.client = get_client()
        self.term_manager = TerminologyManager()
        
        # Few-shot 示例（教学习翻译风格）
        self.few_shot_examples = [
            {
                "source": "So basically what we're doing here is taking the derivative of the loss function.",
                "target": "所以基本上我们在这里做的就是计算损失函数的导数。",
                "note": "保留所以、基本上等口语化表达"
            },
            {
                "source": "This is a really cool technique that allows us to...",
                "target": "这是一个非常酷的技术，它让我们能够……",
                "note": "\"really cool\" 翻译为 非常酷 而非 真的很酷 "
            },
            {
                "source": "Now, you might be wondering why we use attention here.",
                "target": "现在，你可能会想知道为什么我们在这里使用注意力机制。",
                "note": "保留 你可能会想 等对话感"
            }
        ]
    
    def build_translation_prompt(self,
                                 text: str,
                                 context_prev: List[str],
                                 context_next: List[str],
                                 terms: Dict[str, str],
                                 target_duration: float,
                                 target_language: str = '简体中文') -> str:
        """构建优化的翻译提示词"""
        
        max_chars = int(target_duration * 4.5)
        
        # 术语列表
        term_list = "\n".join([f"- {en} → {zh}" for en, zh in list(terms.items())[:15]])
        
        # Few-shot 示例
        examples = "\n\n".join([
            f"原文: {ex['source']}\n译文: {ex['target']}\n注意: {ex['note']}"
            for ex in self.few_shot_examples[:2]
        ])
        
        # 上下文
        context = []
        for i, t in enumerate(context_prev, 1):
            if t: context.append(f"前{i}句: {t}")
        context.append(f"【当前句】: {text}")
        for i, t in enumerate(context_next, 1):
            if t: context.append(f"后{i}句: {t}")
        context_str = "\n".join(context)
        
        return f"""你是专业视频字幕翻译专家，擅长将英文视频翻译成地道、口语化的{target_language}。

# 翻译原则
1. **自然流畅**: 符合{target_language}表达习惯，不要逐字直译
2. **口语化**: 保留"所以"、"其实"、"那么"等语气词
3. **准确性**: 严格使用术语表，保持全文一致
4. **时长匹配**: 译文约{target_duration:.1f}秒，最多{max_chars}个汉字

# 术语表（必须严格遵守）
{term_list}

# 翻译示例（学习风格）
{examples}

# 待翻译内容（含上下文）
{context_str}

# 输出要求
严格输出JSON格式: {{"translation": "译文"}}
只翻译【当前句】，不要翻译上下文！"""
    
    def translate_first_pass(self,
                            text: str,
                            context_prev: List[str],
                            context_next: List[str],
                            terms: Dict[str, str],
                            target_duration: float,
                            target_language: str = '简体中文') -> str:
        """第一遍翻译（注重准确性）"""
        
        prompt = self.build_translation_prompt(
            text, context_prev, context_next, terms, 
            target_duration, target_language
        )
        
        system_prompt = (
            "你是专业视频翻译专家。严格按照JSON格式输出，"
            "使用口语化表达，保持术语一致性。"
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 较低温度保证准确性
                top_p=0.9,
                max_tokens=250
            )
            
            raw = response.choices[0].message.content.strip()
            translation = self._parse_json(raw)
            
            return translation if translation else text
            
        except Exception as e:
            logger.warning(f"⚠️ 第一遍翻译失败: {e}")
            return text
    
    def refine_translation(self,
                          original: str,
                          first_translation: str,
                          target_duration: float,
                          target_language: str = '简体中文') -> str:
        """第二遍翻译（优化流畅度和时长）"""
        
        max_chars = int(target_duration * 4.5)
        current_chars = len(first_translation)
        
        prompt = f"""你是翻译质量审校专家。请优化以下翻译，使其更加自然流畅。

原文: {original}

初译: {first_translation}

优化要求:
1. 保持原意不变
2. 更加口语化、自然
3. 长度控制在 {max_chars} 个汉字内（当前 {current_chars} 字）
4. 去除冗余，使用更简洁的表达
5. 确保时长匹配语音（约{target_duration:.1f}秒）

输出JSON: {{"refined": "优化后的译文"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是专业翻译审校专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,  # 稍高温度增加创造性
                max_tokens=200
            )
            
            raw = response.choices[0].message.content.strip()
            
            # 解析refined字段
            try:
                data = json.loads(raw)
                refined = data.get('refined', first_translation)
            except:
                match = re.search(r'"refined"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                refined = match.group(1) if match else first_translation
            
            return refined.replace('\\"', '"').replace('\\', '')
            
        except Exception as e:
            logger.warning(f"⚠️ 翻译优化失败: {e}")
            return first_translation
    
    def _parse_json(self, raw: str) -> str:
        """解析JSON响应"""
        try:
            data = json.loads(raw)
            return str(data.get('translation', '')).strip()
        except:
            pass
        
        # 正则提取
        match = re.search(r'"translation"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if match:
            return match.group(1).replace('\\"', '"').replace('\\', '')
        
        return ""
    
    def translate_with_quality_check(self,
                                    text: str,
                                    context_prev: List[str],
                                    context_next: List[str],
                                    terms: Dict[str, str],
                                    target_duration: float,
                                    target_language: str = '简体中文') -> Tuple[str, bool]:
        """
        完整翻译流程（两遍 + 校对）
        
        Returns:
            (translation, is_high_quality)
        """
        # 第一遍：准确翻译
        first_pass = self.translate_first_pass(
            text, context_prev, context_next, terms,
            target_duration, target_language
        )
        
        if not first_pass or first_pass == text:
            return text, False
        
        # 第二遍：优化流畅度
        refined = self.refine_translation(
            text, first_pass, target_duration, target_language
        )
        
        # 应用术语替换（确保一致性）
        final = self.term_manager.apply_terms(refined)
        
        # 质量检查
        is_good = self._quality_check(text, final, target_duration)
        
        return final, is_good
    
    def _quality_check(self, source: str, target: str, target_duration: float) -> bool:
        """简单的质量检查"""
        if not target or len(target) < 3:
            return False
        
        # 检查长度
        max_chars = int(target_duration * 5.0)  # 允许10%超出
        if len(target) > max_chars:
            logger.warning(f"⚠️ 译文过长: {len(target)} > {max_chars}")
            return False
        
        # 检查是否有未翻译的大段英文
        english_chars = sum(1 for c in target if c.isalpha() and ord(c) < 128)
        if english_chars > len(target) * 0.3:
            logger.warning(f"⚠️ 英文字符占比过高: {english_chars}/{len(target)}")
            return False
        
        return True


# ===== 主翻译函数 =====
def translate_advanced(folder: str, target_language: str = '简体中文') -> bool:
    """
    使用高级翻译器处理单个视频
    
    Args:
        folder: 视频文件夹路径
        target_language: 目标语言
    
    Returns:
        是否成功
    """
    translation_path = os.path.join(folder, 'translation.json')
    if os.path.exists(translation_path):
        logger.info(f"✅ 翻译已存在: {folder}")
        return True
    
    transcript_path = os.path.join(folder, 'transcript.json')
    if not os.path.exists(transcript_path):
        logger.error(f"❌ 字幕文件不存在: {transcript_path}")
        return False
    
    # 加载字幕
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    logger.info(f"📄 加载了 {len(transcript)} 条字幕")
    
    # 初始化翻译器
    translator = AdvancedTranslator()
    
    # 提取全局术语
    full_text = ' '.join(line.get('text', '') for line in transcript)
    terms = translator.term_manager.extract_terms(full_text, target_language)
    
    # 保存术语库
    terms_path = os.path.join(folder, 'terminology.json')
    with open(terms_path, 'w', encoding='utf-8') as f:
        json.dump(terms, f, indent=2, ensure_ascii=False)
    
    # 翻译每条字幕
    translations = []
    quality_flags = []
    
    for i, line in enumerate(transcript):
        text = line.get('text', '').strip()
        if not text:
            translations.append("")
            quality_flags.append(False)
            continue
        
        # 获取上下文
        context_prev = [
            transcript[j].get('text', '') 
            for j in range(max(0, i-2), i)
        ]
        context_next = [
            transcript[j].get('text', '') 
            for j in range(i+1, min(len(transcript), i+3))
        ]
        
        # 计算目标时长
        start = float(line.get('start', 0))
        end = float(line.get('end', 0))
        vad_duration = line.get('vad_duration')
        target_duration = min(float(vad_duration), end - start) if vad_duration else (end - start)
        
        # 进度显示
        progress = (i + 1) / len(transcript) * 100
        logger.info(f"📈 [{i+1}/{len(transcript)}] ({progress:.1f}%) - {text[:50]}...")
        
        # 翻译
        translation, is_good = translator.translate_with_quality_check(
            text, context_prev, context_next, terms,
            target_duration, target_language
        )
        
        translations.append(translation)
        quality_flags.append(is_good)
        
        logger.info(f"💬 译文: {translation}")
        logger.info(f"     质量: {'✅' if is_good else '⚠️ '} | 时长: {target_duration:.1f}s")
        logger.info("-" * 60)
        
        time.sleep(0.2)  # 避免API限流
    
    # 构建最终结果
    result = []
    for i, line in enumerate(transcript):
        result.append({
            "start": float(line.get('start', 0)),
            "end": float(line.get('end', 0)),
            "text": line.get('text', ''),
            "speaker": line.get('speaker', ''),
            "translation": translations[i]
        })
    
    # 保存结果
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    # 统计
    total = len(quality_flags)
    success = sum(quality_flags)
    stats = {
        'total': total,
        'success': success,
        'success_rate': round(100 * success / total, 2) if total else 0
    }
    
    stats_path = os.path.join(folder, 'translation_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.success(f"✅ 翻译完成: {translation_path}")
    logger.info(f"📊 成功率: {stats['success_rate']:.1f}% ({success}/{total})")
    
    return True


def translate_all_advanced(root_folder: str, target_language: str = '简体中文') -> int:
    """批量翻译所有视频"""
    folders = [
        root for root, _, files in os.walk(root_folder)
        if 'transcript.json' in files and 'translation.json' not in files
    ]
    
    logger.info(f"🎯 找到 {len(folders)} 个待翻译视频")
    
    success_count = 0
    for i, folder in enumerate(folders, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"🎬 处理 ({i}/{len(folders)}): {folder}")
        
        if translate_advanced(folder, target_language):
            success_count += 1
        
        if i < len(folders):
            time.sleep(2)
    
    logger.success(f"🏁 完成! 成功翻译 {success_count}/{len(folders)} 个视频")
    return success_count


if __name__ == '__main__':
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <6}</level> | <cyan>{message}</cyan>"
    )
    
    translate_all_advanced('videos', '简体中文')