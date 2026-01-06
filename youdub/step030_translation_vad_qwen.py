#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级视频字幕翻译模块（Qwen2.5 + 多轮校对 + 智能术语库 + 自动克隆音频提取）

新增功能：
✅ 自动从 audio_vocals.wav 提取每个说话人 ≤10 秒的干净语音片段
✅ 保存为 SPEAKER/SPEAKER_XX_CLONE.wav 供 VoxCPM 使用

作者: Advanced Translation Team
日期: 2026-01-04
版本: 2.1
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

# 尝试导入音频处理库（用于自动提取克隆音频）
try:
    import librosa
    import numpy as np
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    logger.warning("⚠️ librosa 未安装，将跳过自动克隆音频提取")

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


# ===== 全局分析器（新增）=====
def analyze_transcript(transcript: List[Dict], target_language: str = '简体中文') -> Dict[str, str]:
    full_text = ' '.join(line.get('text', '') for line in transcript[:50])
    input_text = full_text[:3000]

    prompt = f"""你是专业视频内容分析师。请从以下字幕中提取关键术语（英文 → {target_language}）。

要求：
1. 提取专有名词、地名、人名、文化概念、技术术语等
2. 输出标准JSON格式

字幕内容:
{input_text}

输出格式（严格JSON）：
{{
  "Paris": "巴黎",
  "UNESCO": "联合国教科文组织",
  "Northern Lights": "北极光"
}}"""

    try:
        client = get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是专业的视频内容分析师，擅长提取多领域术语。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        
        raw = response.choices[0].message.content.strip()
        logger.debug(f"全局分析术语原始输出: {raw[:200]}...")

        try:
            terms = json.loads(raw)
        except:
            terms = {}
            matches = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', raw)
            for en, zh in matches:
                if en and zh:
                    terms[en] = zh

        logger.info(f"🧠 全局分析提取到 {len(terms)} 个术语")
        return terms

    except Exception as e:
        logger.warning(f"⚠️ 全局分析失败: {e}")
        return {}


# ===== 术语提取与管理（增强版）=====
class TerminologyManager:
    def __init__(self):
        self.terms = {}
        self.domain_keywords = {
            "AI": ["transformer", "attention", "neural network", "GPT", "LLM", 
                   "embedding", "tokenizer", "fine-tuning", "reinforcement learning"],
            "Math": ["derivative", "integral", "matrix", "vector", "function",
                    "equation", "theorem", "proof", "optimization"],
            "Programming": ["API", "function", "variable", "algorithm", "database",
                          "framework", "compiler", "debugger", "deployment"],
            "Travel": [
                "destination", "travel", "visit", "tour", "trip", "journey", "vacation",
                "backpack", "explore", "adventure", "culture", "heritage", "UNESCO",
                "landmark", "beach", "mountain", "resort", "itinerary", "passport"
            ]
        }
    
    def detect_domains(self, text: str) -> List[str]:
        detected = []
        text_lower = text.lower()
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score >= 2:
                detected.append(domain)
        return detected or ["通用"]
    
    def extract_terms(self, text: str, target_language: str = '简体中文') -> Dict[str, str]:
        domains = self.detect_domains(text)
        logger.info(f"🔍 检测到领域: {domains}")
        
        domain_desc = "、".join(domains)
        prompt = f"""你是专业术语提取专家。请从以下{domain_desc}领域的文本中提取关键术语。

要求：
1. 提取技术术语、专有名词、关键概念
2. 每个术语提供准确的{target_language}翻译
3. 输出标准JSON格式

文本（节选）:
{text[:1500]}

输出格式（严格JSON）：
{{
  "transformer": "Transformer模型",
  "attention mechanism": "注意力机制"
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

            try:
                terms = json.loads(raw)
            except:
                terms = {}
                matches = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', raw)
                for en, zh in matches:
                    if en and zh:
                        terms[en] = zh

            self.terms.update(terms)
            logger.info(f"✅ 提取到 {len(terms)} 个关键术语")
            return terms
            
        except Exception as e:
            logger.warning(f"⚠️ 术语提取失败: {e}")
            return {}
    
    def apply_terms(self, text: str) -> str:
        if not self.terms:
            return text
        sorted_terms = sorted(self.terms.items(), key=lambda x: len(x[0]), reverse=True)
        for en, zh in sorted_terms:
            if en and zh:
                pattern = r'\b' + re.escape(en) + r'\b'
                text = re.sub(pattern, zh, text, flags=re.IGNORECASE)
        return text


# ===== 高级翻译器 =====
class AdvancedTranslator:
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.client = get_client()
        self.term_manager = TerminologyManager()
        
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
    
    def build_translation_prompt(self, text, context_prev, context_next, terms, target_duration, target_language='简体中文'):
        max_chars = int(target_duration * 4.5)
        term_list = "\n".join([f"- {en} → {zh}" for en, zh in list(terms.items())[:15]])
        examples = "\n\n".join([
            f"原文: {ex['source']}\n译文: {ex['target']}\n注意: {ex['note']}"
            for ex in self.few_shot_examples[:2]
        ])
        
        context = []
        for i, t in enumerate(context_prev, 1):
            if t: context.append(f"前{i}句: {t}")
        context.append(f"【当前句】: {text}")
        for i, t in enumerate(context_next, 1):
            if t: context.append(f"后{i}句: {t}")
        context_str = "\n".join(context)
        
        term_display = term_list if term_list else "无特定术语"
        return f"""你是专业视频字幕翻译专家，擅长将英文视频翻译成地道、口语化的{target_language}。

# 翻译原则
1. **自然流畅**: 符合{target_language}表达习惯，不要逐字直译
2. **口语化**: 保留"所以"、"其实"、"那么"等语气词
3. **准确性**: 严格使用术语表，保持全文一致
4. **时长匹配**: 译文约{target_duration:.1f}秒，最多{max_chars}个汉字

# 术语表（必须严格遵守）
{term_display}

# 翻译示例（学习风格）
{examples}

# 待翻译内容（含上下文）
{context_str}

# 输出要求
严格输出JSON格式: {{"translation": "译文"}}
只翻译【当前句】，不要翻译上下文！"""
    
    def translate_first_pass(self, text, context_prev, context_next, terms, target_duration, target_language='简体中文'):
        prompt = self.build_translation_prompt(text, context_prev, context_next, terms, target_duration, target_language)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是专业视频翻译专家。严格按照JSON格式输出，使用口语化表达，保持术语一致性。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                top_p=0.9,
                max_tokens=250
            )
            raw = response.choices[0].message.content.strip()
            try:
                data = json.loads(raw)
                return str(data.get('translation', '')).strip()
            except:
                match = re.search(r'"translation"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                return match.group(1).replace('\\"', '"').replace('\\', '') if match else ""
        except Exception as e:
            logger.warning(f"⚠️ 第一遍翻译失败: {e}")
            return ""
    
    def refine_translation(self, original, first_translation, target_duration, target_language='简体中文'):
        if not first_translation:
            return original
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

输出JSON: {{"refined": "优化后的译文"}}"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是专业翻译审校专家。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )
            raw = response.choices[0].message.content.strip()
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
    
    def translate_with_quality_check(self, text, context_prev, context_next, terms, target_duration, target_language='简体中文'):
        first_pass = self.translate_first_pass(text, context_prev, context_next, terms, target_duration, target_language)
        if not first_pass:
            return text, False
        refined = self.refine_translation(text, first_pass, target_duration, target_language)
        final = self.term_manager.apply_terms(refined)
        is_good = self._quality_check(text, final, target_duration)
        return final, is_good
    
    def _quality_check(self, source, target, target_duration):
        if not target or len(target) < 3:
            return False
        max_chars = int(target_duration * 5.0)
        if len(target) > max_chars:
            return False
        english_chars = sum(1 for c in target if c.isalpha() and ord(c) < 128)
        if english_chars > len(target) * 0.3:
            return False
        return True


# ===== 新增：自动提取克隆参考音频 =====
def extract_speaker_clips(folder: str, max_duration: float = 30.0):
    """
    为每个说话人提取一段 <= max_duration 秒的干净语音，用于 TTS 克隆
    保存为 SPEAKER/SPEAKER_XX_CLONE.wav
    """
    if not HAS_LIBROSA:
        logger.warning("⚠️ 未安装 librosa，跳过克隆音频提取")
        return

    transcript_path = os.path.join(folder, 'translation.json')
    vocals_path = os.path.join(folder, 'audio_vocals.wav')
    speaker_dir = os.path.join(folder, 'SPEAKER')
    
    if not os.path.exists(transcript_path) or not os.path.exists(vocals_path):
        logger.warning("⚠️ 缺少 translation.json 或 audio_vocals.wav，跳过克隆音频提取")
        return
    
    if not os.path.exists(speaker_dir):
        os.makedirs(speaker_dir)

    # 加载人声音频
    try:
        vocals, sr = librosa.load(vocals_path, sr=16000)
    except Exception as e:
        logger.error(f"❌ 无法加载人声音频: {e}")
        return

    # 按说话人分组片段（选择最长且 <= max_duration 的）
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    
    speaker_segments = {}
    for line in transcript:
        speaker = line.get('speaker', 'SPEAKER_00')
        start = float(line.get('start', 0))
        end = float(line.get('end', 0))
        text = line.get('text', '').strip()
        
        if not text or end - start <= 0.5:  # 忽略太短或空文本
            continue
        
        duration = end - start
        if duration > max_duration:  # 超长则跳过（或可裁剪，但简单起见跳过）
            continue
        
        if speaker not in speaker_segments or duration > speaker_segments[speaker]['duration']:
            speaker_segments[speaker] = {
                'start': start,
                'end': end,
                'text': text,
                'duration': duration
            }
    
    # 保存每个说话人的最佳片段
    for speaker, seg in speaker_segments.items():
        start_sample = int(seg['start'] * sr)
        end_sample = int(seg['end'] * sr)
        clip = vocals[start_sample:end_sample]
        
        if len(clip) == 0:
            continue
        
        output_path = os.path.join(speaker_dir, f"{speaker}_CLONE.wav")
        try:
            import soundfile as sf
            sf.write(output_path, clip, sr)
            logger.info(f"🔊 保存克隆音频: {output_path} ({seg['duration']:.1f}s)")
            
            # 同时保存文本（用于 VoxCPM 的 prompt_text）
            txt_path = output_path.replace('.wav', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(seg['text'])
            logger.info(f"📄 保存克隆文本: {txt_path}")
        except Exception as e:
            logger.error(f"❌ 保存失败: {e}")


# ===== 主翻译函数 =====
def translate_advanced(folder: str, target_language: str = '简体中文') -> bool:
    translation_path = os.path.join(folder, 'translation.json')
    if os.path.exists(translation_path):
        logger.info(f"✅ 翻译已存在: {folder}")
        return True
    
    transcript_path = os.path.join(folder, 'transcript.json')
    if not os.path.exists(transcript_path):
        logger.error(f"❌ 字幕文件不存在: {transcript_path}")
        return False
    
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript = json.load(f)
    logger.info(f"📄 加载了 {len(transcript)} 条字幕")
    
    global_terms = analyze_transcript(transcript, target_language)
    translator = AdvancedTranslator()
    full_text = ' '.join(line.get('text', '') for line in transcript)
    domain_terms = translator.term_manager.extract_terms(full_text, target_language)
    all_terms = {**domain_terms, **global_terms}
    translator.term_manager.terms = all_terms
    
    terms_path = os.path.join(folder, 'terminology.json')
    with open(terms_path, 'w', encoding='utf-8') as f:
        json.dump(all_terms, f, indent=2, ensure_ascii=False)
    
    translations = []
    quality_flags = []
    
    for i, line in enumerate(transcript):
        text = line.get('text', '').strip()
        if not text:
            translations.append("")
            quality_flags.append(False)
            continue
        
        context_prev = [transcript[j].get('text', '') for j in range(max(0, i-2), i)]
        context_next = [transcript[j].get('text', '') for j in range(i+1, min(len(transcript), i+3))]
        
        start = float(line.get('start', 0))
        end = float(line.get('end', 0))
        vad_duration = line.get('vad_duration')
        target_duration = min(float(vad_duration), end - start) if vad_duration else (end - start)
        
        progress = (i + 1) / len(transcript) * 100
        logger.info(f"📈 [{i+1}/{len(transcript)}] ({progress:.1f}%) - {text[:50]}...")
        
        translation, is_good = translator.translate_with_quality_check(
            text, context_prev, context_next, all_terms,
            target_duration, target_language
        )
        
        translations.append(translation)
        quality_flags.append(is_good)
        
        logger.info(f"💬 译文: {translation}")
        logger.info(f"     质量: {'✅' if is_good else '⚠️ '} | 时长: {target_duration:.1f}s")
        logger.info("-" * 60)
        time.sleep(0.2)
    
    result = []
    for i, line in enumerate(transcript):
        result.append({
            "start": float(line.get('start', 0)),
            "end": float(line.get('end', 0)),
            "text": line.get('text', ''),
            "speaker": line.get('speaker', ''),
            "translation": translations[i]
        })
    
    with open(translation_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    total = len(quality_flags)
    success = sum(quality_flags)
    stats = {'total': total, 'success': success, 'success_rate': round(100 * success / total, 2) if total else 0}
    stats_path = os.path.join(folder, 'translation_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.success(f"✅ 翻译完成: {translation_path}")
    logger.info(f"📊 成功率: {stats['success_rate']:.1f}% ({success}/{total})")
    
    # === 新增：自动提取克隆音频 ===
    logger.info("✂️ 正在提取说话人克隆音频...")
    extract_speaker_clips(folder, max_duration=10.0)
    
    return True


def translate_all_advanced(root_folder: str, target_language: str = '简体中文') -> int:
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