# -*- coding: utf-8 -*-
"""
WhisperX 批量字幕生成工具 - Silero VAD 校准版（ONNX 修复 + 单位修正）
功能：
  - 高精度语音识别 + 时间对齐
  - 说话人分离（Diarization）
  - 智能句子合并
  - Silero VAD 校准真实语音时长（ONNX 模式，单位已修正）
  - 中间文件存入 temp/ 目录
  
版本: 1.0
"""

import json
import time
import librosa
import numpy as np
import whisperx
import os
from loguru import logger
import torch
from dotenv import load_dotenv
import glob
import sys
import gc
import soundfile as sf
from whisperx.diarize import DiarizationPipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# 全局模型缓存
whisper_model = None
diarize_model = None
align_model = None
language_code = None
align_metadata = None

# 模型缓存目录
EXISTING_MODEL_DIR = os.getenv("HF_HUB_CACHE", r"C:\model\huggingface\hub")
DEFAULT_DOWNLOAD_ROOT = os.getenv("WHISPER_DOWNLOAD_ROOT", EXISTING_MODEL_DIR)

# 环境变量设置
os.environ["HF_HUB_CACHE"] = EXISTING_MODEL_DIR
os.environ["TORCH_HOME"] = EXISTING_MODEL_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = EXISTING_MODEL_DIR

hf_endpoint = os.getenv("HF_ENDPOINT")
if hf_endpoint:
    os.environ["HF_ENDPOINT"] = hf_endpoint

HF_TOKEN = os.getenv("HF_TOKEN")

def init_whisperx():
    logger.info("=== WhisperX 初始化配置 ===")
    logger.info(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
    logger.info(f"TORCH_HOME: {os.environ.get('TORCH_HOME')}")
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✅ 启用TF32加速")

# ===== 修复：正确处理 Silero VAD 的毫秒单位 =====
def calculate_vad_duration(vad_timestamps, segment_start, segment_end):
    """计算 VAD 语音段在 [segment_start, segment_end] 内的总时长（秒）"""
    total = 0.0
    for ts in vad_timestamps:
        # Silero VAD 的 start/end 单位是毫秒（ms）
        vad_start = ts['start'] / 1000.0  # 转为秒
        vad_end = ts['end'] / 1000.0      # 转为秒
        
        overlap_start = max(vad_start, segment_start)
        overlap_end = min(vad_end, segment_end)
        
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    
    return round(total, 3)
# ===============================================

def load_whisper_model(model_name='large-v3', download_root=None, device='auto'):
    global whisper_model
    if whisper_model is not None:
        return
    
    if download_root is None:
        download_root = DEFAULT_DOWNLOAD_ROOT
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f'🚀 加载 WhisperX 模型: {model_name}')
    logger.info(f'🖥️ 设备: {device}')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    t_start = time.time()
    
    compute_type = "float16"
    if device == 'cpu':
        compute_type = "float32"
    elif torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 8 * 1024**3:
        logger.warning("⚠️ GPU内存<8GB，建议使用medium模型")
    
    whisper_model = whisperx.load_model(
        model_name, 
        download_root=download_root, 
        device=device,
        compute_type=compute_type
    )
    t_end = time.time()
    logger.info(f'✅ WhisperX 模型加载完成: {t_end - t_start:.2f}s')
    
    check_model_cache(download_root)

def unload_whisper_model():
    """
    卸载所有 WhisperX 相关模型，包括：
    - Whisper 主模型
    - 对齐模型
    - 说话人分离模型
    """
    global whisper_model, align_model, diarize_model, language_code, align_metadata
    
    # 卸载 Whisper 主模型
    if whisper_model is not None:
        logger.info("✅ 正在卸载 Whisper 主模型...")
        del whisper_model
        whisper_model = None
    
    # 卸载对齐模型
    if align_model is not None:
        logger.info("✅ 正在卸载对齐模型...")
        del align_model
        del align_metadata
        align_model = None
        align_metadata = None
        language_code = None
    
    # 卸载说话人分离模型
    if diarize_model is not None:
        logger.info("✅ 正在卸载说话人分离模型...")
        del diarize_model
        diarize_model = None
    
    # 清理资源
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 强制垃圾回收
    gc.collect()
    
    logger.info("✅ WhisperX 相关模型已全部卸载")

def load_align_model(language='en', device='auto'):
    global align_model, language_code, align_metadata
    if align_model is not None and language_code == language:
        return
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f'⏱️ 加载对齐模型: {language}')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    t_start = time.time()
    align_model, align_metadata = whisperx.load_align_model(
        language_code=language, 
        device=device,
        model_dir=EXISTING_MODEL_DIR
    )
    t_end = time.time()
    logger.info(f'✅ 对齐模型加载完成: {t_end - t_start:.2f}s')

def load_diarize_model(device='auto'):
    global diarize_model
    if diarize_model is not None:
        return
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info('👥 加载说话人分离模型...')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    t_start = time.time()
    diarize_model = DiarizationPipeline(
        use_auth_token=HF_TOKEN, 
        device=device
    )
    t_end = time.time()
    logger.info(f'✅ 说话人分离模型加载完成: {t_end - t_start:.2f}s')

def check_model_cache(cache_dir):
    try:
        models_found = []
        pattern = os.path.join(cache_dir, "models--*")
        for folder in glob.glob(pattern):
            models_found.append(os.path.basename(folder))
        
        if models_found:
            logger.info(f"📁 缓存中找到 {len(models_found)} 个模型")
            for model in models_found[:5]:
                logger.info(f"  - {model}")
    except Exception as e:
        logger.warning(f"⚠️ 检查模型缓存失败: {str(e)}")

def convert_diarization_result(diarize_segments):
    result = []
    try:
        if hasattr(diarize_segments, 'itertracks'):
            for segment, track, label in diarize_segments.itertracks(yield_label=True):
                result.append({'segment_start': segment.start, 'segment_end': segment.end, 'speaker': label})
        elif hasattr(diarize_segments, 'to_dict'):
            import pandas as pd
            df = diarize_segments
            for _, row in df.iterrows():
                result.append({'segment_start': row.get('start', 0), 'segment_end': row.get('end', 0), 'speaker': row.get('speaker', 'UNKNOWN')})
        elif isinstance(diarize_segments, (list, tuple)):
            for i, item in enumerate(diarize_segments):
                if hasattr(item, 'start') and hasattr(item, 'end'):
                    result.append({'segment_start': item.start, 'segment_end': item.end, 'speaker': getattr(item, 'speaker', f'SPEAKER_{i:02d}')})
                elif isinstance(item, dict):
                    result.append(item)
        else:
            result = {'raw_type': str(type(diarize_segments))}
    except Exception as e:
        logger.error(f"❌ 转换失败: {e}")
        result = {'error': str(e)}
    return result

def merge_segments(transcript, ending='!"\').:;?]}~', max_gap=1.0):
    if not transcript:
        return []
    merged = []
    buffer = transcript[0].copy()
    for i in range(1, len(transcript)):
        current = transcript[i].copy()
        gap = current['start'] - buffer['end']
        should_merge = ((not buffer['text'] or buffer['text'][-1] not in ending) and gap < max_gap and current['text'].strip())
        if should_merge:
            buffer['text'] += ' ' + current['text']
            buffer['end'] = current['end']
            buffer['duration'] = round(buffer['end'] - buffer['start'], 3)
            buffer['vad_duration'] = round(buffer.get('vad_duration', 0) + current.get('vad_duration', 0), 3)
        else:
            merged.append(buffer)
            buffer = current
    if buffer:
        if 'duration' not in buffer:
            buffer['duration'] = round(buffer['end'] - buffer['start'], 3)
        if 'vad_duration' not in buffer:
            buffer['vad_duration'] = buffer['duration']
        merged.append(buffer)
    return merged

def sanitize_transcript(transcript, audio_duration):
    if not transcript:
        return []
    sanitized = []
    prev_end = 0.0
    for seg in transcript:
        try:
            start = float(seg.get('start', 0))
            end = float(seg.get('end', 0))
            text = str(seg.get('text', '')).strip()
            speaker = seg.get('speaker', 'SPEAKER_00')
            if not text or end <= start:
                continue
            start = max(0.0, start)
            end = min(audio_duration, end)
            if end <= start:
                end = start + 0.01
            if start < prev_end:
                start = prev_end
                if end <= start:
                    end = start + 0.01
            start = round(start, 3)
            end = round(end, 3)
            duration = round(end - start, 3)
            vad_duration = min(duration, seg.get('vad_duration', duration))
            sanitized.append({
                'start': start,
                'end': end,
                'duration': duration,
                'vad_duration': vad_duration,
                'text': text,
                'speaker': speaker
            })
            prev_end = end
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"⚠️ 跳过无效段: {e}")
            continue
    return sanitized

def validate_transcript_coverage(transcript, audio_duration, wav_path, folder):
    total_duration = sum(seg['end'] - seg['start'] for seg in transcript)
    coverage_rate = total_duration / audio_duration if audio_duration > 0 else 0
    logger.info(f"📊 时间覆盖分析:")
    logger.info(f"  音频总时长: {audio_duration:.2f}秒")
    logger.info(f"  转录总时长: {total_duration:.2f}秒")
    logger.info(f"  覆盖比例: {coverage_rate:.1%}")
    gaps = []
    last_end = 0
    for i, seg in enumerate(transcript):
        if seg['start'] > last_end:
            gap_duration = seg['start'] - last_end
            gaps.append({'gap_index': len(gaps), 'start': last_end, 'end': seg['start'], 'duration': gap_duration})
        last_end = seg['end']
    if gaps:
        logger.warning(f"⚠️ 发现 {len(gaps)} 个时间空白")
        for gap in gaps[:3]:
            logger.warning(f"  空白{gap['gap_index']}: {gap['start']:.2f}-{gap['end']:.2f} ({gap['duration']:.2f}秒)")
    validation_report = {
        'audio_duration': audio_duration,
        'transcript_duration': total_duration,
        'coverage_rate': coverage_rate,
        'gap_count': len(gaps),
        'gaps': gaps,
        'segment_count': len(transcript)
    }
    report_path = os.path.join(folder, 'validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(validation_report, f, indent=2, ensure_ascii=False)
    logger.info(f"📋 验证报告已保存: {report_path}")
    return validation_report
def transcribe_audio(folder, model_name='large-v3', download_root=None, device='auto', 
                     batch_size=8, diarization=True, min_speakers=None, max_speakers=None):
    transcript_path = os.path.join(folder, 'transcript.json')
    if os.path.exists(transcript_path):
        logger.info(f'✅ 转录已存在: {transcript_path}')
        return True
    
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    if not os.path.exists(wav_path):
        logger.error(f'❌ 音频文件未找到: {wav_path}')
        return False
    
    # 创建临时目录
    temp_dir = os.path.join(folder, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    logger.info(f'📁 临时目录: {temp_dir}')
    
    logger.info(f'🎙️ 开始转录: {wav_path}')
    
    if download_root is None:
        download_root = DEFAULT_DOWNLOAD_ROOT
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 支持对齐的语言列表（WhisperX 官方支持）
    SUPPORTED_ALIGN_LANGUAGES = {
        'en', 'fr', 'de', 'es', 'it', 'pt', 'nl', 'uk', 'ja', 'zh', 'ru',
        'ar', 'cs', 'tr', 'pl', 'ca', 'hu', 'ko', 'vi', 'sw', 'sl', 'lv',
        'fi', 'ro', 'da', 'he', 'el', 'gl', 'eu', 'af', 'lt', 'pa', 'is',
        'ml', 'ms', 'mr', 'ta', 'te', 'ur', 'hi', 'bn', 'gu', 'kn', 'or'
    }

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        load_whisper_model(model_name, download_root, device)
        audio_duration = librosa.get_duration(path=wav_path)
        logger.info(f'⏱️ 音频时长: {audio_duration:.2f}秒')
        
        if audio_duration > 600:
            batch_size = max(4, batch_size // 2)
            logger.info(f'🎬 长音频，降低 batch_size 到 {batch_size}')
                
        # ===== Silero VAD 分析（强制 ONNX 模式）=====
        logger.info('🔊 执行 Silero VAD 分析 (ONNX 模式)...')
        vad_timestamps = None
        try:
            from silero_vad import get_speech_timestamps, load_silero_vad
            model = load_silero_vad()
            audio_vocals, sr = librosa.load(wav_path, sr=16000)
            vad_timestamps = get_speech_timestamps(
                audio_vocals,
                model=model,
                sampling_rate=16000,
                threshold=0.5,
                min_speech_duration_ms=200,
                max_speech_duration_s=15.0,
                min_silence_duration_ms=1000,
                speech_pad_ms=200
            )
            logger.info(f'✅ VAD 检测到 {len(vad_timestamps)} 个语音段')
        except Exception as e:
            logger.error(f"❌ Silero VAD (ONNX) 失败: {e}")
            logger.info("⚠️ 回退到原始 duration")
        # ===================================
        
        logger.info('📝 语音识别...')
        rec_result = whisper_model.transcribe(wav_path, batch_size=batch_size)
        
        if rec_result['language'] == 'nn':
            logger.warning('❓ 未检测到有效语言')
            return False
        
        detected_lang = rec_result['language']
        logger.info(f'🌍 检测到语言: {detected_lang} (置信度可能较低)')
        
        # 保存初始转录
        initial_path = os.path.join(temp_dir, '0_initial_transcription.json')
        with open(initial_path, 'w', encoding='utf-8') as f:
            json.dump(rec_result, f, indent=2, ensure_ascii=False)
        logger.info(f'💾 保存初始转录: {initial_path}')
        
        # ===== 决定是否执行时间对齐 =====
        if detected_lang in SUPPORTED_ALIGN_LANGUAGES:
            logger.info('⏳ 时间对齐...')
            load_align_model(detected_lang, device)
            aligned_result = whisperx.align(
                rec_result['segments'], 
                align_model, 
                align_metadata,
                wav_path, 
                device, 
                return_char_alignments=False
            )
            aligned_path = os.path.join(temp_dir, '1_aligned_transcription.json')
            with open(aligned_path, 'w', encoding='utf-8') as f:
                json.dump(aligned_result, f, indent=2, ensure_ascii=False)
            logger.info(f'💾 保存对齐结果: {aligned_path}')
        else:
            logger.warning(f"⚠️ 语言 '{detected_lang}' 不在支持对齐的语言列表中，跳过对齐步骤")
            aligned_result = rec_result  # 直接使用原始结果
        
        # ===== 说话人分离 =====
        if diarization:
            logger.info('👥 说话人分离...')
            load_diarize_model(device)
            diarize_segments = diarize_model(
                wav_path,
                min_speakers=min_speakers, 
                max_speakers=max_speakers
            )
            
            diarize_path = os.path.join(temp_dir, '2_diarization_raw.json')
            diarize_converted = convert_diarization_result(diarize_segments)
            with open(diarize_path, 'w', encoding='utf-8') as f:
                json.dump(diarize_converted, f, indent=2, ensure_ascii=False)
            logger.info(f'💾 保存说话人分离结果: {diarize_path}')
            
            assigned_result = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            assigned_path = os.path.join(temp_dir, '3_assigned_speakers.json')
            with open(assigned_path, 'w', encoding='utf-8') as f:
                json.dump(assigned_result, f, indent=2, ensure_ascii=False)
            logger.info(f'💾 保存说话人分配结果: {assigned_path}')
        else:
            assigned_result = aligned_result
        
        # ===== 构建最终结果（含 VAD 时长）=====
        logger.info('🔧 构建最终结果（含 VAD 时长）...')
        transcript = []
        for segment in assigned_result['segments']:
            start = float(segment.get('start', 0))
            end = float(segment.get('end', 0.01))
            if end <= start:
                end = start + 0.01
            duration = round(end - start, 3)
            
            # 计算 VAD 时长（使用修复后的单位转换）
            if vad_timestamps is not None:
                vad_duration = calculate_vad_duration(vad_timestamps, start, end)
            else:
                vad_duration = duration
            
            transcript.append({
                'start': start,
                'end': end,
                'duration': duration,
                'vad_duration': vad_duration,
                'text': segment['text'].strip(),
                'speaker': segment.get('speaker', 'SPEAKER_00')
            })
        
        raw_transcript_path = os.path.join(temp_dir, '4_raw_transcript.json')
        with open(raw_transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        logger.info(f'💾 保存未合并结果: {raw_transcript_path}')
        
        # ===== 合并 + 安全化 =====
        logger.info('🔗 合并片段...')
        original_count = len(transcript)
        transcript = merge_segments(transcript)
        merged_count = len(transcript)
        logger.info(f'📊 合并: {original_count} → {merged_count} 个片段')
        
        logger.info('🛡️ 时间戳安全化处理...')
        transcript = sanitize_transcript(transcript, audio_duration)
        if not transcript:
            logger.error('❌ 安全化后无有效字幕')
            return False
        
        # ===== 验证 + 保存 =====
        logger.info('✅ 验证结果...')
        validate_transcript_coverage(transcript, audio_duration, wav_path, folder)
        
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=4, ensure_ascii=False)
        logger.info(f'✅ 转录 {len(transcript)} 个片段 → {transcript_path}')
        
        # ===== 生成说话人音频 =====
        generate_speaker_audio(folder, transcript)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return True
        
    except torch.cuda.OutOfMemoryError:
        logger.error('💥 GPU内存不足!')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return False
        
    except Exception as e:
        logger.exception(f'🔥 转录错误: {e}')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return False
def generate_speaker_audio(folder, transcript):
    """生成说话人音频和对应文本（用于 VoxCPM 语音克隆）"""
    wav_path = os.path.join(folder, 'audio_vocals.wav')
    try:
        audio_data, samplerate = librosa.load(wav_path, sr=24000)
    except Exception as e:
        logger.error(f'❌ 加载音频失败: {e}')
        return

    speaker_audio = {}
    speaker_texts = {}

    delay = 0.05  # 前后扩展 50ms

    for segment in transcript:
        start = max(0, int((segment['start'] - delay) * samplerate))
        end = min(int((segment['end'] + delay) * samplerate), len(audio_data))
        audio_chunk = audio_data[start:end]

        speaker = segment['speaker']
        text = segment['text'].strip()

        if speaker in speaker_audio:
            speaker_audio[speaker] = np.concatenate((speaker_audio[speaker], audio_chunk))
        else:
            speaker_audio[speaker] = audio_chunk

        if speaker in speaker_texts:
            speaker_texts[speaker].append(text)
        else:
            speaker_texts[speaker] = [text]

    speaker_folder = os.path.join(folder, 'SPEAKER')
    os.makedirs(speaker_folder, exist_ok=True)

    for speaker in speaker_audio:
        # === 保存音频：直接使用 soundfile.write ===
        wav_file = os.path.join(speaker_folder, f"{speaker}.wav")
        try:
            sf.write(wav_file, speaker_audio[speaker], 24000)
            logger.info(f'🔊 保存说话人音频: {wav_file}')
        except Exception as e:
            logger.error(f'❌ 保存 {speaker} 音频失败: {e}')

        # === 保存文本 ===
        txt_file = os.path.join(speaker_folder, f"{speaker}.txt")
        full_text = ' '.join(speaker_texts[speaker]).strip()
        try:
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(full_text)
            logger.info(f'📝 保存说话人文本: {txt_file} ({full_text[:50]}...)')
        except Exception as e:
            logger.error(f'❌ 保存 {speaker} 文本失败: {e}')



def transcribe_all_audio_under_folder(folder, model_name='large-v3', download_root=None, 
                                      device='auto', batch_size=8, diarization=True, 
                                      min_speakers=None, max_speakers=None):
    logger.info(f'📁 开始批量转录: {folder}')
    logger.info(f'🤖 模型: {model_name} | 设备: {device}')
    
    if download_root is None:
        download_root = DEFAULT_DOWNLOAD_ROOT
    
    folders_to_process = []
    for root, _, files in os.walk(folder):
        if 'audio_vocals.wav' in files and 'transcript.json' not in files:
            folders_to_process.append(root)
    
    logger.info(f'🎯 找到 {len(folders_to_process)} 个待处理文件夹')
    
    processed, failed = 0, 0
    for i, root in enumerate(folders_to_process, 1):
        logger.info(f'\n{"─" * 50}')
        logger.info(f'🎬 处理 ({i}/{len(folders_to_process)}): {os.path.basename(root)}')
        try:
            if transcribe_audio(root, model_name, download_root, device, batch_size, diarization, min_speakers, max_speakers):
                processed += 1
            else:
                failed += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            logger.exception(f'💥 严重错误 {root}: {e}')
            failed += 1
    
    logger.info(f'\n{"═" * 50}')
    logger.info(f'✅ 完成! 成功: {processed}, 失败: {failed}')
    return f'转录 {processed} 个音频文件 (失败: {failed})'

def regression_test_existing_transcripts(folder):
    logger.info(f"🔍 开始回归测试: {folder}")
    reports = []
    for root, dirs, files in os.walk(folder):
        transcript_path = os.path.join(root, 'transcript.json')
        wav_path = os.path.join(root, 'audio_vocals.wav')
        if os.path.exists(transcript_path) and os.path.exists(wav_path):
            logger.info(f"\n📁 分析: {os.path.basename(root)}")
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    transcript = json.load(f)
                audio_duration = librosa.get_duration(path=wav_path)
                report = validate_transcript_coverage(transcript, audio_duration, wav_path, root)
                report['folder'] = os.path.basename(root)
                reports.append(report)
            except Exception as e:
                logger.error(f"❌ 分析失败 {root}: {e}")
    if reports:
        total_folders = len(reports)
        avg_coverage = sum(r['coverage_rate'] for r in reports) / total_folders
        logger.info(f"\n{'='*60}")
        logger.info(f"📈 回归测试汇总:")
        logger.info(f"  总文件夹数: {total_folders}")
        logger.info(f"  平均覆盖比例: {avg_coverage:.1%}")
    return reports

def main():
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        colorize=True
    )
    logger.add(
        "whisperx_transcribe.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    
    logger.info("🔊 WhisperX 批量转录工具 - Silero VAD 版启动")
    init_whisperx()
    
    target_folder = 'videos'
    if not os.path.exists(target_folder):
        logger.error(f"❌ 目标文件夹未找到: {target_folder}")
        return
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory < 8:
            model_name, batch_size = 'medium', 4
            logger.warning("⚠️ 内存不足: 使用 'medium' 模型")
        else:
            model_name, batch_size = 'large-v3', 8
    else:
        model_name, batch_size = 'large-v3', 8
        logger.info("💻 使用 CPU")
    
    result = transcribe_all_audio_under_folder(
        target_folder, 
        model_name=model_name,
        batch_size=batch_size,
        diarization=True
    )
    
    logger.info("\n" + "="*60)
    logger.info("🔍 对已有转录结果进行验证...")
    regression_test_existing_transcripts(target_folder)
    
    logger.info("\n🎉 所有处理完成！")

if __name__ == '__main__':
    main()