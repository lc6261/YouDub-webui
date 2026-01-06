"""
whisper_asr_reliable_local.py - Local temporary file version for English videos
"""

import os
import subprocess
import json
import tempfile
import time
import sys
import threading
from typing import List, Dict, Optional
import queue
import shutil
import re

# Configuration
WHISPER_DIR = r"C:\whisper-cublas-12.4.0-bin-x64\Release"
WHISPER_EXE = os.path.join(WHISPER_DIR, "whisper-cli.exe")
MODEL = os.path.join(WHISPER_DIR, "ggml-large-v3-q5_0.bin")

# Test file
test_file = "audio_vocals.wav"

# Create local temp directory
LOCAL_TEMP_DIR = os.path.join(os.path.dirname(__file__), "tmp")
os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)
print(f"Local temp directory: {LOCAL_TEMP_DIR}")

class ProcessTimeout(Exception):
    """Process timeout exception"""
    pass

def run_command_with_timeout(cmd, timeout=3600):
    """Run command with timeout and real-time output"""
    def target(queue, cmd, cwd):
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                cwd=cwd,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            
            stdout, stderr = process.communicate()
            
            queue.put({
                'returncode': process.returncode,
                'stdout': stdout,
                'stderr': stderr,
                'process': process
            })
        except Exception as e:
            queue.put({'error': str(e)})
    
    q = queue.Queue()
    thread = threading.Thread(target=target, args=(q, cmd, WHISPER_DIR))
    thread.daemon = True
    thread.start()
    
    try:
        result = q.get(timeout=timeout)
        return result
    except queue.Empty:
        raise ProcessTimeout(f"Command execution timeout ({timeout} seconds)")
    except Exception as e:
        return {'error': str(e)}

def split_audio_file(input_path, segment_duration=600):
    """Split large audio file into segments"""
    if not os.path.exists(input_path):
        return None, []
    
    # Get audio duration
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        duration = float(result.stdout.strip())
    except:
        print("⚠ Unable to get audio duration, using estimation")
        # Estimate based on file size: assume 16000Hz 16-bit mono
        file_size = os.path.getsize(input_path)
        duration = file_size / (16000 * 2)  # 16-bit = 2 bytes
        
    print(f"Total audio duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    
    if duration <= segment_duration * 2:  # Less than 20 minutes, no split needed
        print("Audio is short, no need to split")
        return input_path, []
    
    # Create local split directory
    split_dir = os.path.join(LOCAL_TEMP_DIR, f"split_{int(time.time())}")
    os.makedirs(split_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Split audio
    print(f"Splitting audio (each segment {segment_duration} seconds)...")
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-f", "segment",
        "-segment_time", str(segment_duration),
        "-c", "copy",
        "-map", "0:a",
        os.path.join(split_dir, f"{base_name}_%03d.wav"),
        "-y",
        "-loglevel", "error"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    if result.returncode != 0:
        print(f"⚠ Split failed: {result.stderr[:200]}")
        return input_path, []
    
    # Get split file list
    segment_files = []
    for f in sorted(os.listdir(split_dir)):
        if f.endswith('.wav'):
            segment_files.append(os.path.join(split_dir, f))
    
    print(f"✅ Split completed: {len(segment_files)} segments")
    print(f"Segments saved in: {split_dir}")
    return split_dir, segment_files

def convert_audio(input_path):
    """Convert audio to 16kHz mono"""
    if not os.path.exists(input_path):
        return input_path
    
    # Check audio format
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        codec, sample_rate, channels = result.stdout.strip().split('\n')[:3]
        sample_rate = int(sample_rate)
        channels = int(channels)
        
        print(f"Audio info: {codec}, {sample_rate}Hz, {channels} channels")
        
        # If already in correct format, no conversion needed
        if sample_rate == 16000 and channels == 1 and codec == 'pcm_s16le':
            print("Audio format is already suitable, no conversion needed")
            return input_path
    except:
        print("⚠ Unable to get audio info, performing conversion")
    
    # Create converted file in local temp directory
    timestamp = int(time.time())
    output_path = os.path.join(LOCAL_TEMP_DIR, f"converted_{timestamp}_{os.path.basename(input_path)}")
    
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        "-acodec", "pcm_s16le",
        output_path,
        "-y",
        "-loglevel", "error"
    ]
    
    print("Converting audio format...")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(output_path):
            elapsed = time.time() - start_time
            print(f"✅ Audio conversion successful ({elapsed:.1f} seconds)")
            print(f"Converted file: {output_path}")
            return output_path
        else:
            print(f"⚠ Audio conversion failed: {result.stderr[:200]}")
            return input_path
    except Exception as e:
        print(f"⚠ Conversion error: {e}")
        return input_path

def debug_json_file(json_file):
    """Debug JSON file content"""
    try:
        print(f"\nDebugging JSON file: {json_file}")
        print(f"File size: {os.path.getsize(json_file)} bytes")
        
        # Check file header
        with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
            first_500 = f.read(500)
            print(f"File header (first 500 chars):\n{first_500}")
        
        # Try to parse
        with open(json_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        data = json.loads(content)
        print(f"JSON structure type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            if 'segments' in data:
                segments = data['segments']
                print(f"segments type: {type(segments)}, length: {len(segments) if hasattr(segments, '__len__') else 'N/A'}")
                
                # Print first segment
                if segments and len(segments) > 0:
                    print(f"First segment: {segments[0]}")
        
        return data
    except Exception as e:
        print(f"Debug error: {e}")
        import traceback
        traceback.print_exc()
        return None

def vtt_time_to_seconds(time_str):
    """Convert VTT time format to seconds"""
    # Format: HH:MM:SS.mmm or MM:SS.mmm
    parts = time_str.split(':')
    
    if len(parts) == 3:
        # HH:MM:SS.mmm
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        # MM:SS.mmm
        minutes = float(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        return 0

def time_str_to_seconds(time_str):
    """Convert time string to seconds"""
    # Format: HH:MM:SS.mmm
    if '.' in time_str:
        time_str = time_str.split('.')[0]  # Remove milliseconds
    
    parts = time_str.split(':')
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    return 0

def parse_vtt_file(vtt_path):
    """Parse VTT file to get timestamps"""
    segments = []
    
    try:
        with open(vtt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip WEBVTT header and other metadata
            if line.startswith('WEBVTT') or not line:
                i += 1
                continue
            
            # Try to parse timestamp line (format: 00:00:00.000 --> 00:00:04.000)
            if '-->' in line:
                try:
                    times = line.split('-->')
                    start_str = times[0].strip()
                    end_str = times[1].strip().split()[0]  # Might have position info after
                    
                    # Convert time format
                    start_seconds = vtt_time_to_seconds(start_str)
                    end_seconds = vtt_time_to_seconds(end_str)
                    
                    # Read text lines
                    text_lines = []
                    i += 1
                    while i < len(lines) and lines[i].strip() and not '-->' in lines[i]:
                        text_lines.append(lines[i].strip())
                        i += 1
                    
                    text = ' '.join(text_lines).strip()
                    if text:
                        segments.append({
                            'start': start_seconds,
                            'end': end_seconds,
                            'text': text
                        })
                    
                    continue  # Continue to next segment
                    
                except Exception as e:
                    print(f"Failed to parse VTT timestamp: {e}")
            
            i += 1
        
        return segments
    
    except Exception as e:
        print(f"Failed to parse VTT file: {e}")
        return []

def parse_srt_file(srt_path):
    """Parse SRT file to get timestamps"""
    segments = []
    
    try:
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by empty lines
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Line 0: Index
                # Line 1: Timestamp (00:00:00,000 --> 00:00:04,000)
                # Line 2+: Text
                timestamp_line = lines[1]
                if '-->' in timestamp_line:
                    times = timestamp_line.split('-->')
                    start_str = times[0].strip().replace(',', '.')
                    end_str = times[1].strip().replace(',', '.')
                    
                    # Convert time format
                    start_seconds = vtt_time_to_seconds(start_str)
                    end_seconds = vtt_time_to_seconds(end_str)
                    
                    # Get text
                    text = ' '.join(lines[2:]).strip()
                    if text:
                        segments.append({
                            'start': start_seconds,
                            'end': end_seconds,
                            'text': text
                        })
        
        return segments
    
    except Exception as e:
        print(f"Failed to parse SRT file: {e}")
        return []

def extract_segments_from_stdout(stdout):
    """Try to extract segment information from stdout"""
    segments = []
    
    if not stdout:
        return segments
    
    lines = stdout.strip().split('\n')
    
    # whisper.cpp output usually contains timestamps and text
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to match timestamp pattern [HH:MM:SS --> HH:MM:SS]
        time_pattern = r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]\s*(.+)'
        match = re.match(time_pattern, line)
        
        if match:
            try:
                start_time = match.group(1)
                end_time = match.group(2)
                text = match.group(3).strip()
                
                # Convert time
                start_seconds = vtt_time_to_seconds(start_time)
                end_seconds = vtt_time_to_seconds(end_time)
                
                segments.append({
                    'start': start_seconds,
                    'end': end_seconds,
                    'text': text
                })
            except:
                continue
    
    return segments

def parse_whisper_json(data):
    """Parse whisper.cpp JSON output"""
    segments = []
    
    if isinstance(data, dict):
        # Try different structures
        if 'segments' in data and isinstance(data['segments'], list):
            # Standard structure
            for seg in data['segments']:
                if isinstance(seg, dict):
                    text = seg.get('text', '').strip()
                    if text:
                        segments.append({
                            'start': float(seg.get('start', 0)),
                            'end': float(seg.get('end', 0)),
                            'text': text
                        })
        
        elif 'result' in data and isinstance(data['result'], dict):
            # Might be another structure
            result = data['result']
            if 'segments' in result and isinstance(result['segments'], list):
                for seg in result['segments']:
                    if isinstance(seg, dict):
                        text = seg.get('text', '').strip()
                        if text:
                            segments.append({
                                'start': float(seg.get('start', 0)),
                                'end': float(seg.get('end', 0)),
                                'text': text
                            })
        
        elif 'transcription' in data:
            # Only full text
            text = data['transcription'].strip()
            if text:
                segments.append({
                    'start': 0,
                    'end': 0,
                    'text': text
                })
    
    elif isinstance(data, list):
        # Direct segment list
        for item in data:
            if isinstance(item, dict):
                text = item.get('text', '').strip()
                if text:
                    segments.append({
                        'start': float(item.get('start', 0)),
                        'end': float(item.get('end', 0)),
                        'text': text
                    })
    
    return segments

def transcribe_segments(segment_files, language="en"):
    """Transcribe multiple audio segments"""
    all_segments = []
    total_files = len(segment_files)
    
    for i, segment_file in enumerate(segment_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing segment {i}/{total_files}: {os.path.basename(segment_file)}")
        print(f"{'='*60}")
        
        file_size = os.path.getsize(segment_file) / 1024 / 1024
        print(f"Segment size: {file_size:.1f} MB")
        print(f"Segment path: {segment_file}")
        
        try:
            segments = transcribe_with_json_single(segment_file, language)
            if segments:
                # Adjust timestamps
                time_offset = (i-1) * 600  # Assuming each segment is 600 seconds
                for seg in segments:
                    seg['start'] += time_offset
                    seg['end'] += time_offset
                all_segments.extend(segments)
                print(f"✅ Segment transcription completed: {len(segments)} segments")
                if segments:
                    preview = segments[0]['text'][:100] + "..." if len(segments[0]['text']) > 100 else segments[0]['text']
                    print(f"Sample segment: {preview}")
            else:
                print(f"⚠ Segment transcription failed or no content")
        except Exception as e:
            print(f"❌ Segment transcription error: {e}")
            import traceback
            traceback.print_exc()
    
    # Return split directory path for cleanup
    split_dir = os.path.dirname(segment_files[0]) if segment_files else None
    return all_segments, split_dir

def transcribe_with_json_single(audio_path, language="en"):
    """Transcribe single audio file"""
    print(f"Starting transcription: {os.path.basename(audio_path)}")
    
    # Convert audio
    audio_to_use = convert_audio(audio_path)
    print(f"Using audio file: {audio_to_use}")
    
    # Create output file in local temp directory
    timestamp = int(time.time())
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_base = os.path.join(LOCAL_TEMP_DIR, f"whisper_out_{timestamp}_{base_name}")
    
    # Build command - use reliable parameters
    cmd = [
        WHISPER_EXE,
        "--model", MODEL,
        "--file", os.path.abspath(audio_to_use),
        "--language", language,
        "--output-file", output_base,
        "--output-txt",  # Text output
        "--output-vtt",  # VTT output with timestamps
        "--output-srt",  # SRT output with timestamps
        "--threads", "4",
        "--beam-size", "1",  # Faster settings
        "--print-progress"
    ]
    
    print(f"Executing whisper command...")
    print(f"Output base: {output_base}")
    
    start_time = time.time()
    
    try:
        # Run directly and capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True,
            cwd=WHISPER_DIR,
            encoding='utf-8',
            errors='replace'
        )
        
        # Read output
        stdout, stderr = process.communicate(timeout=1800)  # 30 minute timeout
        
        elapsed = time.time() - start_time
        print(f"Transcription time: {elapsed:.1f} seconds")
        print(f"Return code: {process.returncode}")
        
        if process.returncode != 0:
            print(f"❌ Transcription failed")
            if stderr:
                print(f"Error details: {stderr[:500]}")
            return []
        
        # Find output files
        txt_file = output_base + ".txt"
        vtt_file = output_base + ".vtt"
        srt_file = output_base + ".srt"
        
        print(f"Looking for output files:")
        print(f"  TXT file: {txt_file} - exists: {os.path.exists(txt_file)}")
        print(f"  VTT file: {vtt_file} - exists: {os.path.exists(vtt_file)}")
        print(f"  SRT file: {srt_file} - exists: {os.path.exists(srt_file)}")
        
        segments = []
        
        # First try SRT file (most reliable for timestamps)
        if os.path.exists(srt_file):
            print(f"Parsing SRT file: {srt_file}")
            segments = parse_srt_file(srt_file)
            if segments:
                print(f"✅ Extracted {len(segments)} segments from SRT")
                return segments
        
        # Then try VTT file
        if os.path.exists(vtt_file) and not segments:
            print(f"Parsing VTT file: {vtt_file}")
            segments = parse_vtt_file(vtt_file)
            if segments:
                print(f"✅ Extracted {len(segments)} segments from VTT")
                return segments
        
        # If no timestamp files, try to extract from stdout
        if not segments:
            print(f"Trying to extract segments from stdout...")
            segments = extract_segments_from_stdout(stdout)
            if segments:
                print(f"✅ Extracted {len(segments)} segments from stdout")
                return segments
        
        # Last resort: use TXT file
        if os.path.exists(txt_file) and not segments:
            print(f"Reading from TXT file: {txt_file}")
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    # If no time info, at least return text
                    segments = [{'start': 0, 'end': 0, 'text': text}]
                    print(f"✅ Got text from TXT file (no timestamps)")
                    return segments
        
        return segments
        
    except subprocess.TimeoutExpired:
        print(f"❌ Command execution timeout (1800 seconds)")
        return []
    except Exception as e:
        print(f"❌ Transcription process error: {e}")
        import traceback
        traceback.print_exc()
        return []

def transcribe_with_json(audio_path, language="en", enable_split=True):
    """Transcribe audio file with support for large file splitting"""
    print(f"\nStarting timestamped transcription: {os.path.basename(audio_path)}")
    
    # Check file size
    file_size = os.path.getsize(audio_path) / 1024 / 1024
    print(f"File size: {file_size:.1f} MB")
    
    if file_size > 200 and enable_split:  # Split if larger than 200MB
        print("Large file, enabling split mode...")
        split_dir, segments = split_audio_file(audio_path, segment_duration=600)
        
        if segments:
            all_segments, _ = transcribe_segments(segments, language)
            
            # Clean up split directory
            if split_dir and os.path.exists(split_dir):
                try:
                    shutil.rmtree(split_dir)
                    print(f"Cleaned split directory: {split_dir}")
                except:
                    print(f"⚠ Unable to clean split directory: {split_dir}")
            
            return all_segments
        else:
            print("Split failed, trying direct processing...")
    
    # Direct processing
    return transcribe_with_json_single(audio_path, language)

def save_results(segments, base_name="transcription"):
    """Save transcription results"""
    if not segments:
        print("❌ No results to save")
        return False
    
    try:
        # Sort by time
        segments.sort(key=lambda x: x['start'])
        
        # Save as JSON
        json_file = f"{base_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"✅ JSON saved: {json_file}")
        
        # Save as timestamped text
        txt_file = f"{base_name}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            for seg in segments:
                if seg['start'] > 0 or seg['end'] > 0:
                    start_min = seg['start'] / 60
                    end_min = seg['end'] / 60
                    f.write(f"[{start_min:.2f}m - {end_min:.2f}m] {seg['text']}\n")
                else:
                    f.write(f"{seg['text']}\n")
        print(f"✅ Timestamped text saved: {txt_file}")
        
        # Save as merged text
        merged_file = f"{base_name}_merged.txt"
        with open(merged_file, 'w', encoding='utf-8') as f:
            merged_text = " ".join(seg['text'] for seg in segments)
            f.write(merged_text)
        print(f"✅ Merged text saved: {merged_file}")
        
        # Save as SRT format
        srt_file = f"{base_name}.srt"
        with open(srt_file, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(segments, 1):
                # Format times for SRT
                start_time = format_time_srt(seg['start'])
                end_time = format_time_srt(seg['end'])
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{seg['text']}\n\n")
        print(f"✅ SRT file saved: {srt_file}")
        
        # Statistics
        total_chars = sum(len(seg['text']) for seg in segments)
        if segments and segments[-1]['end'] > 0:
            total_duration = segments[-1]['end']
            print(f"\n📊 Statistics:")
            print(f"  Number of segments: {len(segments)}")
            print(f"  Total characters: {total_chars}")
            print(f"  Audio duration: {total_duration/60:.1f} minutes")
            if total_duration > 0:
                print(f"  Average speaking speed: {total_chars/(total_duration/60):.0f} words/min")
        
        return True
        
    except Exception as e:
        print(f"❌ Save failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def format_time_srt(seconds):
    """Format time for SRT file"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def cleanup_temp_files():
    """Clean up temporary files"""
    print(f"\nCleaning temp directory: {LOCAL_TEMP_DIR}")
    
    if os.path.exists(LOCAL_TEMP_DIR):
        try:
            # Only delete temp files, keep directory
            for filename in os.listdir(LOCAL_TEMP_DIR):
                file_path = os.path.join(LOCAL_TEMP_DIR, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"  Deleted: {filename}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"  Deleted directory: {filename}")
                except Exception as e:
                    print(f"  Cannot delete {filename}: {e}")
            
            print("✅ Temp files cleaned up")
        except Exception as e:
            print(f"⚠ Cleanup failed: {e}")

def main():
    print("="*60)
    print("whisper.cpp Large File Transcription Tool (Local Temp Files)")
    print("="*60)
    print(f"Local temp directory: {LOCAL_TEMP_DIR}")
    
    # Check environment
    if not os.path.exists(WHISPER_EXE):
        print(f"❌ Whisper executable not found: {WHISPER_EXE}")
        return
    
    if not os.path.exists(MODEL):
        print(f"❌ Model file not found: {MODEL}")
        return
    
    # Check FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=10)
        print("✅ FFmpeg available")
    except:
        print("⚠ FFmpeg may not be installed or not in PATH")
    
    print(f"✅ Environment check passed")
    print(f"   Executable: {WHISPER_EXE}")
    print(f"   Model file: {MODEL}")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    # Show file info
    file_size = os.path.getsize(test_file) / 1024 / 1024
    print(f"\nTranscription file: {test_file}")
    print(f"File size: {file_size:.1f} MB")
    
    if file_size > 1000:  # Larger than 1GB
        print("⚠ Warning: Very large file, transcription may take hours")
        print("Suggest splitting audio or using more powerful hardware")
        
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Transcription cancelled")
            return
    
    print("\nSelect transcription method:")
    print("  1. Timestamped transcription (with split for large files)")
    print("  2. Timestamped transcription (no split, for small files)")
    print("  3. Test single segment only")
    
    choice = input("Choice (1/2/3): ").strip()
    
    start_time = time.time()
    segments = []
    
    if choice == "1":
        segments = transcribe_with_json(test_file, "en", enable_split=True)
    elif choice == "2":
        segments = transcribe_with_json(test_file, "en", enable_split=False)
    elif choice == "3":
        # Test mode: process only first 10-minute segment
        print("\nTest mode: processing first 10-minute segment only")
        split_dir, segment_files = split_audio_file(test_file, segment_duration=600)
        if segment_files and len(segment_files) > 0:
            segments, _ = transcribe_segments([segment_files[0]], "en")
            
            # Clean up other segments
            for i in range(1, len(segment_files)):
                try:
                    os.remove(segment_files[i])
                except:
                    pass
            
            # Clean up split directory
            if split_dir and os.path.exists(split_dir):
                try:
                    shutil.rmtree(split_dir)
                except:
                    pass
    else:
        print("Invalid choice, using default method (with split)")
        segments = transcribe_with_json(test_file, "en", enable_split=True)
    
    total_time = time.time() - start_time
    
    if segments:
        print(f"\n✅ Transcription successful!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Total segments: {len(segments)}")
        
        print("\nFirst 10 segments:")
        for i, seg in enumerate(segments[:10]):
            start_min = seg['start'] / 60
            end_min = seg['end'] / 60
            text_preview = seg['text'][:50] + "..." if len(seg['text']) > 50 else seg['text']
            print(f"  [{i+1:2d}] [{start_min:6.2f}m - {end_min:6.2f}m]: {text_preview}")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"transcription_{timestamp}"
        
        print(f"\nSaving transcription results...")
        save_results(segments, base_name)
    else:
        print("\n❌ Transcription failed, please check:")
        print("   1. File is not corrupted")
        print("   2. Audio format is supported")
        print("   3. Whisper model is complete")
        print("   4. Enough RAM is available (large files need lots of RAM)")
    
    # Clean up temp files
    cleanup_temp_files()
    
    print("\n" + "="*60)
    print("Complete")
    print("="*60)

# Compatibility interface function
def transcribe_with_whisper_cpp(wav_path: str, language: str = "en") -> List[Dict[str, any]]:
    """Compatibility interface function"""
    return transcribe_with_json(wav_path, language, enable_split=True)

if __name__ == "__main__":
    main()