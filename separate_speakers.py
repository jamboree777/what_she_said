import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinterdnd2 import DND_FILES, TkinterDnD

def separate_speakers(input_path):
    """WAV 파일에서 화자를 분리합니다"""
    # 입력 파일의 디렉토리와 파일명 분리
    directory = os.path.dirname(input_path)
    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join(directory, f"{filename}_speaker_separation")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 상태바 초기화
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(
        main_frame,
        variable=progress_var,
        maximum=100,
        mode='determinate'
    )
    progress_bar.pack(fill=tk.X, padx=20, pady=10)
    
    # 현재 작업 표시 레이블
    current_task_label = ttk.Label(
        main_frame,
        text="Ready...",
        font=("Arial", 10)
    )
    current_task_label.pack(pady=5)
    
    status_text.delete(1.0, tk.END)
    status_text.insert(tk.END, f"Starting speaker separation: {input_path}\n")
    root.update()
    
    def update_progress(task, progress):
        """진행 상태 업데이트"""
        current_task_label.config(text=task)
        progress_var.set(progress)
        root.update()
    
    # pyannote.audio를 사용하여 화자 분리
    command = [
        'python', '-c',
        f'''
import torch
from pyannote.audio import Pipeline
import os
import sys
import traceback
import subprocess
import tempfile
import json
import numpy as np
import time

def convert_to_wav(input_path):
    """Convert audio file to WAV format"""
    try:
        # Create temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        
        # Convert to WAV using ffmpeg
        command = [
            "ffmpeg",
            "-i", input_path,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            temp_wav.name
        ]
        
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        
        if result.returncode != 0:
            raise Exception(f"Conversion failed: {{result.stderr}}")
            
        return temp_wav.name
    except Exception as e:
        print(f"Error during file conversion: {{str(e)}}")
        return None

def create_pipeline():
    try:
        print("Initializing Pipeline...")
        # Direct model loading with custom parameters
        HF_TOKEN = os.getenv('HF_TOKEN')
        if not HF_TOKEN:
            print("경고: HF_TOKEN 환경 변수가 설정되지 않았습니다")
            print("PowerShell에서 다음 명령어를 실행하세요:")
            print("$env:HF_TOKEN = 'your_token_here'")
            sys.exit(1)
            
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN
        )
        
        # Configure pipeline parameters
        pipeline.instantiate({{
            "segmentation": {{
                "min_duration_on": 0.5,  # 최소 음성 구간 길이 (초)
                "min_duration_off": 0.5,  # 최소 묵음 구간 길이 (초)
                "onset": 0.5,  # 음성 시작 임계값
                "offset": 0.5,  # 음성 종료 임계값
                "min_activity": 0.5,  # 최소 활동 임계값
            }},
            "clustering": {{
                "min_cluster_size": 15,  # 최소 클러스터 크기
                "threshold": 0.7,  # 클러스터링 임계값
            }}
        }})
        
        print("Pipeline initialization complete")
        return pipeline
    except Exception as e:
        print(f"Error during Pipeline initialization: {{str(e)}}")
        print("Detailed error:")
        traceback.print_exc()
        return None

def format_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{{hours:02d}}:{{minutes:02d}}:{{seconds:05.2f}}"

def calculate_confidence_score(processing_time, duration, is_silence):
    """Calculate confidence score (0-100) based on processing time and duration"""
    if is_silence:
        return 100  # 묵음 구간은 최고 신뢰도
    
    ratio = processing_time / duration
    if ratio <= 0.1:
        return 90  # 매우 높은 신뢰도
    elif ratio <= 0.2:
        return 80  # 높은 신뢰도
    elif ratio <= 0.3:
        return 70  # 중간 신뢰도
    elif ratio <= 0.4:
        return 60  # 낮은 신뢰도
    else:
        return 50  # 매우 낮은 신뢰도

try:
    # Convert to WAV
    print("Converting audio to WAV format...")
    wav_path = convert_to_wav("{input_path}")
    if wav_path is None:
        raise Exception("Audio conversion failed")
    print("WAV conversion complete")

    # Create Pipeline
    print("Initializing Pipeline...")
    pipeline = create_pipeline()
    if pipeline is None:
        raise Exception("Pipeline creation failed")
    print("Pipeline initialization complete")

    # Start speaker separation
    print("Starting speaker separation...")
    start_time = time.time()
    diarization = pipeline(wav_path)
    total_processing_time = time.time() - start_time

    # Sort results by time
    print("Sorting results...")
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        duration = turn.duration
        # 각 구간의 처리 시간을 전체 처리 시간의 비율로 계산
        processing_time = (duration / diarization.duration) * total_processing_time
        # 묵음 구간 여부 확인
        is_silence = not turn.is_speech if hasattr(turn, 'is_speech') else False
        segments.append((turn.start, turn.end, speaker, duration, processing_time, is_silence))
    segments.sort(key=lambda x: x[0])

    # Create speaker mapping
    print("Creating speaker mapping...")
    speaker_mapping = {{}}
    speaker_count = 1
    for _, _, speaker, _, _, _ in segments:
        if speaker not in speaker_mapping:
            speaker_mapping[speaker] = f"Speaker {{speaker_count}}"
            speaker_count += 1

    # Calculate processing time statistics
    processing_times = [pt for _, _, _, _, pt, _ in segments]
    avg_processing_time = np.mean(processing_times)
    std_processing_time = np.std(processing_times)
    high_processing_threshold = avg_processing_time + std_processing_time

    # Save results in JSON format
    print("Saving results...")
    result_data = {{
        "segments": [
            {{
                "start": start,
                "end": end,
                "speaker": speaker_mapping[speaker],
                "start_time": format_time(start),
                "end_time": format_time(end),
                "duration": duration,
                "processing_time": processing_time,
                "processing_ratio": processing_time / duration,
                "confidence_score": calculate_confidence_score(processing_time, duration, is_silence),
                "is_silence": is_silence,
                "is_high_processing": processing_time > high_processing_threshold
            }}
            for start, end, speaker, duration, processing_time, is_silence in segments
        ],
        "speaker_mapping": {{
            mapped: original
            for original, mapped in speaker_mapping.items()
        }},
        "processing_statistics": {{
            "total_processing_time": float(total_processing_time),
            "average_processing_time": float(avg_processing_time),
            "std_dev": float(std_processing_time),
            "high_processing_threshold": float(high_processing_threshold)
        }}
    }}

    # Save as JSON file
    json_path = os.path.join(os.path.dirname(wav_path), "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    # Save as text file
    txt_path = os.path.join(os.path.dirname(wav_path), "result.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Speaker Separation Results ===\\n\\n")
        
        # Write processing statistics
        f.write("[Processing Statistics]\\n")
        f.write(f"Total Processing Time: {{total_processing_time:.2f}} seconds\\n")
        f.write(f"Average Processing Time per Segment: {{avg_processing_time:.2f}} seconds\\n")
        f.write(f"Standard Deviation: {{std_processing_time:.2f}} seconds\\n")
        f.write(f"High Processing Threshold: {{high_processing_threshold:.2f}} seconds\\n\\n")
        
        f.write("[Segment Information]\\n")
        f.write("Start Time | End Time | Speaker | Duration | Process Time | Ratio | Score | Type | Status\\n")
        f.write("-" * 100 + "\\n")
        for start, end, speaker, duration, processing_time, is_silence in segments:
            mapped_speaker = speaker_mapping[speaker]
            ratio = processing_time / duration
            confidence_score = calculate_confidence_score(processing_time, duration, is_silence)
            segment_type = "Silence" if is_silence else "Speech"
            is_high = processing_time > high_processing_threshold
            status = "⚠️ High Processing" if is_high else "✓"
            f.write(f"{{format_time(start)}} | {{format_time(end)}} | {{mapped_speaker}} | {{duration:.2f}}s | {{processing_time:.2f}}s | {{ratio:.2f}} | {{confidence_score:3d}} | {{segment_type:7}} | {{status}}\\n")
        
        f.write("\\n[Speaker Information]\\n")
        f.write("Speaker ID | Original ID\\n")
        f.write("-" * 30 + "\\n")
        for original, mapped in speaker_mapping.items():
            f.write(f"{{mapped:8}} | {{original}}\\n")

    # Print results
    with open(txt_path, "r", encoding="utf-8") as f:
        print(f.read())

    # Clean up temporary files
    try:
        os.unlink(wav_path)
    except:
        pass

except Exception as e:
    print(f"Error occurred: {{str(e)}}")
    print("Detailed error:")
    traceback.print_exc()
    raise
        '''
    ]
    
    try:
        # 진행 상태 업데이트 함수
        def update_status(line):
            if "Converting audio to WAV format" in line:
                update_progress("Converting audio to WAV...", 10)
            elif "WAV conversion complete" in line:
                update_progress("WAV conversion complete", 20)
            elif "Initializing Pipeline" in line:
                update_progress("Initializing Pipeline...", 30)
            elif "Pipeline initialization complete" in line:
                update_progress("Pipeline initialization complete", 40)
            elif "Starting speaker separation" in line:
                update_progress("Separating speakers...", 50)
            elif "Sorting results" in line:
                update_progress("Sorting results...", 70)
            elif "Creating speaker mapping" in line:
                update_progress("Creating speaker mapping...", 80)
            elif "Saving results" in line:
                update_progress("Saving results...", 90)
            elif "=== Speaker Separation Results ===" in line:
                update_progress("Complete!", 100)
        
        # 프로세스 실행 및 실시간 출력 처리
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # 실시간으로 출력 처리
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                update_status(line)
                status_text.insert(tk.END, line)
                status_text.see(tk.END)
                root.update()
        
        # 에러 체크
        if process.returncode != 0:
            error = process.stderr.read()
            status_text.delete(1.0, tk.END)
            status_text.insert(tk.END, f"Speaker separation failed:\n{error}")
            return None
        
        # 결과를 파일로 저장
        output_file = os.path.join(output_dir, "speaker_separation_result.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(status_text.get(1.0, tk.END))
        
        return output_file
    except Exception as e:
        status_text.delete(1.0, tk.END)
        status_text.insert(tk.END, f"Unexpected error:\n{str(e)}")
        return None

def drop(event):
    """파일 드롭 이벤트 처리"""
    file_path = event.data
    # 중괄호 제거 (Windows 드래그 앤 드롭 형식)
    file_path = file_path.strip('{}')
    if os.path.exists(file_path):
        separate_speakers(file_path)
    else:
        status_text.delete(1.0, tk.END)
        status_text.insert(tk.END, f"오류: 파일을 찾을 수 없습니다: {file_path}")

# GUI 생성
root = TkinterDnD.Tk()
root.title("화자 분리 프로그램")
root.geometry("800x600")

# 스타일 설정
style = ttk.Style()
style.configure("TLabel", padding=10)
style.configure("TButton", padding=10)

# 메인 프레임
main_frame = ttk.Frame(root, padding="20")
main_frame.pack(fill=tk.BOTH, expand=True)

# 안내 레이블
guide_label = ttk.Label(
    main_frame,
    text="WAV 파일을 여기에 드래그 앤 드롭하세요",
    font=("Arial", 12)
)
guide_label.pack(pady=20)

# 드롭 영역
drop_frame = ttk.Frame(main_frame, relief="solid", borderwidth=2)
drop_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

drop_label = ttk.Label(
    drop_frame,
    text="파일을 여기에 드롭하세요",
    font=("Arial", 14)
)
drop_label.pack(expand=True)

# 상태 표시 영역 (스크롤 가능한 텍스트 위젯)
status_text = scrolledtext.ScrolledText(
    main_frame,
    wrap=tk.WORD,
    width=80,
    height=20,
    font=("Consolas", 10)
)
status_text.pack(pady=20, fill=tk.BOTH, expand=True)

# 드래그 앤 드롭 설정
drop_frame.drop_target_register(DND_FILES)
drop_frame.dnd_bind('<<Drop>>', drop)

# 실행
if __name__ == "__main__":
    root.mainloop() 