import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
from tkinterdnd2 import DND_FILES, TkinterDnD

def convert_to_wav(input_path):
    """오디오 파일을 WAV 형식으로 변환합니다"""
    # 입력 파일의 디렉토리와 파일명 분리
    directory = os.path.dirname(input_path)
    filename = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(directory, f"{filename}_변환.wav")
    
    status_label.config(text=f"변환 시작: {input_path} -> {output_path}")
    root.update()
    
    # ffmpeg를 사용하여 WAV로 변환
    command = [
        'ffmpeg', '-i', input_path,
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y', output_path
    ]
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        status_label.config(text=f"변환 성공!\n변환된 파일: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        status_label.config(text=f"변환 실패: {e.stderr}")
        return None
    except Exception as e:
        status_label.config(text=f"예상치 못한 오류: {str(e)}")
        return None

def drop(event):
    """파일 드롭 이벤트 처리"""
    file_path = event.data
    # 중괄호 제거 (Windows 드래그 앤 드롭 형식)
    file_path = file_path.strip('{}')
    if os.path.exists(file_path):
        convert_to_wav(file_path)
    else:
        status_label.config(text=f"오류: 파일을 찾을 수 없습니다: {file_path}")

# GUI 생성
root = TkinterDnD.Tk()
root.title("오디오 파일 WAV 변환기")
root.geometry("600x400")

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
    text="변환할 오디오 파일을 여기에 드래그 앤 드롭하세요",
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

# 상태 표시 레이블
status_label = ttk.Label(
    main_frame,
    text="대기 중...",
    wraplength=500
)
status_label.pack(pady=20)

# 드래그 앤 드롭 설정
drop_frame.drop_target_register(DND_FILES)
drop_frame.dnd_bind('<<Drop>>', drop)

# 실행
if __name__ == "__main__":
    root.mainloop() 