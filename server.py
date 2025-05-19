import os
import torch
import whisperx
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from kiwipiepy import Kiwi
from soynlp.word import WordExtractor
from pyannote.audio import Pipeline
import gc
import subprocess
import tempfile
import traceback
import logging
from logging.handlers import RotatingFileHandler

# 로깅 설정
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 파일 핸들러 설정
file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'server.log'),
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))

# 콘솔 핸들러 설정
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))

# 로거 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

app = Flask(__name__)
CORS(app)

# 전역 변수로 모델 선언 (메모리 효율을 위해)
whisper_model = None
diarization_model = None
kiwi = None
word_extractor = None

def convert_to_wav(input_path):
    """오디오 파일을 WAV 형식으로 변환합니다"""
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "temp_audio.wav")
    
    logger.info(f"오디오 파일 변환 시작: {input_path} -> {output_path}")
    
    # ffmpeg를 사용하여 WAV로 변환
    command = [
        'ffmpeg', '-i', input_path,
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y', output_path
    ]
    
    try:
        # 환경 변수 설정으로 인코딩 문제 해결
        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=my_env
        )
        logger.info("오디오 변환 성공")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = f"오디오 변환 실패: {e.stderr}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"오디오 변환 중 예상치 못한 오류: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def convert_m4p_to_wav(input_path):
    """M4P 파일을 WAV 형식으로 변환합니다"""
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "temp_audio.wav")
    
    logger.info(f"M4P 파일 변환 시작: {input_path} -> {output_path}")
    
    # ffmpeg를 사용하여 WAV로 변환
    command = [
        'ffmpeg', '-i', input_path,
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        '-y', output_path
    ]
    
    try:
        # 환경 변수 설정으로 인코딩 문제 해결
        my_env = os.environ.copy()
        my_env["PYTHONIOENCODING"] = "utf-8"
        
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=my_env
        )
        logger.info("M4P 변환 성공")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = f"M4P 변환 실패: {e.stderr}"
        logger.error(error_msg)
        raise Exception(error_msg)
    except Exception as e:
        error_msg = f"M4P 변환 중 예상치 못한 오류: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def load_models():
    """필요한 모델들을 로드합니다"""
    global whisper_model, diarization_model, kiwi, word_extractor
    
    try:
        if whisper_model is None:
            logger.info("WhisperX 모델 로딩 중...")
            whisper_model = whisperx.load_model("base", device="cpu", compute_type="int8")
        
        if diarization_model is None:
            logger.info("화자 분리 모델 로딩 중...")
            diarization_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.getenv('HF_TOKEN')
            )
            diarization_model.to(torch.device("cpu"))
        
        if kiwi is None:
            logger.info("Kiwi 형태소 분석기 로딩 중...")
            kiwi = Kiwi()
        
        if word_extractor is None:
            logger.info("Soynlp 단어 추출기 초기화 중...")
            word_extractor = WordExtractor()
            
    except Exception as e:
        error_msg = f"모델 로딩 실패: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        raise Exception(error_msg)

def unload_models():
    """메모리 관리를 위해 모델을 언로드합니다"""
    global whisper_model, diarization_model, kiwi, word_extractor
    
    try:
        if whisper_model is not None:
            del whisper_model
            whisper_model = None
        
        if diarization_model is not None:
            del diarization_model
            diarization_model = None
        
        if kiwi is not None:
            del kiwi
            kiwi = None
        
        if word_extractor is not None:
            del word_extractor
            word_extractor = None
        
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("모델 언로드 완료")
        
    except Exception as e:
        error_msg = f"모델 언로드 실패: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

@app.route('/')
def index():
    """메인 페이지를 렌더링합니다"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """오디오 파일을 업로드하고 처리합니다"""
    temp_path = None
    wav_path = None
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '선택된 파일이 없습니다'}), 400
        
        # 임시 파일로 저장
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        logger.info(f"파일 저장 완료: {temp_path}")
        
        # M4P 파일인 경우 WAV로 변환
        if temp_path.lower().endswith('.m4p'):
            wav_path = convert_m4p_to_wav(temp_path)
        else:
            wav_path = convert_to_wav(temp_path)
        
        # 모델 로드
        load_models()
        
        # 음성 인식
        logger.info("음성 인식 시작...")
        result = whisper_model.transcribe(wav_path, batch_size=16)
        logger.info("음성 인식 완료")
        
        # 화자 분리
        logger.info("화자 분리 시작...")
        diarize_segments = diarization_model(wav_path)
        logger.info("화자 분리 완료")
        
        # 결과 통합
        try:
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as e:
            logger.error(f"화자 분리 결과 통합 실패: {str(e)}")
            # 화자 분리 실패 시 기본 결과 반환
            result = {
                'segments': [
                    {
                        'text': seg['text'],
                        'start': seg['start'],
                        'end': seg['end'],
                        'speaker': 'UNKNOWN'
                    }
                    for seg in result['segments']
                ]
            }
        
        # 메모리 정리
        unload_models()
        
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
        
        return jsonify({
            'success': True,
            'segments': result['segments']
        })
        
    except Exception as e:
        error_msg = f"처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        # 메모리 정리
        try:
            unload_models()
        except:
            pass
        
        # 임시 파일 삭제
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        
        return jsonify({'error': error_msg}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태를 확인합니다"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Hugging Face 토큰 확인
    if not os.getenv('HF_TOKEN'):
        logger.warning("HF_TOKEN 환경 변수가 설정되지 않았습니다")
        print("경고: HF_TOKEN 환경 변수가 설정되지 않았습니다")
        print("Hugging Face 토큰을 설정하려면 다음 명령어를 실행하세요:")
        print("$env:HF_TOKEN = 'your_token_here'")
    
    # 서버 실행
    app.run(host='127.0.0.1', port=5000, debug=True) 