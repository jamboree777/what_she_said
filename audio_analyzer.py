import os
import librosa
import numpy as np
from pyannote.audio import Pipeline
from transformers import pipeline, AutoModelForSequenceClassification, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import torch
import tempfile
import logging
import soundfile as sf
import warnings
import subprocess
import shutil
import wave
import audioread
import sounddevice as sd
from dotenv import load_dotenv
from datetime import datetime
from pydub import AudioSegment
import json

# .env 파일 로드
load_dotenv()

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Hugging Face 토큰 설정
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")

def find_ffmpeg():
    """ffmpeg 실행 파일을 찾는 함수"""
    # 가능한 ffmpeg 경로들
    possible_paths = [
        r"C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg"  # PATH에서 찾기
    ]
    
    # 각 경로 확인
    for path in possible_paths:
        try:
            if path == "ffmpeg":
                # PATH에서 찾기
                result = subprocess.run(["where", "ffmpeg"], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip().split('\n')[0]
            else:
                # 직접 경로 확인
                if os.path.exists(path):
                    return path
        except Exception:
            continue
    
    raise FileNotFoundError("ffmpeg를 찾을 수 없습니다. ffmpeg가 설치되어 있는지 확인해주세요.")

def convert_to_wav(audio_file):
    """오디오 파일을 WAV 형식으로 변환"""
    try:
        logger.info(f"파일 변환 시작: {audio_file}")
        
        # 임시 WAV 파일 생성
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_wav.close()
        
        # ffmpeg를 사용하여 변환
        try:
            # ffmpeg 경로 찾기
            ffmpeg_path = find_ffmpeg()
            logger.info(f"ffmpeg 경로: {ffmpeg_path}")
            
            # 변환 명령 실행 (16kHz, 16bit, mono)
            subprocess.run([
                ffmpeg_path, 
                '-i', audio_file,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',  # 기존 파일 덮어쓰기
                temp_wav.name
            ], check=True, capture_output=True)
            
            # 변환된 파일 검증
            if not os.path.exists(temp_wav.name):
                raise FileNotFoundError(f"변환된 파일을 찾을 수 없습니다: {temp_wav.name}")
            
            # WAV 파일 형식 검증
            try:
                with wave.open(temp_wav.name, 'rb') as wav_file:
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    frame_rate = wav_file.getframerate()
                    
                    if channels != 1 or sample_width != 2 or frame_rate != 16000:
                        raise ValueError(f"잘못된 WAV 파일 형식: channels={channels}, width={sample_width}, rate={frame_rate}")
            except Exception as e:
                raise ValueError(f"WAV 파일 검증 실패: {str(e)}")
            
            logger.info(f"ffmpeg 변환 성공: {temp_wav.name}")
            return temp_wav.name
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg 변환 실패: {e.stderr.decode()}")
            raise
        except Exception as e:
            logger.error(f"ffmpeg 실행 중 오류 발생: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"파일 변환 중 오류 발생: {str(e)}")
        raise

def load_audio(audio_file):
    """오디오 파일을 로드하는 함수"""
    try:
        # PySoundFile로 시도
        try:
            audio, sr = librosa.load(audio_file, sr=16000, mono=True)
            return audio, sr
        except Exception as e:
            logger.warning(f"PySoundFile 로드 실패, audioread로 시도: {str(e)}")
        
        # audioread로 시도
        try:
            with audioread.audio_open(audio_file) as f:
                audio = np.frombuffer(f.read_raw(), dtype=np.float32)
                sr = f.samplerate
                if sr != 16000:
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000
                return audio, sr
        except Exception as e:
            logger.warning(f"audioread 로드 실패, sounddevice로 시도: {str(e)}")
        
        # sounddevice로 시도
        try:
            audio, sr = sd.read(audio_file)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)  # 스테레오를 모노로 변환
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            return audio, sr
        except Exception as e:
            logger.error(f"모든 오디오 로드 방법 실패: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"오디오 파일 로드 실패: {str(e)}")
        raise

def initialize_models():
    """모델 초기화 함수"""
    try:
        # 화자 분할 초기화
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN
        )
        
        # 감정 분석 모델 초기화
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            num_labels=7  # 감정 클래스 수
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        
        # 감정 분석 파이프라인 초기화
        emotion_analyzer = pipeline(
            "audio-classification",
            model=model,
            feature_extractor=feature_extractor,
            device=-1  # CPU 사용
        )
        
        return diarization_pipeline, emotion_analyzer
    except Exception as e:
        logger.error(f"모델 초기화 중 오류 발생: {str(e)}")
        raise

def analyze_audio(audio_file):
    try:
        logger.info(f"분석 시작: {audio_file}")
        
        # 파일 존재 확인
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {audio_file}")
        
        # m4a 파일을 WAV로 변환
        if audio_file.lower().endswith('.m4a'):
            audio_file = convert_to_wav(audio_file)
        
        # 모델 초기화
        diarization_pipeline, emotion_analyzer = initialize_models()
        
        # 화자 분할 실행
        diarization = diarization_pipeline(audio_file)
        
        # 오디오 파일 로드
        audio, sr = load_audio(audio_file)
        
        # 결과 저장
        results = []
        
        # 각 세그먼트에 대해 감정 분석 수행
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            try:
                # 오디오 세그먼트 추출
                start_sample = int(turn.start * sr)
                end_sample = int(turn.end * sr)
                segment = audio[start_sample:end_sample]
                
                # 세그먼트가 너무 짧으면 건너뛰기
                if len(segment) < sr * 0.5:  # 0.5초 미만인 경우
                    continue
                    
                # 감정 분석 수행
                emotion_result = emotion_analyzer(segment)
                
                results.append({
                    'speaker': speaker,
                    'start': turn.start,
                    'end': turn.end,
                    'emotion': emotion_result[0]['label'],
                    'confidence': float(emotion_result[0]['score'])
                })
            except Exception as e:
                logger.error(f"세그먼트 처리 중 오류 발생: {str(e)}")
                continue
        
        # 임시 파일 삭제
        if audio_file.endswith('.wav') and os.path.exists(audio_file):
            try:
                os.unlink(audio_file)
                logger.info(f"임시 파일 삭제 완료: {audio_file}")
            except Exception as e:
                logger.error(f"임시 파일 삭제 중 오류 발생: {str(e)}")
            
        return results
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        raise

class AudioAnalyzer:
    def __init__(self):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HF_TOKEN")
        )
        self.recording = False
        self.audio_data = []
        self.sample_rate = 16000  # pyannote 권장 샘플링 레이트
        self.channels = 1  # 모노 채널
        self.speakers = {}
        self.current_speaker_id = 0

    def start_recording(self):
        """녹음 시작"""
        self.recording = True
        self.audio_data = []
        
        def callback(indata, frames, time, status):
            if status:
                print(status)
            if self.recording:
                # 스테레오를 모노로 변환 (필요한 경우)
                if indata.shape[1] > 1:
                    indata = indata.mean(axis=1, keepdims=True)
                self.audio_data.append(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            callback=callback
        ):
            print("녹음이 시작되었습니다. 중지하려면 Ctrl+C를 누르세요.")
            while self.recording:
                sd.sleep(100)

    def stop_recording(self):
        """녹음 중지"""
        self.recording = False
        if self.audio_data:
            # 모든 오디오 데이터를 하나로 합치기
            audio_data = np.concatenate(self.audio_data, axis=0)
            
            # 정규화 (-1.0 ~ 1.0 범위로)
            audio_data = np.clip(audio_data, -1.0, 1.0)
            
            # 16비트 PCM으로 변환
            audio_data = (audio_data * 32767).astype(np.int16)
            
            # 파일 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.wav"
            
            # WAV 파일로 저장
            with sf.SoundFile(filename, 'w', self.sample_rate, self.channels, 'PCM_16') as f:
                f.write(audio_data)
            
            return filename
        return None

    def analyze_speakers(self, audio_file):
        """화자 분리 분석"""
        diarization = self.pipeline(audio_file)
        results = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "text": "",  # STT 결과가 들어갈 자리
                "confidence": 0.0,  # 화자 분리 신뢰도
                "emotion": None  # 감정 분석 결과
            })
        
        return results

    def add_speaker(self, name):
        """새로운 화자 추가"""
        if len(self.speakers) >= 10:
            raise ValueError("최대 10명의 화자만 등록할 수 있습니다.")
        
        self.speakers[f"SPEAKER_{self.current_speaker_id}"] = name
        self.current_speaker_id += 1
        return self.current_speaker_id - 1

    def save_transcript(self, results, output_file):
        """분석 결과를 JSON 형식으로 저장"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "speakers": self.speakers,
                "transcript": results
            }, f, ensure_ascii=False, indent=2)

    def get_unclear_segments(self, results, confidence_threshold=0.7):
        """신뢰도가 낮은 구간 반환"""
        return [segment for segment in results if segment["confidence"] < confidence_threshold]

if __name__ == "__main__":
    # 오디오 파일 경로 설정
    audio_file = "path/to/your/audio.wav"
    
    # 분석 실행
    results = analyze_audio(audio_file)
    
    # 결과 출력
    for result in results:
        print(f"화자: {result['speaker']}")
        print(f"시간: {result['start']:.2f}s - {result['end']:.2f}s")
        print(f"감정: {result['emotion']} (신뢰도: {result['confidence']:.2f})")
        print("-" * 50) 