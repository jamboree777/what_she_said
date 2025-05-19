import os
import torch
import numpy as np
import librosa
import soundfile as sf
from pyannote.audio import Pipeline
from transformers import pipeline
from datetime import datetime

class SpeechAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # 화자 분리 파이프라인 초기화
        self.speaker_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HF_TOKEN")
        ).to(self.device)
        
        # 음성 인식 파이프라인 초기화
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="kakao-enterprise/vosk-model-ko-0.22",
            device=0 if self.device == "cuda" else -1
        )
        
    def analyze_audio(self, audio_path):
        """오디오 파일을 분석하여 화자 분리와 음성 인식을 수행합니다."""
        try:
            # 오디오 파일 로드
            waveform, sample_rate = librosa.load(audio_path, sr=16000)
            
            # 화자 분리 수행
            diarization = self.speaker_pipeline({
                "waveform": torch.from_numpy(waveform).to(self.device),
                "sample_rate": sample_rate
            })
            
            # 결과 저장
            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # 해당 구간의 오디오 추출
                start_sample = int(turn.start * sample_rate)
                end_sample = int(turn.end * sample_rate)
                segment = waveform[start_sample:end_sample]
                
                # 음성 인식 수행
                transcription = self.asr_pipeline(segment)
                
                results.append({
                    "speaker": speaker,
                    "start_time": turn.start,
                    "end_time": turn.end,
                    "text": transcription["text"],
                    "confidence": transcription.get("confidence", 0.0)
                })
            
            return results
            
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            return None
    
    def save_results(self, results, output_path):
        """분석 결과를 텍스트 파일로 저장합니다."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for result in results:
                    f.write(f"[{result['start_time']:.2f}s - {result['end_time']:.2f}s] "
                           f"화자 {result['speaker']} (신뢰도: {result['confidence']:.2f})\n")
                    f.write(f"{result['text']}\n\n")
                    
            print(f"Results saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False

if __name__ == "__main__":
    # 환경 변수에서 Hugging Face 토큰 가져오기
    if not os.getenv("HF_TOKEN"):
        print("Please set your Hugging Face token as HF_TOKEN environment variable")
        exit(1)
    
    # 분석기 초기화
    analyzer = SpeechAnalyzer()
    
    # 테스트 오디오 파일 분석
    audio_path = "test.wav"  # 테스트용 오디오 파일 경로
    if os.path.exists(audio_path):
        results = analyzer.analyze_audio(audio_path)
        if results:
            analyzer.save_results(results, "analysis_results.txt")
    else:
        print(f"Test audio file not found: {audio_path}") 