from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pyngrok.ngrok as ngrok
import os
import torch
import whisperx
from pyannote.audio import Pipeline
import tempfile
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from kiwipiepy import Kiwi
import json
import numpy as np
from collections import defaultdict
from soynlp.word import WordExtractor
import gc

app = Flask(__name__)
CORS(app)

# 모델 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 전역 변수로 모델 선언 (메모리 효율을 위해)
whisper_model = None
diarization_model = None
kiwi = None
word_extractor = None

def load_models():
    """필요한 모델들을 로드합니다."""
    global whisper_model, diarization_model, kiwi, word_extractor
    
    if whisper_model is None:
        print("WhisperX 모델 로딩 중...")
        whisper_model = whisperx.load_model("large-v3", device="cpu", compute_type="int8")
    
    if diarization_model is None:
        print("화자 분리 모델 로딩 중...")
        diarization_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv('HF_TOKEN'), device="cpu")
    
    if kiwi is None:
        print("Kiwi 형태소 분석기 로딩 중...")
        kiwi = Kiwi()
    
    if word_extractor is None:
        print("Soynlp 단어 추출기 초기화 중...")
        word_extractor = WordExtractor()

def unload_models():
    """메모리 관리를 위해 모델을 언로드합니다."""
    global whisper_model, diarization_model, kiwi, word_extractor
    
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

# 도메인 전문용어 사전
domain_terms = {
    "의학": ["진단", "처방", "증상", "치료"],
    "법률": ["소송", "계약", "법원", "판결"],
    "IT": ["프로그램", "코드", "서버", "데이터베이스"]
}

# 사투리 사전
dialect_dict = {
    "먹나": "먹니",
    "가나": "가니",
    "하나": "하니"
}

# 화자 프로필 저장소
speaker_profiles = defaultdict(dict)

# 피드백 데이터 저장소
feedback_data = []

def preprocess_text(text):
    """텍스트 전처리"""
    # 사투리 변환
    for dialect, standard in dialect_dict.items():
        text = text.replace(dialect, standard)
    return text

def analyze_korean_text(text):
    """한국어 텍스트 분석"""
    # 형태소 분석
    tokens = kiwi.analyze(text)
    
    # 전문용어 감지
    detected_terms = []
    for domain, terms in domain_terms.items():
        for term in terms:
            if term in text:
                detected_terms.append({"domain": domain, "term": term})
    
    return {
        "tokens": tokens,
        "domain_terms": detected_terms
    }

def calculate_confidence(segment, speaker_profile=None):
    """신뢰도 계산"""
    base_confidence = segment.get("confidence", 0.8)
    
    # 화자 프로필이 있는 경우 추가 검증
    if speaker_profile:
        # 화자 음성 특성과의 유사도 계산
        similarity = calculate_speaker_similarity(segment, speaker_profile)
        base_confidence = (base_confidence + similarity) / 2
    
    return base_confidence

def calculate_speaker_similarity(segment, speaker_profile):
    """화자 음성 특성 유사도 계산"""
    # TODO: 실제 음성 특성 비교 로직 구현
    return 0.8  # 임시값

def detect_new_speaker(segment, known_speakers):
    """새로운 화자 감지"""
    # TODO: 실제 화자 감지 로직 구현
    return len(known_speakers) == 0

@app.route('/upload', methods=['POST'])
def upload_file():
    """오디오 파일을 업로드하고 처리합니다."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '선택된 파일이 없습니다'}), 400
        
        # 임시 파일로 저장
        temp_path = f"temp_{file.filename}"
        file.save(temp_path)
        
        try:
            # 모델 로드
            load_models()
            
            # 음성 인식
            print("음성 인식 중...")
            result = whisper_model.transcribe(temp_path, batch_size=16)
            
            # 화자 분리
            print("화자 분리 중...")
            diarize_segments = diarization_model(temp_path)
            
            # 결과 통합
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # 메모리 정리
            unload_models()
            
            # 임시 파일 삭제
            os.remove(temp_path)
            
            return jsonify({
                'success': True,
                'segments': result['segments']
            })
            
        except Exception as e:
            # 에러 발생 시 메모리 정리
            unload_models()
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_transcript', methods=['POST'])
def update_transcript():
    """사용자가 수정한 텍스트 업데이트"""
    data = request.json
    try:
        # 수정된 텍스트 처리
        updated_text = preprocess_text(data['text'])
        analysis = analyze_korean_text(updated_text)
        
        # 피드백 데이터 저장
        feedback_data.append({
            "original_text": data.get("original_text", ""),
            "updated_text": updated_text,
            "timestamp": datetime.now().isoformat(),
            "feedback_type": data.get("feedback_type", "manual_edit")
        })
        
        return jsonify({
            'message': 'Transcript updated successfully',
            'updated_text': updated_text,
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_domain_term', methods=['POST'])
def add_domain_term():
    """도메인 전문용어 추가"""
    data = request.json
    try:
        domain = data['domain']
        term = data['term']
        
        if domain not in domain_terms:
            domain_terms[domain] = []
        
        if term not in domain_terms[domain]:
            domain_terms[domain].append(term)
        
        return jsonify({
            'message': 'Domain term added successfully',
            'domain_terms': domain_terms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/add_dialect', methods=['POST'])
def add_dialect():
    """사투리 사전 추가"""
    data = request.json
    try:
        dialect = data['dialect']
        standard = data['standard']
        
        dialect_dict[dialect] = standard
        
        return jsonify({
            'message': 'Dialect added successfully',
            'dialect_dict': dialect_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_speaker_profile', methods=['POST'])
def update_speaker_profile():
    """화자 프로필 업데이트"""
    data = request.json
    try:
        speaker = data['speaker']
        profile_data = data['profile']
        
        speaker_profiles[speaker].update(profile_data)
        
        return jsonify({
            'message': 'Speaker profile updated successfully',
            'profile': speaker_profiles[speaker]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_feedback_stats', methods=['GET'])
def get_feedback_stats():
    """피드백 통계 조회"""
    try:
        stats = {
            "total_feedbacks": len(feedback_data),
            "feedback_types": defaultdict(int),
            "confidence_trends": []
        }
        
        for feedback in feedback_data:
            stats["feedback_types"][feedback["feedback_type"]] += 1
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태를 확인합니다."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # ngrok 설정
    ngrok_tunnel = ngrok.connect(5000)
    print(f' * ngrok tunnel "{ngrok_tunnel.public_url}" -> http://127.0.0.1:5000')
    
    # 서버 실행
    app.run(host='0.0.0.0', port=5000) 