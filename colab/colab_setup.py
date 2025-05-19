!pip install pyannote.audio
!pip install whisperx
!pip install flask
!pip install flask-cors
!pip install pyngrok
!pip install torch torchaudio
!pip install transformers
!pip install librosa
!pip install soundfile
!pip install numpy
!pip install pandas
!pip install matplotlib
!pip install scikit-learn
!pip install pydub
!pip install webrtcvad
!pip install speechbrain

# 한국어 처리 관련 라이브러리
!pip install kobert
!pip install kogpt
!pip install kiwipiepy
!pip install soynlp
!pip install konlpy
!pip install mecab-python
!pip install sentencepiece
!pip install fasttext

# Hugging Face 토큰 설정
import os
os.environ["HF_TOKEN"] = "your_hugging_face_token"  # 여기에 본인의 Hugging Face 토큰을 입력하세요

# GPU 사용 가능 여부 확인
import torch
print(f"GPU 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}") 