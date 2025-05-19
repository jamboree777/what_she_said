# 화자 분리 프로그램 (Speaker Separation Program)

이 프로그램은 오디오 파일에서 여러 화자의 음성을 분리하고 분석하는 도구입니다.

## 주요 기능

- WAV 파일에서 화자 분리
- 드래그 앤 드롭 인터페이스
- 실시간 처리 상태 표시
- 신뢰도 점수 계산
- 묵음 구간 감지
- 상세한 처리 결과 리포트

## 설치 방법

1. Python 3.9 이상 설치
2. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 사용 방법

1. 프로그램 실행:
```bash
python separate_speakers.py
```

2. WAV 파일을 프로그램 창에 드래그 앤 드롭
3. 처리 결과는 입력 파일과 동일한 디렉토리의 `{파일명}_speaker_separation` 폴더에 저장됩니다.

## 결과 파일

- `result.txt`: 상세한 처리 결과
- `result.json`: JSON 형식의 처리 결과

## 시스템 요구사항

- Python 3.9 이상
- Windows 10 이상
- 최소 4GB RAM
- NVIDIA GPU 권장 (CUDA 지원)

## 라이선스

MIT License

## 의존성 패키지

- pyannote.audio 3.3.2
- torch 2.7.0
- torchaudio 2.7.0
- numpy 2.2.6
- speechbrain 1.0.3
- tkinterdnd2 