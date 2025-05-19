from flask import Flask, request, jsonify, send_from_directory, render_template, send_file
from werkzeug.utils import secure_filename
import os
from audio_analyzer import analyze_audio, AudioAnalyzer
import webbrowser
import logging
from flask_cors import CORS
from datetime import datetime
from speech_analyzer import SpeechAnalyzer

app = Flask(__name__)
CORS(app)

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 업로드 폴더 설정
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"업로드 폴더 생성: {UPLOAD_FOLDER}")

# 분석 결과를 저장할 디렉토리
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 제한

# 허용된 파일 확장자
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

analyzer = AudioAnalyzer()

def allowed_file(filename):
    """허용된 파일 형식인지 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def ensure_upload_folder():
    """업로드 폴더가 존재하는지 확인하고 생성"""
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            logger.info(f"업로드 폴더 생성됨: {UPLOAD_FOLDER}")
    except Exception as e:
        logger.error(f"업로드 폴더 생성 중 오류 발생: {str(e)}")
        raise

def cleanup_uploads():
    """uploads 폴더의 모든 파일을 삭제"""
    try:
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"파일 삭제 중 오류 발생: {str(e)}")
    except Exception as e:
        logger.error(f"폴더 정리 중 오류 발생: {str(e)}")

def open_browser():
    """기존 브라우저 창을 재사용하여 URL 열기"""
    url = 'http://localhost:5000'
    try:
        # 기본 브라우저의 컨트롤러 가져오기
        browser = webbrowser.get()
        # 기존 창에서 URL 열기
        browser.open(url, new=0, autoraise=True)
    except Exception as e:
        logger.error(f"브라우저 열기 실패: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        # 파일 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # 오디오 분석 수행
        results = analyzer.analyze_audio(filepath)
        if results:
            # 결과 저장
            result_filename = f"{timestamp}_analysis.txt"
            result_path = os.path.join(RESULTS_FOLDER, result_filename)
            analyzer.save_results(results, result_path)
            
            return jsonify({
                'message': 'File uploaded and analyzed successfully',
                'results': results,
                'result_file': result_filename
            })
        else:
            return jsonify({'error': 'Error analyzing audio'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(
        os.path.join(RESULTS_FOLDER, filename),
        as_attachment=True,
        download_name=filename
    )

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/analyze/<filename>', methods=['GET'])
def analyze(filename):
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            logger.error(f"파일을 찾을 수 없습니다: {filepath}")
            return jsonify({'error': '파일을 찾을 수 없습니다'}), 404
            
        # 파일 분석
        results = analyze_audio(filepath)
        
        # 분석 완료 후 파일 삭제
        try:
            os.remove(filepath)
            logger.info(f"분석 완료 후 파일 삭제: {filepath}")
        except Exception as e:
            logger.error(f"파일 삭제 중 오류 발생: {str(e)}")
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    try:
        analyzer.start_recording()
        return jsonify({"status": "success", "message": "녹음이 시작되었습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    try:
        filename = analyzer.stop_recording()
        if filename:
            return jsonify({
                "status": "success",
                "message": "녹음이 저장되었습니다.",
                "filename": filename
            })
        return jsonify({"status": "error", "message": "녹음 파일이 생성되지 않았습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    try:
        audio_file = request.json.get('filename')
        if not audio_file:
            return jsonify({"status": "error", "message": "오디오 파일이 지정되지 않았습니다."})
        
        results = analyzer.analyze_speakers(audio_file)
        return jsonify({
            "status": "success",
            "results": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/add_speaker', methods=['POST'])
def add_speaker():
    try:
        name = request.json.get('name')
        if not name:
            return jsonify({"status": "error", "message": "화자 이름이 지정되지 않았습니다."})
        
        speaker_id = analyzer.add_speaker(name)
        return jsonify({
            "status": "success",
            "speaker_id": speaker_id,
            "speakers": analyzer.speakers
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/save_transcript', methods=['POST'])
def save_transcript():
    try:
        results = request.json.get('results')
        if not results:
            return jsonify({"status": "error", "message": "분석 결과가 없습니다."})
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"transcript_{timestamp}.json"
        analyzer.save_transcript(results, output_file)
        
        return jsonify({
            "status": "success",
            "message": "녹취록이 저장되었습니다.",
            "filename": output_file
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    ensure_upload_folder()
    open_browser()  # 브라우저를 한 번만 열기
    app.run(host='localhost', port=5000, debug=False)  # 디버그 모드 비활성화 