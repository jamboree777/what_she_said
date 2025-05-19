document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const speakerContent = document.getElementById('speakerContent');
    const emotionContent = document.getElementById('emotionContent');

    // 파일 선택 시 이벤트 처리
    fileInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            // 파일 크기 체크 (16MB)
            if (file.size > 16 * 1024 * 1024) {
                alert('파일 크기는 16MB를 초과할 수 없습니다.');
                fileInput.value = '';
                return;
            }
            
            // 파일 확장자 체크
            const fileExtension = file.name.split('.').pop().toLowerCase();
            const allowedExtensions = ['wav', 'mp3', 'ogg', 'm4a'];
            
            // 파일이 오디오 타입인지 확인
            if (!file.type.startsWith('audio/') && !allowedExtensions.includes(fileExtension)) {
                alert('지원되지 않는 파일 형식입니다. (지원 형식: WAV, MP3, OGG, M4A)');
                fileInput.value = '';
                return;
            }

            console.log('파일 정보:', {
                name: file.name,
                type: file.type,
                extension: fileExtension
            });
        }
    });

    analyzeBtn.addEventListener('click', async () => {
        const file = fileInput.files[0];
        if (!file) {
            alert('파일을 선택해주세요.');
            return;
        }

        try {
            // 로딩 상태 표시
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = '분석 중...';
            speakerContent.innerHTML = '<p>분석 중입니다. 잠시만 기다려주세요...</p>';
            emotionContent.innerHTML = '';

            // 파일 업로드
            const formData = new FormData();
            formData.append('file', file);

            const uploadResponse = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                const errorData = await uploadResponse.json();
                throw new Error(errorData.error || '파일 업로드 실패');
            }

            const uploadResult = await uploadResponse.json();
            
            // 분석 요청 (GET 메서드로 변경)
            const analyzeResponse = await fetch(`/analyze/${uploadResult.filename}`, {
                method: 'GET'
            });

            if (!analyzeResponse.ok) {
                const errorData = await analyzeResponse.json();
                throw new Error(errorData.error || '분석 실패');
            }

            const analysisResult = await analyzeResponse.json();
            displayResults(analysisResult);
        } catch (error) {
            console.error('처리 중 오류 발생:', error);
            speakerContent.innerHTML = `<p class="error">오류 발생: ${error.message}</p>`;
            emotionContent.innerHTML = '';
        } finally {
            // 버튼 상태 복원
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '분석하기';
        }
    });

    function displayResults(results) {
        // 화자 분석 결과 표시
        const speakerHtml = results.map(result => `
            <div class="speaker-segment">
                <h4>화자 ${result.speaker}</h4>
                <p>시간: ${formatTime(result.start)} - ${formatTime(result.end)}</p>
                <p>감정: ${result.emotion} (신뢰도: ${(result.confidence * 100).toFixed(1)}%)</p>
            </div>
        `).join('');

        speakerContent.innerHTML = speakerHtml;

        // 감정 분석 요약
        const emotions = {};
        results.forEach(result => {
            emotions[result.emotion] = (emotions[result.emotion] || 0) + 1;
        });

        const emotionHtml = `
            <h4>감정 분포</h4>
            <div class="emotion-summary">
                ${Object.entries(emotions).map(([emotion, count]) => `
                    <div class="emotion-item">
                        <span class="emotion-label">${emotion}</span>
                        <span class="emotion-count">${count}회</span>
                    </div>
                `).join('')}
            </div>
        `;

        emotionContent.innerHTML = emotionHtml;
    }

    function formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
}); 