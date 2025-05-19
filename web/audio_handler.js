class AudioHandler {
  constructor() {
    this.mediaRecorder = null;
    this.audioChunks = [];
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.audioElement = new Audio();
  }

  async startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        this.audioChunks.push(event.data);
      };

      this.mediaRecorder.start();
      return true;
    } catch (error) {
      console.error('녹음 시작 실패:', error);
      return false;
    }
  }

  stopRecording() {
    return new Promise((resolve) => {
      this.mediaRecorder.onstop = () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        resolve(audioUrl);
      };
      this.mediaRecorder.stop();
    });
  }

  playAudio(audioUrl) {
    this.audioElement.src = audioUrl;
    return this.audioElement.play();
  }

  pauseAudio() {
    this.audioElement.pause();
  }

  stopAudio() {
    this.audioElement.pause();
    this.audioElement.currentTime = 0;
  }

  seekTo(position) {
    this.audioElement.currentTime = position;
  }

  getCurrentTime() {
    return this.audioElement.currentTime;
  }

  getDuration() {
    return this.audioElement.duration;
  }

  onTimeUpdate(callback) {
    this.audioElement.ontimeupdate = callback;
  }

  onEnded(callback) {
    this.audioElement.onended = callback;
  }
}

// 전역 객체로 내보내기
window.audioHandler = new AudioHandler(); 