const frame = document.querySelector('#frame');
const photo = document.querySelector('#photo');
const camera = document.querySelector('#camera');
const cameraButton = document.querySelector('#cameraButton');
const captureButton = document.querySelector('#capture');
const upload = document.querySelector('#upload');
const status = document.querySelector('#status');
const toast = document.querySelector('#toast');
const gestureSound = document.querySelector('#gestureSound');
const peaceSound = document.querySelector('#peaceSound');
const voiceCaption = document.querySelector('#voiceCaption');

let stream = null;
let landmarker = null;
let raf = 0;
let lastDetection = 0;
let blurPose = false;
let activeGesture = null;
let candidateGesture = null;
let candidateFrames = 0;
let noGestureFrames = 0;
let timer;

const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

function say(message) {
  toast.textContent = message;
  toast.classList.add('show');
  clearTimeout(timer);
  timer = setTimeout(() => toast.classList.remove('show'), 2400);
}

function applyPose(blur) {
  blurPose = blur;
  frame.classList.toggle('revealed', !blur);
}

function isTwoFingerPose(landmarks) {
  const indexUp = landmarks[8].y < landmarks[6].y - 0.015;
  const middleUp = landmarks[12].y < landmarks[10].y - 0.015;
  const ringDown = landmarks[16].y > landmarks[14].y - 0.005;
  const pinkyDown = landmarks[20].y > landmarks[18].y - 0.005;
  return indexUp && middleUp && ringDown && pinkyDown;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function isFistPose(landmarks) {
  return [[8, 6], [12, 10], [16, 14], [20, 18]].every(([tip, pip]) =>
    distance(landmarks[0], landmarks[tip]) < distance(landmarks[0], landmarks[pip]) * 1.18
  );
}

function showVoiceCaption(message) {
  voiceCaption.textContent = message;
  voiceCaption.hidden = false;
  requestAnimationFrame(() => voiceCaption.classList.add('show'));
}

function hideVoiceCaption() {
  voiceCaption.classList.remove('show');
  voiceCaption.hidden = true;
}

function stopGestureSound() {
  [gestureSound, peaceSound].forEach((sound) => {
    sound.pause();
    sound.currentTime = 0;
  });
  hideVoiceCaption();
}

function playGestureSound(audio, message) {
  stopGestureSound();
  showVoiceCaption(message);
  audio.play().catch(() => {
    if (activeGesture) say('Browser menahan suara. Klik halaman, lalu coba lagi.');
  });
}

function activateGesture(gesture) {
  activeGesture = gesture;
  if (gesture === 'peace') {
    applyPose(true);
    status.textContent = 'Pose ✌️ terdeteksi — blur dan suara aktif.';
    playGestureSound(peaceSound, 'FOTO KITA BLUR');
  } else {
    applyPose(false);
    status.textContent = 'Kepalan tangan terdeteksi — suara aktif.';
    playGestureSound(gestureSound, 'HIDUP JOKOWI!');
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    say('Kamera perlu halaman HTTPS atau localhost.');
    return;
  }

  cameraButton.disabled = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 720 }, height: { ideal: 900 } },
      audio: false,
    });
    camera.srcObject = stream;
    camera.hidden = false;
    await camera.play();
    frame.classList.add('camera-live');
    cameraButton.textContent = 'Matikan kamera';
    captureButton.disabled = false;
    status.textContent = 'Kamera aktif — angkat ✌️ atau kepalkan tangan.';
    loadHandDetector();
  } catch (error) {
    stopCamera();
    console.error(error);
    say(error.name === 'NotAllowedError'
      ? 'Izin kamera ditolak. Izinkan kamera di browser.'
      : 'Kamera tidak tersedia. Izinkan akses kamera di browser.');
  } finally {
    cameraButton.disabled = false;
  }
}

async function loadHandDetector() {
  try {
    status.textContent = 'Kamera aktif — memuat deteksi pose tangan…';
    const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
    if (!stream) return;
    const fileset = await vision.FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
    const detector = await vision.HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'VIDEO',
      numHands: 1,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
    });
    if (!stream) {
      detector.close();
      return;
    }
    landmarker = detector;
    status.textContent = 'Kamera aktif — angkat ✌️ atau kepalkan tangan.';
    detectLoop();
  } catch (error) {
    console.error(error);
    if (stream) status.textContent = 'Kamera aktif, tapi deteksi pose gagal dimuat. Periksa koneksi internet.';
  }
}

function detectLoop(now = 0) {
  if (!stream || !landmarker) return;
  if (camera.readyState >= 2 && now - lastDetection > 90) {
    lastDetection = now;
    try {
      const result = landmarker.detectForVideo(camera, now);
      const hands = result.landmarks || [];
      const fistFound = hands.some(isFistPose);
      const peaceFound = hands.some(isTwoFingerPose);
      const detectedGesture = fistFound ? 'fist' : peaceFound ? 'peace' : null;

      if (detectedGesture) {
        noGestureFrames = 0;
        if (candidateGesture === detectedGesture) candidateFrames++;
        else {
          candidateGesture = detectedGesture;
          candidateFrames = 1;
        }
        if (candidateFrames >= 2 && activeGesture !== detectedGesture) {
          activateGesture(detectedGesture);
        }
      } else {
        candidateGesture = null;
        candidateFrames = 0;
        noGestureFrames++;
        if (noGestureFrames >= 2 && activeGesture) {
          activeGesture = null;
          stopGestureSound();
          applyPose(false);
          status.textContent = 'Kamera aktif — angkat ✌️ atau kepalkan tangan.';
        }
      }
    } catch (error) {
      console.error(error);
    }
  }
  raf = requestAnimationFrame(detectLoop);
}

function stopCamera() {
  cancelAnimationFrame(raf);
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  camera.srcObject = null;
  camera.hidden = true;
  frame.classList.remove('camera-live');
  cameraButton.textContent = 'Pakai kamera 📷';
  captureButton.disabled = true;
  activeGesture = null;
  candidateGesture = null;
  candidateFrames = 0;
  noGestureFrames = 0;
  stopGestureSound();
  applyPose(false);
  if (landmarker) {
    landmarker.close();
    landmarker = null;
  }
}

cameraButton.addEventListener('click', () => stream ? stopCamera() : startCamera());

captureButton.addEventListener('click', () => {
  if (!stream) return;
  const canvas = document.createElement('canvas');
  canvas.width = camera.videoWidth;
  canvas.height = camera.videoHeight;
  const context = canvas.getContext('2d');
  context.translate(canvas.width, 0);
  context.scale(-1, 1);
  context.drawImage(camera, 0, 0);
  if (blurPose) {
    const temp = document.createElement('canvas');
    temp.width = canvas.width;
    temp.height = canvas.height;
    const tempContext = temp.getContext('2d');
    tempContext.filter = 'blur(24px) brightness(.8)';
    tempContext.drawImage(canvas, 0, 0);
    context.filter = 'none';
    context.drawImage(temp, 0, 0);
  }
  const link = document.createElement('a');
  link.download = 'foto-kita-blur.jpg';
  link.href = canvas.toDataURL('image/jpeg', 0.92);
  link.click();
  say('Foto berhasil disimpan ✨');
});

upload.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  stopCamera();
  photo.src = URL.createObjectURL(file);
  photo.style.display = '';
  frame.classList.remove('revealed');
  status.textContent = 'Foto pilihanmu siap — efek blur bisa dibuka dengan sentuhan.';
});

frame.addEventListener('click', () => {
  if (stream) return;
  frame.classList.toggle('revealed');
});

document.querySelector('#share').addEventListener('click', async () => {
  try {
    if (navigator.share) {
      await navigator.share({ title: 'Foto kita blur', text: 'Yang blur fotonya, perasaannya jangan.', url: location.href });
    } else {
      await navigator.clipboard.writeText(location.href);
      say('Link sudah disalin ✨');
    }
  } catch (error) {
    if (error.name !== 'AbortError') say('Belum bisa membagikan dari browser ini');
  }
});

window.addEventListener('pagehide', stopCamera);
