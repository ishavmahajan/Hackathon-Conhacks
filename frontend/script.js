// ==============================================
// DOM ELEMENTS
// ==============================================
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const cameraBtn   = document.getElementById('camera-btn');
const speakBtn    = document.getElementById('speak-btn');
const clearBtn    = document.getElementById('clear-btn');
const translationEl = document.getElementById('translation');
const confidenceEl  = document.getElementById('confidence');
const statusDot     = document.getElementById('status-dot');
const statusText    = document.getElementById('status-text');
const videoBox      = document.getElementById('video-box');
const overlay       = document.getElementById('camera-off-overlay');
const guide         = document.getElementById('guide');
const spotlight     = document.getElementById('spotlight');
const outputCard    = document.getElementById('output-card');
const historyEl     = document.getElementById('output-history');

// ==============================================
// STATE
// ==============================================
let stream            = null;
let isCameraOn        = false;
let currentTranslation = '';
let animationId       = null;
let predictionInterval = null;
let detectionHistory  = [];

const BACKEND_URL    = 'http://127.0.0.1:5000';
const MAX_HISTORY    = 8;
const PREDICT_INTERVAL_MS = 1000;

// ==============================================
// MOUSE-TRACKING SPOTLIGHT
// ==============================================
videoBox.addEventListener('mousemove', (e) => {
    const rect = videoBox.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    spotlight.style.left = x + 'px';
    spotlight.style.top  = y + 'px';
});

videoBox.addEventListener('mouseleave', () => {
    spotlight.style.opacity = '0';
});

videoBox.addEventListener('mouseenter', () => {
    if (isCameraOn) spotlight.style.opacity = '1';
});

// ==============================================
// CAMERA TOGGLE
// ==============================================
cameraBtn.addEventListener('click', () => {
    isCameraOn ? stopCamera() : startCamera();
});

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
        });

        video.srcObject = stream;
        video.classList.add('active');
        overlay.classList.add('hidden');
        videoBox.classList.add('active');
        spotlight.style.opacity = '1';

        isCameraOn = true;

        // Update button
        cameraBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                <rect x="6" y="4" width="4" height="16" rx="1"/>
                <rect x="14" y="4" width="4" height="16" rx="1"/>
            </svg>
            Stop Camera
        `;
        cameraBtn.classList.add('stop');

        updateStatus('active', 'Camera on — show your sign');
        setGuide('Hold your hand steady in the frame');

        video.addEventListener('loadeddata', startDetection, { once: true });

    } catch (err) {
        updateStatus('error', 'Camera access denied');
        setGuide('Allow camera access and try again');
        console.error('[SignBridge] Camera error:', err);
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(t => t.stop());
    }

    video.srcObject = null;
    video.classList.remove('active');
    overlay.classList.remove('hidden');
    videoBox.classList.remove('active');
    spotlight.style.opacity = '0';

    isCameraOn = false;

    // Reset button
    cameraBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Start Camera
    `;
    cameraBtn.classList.remove('stop');

    cancelAnimationFrame(animationId);
    clearInterval(predictionInterval);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    updateStatus('off', 'Camera off — Ready to translate');
    resetTranslation();
    setGuide('Position your hand clearly in the frame');

    speakBtn.disabled = true;
    clearBtn.disabled  = true;
}

// ==============================================
// STATUS UPDATES
// ==============================================
function updateStatus(state, message) {
    statusDot.className = 'status-dot';
    if (state === 'active')    statusDot.classList.add('active');
    if (state === 'detecting') statusDot.classList.add('detecting');
    if (state === 'error')     statusDot.classList.add('error');
    statusText.textContent = message;
}

function setGuide(msg) {
    guide.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle;margin-right:5px;opacity:0.45;">
            <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v4"/>
            <path d="M18 11a2 2 0 0 1 4 0v3a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15"/>
        </svg>
        ${msg}
    `;
}

// ==============================================
// SEND FRAME TO BACKEND
// ==============================================
async function sendFrameToBackend() {
    if (!isCameraOn) return;

    try {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width  = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        // Draw mirrored frame to match what user sees
        tempCtx.translate(tempCanvas.width, 0);
        tempCtx.scale(-1, 1);
        tempCtx.drawImage(video, 0, 0);

        const blob = await new Promise(res => tempCanvas.toBlob(res, 'image/jpeg', 0.85));
        const formData = new FormData();
        formData.append('frame', blob, 'frame.jpg');

        const response = await fetch(`${BACKEND_URL}/predict_frame`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Bad response');

        const data = await response.json();

        if (data.gesture && data.gesture !== 'NO_HAND' && data.gesture !== 'NO_GESTURE') {
            setTranslation(data.gesture, data.confidence);
        } else {
            updateStatus('active', 'Camera on — no gesture detected');
        }

    } catch (err) {
        // Silent fail — demo mode
        updateStatus('active', 'Camera on — awaiting detection');
    }
}

// ==============================================
// DETECTION LOOP
// ==============================================
function startDetection() {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;

    function drawFrame() {
        if (!isCameraOn) return;
        // Mirror the canvas draw
        ctx.save();
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
        animationId = requestAnimationFrame(drawFrame);
    }

    drawFrame();
    predictionInterval = setInterval(sendFrameToBackend, PREDICT_INTERVAL_MS);
}

// ==============================================
// SET TRANSLATION
// ==============================================
function setTranslation(text, confidence = null) {
    if (!text || text === 'NO_GESTURE' || text === 'NO_HAND') {
        translationEl.textContent = '—';
        translationEl.className = 'output-text no-gesture';
        confidenceEl.textContent = '';
        outputCard.classList.remove('has-result');
        updateStatus('active', 'Camera on — no gesture detected');
        return;
    }

    currentTranslation = text;

    // Animate text swap
    translationEl.style.opacity = '0';
    translationEl.style.transform = 'translateY(6px)';
    setTimeout(() => {
        translationEl.textContent = text;
        translationEl.className = 'output-text detected';
        translationEl.style.opacity = '1';
        translationEl.style.transform = 'translateY(0)';
        translationEl.style.transition = 'opacity 0.2s var(--ease-expo), transform 0.2s var(--ease-expo)';
    }, 120);

    confidenceEl.textContent = confidence
        ? `${Math.round(confidence * 100)}% confidence`
        : '';

    outputCard.classList.add('has-result');
    updateStatus('detecting', `Detected: ${text}`);

    speakBtn.disabled = false;
    clearBtn.disabled  = false;

    // Add to history
    addToHistory(text);
}

// ==============================================
// HISTORY CHIPS
// ==============================================
function addToHistory(text) {
    // Avoid duplicates at the front
    if (detectionHistory[0] === text) return;

    detectionHistory.unshift(text);
    if (detectionHistory.length > MAX_HISTORY) {
        detectionHistory.pop();
    }
    renderHistory();
}

function renderHistory() {
    historyEl.innerHTML = '';
    detectionHistory.forEach((item, i) => {
        const chip = document.createElement('span');
        chip.className = 'history-chip';
        chip.textContent = item;
        chip.style.opacity = i === 0 ? '1' : String(Math.max(0.25, 1 - i * 0.12));
        chip.setAttribute('title', `Previous detection: ${item}`);
        historyEl.appendChild(chip);
    });
}

// ==============================================
// RESET
// ==============================================
function resetTranslation() {
    currentTranslation = '';
    translationEl.textContent = '—';
    translationEl.className = 'output-text';
    translationEl.style.opacity = '';
    translationEl.style.transform = '';
    translationEl.style.transition = '';
    confidenceEl.textContent = '';
    outputCard.classList.remove('has-result');
    detectionHistory = [];
    historyEl.innerHTML = '';
}

// ==============================================
// SPEAK
// ==============================================
speakBtn.addEventListener('click', () => {
    if (!currentTranslation || !('speechSynthesis' in window)) return;

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(currentTranslation);
    utterance.rate  = 0.95;
    utterance.pitch = 1;
    utterance.lang  = 'en-US';

    speakBtn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
            <line x1="23" y1="9" x2="17" y2="15"/>
            <line x1="17" y1="9" x2="23" y2="15"/>
        </svg>
        Speaking…
    `;
    speakBtn.disabled = true;

    utterance.onend = () => {
        speakBtn.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
            </svg>
            Speak
        `;
        speakBtn.disabled = !currentTranslation;
    };

    window.speechSynthesis.speak(utterance);
});

// ==============================================
// CLEAR
// ==============================================
clearBtn.addEventListener('click', () => {
    resetTranslation();
    speakBtn.disabled = true;
    clearBtn.disabled  = true;
    updateStatus('active', 'Camera on — show your sign');
    setGuide('Hold your hand steady in the frame');
});

// ==============================================
// BACKEND HEALTH CHECK
// ==============================================
async function checkBackend() {
    try {
        const res = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
            console.info('[SignBridge] Backend connected at', BACKEND_URL);
        }
    } catch {
        console.info('[SignBridge] Backend not reachable — running in demo mode.');
    }
}

// ==============================================
// KEYBOARD SHORTCUT: Space = toggle camera
// ==============================================
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        cameraBtn.click();
    }
    if (e.code === 'KeyS' && currentTranslation) {
        speakBtn.click();
    }
    if (e.code === 'KeyC' && currentTranslation) {
        clearBtn.click();
    }
});

// ==============================================
// INIT
// ==============================================
updateStatus('off', 'Camera off — Ready to translate');
translationEl.textContent = '—';
checkBackend();