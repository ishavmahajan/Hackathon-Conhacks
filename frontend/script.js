// ==============================================
// DOM ELEMENTS
// ==============================================
const video         = document.getElementById('video');
const canvas        = document.getElementById('canvas');
const ctx           = canvas.getContext('2d');
const cameraBtn     = document.getElementById('camera-btn');
const speakBtn      = document.getElementById('speak-btn');
const clearBtn      = document.getElementById('clear-btn');
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
let stream              = null;
let isCameraOn          = false;
let currentTranslation  = '';
let animationId         = null;
let predictionInterval  = null;
let detectionHistory    = [];
let isSpeaking          = false;

const BACKEND_URL         = 'http://127.0.0.1:5000';
const MAX_HISTORY         = 8;
const PREDICT_INTERVAL_MS = 900;

// ==============================================
// MOUSE-TRACKING SPOTLIGHT
// ==============================================
videoBox.addEventListener('mousemove', (e) => {
    const rect = videoBox.getBoundingClientRect();
    spotlight.style.left = (e.clientX - rect.left) + 'px';
    spotlight.style.top  = (e.clientY - rect.top)  + 'px';
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
    if (stream) stream.getTracks().forEach(t => t.stop());

    video.srcObject = null;
    video.classList.remove('active');
    overlay.classList.remove('hidden');
    videoBox.classList.remove('active');
    spotlight.style.opacity = '0';

    isCameraOn = false;

    cameraBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
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
    clearBtn.disabled = true;
}

// ==============================================
// STATUS & GUIDE HELPERS
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
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
             style="vertical-align:middle;margin-right:5px;opacity:0.45;">
            <path d="M18 11V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0
                     M14 10V4a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v0
                     M10 10.5V6a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v4"/>
            <path d="M18 11a2 2 0 0 1 4 0v3a8 8 0 0 1-8 8h-2
                     c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6
                     a2 2 0 0 1 2.83-2.82L7 15"/>
        </svg>
        ${msg}
    `;
}

// ==============================================
// DETECTION LOOP
// The display <video> uses CSS scaleX(-1) for a natural selfie mirror.
// The canvas draws raw un-mirrored pixels — these are what we ship to
// the backend. MediaPipe was trained on un-mirrored webcam frames, so
// sending un-flipped data is critical for correct landmark orientation.
// ==============================================
function startDetection() {
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;

    function drawFrame() {
        if (!isCameraOn) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        animationId = requestAnimationFrame(drawFrame);
    }

    drawFrame();
    predictionInterval = setInterval(sendFrameToBackend, PREDICT_INTERVAL_MS);
}

// ==============================================
// SEND FRAME TO BACKEND
// _busy flag prevents overlapping requests, which cause stale
// predictions to overwrite fresh ones and tank confidence scores.
// ==============================================
sendFrameToBackend._busy = false;

async function sendFrameToBackend() {
    if (!isCameraOn || sendFrameToBackend._busy) return;
    sendFrameToBackend._busy = true;

    try {
        // Read directly from the raw <video> element.
        // DO NOT apply ctx.scale(-1,1) — flipping swaps left/right
        // hand landmark coordinates and causes misclassifications
        // (e.g. showing "8" instead of "1"). JPEG 0.95 preserves the
        // fine finger-edge detail that MediaPipe uses for confidence.
        const w   = video.videoWidth;
        const h   = video.videoHeight;
        const tmp = document.createElement('canvas');
        tmp.width  = w;
        tmp.height = h;
        tmp.getContext('2d').drawImage(video, 0, 0, w, h);

        const blob = await new Promise(res => tmp.toBlob(res, 'image/jpeg', 0.95));
        const fd   = new FormData();
        fd.append('frame', blob, 'frame.jpg');

        const response = await fetch(`${BACKEND_URL}/predict_frame`, {
            method: 'POST',
            body: fd,
            signal: AbortSignal.timeout(3000)
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);

        const data = await response.json();

        if (data.gesture && data.gesture !== 'NO_HAND' && data.gesture !== 'NO_GESTURE') {
            setTranslation(data.gesture, data.confidence);
        } else {
            updateStatus('active', 'Camera on — no gesture detected');
        }

    } catch (err) {
        if (err.name !== 'AbortError') {
            updateStatus('active', 'Camera on — awaiting detection');
        }
    } finally {
        sendFrameToBackend._busy = false;
    }
}

// ==============================================
// SET TRANSLATION
// ==============================================
function setTranslation(text, confidence = null) {
    if (!text || text === 'NO_GESTURE' || text === 'NO_HAND') {
        translationEl.textContent      = '—';
        translationEl.className        = 'output-text no-gesture';
        translationEl.style.opacity    = '';
        translationEl.style.transform  = '';
        translationEl.style.transition = '';
        confidenceEl.textContent       = '';
        outputCard.classList.remove('has-result');
        updateStatus('active', 'Camera on — no gesture detected');
        return;
    }

    currentTranslation = text;

    // Smooth swap animation
    translationEl.style.transition = '';
    translationEl.style.opacity    = '0';
    translationEl.style.transform  = 'translateY(8px)';

    setTimeout(() => {
        translationEl.textContent      = text;
        translationEl.className        = 'output-text detected';
        translationEl.style.transition = 'opacity 0.2s var(--ease-expo), transform 0.2s var(--ease-expo)';
        translationEl.style.opacity    = '1';
        translationEl.style.transform  = 'translateY(0)';
    }, 110);

    confidenceEl.textContent = confidence
        ? `${Math.round(confidence * 100)}% confidence`
        : '';

    outputCard.classList.add('has-result');
    updateStatus('detecting', `Detected: ${text}`);

    speakBtn.disabled = false;
    clearBtn.disabled = false;

    addToHistory(text);
}

// ==============================================
// HISTORY CHIPS
// ==============================================
function addToHistory(text) {
    if (detectionHistory[0] === text) return;
    detectionHistory.unshift(text);
    if (detectionHistory.length > MAX_HISTORY) detectionHistory.pop();
    renderHistory();
}

function renderHistory() {
    historyEl.innerHTML = '';
    detectionHistory.forEach((item, i) => {
        const chip       = document.createElement('span');
        chip.className   = 'history-chip';
        chip.textContent = item;
        chip.style.opacity = i === 0 ? '1' : String(Math.max(0.25, 1 - i * 0.12));
        chip.title       = `Previous detection: ${item}`;
        historyEl.appendChild(chip);
    });
}

// ==============================================
// RESET
// ==============================================
function resetTranslation() {
    currentTranslation             = '';
    translationEl.textContent      = '—';
    translationEl.className        = 'output-text';
    translationEl.style.opacity    = '';
    translationEl.style.transform  = '';
    translationEl.style.transition = '';
    confidenceEl.textContent       = '';
    outputCard.classList.remove('has-result');
    detectionHistory               = [];
    historyEl.innerHTML            = '';
}

// ==============================================
// SPEAK
// ==============================================
speakBtn.addEventListener('click', () => {
    if (!currentTranslation || !('speechSynthesis' in window) || isSpeaking) return;

    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(currentTranslation);
    utterance.rate  = 0.95;
    utterance.pitch = 1;
    utterance.lang  = 'en-US';

    isSpeaking        = true;
    speakBtn.disabled = true;
    speakBtn.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
            <line x1="23" y1="9" x2="17" y2="15"/>
            <line x1="17" y1="9" x2="23" y2="15"/>
        </svg>
        Speaking…
    `;

    const resetSpeakBtn = () => {
        isSpeaking        = false;
        speakBtn.disabled = !currentTranslation;
        speakBtn.innerHTML = `
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
            </svg>
            Speak
        `;
    };

    utterance.onend   = resetSpeakBtn;
    utterance.onerror = resetSpeakBtn;

    window.speechSynthesis.speak(utterance);
});

// ==============================================
// CLEAR
// ==============================================
clearBtn.addEventListener('click', () => {
    resetTranslation();
    speakBtn.disabled = true;
    clearBtn.disabled = true;
    if (isCameraOn) {
        updateStatus('active', 'Camera on — show your sign');
        setGuide('Hold your hand steady in the frame');
    }
});

// ==============================================
// KEYBOARD SHORTCUTS
// Space  → toggle camera
// S      → speak
// C      → clear
// Escape → stop camera
// ==============================================
document.addEventListener('keydown', (e) => {
    if (e.target !== document.body) return;
    if (e.code === 'Space') {
        e.preventDefault();
        cameraBtn.click();
    }
    if (e.code === 'KeyS' && currentTranslation && !isSpeaking) speakBtn.click();
    if (e.code === 'KeyC' && currentTranslation) clearBtn.click();
    if (e.code === 'Escape' && isCameraOn) stopCamera();
});

// ==============================================
// BACKEND HEALTH CHECK
// ==============================================
async function checkBackend() {
    try {
        const res = await fetch(`${BACKEND_URL}/health`, {
            signal: AbortSignal.timeout(3000)
        });
        if (res.ok) {
            const data = await res.json();
            console.info(`[SignBridge] Backend connected. Model loaded: ${data.model_loaded}`);
            statusText.textContent = data.model_loaded
                ? 'Ready — model loaded ✓'
                : 'Ready — no model found';
        }
    } catch {
        console.info('[SignBridge] Backend offline — demo mode.');
    }
}

// ==============================================
// INIT
// ==============================================
updateStatus('off', 'Camera off — Ready to translate');
translationEl.textContent = '—';
checkBackend();