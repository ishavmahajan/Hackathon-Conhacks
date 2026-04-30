console.log('script.js loaded');

const API_URL = '/api/predict';

const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const statusText = document.getElementById('status-text');
const statusDot = document.getElementById('status-dot');
const cameraOffOverlay = document.getElementById('camera-off-overlay');
const spotlight = document.getElementById('spotlight');
const scanLine = document.getElementById('scan-line');
const cameraBtn = document.getElementById('camera-btn');
const speakBtn = document.getElementById('speak-btn');
const clearBtn = document.getElementById('clear-btn');
const translationEl = document.getElementById('translation');
const confidenceEl = document.getElementById('confidence');
const outputHistory = document.getElementById('output-history');

let lastTranslation = '';
let isProcessing = false;
let stream = null;
let animationId = null;
let predictTimer = null;
let latestLandmarks = null; // { pose, leftHand, rightHand }

// ─── MediaPipe Setup ───
const holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
});
holistic.setOptions({
    modelComplexity: 0,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

// When MediaPipe finishes processing a frame, store the landmark data
holistic.onResults((results) => {
    latestLandmarks = {
        pose: results.poseLandmarks || null,
        leftHand: results.leftHandLandmarks || null,
        rightHand: results.rightHandLandmarks || null
    };
});

// ─── UI Helpers ───
function setStatus(text, state = 'idle') {
    statusText.textContent = text;
    statusDot.className = `status-dot ${state}`;
}

function toggleControls(enabled) {
    speakBtn.disabled = !enabled;
    clearBtn.disabled = !enabled;
}

function resetOutput() {
    translationEl.textContent = '—';
    confidenceEl.textContent = '';
    outputHistory.innerHTML = '';
    lastTranslation = '';
    toggleControls(false);
}

function addHistoryItem(label, translation, confidence) {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
        <strong>${label || translation || 'Detected'}</strong>
        <span>${translation}</span>
        <small>${confidence ? `Confidence: ${(confidence * 100).toFixed(0)}%` : ''}</small>
    `;
    outputHistory.prepend(item);
    if (outputHistory.children.length > 6) {
        outputHistory.removeChild(outputHistory.lastChild);
    }
}

function updateOutput(result) {
    const text = result.translation || result.label || 'No sign detected';
    const confidence = typeof result.confidence === 'number' ? result.confidence : null;
    translationEl.textContent = text;
    confidenceEl.textContent = confidence !== null ? `Confidence ${Math.round(confidence * 100)}%` : '';
    addHistoryItem(result.label || text, text, confidence);
    lastTranslation = text;
    toggleControls(true);
}

// ─── Feature Extraction (matches Python collect_live_data.py exactly) ───
function extractHandData(handLandmarks, anchor) {
    if (!handLandmarks || !anchor) return new Array(63).fill(0.0);
    let data = [];
    for (let lm of handLandmarks) {
        data.push(lm.x - anchor.x, lm.y - anchor.y, lm.z - anchor.z);
    }
    return data;
}

function extractPoseData(poseLandmarks) {
    const selectedIndices = [0, 11, 12, 13, 14, 15, 16];
    if (!poseLandmarks || poseLandmarks.length < 17) return new Array(21).fill(0.0);

    let ls = poseLandmarks[11]; // left shoulder
    let rs = poseLandmarks[12]; // right shoulder
    if (!ls || !rs) return new Array(21).fill(0.0);

    let cx = (ls.x + rs.x) / 2;
    let cy = (ls.y + rs.y) / 2;
    let cz = (ls.z + rs.z) / 2;

    let data = [];
    for (let idx of selectedIndices) {
        let lm = poseLandmarks[idx];
        if (!lm) { data.push(0, 0, 0); continue; }
        data.push(lm.x - cx, lm.y - cy, lm.z - cz);
    }
    return data;
}

// ─── Drawing Loop (60fps, lightweight) ───
// Mirror via CSS so the canvas draw is just a simple copy
canvas.style.transform = 'scaleX(-1)';

function drawLoop() {
    if (video.videoWidth && video.videoHeight) {
        if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
        if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
    }
    animationId = requestAnimationFrame(drawLoop);
}

// ─── MediaPipe Send Loop (throttled, every 150ms) ───
const mpCanvas = document.createElement('canvas');
const mpCtx = mpCanvas.getContext('2d');
let mpBusy = false;

async function sendToMediaPipe() {
    if (mpBusy || !video.videoWidth || !video.videoHeight) return;
    mpBusy = true;

    try {
        // Mirror the frame to match Python's cv2.flip(frame, 1)
        if (mpCanvas.width !== video.videoWidth) mpCanvas.width = video.videoWidth;
        if (mpCanvas.height !== video.videoHeight) mpCanvas.height = video.videoHeight;
        mpCtx.save();
        mpCtx.scale(-1, 1);
        mpCtx.translate(-mpCanvas.width, 0);
        mpCtx.drawImage(video, 0, 0);
        mpCtx.restore();

        await holistic.send({ image: mpCanvas });
    } catch (e) {
        console.warn('holistic.send failed:', e);
    } finally {
        mpBusy = false;
    }
}

// ─── Prediction Loop (every 300ms, reads latest landmarks) ───
async function runPrediction() {
    if (!latestLandmarks || isProcessing) return;

    const { pose, leftHand, rightHand } = latestLandmarks;
    const hasHands = leftHand || rightHand;

    if (!hasHands) {
        if (!pose) {
            setStatus('No body detected — move closer', 'idle');
        } else {
            setStatus('Show your hands to the camera', 'detecting');
        }
        return;
    }

    isProcessing = true;
    setStatus('Detecting sign...', 'active');

    try {
        let poseData = extractPoseData(pose);
        let leftAnchor = pose ? pose[11] : null;
        let rightAnchor = pose ? pose[12] : null;
        let leftHandData = extractHandData(leftHand, leftAnchor);
        let rightHandData = extractHandData(rightHand, rightAnchor);

        let features = poseData.concat(leftHandData, rightHandData);
        // Pad or trim to exactly 147
        if (features.length < 147) features = features.concat(new Array(147 - features.length).fill(0.0));
        else if (features.length > 147) features = features.slice(0, 147);

        const res = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features })
        });

        if (res.ok) {
            updateOutput(await res.json());
        } else {
            setStatus('Backend error: ' + (await res.text()), 'error');
        }
    } catch (err) {
        setStatus('Cannot reach backend — is app.py running?', 'error');
    } finally {
        isProcessing = false;
    }
}

// ─── Camera Start / Stop ───
async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        });
        video.srcObject = stream;
        await video.play();

        cameraOffOverlay.style.display = 'none';
        video.style.display = 'block';
        video.style.opacity = '0';
        video.style.position = 'absolute';
        scanLine.style.opacity = '1';
        spotlight.style.opacity = '1';
        cameraBtn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <line x1="5" y1="12" x2="19" y2="12"/>
            </svg>
            Stop Camera`;
        cameraBtn.setAttribute('aria-label', 'Stop camera');
        setStatus('Camera on — Waiting for MediaPipe to load...', 'detecting');
        resetOutput();

        // Start all three independent loops
        animationId = requestAnimationFrame(drawLoop);                     // 60fps draw
        setInterval(sendToMediaPipe, 100);                                  // 10fps to MediaPipe
        predictTimer = setInterval(runPrediction, 200);                    // 5fps predictions

    } catch (error) {
        console.error('Camera start failed:', error);
        setStatus('Camera access denied or unavailable', 'error');
    }
}

function stopCamera() {
    if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
    if (predictTimer) { clearInterval(predictTimer); predictTimer = null; }
    latestLandmarks = null;

    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.srcObject = null;
    video.style.display = 'none';
    cameraOffOverlay.style.display = 'flex';
    scanLine.style.opacity = '0';
    spotlight.style.opacity = '0';
    cameraBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        Start Camera`;
    cameraBtn.setAttribute('aria-label', 'Start camera');
    setStatus('Camera off — Ready to translate', 'idle');
}

function speakTranslation() {
    const text = lastTranslation || translationEl.textContent;
    if (!text || text === '—') return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
}

// ─── Event Listeners ───
cameraBtn.addEventListener('click', () => { stream ? stopCamera() : startCamera(); });
speakBtn.addEventListener('click', speakTranslation);
clearBtn.addEventListener('click', () => {
    resetOutput();
    setStatus('Camera on — Ready to detect new signs', 'active');
});
window.addEventListener('beforeunload', stopCamera);

resetOutput();
setStatus('Camera off — Ready to translate', 'idle');
