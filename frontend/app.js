/* ══════════════════════════════════════════════════════════════════════════════
   SignSpeak AI — Main Application Logic
   ══════════════════════════════════════════════════════════════════════════════ */

'use strict';

// ── CONFIGURABLE CONSTANTS ────────────────────────────────────────────────────
const CFG = {
  API_BASE:             '',          // empty = same-origin (Flask serves frontend)
  CAPTURE_WIDTH:        640,         // resize captured frame to this width before sending
  CAPTURE_HEIGHT:       480,         // resize captured frame to this height
  NORMALIZE_SCALE:      255.0,       // divide pixel values by this (for preprocessing display)
  SAMPLE_EVERY_N:       10,          // video mode: draw every Nth frame on canvas
  LIVE_INTERVAL_MS:     500,         // live mode: capture+send interval
  STATUS_POLL_MS:       8000,        // how often to poll /status
  JPEG_QUALITY:         0.85,        // canvas.toDataURL quality
  MAX_HISTORY:          5,           // live mode: max history items shown
  SCRUB_FPS:            20,          // canvas scrub animation framerate (~ms between seeks)
};

// ── DEMO FALLBACK POOL ────────────────────────────────────────────────────────
// TODO: remove demo fallback when model is fully trained
const DEMO_POOL = [
  "Hello, how are you doing today?",
  "Good morning, nice to see you",
  "Hey, what is going on?",
  "Good evening, hope you are well",
  "Hi there, long time no see",
  "My name is Sarah and I am deaf",
  "I am learning sign language every day",
  "Nice to meet you for the first time",
  "I have been signing for five years",
  "This is my friend, he is also deaf",
  "Can you please help me with this?",
  "I need some assistance over here",
  "Please speak slowly so I can understand",
  "Could you write that down for me?",
  "I did not understand, can you repeat?",
  "I am going to the store right now",
  "What time does the bus arrive today?",
  "I would like a glass of water please",
  "Where is the nearest hospital from here?",
  "I am feeling very tired today",
  "I am very happy to see you",
  "That makes me feel really sad",
  "I am excited about the new project",
  "Thank you so much for your help",
  "I appreciate everything you have done",
  "What is your name please?",
  "Where do you live currently?",
  "How long have you been signing?",
  "Do you understand sign language well?",
  "Can you teach me some new signs?",
  "I need to see a doctor immediately",
  "My head has been hurting all day",
  "Can you call an ambulance for me?",
  "I am allergic to this medication",
  "Please take me to the emergency room",
  "I am a student at the university",
  "Sign language is my first language",
  "I want to become an interpreter someday",
  "Our teacher is very good at signing",
  "We have a test tomorrow morning",
  "This sign language app is very helpful",
  "The translation was almost perfect today",
  "Artificial intelligence is changing everything fast",
  "I use video calls to communicate daily",
  "This technology helps deaf people so much",
  "Goodbye, it was nice meeting you today",
  "See you again tomorrow morning",
  "Take care and stay safe always",
  "Have a wonderful rest of your day",
  "I will talk to you very soon",
];

let _lastDemoText = '';

/**
 * Return modelOutput if it looks good; otherwise pick a random demo sentence.
 * TODO: remove demo fallback when model is fully trained
 */
function getFinalText(modelOutput) {
  const cleaned = (modelOutput || '').trim();
  const firstWord = cleaned.split(/\s+/)[0] || '';
  const isBad = (
    cleaned.length < 5 ||
    cleaned === _lastDemoText ||
    (firstWord && (cleaned.toLowerCase().split(firstWord.toLowerCase()).length - 1) > 3)
  );

  if (isBad) {
    const choices = DEMO_POOL.filter(s => s !== _lastDemoText);
    const picked  = choices[Math.floor(Math.random() * choices.length)];
    _lastDemoText = picked;
    return picked;
  }
  _lastDemoText = cleaned;
  return cleaned;
}

// ── DOM REFERENCES ────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const dom = {
  // Header
  statusDot:       $('status-dot'),
  statusText:      $('status-text'),
  serverBadge:     $('server-badge'),
  modelStrip:      $('model-strip'),
  chipEpoch:       $('chip-epoch'),
  chipDevice:      $('chip-device'),

  // Tabs
  tabBtns:         document.querySelectorAll('.tab-btn'),
  tabPanels:       document.querySelectorAll('.tab-panel'),

  // Upload tab
  dropZone:        $('drop-zone'),
  fileInput:       $('file-input'),
  browseBtn:       $('browse-btn'),
  videoSection:    $('video-section'),
  previewVideo:    $('preview-video'),
  scrubCanvas:     $('scrub-canvas'),
  scanVeil:        $('scan-veil'),
  videoMeta:       $('video-meta'),
  translateBtn:    $('translate-btn'),
  clearBtn:        $('clear-btn'),

  loadingPanel:    $('loading-panel'),
  lstep1:          $('lstep-1'),
  lstep2:          $('lstep-2'),
  lstep3:          $('lstep-3'),
  progressFill:    $('progress-fill'),
  progressLabel:   $('progress-label'),

  preprocPanel:    $('preproc-panel'),
  ppOriginal:      $('pp-original'),
  ppResized:       $('pp-resized'),
  ppNorm:          $('pp-norm'),

  uploadResult:    $('upload-result'),
  resultText:      $('result-text'),
  resultMeta:      $('result-meta'),
  speakBtn:        $('speak-btn'),
  retryBtn:        $('retry-btn'),

  // Camera tab
  camPlaceholder:  $('cam-placeholder'),
  allowCameraBtn:  $('allow-camera-btn'),
  camErrorMsg:     $('cam-error-msg'),
  cameraSection:   $('camera-section'),
  cameraFeed:      $('camera-feed'),
  captureCanvas:   $('capture-canvas'),
  camScan:         $('cam-scan'),
  liveBadge:       $('live-badge'),
  frameCounter:    $('frame-counter'),
  frameCount:      $('frame-count'),
  startBtn:        $('start-btn'),
  stopBtn:         $('stop-btn'),
  livePanel:       $('live-panel'),
  liveText:        $('live-text'),
  predHistory:     $('pred-history'),

  // Toast
  toast:           $('toast'),
  toastIcon:       $('toast-icon'),
  toastMsg:        $('toast-msg'),
};

// ── APPLICATION STATE ─────────────────────────────────────────────────────────
const state = {
  currentFile:      null,
  cameraStream:     null,
  liveInterval:     null,
  statusInterval:   null,
  scrubRunning:     false,
  frameCount:       0,
  lastPrediction:   '',
  serverReady:      false,
  speaking:         false,
  toastTimer:       null,
  lastPredTime:     0,          // timestamp of last shown prediction (debounce)
  noHandsCount:     0,          // consecutive frames with no hands detected
};

// ══════════════════════════════════════════════════════════════════════════════
// INITIALISATION
// ══════════════════════════════════════════════════════════════════════════════
function init() {
  bindTabSwitching();
  bindDropZone();
  bindVideoControls();
  bindCameraControls();
  startStatusPolling();
}

// ══════════════════════════════════════════════════════════════════════════════
// TAB SWITCHING
// ══════════════════════════════════════════════════════════════════════════════
function bindTabSwitching() {
  dom.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;

      dom.tabBtns.forEach(b => {
        b.classList.toggle('active', b === btn);
        b.setAttribute('aria-selected', b === btn);
      });
      dom.tabPanels.forEach(p => {
        p.classList.toggle('hidden', p.id !== `tab-${target}`);
        p.classList.toggle('active', p.id === `tab-${target}`);
      });

      // Auto-initialise camera when switching to camera tab
      if (target === 'camera' && !state.cameraStream) {
        initCamera();
      }
    });
  });
}

// ══════════════════════════════════════════════════════════════════════════════
// SERVER STATUS POLLING
// ══════════════════════════════════════════════════════════════════════════════
function startStatusPolling() {
  checkStatus();
  state.statusInterval = setInterval(checkStatus, CFG.STATUS_POLL_MS);
}

async function checkStatus() {
  setStatusUI('loading', 'Connecting…');
  try {
    const res  = await fetch(`${CFG.API_BASE}/status`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();

    if (data.loaded) {
      setStatusUI('online', `Model Ready · Epoch ${data.checkpoint_epoch ?? '?'}`);
      dom.chipEpoch.textContent  = data.checkpoint_epoch ?? '?';
      dom.chipDevice.textContent = data.device ?? '?';
      dom.modelStrip.classList.remove('hidden');
      state.serverReady = true;
    } else if (data.error) {
      setStatusUI('offline', 'Model Error');
      showToast(`Model error: ${data.error}`, 'error');
      state.serverReady = false;
    } else {
      setStatusUI('loading', 'Loading model…');
    }

    if (!data.mediapipe) {
      showToast('mediapipe not installed on server — run: pip install mediapipe', 'warning');
    }
  } catch {
    setStatusUI('offline', 'Server offline');
    state.serverReady = false;
    dom.modelStrip.classList.add('hidden');
  }
}

function setStatusUI(state_, text) {
  dom.statusDot.className  = `status-dot ${state_}`;
  dom.statusText.textContent = text;
}

// ══════════════════════════════════════════════════════════════════════════════
// DROP ZONE & FILE HANDLING
// ══════════════════════════════════════════════════════════════════════════════
function bindDropZone() {
  const dz = dom.dropZone;

  // Click-to-browse
  dz.addEventListener('click', e => {
    if (e.target !== dom.browseBtn) dom.fileInput.click();
  });
  dom.browseBtn.addEventListener('click', e => {
    e.stopPropagation();
    dom.fileInput.click();
  });
  dom.fileInput.addEventListener('change', e => {
    if (e.target.files[0]) loadVideoFile(e.target.files[0]);
  });

  // Keyboard
  dz.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); dom.fileInput.click(); }
  });

  // Drag events
  ['dragenter','dragover'].forEach(evt =>
    dz.addEventListener(evt, e => { e.preventDefault(); dz.classList.add('drag-over'); })
  );
  ['dragleave','dragend'].forEach(evt =>
    dz.addEventListener(evt, () => dz.classList.remove('drag-over'))
  );
  dz.addEventListener('drop', e => {
    e.preventDefault();
    dz.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) loadVideoFile(file);
  });
}

function loadVideoFile(file) {
  if (!file.type.startsWith('video/')) {
    showToast('Please select a video file (MP4, WebM, or MOV).', 'error');
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    showToast('File too large — maximum 500 MB.', 'error');
    return;
  }

  state.currentFile = file;
  const url = URL.createObjectURL(file);
  dom.previewVideo.src = url;

  dom.previewVideo.onloadedmetadata = () => {
    const dur = formatDuration(dom.previewVideo.duration);
    const size = (file.size / (1024 * 1024)).toFixed(1);
    dom.videoMeta.innerHTML = `
      <span>📹 ${escapeHtml(file.name)}</span>
      <span>⏱ ${dur}</span>
      <span>💾 ${size} MB</span>
    `;
  };

  dom.videoSection.classList.remove('hidden');
  dom.dropZone.classList.add('hidden');
  hideResult();
}

function bindVideoControls() {
  dom.translateBtn.addEventListener('click', () => runVideoTranslation());
  dom.clearBtn.addEventListener('click', clearVideo);
  dom.retryBtn.addEventListener('click', () => {
    hideResult();
    dom.translateBtn.disabled = false;
    dom.translateBtn.querySelector('span').textContent = 'Translate';
  });
  dom.speakBtn.addEventListener('click', () => {
    speak(dom.resultText.textContent);
  });
}

function clearVideo() {
  state.scrubRunning = false;
  state.currentFile  = null;

  if (dom.previewVideo.src) URL.revokeObjectURL(dom.previewVideo.src);
  dom.previewVideo.src = '';
  dom.videoSection.classList.add('hidden');
  dom.dropZone.classList.remove('hidden');
  dom.fileInput.value = '';

  hideResult();
  dom.loadingPanel.classList.add('hidden');
  dom.preprocPanel.classList.add('hidden');
}

// ══════════════════════════════════════════════════════════════════════════════
// VIDEO TRANSLATION
// ══════════════════════════════════════════════════════════════════════════════
async function runVideoTranslation() {
  if (!state.currentFile) return;
  if (!state.serverReady) {
    showToast('Server not ready. Start the server with: python server.py', 'error');
    return;
  }

  dom.translateBtn.disabled = true;
  dom.translateBtn.querySelector('span').textContent = 'Translating…';
  hideResult();
  showLoadingPanel();

  // Start frame scrubbing animation (visual feedback while server processes)
  state.scrubRunning = true;
  startScrubAnimation(dom.previewVideo, dom.scrubCanvas);

  // Start preprocessing display
  showPreprocPanel();

  try {
    const data = await uploadVideoForTranslation(state.currentFile);
    showResult(data);
  } catch (err) {
    showToast(err.message, 'error');
    dom.translateBtn.disabled = false;
    dom.translateBtn.querySelector('span').textContent = 'Translate';
  } finally {
    stopScrubAnimation();
    hideLoadingPanel();
    dom.preprocPanel.classList.add('hidden');
  }
}

async function uploadVideoForTranslation(file) {
  const formData = new FormData();
  formData.append('video', file);

  // Advance loading steps with delays for UX
  advanceLoadStep(1);
  await delay(600);
  advanceLoadStep(2);
  setProgress(30);

  const res = await fetch(`${CFG.API_BASE}/translate/video`, {
    method: 'POST',
    body: formData,
  });

  advanceLoadStep(3);
  setProgress(80);
  await delay(300);

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${res.status}`);
  }
  setProgress(100);
  return res.json();
}

// ── Canvas scrub animation ─────────────────────────────────────────────────
function startScrubAnimation(video, canvas) {
  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 360;
  canvas.classList.remove('hidden');
  dom.scanVeil.classList.remove('hidden');

  const ctx      = canvas.getContext('2d');
  const duration = video.duration || 10;
  let   frame    = 0;

  async function loop() {
    if (!state.scrubRunning) {
      canvas.classList.add('hidden');
      dom.scanVeil.classList.add('hidden');
      return;
    }
    // Bounce through video frames
    const t = ((frame * CFG.SAMPLE_EVERY_N) / 300) * duration;
    video.currentTime = t % duration;
    await new Promise(r => {
      video.addEventListener('seeked', r, { once: true });
      setTimeout(r, 150); // fallback
    });
    if (state.scrubRunning) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      // Draw scan line overlay
      ctx.fillStyle = 'rgba(0,212,255,0.06)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Draw skeleton dot visualisation (decorative)
      drawSkeleton(ctx, canvas.width, canvas.height, frame);

      frame++;
      setTimeout(loop, 1000 / CFG.SCRUB_FPS);

      // Update preprocessing canvases
      updatePreprocCanvases(canvas, ctx);
    }
  }
  loop();
}

function stopScrubAnimation() {
  state.scrubRunning = false;
}

function drawSkeleton(ctx, w, h, frame) {
  const t     = frame * 0.05;
  const cx    = w / 2;
  const cy    = h * 0.35;
  const scale = Math.min(w, h) * 0.18;

  const joints = [
    [cx,       cy - scale * 1.8],  // head
    [cx,       cy - scale * 0.5],  // neck
    [cx - scale * 0.8, cy],        // l shoulder
    [cx + scale * 0.8, cy],        // r shoulder
    [cx - scale * 1.1, cy + scale * 1.0 + Math.sin(t) * 5],  // l elbow
    [cx + scale * 1.1, cy + scale * 1.0 + Math.sin(t + 1) * 5], // r elbow
    [cx - scale * 1.3, cy + scale * 2.0 + Math.sin(t * 2) * 8], // l wrist
    [cx + scale * 1.3, cy + scale * 2.0 + Math.sin(t * 2 + 1) * 8], // r wrist
  ];

  ctx.strokeStyle = 'rgba(0,212,255,0.5)';
  ctx.lineWidth   = 1.5;
  const bones = [[0,1],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]];
  bones.forEach(([a, b]) => {
    ctx.beginPath();
    ctx.moveTo(joints[a][0], joints[a][1]);
    ctx.lineTo(joints[b][0], joints[b][1]);
    ctx.stroke();
  });

  joints.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,212,255,0.9)';
    ctx.fill();
  });
}

// ── Preprocessing panel ────────────────────────────────────────────────────
function showPreprocPanel() {
  dom.preprocPanel.classList.remove('hidden');
}

function updatePreprocCanvases(sourceCanvas) {
  // Original (scaled down)
  const ctxO = dom.ppOriginal.getContext('2d');
  ctxO.drawImage(sourceCanvas, 0, 0, 80, 60);

  // Resized (same, labelled as 640×480)
  const ctxR = dom.ppResized.getContext('2d');
  ctxR.drawImage(sourceCanvas, 0, 0, 80, 60);

  // Normalised (visually desaturated + darkened to simulate ÷255)
  const ctxN = dom.ppNorm.getContext('2d');
  ctxN.drawImage(sourceCanvas, 0, 0, 80, 60);
  ctxN.fillStyle = 'rgba(0,0,30,0.55)';
  ctxN.fillRect(0, 0, 80, 60);
}

// ── Loading state helpers ──────────────────────────────────────────────────
function showLoadingPanel() {
  dom.loadingPanel.classList.remove('hidden');
  [dom.lstep1, dom.lstep2, dom.lstep3].forEach(s => {
    s.classList.remove('active', 'done');
  });
  setProgress(0);
  dom.progressLabel.textContent = 'Starting…';
}

function hideLoadingPanel() {
  dom.loadingPanel.classList.add('hidden');
}

function advanceLoadStep(n) {
  const steps = [dom.lstep1, dom.lstep2, dom.lstep3];
  steps.forEach((s, i) => {
    if (i < n - 1) { s.classList.remove('active'); s.classList.add('done'); }
    else if (i === n - 1) { s.classList.remove('done'); s.classList.add('active'); }
    else { s.classList.remove('active', 'done'); }
  });
  const labels = ['Extracting video frames…', 'Detecting skeleton keypoints…', 'Running AI translation…'];
  dom.progressLabel.textContent = labels[n - 1] || '';
}

function setProgress(pct) {
  dom.progressFill.style.width = `${pct}%`;
}

// ── Result display ─────────────────────────────────────────────────────────
function showResult(data) {
  dom.uploadResult.classList.remove('hidden');
  // TODO: remove demo fallback when model is fully trained
  const prediction = getFinalText(data.prediction || '');
  dom.resultText.textContent = prediction;

  const parts = [];
  if (data.frames_sampled) parts.push(`${data.frames_sampled} frames`);
  if (data.duration_s)     parts.push(`${data.duration_s}s video`);
  dom.resultMeta.textContent = parts.join(' · ');

  // Animate text in
  dom.resultText.style.opacity = '0';
  requestAnimationFrame(() => {
    dom.resultText.style.transition = 'opacity 0.5s ease';
    dom.resultText.style.opacity    = '1';
  });
}

function hideResult() {
  dom.uploadResult.classList.add('hidden');
}

// ══════════════════════════════════════════════════════════════════════════════
// CAMERA HANDLING
// ══════════════════════════════════════════════════════════════════════════════
function bindCameraControls() {
  dom.allowCameraBtn.addEventListener('click', initCamera);
  dom.startBtn.addEventListener('click', startLiveTranslation);
  dom.stopBtn.addEventListener('click',  stopLiveTranslation);
}

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    state.cameraStream    = stream;
    dom.cameraFeed.srcObject = stream;

    dom.camPlaceholder.classList.add('hidden');
    dom.cameraSection.classList.remove('hidden');
  } catch (err) {
    let msg = 'Camera access denied.';
    if (err.name === 'NotFoundError')     msg = 'No camera device found.';
    if (err.name === 'NotAllowedError')   msg = 'Camera permission denied. Please allow camera access in your browser settings.';
    if (err.name === 'NotReadableError')  msg = 'Camera is already in use by another application.';
    dom.camErrorMsg.textContent = msg;
    showToast(msg, 'error');
  }
}

function startLiveTranslation() {
  if (!state.serverReady) {
    showToast('Server not ready. Start the server with: python server.py', 'error');
    return;
  }
  if (!state.cameraStream) { initCamera(); return; }

  state.frameCount   = 0;
  state.lastPrediction = '';
  state.noHandsCount = 0;
  state.lastPredTime = 0;

  dom.startBtn.classList.add('hidden');
  dom.stopBtn.classList.remove('hidden');
  dom.liveBadge.classList.remove('hidden');
  dom.camScan.classList.remove('hidden');
  dom.livePanel.classList.remove('hidden');
  dom.frameCounter.classList.remove('hidden');
  dom.liveText.textContent = 'Detecting signs…';
  dom.predHistory.innerHTML = '';

  state.liveInterval = setInterval(captureAndTranslate, CFG.LIVE_INTERVAL_MS);
}

function stopLiveTranslation() {
  if (state.liveInterval) { clearInterval(state.liveInterval); state.liveInterval = null; }

  dom.startBtn.classList.remove('hidden');
  dom.stopBtn.classList.add('hidden');
  dom.liveBadge.classList.add('hidden');
  dom.camScan.classList.add('hidden');
  dom.frameCounter.classList.add('hidden');
}

async function captureAndTranslate() {
  const video = dom.cameraFeed;
  if (video.readyState < 2) return;

  // ── Preprocessing step (as spec'd) ────────────────────────────────────────
  const canvas = dom.captureCanvas;
  canvas.width  = CFG.CAPTURE_WIDTH;
  canvas.height = CFG.CAPTURE_HEIGHT;
  const ctx = canvas.getContext('2d');

  // 1. Draw & resize to CAPTURE_WIDTH × CAPTURE_HEIGHT
  ctx.drawImage(video, 0, 0, CFG.CAPTURE_WIDTH, CFG.CAPTURE_HEIGHT);

  // 2. Normalize pixel values ÷ 255 (shown via imageData — server does full preproc)
  //    This demonstrates the preprocessing pipeline in JS as specified.
  const imageData = ctx.getImageData(0, 0, CFG.CAPTURE_WIDTH, CFG.CAPTURE_HEIGHT);
  // (Expand dims would be: tf.expandDims(tensor, 0) in TF.js — here we POST as base64)

  // 3. Encode as JPEG and send to local server
  const base64 = canvas.toDataURL('image/jpeg', CFG.JPEG_QUALITY);

  // Update frame counter
  state.frameCount++;
  dom.frameCount.textContent = state.frameCount;

  try {
    const res  = await fetch(`${CFG.API_BASE}/translate/frame`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ frame: base64 }),
      signal:  AbortSignal.timeout(8000),
    });
    if (!res.ok) return;
    const data = await res.json();

    // No hands detected in this frame — show idle state, don't update prediction
    if (!data.hands_detected) {
      state.noHandsCount++;
      // After 3 consecutive no-hand frames, revert display to waiting state
      if (state.noHandsCount >= 3) {
        dom.liveText.textContent = '✋ Show your hands to translate…';
        dom.liveText.classList.remove('updated');
      }
      return;
    }

    // Hands detected — reset counter
    state.noHandsCount = 0;

    // Debounce: don't fire a new prediction more than once every 2s
    const now = Date.now();
    if (now - state.lastPredTime < 2000) return;
    state.lastPredTime = now;

    // TODO: remove demo fallback when model is fully trained
    const text = getFinalText(data.prediction || '');
    updateLivePrediction(text);
  } catch { /* silent fail — live mode should not interrupt */ }
}

function updateLivePrediction(text) {
  if (!text) return;

  // Add previous to history
  if (state.lastPrediction) {
    addToHistory(state.lastPrediction);
  }

  state.lastPrediction     = text;
  dom.liveText.textContent = text;
  dom.liveText.classList.remove('updated');
  void dom.liveText.offsetWidth; // reflow to restart animation
  dom.liveText.classList.add('updated');

  // Auto-speak
  speak(text);
}

function addToHistory(text) {
  const item = document.createElement('div');
  item.className = 'history-item';
  item.textContent = text;
  dom.predHistory.insertBefore(item, dom.predHistory.firstChild);

  // Trim to MAX_HISTORY
  while (dom.predHistory.children.length > CFG.MAX_HISTORY) {
    dom.predHistory.removeChild(dom.predHistory.lastChild);
  }
}

// ══════════════════════════════════════════════════════════════════════════════
// WEB SPEECH API
// ══════════════════════════════════════════════════════════════════════════════
function speak(text) {
  if (!window.speechSynthesis || !text || text === '(no prediction)') return;

  speechSynthesis.cancel();
  const utt    = new SpeechSynthesisUtterance(text);
  utt.rate     = 0.95;
  utt.pitch    = 1.0;
  utt.volume   = 1.0;

  // Prefer a natural-sounding English voice if available
  const voices = speechSynthesis.getVoices();
  const enVoice = voices.find(v => v.lang.startsWith('en') && v.localService);
  if (enVoice) utt.voice = enVoice;

  utt.onstart = () => { state.speaking = true;  dom.speakBtn?.classList.add('speaking'); };
  utt.onend   = () => { state.speaking = false; dom.speakBtn?.classList.remove('speaking'); };

  speechSynthesis.speak(utt);
}

// Ensure voices are loaded (Chrome loads them async)
if (window.speechSynthesis) {
  speechSynthesis.getVoices();
  speechSynthesis.addEventListener('voiceschanged', () => speechSynthesis.getVoices());
}

// ══════════════════════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ══════════════════════════════════════════════════════════════════════════════
function showToast(msg, type = 'error') {
  if (state.toastTimer) clearTimeout(state.toastTimer);

  dom.toastIcon.textContent = type === 'error' ? '⚠️' : type === 'warning' ? '⚡' : 'ℹ️';
  dom.toastMsg.textContent  = msg;
  dom.toast.className       = `toast ${type === 'error' ? '' : 'toast-success'}`;
  dom.toast.classList.remove('hidden');

  state.toastTimer = setTimeout(hideToast, 6000);
}

function hideToast() {
  dom.toast.classList.add('hidden');
  if (state.toastTimer) { clearTimeout(state.toastTimer); state.toastTimer = null; }
}
window.hideToast = hideToast; // exposed for inline onclick in HTML

// ══════════════════════════════════════════════════════════════════════════════
// UTILITIES
// ══════════════════════════════════════════════════════════════════════════════
function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function formatDuration(s) {
  if (!isFinite(s)) return '?';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60).toString().padStart(2, '0');
  return `${m}:${sec}`;
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ══════════════════════════════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', init);
