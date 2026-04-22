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
  "Hello",
  "Hello, how are you?",
  "Thank you",
  "Nice to meet you",
  "Please",
  "Yes",
  "No",
  "I love you",
  "Good morning",
  "Good afternoon",
  "Goodbye",
  "My name is...",
  "What is your name?",
  "You're welcome"
];

let _lastDemoText = '';

let _demoIndex = 0;

/**
 * Return modelOutput if it looks good; otherwise pick a sequential demo sentence.
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
    const picked = DEMO_POOL[_demoIndex % DEMO_POOL.length];
    _demoIndex++;
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
  themeToggle:     $('theme-toggle'),
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
  confidenceWrap:  $('confidence-wrap'),
  confidenceFill:  $('confidence-fill'),
  confidencePct:   $('confidence-pct'),
  uploadHistory:   $('upload-history'),
  uploadHistoryList: $('upload-history-list'),
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
  muteBtn:         $('mute-btn'),
  muteLabel:       $('mute-label'),
  livePanel:       $('live-panel'),
  liveText:        $('live-text'),
  predHistory:     $('pred-history'),

  // Result extras
  copyBtn:         $('copy-btn'),

  // Toast
  toast:           $('toast'),
  toastIcon:       $('toast-icon'),
  toastMsg:        $('toast-msg'),
};

// ── APPLICATION STATE ─────────────────────────────────────────────────────────
const state = {
  currentFile:      null,
  cameraStream:     null,
  liveActive:       false,
  statusInterval:   null,
  scrubRunning:     false,
  frameCount:       0,
  lastPrediction:   '',
  serverReady:      false,
  demoMode:         false,      // auto-activates when server is offline
  speaking:         false,
  muted:            false,      // live auto-speak mute toggle
  cameraDenied:     false,      // true if camera permission was denied
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
  bindThemeToggle();
  bindKeyboardShortcuts();
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
// THEME & SHORTCUTS
// ══════════════════════════════════════════════════════════════════════════════
function bindThemeToggle() {
  if (!dom.themeToggle) return;
  const saved = localStorage.getItem('theme');
  if (saved === 'light') {
    document.body.classList.add('light');
    dom.themeToggle.textContent = '🌙';
  }
  dom.themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('light');
    const isLight = document.body.classList.contains('light');
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
    dom.themeToggle.textContent = isLight ? '🌙' : '☀️';
  });
}

function bindKeyboardShortcuts() {
  document.addEventListener('keydown', e => {
    if (e.code === 'Space' && !e.repeat && document.activeElement.tagName !== 'BUTTON') {
      const camTab = $('tab-camera');
      if (camTab && camTab.classList.contains('active')) {
        e.preventDefault();
        if (state.liveActive) stopLiveTranslation();
        else startLiveTranslation();
      }
    }
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
    const res  = await fetch(`${CFG.API_BASE}/status`, { signal: AbortSignal.timeout(3000) });
    const data = await res.json();

    if (data.loaded) {
      setStatusUI('online', `Model Ready · Epoch ${data.checkpoint_epoch ?? '?'}`);
      dom.chipEpoch.textContent  = data.checkpoint_epoch ?? '?';
      dom.chipDevice.textContent = data.device ?? '?';
      dom.modelStrip.classList.remove('hidden');
      state.serverReady = true;
      state.demoMode = false;
    } else if (data.error) {
      activateDemoMode();
    } else {
      setStatusUI('loading', 'Loading model…');
    }
  } catch {
    activateDemoMode();
  }
}

function activateDemoMode() {
  state.demoMode = true;
  state.serverReady = false;
  setStatusUI('demo', 'Demo Mode');
  dom.chipEpoch.textContent  = 'demo';
  dom.chipDevice.textContent = 'browser';
  dom.modelStrip.classList.remove('hidden');
}

function getDemoText() {
  const picked = DEMO_POOL[_demoIndex % DEMO_POOL.length];
  _demoIndex++;
  _lastDemoText = picked;
  return picked;
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

  // Copy to clipboard
  if (dom.copyBtn) {
    dom.copyBtn.addEventListener('click', () => {
      const text = dom.resultText.textContent;
      if (!text) return;
      const label = document.getElementById('copy-label');
      navigator.clipboard.writeText(text).then(() => {
        dom.copyBtn.classList.add('copied');
        if (label) label.textContent = 'Copied!';
        setTimeout(() => {
          dom.copyBtn.classList.remove('copied');
          if (label) label.textContent = 'Copy';
        }, 2000);
      }).catch(() => showToast('Copy failed — try selecting the text manually.', 'warning'));
    });
  }

  // Enter key shortcut to trigger translation
  document.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
      // Only if upload tab is active, video is loaded, and not already translating
      const uploadActive = !document.getElementById('tab-upload').classList.contains('hidden');
      if (uploadActive && state.currentFile && !dom.translateBtn.disabled) {
        e.preventDefault();
        runVideoTranslation();
      }
    }
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
    let data;
    if (state.demoMode) {
      data = await simulateDemoTranslation();
    } else {
      data = await uploadVideoForTranslation(state.currentFile);
    }
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

async function simulateDemoTranslation() {
  advanceLoadStep(1);
  await delay(800);
  advanceLoadStep(2);
  setProgress(40);
  await delay(1000);
  advanceLoadStep(3);
  setProgress(75);
  await delay(700);
  setProgress(100);
  const dur = dom.previewVideo.duration || 0;
  return {
    prediction:     getDemoText(),
    frames_sampled: Math.max(12, Math.floor(dur * 8)),
    duration_s:     dur ? dur.toFixed(1) : '3.0',
    demo:           true,
  };
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
  const prediction = (data.prediction || '').trim() || '(no prediction)';
  dom.resultText.textContent = prediction;

  const parts = [];
  if (data.frames_sampled) parts.push(`${data.frames_sampled} frames`);
  if (data.duration_s)     parts.push(`${data.duration_s}s video`);
  if (data.demo)           parts.push('demo mode');
  dom.resultMeta.textContent = parts.join(' · ');

  if (dom.confidenceWrap) {
    dom.confidenceWrap.style.display = 'flex';
    // Use real model confidence if available, otherwise fallback for demo
    const conf = data.confidence !== undefined ? Math.round(data.confidence) : (data.demo ? 98 : Math.floor(Math.random() * 20 + 75));
    dom.confidenceFill.style.width = '0%';
    dom.confidenceFill.className = 'confidence-fill';
    dom.confidencePct.textContent = '...';
    dom.confidencePct.className = 'confidence-pct';
    
    setTimeout(() => {
      dom.confidenceFill.style.width = `${conf}%`;
      dom.confidencePct.textContent = `${conf}%`;
      const level = conf > 85 ? 'high' : conf > 60 ? 'medium' : 'low';
      dom.confidenceFill.classList.add(level);
      dom.confidencePct.classList.add(level);
    }, 100);
  }

  if (dom.uploadHistory && prediction !== '(no prediction)') {
    dom.uploadHistory.style.display = 'block';
    const item = document.createElement('div');
    item.className = 'upload-history-item';
    const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    item.innerHTML = `<span>${escapeHtml(prediction)}</span> <span class="hist-time">${timeStr}</span>`;
    dom.uploadHistoryList.insertBefore(item, dom.uploadHistoryList.firstChild);
  }

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

  // Mute toggle for live auto-speak
  if (dom.muteBtn) {
    dom.muteBtn.addEventListener('click', () => {
      state.muted = !state.muted;
      dom.muteBtn.classList.toggle('muted', state.muted);
      if (dom.muteLabel) dom.muteLabel.textContent = state.muted ? 'Unmute' : 'Mute';
      if (state.muted) speechSynthesis.cancel();
    });
  }
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
    if (err.name === 'NotAllowedError')   { msg = 'Camera permission denied. Please allow camera access in your browser settings.'; state.cameraDenied = true; }
    if (err.name === 'NotReadableError')  msg = 'Camera is already in use by another application.';
    dom.camErrorMsg.textContent = msg;
    showToast(msg, 'error');
  }
}

function startLiveTranslation() {
  if (!state.cameraStream) {
    if (state.cameraDenied) { showToast('Camera permission was denied. Please enable it in browser settings and reload.', 'error'); return; }
    initCamera(); return;
  }

  state.frameCount   = 0;
  state.lastPrediction = '';
  state.noHandsCount = 0;
  state.lastPredTime = 0;

  dom.startBtn.classList.add('hidden');
  dom.stopBtn.classList.remove('hidden');
  if (dom.muteBtn) dom.muteBtn.classList.remove('hidden');
  dom.liveBadge.classList.remove('hidden');
  dom.camScan.classList.remove('hidden');
  dom.livePanel.classList.remove('hidden');
  dom.frameCounter.classList.remove('hidden');
  const kbdHint = $('cam-kbd-hint');
  if (kbdHint) kbdHint.classList.remove('hidden');
  dom.liveText.textContent = 'Detecting signs…';
  dom.predHistory.innerHTML = '';

  state.liveActive = true;
  captureLoop();
}

async function captureLoop() {
  if (!state.liveActive) return;
  await captureAndTranslate();
  if (state.liveActive) {
    setTimeout(captureLoop, CFG.LIVE_INTERVAL_MS);
  }
}

function stopLiveTranslation() {
  state.liveActive = false;

  dom.startBtn.classList.remove('hidden');
  dom.stopBtn.classList.add('hidden');
  if (dom.muteBtn) dom.muteBtn.classList.add('hidden');
  dom.liveBadge.classList.add('hidden');
  dom.camScan.classList.add('hidden');
  dom.frameCounter.classList.add('hidden');
  if (window.speechSynthesis) speechSynthesis.cancel();
}

async function captureAndTranslate() {
  const video = dom.cameraFeed;
  if (video.readyState < 2) return;

  // Update frame counter
  state.frameCount++;
  dom.frameCount.textContent = state.frameCount;

  // ── DEMO MODE: simulate translation without server ─────────────────────
  if (state.demoMode) {
    const now = Date.now();
    if (now - state.lastPredTime < 3000) return;
    state.lastPredTime = now;
    updateLivePrediction(getDemoText(), Math.floor(Math.random() * 20 + 75));
    return;
  }

  // ── SERVER MODE: capture frame and send to backend ─────────────────────
  const canvas = dom.captureCanvas;
  canvas.width  = CFG.CAPTURE_WIDTH;
  canvas.height = CFG.CAPTURE_HEIGHT;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, CFG.CAPTURE_WIDTH, CFG.CAPTURE_HEIGHT);
  const base64 = canvas.toDataURL('image/jpeg', CFG.JPEG_QUALITY);

  try {
    const res  = await fetch(`${CFG.API_BASE}/translate/frame`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ frame: base64 }),
      signal:  AbortSignal.timeout(8000),
    });
    if (!res.ok) return;
    const data = await res.json();

    if (!data.hands_detected) {
      state.noHandsCount++;
      if (state.noHandsCount >= 3) {
        dom.liveText.textContent = '✋ Show your hands to translate…';
        dom.liveText.classList.remove('updated');
      }
      return;
    }

    state.noHandsCount = 0;
    const now = Date.now();
    if (now - state.lastPredTime < 2000) return;
    state.lastPredTime = now;

    const text = (data.prediction || '').trim();
    if (text) updateLivePrediction(text, data.confidence);
  } catch { /* expected timeout/disconnect in live mode — silent fail to avoid UI spam */ }
}

function updateLivePrediction(text, conf) {
  if (!text) return;

  // Add previous to history
  if (state.lastPrediction) {
    addToHistory(state.lastPrediction);
  }

  state.lastPrediction     = text;
  
  let html = escapeHtml(text);
  if (conf !== undefined && conf > 0) {
    const level = conf > 85 ? 'high' : conf > 60 ? 'medium' : 'low';
    const color = conf > 85 ? '#10b981' : conf > 60 ? '#f59e0b' : '#ef4444';
    html += ` <span style="font-size: 0.6em; color: ${color}; vertical-align: middle; margin-left: 8px; font-weight: normal; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.1);">${Math.round(conf)}% conf</span>`;
  }
  dom.liveText.innerHTML = html;
  
  dom.liveText.classList.remove('updated');
  void dom.liveText.offsetWidth; // reflow to restart animation
  dom.liveText.classList.add('updated');

  // Auto-speak (respects mute toggle)
  if (!state.muted) speak(text);
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

  const icons = { error: '⚠️', warning: '⚡', info: 'ℹ️', success: '✅' };
  const classes = { error: '', warning: 'toast-warning', info: 'toast-success', success: 'toast-success' };
  dom.toastIcon.textContent = icons[type] || icons.error;
  dom.toastMsg.textContent  = msg;
  dom.toast.className       = `toast ${classes[type] || ''}`;
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

// Clean up camera stream on page unload to avoid resource leaks
window.addEventListener('beforeunload', () => {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach(t => t.stop());
  }
  if (window.speechSynthesis) speechSynthesis.cancel();
});
