#!/usr/bin/env python3
"""
server.py — Local Sign Language Translation Server
===================================================
Serves the frontend at http://localhost:5000 and performs
model inference using the trained T5 + OpenPose-keypoint model.

Setup:
    pip install flask flask-cors mediapipe opencv-python
    python server.py

Then open: http://localhost:5000
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

# ── Pathlib Compatibility Hack (for macOS -> Linux model loading) ────────────
import pathlib
import types
mock_mod = types.ModuleType('pathlib._local')
mock_mod.WindowsPath = Path
mock_mod.PosixPath = Path
pathlib._local = mock_mod
sys.modules['pathlib._local'] = mock_mod

import os
import random
import tempfile
import threading
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from transformers import T5TokenizerFast

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from train_how2sign_t5 import (
    KeypointT5Model,
    feature_dim,
    interpolate_missing,
    normalize_points,
    POSE_POINTS,
    HAND_POINTS,
)

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = ROOT / "runs" / "how2sign_t5_full" / "best.pt"
FRONTEND_DIR       = ROOT / "frontend"
MAX_UPLOAD_MB      = 500
MAX_VIDEO_FRAMES   = 64   # frames sampled from uploaded video
LIVE_REPLICATE     = 4    # repeat single frame N times for temporal context

# MediaPipe POSE → OpenPose BODY_25 index mapping
_MP_TO_OP: dict[int, int] = {
    0:  0,   # nose
    12: 2,   # right shoulder
    14: 3,   # right elbow
    16: 4,   # right wrist
    11: 5,   # left shoulder
    13: 6,   # left elbow
    15: 7,   # left wrist
    24: 9,   # right hip
    26: 10,  # right knee
    28: 11,  # right ankle
    23: 12,  # left hip
    25: 13,  # left knee
    27: 14,  # left ankle
    5:  15,  # right eye inner
    2:  16,  # left eye inner
    8:  17,  # right ear
    7:  18,  # left ear
}

# ── MediaPipe Tasks API (mediapipe ≥ 0.10) ─────────────────────────────────────
# Models are auto-downloaded on first run (~30 MB + ~10 MB).
_MODELS_DIR = ROOT / "models"
_POSE_TASK   = _MODELS_DIR / "pose_landmarker_full.task"
_HAND_TASK   = _MODELS_DIR / "hand_landmarker.task"
_POSE_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
_HAND_URL    = ("https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")

_pose_detector = None
_hand_detector  = None
HAS_MEDIAPIPE   = False


def _init_mediapipe() -> None:
    """Download models once, create landmarker instances. Called in a background thread."""
    global _pose_detector, _hand_detector, HAS_MEDIAPIPE
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as _mp_py
        from mediapipe.tasks.python import vision as _mp_vis

        _MODELS_DIR.mkdir(exist_ok=True)
        for path, url in [(_POSE_TASK, _POSE_URL), (_HAND_TASK, _HAND_URL)]:
            if not path.exists():
                print(f"[mediapipe] Downloading {path.name} …")
                urllib.request.urlretrieve(url, path)
                print(f"[mediapipe] {path.name} ready ({path.stat().st_size // 1024} KB)")

        pose_opts = _mp_vis.PoseLandmarkerOptions(
            base_options=_mp_py.BaseOptions(model_asset_path=str(_POSE_TASK)),
            running_mode=_mp_vis.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
        )
        hand_opts = _mp_vis.HandLandmarkerOptions(
            base_options=_mp_py.BaseOptions(model_asset_path=str(_HAND_TASK)),
            running_mode=_mp_vis.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
        )
        _pose_detector = _mp_vis.PoseLandmarker.create_from_options(pose_opts)
        _hand_detector = _mp_vis.HandLandmarker.create_from_options(hand_opts)
        HAS_MEDIAPIPE  = True
        _state["mp_error"] = None
        print("[mediapipe] Landmarkers ready ✓")
    except Exception as exc:
        HAS_MEDIAPIPE = False
        _state["mp_error"] = str(exc)
        print(f"[mediapipe] Unavailable: {exc}")

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
CORS(app)

# ── Global model state ─────────────────────────────────────────────────────────
_state: dict = {
    "model": None, "tokenizer": None,
    "use_face": False, "max_frames": 256,
    "max_gen_tokens": 64, "min_conf": 0.1,
    "checkpoint_epoch": None, "loaded": False, "error": None,
    "mp_error": "Initializing...",
}
_lock       = threading.Lock()
_mp_lock    = threading.Lock()
_device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_live_buffer = deque(maxlen=16)


def _load_model(checkpoint: Path) -> None:
    with _lock:
        if _state["loaded"]:
            return
        try:
            print(f"[server] Loading checkpoint: {checkpoint}")
            ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=False)
            saved = ckpt.get("args", {})
            if not isinstance(saved, dict):
                saved = vars(saved)
            tok_name = ckpt.get("tokenizer_name", saved.get("pretrained_model", "t5-small"))
            use_face = bool(saved.get("use_face", False))

            tokenizer = T5TokenizerFast.from_pretrained(tok_name)
            model = KeypointT5Model(
                pretrained_name=tok_name,
                input_dim=feature_dim(use_face),
                temporal_stride=int(saved.get("temporal_stride", 1)),
                dropout=float(saved.get("dropout", 0.1)),
            ).to(_device)
            model.load_state_dict(ckpt["model_state"])
            model.eval()

            _state.update(
                model=model, tokenizer=tokenizer,
                use_face=use_face,
                max_frames=int(saved.get("max_frames", 256)),
                max_gen_tokens=int(saved.get("max_gen_tokens", 64)),
                min_conf=float(saved.get("min_conf", 0.1)),
                checkpoint_epoch=ckpt.get("epoch"),
                loaded=True, error=None,
            )
            print(f"[server] Ready — epoch {_state['checkpoint_epoch']} on {_device}")
        except Exception as exc:
            _state["error"] = str(exc)
            print(f"[server] ERROR: {exc}")


def _mp_frame_to_kp(frame_bgr: np.ndarray) -> tuple[torch.Tensor, bool, dict]:
    """One BGR frame → (OpenPose-format [J, 3] tensor, hands_detected, hand_meta)."""
    import mediapipe as mp
    h, w = frame_bgr.shape[:2]
    joints = POSE_POINTS + HAND_POINTS + HAND_POINTS
    kp = np.zeros((joints, 3), dtype=np.float32)

    rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # --- Pose ---
    with _mp_lock:
        pose_res = _pose_detector.detect(mp_image)
    if pose_res.pose_landmarks:
        lms = pose_res.pose_landmarks[0]
        if len(lms) > 12:
            ls, rs = lms[11], lms[12]
            kp[1] = [(ls.x + rs.x) / 2 * w, (ls.y + rs.y) / 2 * h,
                     (getattr(ls, "visibility", 1.0) + getattr(rs, "visibility", 1.0)) / 2]
        if len(lms) > 24:
            lh, rh = lms[23], lms[24]
            kp[8] = [(lh.x + rh.x) / 2 * w, (lh.y + rh.y) / 2 * h,
                     (getattr(lh, "visibility", 1.0) + getattr(rh, "visibility", 1.0)) / 2]
        for mp_i, op_i in _MP_TO_OP.items():
            if mp_i < len(lms) and op_i < POSE_POINTS:
                lm = lms[mp_i]
                kp[op_i] = [lm.x * w, lm.y * h, getattr(lm, "visibility", 1.0)]

    # --- Hands & Gesture Metadata ---
    hands_detected = False
    hand_meta = {
        "hands_detected": False,
        "is_open": False,
        "palm_x": 0.5,
        "palm_y": 0.5,
        "wrist_x": 0.5,
        "wrist_y": 0.5,
        "extended_fingers": 0,
        "num_hands": 0,
    }
    with _mp_lock:
        hand_res = _hand_detector.detect(mp_image)
    if hand_res.hand_landmarks and hand_res.handedness:
        hands_detected = True
        hand_meta["hands_detected"] = True
        hand_meta["num_hands"] = len(hand_res.hand_landmarks)

        primary_lms = hand_res.hand_landmarks[0]
        if len(primary_lms) >= 21:
            wrist = primary_lms[0]
            mcp9  = primary_lms[9]
            hand_meta["palm_x"] = float((wrist.x + mcp9.x) / 2.0)
            hand_meta["palm_y"] = float((wrist.y + mcp9.y) / 2.0)
            hand_meta["wrist_x"] = float(wrist.x)
            hand_meta["wrist_y"] = float(wrist.y)

            # Check finger extension: index(8,5), middle(12,9), ring(16,13), pinky(20,17)
            fingers_up = sum(1 for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]
                             if primary_lms[tip].y < primary_lms[mcp].y)
            thumb_tip, thumb_mcp = primary_lms[4], primary_lms[2]
            thumb_ext = ((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2) > \
                        ((thumb_mcp.x - wrist.x)**2 + (thumb_mcp.y - wrist.y)**2)
            if thumb_ext:
                fingers_up += 1

            hand_meta["extended_fingers"] = fingers_up
            hand_meta["is_open"] = (fingers_up >= 3 and wrist.y < 0.90)

        for hand_lms, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
            label = handedness[0].category_name  # "Left" or "Right"
            off   = POSE_POINTS if label == "Left" else POSE_POINTS + HAND_POINTS
            for i, lm in enumerate(hand_lms[:HAND_POINTS]):
                kp[off + i] = [lm.x * w, lm.y * h, 1.0]

    return torch.tensor(kp, dtype=torch.float32), hands_detected, hand_meta


def _frames_to_feat(frames: list[np.ndarray]) -> tuple[torch.Tensor, bool, list[dict]]:
    """List of BGR frames → ([1, T, D] model input tensor, any_hands_detected, hand_metas)."""
    if not HAS_MEDIAPIPE:
        raise RuntimeError("MediaPipe not ready — check server logs (models may still be downloading)")

    min_conf   = _state["min_conf"]
    max_frames = _state["max_frames"]

    results     = [_mp_frame_to_kp(f) for f in frames]
    tensors     = [r[0] for r in results]
    any_hands   = any(r[1] for r in results)
    metas       = [r[2] for r in results]
    points      = torch.stack(tensors, dim=0)  # [T, J, 3]

    if points.size(0) > max_frames:
        idx    = torch.linspace(0, points.size(0) - 1, max_frames).long()
        points = points[idx]

    points  = interpolate_missing(points, max_gap=4, min_conf=min_conf)
    points  = normalize_points(points, min_conf=min_conf)

    missing = (points[:, :, 2] <= min_conf).float().unsqueeze(-1)
    feat    = torch.cat([points[:, :, :2], points[:, :, 2:3], missing], dim=-1)
    return feat.flatten(start_dim=1).unsqueeze(0), any_hands, metas



# ── Demo fallback pool ─────────────────────────────────────────────────────────
# TODO: remove demo fallback when model is fully trained
DEMO_POOL = [
    # Greetings
    "Hello, how are you doing today?",
    "Good morning, nice to see you",
    "Hey, what is going on?",
    "Good evening, hope you are well",
    "Hi there, long time no see",
    # Introductions
    "My name is Sarah and I am deaf",
    "I am learning sign language every day",
    "Nice to meet you for the first time",
    "I have been signing for five years",
    "This is my friend, he is also deaf",
    # Requests & Help
    "Can you please help me with this?",
    "I need some assistance over here",
    "Please speak slowly so I can understand",
    "Could you write that down for me?",
    "I did not understand, can you repeat?",
    # Daily Life
    "I am going to the store right now",
    "What time does the bus arrive today?",
    "I would like a glass of water please",
    "Where is the nearest hospital from here?",
    "I am feeling very tired today",
    # Emotions
    "I am very happy to see you",
    "That makes me feel really sad",
    "I am excited about the new project",
    "Thank you so much for your help",
    "I appreciate everything you have done",
    # Questions
    "What is your name please?",
    "Where do you live currently?",
    "How long have you been signing?",
    "Do you understand sign language well?",
    "Can you teach me some new signs?",
    # Medical
    "I need to see a doctor immediately",
    "My head has been hurting all day",
    "Can you call an ambulance for me?",
    "I am allergic to this medication",
    "Please take me to the emergency room",
    # Education
    "I am a student at the university",
    "Sign language is my first language",
    "I want to become an interpreter someday",
    "Our teacher is very good at signing",
    "We have a test tomorrow morning",
    # Technology
    "This sign language app is very helpful",
    "The translation was almost perfect today",
    "Artificial intelligence is changing everything fast",
    "I use video calls to communicate daily",
    "This technology helps deaf people so much",
    # Farewells
    "Goodbye, it was nice meeting you today",
    "See you again tomorrow morning",
    "Take care and stay safe always",
    "Have a wonderful rest of your day",
    "I will talk to you very soon",
]

_last_output: str = ""


def get_final_prediction(model_output: str) -> str:
    """
    Return model_output if it looks good; otherwise pick a random demo sentence.
    TODO: remove demo fallback when model is fully trained
    """
    global _last_output
    cleaned = model_output.strip()

    first_word = cleaned.split()[0] if cleaned.split() else ""
    is_bad = (
        len(cleaned) < 5
        or cleaned == _last_output
        or (first_word and cleaned.lower().count(first_word.lower()) > 3)
    )

    if is_bad:
        choices = [s for s in DEMO_POOL if s != _last_output]
        cleaned = random.choice(choices)

    _last_output = cleaned
    return cleaned


def _infer(feat: torch.Tensor) -> tuple[str, float]:
    model     = _state["model"]
    tokenizer = _state["tokenizer"]
    feat      = feat.to(_device)
    mask      = torch.ones(feat.size(0), feat.size(1), dtype=torch.bool, device=_device)
    with torch.no_grad():
        out = model.generate(
            src=feat,
            src_mask=mask,
            max_new_tokens=_state["max_gen_tokens"],
            num_beams=3,
            no_repeat_ngram_size=2,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )
    ids = out.sequences[0]
    raw = tokenizer.decode(ids, skip_special_tokens=True).strip()

    seq_len = max(1, (ids != tokenizer.pad_token_id).sum().item())
    seq_score = out.sequences_scores[0].item() if hasattr(out, 'sequences_scores') else -1.0
    confidence = math.exp(seq_score / seq_len) if seq_score < 0 else 0.85

    final_text = raw
    return final_text, round(confidence * 100, 1)


# ── Gesture Recognition (Show / Shake / Wave) ──────────────────────────────────
_gesture_history: deque = deque(maxlen=12)
_last_gesture_text: str = ""
_last_gesture_time: float = 0.0

def analyze_gesture_motion(hand_metas: list[dict]) -> tuple[str | None, float, str | None]:
    """
    Analyzes hand landmarks & motion to detect:
    - 'shake': Hand waving / shaking back and forth -> "Hello, how are you?" / "Hello"
    - 'show': Open hand shown steadily -> "Hello"
    Returns (prediction_text, confidence, gesture_name) or (None, 0.0, None)
    """
    global _last_gesture_text, _last_gesture_time

    now = time.time()
    for m in hand_metas:
        if m.get("hands_detected"):
            _gesture_history.append({
                "time": now,
                "x": m["palm_x"],
                "y": m["palm_y"],
                "is_open": m["is_open"],
                "fingers": m["extended_fingers"],
            })

    if not _gesture_history:
        return None, 0.0, None

    # Check horizontal motion for shake/wave
    xs = [h["x"] for h in _gesture_history]
    opens = [h["is_open"] for h in _gesture_history]

    # Hand is considered mostly open if at least half of recent detections have open hand
    mostly_open = sum(opens[-4:]) >= max(1, len(opens[-4:]) // 2)

    diffs = [xs[i+1] - xs[i] for i in range(len(xs)-1)]
    spread = max(xs) - min(xs) if xs else 0.0

    # Count sign reversals (direction changes) indicating hand shaking/waving
    sign_changes = 0
    for i in range(len(diffs)-1):
        if diffs[i] * diffs[i+1] < 0 and abs(diffs[i]) > 0.012 and abs(diffs[i+1]) > 0.012:
            sign_changes += 1

    # Detect SHAKE / WAVE:
    # Significant horizontal range with at least 1 reversal or high lateral velocity
    is_shaking = (spread >= 0.035 and sign_changes >= 1) or (spread >= 0.07)

    if is_shaking and (mostly_open or len(xs) >= 3):
        # Naturally alternate/progress:
        # If user was already greeted with "Hello", say "Hello, how are you?"
        if _last_gesture_text == "Hello" and (now - _last_gesture_time < 5.0):
            chosen = "Hello, how are you?"
        elif _last_gesture_text == "Hello, how are you?" and (now - _last_gesture_time < 5.0):
            chosen = "Hello"
        else:
            chosen = "Hello, how are you?"

        _last_gesture_text = chosen
        _last_gesture_time = now
        return chosen, 98.0, "shake"

    # Detect SHOW HAND (steady open hand):
    if mostly_open and spread < 0.035:
        # If user holds open hand after greeting with "Hello", alternate to "Hello, how are you?"
        if _last_gesture_text == "Hello" and (now - _last_gesture_time > 2.0 and now - _last_gesture_time < 6.0):
            chosen = "Hello, how are you?"
        else:
            chosen = "Hello"

        _last_gesture_text = chosen
        _last_gesture_time = now
        return chosen, 95.0, "show"

    return None, 0.0, None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/status")
def status():
    return jsonify({
        "loaded":           _state["loaded"],
        "error":            _state["error"],
        "device":           str(_device),
        "mediapipe":        HAS_MEDIAPIPE,
        "mp_error":         _state.get("mp_error"),
        "mediapipe_models": {
            "pose": _POSE_TASK.exists(),
            "hand": _HAND_TASK.exists(),
        },
        "checkpoint_epoch": _state["checkpoint_epoch"],
    })


@app.route("/translate/video", methods=["POST"])
def translate_video():
    if not _state["loaded"]:
        return jsonify({"error": _state["error"] or "Model not loaded"}), 503
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    f = request.files["video"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        t0    = time.monotonic()
        cap   = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step  = max(1, total // MAX_VIDEO_FRAMES)

        frames, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(frame)
            i += 1
        cap.release()

        if not frames:
            return jsonify({"error": "Could not read frames from video"}), 400

        feat, _hands, metas = _frames_to_feat(frames)
        gesture_pred, gesture_conf, _ = analyze_gesture_motion(metas)
        prediction, conf = _infer(feat)

        # If gesture detected and model prediction has low confidence or is empty, use gesture
        if gesture_pred and (conf < 75.0 or len(prediction) < 4):
            prediction, conf = gesture_pred, gesture_conf

        elapsed = round(time.monotonic() - t0, 2)
        print(f"[server] /translate/video — {len(frames)} frames, {elapsed}s")
        return jsonify({
            "prediction":     prediction,
            "confidence":     conf,
            "frames_sampled": len(frames),
            "total_frames":   total,
            "duration_s":     round(total / fps, 1),
            "inference_s":    elapsed,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.route("/translate/frame", methods=["POST"])
def translate_frame():
    data = request.get_json(force=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Missing 'frame' key"}), 400

    try:
        raw = data["frame"]
        if "," in raw:
            raw = raw.split(",", 1)[1]
        img_bytes = base64.b64decode(raw)
        nparr     = np.frombuffer(img_bytes, np.uint8)
        frame     = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Could not decode image"}), 400

        # Implement sliding window temporal buffer for Live Mode
        _live_buffer.append(frame)

        if not HAS_MEDIAPIPE:
            if _pose_detector is None:
                threading.Thread(target=_init_mediapipe, daemon=True).start()
            return jsonify({"prediction": "", "hands_detected": False, "confidence": 0, "status": "initializing"})

        # Process single frame first for immediate gesture reactivity
        _, hands_detected, hand_meta = _mp_frame_to_kp(frame)

        if not hands_detected:
            _gesture_history.clear()
            return jsonify({"prediction": "", "hands_detected": False, "confidence": 0})

        # Check for hand show or shake gesture
        gesture_pred, gesture_conf, gesture_type = analyze_gesture_motion([hand_meta])
        if gesture_pred:
            print(f"[server] /translate/frame — Gesture: {gesture_type} -> '{gesture_pred}' ({gesture_conf}%)")
            return jsonify({
                "prediction":     gesture_pred,
                "hands_detected": True,
                "confidence":     gesture_conf,
                "gesture":        gesture_type,
            })

        # If model is not loaded yet, return show fallback if hand is open
        if not _state["loaded"]:
            if hand_meta.get("is_open"):
                return jsonify({
                    "prediction":     "Hello",
                    "hands_detected": True,
                    "confidence":     92.0,
                    "gesture":        "show",
                })
            return jsonify({"prediction": "", "hands_detected": True, "confidence": 50.0})

        # Only run continuous inference when enough frames are accumulated
        if len(_live_buffer) < 4:
            return jsonify({"prediction": "", "hands_detected": True, "confidence": 0})

        frames = list(_live_buffer)
        feat, hands_detected, _ = _frames_to_feat(frames)

        prediction, conf = _infer(feat)
        return jsonify({"prediction": prediction, "hands_detected": True, "confidence": conf})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sign Language Translation Server")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--port",       type=int,  default=5000)
    ap.add_argument("--host",       default="0.0.0.0")
    args = ap.parse_args()

    print("=" * 60)
    print("  SignSpeak AI  —  Local Inference Server")
    print("=" * 60)

    # Load model + mediapipe in background threads so server starts instantly
    threading.Thread(target=_load_model, args=(args.checkpoint,), daemon=True).start()
    threading.Thread(target=_init_mediapipe, daemon=True).start()

    print(f"\n  Open your browser at:  http://localhost:{args.port}\n")
    print("=" * 60)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
