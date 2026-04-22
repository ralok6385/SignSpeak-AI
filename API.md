# SignSpeak AI — REST API Reference

Base URL: `http://localhost:5000`

---

## `GET /health`

Health check endpoint.

**Response:**
```json
{ "status": "ok" }
```

---

## `GET /status`

Returns model loading status, device info, and MediaPipe readiness.

**Response:**
```json
{
  "loaded": true,
  "error": null,
  "device": "cuda",
  "mediapipe": true,
  "mediapipe_models": {
    "pose": true,
    "hand": true
  },
  "checkpoint_epoch": 15
}
```

---

## `POST /translate/video`

Upload a sign language video for translation.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `video` — video file (MP4, WebM, MOV — max 500 MB)

**Example:**
```bash
curl -X POST http://localhost:5000/translate/video \
  -F "video=@my_sign_video.mp4"
```

**Response (200):**
```json
{
  "prediction": "Hello, how are you doing today?",
  "confidence": 87.3,
  "frames_sampled": 48,
  "total_frames": 240,
  "duration_s": 8.0,
  "inference_s": 2.15
}
```

**Error (503 — model not loaded):**
```json
{ "error": "Model not loaded" }
```

---

## `POST /translate/frame`

Send a single webcam frame for live translation. The server maintains a sliding window buffer (16 frames) for temporal context.

**Request:**
- Content-Type: `application/json`
- Body:

```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/translate/frame \
  -H "Content-Type: application/json" \
  -d '{"frame": "data:image/jpeg;base64,/9j/4AAQ..."}'
```

**Response (200 — hands detected):**
```json
{
  "prediction": "Good morning, nice to see you",
  "hands_detected": true,
  "confidence": 82.1
}
```

**Response (200 — no hands visible):**
```json
{
  "prediction": "",
  "hands_detected": false,
  "confidence": 0
}
```

**Response (200 — buffering, need more frames):**
```json
{
  "prediction": "",
  "hands_detected": true
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `400` | Bad request (missing file, unreadable video) |
| `500` | Server error during inference |
| `503` | Model not yet loaded |

## Notes

- The `/translate/frame` endpoint uses a sliding window buffer (`deque(maxlen=16)`). The first 3 frames return empty predictions while the buffer fills.
- Inference only runs when hands are detected by MediaPipe. This avoids garbage predictions on empty frames.
- The `confidence` field is a percentage (0–100) derived from the T5 model's beam search sequence score.
