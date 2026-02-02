from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
import base64
import tempfile
import os
import librosa
import numpy as np

# =========================
# CONFIG
# =========================
API_KEY = "test123"
MAX_BASE64_SIZE = 1_000_000 # ~750KB (GUVI safe limit)

app = FastAPI(
    title="AI-Generated Voice Detection API",
    version="1.0"
)

# =========================
# ROOT (IMPORTANT FOR GUVI)
# =========================
@app.get("/")
def root():
    return {
        "status": "running",
        "service": "AI-Generated Voice Detection API",
        "docs": "/docs"
    }

# =========================
# REQUEST SCHEMA (GUVI FORMAT)
# =========================
class VoiceRequest(BaseModel):
    language: str
    audio_format: str
    audio_base64: str
    message: Optional[str] = None

# =========================
# BASE64 → TEMP AUDIO
# =========================
def decode_base64_audio(audio_base64: str, audio_format: str) -> str:
    if len(audio_base64) > MAX_BASE64_SIZE:
        raise ValueError("Audio too large for processing")

    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise ValueError("Invalid Base64 audio data")

    suffix = f".{audio_format.lower()}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        return tmp.name

# =========================
# AI vs HUMAN DETECTION
# =========================
def detect_ai_or_human(audio, sr):
    energy = np.mean(audio ** 2)
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    energy_n = min(energy * 10, 1.0)
    zcr_n = min(zcr * 10, 1.0)
    centroid_n = min(centroid / 4000, 1.0)

    ai_score = (
        0.4 * (1 - energy_n) +
        0.3 * (1 - zcr_n) +
        0.3 * (1 - centroid_n)
    )

    ai_score = round(ai_score, 2)

    if ai_score > 0.55:
        return (
            "AI_GENERATED",
            round(ai_score * 100, 1),
            "Over-smooth spectral patterns and low natural variation detected"
        )
    else:
        return (
            "HUMAN",
            round((1 - ai_score) * 100, 1),
            "Natural pitch variation and speech irregularities detected"
        )

# =========================
# MAIN ENDPOINT (GUVI)
# =========================
@app.post("/detect-voice")
def detect_voice(
    data: VoiceRequest,
    x_api_key: str = Header(None)
):
    # -------- AUTH --------
    if not x_api_key or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    audio_path = None

    try:
        audio_path = decode_base64_audio(
            data.audio_base64,
            data.audio_format
        )

        audio, sr = librosa.load(audio_path, sr=16000)

        if len(audio) < sr:
            raise ValueError("Audio too short")

        prediction, confidence, explanation = detect_ai_or_human(audio, sr)

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "language": data.language,
            "explanation": explanation
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)