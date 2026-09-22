"""
Audio/video transcription using faster-whisper — runs entirely locally,
no API cost, no internet needed once the model is downloaded once.
"""
import os

# Allow Hugging Face to download models when a token is configured.
if os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"

from faster_whisper import WhisperModel

# "base" is a good balance of speed/accuracy for a hackathon/demo on a laptop CPU.
# Options (fastest to most accurate): tiny, base, small, medium, large-v3
_model = WhisperModel("base", device="cpu", compute_type="int8")


def transcribe_audio(file_path: str) -> dict:
    """
    Transcribes an audio/video file. Returns the full text plus timestamped
    segments (useful later if you want a "jump to this moment" UI feature).
    """
    segments, info = _model.transcribe(file_path, beam_size=5)

    segment_list = []
    full_text_parts = []
    for seg in segments:
        segment_list.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        full_text_parts.append(seg.text.strip())

    return {
        "text": " ".join(full_text_parts),
        "segments": segment_list,
        "language": info.language,
        "duration": round(info.duration, 2),
    }
