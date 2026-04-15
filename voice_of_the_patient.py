"""
voice_of_the_patient.py — Audio Recording & Speech-to-Text
------------------------------------------------------------
Handles the "ears" of the AI doctor:
  1. Records audio from the user's microphone and saves it as an MP3.
     (Used for the standalone CLI — Gradio handles recording in the web UI.)
  2. Transcribes the audio file to text using Groq's Whisper large-v3 model.

FFmpeg is required by pydub for MP3 encoding. This module auto-detects it on
the system PATH and common install locations so it works on Windows, macOS,
and Linux without manual configuration.

External dependencies: groq, speechrecognition, pydub (pip install as above)
"""

import logging
import os
import shutil
import sys
from io import BytesIO

import groq
import speech_recognition as sr
from groq import Groq
from pydub import AudioSegment
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import API_TIMEOUT_SECONDS, STT_MODEL

logger = logging.getLogger(__name__)


# ── FFmpeg Auto-Detection ─────────────────────────────────────────────────────

def _configure_ffmpeg() -> None:
    """Locate FFmpeg and point pydub at it.

    Strategy:
      1. Check whether 'ffmpeg' is already on the system PATH (the ideal case —
         installed via apt, brew, winget, etc.).
      2. If not, probe a list of common manual-install locations.
      3. If still not found, log a warning. pydub will raise its own error
         later if it actually needs FFmpeg to convert audio.

    This approach works across Windows, macOS, Linux, and HuggingFace Spaces
    without any hardcoded paths in production code.
    """
    # Fast path: ffmpeg is already discoverable — pydub finds it automatically.
    if shutil.which("ffmpeg"):
        logger.info("FFmpeg found on system PATH.")
        return

    # Suffix for Windows executables.
    exe = "ffmpeg" + (".exe" if sys.platform == "win32" else "")
    probe = "ffprobe" + (".exe" if sys.platform == "win32" else "")

    # Ordered list of common install directories to try.
    candidates = [
        r"C:\ffmpeg\bin",       # Windows: installer default (adds \bin)
        r"C:\ffmpeg",           # Windows: manual extract (no \bin subfolder)
        "/usr/local/bin",       # macOS: Homebrew (Intel)
        "/opt/homebrew/bin",    # macOS: Homebrew (Apple Silicon)
        "/usr/bin",             # Linux: apt/yum system install
    ]

    for directory in candidates:
        ffmpeg_bin = os.path.join(directory, exe)
        if os.path.isfile(ffmpeg_bin):
            # Add the directory to PATH so child processes (pydub) can find it.
            os.environ["PATH"] += os.pathsep + directory
            # Also set pydub's explicit converter paths for reliability.
            AudioSegment.converter = ffmpeg_bin
            AudioSegment.ffmpeg = ffmpeg_bin
            ffprobe_bin = os.path.join(directory, probe)
            if os.path.isfile(ffprobe_bin):
                AudioSegment.ffprobe = ffprobe_bin
            logger.info("FFmpeg configured from %s", ffmpeg_bin)
            return

    # Not found anywhere — warn the user but don't crash at import time.
    logger.warning(
        "FFmpeg not found. Install it and add it to PATH, or place it in "
        "C:\\ffmpeg (Windows) / /usr/local/bin (macOS) / /usr/bin (Linux). "
        "Audio conversion will fail if FFmpeg is required."
    )


# Run detection once at module import — subsequent imports reuse the result.
_configure_ffmpeg()


# ── Microphone Recording ──────────────────────────────────────────────────────

def record_audio(file_path: str, timeout: int = 20, phrase_time_limit: int = None) -> None:
    """Record audio from the default microphone and save it as an MP3.

    This function is used for the standalone CLI workflow. When the app is
    running in Gradio, the browser handles recording and passes a file path
    directly — this function is not called in that case.

    Args:
        file_path:        Where to save the recorded audio (must end in .mp3).
        timeout:          Seconds to wait for speech to begin before giving up.
        phrase_time_limit: Max seconds to record a single utterance.
    """
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        # Calibrate for background noise before listening (improves accuracy).
        logger.info("Calibrating for ambient noise — please wait...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        logger.info("Listening — speak your symptoms now.")

        # listen() blocks until silence is detected or phrase_time_limit is reached.
        audio_data = recognizer.listen(
            source, timeout=timeout, phrase_time_limit=phrase_time_limit
        )
        logger.info("Recording complete.")

    # Convert the raw WAV bytes captured by SpeechRecognition into an MP3 file.
    # pydub handles the WAV → MP3 transcoding via FFmpeg under the hood.
    wav_data = audio_data.get_wav_data()
    audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
    audio_segment.export(file_path, format="mp3", bitrate="128k")
    logger.info("Audio saved to %s", file_path)


# ── Speech-to-Text Transcription ──────────────────────────────────────────────

@retry(
    # Only retry on errors that might resolve on their own.
    # FileNotFoundError and AuthenticationError are not retried.
    retry=retry_if_exception_type((
        groq.APIConnectionError,  # Network blip — worth retrying immediately.
        groq.APITimeoutError,     # Slow response — back off and try again.
        groq.InternalServerError, # Groq 5xx — temporary server fault.
        groq.RateLimitError,      # Quota hit — exponential backoff helps here.
    )),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def transcribe_with_groq(audio_filepath: str, GROQ_API_KEY: str, stt_model: str = STT_MODEL) -> str:
    """Transcribe an MP3/WAV audio file to text using Groq Whisper.

    Decorated with @retry (tenacity) — automatically retries up to 3 times on
    transient errors with exponential backoff (2s, 4s, 8s).

    Whisper is an OpenAI-developed model hosted on Groq's fast inference
    infrastructure. It handles accents, background noise, and medical
    terminology better than browser-based speech APIs.

    Args:
        audio_filepath: Path to the audio file recorded by the patient.
        GROQ_API_KEY:   Groq API key (read from .env in app.py).
        stt_model:      Whisper model variant (default: whisper-large-v3).

    Returns:
        Plain-text string of what the patient said.
    """
    client = Groq(api_key=GROQ_API_KEY)
    logger.info("Transcribing '%s' with model '%s'.", audio_filepath, stt_model)

    # Open the audio file in binary mode and stream it directly to the API.
    # The context manager guarantees the file handle is closed after the call.
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en",   # Hint: English speeds up decoding slightly.
        )

    logger.info("Transcription complete — %d characters.", len(transcription.text))
    return transcription.text
