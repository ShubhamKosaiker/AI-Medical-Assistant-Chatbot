"""
voice_of_the_doctor.py — Text-to-Speech Synthesis
---------------------------------------------------
Handles the "voice" of the AI doctor — converts the LLM's text response
into a spoken MP3 file that plays in the Gradio audio component.

Two TTS engines are available:
  • gTTS  (Google Text-to-Speech) — free, no API key required, used by default.
  • ElevenLabs — premium, highly natural voice, requires ELEVENLABS_API_KEY.

To switch to ElevenLabs, replace text_to_speech_with_gtts with
text_to_speech_with_elevenlabs in app.py's process_inputs function.

Audio playback (afplay / aplay) is only triggered in local development.
When deployed on HuggingFace Spaces or Docker, Gradio's audio component
streams the file to the browser — no server-side playback is needed.

External dependencies: gtts, elevenlabs (pip install as above)
"""

import logging
import os
import platform
import subprocess

from gtts import gTTS

from config import ELEVENLABS_OUTPUT_FORMAT, ELEVENLABS_VOICE_ID, TTS_ELEVENLABS_MODEL

logger = logging.getLogger(__name__)


# ── Local Playback Helper ─────────────────────────────────────────────────────

def _play_audio(output_filepath: str) -> None:
    """Play an audio file using the platform's native audio tool.

    This is a developer convenience — it lets you hear the TTS output when
    running the app locally via CLI. Errors here are non-fatal; the file
    has already been saved and will be served by Gradio regardless.

    Platform mapping:
      • macOS   → afplay  (built-in, no install needed)
      • Linux   → aplay   (ALSA; alternative: mpg123 or ffplay)
      • Windows → skipped (Gradio's browser player handles playback)
    """
    os_name = platform.system()
    try:
        if os_name == "Darwin":   # macOS
            subprocess.run(["afplay", output_filepath], check=True)
        elif os_name == "Linux":
            subprocess.run(["aplay", output_filepath], check=True)
        # Windows: audio is delivered via Gradio's UI — no local autoplay needed.
    except Exception as e:
        logger.error("Local audio playback failed: %s", e)


# ── Google Text-to-Speech (Free) ──────────────────────────────────────────────

def text_to_speech_with_gtts(input_text: str, output_filepath: str) -> str:
    """Convert text to speech using Google TTS and save the result as an MP3.

    gTTS is the default engine. It's free, requires no API key, and produces
    clear (if slightly robotic) speech. It works out of the box on HuggingFace
    Spaces and Docker without any extra credentials.

    Args:
        input_text:      The doctor's diagnosis text to be spoken.
        output_filepath: Path where the MP3 file will be written.

    Returns:
        output_filepath — passed back so Gradio can serve the file directly.
    """
    logger.info("Generating speech with gTTS (%d characters).", len(input_text))

    # gTTS streams audio from Google's servers and writes it to disk.
    audioobj = gTTS(text=input_text, lang="en", slow=False)
    audioobj.save(output_filepath)

    # Attempt local playback (no-op on Windows / HuggingFace).
    _play_audio(output_filepath)

    return output_filepath


# ── ElevenLabs Text-to-Speech (Premium) ──────────────────────────────────────

def text_to_speech_with_elevenlabs(input_text: str, output_filepath: str) -> str:
    """Convert text to speech using ElevenLabs and save the result as an MP3.

    ElevenLabs produces significantly more natural, human-like speech than gTTS.
    Requires a paid ElevenLabs account and the ELEVENLABS_API_KEY environment
    variable to be set. Voice characteristics are controlled by ELEVENLABS_VOICE_ID
    in config.py.

    Args:
        input_text:      The doctor's diagnosis text to be spoken.
        output_filepath: Path where the MP3 file will be written.

    Returns:
        output_filepath — passed back so Gradio can serve the file directly.
    """
    # Lazy import: only loads the ElevenLabs SDK when this function is actually called,
    # so the app doesn't fail at startup if elevenlabs isn't installed.
    from elevenlabs import save
    from elevenlabs.client import ElevenLabs

    logger.info("Generating speech with ElevenLabs (%d characters).", len(input_text))

    client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))

    # convert() returns a generator of audio bytes; save() streams it to disk.
    audio = client.text_to_speech.convert(
        text=input_text,
        voice_id=ELEVENLABS_VOICE_ID,           # Which voice clone to use.
        model_id=TTS_ELEVENLABS_MODEL,          # eleven_turbo_v2 = fast + high quality.
        output_format=ELEVENLABS_OUTPUT_FORMAT, # mp3_44100_128 = CD quality MP3.
    )
    save(audio, output_filepath)

    # Attempt local playback (no-op on Windows / HuggingFace).
    _play_audio(output_filepath)

    return output_filepath
