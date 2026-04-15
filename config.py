"""
config.py — Central configuration for AI Medical Assistant Chatbot
------------------------------------------------------------------
All constants, model identifiers, limits, and the system prompt live here.
Changing a value in one place updates the entire application — no hunting
through multiple files to find a hardcoded string.
"""

# ── AI Model Identifiers ──────────────────────────────────────────────────────
# Groq-hosted LLaMA 4 Scout: handles both the vision (image analysis) and
# language (diagnosis generation) tasks in a single API call.
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# Groq-hosted Whisper: converts patient's spoken audio to text (speech-to-text).
STT_MODEL = "whisper-large-v3"

# ElevenLabs model for premium, natural-sounding voice synthesis.
TTS_ELEVENLABS_MODEL = "eleven_turbo_v2"

# ElevenLabs voice clone ID — the specific voice used for the AI doctor.
# Replace with your own voice ID from the ElevenLabs dashboard if preferred.
ELEVENLABS_VOICE_ID = "JjpPU2Do2isL2c5DkxV2"

# Audio encoding format for ElevenLabs output (MP3, 44.1 kHz, 128 kbps).
ELEVENLABS_OUTPUT_FORMAT = "mp3_44100_128"


# ── Safety Limits ─────────────────────────────────────────────────────────────
# Maximum image file size accepted before rejecting the upload.
# Prevents accidental 50 MB screenshots from being sent to the API.
MAX_IMAGE_SIZE_MB = 10

# Maximum characters taken from the Whisper transcription before truncation.
# Guards against runaway long-form audio eating prompt tokens.
MAX_TRANSCRIPTION_CHARS = 2000

# Seconds to wait for a Groq API response before raising a timeout error.
# Prevents the UI from hanging indefinitely if the API is slow.
API_TIMEOUT_SECONDS = 30


# ── System Prompt ─────────────────────────────────────────────────────────────
# This is the instruction prepended to every LLM request.
# It shapes the tone, format, and style of the AI doctor's response.
# Key behaviours enforced:
#   - Speaks like a real doctor, not an AI chatbot
#   - No markdown, no bullet points, no numbered lists
#   - Concise (2 sentences max)
#   - References the image directly ("With what I see...")
SYSTEM_PROMPT = (
    "You have to act as a professional doctor. I know you are not, but this is a "
    "learning project. What's in this image? Do you find anything wrong with it "
    "medically? If you make a differential, suggest some remedies for them.If anything looks suspicious or wrong ask them to visit specialized doctor "
    "Do not add any numbers or special characters in your response. Your response "
    "should be in one long paragraph. Also always answer as if you are answering "
    "a real person. Don't say 'In the image I see' but say 'With what I see, I "
    "think you have ...' Don't respond as an AI model in markdown; your answer "
    "should mimic that of an actual doctor, not an AI bot. Keep your answer "
    "concise. No preamble — start your answer right away."
)
