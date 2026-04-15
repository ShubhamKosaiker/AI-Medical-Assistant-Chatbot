# ── AI Medical Assistant Chatbot — Docker Image ───────────────────────────────
#
# Build:  docker build -t ai-medical-chatbot .
# Run:    docker compose up --build
#
# The image installs FFmpeg at the system level so pydub can encode audio
# without any additional configuration. Port 7860 is Gradio's default and
# also what HuggingFace Spaces expects.

FROM python:3.11-slim

# Install system-level dependencies:
#   ffmpeg         — required by pydub to encode/decode audio (MP3, WAV)
#   portaudio19-dev — required to compile PyAudio (microphone access)
#   gcc, python3-dev — build tools needed to compile PyAudio's C extension
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        portaudio19-dev \
        gcc \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first — Docker caches this layer
# separately so rebuilds are fast when only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application source files — test files, .env, and media
# are excluded via .dockerignore.
COPY config.py \
     app.py \
     brain_of_the_doctor.py \
     voice_of_the_patient.py \
     voice_of_the_doctor.py \
     ./

# Expose the port Gradio listens on.
EXPOSE 7860

# Run as a non-root user — best practice for container security.
RUN useradd --create-home appuser
USER appuser

# Start the Gradio app.
CMD ["python", "app.py"]
