"""
app.py — AI Medical Assistant Chatbot (Main Entry Point)
---------------------------------------------------------
Orchestrates all components and serves the Gradio web interface.

Data flow for each consultation:
  1. Patient speaks symptoms  →  Gradio captures audio as a temp file.
  2. (Optional) Patient uploads a medical image.
  3. process_inputs() is called:
       a. Whisper (via Groq) transcribes the audio to text.
       b. LLaMA 4 Scout (via Groq) analyses image + text and returns a diagnosis.
       c. gTTS converts the diagnosis text to an MP3 file.
  4. Gradio displays the transcription, diagnosis, and plays the audio.
  5. The exchange is appended to the session's conversation history (gr.Chatbot).

Run locally:  python app.py
Deploy:       push to HuggingFace Spaces — Gradio serves on port 7860 automatically.
Docker:       docker compose up --build
"""

import logging
import os
import tempfile

import gradio as gr
from dotenv import load_dotenv

from brain_of_the_doctor import analyze_image_with_query, encode_image
from config import MAX_TRANSCRIPTION_CHARS, STT_MODEL, SYSTEM_PROMPT, VISION_MODEL
from voice_of_the_doctor import text_to_speech_with_gtts
from voice_of_the_patient import transcribe_with_groq

# ── Logging Setup ─────────────────────────────────────────────────────────────
# Configures structured logging for all modules (brain, voice_patient, voice_doctor).
# INFO level shows normal operation; swap to DEBUG for full request/response detail.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Environment Validation ────────────────────────────────────────────────────
# Load variables from the .env file (GROQ_API_KEY, ELEVENLABS_API_KEY, etc.).
# dotenv silently skips if .env doesn't exist — HuggingFace injects secrets directly.
load_dotenv()

# Keys that MUST be set before the app can function.
_REQUIRED_ENV_KEYS = ["GROQ_API_KEY"]


def _validate_env() -> None:
    """Crash early with a clear message if required API keys are missing.

    Without this check, the app would start, accept user input, and then fail
    with a cryptic AuthenticationError deep inside the Groq SDK — confusing to debug.
    """
    missing = [k for k in _REQUIRED_ENV_KEYS if not os.environ.get(k)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}.\n"
            "  → Create a '.env' file in the project root with these keys.\n"
            "  → Example:  GROQ_API_KEY=gsk_xxxxxxxxxxxx"
        )


# Validate immediately at startup — before Gradio even launches.
_validate_env()


# ── Core Processing Pipeline ──────────────────────────────────────────────────

def process_inputs(audio_filepath: str, image_filepath: str, history: list):
    """Run the full STT → LLM → TTS pipeline for one patient consultation.

    This function is wired to the Gradio Submit button. Gradio calls it with
    the current values of all input components and expects a tuple matching
    the output components defined in submit_btn.click().

    Args:
        audio_filepath: Temp file path to the patient's recorded speech (MP3/WAV).
                        Gradio writes this automatically when the user records audio.
        image_filepath: Temp file path to an uploaded medical image, or None.
        history:        Current chatbot message list (list of role/content dicts).
                        Passed in by gr.Chatbot as both input and output.

    Returns:
        4-tuple: (transcribed_text, doctor_response, audio_path, updated_history)
        Each element maps to one Gradio output component.
    """
    try:
        # Guard: no audio means nothing to process.
        if not audio_filepath:
            return "No audio provided.", "Please record your symptoms first.", None, history

        # ── Step 1: Speech → Text ─────────────────────────────────────────────
        # Whisper converts the patient's voice recording to a text string.
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=audio_filepath,
            stt_model=STT_MODEL,
        )

        # Truncate if unreasonably long (e.g., user left mic on for minutes).
        if len(speech_to_text_output) > MAX_TRANSCRIPTION_CHARS:
            speech_to_text_output = speech_to_text_output[:MAX_TRANSCRIPTION_CHARS]
            logger.warning("Transcription truncated to %d chars.", MAX_TRANSCRIPTION_CHARS)

        # ── Step 2: Image + Text → Diagnosis ─────────────────────────────────
        if image_filepath:
            # Encode the image to base64 (validates size and type as a side effect).
            encoded = encode_image(image_filepath)
            # Prepend the system prompt to the patient's speech so the model
            # understands both the role it should play and what the patient said.
            doctor_response = analyze_image_with_query(
                query=SYSTEM_PROMPT + speech_to_text_output,
                encoded_image=encoded,
                model=VISION_MODEL,
            )
        else:
            # No image: provide a helpful nudge rather than a blank response.
            doctor_response = (
                "No image was provided. Based on your description alone: "
                "please share a photo for a more accurate assessment."
            )

        # ── Step 3: Text → Speech ─────────────────────────────────────────────
        # NamedTemporaryFile creates a unique file on disk; delete=False keeps
        # it alive after closing so Gradio can read and serve it to the browser.
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        voice_of_doctor = text_to_speech_with_gtts(
            input_text=doctor_response,
            output_filepath=tmp_path,
        )

        # ── Step 4: Update Conversation History ───────────────────────────────
        # Append this exchange to the chatbot history in OpenAI messages format.
        # gr.Chatbot renders "user" messages on the left, "assistant" on the right.
        updated_history = history + [
            {"role": "user",      "content": speech_to_text_output},
            {"role": "assistant", "content": doctor_response},
        ]

        return speech_to_text_output, doctor_response, voice_of_doctor, updated_history

    except ValueError as e:
        # Raised by encode_image() for bad file type or oversized images.
        logger.warning("Input validation error: %s", e)
        return "", f"Input error: {e}", None, history

    except Exception as e:
        # Catch-all for API failures, network errors, etc.
        # Log the full traceback for debugging, but show a clean message in the UI.
        logger.error("Unexpected error during processing: %s", e, exc_info=True)
        return "", "Something went wrong. Please try again in a moment.", None, history


# ── Custom CSS ────────────────────────────────────────────────────────────────
# Dark theme inspired by Catppuccin Mocha colour palette.
# Injected into Gradio via gr.Blocks(css=...) — overrides default Gradio styles.
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600&family=Inter:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    background: #0d1117 !important;
    font-family: 'Inter', sans-serif !important;
    color: #cdd6f4 !important;
    min-height: 100vh !important;
}
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 1.5rem 3rem !important;
}

/* ── Header ── */
.app-header { text-align: center; padding: 3rem 0 2.5rem; margin-bottom: 2rem; border-bottom: 1px solid rgba(255,255,255,0.07); }
.app-header h1 { font-family: 'Syne', sans-serif !important; font-size: 3rem !important; font-weight: 600 !important; color: #ffffff !important; letter-spacing: -1px; line-height: 1.1; margin-bottom: 0.6rem !important; }
.app-header .subtitle { color: #6c7a8d; font-size: 0.9rem; font-weight: 300; letter-spacing: 0.3px; }
.status-badge { display: inline-flex; align-items: center; gap: 6px; background: rgba(62,207,142,0.1); border: 1px solid rgba(62,207,142,0.25); border-radius: 20px; padding: 4px 14px; font-size: 0.75rem; color: #3ecf8e; margin-top: 1rem; }
.dot { width: 6px; height: 6px; background: #3ecf8e; border-radius: 50%; animation: blink 2s infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* ── Panel labels ── */
.panel-label { font-size: 0.68rem !important; font-weight: 500 !important; letter-spacing: 2.5px !important; text-transform: uppercase !important; color: #89b4fa !important; margin-bottom: 1rem !important; padding-bottom: 0.5rem !important; border-bottom: 1px solid rgba(137,180,250,0.15) !important; display: block !important; }

/* ── Component cards ── */
.gr-audio-container, .gr-image-container, .gr-audio, .gr-image,
[data-testid="audio"], [data-testid="image"] { background: #161b27 !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; overflow: hidden !important; }

/* ── Textboxes ── */
.gr-textbox, [data-testid="textbox"] { background: #161b27 !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; overflow: hidden !important; }
textarea { background: #161b27 !important; border: none !important; border-radius: 0 !important; color: #e2e8f0 !important; font-family: 'Inter', sans-serif !important; font-size: 0.92rem !important; line-height: 1.7 !important; padding: 1rem 1.1rem !important; resize: none !important; caret-color: #89b4fa !important; }
textarea::placeholder { color: #3d4a5c !important; }
label span, .gr-block label, .block label span { color: #89b4fa !important; font-size: 0.8rem !important; font-weight: 500 !important; letter-spacing: 0.5px !important; }

/* ── Analyze button ── */
#analyze-btn { background: #89b4fa !important; border: none !important; border-radius: 12px !important; color: #0d1117 !important; font-family: 'Syne', sans-serif !important; font-weight: 600 !important; font-size: 0.95rem !important; letter-spacing: 0.5px !important; padding: 0.85rem 2rem !important; cursor: pointer !important; width: 100% !important; transition: background 0.2s, transform 0.1s !important; margin-top: 0.5rem !important; }
#analyze-btn:hover { background: #b4ceff !important; transform: translateY(-1px) !important; }
#analyze-btn:active { transform: translateY(0) !important; }

/* ── Conversation history / chatbot ── */
.chatbot-panel { margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.07); }
.message.user .bubble-wrap { background: #1e2535 !important; }
.message.bot  .bubble-wrap { background: #162032 !important; }

/* ── Clear button ── */
#clear-btn { background: transparent !important; border: 1px solid rgba(255,255,255,0.12) !important; border-radius: 8px !important; color: #6c7a8d !important; font-size: 0.8rem !important; padding: 0.4rem 1rem !important; cursor: pointer !important; transition: border-color 0.2s, color 0.2s !important; }
#clear-btn:hover { border-color: rgba(255,255,255,0.25) !important; color: #cdd6f4 !important; }

/* ── Audio output waveform ── */
.gr-audio [data-testid="waveform"], .gr-audio .waveform-container { background: #1e2535 !important; }

/* ── Footer ── */
.app-footer { text-align: center; margin-top: 2.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.05); color: #2d3748; font-size: 0.75rem; letter-spacing: 0.3px; }
"""

# ── Gradio UI Layout ──────────────────────────────────────────────────────────
# gr.Blocks gives full layout control (vs gr.Interface which is one-column only).
with gr.Blocks(title="AI Doctor Assistant") as iface:

    # Header — pure HTML injected into the page.
    gr.HTML("""
        <div class="app-header">
            <h1>🩺 AI Doctor Assistant</h1>
            <p class="subtitle">Voice powered · AI diagnostics · Instant medical insights</p>
            <div class="status-badge">
                <span class="dot"></span>
                Groq &nbsp;·&nbsp; LLaMA 4 Scout &nbsp;·&nbsp; Whisper
            </div>
        </div>
    """)

    # ── Main two-column consultation row ──────────────────────────────────────
    with gr.Row(equal_height=False):

        # Left column: patient inputs
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<span class="panel-label">📋 &nbsp; Your Input</span>')

            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",         # Gradio writes audio to a temp file and passes the path.
                label="Speak your symptoms",
            )
            image_input = gr.Image(
                type="filepath",         # Same pattern — temp file path passed to process_inputs.
                label="Upload an image (optional)",
            )
            submit_btn = gr.Button(
                "Analyze →",
                variant="primary",
                elem_id="analyze-btn",   # CSS targets this ID for button styling.
            )

        # Right column: AI doctor outputs
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<span class="panel-label">💬 &nbsp; Consultation</span>')

            transcription_out = gr.Textbox(
                label="What you said",
                lines=3,
                interactive=False,       # Read-only — populated by process_inputs.
                placeholder="Your transcribed speech appears here...",
            )
            response_out = gr.Textbox(
                label="Doctor's diagnosis",
                lines=7,
                interactive=False,
                placeholder="The AI doctor's response appears here...",
            )
            audio_out = gr.Audio(
                label="Doctor's voice",
                interactive=False,       # Playback only — user can't upload here.
            )

    # ── Conversation history row ──────────────────────────────────────────────
    # Shows all consultations from the current session as a chat thread.
    with gr.Row():
        with gr.Column():
            gr.HTML('<div class="chatbot-panel"><span class="panel-label">🗂 &nbsp; Session History</span></div>')

            # gr.Chatbot stores and displays the conversation history.
            # Gradio 6+ natively uses the {role, content} message dict format —
            # compatible with standard chat APIs and easy to extend.
            chatbot = gr.Chatbot(
                label="",
                height=320,
                show_label=False,
                # Stethoscope emoji as the doctor's avatar in the chat thread.
                avatar_images=(
                    None,
                    "https://em-content.zobj.net/source/twitter/376/stethoscope_1fa7a.png",
                ),
            )
            clear_btn = gr.Button("Clear History", elem_id="clear-btn", size="sm")

    # ── Event Wiring ──────────────────────────────────────────────────────────
    # Submit button → runs process_inputs → updates all 4 output components.
    # chatbot is passed as BOTH input (existing history) and output (updated history).
    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input, chatbot],
        outputs=[transcription_out, response_out, audio_out, chatbot],
    )

    # Clear button → resets chatbot to an empty list (no args needed).
    clear_btn.click(fn=lambda: [], outputs=[chatbot])

    # Disclaimer footer
    gr.HTML("""
        <div class="app-footer">
            ⚠️ For educational purposes only &nbsp;·&nbsp; Not a substitute for professional medical advice
        </div>
    """)

# ── Launch ────────────────────────────────────────────────────────────────────
# Guard ensures launch() is only called when running the script directly,
# NOT when the module is imported by pytest or other tools.
# server_name="0.0.0.0" binds to all interfaces — required for Docker and
# HuggingFace Spaces. css= is passed to launch() in Gradio 6+.
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860, css=custom_css)
