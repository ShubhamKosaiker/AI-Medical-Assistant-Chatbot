#Voicebot UI using Gradio
import os
import gradio as gr
from brain_of_the_doctor import encode_image,analyze_image_with_query

from voice_of_the_patient import record_audio,transcribe_with_groq

from voice_of_the_doctor import text_to_speech_with_gtts,text_to_speech_with_elevenlabs


system_prompt=""" You have to act as a professional doctor,I know you are not but this is a learning project.
                What's in this image?. Do you find anything wrong with it medically?
                If you make differential, suggest some remedies for them. Do not add any numbers or special characters
                in your response. Your response should be in one long paragraph. Also always answer as you are answering
                to real person. Don't say 'In the image I see' but say 'With what i see, i think you have ...'
                Don't respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot,
                Keep your answer concise (max 2 sentences). No preamble ,start your answer right way please"""


def process_inputs(audio_filepath,image_filepath):
    speech_to_text_output=transcribe_with_groq( GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
                                               audio_filepath=audio_filepath,
                                               stt_model="whisper-large-v3"

    )

    # Handle the image input
    if image_filepath:
        doctor_response= analyze_image_with_query(query=system_prompt+speech_to_text_output,encoded_image=encode_image(image_filepath),model="meta-llama/llama-4-scout-17b-16e-instruct")
    else:
        doctor_response= "No image provided for me to analyze"

    voice_of_doctor= text_to_speech_with_elevenlabs(input_text=doctor_response,output_filepath="final.mp3")

    return speech_to_text_output,doctor_response,voice_of_doctor



# #Create the interface

# iface=gr.Interface(
#     fn=process_inputs,
#     inputs=[
#         gr.Audio(sources=["microphone"],type="filepath"),
#         gr.Image(type="filepath")
#     ],
#     outputs=[
#         gr.Textbox(label="Speech to Text"),
#         gr.Textbox(label="Doctor's Response"),
#         gr.Audio("Temp.mp3")

#     ],
#     title="AI Doctor"
# )

# iface.launch(debug=True)

# ── Custom CSS ──────────────────────────────────────────────────────────────
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
.app-header {
    text-align: center;
    padding: 3rem 0 2.5rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.app-header h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.6rem !important;
}
.app-header .subtitle {
    color: #6c7a8d;
    font-size: 0.9rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(62, 207, 142, 0.1);
    border: 1px solid rgba(62, 207, 142, 0.25);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    color: #3ecf8e;
    margin-top: 1rem;
}
.dot {
    width: 6px; height: 6px;
    background: #3ecf8e;
    border-radius: 50%;
    animation: blink 2s infinite;
}
@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* ── Panel labels ── */
.panel-label {
    font-size: 0.68rem !important;
    font-weight: 500 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: #89b4fa !important;
    margin-bottom: 1rem !important;
    padding-bottom: 0.5rem !important;
    border-bottom: 1px solid rgba(137,180,250,0.15) !important;
    display: block !important;
}

/* ── Component cards ── */
.gr-audio-container, .gr-image-container,
.gr-audio, .gr-image,
[data-testid="audio"], [data-testid="image"] {
    background: #161b27 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Textboxes ── */
.gr-textbox, [data-testid="textbox"] {
    background: #161b27 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}
textarea {
    background: #161b27 !important;
    border: none !important;
    border-radius: 0 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    line-height: 1.7 !important;
    padding: 1rem 1.1rem !important;
    resize: none !important;
    caret-color: #89b4fa !important;
}
textarea::placeholder { color: #3d4a5c !important; }

/* Label text above components */
label span, .gr-block label, .block label span {
    color: #89b4fa !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
}

/* ── Analyze button ── */
#analyze-btn {
    background: #89b4fa !important;
    border: none !important;
    border-radius: 12px !important;
    color: #0d1117 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    width: 100% !important;
    transition: background 0.2s, transform 0.1s !important;
    margin-top: 0.5rem !important;
}
#analyze-btn:hover {
    background: #b4ceff !important;
    transform: translateY(-1px) !important;
}
#analyze-btn:active { transform: translateY(0) !important; }

/* ── Audio output player ── */
.gr-audio [data-testid="waveform"],
.gr-audio .waveform-container {
    background: #1e2535 !important;
}

/* ── Footer ── */
.app-footer {
    text-align: center;
    margin-top: 2.5rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
    color: #2d3748;
    font-size: 0.75rem;
    letter-spacing: 0.3px;
}
"""

with gr.Blocks(css=custom_css, title="AI Doctor") as iface:

    gr.HTML("""
        <div class="app-header">
            <h1>🩺 AI Doctor</h1>
            <p class="subtitle">Describe your symptoms · Upload an image · Get an instant consultation</p>
            <div class="status-badge">
                <span class="dot"></span>
                Groq &nbsp;·&nbsp; ElevenLabs &nbsp;·&nbsp; LLaMA 4 Scout
            </div>
        </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<span class="panel-label">📋 &nbsp; Your Input</span>')
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Speak your symptoms"
            )
            image_input = gr.Image(
                type="filepath",
                label="Upload an image (optional)"
            )
            submit_btn = gr.Button(
                "Analyze →",
                variant="primary",
                elem_id="analyze-btn"
            )

        with gr.Column(scale=1, min_width=300):
            gr.HTML('<span class="panel-label">💬 &nbsp; Consultation</span>')
            transcription_out = gr.Textbox(
                label="What you said",
                lines=3,
                interactive=False,
                placeholder="Your transcribed speech appears here..."
            )
            response_out = gr.Textbox(
                label="Doctor's diagnosis",
                lines=7,
                interactive=False,
                placeholder="The AI doctor's response appears here..."
            )
            audio_out = gr.Audio(
                label="Doctor's voice",
                interactive=False
            )

    submit_btn.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[transcription_out, response_out, audio_out]
    )

    gr.HTML("""
        <div class="app-footer">
            ⚠️ For educational purposes only &nbsp;·&nbsp; Not a substitute for professional medical advice
        </div>
    """)

iface.launch(debug=True)