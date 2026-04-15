"""
test_app.py — Integration-level tests for the main processing pipeline
-----------------------------------------------------------------------
Tests cover process_inputs() — the central function wired to the Gradio
Submit button. All external I/O (Groq, gTTS, ElevenLabs) is mocked so
these tests run instantly without network access or API keys.

Test categories:
  • No audio provided    → early return with helpful message
  • With image           → full pipeline, history accumulation
  • Without image        → text-only fallback response
  • Error handling       → ValueError and unexpected exceptions
"""

from unittest.mock import patch

import pytest

from app import process_inputs


class TestNoAudioProvided:
    """process_inputs() should return early if no audio file is given."""

    def test_returns_four_tuple(self):
        result = process_inputs(None, None, [])
        assert len(result) == 4

    def test_audio_output_is_none(self):
        _, _, audio, _ = process_inputs(None, None, [])
        assert audio is None

    def test_history_is_unchanged(self):
        prior_history = [{"role": "user", "content": "prior"}]
        _, _, _, history = process_inputs(None, None, prior_history)
        assert history == prior_history

    def test_response_contains_helpful_message(self):
        transcription, response, _, _ = process_inputs(None, None, [])
        combined = (transcription + response).lower()
        assert "audio" in combined


class TestFullPipelineWithImage:
    """Full STT → LLM → TTS pipeline when both audio and image are provided."""

    def test_returns_correct_transcription(self, sample_audio, sample_image):
        with patch("app.transcribe_with_groq", return_value="I have a rash on my arm"), \
             patch("app.encode_image", return_value="base64img"), \
             patch("app.analyze_image_with_query", return_value="You have contact dermatitis."), \
             patch("app.text_to_speech_with_gtts", return_value="/tmp/response.mp3"):
            transcription, _, _, _ = process_inputs(sample_audio, sample_image, [])
        assert transcription == "I have a rash on my arm"

    def test_returns_correct_doctor_response(self, sample_audio, sample_image):
        with patch("app.transcribe_with_groq", return_value="My skin is red"), \
             patch("app.encode_image", return_value="img"), \
             patch("app.analyze_image_with_query", return_value="Possible eczema."), \
             patch("app.text_to_speech_with_gtts", return_value="/tmp/x.mp3"):
            _, response, _, _ = process_inputs(sample_audio, sample_image, [])
        assert response == "Possible eczema."

    def test_returns_audio_filepath(self, sample_audio, sample_image):
        with patch("app.transcribe_with_groq", return_value="text"), \
             patch("app.encode_image", return_value="img"), \
             patch("app.analyze_image_with_query", return_value="diagnosis"), \
             patch("app.text_to_speech_with_gtts", return_value="/tmp/audio.mp3"):
            _, _, audio, _ = process_inputs(sample_audio, sample_image, [])
        assert audio == "/tmp/audio.mp3"

    def test_history_gets_user_and_assistant_turns(self, sample_audio, sample_image):
        """Each consultation must add exactly one user turn and one assistant turn."""
        with patch("app.transcribe_with_groq", return_value="My knee hurts"), \
             patch("app.encode_image", return_value="img"), \
             patch("app.analyze_image_with_query", return_value="Possible tendinitis."), \
             patch("app.text_to_speech_with_gtts", return_value="/tmp/x.mp3"):
            _, _, _, history = process_inputs(sample_audio, sample_image, [])

        assert len(history) == 2
        assert history[0] == {"role": "user",      "content": "My knee hurts"}
        assert history[1] == {"role": "assistant",  "content": "Possible tendinitis."}

    def test_existing_history_is_preserved_and_extended(self, sample_audio, sample_image):
        """New turns must be appended — prior history must not be overwritten."""
        prior = [
            {"role": "user",      "content": "first question"},
            {"role": "assistant", "content": "first answer"},
        ]
        with patch("app.transcribe_with_groq", return_value="second question"), \
             patch("app.encode_image", return_value="img"), \
             patch("app.analyze_image_with_query", return_value="second answer"), \
             patch("app.text_to_speech_with_gtts", return_value="/tmp/x.mp3"):
            _, _, _, history = process_inputs(sample_audio, sample_image, prior)

        assert len(history) == 4
        assert history[0]["content"] == "first question"
        assert history[2]["content"] == "second question"


class TestPipelineWithoutImage:
    """When no image is provided, the app should return a text-only fallback."""

    def test_responds_without_image(self, sample_audio):
        with patch("app.transcribe_with_groq", return_value="I feel dizzy"), \
             patch("app.text_to_speech_with_gtts", return_value="/tmp/out.mp3"):
            transcription, response, audio, _ = process_inputs(sample_audio, None, [])

        assert transcription == "I feel dizzy"
        assert "no image" in response.lower()
        assert audio == "/tmp/out.mp3"

    def test_encode_image_is_not_called_without_image(self, sample_audio):
        """encode_image must never be called when image_filepath is None."""
        with patch("app.transcribe_with_groq", return_value="text"), \
             patch("app.encode_image") as mock_encode, \
             patch("app.text_to_speech_with_gtts", return_value="/tmp/x.mp3"):
            process_inputs(sample_audio, None, [])
        mock_encode.assert_not_called()


class TestErrorHandling:
    """process_inputs() must handle errors gracefully — no raw tracebacks in the UI."""

    def test_value_error_from_encode_image_shows_friendly_message(
        self, sample_audio, sample_image
    ):
        """Image validation errors (size, type) should surface as readable UI messages."""
        with patch("app.transcribe_with_groq", return_value="text"), \
             patch("app.encode_image", side_effect=ValueError("Image too large (12.3 MB)")):
            _, response, audio, _ = process_inputs(sample_audio, sample_image, [])

        assert "Input error" in response
        assert audio is None

    def test_unexpected_exception_shows_generic_message(self, sample_audio, sample_image):
        """Unexpected errors (network, API down) must never expose stack traces."""
        with patch("app.transcribe_with_groq", side_effect=RuntimeError("Groq API unavailable")):
            _, response, audio, _ = process_inputs(sample_audio, sample_image, [])

        assert "went wrong" in response.lower()
        assert audio is None

    def test_history_is_unchanged_on_any_error(self, sample_audio, sample_image):
        """A failed request must never corrupt the existing conversation history."""
        prior = [
            {"role": "user",      "content": "previous"},
            {"role": "assistant", "content": "previous answer"},
        ]
        with patch("app.transcribe_with_groq", side_effect=RuntimeError("boom")):
            _, _, _, history = process_inputs(sample_audio, sample_image, prior)

        assert history == prior
