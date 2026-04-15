"""
test_voice_of_the_patient.py — Tests for speech-to-text transcription
----------------------------------------------------------------------
Tests cover:
  • transcribe_with_groq() — correct Groq client usage, API key forwarding,
    model selection, and error handling for missing files.

The Groq audio transcription API is mocked so no real API call is made.
"""

from unittest.mock import MagicMock, patch

import pytest

from voice_of_the_patient import transcribe_with_groq


class TestTranscribeWithGroq:
    """Tests for the transcribe_with_groq() function."""

    def test_returns_transcribed_text(self, sample_audio):
        """Happy path: the function should return the text from the Whisper response."""
        mock_transcription = MagicMock()
        mock_transcription.text = "I have had a persistent headache for two days."

        with patch("voice_of_the_patient.Groq") as MockGroq:
            MockGroq.return_value.audio.transcriptions.create.return_value = mock_transcription
            result = transcribe_with_groq(
                audio_filepath=sample_audio,
                GROQ_API_KEY="test-key",
            )

        assert result == "I have had a persistent headache for two days."

    def test_api_key_is_passed_to_groq_client(self, sample_audio):
        """The provided API key must be forwarded to the Groq() constructor."""
        mock_transcription = MagicMock()
        mock_transcription.text = "test"

        with patch("voice_of_the_patient.Groq") as MockGroq:
            MockGroq.return_value.audio.transcriptions.create.return_value = mock_transcription
            transcribe_with_groq(audio_filepath=sample_audio, GROQ_API_KEY="my-secret-key")

        # Groq() must be called with exactly this key.
        MockGroq.assert_called_once_with(api_key="my-secret-key")

    def test_uses_default_stt_model_from_config(self, sample_audio):
        """When no stt_model is given, STT_MODEL from config.py should be used."""
        from config import STT_MODEL

        mock_transcription = MagicMock()
        mock_transcription.text = "test"

        with patch("voice_of_the_patient.Groq") as MockGroq:
            mock_create = MockGroq.return_value.audio.transcriptions.create
            mock_create.return_value = mock_transcription
            transcribe_with_groq(audio_filepath=sample_audio, GROQ_API_KEY="key")
            _, kwargs = mock_create.call_args

        assert kwargs["model"] == STT_MODEL

    def test_custom_stt_model_is_forwarded_to_api(self, sample_audio):
        """An explicit stt_model argument must be passed through to the API call."""
        mock_transcription = MagicMock()
        mock_transcription.text = "test"

        with patch("voice_of_the_patient.Groq") as MockGroq:
            mock_create = MockGroq.return_value.audio.transcriptions.create
            mock_create.return_value = mock_transcription
            transcribe_with_groq(
                audio_filepath=sample_audio,
                GROQ_API_KEY="key",
                stt_model="whisper-large-v3-turbo",
            )
            _, kwargs = mock_create.call_args

        assert kwargs["model"] == "whisper-large-v3-turbo"

    def test_missing_audio_file_raises_file_not_found(self, tmp_path):
        """If the audio file doesn't exist on disk, FileNotFoundError should propagate."""
        with patch("voice_of_the_patient.Groq"):
            with pytest.raises(FileNotFoundError):
                transcribe_with_groq(
                    audio_filepath=str(tmp_path / "nonexistent.mp3"),
                    GROQ_API_KEY="key",
                )
