"""
test_voice_of_the_doctor.py — Tests for text-to-speech synthesis
-----------------------------------------------------------------
Tests cover:
  • text_to_speech_with_gtts()      — gTTS calls, file saving, local playback.
  • text_to_speech_with_elevenlabs() — ElevenLabs API calls, config values,
                                       audio saving.

Both gTTS and the ElevenLabs SDK are mocked — no real API calls or audio
files are generated during testing.
"""

from unittest.mock import MagicMock, patch

import pytest

from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs


class TestTextToSpeechWithGtts:
    """Tests for the gTTS-based TTS function (default engine)."""

    def test_returns_output_filepath(self, tmp_path):
        """The function should return the path it was given so Gradio can serve the file."""
        output_path = str(tmp_path / "out.mp3")

        with patch("voice_of_the_doctor.gTTS") as MockGTTS, \
             patch("voice_of_the_doctor._play_audio"):
            MockGTTS.return_value = MagicMock()
            result = text_to_speech_with_gtts("Hello world", output_path)

        assert result == output_path

    def test_gtts_is_called_with_correct_arguments(self, tmp_path):
        """gTTS must receive the input text, English language, and normal speed."""
        output_path = str(tmp_path / "out.mp3")

        with patch("voice_of_the_doctor.gTTS") as MockGTTS, \
             patch("voice_of_the_doctor._play_audio"):
            mock_tts = MagicMock()
            MockGTTS.return_value = mock_tts
            text_to_speech_with_gtts("Test sentence.", output_path)

        MockGTTS.assert_called_once_with(text="Test sentence.", lang="en", slow=False)

    def test_audio_is_saved_to_correct_path(self, tmp_path):
        """The gTTS object's save() method must be called with the output filepath."""
        output_path = str(tmp_path / "out.mp3")

        with patch("voice_of_the_doctor.gTTS") as MockGTTS, \
             patch("voice_of_the_doctor._play_audio"):
            mock_tts = MagicMock()
            MockGTTS.return_value = mock_tts
            text_to_speech_with_gtts("Hello", output_path)

        mock_tts.save.assert_called_once_with(output_path)

    def test_play_audio_is_called_after_saving(self, tmp_path):
        """Local playback should be attempted once the file has been saved."""
        output_path = str(tmp_path / "out.mp3")

        with patch("voice_of_the_doctor.gTTS") as MockGTTS, \
             patch("voice_of_the_doctor._play_audio") as mock_play:
            MockGTTS.return_value = MagicMock()
            text_to_speech_with_gtts("Hello", output_path)

        mock_play.assert_called_once_with(output_path)


class TestTextToSpeechWithElevenLabs:
    """Tests for the ElevenLabs-based TTS function (premium engine).

    ElevenLabs and save are lazy-imported inside the function body to avoid
    a hard startup dependency. Tests therefore patch them at their source
    module paths rather than via voice_of_the_doctor's namespace.
    """

    def test_returns_output_filepath(self, tmp_path):
        """The function should return the path it was given so Gradio can serve the file."""
        output_path = str(tmp_path / "out.mp3")

        with patch("elevenlabs.client.ElevenLabs") as MockEL, \
             patch("elevenlabs.save"), \
             patch("voice_of_the_doctor._play_audio"):
            MockEL.return_value.text_to_speech.convert.return_value = b""
            result = text_to_speech_with_elevenlabs("Doctor response.", output_path)

        assert result == output_path

    def test_elevenlabs_uses_config_values(self, tmp_path):
        """Voice ID, model ID, and output format must come from config.py — not hardcoded."""
        from config import ELEVENLABS_VOICE_ID, TTS_ELEVENLABS_MODEL, ELEVENLABS_OUTPUT_FORMAT

        output_path = str(tmp_path / "out.mp3")

        with patch("elevenlabs.client.ElevenLabs") as MockEL, \
             patch("elevenlabs.save"), \
             patch("voice_of_the_doctor._play_audio"):
            mock_convert = MockEL.return_value.text_to_speech.convert
            mock_convert.return_value = b""
            text_to_speech_with_elevenlabs("text", output_path)

        mock_convert.assert_called_once_with(
            text="text",
            voice_id=ELEVENLABS_VOICE_ID,
            model_id=TTS_ELEVENLABS_MODEL,
            output_format=ELEVENLABS_OUTPUT_FORMAT,
        )

    def test_audio_bytes_are_saved_to_filepath(self, tmp_path):
        """The save() helper must be called with the audio bytes and the output path."""
        output_path = str(tmp_path / "out.mp3")
        fake_audio = b"fake_audio_bytes_here"

        with patch("elevenlabs.client.ElevenLabs") as MockEL, \
             patch("elevenlabs.save") as mock_save, \
             patch("voice_of_the_doctor._play_audio"):
            MockEL.return_value.text_to_speech.convert.return_value = fake_audio
            text_to_speech_with_elevenlabs("text", output_path)

        mock_save.assert_called_once_with(fake_audio, output_path)
