"""
test_brain_of_the_doctor.py — Tests for image encoding and LLM analysis
------------------------------------------------------------------------
Tests cover:
  • encode_image() — file validation (extension, size) and base64 output.
  • analyze_image_with_query() — correct API call construction and response handling.

All Groq API calls are mocked with unittest.mock so no real API key or
network connection is needed to run these tests.
No Pillow/PIL dependency — raw bytes are sufficient for encoding tests.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from brain_of_the_doctor import analyze_image_with_query, encode_image


class TestEncodeImage:
    """Tests for the encode_image() function."""

    def test_valid_jpeg_returns_non_empty_base64_string(self, sample_image):
        """Happy path: a valid JPEG file should produce a non-empty base64 string."""
        result = encode_image(sample_image)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_valid_png_is_accepted(self, sample_png):
        """PNG is in ALLOWED_IMAGE_EXTENSIONS and should be accepted without error."""
        result = encode_image(sample_png)
        assert isinstance(result, str)

    def test_unsupported_extension_raises_value_error(self, tmp_path):
        """A .txt file should be rejected before any I/O with a clear error message."""
        bad_path = str(tmp_path / "report.txt")
        with open(bad_path, "w") as f:
            f.write("not an image")
        with pytest.raises(ValueError, match="Unsupported image type"):
            encode_image(bad_path)

    def test_oversized_image_raises_value_error(self, sample_image):
        """Images exceeding MAX_IMAGE_SIZE_MB should be rejected with a size error."""
        # Patch os.path.getsize to simulate a 15 MB file without creating one.
        with patch("os.path.getsize", return_value=15 * 1024 * 1024):
            with pytest.raises(ValueError, match="too large"):
                encode_image(sample_image)

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """Attempting to encode a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            encode_image(str(tmp_path / "ghost.jpg"))


class TestAnalyzeImageWithQuery:
    """Tests for the analyze_image_with_query() function."""

    def _mock_completion(self, content: str):
        """Build a minimal mock that mimics Groq's ChatCompletion response."""
        mock = MagicMock()
        mock.choices[0].message.content = content
        return mock

    def test_returns_model_response_text(self):
        """The function should return exactly what the model's message content contains."""
        expected = "With what I see, I think you have mild acne."
        with patch("brain_of_the_doctor.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = (
                self._mock_completion(expected)
            )
            result = analyze_image_with_query(
                query="What do you see?",
                encoded_image="base64encodedstring",
            )
        assert result == expected

    def test_uses_default_vision_model_from_config(self):
        """When no model is specified, VISION_MODEL from config.py should be used."""
        from config import VISION_MODEL
        with patch("brain_of_the_doctor.Groq") as MockGroq:
            mock_create = MockGroq.return_value.chat.completions.create
            mock_create.return_value = self._mock_completion("ok")
            analyze_image_with_query(query="test", encoded_image="abc")
            _, kwargs = mock_create.call_args
        assert kwargs["model"] == VISION_MODEL

    def test_custom_model_is_forwarded_to_api(self):
        """An explicit model argument must be passed through to the Groq client."""
        with patch("brain_of_the_doctor.Groq") as MockGroq:
            mock_create = MockGroq.return_value.chat.completions.create
            mock_create.return_value = self._mock_completion("ok")
            analyze_image_with_query(
                query="test", encoded_image="abc", model="custom-model-id"
            )
            _, kwargs = mock_create.call_args
        assert kwargs["model"] == "custom-model-id"

    def test_api_timeout_is_set(self):
        """API calls must include a timeout to prevent the UI from hanging."""
        from config import API_TIMEOUT_SECONDS
        with patch("brain_of_the_doctor.Groq") as MockGroq:
            mock_create = MockGroq.return_value.chat.completions.create
            mock_create.return_value = self._mock_completion("ok")
            analyze_image_with_query(query="test", encoded_image="abc")
            _, kwargs = mock_create.call_args
        assert kwargs["timeout"] == API_TIMEOUT_SECONDS
