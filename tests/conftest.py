"""
conftest.py — Shared pytest configuration and fixtures
-------------------------------------------------------
pytest automatically loads this file before running any test.

Key responsibilities:
  1. Set required environment variables BEFORE any app module is imported.
     This prevents _validate_env() in app.py from raising EnvironmentError
     during test collection.
  2. Provide reusable fixtures (sample image, sample audio) that individual
     test modules can request by name.

Note: image fixtures use raw JPEG/PNG header bytes rather than Pillow to
avoid an extra dependency — encode_image() only reads and base64-encodes
bytes, so real image content is not required.
"""

import os

import pytest

# ── Pre-set environment variables ─────────────────────────────────────────────
# These must be set at module level (not inside a fixture) because app.py calls
# _validate_env() at import time — before any fixture can run.
# setdefault() means real keys in the shell environment are never overwritten,
# so tests also work if the developer has a real .env loaded.
os.environ.setdefault("GROQ_API_KEY", "test-groq-key-do-not-use-in-production")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-elevenlabs-key-do-not-use")


# ── Shared Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def sample_image(tmp_path):
    """Create a minimal valid JPEG file in a temporary directory.

    Writes a real JPEG file header (SOI marker + minimal APP0 segment) so
    os.path.getsize() and open() both work correctly. Full image content
    is not required because encode_image() only base64-encodes the raw bytes.
    No Pillow dependency needed.
    """
    img_path = tmp_path / "sample.jpg"
    # Minimal JPEG: SOI (FF D8) + APP0 marker (FF E0) — enough to be a real file.
    jpeg_header = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xd9"  # EOI marker
    )
    img_path.write_bytes(jpeg_header)
    return str(img_path)


@pytest.fixture()
def sample_png(tmp_path):
    """Create a minimal PNG file (valid PNG header + IEND chunk)."""
    png_path = tmp_path / "sample.png"
    # Minimal PNG: signature + IHDR + IDAT + IEND
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"                   # PNG signature
        b"\x00\x00\x00\rIHDR"                  # IHDR chunk length + type
        b"\x00\x00\x00\x01\x00\x00\x00\x01"   # 1x1 pixel
        b"\x08\x02\x00\x00\x00"                # 8-bit RGB
        b"\x90wS\xde"                           # CRC
        b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f" # Minimal IDAT
        b"\x00\x00\x01\x01\x00\x05\x18\xd8N"  # chunk data + CRC
        b"\x00\x00\x00\x00IEND\xaeB`\x82"      # IEND
    )
    png_path.write_bytes(png_bytes)
    return str(png_path)


@pytest.fixture()
def sample_audio(tmp_path):
    """Create a minimal fake MP3 file in a temporary directory.

    The content doesn't need to be valid audio because the Groq transcription
    client is mocked in all tests that use this fixture. We just need a file
    that exists on disk so open() succeeds.
    """
    audio_path = tmp_path / "sample.mp3"
    # Minimal MP3 sync word header — enough to pass open() but not real audio.
    audio_path.write_bytes(b"\xff\xfb\x90\x00" * 100)
    return str(audio_path)
