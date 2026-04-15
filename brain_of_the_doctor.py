"""
brain_of_the_doctor.py — Vision + LLM Analysis
------------------------------------------------
Responsible for the "thinking" part of the AI doctor:
  1. Encodes the patient's uploaded image into base64 so it can be
     sent over the Groq REST API.
  2. Sends the encoded image together with the doctor's instruction
     prompt to the LLaMA 4 Scout multimodal model.
  3. Returns the model's natural-language diagnosis.

External dependency: Groq Python SDK (pip install groq)
"""

import base64
import logging
import os

import groq
from groq import Groq
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import API_TIMEOUT_SECONDS, MAX_IMAGE_SIZE_MB, VISION_MODEL

# Module-level logger — output is controlled by the root logger configured in app.py.
logger = logging.getLogger(__name__)

# Pre-compute the byte limit once so we're not multiplying on every call.
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024

# Only these extensions are accepted; anything else (PDF, TIFF, etc.) is rejected
# before making an API call, saving quota and giving the user a clear error.
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def encode_image(image_path: str) -> str:
    """Convert a local image file to a base64-encoded string.

    The Groq vision API expects images embedded as base64 data URIs rather
    than file uploads, so this step is required before every API call.

    Args:
        image_path: Absolute or relative path to the image file.

    Returns:
        Base64-encoded string of the image's raw bytes.

    Raises:
        ValueError: If the file extension is not in ALLOWED_IMAGE_EXTENSIONS,
                    or if the file exceeds MAX_IMAGE_SIZE_MB.
        FileNotFoundError: If no file exists at image_path.
    """
    # Validate file extension — catch unsupported formats early.
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image type '{ext}'. Allowed types: {ALLOWED_IMAGE_EXTENSIONS}"
        )

    # Validate file size — prevent sending huge files to the API.
    size = os.path.getsize(image_path)
    if size > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image too large ({size / 1024 / 1024:.1f} MB). "
            f"Maximum allowed size is {MAX_IMAGE_SIZE_MB} MB."
        )

    # Read and encode — context manager ensures the file handle is always closed.
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@retry(
    # Retry on transient network/server errors only — not on auth or bad input.
    retry=retry_if_exception_type((
        groq.APIConnectionError,  # DNS failure, TCP reset, etc.
        groq.APITimeoutError,     # Server took too long to respond.
        groq.InternalServerError, # Groq 5xx — temporary server fault.
        groq.RateLimitError,      # Quota hit — back off and retry.
    )),
    stop=stop_after_attempt(3),                        # Give up after 3 total tries.
    wait=wait_exponential(multiplier=1, min=2, max=10), # 2s → 4s → 8s between retries.
    reraise=True,                                       # Re-raise the last exception if all attempts fail.
)
def analyze_image_with_query(query: str, encoded_image: str, model: str = VISION_MODEL) -> str:
    """Send an image + text query to the Groq multimodal LLM and return the response.

    Decorated with @retry (tenacity) — automatically retries up to 3 times on
    transient errors (connection drops, timeouts, rate limits, 5xx responses)
    with exponential backoff (2s, 4s, 8s). Auth errors and bad requests are
    NOT retried since they require user action to fix.

    The message structure follows the OpenAI-compatible vision format:
      - A "text" part carries the system prompt + patient's transcribed speech.
      - An "image_url" part carries the base64-encoded image as a data URI.

    Args:
        query:         The full prompt text (system prompt + patient's spoken text).
        encoded_image: Base64 string of the image, produced by encode_image().
        model:         Groq model ID to use (defaults to VISION_MODEL in config).

    Returns:
        The model's plain-text response string.
    """
    # Initialise the Groq client — it reads GROQ_API_KEY from the environment automatically.
    client = Groq()
    logger.info("Sending image + query to model '%s'.", model)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    # Text portion: combined system instructions + patient description.
                    {"type": "text", "text": query},
                    # Image portion: embedded as a base64 data URI (JPEG assumed).
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    },
                ],
            }
        ],
        model=model,
        timeout=API_TIMEOUT_SECONDS,  # Avoid hanging if Groq is slow to respond.
    )

    response = chat_completion.choices[0].message.content
    logger.info("Model responded with %d characters.", len(response))
    return response
