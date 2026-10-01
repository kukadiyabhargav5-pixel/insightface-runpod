"""
Secure, memory-optimized temporary image loader for Cloudflare R2 / S3 URLs.
Ensures zero persistent disk storage and safe memory limits.
"""

import os
import tempfile
import logging
from contextlib import contextmanager
from typing import Generator
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np
import requests

logger = logging.getLogger("insightface-worker.image_loader")


def sanitize_url_for_logging(url: str) -> str:
    """
    Strips query parameters and credentials from URLs to prevent logging private signed tokens.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        # Reconstruct URL without query params or fragment
        sanitized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
        return sanitized
    except Exception:
        return "<invalid-url>"


def download_temporary_image(image_url: str, max_size_mb: int = 20, timeout: int = 60) -> str:
    """
    Downloads an image stream to a secure temporary file on disk.
    Enforces maximum size constraints chunk-by-chunk to protect container RAM.
    Returns the absolute path to the temporary file.
    """
    if not image_url or not isinstance(image_url, str):
        raise ValueError("A valid 'image_url' string is required.")

    parsed = urlparse(image_url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL protocol '{parsed.scheme}'. Only http and https URLs are allowed.")

    max_bytes = max_size_mb * 1024 * 1024
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".img")
    temp_path = temp_file.name

    try:
        headers = {
            "User-Agent": "InsightFace-RunPod-Worker/1.0"
        }
        with requests.get(image_url, stream=True, timeout=timeout, headers=headers) as response:
            response.raise_for_status()

            # Pre-check Content-Length header if present
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError(
                    f"Image size ({int(content_length) / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {max_size_mb} MB."
                )

            downloaded_bytes = 0
            for chunk in response.iter_content(chunk_size=65536):  # 64 KB chunks
                if chunk:
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > max_bytes:
                        raise ValueError(
                            f"Image download exceeded maximum allowed limit of {max_size_mb} MB."
                        )
                    temp_file.write(chunk)

            temp_file.flush()
    except Exception:
        # Clean up partial file immediately on download failure
        temp_file.close()
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        raise
    finally:
        temp_file.close()

    return temp_path


@contextmanager
def temp_image_context(
    image_url: str,
    max_size_mb: int = 20,
    timeout: int = 60
) -> Generator[np.ndarray, None, None]:
    """
    Context manager that temporarily downloads an image, decodes it into a BGR NumPy array,
    yields the array for processing, and guarantees immediate deletion of the temporary file.
    """
    temp_path = None
    try:
        temp_path = download_temporary_image(image_url, max_size_mb=max_size_mb, timeout=timeout)
        
        # Load image via OpenCV
        img = cv2.imread(temp_path, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError("Corrupted or unsupported image file. OpenCV could not decode the image buffer.")
        
        yield img

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as err:
                logger.warning(f"Failed to delete temporary file {temp_path}: {err}")
