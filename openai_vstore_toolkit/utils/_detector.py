"""
Hybrid file-type detection utilities.
Maps files to canonical extensions and MIME types for reliable handling and indexing.

Features

    - Library-first: python-magic for broad MIME, filetype.py/OOXML sniff for Office; fallback to signatures + text heuristics.
    - Ordered strategy to maximize accuracy and consistency.
    - Unified API returning canonical {ext, mime} plus a debug reason for traceability.
"""

from __future__ import annotations
from typing import Optional
from loguru import logger

try:
    import magic
except Exception:
    magic = None
try:
    import filetype
except Exception:
    filetype = None

from openai_vstore_toolkit.utils._file_type import (
    get_mime_map,
    get_mime_to_ext,
    get_alias_mime_to_canonical,
)
from ._nhpo_detector import NHPODetector
from openai_vstore_toolkit.utils._models import DetectedType

ALIAS_MIME_TO_CANONICAL = get_alias_mime_to_canonical()
MIME_TO_EXT = get_mime_to_ext()
MIME_MAP = get_mime_map()


# --- Các hàm xử lý với thư viện ---
def _detect_with_filetype(content: bytes) -> Optional[DetectedType]:
    """Attempts detection using the filetype.py library.
    Returns None if the library is not available or detection fails.
    Args:
        content (bytes): The binary content of the file to be analyzed.
    Returns:
        Optional[DetectedType]: A DetectedType object if detection is successful, otherwise None.
    Note:
        filetype.py may not recognize all file types, especially text-based formats.
    """
    if filetype is None:
        return None
    try:
        kind = filetype.guess(content[:261])
        if kind is not None:
            logger.debug(f"filetype.py detected mime: {kind.mime}")
            return DetectedType(
                f".{kind.extension}", kind.mime, f"filetype.py:{kind.extension}"
            )
    except Exception as e:
        logger.warning(f"_detect_with_filetype failed: {e}")
    return None


def _detect_with_magic(
    content: bytes, original_name: Optional[str]
) -> Optional[DetectedType]:
    """Attempts detection using the python-magic library.
    Args:
        content (bytes): The binary content of the file to be analyzed.
        original_name (Optional[str]): The original filename, used for extension-based hints.
    Returns:
        Optional[DetectedType]: A DetectedType object if detection is successful, otherwise None.
    """
    if magic is None:
        return None
    try:
        mime = magic.from_buffer(buffer=content[:8192], mime=True).lower()
        logger.debug(f"magic detected mime: {mime}")
        ext = (
            MIME_TO_EXT.get(mime) or ALIAS_MIME_TO_CANONICAL.get(mime, (None, None))[0]
        )
        if ext:
            return DetectedType(ext, mime, "magic:direct")
    except Exception as e:
        logger.exception(f"_detect_with_magic failed: {e}")
    return None


# ---------- Public Detector Class ----------
class FileTypeDetector:
    """Hybrid file-type detector using multiple strategies."""

    @staticmethod
    def detect(content: bytes, original_name: Optional[str] = None) -> DetectedType:
        """Orchestrates file type detection using a hybrid library-first strategy.
        Returns a DetectedType with canonical extension, MIME type, and debug reason.
        Features:
            - Priority 1: Use filetype.py (strong for OOXML, etc.)
            - Priority 2: Use python-magic (broad database)
            - Priority 3: Use the in-house fallback detector
        Args:
            content (bytes): The binary content of the file to be analyzed.
            original_name (Optional[str]): The original filename, used for extension-based hints.
        Returns:
            DetectedType: A DetectedType object with detected extension, MIME type, and reason.
        Note:
            - If all methods fail, defaults to .bin with application/octet-stream.
            - Exceptions are caught and logged; detection continues with fallback methods.
        """
        try:
            # Priority 1: Use filetype.py (strong for OOXML, etc.)
            dt = _detect_with_filetype(content)
            if dt:
                logger.debug("Strategy chosen: filetype.py")
                return dt

            # Priority 2: Use python-magic (broad database)
            dt = _detect_with_magic(content, original_name)
            generic_mimes = {"text/plain", "application/octet-stream"}
            if dt and dt.mime not in generic_mimes:
                logger.debug(
                    f"Strategy chosen: python-magic (specific type: {dt.mime})"
                )
                return dt

            # Priority 3: Use the in-house fallback detector
            nhpo = NHPODetector()
            logger.debug("Libraries failed, using NHPO fallback detector.")
            dt = nhpo.detect(content, original_name)
            return dt
        except Exception as e:
            logger.exception(f"FileTypeDetector.detect failed: {e}")
            return DetectedType(".bin", "application/octet-stream", "error:fallback")
