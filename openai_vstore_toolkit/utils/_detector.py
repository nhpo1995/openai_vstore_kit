# utils/detector.py
from __future__ import annotations

import json
import os
import mimetypes
import zipfile
from loguru import logger

from io import BytesIO
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import magic  # provided by python-magic-bin
except Exception:
    magic = None  # graceful fallback if lib not available

# ===== Strong signatures (fallback path) =====
PDF_SIG = b"%PDF-"
ZIP_SIG = b"PK\x03\x04"
ZIP_EMPTY_SIG = b"PK\x05\x06"
PNG_SIG = b"\x89PNG\r\n\x1a\n"
JPG_SIG_PREFIX = b"\xff\xd8\xff"
OLE_CFB_SIG = (
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy DOC/XLS/PPT (Compound File Binary)
)

# ===== Canonical MIME map =====
MIME_MAP = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".zip": "application/zip",
    ".bin": "application/octet-stream",
    ".ole": "application/vnd.ms-office",
}

MIME_TO_EXT = {v: k for k, v in MIME_MAP.items()}
ZIP_MIME_ALIASES = {
    "application/zip",
    "application/x-zip-compressed",
    "multipart/x-zip",
}
TEXT_PREFIX = "text/"


@dataclass
class DetectedType:
    """Detected file infomation

    Attributes:
        ext (str): canonical extension (leading dot)
        mime (str): MIME type
        reason (str): debug info
        oxml_inner (Optional[str]): "word/" | "ppt/" | "xl/" if OOXML inside ZIP
    """

    ext: str
    mime: str
    reason: str
    oxml_inner: Optional[str] = None


# ---------- Helpers ----------
def _sniff_ooxml_from_zip(buf: bytes) -> Optional[Tuple[str, str]]:
    """
    Inspect ZIP for Office OOXML structure:
        - xl/   → .xlsx
        - word/ → .docx
        - ppt/  → .pptx
    args:
            buf (bytes): file content in binary format
    Returns:
            tuple[str, str] or None: (ext, sentinel_dir) or None.
    """
    try:
        with zipfile.ZipFile(file=BytesIO(buf)) as zf:
            for ext, sentinel in (
                (".xlsx", "xl/workbook.xml"),
                (".docx", "word/document.xml"),
                (".pptx", "ppt/presentation.xml"),
            ):
                try:
                    zf.getinfo(sentinel)  # tra cứu trực tiếp
                    return (ext, sentinel.split("/")[0] + "/")
                except KeyError:
                    continue
    except Exception as e:
        logger.exception(f"_sniff_ooxml_from_zip failed: {e}")
    return None


def _guess_ext_from_mime(mime: str) -> str:
    if mime in MIME_TO_EXT:
        return MIME_TO_EXT[mime]
    # fallback to stdlib mapping (may return weird variants like ".jpe")
    ext = mimetypes.guess_extension(mime)
    if ext:
        if ext == ".jpe":
            return ".jpg"
        return ext
    return ".bin"


def _is_plaintext(buf: bytes, min_ratio: float = 0.95) -> bool:
    """
    Heuristic nhanh để coi file là plaintext:
    - Không chứa NUL (\x00)
    - Decode được UTF-8 (bỏ lỗi)
    - Tỉ lệ ký tự in được > min_ratio
    """
    if not buf:
        return True
    if b"\x00" in buf:  # có NUL => thường là binary
        return False

    sample = buf[:65536]
    txt = sample.decode("utf-8", errors="ignore")
    printable = sum(ch.isprintable() or ch.isspace() for ch in txt)
    return (printable / max(1, len(txt))) >= min_ratio


def _apply_text_heuristics(content: bytes, current_mime: str) -> Optional[DetectedType]:
    """If reported as generic text, refine to markdown/csv/tsv/txt.

    Args:
        content (bytes): file content in binary format
        current_mime (str): file mime type

    Returns:
        DetectedType or None:
        DetctedType(
                ext: str,
                mime: str,
                reason: str,
                oxml_inner: Optional[str] = None
            )
    """
    if not current_mime.startswith(TEXT_PREFIX):
        return None
    try:
        text = content[:4096]
        # Detect UTF-16/UTF-8
        if text.startswith(b"\xff\xfe") or text.startswith(b"\xfe\xff"):
            encoding = "utf-16"
        elif text.startswith(b"\xef\xbb\xbf"):
            encoding = "utf-8-sig"
        else:
            encoding = "utf-8"

        sample = text.decode(encoding, errors="ignore").strip()
        # JSON (pre-check json)
        if (text.startswith("{") and text.endswith("}")) or (
            text.startswith("[") and text.endswith("]")
        ):
            try:
                json.loads(text)  # validate nhanh
                return DetectedType(
                    ".json", MIME_MAP[".json"], "magic:text + heuristic:json"
                )
            except Exception:
                pass
        # Markdown hints
        if any(tok in sample for tok in ("```", "\n# ", "\n## ", "|---")):
            return DetectedType(".md", MIME_MAP[".md"], "magic:text + heuristic:md")
        # Delimited text: CSV/TSV
        lines = [ln for ln in sample.splitlines() if ln.strip()]
        if lines:
            sep = "," if sample.count(",") >= sample.count("\t") else "\t"
            cols = [len(ln.split(sep)) for ln in lines[:20]]
            # columns mostly consistent and >1 → delimited
            if len(set(cols)) <= 3 and max(cols) > 1:
                if sep == ",":
                    return DetectedType(
                        ".csv", MIME_MAP[".csv"], "magic:text + heuristic:csv"
                    )
                return DetectedType(
                    ".tsv", MIME_MAP[".tsv"], "magic:text + heuristic:tsv"
                )
        # Plain text
        if _is_plaintext(content):
            return DetectedType(".txt", MIME_MAP[".txt"], "heuristic:plaintext")
        return None
    except Exception:
        return None


def _detect_with_magic(content: bytes) -> Optional[DetectedType]:
    """Try python-magic (libmagic). Returns DetectedType or None if unavailable/fails.

    args:
        content (bytes): file content in binary format

    Returns:
        DetectedType: Returns DetectedType or None if unavailable/fails
    """
    if magic is None:
        return None
    try:
        mime = magic.from_buffer(buffer=content[:8192], mime=True)
    except Exception as e:
        logger.exception(f"_detect_with_magic failed: {e}")
        return None
    if not mime:
        logger.warning(f"_detect_with_magic but no mime found!")
        return None
    # If it's ZIP, check if it's actually an OOXML container
    if mime in ZIP_MIME_ALIASES:
        logger.debug("mime is a zip")
        ooxml = _sniff_ooxml_from_zip(content)
        if ooxml:
            ext, sentinel = ooxml
            return DetectedType(
                ext, MIME_MAP[ext], f"magic+ooxml:{sentinel}", oxml_inner=sentinel
            )
        return DetectedType(".zip", MIME_MAP[".zip"], "magic:zip non-OOXML")
    # If generic text, refine
    refined = _apply_text_heuristics(content, mime)
    if refined:
        return refined
    # Map MIME → extension
    ext = _guess_ext_from_mime(mime)
    # Normalize JPEG extension
    if mime == "image/jpeg" and ext not in (".jpg", ".jpeg"):
        ext = ".jpg"
    return DetectedType(ext, mime, "magic:mime")


# ---------- Public Detector ----------


class FileTypeDetector:
    def __init__(self) -> None:
        pass

    @staticmethod
    def detect(content: bytes, original_name: Optional[str] = None) -> DetectedType:
        """
        Priority:
            1) python-magic (libmagic) if available
            2) Fallback to strong signatures (PDF/PNG/JPEG/OLE/ZIP)
            3) Fallback to text heuristics (md/csv/tsv/txt)
            4) Unknown → .bin
        """
        # libmagic
        dt = _detect_with_magic(content)
        if dt is not None:
            logger.debug("magic got detected")
            return dt
        logger.debug("magic detected failed, then fallback to signatures-based")

        # Fallback: signature-based
        head = content[:16]
        name_ext = os.path.splitext(original_name)[1].lower() if original_name else None
        if head.startswith(PDF_SIG):
            return DetectedType(
                ext=".pdf", mime=MIME_MAP[".pdf"], reason="fallback sig:%PDF-"
            )
        if head.startswith(PNG_SIG):
            return DetectedType(
                ext=".png", mime=MIME_MAP[".png"], reason="fallback sig:PNG"
            )
        if head.startswith(JPG_SIG_PREFIX):
            return DetectedType(
                ext=".jpg", mime=MIME_MAP[".jpg"], reason="fallback sig:JPEG"
            )
        # Legacy Office compound file
        if head.startswith(OLE_CFB_SIG):
            if name_ext in {".doc", ".xls", ".ppt"}:
                return DetectedType(
                    name_ext, MIME_MAP[name_ext], f"fallback sig:OLE + hint:{name_ext}"
                )
            return DetectedType(
                ".ole", MIME_MAP[".ole"], "fallback sig:OLE undetermined"
            )
        # ZIP container → OOXML or plain zip
        if head.startswith(ZIP_SIG) or head.startswith(ZIP_EMPTY_SIG):
            ooxml = _sniff_ooxml_from_zip(content)
            if ooxml:
                ext, sentinel = ooxml
                return DetectedType(
                    ext,
                    MIME_MAP[ext],
                    f"fallback zip+ooxml:{sentinel}",
                    oxml_inner=sentinel,
                )
            return DetectedType(".zip", MIME_MAP[".zip"], "fallback sig:ZIP non-OOXML")
        # 3) Fallback text heuristics
        refined = _apply_text_heuristics(content, "text/plain")
        if refined:
            return refined
        # 4) Unknown
        return DetectedType(".bin", MIME_MAP[".bin"], "fallback:unknown")
