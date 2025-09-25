# utils/_file_type.py
"""
Single source of truth for the formats directly indexable by the OpenAI File Search tool.
If OpenAI updates support, update here and the rest of the pipeline will respect it.

This module intentionally avoids exporting mutable globals.
Use the provided getters and helper predicates instead.
"""

from __future__ import annotations
from typing import Dict, Set


# ----------------- Supported sets (internal, do not export directly) -----------------
_SUPPORTED_EXT: Set[str] = {
    # office / docs
    ".pdf",
    ".doc",
    ".docx",
    ".pptx",
    # text / markup
    ".txt",
    ".md",
    ".html",
    ".json",
    ".tex",
    # code
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".rb",
    ".php",
    ".go",
    ".sh",
}

# These are the exact MIME strings we accept as "supported" in detection.
_SUPPORTED_MIME: Set[str] = {
    # office
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # text / markup / data
    "text/plain",
    "text/markdown",
    "text/html",
    "application/json",
    # code (as requested list)
    "text/x-python",
    "text/x-script.python",
    "text/javascript",
    "application/typescript",
    "text/x-java",
    "text/x-c",
    "text/x-c++",
    "text/x-csharp",
    "text/x-ruby",
    "text/x-php",
    "text/x-golang",
    "application/x-sh",
    "text/x-tex",
    # common doc
    "application/pdf",
}

# Convenience set used by stager for final outputs
_INDEXABLE_EXT: Set[str] = {
    ".pdf",
    ".doc",
    ".docx",
    ".pptx",
    ".txt",
    ".md",
    ".html",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".rb",
    ".php",
    ".tex",
    ".json",
    ".go",
    ".sh",
}

# ----------------- MIME type mappings -----------------
# Canonical ext → canonical MIME (one canonical per ext)
_MIME_MAP: Dict[str, str] = {
    # office / archives still needed for detector
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".ppt": "application/vnd.ms-powerpoint",  # kept for OLE fallback
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xls": "application/vnd.ms-excel",  # kept for OLE fallback
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    # text / markup / data
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
    ".json": "application/json",
    ".tex": "text/x-tex",
    # code (canonical)
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "application/typescript",
    ".java": "text/x-java",
    ".c": "text/x-c",
    ".cpp": "text/x-c++",
    ".cs": "text/x-csharp",
    ".rb": "text/x-ruby",
    ".php": "text/x-php",
    ".go": "text/x-golang",
    ".sh": "application/x-sh",
    # binary/others
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".zip": "application/zip",
    ".bin": "application/octet-stream",
    ".ole": "application/vnd.ms-office",
}

# Reverse map for detection (canonical only)
_MIME_TO_EXT: Dict[str, str] = {v: k for k, v in _MIME_MAP.items()}

# ---------- Aliases & normalizations ----------
# Some detectors or systems may return these MIME variants. We normalize them to (ext, canonical_mime).
# IMPORTANT: All canonical_mime must appear in _SUPPORTED_MIME and in _MIME_MAP values.
_ALIAS_MIME_TO_CANONICAL: Dict[str, tuple[str, str]] = {
    # PDF
    "application/x-pdf": (".pdf", "application/pdf"),
    "application/acrobat": (".pdf", "application/pdf"),
    "application/vnd.pdf": (".pdf", "application/pdf"),
    "application/nappdf": (".pdf", "application/pdf"),
    # JS common alias
    "application/javascript": (".js", "text/javascript"),
    "text/ecmascript": (".js", "text/javascript"),
    "application/ecmascript": (".js", "text/javascript"),
    # Python variant already supported; keep mapping anyway
    "text/x-script.python": (".py", "text/x-python"),
    # TypeScript sometimes shows as text/x-typescript
    "text/x-typescript": (".ts", "application/typescript"),
    # Shell may appear as text/x-sh
    "text/x-sh": (".sh", "application/x-sh"),
}

# ZIP aliases (used in detector ZIP handling)
_ZIP_MIME_ALIASES: Set[str] = {
    "application/zip",
    "application/x-zip-compressed",
    "multipart/x-zip",
}

# PDF aliases (for detector)
_PDF_MIME_ALIASES: Set[str] = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "application/vnd.pdf",
    "application/nappdf",
}

# libmagic names for OLE/CFB containers (DOC/XLS/PPT/…)
_OLE_MIME_ALIASES: Set[str] = {
    "application/CDFV2",
    "application/x-ole-storage",
    "application/x-cfb",
    "application/vnd.ms-office",
}

# jpg oddities
_JPEG_MIME_ALIASES: Set[str] = {
    "image/jpeg",
    "image/jpg",
    "image/pjpeg",
}

# markdown alias
_MARKDOWN_MIME_ALIASES: Set[str] = {
    "text/markdown",
    "text/x-markdown",
}

# csv/tsv alias
_CSV_MIME_ALIASES: Set[str] = {
    "text/csv",
    "text/x-comma-separated-values",
    "application/csv",
}

_TSV_MIME_ALIASES: Set[str] = {
    "text/tab-separated-values",
}

# text/* catch-all prefix
_TEXT_PREFIX: str = "text/"


# ----------------- Getter APIs (return copies to avoid external mutation) -----------------
def get_supported_ext() -> Set[str]:
    return set(_SUPPORTED_EXT)


def get_supported_mime() -> Set[str]:
    return set(_SUPPORTED_MIME)


def get_indexable_ext() -> Set[str]:
    return set(_INDEXABLE_EXT)


def get_mime_map() -> Dict[str, str]:
    return dict(_MIME_MAP)


def get_mime_to_ext() -> Dict[str, str]:
    return dict(_MIME_TO_EXT)


def get_alias_mime_to_canonical() -> Dict[str, tuple[str, str]]:
    return dict(_ALIAS_MIME_TO_CANONICAL)


def get_zip_mime_aliases() -> Set[str]:
    return set(_ZIP_MIME_ALIASES)


def get_pdf_mime_aliases() -> Set[str]:
    return set(_PDF_MIME_ALIASES)


def get_ole_mime_aliases() -> Set[str]:
    return set(_OLE_MIME_ALIASES)


def get_jpeg_mime_aliases() -> Set[str]:
    return set(_JPEG_MIME_ALIASES)


def get_markdown_mime_aliases() -> Set[str]:
    return set(_MARKDOWN_MIME_ALIASES)


def get_csv_mime_aliases() -> Set[str]:
    return set(_CSV_MIME_ALIASES)


def get_tsv_mime_aliases() -> Set[str]:
    return set(_TSV_MIME_ALIASES)


def get_text_prefix() -> str:
    return _TEXT_PREFIX


# ----------------- Helper predicates (stable API) -----------------
def is_supported_ext(ext: str) -> bool:
    if not ext:
        return False
    if not ext.startswith("."):
        ext = "." + ext
    return ext.lower() in _SUPPORTED_EXT


def is_supported_mime(mime: str) -> bool:
    if not mime:
        return False
    return mime.lower() in _SUPPORTED_MIME


def is_indexable_ext(ext: str) -> bool:
    if not ext:
        return False
    if not ext.startswith("."):
        ext = "." + ext
    return ext.lower() in _INDEXABLE_EXT
