"""
Single source of truth for the formats supported by the toolkit.
If OpenAI updates support, update here and the rest of the pipeline will respect it.

This module intentionally avoids exporting mutable globals.
Use the provided getters and helper predicates instead.
"""

from __future__ import annotations
from typing import Dict, Set

# Define all supported types, mappings, and aliases as internal constants
_SUPPORTED_EXT: Set[str] = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".csv",
    ".tsv",
    ".txt",
    ".md",
    ".html",
    ".json",
    ".tex",
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
    ".png",
    ".jpg",
    ".jpeg",
    ".zip",
    ".ole",
    ".bin",
}
_SUPPORTED_MIME: Set[str] = {
    "application/pdf",
    "application/msword",
    "application/vnd.ms-powerpoint",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "text/plain",
    "text/markdown",
    "text/html",
    "application/json",
    "text/csv",
    "text/tab-separated-values",
    "text/x-tex",
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
    "image/png",
    "image/jpeg",
    "application/zip",
    "application/octet-stream",
    "application/vnd.ms-office",
}
_INDEXABLE_EXT: Set[str] = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
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
    ".csv",
    ".tsv",
    ".go",
    ".sh",
}
_MIME_MAP: Dict[str, str] = {
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
    ".html": "text/html",
    ".json": "application/json",
    ".tex": "text/x-tex",
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
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".zip": "application/zip",
    ".bin": "application/octet-stream",
    ".ole": "application/vnd.ms-office",
}
_MIME_TO_EXT: Dict[str, str] = {v: k for k, v in _MIME_MAP.items()}
_ALIAS_MIME_TO_CANONICAL: Dict[str, tuple[str, str]] = {
    "application/x-pdf": (".pdf", "application/pdf"),
    "application/javascript": (".js", "text/javascript"),
    "text/x-script.python": (".py", "text/x-python"),
    "text/x-typescript": (".ts", "application/typescript"),
}
_ZIP_MIME_ALIASES: Set[str] = {
    "application/zip",
    "application/x-zip-compressed",
    "multipart/x-zip",
}
_PDF_MIME_ALIASES: Set[str] = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "application/vnd.pdf",
    "application/nappdf",
}
_OLE_MIME_ALIASES: Set[str] = {
    "application/CDFV2",
    "application/x-ole-storage",
    "application/x-cfb",
    "application/vnd.ms-office",
}
_JPEG_MIME_ALIASES: Set[str] = {"image/jpeg", "image/jpg", "image/pjpeg"}
_MARKDOWN_MIME_ALIASES: Set[str] = {"text/markdown", "text/x-markdown"}
_CSV_MIME_ALIASES: Set[str] = {
    "text/csv",
    "text/x-comma-separated-values",
    "application/csv",
}
_TSV_MIME_ALIASES: Set[str] = {"text/tab-separated-values"}
_TEXT_PREFIX: str = "text/"


# Getter functions to provide safe, read-only access to the constants
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


# Predicate functions for easy checking
def is_supported_ext(ext: str) -> bool:
    if not ext:
        return False
    if not ext.startswith("."):
        ext = "." + ext
    return ext.lower() in _SUPPORTED_EXT


def is_supported_mime(mime: str) -> bool:
    return bool(mime and mime.lower() in _SUPPORTED_MIME)


def is_indexable_ext(ext: str) -> bool:
    if not ext:
        return False
    if not ext.startswith("."):
        ext = "." + ext
    return ext.lower() in _INDEXABLE_EXT
