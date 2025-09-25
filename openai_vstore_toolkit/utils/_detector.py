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

# ---- Single source of truth: import getters & helpers from _file_type ----
from ._file_type import (
    get_supported_ext,
    get_supported_mime,
    get_indexable_ext,
    get_mime_map,
    get_mime_to_ext,
    get_alias_mime_to_canonical,
    get_zip_mime_aliases,
    get_pdf_mime_aliases,
    get_ole_mime_aliases,
    get_jpeg_mime_aliases,
    get_markdown_mime_aliases,
    get_csv_mime_aliases,
    get_tsv_mime_aliases,
    get_text_prefix,
    is_supported_ext,
    is_supported_mime,
)

# Snapshots (read-only usage)
SUPPORTED_EXT = get_supported_ext()
SUPPORTED_MIME = get_supported_mime()
INDEXABLE_EXT = get_indexable_ext()

MIME_MAP = get_mime_map()
MIME_TO_EXT = get_mime_to_ext()
ALIAS_MIME_TO_CANONICAL = get_alias_mime_to_canonical()

ZIP_MIME_ALIASES = get_zip_mime_aliases()
PDF_MIME_ALIASES = get_pdf_mime_aliases()
OLE_MIME_ALIASES = get_ole_mime_aliases()
JPEG_MIME_ALIASES = get_jpeg_mime_aliases()
MARKDOWN_MIME_ALIASES = get_markdown_mime_aliases()
CSV_MIME_ALIASES = get_csv_mime_aliases()
TSV_MIME_ALIASES = get_tsv_mime_aliases()

TEXT_PREFIX = get_text_prefix()

# ===== Strong signatures (fallback path) =====
PDF_SIG = b"%PDF-"
ZIP_SIG = b"PK\x03\x04"
ZIP_EMPTY_SIG = b"PK\x05\x06"
PNG_SIG = b"\x89PNG\r\n\x1a\n"
JPG_SIG_PREFIX = b"\xff\xd8\xff"
OLE_CFB_SIG = (
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy DOC/XLS/PPT (Compound File Binary)
)


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
    # Prefer explicit mapping
    mime_to_ext = get_mime_to_ext()
    if mime in mime_to_ext:
        return mime_to_ext[mime]
    # Try alias → canonical
    alias = ALIAS_MIME_TO_CANONICAL.get(mime)
    if alias:
        ext, canonical_mime = alias
        return ext
    # fallback to stdlib mapping (may return variants like ".jpe")
    ext = mimetypes.guess_extension(mime)
    if ext:
        if ext == ".jpe":
            return ".jpg"
        return ext
    return ".bin"


def _canonical_mime_for_ext(ext: str) -> Optional[str]:
    return MIME_MAP.get(ext)


def _is_plaintext(buf: bytes, min_ratio: float = 0.95) -> bool:
    """
    Quick heuristic to treat file as plaintext:
    - No NUL (\x00)
    - Decode UTF-8 (ignore errors)
    - Printable character ratio > min_ratio
    """
    if not buf:
        return True
    if b"\x00" in buf:  # có NUL => thường là binary
        return False

    sample = buf[:65536]
    txt = sample.decode("utf-8", errors="ignore")
    printable = sum(ch.isprintable() or ch.isspace() for ch in txt)
    return (printable / max(1, len(txt))) >= min_ratio


# --- NDJSON heuristic (unsupported) ---
def _looks_like_ndjson(content: bytes, max_scan_lines: int = 200) -> bool:
    """
    Heuristic: NDJSON = nhiều dòng, mỗi dòng (sau strip) trông giống JSON object/array.
    Chỉ quét một số dòng đầu để nhanh và an toàn.
    """
    try:
        head = content.splitlines()[:max_scan_lines]
        if len(head) < 2:
            return False  # NDJSON thường có >= 2 dòng JSON
        json_like = 0
        nonempty = 0
        for ln in head:
            s = ln.strip()
            if not s:
                continue
            nonempty += 1
            if s.endswith(b","):
                s = s[:-1].rstrip()
            if (s.startswith(b"{") and s.endswith(b"}")) or (
                s.startswith(b"[") and s.endswith(b"]")
            ):
                json_like += 1
        return nonempty > 0 and json_like >= max(2, int(nonempty * 0.6))
    except Exception:
        return False


def _apply_text_heuristics(
    content: bytes, current_mime: str, original_name: Optional[str]
) -> Optional[DetectedType]:
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
        # Normalize markdown/csv/tsv aliases to canonical ones.
        if current_mime in MARKDOWN_MIME_ALIASES:
            return DetectedType(".md", MIME_MAP[".md"], "magic:text markdown alias")
        if current_mime in CSV_MIME_ALIASES:
            return DetectedType(".csv", MIME_MAP[".csv"], "magic:text csv alias")
        if current_mime in TSV_MIME_ALIASES:
            return DetectedType(".tsv", MIME_MAP[".tsv"], "magic:text tsv alias")
        # JSON direct
        if current_mime == "application/json":
            return DetectedType(".json", MIME_MAP[".json"], "magic:json")
        return None

    try:
        if is_supported_mime(current_mime):
            ext = MIME_TO_EXT.get(current_mime)
            if ext and is_supported_ext(ext):
                return DetectedType(ext, current_mime, "magic:text passthrough")

        text = content[:4096]
        # Detect UTF-16/UTF-8
        if text.startswith(b"\xff\xfe") or text.startswith(b"\xfe\xff"):
            encoding = "utf-16"
        elif text.startswith(b"\xef\xbb\xbf"):
            encoding = "utf-8-sig"
        else:
            encoding = "utf-8"

        sample = text.decode(encoding, errors="ignore").strip()

        # --- JSON / NDJSON ---
        if _looks_like_ndjson(content):
            return DetectedType(
                ".bin", MIME_MAP[".bin"], "magic:text ndjson unsupported"
            )
        try:
            if len(content) <= 2 * 1024 * 1024:  # 2MB
                json.loads(content.decode(encoding, errors="ignore"))
                return DetectedType(
                    ".json", MIME_MAP[".json"], "magic:text + heuristic:json"
                )
            else:
                stripped = content.strip()
                if (stripped.startswith(b"{") and stripped.endswith(b"}")) or (
                    stripped.startswith(b"[") and stripped.endswith(b"]")
                ):
                    return DetectedType(
                        ".json",
                        MIME_MAP[".json"],
                        "magic:text + heuristic:json-boundary",
                    )
        except Exception:
            pass
        # --- END JSON/NDJSON ---

        # If magic said generic text BUT filename extension is a known code ext → rescue to canonical
        if original_name:
            name_ext = os.path.splitext(original_name)[1].lower()
            if (
                is_supported_ext(name_ext)
                and name_ext in MIME_MAP
                and name_ext not in {".txt", ".md", ".html"}
            ):
                canon = _canonical_mime_for_ext(name_ext)
                if canon and is_supported_mime(canon):
                    return DetectedType(
                        name_ext, canon, f"heuristic:ext rescue for code ({name_ext})"
                    )

        # Markdown hints
        if any(tok in sample for tok in ("```", "\n# ", "\n## ", "|---")):
            return DetectedType(".md", MIME_MAP[".md"], "magic:text + heuristic:md")

        # Delimited text: CSV/TSV
        lines = [ln for ln in sample.splitlines() if ln.strip()]
        if lines:
            sep = "," if sample.count(",") >= sample.count("\t") else "\t"
            cols = [len(ln.split(sep)) for ln in lines[:20]]
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
            logger.debug("is_plaintext detected")
            return DetectedType(".txt", MIME_MAP[".txt"], "heuristic:plaintext")
        return None
    except Exception:
        return None


def _normalize_magic_mime(mime: str) -> Optional[DetectedType]:
    """
    If mime is a known alias, normalize to (ext, canonical_mime).
    Return None if no alias matched.
    """
    norm = ALIAS_MIME_TO_CANONICAL.get(mime)
    if not norm:
        return None
    ext, canonical_mime = norm
    return DetectedType(ext, canonical_mime, f"magic:alias normalized from {mime}")


def _detect_with_magic(
    content: bytes, original_name: Optional[str]
) -> Optional[DetectedType]:
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
        logger.debug(f"magic mime: {mime}")
    except Exception as e:
        logger.exception(f"_detect_with_magic failed: {e}")
        return None
    if not mime:
        logger.warning(f"_detect_with_magic but no mime found!")
        return None

    mime = mime.lower()

    direct_ext = MIME_TO_EXT.get(mime)
    if is_supported_mime(mime) and direct_ext and is_supported_ext(direct_ext):
        return DetectedType(direct_ext, mime, "magic:direct supported mime")

    alias_dt = _normalize_magic_mime(mime)
    if alias_dt:
        return alias_dt

    if mime in PDF_MIME_ALIASES:
        return DetectedType(".pdf", MIME_MAP[".pdf"], "magic:pdf alias normalized")

    if mime in JPEG_MIME_ALIASES:
        return DetectedType(".jpg", "image/jpeg", "magic:jpeg alias normalized")

    if mime in MARKDOWN_MIME_ALIASES:
        return DetectedType(".md", MIME_MAP[".md"], "magic:markdown alias normalized")

    if mime in CSV_MIME_ALIASES:
        return DetectedType(".csv", MIME_MAP[".csv"], "magic:csv alias normalized")
    if mime in TSV_MIME_ALIASES:
        return DetectedType(".tsv", MIME_MAP[".tsv"], "magic:tsv alias normalized")

    if mime in OLE_MIME_ALIASES:
        return DetectedType(".ole", MIME_MAP[".ole"], "magic:ole/cfb alias normalized")

    if mime in ZIP_MIME_ALIASES:
        logger.debug("mime is a zip")
        ooxml = _sniff_ooxml_from_zip(content)
        if ooxml:
            ext, sentinel = ooxml
            return DetectedType(
                ext, MIME_MAP[ext], f"magic+ooxml:{sentinel}", oxml_inner=sentinel
            )
        return DetectedType(".zip", MIME_MAP[".zip"], "magic:zip non-OOXML")

    if mime == "application/json":
        return DetectedType(".json", MIME_MAP[".json"], "magic:json")

    if mime in ("application/x-empty", "inode/x-empty"):
        if original_name:
            name_ext = os.path.splitext(original_name)[1].lower()
            if is_supported_ext(name_ext):
                canon = MIME_MAP.get(name_ext)
                if canon and is_supported_mime(canon):
                    return DetectedType(
                        name_ext, canon, "magic:empty -> rescued by filename"
                    )
        return DetectedType(".bin", MIME_MAP[".bin"], "magic:empty unsupported")

    head4 = content[:4]
    if head4.startswith(ZIP_SIG) or head4.startswith(ZIP_EMPTY_SIG):
        ooxml = _sniff_ooxml_from_zip(content)
        if ooxml:
            ext, sentinel = ooxml
            return DetectedType(
                ext,
                MIME_MAP[ext],
                f"magic(header zip)+ooxml:{sentinel}",
                oxml_inner=sentinel,
            )
        return DetectedType(".zip", MIME_MAP[".zip"], "magic(header zip):zip non-OOXML")

    refined = _apply_text_heuristics(content, mime, original_name)
    if refined:
        return refined

    ext = _guess_ext_from_mime(mime)
    if mime == "image/jpeg" and ext not in (".jpg", ".jpeg"):
        ext = ".jpg"

    return DetectedType(ext, mime, "magic:mime")


# --- enforce supported list only (with rescue by original_name and canonicalization) ---
def _enforce_supported(dt: DetectedType, original_name: Optional[str]) -> DetectedType:
    """
    If outside curated supported sets, try to rescue using:
      - original_name's extension (if supported),
      - canonicalization by ext or alias,
    Otherwise → .bin / application/octet-stream.
    """
    ext_ok = is_supported_ext(dt.ext)
    mime_ok = is_supported_mime(dt.mime)

    if ext_ok and mime_ok:
        return dt

    if ext_ok and not mime_ok:
        canon = _canonical_mime_for_ext(dt.ext)
        if canon and is_supported_mime(canon):
            return DetectedType(
                dt.ext, canon, f"{dt.reason} -> canonicalized mime for {dt.ext}"
            )
        return DetectedType(
            ".bin", MIME_MAP[".bin"], f"{dt.reason} -> unsupported (bad mime)"
        )

    if mime_ok and not ext_ok:
        ext = MIME_TO_EXT.get(dt.mime)
        if ext and is_supported_ext(ext):
            return DetectedType(
                ext, dt.mime, f"{dt.reason} -> canonicalized ext from mime"
            )
        alias = ALIAS_MIME_TO_CANONICAL.get(dt.mime)
        if alias:
            ext2, canon_mime = alias
            if is_supported_ext(ext2) and is_supported_mime(canon_mime):
                return DetectedType(
                    ext2, canon_mime, f"{dt.reason} -> alias canonicalized"
                )

    if original_name and is_supported_mime(dt.mime):
        name_ext = os.path.splitext(original_name)[1].lower()
        if is_supported_ext(name_ext):
            canon = _canonical_mime_for_ext(name_ext) or dt.mime
            if is_supported_mime(canon):
                return DetectedType(
                    name_ext,
                    canon,
                    f"{dt.reason} -> rescued by original_name:{name_ext}",
                )

    if ext_ok and mime_ok:
        return dt

    return DetectedType(".bin", MIME_MAP[".bin"], f"{dt.reason} -> unsupported")


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
        dt = _detect_with_magic(content, original_name)
        if dt is not None:
            logger.debug("magic got detected")
            return _enforce_supported(dt, original_name)
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
        # 3) Fallback text heuristics (with filename rescue for code)
        refined = _apply_text_heuristics(content, "text/plain", original_name)
        if refined:
            return _enforce_supported(refined, original_name)
        # 4) Unknown
        return DetectedType(".bin", MIME_MAP[".bin"], "fallback:unknown")
