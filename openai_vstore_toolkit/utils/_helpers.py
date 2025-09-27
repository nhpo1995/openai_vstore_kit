"""High-level helper functions for file processing.

This module provides the main business logic for retrieving and preparing
files from various sources for further use.

Features:
    - Handles file retrieval from both remote URLs and local file paths.
    - Orchestrates the full processing pipeline: fetches content, detects
        the true file type, and validates it.
    - Returns a standardized `FileDetail` object, ready for consumption.
"""

import os
import random
import re
import urllib.parse
from datetime import datetime
from io import BytesIO
from typing import List, Mapping, Optional

import requests
from loguru import logger
from requests.exceptions import RequestException

from openai_vstore_toolkit.utils._detector import DetectedType, FileTypeDetector
from openai_vstore_toolkit.utils._file_type import is_supported_ext
from openai_vstore_toolkit.utils._models import FileDetail

# Hard cap for any file size (50 MiB). Use 50 * 1_000_000 if you prefer decimal MB.
MAX_BYTES = 50 * 1024 * 1024


class Helper:
    """
    Helper functions for file processing, including determining MIME types,
    extracting extensions, and handling file content retrieval from URLs or local paths.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------
    # Private helpers
    # ------------------------------
    @staticmethod
    def _read_url_with_cap(resp: requests.Response, max_bytes: int = MAX_BYTES) -> Optional[bytes]:
        """Stream an HTTP response body with a hard size cap. Return bytes or None if exceeded."""
        chunks: List[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=8192):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                logger.debug(f"Content too large while streaming: {total} > {max_bytes}")
                return None
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _read_local_with_cap(path: str, max_bytes: int = MAX_BYTES) -> Optional[bytes]:
        """Read a local file in chunks with a hard size cap. Return bytes or None if exceeded."""
        chunks: List[bytes] = []
        total = 0
        with open(path, "rb") as f:
            while True:
                buf = f.read(8192)
                if not buf:
                    break
                total += len(buf)
                if total > max_bytes:
                    logger.debug(f"Local file too large while reading: {total} > {max_bytes}")
                    return None
                chunks.append(buf)
        return b"".join(chunks)

    @staticmethod
    def _numeric_suffix() -> str:
        """Return a purely numeric suffix: UTC timestamp + 6 random digits (helps avoid collisions)."""
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        rand6 = "".join(random.choice("0123456789") for _ in range(6))
        return f"{ts}{rand6}"

    @staticmethod
    def _clean_name(raw: str) -> str:
        """Decode percent-encoding, trim quotes, and strip any path segments (avoid traversal)."""
        s = urllib.parse.unquote(raw).strip().strip('"')
        # Split on both POSIX '/' and Windows '\' to be platform-agnostic
        s = re.split(r"[\\/]", s)[-1]
        return s

    @staticmethod
    def _derive_original_name(url: str, headers: Mapping[str, str], *, fallback_base: str = "download_noname") -> str:
        """
        Unified name derivation:
        1) Content-Disposition: filename*=  → filename="..." → filename=...
        2) URL basename
        3) Fallback: download_noname_<timestamp><6digits>
        """
        cd = headers.get("Content-Disposition")
        if cd:
            # RFC 5987/6266: filename*=UTF-8''encoded
            m = re.search(r'filename\*\s*=\s*[^\'"]+\'\'([^\s;]+)', cd, flags=re.I)
            if m:
                return Helper._clean_name(m.group(1))
            # filename="..."
            m = re.search(r'filename\s*=\s*"([^"]+)"', cd, flags=re.I)
            if m:
                return Helper._clean_name(m.group(1))
            # filename=...
            m = re.search(r'filename\s*=\s*([^";]+)', cd, flags=re.I)
            if m:
                return Helper._clean_name(m.group(1))

        # Fallback: URL basename (decoded)
        name = Helper._clean_name(os.path.basename(urllib.parse.urlparse(url).path))
        if name:
            return name

        # Last resort: collision-safe numeric suffix
        return f"{fallback_base}_{Helper._numeric_suffix()}"

    # ------------------------------
    # Public helpers
    # ------------------------------
    @staticmethod
    def standardize_store_name(store_name: str) -> str:
        """Normalize a name with these rules:
            - Lowercase.
            - Only [a-z], '_' and digits are allowed.
            - Digits may appear only at the very end (one or more).
            - No digit is allowed immediately after an underscore.
            - Non-allowed chars are converted to '_', consecutive '_' collapse to one,
            leading/trailing '_' are stripped.

        Examples:
            "My Store"      -> "my_store"
            "My-Store123"   -> "my_store123"
            "my_store123"   -> "my_store123"   (valid)
            "1_mystore"     -> "mystore"       (leading digit removed)
            "my_1store"     -> "my_store"      (digit inside removed)
            "my__store***5" -> "my_store5"
            "my_" + "123"   -> "my123"         (no digit right after '_')
        """
        s = store_name.lower().strip()
        s = re.sub(r"[^a-z0-9_]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        # Split into head (no digits allowed) and optional trailing digits
        m = re.match(r"^(.*?)(\d+)?$", s)
        head, tail_digits = (m.group(1), m.group(2) or "") if m else (s, "")
        head = re.sub(r"\d+", "", head)
        head = re.sub(r"_+", "_", head).strip("_")
        if head.endswith("_") and tail_digits:
            head = head.rstrip("_")
        result = head + tail_digits
        result = re.sub(r"_+", "_", result).strip("_")
        return result

    @staticmethod
    def standardize_file_name(filename: str) -> str:
        """
        Normalize filename with these rules:
        - Lowercase
        - Keep all characters except:
            - Replace spaces " " and dashes "-" with "_"
            - Replace dots "." in the stem with "_"
        - Only the last dot is preserved (before extension)
        - Extension is kept but forced to lowercase (letters/digits only)

        Examples:
            "My File.PDF"           -> "my_file.pdf"
            "My-File-Name.TXT"      -> "my_file_name.txt"
            "report.v1.final.DOCX"  -> "report_v1_final.docx"
            "12@file8&name.docx"    -> "12@file8&name.docx"  (allowed symbols)
            "complex name-v2.PY"    -> "complex_name_v2.py"
        """
        filename = filename.strip().lower()
        parts = filename.rsplit(".", 1)
        if len(parts) == 2:
            stem, ext = parts
        else:
            stem, ext = filename, ""
        stem = re.sub(r"[ .-]+", "_", stem)
        stem = re.sub(r"_+", "_", stem).strip("_")
        ext = re.sub(r"[^a-z0-9]+", "", ext)
        if not ext:
            raise ValueError(f"Filename '{filename}' is missing an extension")
        return f"{stem}.{ext}"

    @staticmethod
    def _get_detail_from_url(url: str) -> Optional[FileDetail]:
        """
        Fetches file details from a remote URL.
        Args:
            url (str): The URL of the file to fetch.
        Returns:
            Optional[FileDetail]: The file details if successful, otherwise None.
        Note:
            - Only supports HTTP/HTTPS URLs.
            - Prefers filename from Content-Disposition header if available.
            - Validates file type and extension against supported formats.
        """
        try:
            with requests.get(url, stream=True, timeout=(5, 30), allow_redirects=True) as resp:
                resp.raise_for_status()

                # Unified original name derivation (header → URL → fallback)
                original_name = Helper._derive_original_name(url, resp.headers)
                logger.debug(f"Original name hint: {original_name}")

                # Early header-based size cap (best-effort)
                cl = resp.headers.get("Content-Length")
                if cl and cl.isdigit() and int(cl) > MAX_BYTES:
                    logger.debug(f"Content too large (header): {cl} > {MAX_BYTES}")
                    return None

                # Stream with a hard cap to protect memory even without Content-Length
                content = Helper._read_url_with_cap(resp, max_bytes=MAX_BYTES)
                if content is None:
                    return None

                # Detect real type using bytes + name hint
                detail: DetectedType = FileTypeDetector.detect(content=content, original_name=original_name)

                # Accept only directly indexable files
                if not is_supported_ext(ext=detail.ext):
                    logger.warning(f"Not directly indexable by File Search: ext={detail.ext}, mime={detail.mime}")
                    return None

                # Build stem from derived name, then standardize with detected extension
                stem = os.path.splitext(original_name)[0]
                filename = Helper.standardize_file_name(f"{stem}{detail.ext}")

                logger.debug(f"Final filename: {filename} (mime={detail.mime})")
                return FileDetail(filename, detail.mime, BytesIO(content))

        except RequestException as e:
            logger.error(f"URL get error: {e}")
            return None
        except Exception as e:
            logger.error(f"URL unexpected error: {e}")
            return None

    @staticmethod
    def _get_detail_from_local_path(file_path: str) -> Optional[FileDetail]:
        """
        Fetches file details from a local file path.
        Args:
            file_path (str): The path of the local file.
        Returns:
            Optional[FileDetail]: The file details if successful, otherwise None.
        Note:
            - Validates file existence.
            - Validates file type and extension against supported formats.
            - Uses the file name from the path.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        try:
            # Try to check size before reading; if it fails, fall back to capped read.
            try:
                fsize = os.path.getsize(file_path)
            except OSError:
                fsize = None

            if fsize is not None and fsize > MAX_BYTES:
                logger.debug(f"Local file too large: {fsize} > {MAX_BYTES}")
                return None

            if fsize is None:
                # Unknown size → read with cap to enforce the limit.
                content = Helper._read_local_with_cap(file_path, MAX_BYTES)
                if content is None:
                    return None
            else:
                # Size is known and within limit → safe to read fully.
                with open(file_path, "rb") as f:
                    content = f.read()

            base = os.path.splitext(os.path.basename(file_path))[0]  # drop fake ext
            detail: DetectedType = FileTypeDetector.detect(content, original_name=base)

            if not is_supported_ext(ext=detail.ext):
                logger.warning(f"Not directly indexable by File Search: ext={detail.ext}, mime={detail.mime}")
                return None  # For temporary, only accept directly indexable files

            # Compose → standardize (keeps one '.' and validates ext)
            filename = Helper.standardize_file_name(f"{base}{detail.ext}")

            logger.debug(f"Final filename: {filename} (mime={detail.mime})")
            return FileDetail(filename, detail.mime, BytesIO(content))
        except Exception as e:
            logger.error(f"Local process error: {e}")
            return None

    @staticmethod
    def get_file_detail(file_paths: List[str]) -> List[FileDetail]:
        """
        Dispatcher: URL vs local path.
        Args:
            file_paths (List[str]): List of file paths or URLs.
        Returns:
            List[FileDetail]: List of successfully processed file details.
        Note:
            - Each path is processed according to its type (URL or local).
        """
        details = []
        for p in file_paths:
            d = (
                Helper._get_detail_from_url(p)
                if p.startswith(("http://", "https://"))
                else Helper._get_detail_from_local_path(p)
            )
            if d:
                logger.debug(f"detected -> name:{d.file_name}, mime:{d.mime_type}")
                details.append(d)
            else:
                logger.error(f"Failed to get file detail: {p}")
        return details
