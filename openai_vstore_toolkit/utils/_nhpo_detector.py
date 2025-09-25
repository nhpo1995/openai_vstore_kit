from __future__ import annotations
import os
import json
import zipfile
from io import BytesIO
from typing import Optional
from loguru import logger

from openai_vstore_toolkit.utils._file_type import get_mime_map, get_mime_to_ext
from openai_vstore_toolkit.utils._models import DetectedType

MIME_MAP = get_mime_map()
MIME_TO_EXT = get_mime_to_ext()


class NHPODetector:
    """
    Manual file-type detector (no python-magic).

    Detection priority:
        1) Binary signatures: PDF/PNG/JPEG/OLE/ZIP
        2) ZIP: sniff OOXML (.docx/.xlsx/.pptx); if sniff fails but filename suggests one of those → use that;
            otherwise treat as .zip
        3) Text heuristics (for text-like content):
            - Code (STRICTLY limited to 11 extensions): .py, .js, .ts, .java, .c, .cpp, .cs, .rb, .php, .go, .sh
                (detected via shebang + language cues, or rescued by filename)
            - JSON, CSV/TSV, HTML, Markdown, TeX
            - Otherwise plain text (.txt)
        4) Unknown → .bin

    Notes:
        - Code detection by content returns ONLY one of the 11 allowed extensions.
        - Filename "rescue" for code also respects the same 11 extensions.
    """

    # --- Fixed binary signatures ---
    PDF_SIG = b"%PDF-"
    ZIP_SIG = b"PK\x03\x04"
    ZIP_EMPTY_SIG = b"PK\x05\x06"
    PNG_SIG = b"\x89PNG\r\n\x1a\n"
    JPG_SIG_PREFIX = b"\xff\xd8\xff"
    OLE_CFB_SIG = (
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # Legacy DOC/XLS/PPT (Compound File Binary)
    )

    # Only these 11 code extensions are supported for code detection
    CODE_EXT = {
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
    TEXTY_EXT = CODE_EXT | {".md", ".json", ".html", ".tex", ".txt", ".csv", ".tsv"}

    @classmethod
    def detect(
        cls, content: bytes, original_name: Optional[str] = None
    ) -> DetectedType:
        """
        Detect the file type of the given content.

        Args:
            content: Raw file bytes.
            original_name: Optional original filename (used for "rescue" hints).

        Returns:
            DetectedType with canonical extension, MIME, and a debug reason.
        """
        head = content[:8]

        # 1) Strong binary signatures
        if head.startswith(cls.PDF_SIG):
            return DetectedType(".pdf", MIME_MAP[".pdf"], "sig:pdf")
        if head.startswith(cls.PNG_SIG):
            return DetectedType(".png", MIME_MAP[".png"], "sig:png")
        if head.startswith(cls.JPG_SIG_PREFIX):
            return DetectedType(".jpg", MIME_MAP[".jpg"], "sig:jpeg")

        # Legacy Office (OLE)
        if head.startswith(cls.OLE_CFB_SIG):
            if original_name:
                ext = os.path.splitext(original_name)[1].lower()
                if ext in {".doc", ".xls", ".ppt"}:
                    return DetectedType(ext, MIME_MAP[ext], f"sig:ole+{ext}")
            return DetectedType(".ole", MIME_MAP[".ole"], "sig:ole")

        # ZIP (OOXML or plain ZIP)
        if head.startswith(cls.ZIP_SIG) or head.startswith(cls.ZIP_EMPTY_SIG):
            ooxml = cls._sniff_ooxml(content)
            if ooxml:
                return ooxml
            if original_name:
                ext = os.path.splitext(original_name)[1].lower()
                if ext in {".docx", ".xlsx", ".pptx"}:
                    logger.debug(f"sig:zip rescued by filename as {ext}")
                    return DetectedType(ext, MIME_MAP[ext], f"sig rescue by name:{ext}")
            return DetectedType(".zip", MIME_MAP[".zip"], "sig:zip")

        # 2) Text-like path: apply textual heuristics if the content looks like text
        if cls._is_text(content):
            return cls._heuristic_text(content, original_name)

        # 3) Fallback to binary/octet-stream
        return DetectedType(".bin", MIME_MAP[".bin"], "fallback:unknown")

    # ----------------- Helpers -----------------
    @staticmethod
    def _sniff_ooxml(buf: bytes) -> Optional[DetectedType]:
        """
        Inspect a ZIP container to identify OOXML Office types.

        Returns:
            DetectedType for .docx/.xlsx/.pptx if found; otherwise None.
        """
        try:
            with zipfile.ZipFile(BytesIO(buf)) as zf:
                names = zf.namelist()
                if any(n.startswith("word/") for n in names):
                    logger.debug("_sniff_ooxml: detected DOCX")
                    return DetectedType(
                        ".docx", MIME_MAP[".docx"], "zip+ooxml:docx", "word/"
                    )
                if any(n.startswith("xl/") for n in names):
                    logger.debug("_sniff_ooxml: detected XLSX")
                    return DetectedType(
                        ".xlsx", MIME_MAP[".xlsx"], "zip+ooxml:xlsx", "xl/"
                    )
                if any(n.startswith("ppt/") for n in names):
                    logger.debug("_sniff_ooxml: detected PPTX")
                    return DetectedType(
                        ".pptx", MIME_MAP[".pptx"], "zip+ooxml:pptx", "ppt/"
                    )
        except Exception as e:
            logger.debug(f"_sniff_ooxml failed: {e}")
        return None

    @staticmethod
    def _is_text(buf: bytes, min_ratio: float = 0.80) -> bool:
        """
        Lightweight check whether the content is likely text.

        Heuristic:
            - Reject if NUL byte is present.
            - Decode up to 64 KiB as UTF-8 (ignore errors) and measure printable ratio.
        """
        if not buf:
            return True
        if b"\x00" in buf:
            return False
        sample = buf[:65536].decode("utf-8", errors="ignore")
        printable = sum(ch.isprintable() or ch.isspace() for ch in sample)
        return (printable / max(1, len(sample))) >= min_ratio

    @staticmethod
    def _guess_code_ext(txt: str) -> Optional[str]:
        """
        Guess a code extension (STRICTLY among the 11 supported) based on shebang and keywords.

        Returns:
            One of {'.py','.js','.ts','.java','.c','.cpp','.cs','.rb','.php','.go','.sh'} if confidence is sufficient;
            otherwise None.
        """
        # Shebang on the first line
        first = (txt.splitlines()[0] if txt else "").lower()
        if first.startswith("#!"):
            if "python" in first:
                return ".py"
            if "node" in first or "deno" in first:
                return ".js"
            if "bash" in first or "sh" in first or "dash" in first:
                return ".sh"
            if "php" in first:
                return ".php"
            if "ruby" in first or "jruby" in first:
                return ".rb"
            if "go" in first:
                return ".go"

        # Keyword scoring (very lightweight, order does not imply priority)
        t = txt
        # Special case: PHP tag
        if "<?php" in t:
            return ".php"

        scores = {
            ".ts": sum(
                k in t
                for k in [
                    "interface ",
                    "type ",
                    "enum ",
                    "readonly ",
                    "implements ",
                    " as const",
                ]
            ),
            ".js": sum(
                k in t
                for k in [
                    "function ",
                    "=>",
                    "const ",
                    "let ",
                    "export ",
                    "import ",
                    "console.log(",
                ]
            ),
            ".py": sum(
                k in t for k in ["def ", "from ", "import ", "class ", "__name__"]
            ),
            ".java": sum(
                k in t
                for k in [
                    "package ",
                    "public class ",
                    "class ",
                    "public static void main",
                ]
            ),
            ".cpp": sum(
                k in t for k in ["#include", "std::", "using namespace", "template<"]
            ),
            ".c": sum(k in t for k in ["#include", "int main(", "printf("]),
            ".cs": sum(
                k in t
                for k in [
                    "using System",
                    "namespace ",
                    "Console.WriteLine",
                    "public class ",
                ]
            ),
            ".go": sum(
                k in t for k in ["package ", "func ", "fmt.", "import (", "go func("]
            ),
            ".rb": sum(k in t for k in ["def ", "\nend\n", "class ", "module "]),
            ".sh": sum(
                k in t for k in ["#!/bin/sh", "#!/bin/bash", "\nif [", "\nfi", "echo "]
            ),
        }
        ext, sc = max(scores.items(), key=lambda kv: kv[1])
        return ext if sc >= 2 else None  # threshold to avoid false positives

    @classmethod
    def _heuristic_text(
        cls, content: bytes, original_name: Optional[str]
    ) -> DetectedType:
        """
        Heuristic path for text-like content:
            - Prefer filename rescue for known text-like types (with strict code set)
            - Then try code by content (strict set)
            - Then JSON, HTML, CSV/TSV, Markdown, TeX
            - Else plain text
        """
        txt = content[:65536].decode("utf-8", errors="ignore")
        name_ext = os.path.splitext(original_name)[1].lower() if original_name else None

        # A) Filename rescue for text-like types
        if name_ext in cls.TEXTY_EXT and name_ext in MIME_MAP and name_ext != ".txt":
            # Code rescue allowed only for the strict CODE_EXT set
            if name_ext in cls.CODE_EXT or name_ext in {
                ".md",
                ".json",
                ".html",
                ".tex",
                ".csv",
                ".tsv",
            }:
                logger.debug(f"rescue by filename: {name_ext}")
                return DetectedType(
                    name_ext, MIME_MAP[name_ext], f"rescue by name:{name_ext}"
                )

        # B) Code by content (strict set)
        code_ext = cls._guess_code_ext(txt)
        if code_ext and code_ext in cls.CODE_EXT:
            logger.debug(f"heuristic code detected: {code_ext}")
            return DetectedType(code_ext, MIME_MAP[code_ext], "heuristic:code")

        # C) JSON
        try:
            json.loads(txt)
            return DetectedType(".json", MIME_MAP[".json"], "heuristic:json")
        except Exception:
            pass

        # D) HTML (tolerant)
        low = txt.lower()
        if (
            ("<!doctype html" in low)
            or ("<html" in low)
            or ("<head" in low)
            or ("<body" in low)
        ):
            return DetectedType(".html", MIME_MAP[".html"], "heuristic:html")

        # E) CSV/TSV (>=70% lines with >=2 columns)
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        if lines:
            comma_cols = [len(ln.split(",")) for ln in lines[:50]]
            tab_cols = [len(ln.split("\t")) for ln in lines[:50]]
            if comma_cols and sum(c > 1 for c in comma_cols) / len(comma_cols) >= 0.7:
                return DetectedType(".csv", MIME_MAP[".csv"], "heuristic:csv")
            if tab_cols and sum(c > 1 for c in tab_cols) / len(tab_cols) >= 0.7:
                return DetectedType(".tsv", MIME_MAP[".tsv"], "heuristic:tsv")

        # F) Markdown
        if any(
            tok in txt for tok in ("```", "\n# ", "\n## ", "|---")
        ) or txt.startswith("# "):
            return DetectedType(".md", MIME_MAP[".md"], "heuristic:md")

        # G) TeX
        if (
            ("\\documentclass" in txt)
            or ("\\begin{document}" in txt)
            or ("\\usepackage" in txt)
        ):
            return DetectedType(".tex", MIME_MAP[".tex"], "heuristic:tex")

        # H) Plain text
        return DetectedType(".txt", MIME_MAP[".txt"], "heuristic:txt")
