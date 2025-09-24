# utils/stager.py
from __future__ import annotations
from pathlib import Path
import zipfile, subprocess, shutil, csv
from typing import List, Optional
import pandas as pd

# Optional OCR
_has_pil = False


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


SOFFICE_BIN = _which("soffice") or _which("libreoffice")
OCR_AVAILABLE = False  # luôn false vì đã bỏ logic ảnh

# Final indexable formats used by the File Search tool
from ._supported import INDEXABLE_EXT as INDEXABLE

# Tuning for large tables → split Markdown chunks
MAX_ROWS_PER_MD = 2000
MAX_COLS_SHOW = 200


# ----------------- Utilities -----------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(name: str) -> str:
    for ch in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        name = name.replace(ch, "_")
    return name.strip() or "sheet"


def _to_markdown(df: pd.DataFrame) -> str:
    if df.shape[1] > MAX_COLS_SHOW:
        df = df.iloc[:, :MAX_COLS_SHOW].copy()
    df = df.fillna("")
    return df.to_markdown(index=False)


def _detect_csv_delimiter(sample_path: str) -> str:
    with open(sample_path, "r", encoding="utf-8", errors="ignore") as f:
        head = "".join([next(f) for _ in range(10) if True])
    try:
        dialect = csv.Sniffer().sniff(head, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return "," if head.count(",") >= head.count("\t") else "\t"


def _read_csv_resilient(path: str) -> pd.DataFrame:
    delim = _detect_csv_delimiter(path)
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep=delim, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, sep=delim, encoding="utf-8", on_bad_lines="skip")


def _split_dataframe(df: pd.DataFrame, base_name: str):
    if len(df) <= MAX_ROWS_PER_MD:
        return [(base_name, df)]
    parts = []
    start, part = 0, 1
    while start < len(df):
        end = min(start + MAX_ROWS_PER_MD, len(df))
        parts.append((f"{base_name}__part{part}", df.iloc[start:end].copy()))
        start = end
        part += 1
    return parts


# ----------------- Converters -----------------
def convert_legacy_office(src: str) -> str:
    """
    Convert .doc/.xls/.ppt → .docx/.xlsx/.pptx using LibreOffice (headless).
    """
    p = Path(src)
    ext = p.suffix.lower()
    target_map = {".xls": "xlsx", ".ppt": "pptx"}
    if ext not in target_map:
        raise ValueError(f"Unsupported legacy office: {ext}")
    if not SOFFICE_BIN:
        raise RuntimeError(
            "LibreOffice not found. Install 'soffice' to convert legacy Office formats."
        )
    out_dir = p.parent
    target = target_map[ext]
    subprocess.run(
        [
            SOFFICE_BIN,
            "--headless",
            "--convert-to",
            target,
            str(p),
            "--outdir",
            str(out_dir),
        ],
        check=True,
    )
    return str(out_dir / f"{p.stem}.{target}")


def tabular_to_md(src: str, out_dir: str) -> List[str]:
    """
    Convert CSV/XLSX/XLS into one or multiple Markdown files (.md) for indexing.
    - CSV: 1 file .md
    - XLSX/XLS: mỗi sheet 1 file .md
    """
    p = Path(src)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    outs: List[str] = []

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(src)
        md_path = Path(out_dir) / f"{p.stem}.md"
        md_path.write_text(
            f"# {p.name}\n\n" + df.to_markdown(index=False), encoding="utf-8"
        )
        outs.append(str(md_path))
        return outs

    # XLS/XLSX → mỗi sheet thành 1 .md
    xls = pd.ExcelFile(src)
    for sheet in xls.sheet_names:
        df = pd.read_excel(src, sheet_name=sheet)
        safe_sheet = str(sheet).replace("/", "_").replace("\\", "_").strip()
        md_path = Path(out_dir) / f"{p.stem}__{safe_sheet}.md"
        md_path.write_text(
            f"# {p.name} — Sheet: {sheet}\n\n" + df.to_markdown(index=False),
            encoding="utf-8",
        )
        outs.append(str(md_path))
    return outs


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> List[str]:
    """
    Safe unzip (prevents zip-slip). Returns extracted file paths (files only).
    """
    extracted: List[str] = []
    for member in zf.infolist():
        mf = Path(member.filename)
        if mf.is_absolute() or ".." in mf.parts:
            continue
        target = dest / mf
        if member.is_dir():
            _ensure_dir(target)
            continue
        _ensure_dir(target.parent)
        with zf.open(member) as src_f, open(target, "wb") as dst_f:
            shutil.copyfileobj(src_f, dst_f)
        extracted.append(str(target))
    return extracted


def unzip_stage(src: str, out_dir: str) -> List[str]:
    _ensure_dir(Path(out_dir))
    staged: List[str] = []
    with zipfile.ZipFile(src, "r") as zf:
        files = _safe_extract_zip(zf, Path(out_dir))
        staged.extend(files)
    manifest = Path(out_dir) / "ZIP_MANIFEST.md"
    manifest.write_text(
        "# ZIP Manifest\n" + "\n".join(sorted([Path(s).name for s in staged])),
        encoding="utf-8",
    )
    staged.append(str(manifest))
    return staged


# ----------------- Main entry -----------------
def stage_for_vectorstore(src: str, workdir: str = "vstore_stage") -> List[str]:
    """
    Returns a list of paths that are READY to be indexed by File Search.
    Unsupported formats are converted to indexable outputs first.
    """
    p = Path(src)
    ext = p.suffix.lower()

    if ext in INDEXABLE:
        return [str(p.resolve())]

    if ext in {".doc", ".xls", ".ppt"}:
        if not SOFFICE_BIN:
            raise RuntimeError(
                "Legacy Office requires LibreOffice to convert (.doc/.xls/.ppt → OOXML)."
            )
        converted = convert_legacy_office(src)
        return stage_for_vectorstore(converted, workdir)

    if ext in {".xlsx", ".xls", ".csv"}:
        out_dir = Path(workdir) / f"{_sanitize_filename(p.stem)}_md"
        return tabular_to_md(src, str(out_dir))

    if ext == ".zip":
        base_out = Path(workdir) / f"{_sanitize_filename(p.stem)}_unzip"
        ready: List[str] = []
        for q in unzip_stage(src, str(base_out)):
            ready.extend(stage_for_vectorstore(q, workdir))
        # unique preserve order
        seen, uniq = set(), []
        for r in ready:
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        return uniq

    # Fallback: try to read as text and write .txt
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        out_dir = Path(workdir)
        _ensure_dir(out_dir)
        out = out_dir / f"{_sanitize_filename(p.stem)}.txt"
        out.write_text(txt, encoding="utf-8")
        return [str(out)]
    except Exception:
        return []
