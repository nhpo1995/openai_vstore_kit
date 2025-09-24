from __future__ import annotations
from typing import Optional
import olefile


def ole_office_kind(content: bytes) -> Optional[str]:
    """
    Return'.doc' | '.xls' | '.ppt' if recognize; None if its not OOXML.
    """
    try:
        with olefile.OleFileIO(content) as ole:
            streams = {"/".join(p) for p in ole.listdir(streams=True)}
            if any(s.endswith("WordDocument") for s in streams):
                return ".doc"
            if any(s.endswith(("Workbook", "Book")) for s in streams):
                return ".xls"
            if any(s.endswith("PowerPoint Document") for s in streams):
                return ".ppt"
    except Exception:
        pass
    return None
