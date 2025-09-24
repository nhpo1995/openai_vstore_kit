import os
import re
import urllib.parse
from io import BytesIO
from typing import List, Optional
from loguru import logger

import requests
from requests.exceptions import RequestException
from openai_vstore_toolkit.utils._detector import DetectedType, FileTypeDetector
from openai_vstore_toolkit.utils._models import FileDetail
from openai_vstore_toolkit.utils._supported import (
    is_supported_ext,
    is_supported_mime,
)


class Helpers:
    """
    Helper functions for file processing, including determining MIME types,
    extracting extensions, and handling file content retrieval from URLs or local paths.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _basename_from_url(url: str, fallback: str = "download") -> str:
        parsed = urllib.parse.urlparse(url)
        name = os.path.basename(parsed.path) or fallback
        return os.path.splitext(name)[0]  # drop possibly fake ext

    @staticmethod
    def get_detail_from_url(url: str) -> Optional[FileDetail]:
        try:
            with requests.get(url, stream=True) as resp:
                resp.raise_for_status()
                # Prefer content-disposition filename if present, but ignore its extension
                original_name = None
                cd = resp.headers.get("Content-Disposition")
                if cd:
                    m = re.findall(pattern='filename="(.+?)"', string=cd)
                    if m:
                        original_name = m[0]
                if not original_name:
                    original_name = os.path.basename(
                        p=urllib.parse.urlparse(url=url).path
                    )
                content = resp.raw.read()
                detail: DetectedType = FileTypeDetector.detect(
                    content=content, original_name=original_name
                )
                basename = Helpers._basename_from_url(url=url, fallback=original_name)
                filename = f"{basename}.{detail.ext}"
                if not is_supported_ext(ext=detail.ext) and not is_supported_mime(
                    mime=detail.mime
                ):
                    logger.warning(
                        f"Not directly indexable by File Search: ext={detail.ext}, mime={detail.mime}"
                    )

                return FileDetail(filename, detail.mime, BytesIO(content))
        except RequestException as e:
            logger.error(f"URL get error: {e}")
            return None
        except Exception as e:
            logger.error(f"URL unexpected error: {e}")
            return None

    @staticmethod
    def get_detail_from_local_path(file_path: str) -> Optional[FileDetail]:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            base = os.path.splitext(os.path.basename(file_path))[0]  # drop fake ext
            dt: DetectedType = FileTypeDetector.detect(content, original_name=base)
            filename = f"{base}{dt.ext}"

            if not is_supported_ext(ext=dt.ext) and not is_supported_mime(mime=dt.mime):
                logger.warning(
                    f"Not directly indexable by File Search: ext={dt.ext}, mime={dt.mime}"
                )

            return FileDetail(filename, dt.mime, BytesIO(content))
        except Exception as e:
            logger.error(f"Local process error: {e}")
            return None

    @staticmethod
    def get_file_detail(file_paths: List[str]) -> List[FileDetail]:
        """
        Dispatcher: URL vs local path.
        """
        details = []
        for p in file_paths:
            d = (
                Helpers.get_detail_from_url(p)
                if p.startswith(("http://", "https://"))
                else Helpers.get_detail_from_local_path(p)
            )
            if d:
                logger.debug(f"detected -> name:{d.file_name}, mime:{d.mime_type}")
                details.append(d)
            else:
                logger.error(f"Failed to get file detail: {p}")
        return details
