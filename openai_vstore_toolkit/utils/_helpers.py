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
import re
import urllib.parse
from io import BytesIO
from typing import List, Optional
from loguru import logger

import requests
from requests.exceptions import RequestException
from openai_vstore_toolkit.utils._detector import DetectedType, FileTypeDetector
from openai_vstore_toolkit.utils._models import FileDetail
from openai_vstore_toolkit.utils._file_type import is_supported_ext


class Helper:
    """
    Helper functions for file processing, including determining MIME types,
    extracting extensions, and handling file content retrieval from URLs or local paths.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _basename_from_url(url: str, fallback: str = "download") -> str:
        """Extracts the base name from a URL.

        Args:
            url (str): The URL from which to extract the base name.
            fallback (str): The fallback name if extraction fails.

        Returns:
            str: The extracted base name.
        """
        parsed = urllib.parse.urlparse(url)
        name = os.path.basename(parsed.path) or fallback
        return os.path.splitext(name)[0]

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
            with requests.get(url, stream=True) as resp:
                resp.raise_for_status()
                original_name = None
                cd = resp.headers.get("Content-Disposition")
                if cd:
                    m = re.findall(pattern='filename="(.+?)"', string=cd)
                    if m:
                        original_name = m[0]
                        logger.debug(
                            f"Filename from Content-Disposition: {original_name}"
                        )
                if not original_name:
                    original_name = os.path.basename(
                        p=urllib.parse.urlparse(url=url).path
                    )
                content = resp.raw.read()
                detail: DetectedType = FileTypeDetector.detect(
                    content=content, original_name=original_name
                )
                if not is_supported_ext(ext=detail.ext):
                    logger.warning(
                        f"Not directly indexable by File Search: ext={detail.ext}, mime={detail.mime}"
                    )
                    return None  # For temporary, only accept directly indexable files
                basename = Helper._basename_from_url(url=url, fallback=original_name)
                logger.debug(f"Determined base name: {basename}")
                filename = f"{basename}{detail.ext}"
                if not is_supported_ext(ext=detail.ext):
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
            with open(file_path, "rb") as f:
                content = f.read()
            base = os.path.splitext(os.path.basename(file_path))[0]  # drop fake ext
            detail: DetectedType = FileTypeDetector.detect(content, original_name=base)
            if not is_supported_ext(ext=detail.ext):
                logger.warning(
                    f"Not directly indexable by File Search: ext={detail.ext}, mime={detail.mime}"
                )
                return None  # For temporary, only accept directly indexable files
            filename = f"{base}{detail.ext}"
            if not is_supported_ext(ext=detail.ext):
                logger.warning(
                    f"Not directly indexable by File Search: ext={detail.ext}, mime={detail.mime}"
                )
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
