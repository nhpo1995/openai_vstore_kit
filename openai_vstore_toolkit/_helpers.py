import os
import pprint
import re
import mimetypes
import urllib.parse
from io import BytesIO
from typing import List, NamedTuple, Optional
from loguru import logger

import magic
import requests
from requests.exceptions import RequestException

from openai_vstore_toolkit._exceptions import FileExtensionError, FileProcessingError
from openai_vstore_toolkit._models import FileDetail

import mimetypes

__CHUNK_SIZE = 2048


class Helpers:
    """
    Helper functions for file processing, including determining MIME types,
    extracting extensions, and handling file content retrieval from URLs or local paths.
    """

    def __init__(self):
        pass

    @staticmethod
    def _get_mime_and_extension(content_chunk: bytes) -> tuple[str, str]:
        """Determines the MIME type and file extension from a chunk of file content.

        This helper is separated for reusability and clear responsibility.

        Args:
            content_chunk: The first few bytes of the file content.

        Returns:
            A tuple containing the MIME type and the corresponding file extension.

        Raises:
            FileExtensionError: If the file extension cannot be guessed from the
                MIME type.
        """
        mime_type = magic.from_buffer(content_chunk, mime=True)
        extension = mimetypes.guess_extension(mime_type)
        logger.debug(f"mime_type: {mime_type}, extension: {extension}")
        if not extension:
            raise FileExtensionError(
                f"Could not determine file extension for MIME type: {mime_type}"
            )
        return mime_type, extension

    @staticmethod
    def get_detail_from_url(url: str, chunk_size: int = 2048) -> Optional[FileDetail]:
        """Efficiently retrieves file details from a URL.

        It streams the response, reading only a small initial chunk to determine
        the file type before downloading the entire content.

        Args:
            url: The URL of the file to process.

        Returns:
            A FileDetail object if successful, otherwise None.
        """
        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                content_chunk = response.raw.read(chunk_size)
                mime_type, extension = Helpers._get_mime_and_extension(
                    content_chunk=content_chunk
                )
                base_name = None
                if "Content-Disposition" in response.headers:
                    cd = response.headers["Content-Disposition"]
                    fname_match = re.findall('filename="(.+?)"', string=cd)
                    if fname_match:
                        base_name = os.path.splitext(fname_match[0])[0]
                if not base_name:
                    parsed_url = urllib.parse.urlparse(url)
                    base_name = os.path.splitext(os.path.basename(parsed_url.path))[0]
                remaining_content = response.raw.read()
                full_content = content_chunk + remaining_content

                return FileDetail(
                    file_name=f"{base_name}{extension}",
                    mime_type=mime_type,
                    content=BytesIO(initial_bytes=full_content),
                )
        except RequestException as e:
            print(f"Error downloading file from URL '{url}': {e}")
            return None
        except FileProcessingError as e:
            print(f"Error processing file from URL '{url}': {e}")
            return None
        except Exception as e:
            print(f"Unexpected error processing file from URL '{url}': {e}")
            return None

    @staticmethod
    def get_detail_from_local_path(
        file_path: str, chunk_size: int = 2048
    ) -> Optional[FileDetail]:
        """Retrieves file details from a local file path.

        Args:
            file_path: The local path to the file.

        Returns:
            A FileDetail object if successful, otherwise None.
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found at path '{file_path}'")
            return None
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
            content_chunk = file_content[:chunk_size]
            mime_type, extension = Helpers._get_mime_and_extension(
                content_chunk=content_chunk
            )
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            return FileDetail(
                file_name=f"{base_name}{extension}",
                mime_type=mime_type,
                content=BytesIO(file_content),
            )
        except (IOError, FileProcessingError) as e:
            logger.exception(f"Error processing local file '{file_path}': {e}")
            return None
        except Exception as e:
            logger.exception(
                f"Unexpected error processing local file '{file_path}': {e}"
            )
            return None

    # --- Main Dispatcher Function ---
    @staticmethod
    def get_file_detail(file_paths: List[str]) -> List[FileDetail]:
        """Gets file details (name, MIME type, content) from a URL or a local path.

        This function acts as a dispatcher, determining whether the input is a URL
        or a local path and calling the appropriate helper function.

        Args:
            file_path: The URL or local file path.

        Returns:
            A FileDetail object containing file information, or None if an error occurs.
        """
        file_details = []
        for path in file_paths:
            if path.startswith(("http://", "https://")):
                file_detail = Helpers.get_detail_from_url(path)
            else:
                file_detail = Helpers.get_detail_from_local_path(path)
            if file_detail:
                logger.debug(f"Added file detail: {file_detail.file_name}")
                file_details.append(file_detail)

        return file_details
