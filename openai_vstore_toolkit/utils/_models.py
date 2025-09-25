"""Data models for file handling and search results.

This module defines the primary data structures used to represent files
and structured API responses within the application.

Models:
    - FileDetail: A container for a file's name, MIME type, and
        binary content (`BytesIO`).
    - FileSearchResponse: A Pydantic model that structures the results
        from a file search tool call.
"""

from io import BytesIO
from typing import List
from pydantic import BaseModel, ConfigDict, Field
from openai.types.responses.response_file_search_tool_call import Result


class FileSearchResponse(BaseModel):
    """Structured response for file search results.

    Attributes:
        file_id (str | None): The unique ID of the file.
        filename (str | None): The name of the file.
        details (List[Result]): List of search results with details.
    """

    file_id: str | None = Field(..., description="The unique ID of the file.")
    filename: str | None = Field(..., description="The name of the file.")
    details: List[Result]


class FileDetail:
    """Lightweight container passed into files.create(...).

    Attributes:
        file_name (str): Name of the file.
        mime_type (str): MIME type of the file.
        content (BytesIO): Binary content of the file.
    """

    def __init__(self, file_name: str, mime_type: str, content: BytesIO):
        self.file_name = file_name
        self.mime_type = mime_type
        self.content = content

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # For allowing taking binary content

    def __repr__(self) -> str:
        return f"file_name='{self.file_name}' mime_type='{self.mime_type}' content=<_io.BytesIO>"
