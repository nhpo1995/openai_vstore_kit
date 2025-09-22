from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class ResultDetail(BaseModel):
    attributes: Optional[Dict[str, Union[str, float, bool]]] = None
    file_name: Optional[str] = None
    quote: Optional[str] = None
    score: Optional[float] = None


class FileSearchResponse(BaseModel):
    """
    Represents the structured response from a file search operation,
    including the AI's answer and detailed source information.
    """

    file_id: str | None = Field(..., description="The unique ID of the file.")
    details: List[ResultDetail]


class FileDetail(BaseModel):
    """_summary_

    Attributes:
        file_name (str): file's name
        mine_type (str): file type
        byte_content (BytesIO): file content in bytes
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_name: str
    mime_type: str
    content: BytesIO
