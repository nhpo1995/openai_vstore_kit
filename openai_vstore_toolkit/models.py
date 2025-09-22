from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ResultDetail(BaseModel):
    attributes: Optional[Dict[str, Union[str, float, bool]]] = None

    text: Optional[str] = None
    """The text that was retrieved from the file."""

    score: Optional[float] = None
    """The relevance score of the file - a value between 0 and 1."""


class FileSearchResponse(BaseModel):
    """
    Represents the structured response from a file search operation,
    including the AI's answer and detailed source information.
    """

    file_id: str = Field(..., description="The unique ID of the file.")
    detail: List[ResultDetail]
