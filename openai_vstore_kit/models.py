# openai_vstore/models.py
from pydantic import BaseModel, model_validator
from typing import Dict, Literal
from openai.types.vector_store import FileCounts, VectorStore


class StaticDetails(BaseModel):
    """Schema for the nested 'static' object in chunking strategy."""

    max_chunk_size_tokens: int
    chunk_overlap_tokens: int

    @model_validator(mode="after")
    def check_overlap_rule(self) -> "StaticDetails":
        size = self.max_chunk_size_tokens
        overlap = self.chunk_overlap_tokens
        if overlap > size / 2:
            raise ValueError(
                f"Overlap ({overlap}) cannot be more than half of chunk size ({size})."
            )
        return self


class StaticStrategy(BaseModel):
    """Schema for the 'static' chunking strategy object."""

    type: Literal["static"]
    static: StaticDetails


class StoreDetail(BaseModel):
    id: str
    name: str
    created_at: int
    file_counts: FileCounts
