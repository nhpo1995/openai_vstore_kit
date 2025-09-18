import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Literal, TypedDict, cast
from tqdm import tqdm
import os
from loguru import logger
import base64
import requests
from io import BytesIO
from openai import BaseModel, OpenAI
from dotenv import load_dotenv
from openai.types.static_file_chunking_strategy_object_param import (
    StaticFileChunkingStrategyObjectParam,
)
from openai.types import StaticFileChunkingStrategyParam
from openai.types.auto_file_chunking_strategy_param import AutoFileChunkingStrategyParam
from openai.types import FileChunkingStrategyParam
from openai import NOT_GIVEN, NotGiven
from pydantic import model_validator

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()


def _create_file(file_path: str, client: OpenAI):
    if file_path.startswith("http://") or file_path.startswith("https://"):
        # Download the file content from the URL
        response = requests.get(file_path)
        file_content = BytesIO(response.content)
        file_name = file_path.split("/")[-1]
        file_tuple = (file_name, file_content)
        file_response = client.files.create(file=file_tuple, purpose="assistants")
    else:
        # Handle local file path
        with open(file_path, "rb") as file_content:
            file_response = client.files.create(file=file_content, purpose="assistants")
        file_name = os.path.basename(file_path)
    return file_response, str(file_name)


created_file, file_name = _create_file(
    file_path="https://cdn.openai.com/API/docs/deep_research_blog.pdf", client=client
)


def _custom_chunk_strategy(
    max_chunk_size: int,
    chunk_overlap: int,
) -> StaticFileChunkingStrategyObjectParam:
    custom_chunking_strategy: StaticFileChunkingStrategyObjectParam = {
        "type": "static",
        "static": {
            "max_chunk_size_tokens": max_chunk_size,
            "chunk_overlap_tokens": chunk_overlap,
        },
    }
    return custom_chunking_strategy


class StaticDetails(BaseModel):
    max_chunk_size_tokens: int
    chunk_overlap_tokens: int

    # Đây là validator
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
    type: Literal["static"]
    static: StaticDetails


def _prepare_chunking_strategy(
    self, max_chunk_size, chunk_overlap
) -> StaticFileChunkingStrategyObjectParam | NotGiven:
    """Helper để tạo chunking strategy một cách an toàn."""
    if max_chunk_size is not None and chunk_overlap is not None:
        try:
            strategy_model = StaticStrategy(
                type="static",
                static=StaticDetails(
                    max_chunk_size_tokens=max_chunk_size,
                    chunk_overlap_tokens=chunk_overlap,
                ),
            )
            return cast(
                StaticFileChunkingStrategyObjectParam, strategy_model.model_dump()
            )
        except ValueError as e:
            logger.error(f"Invalid chunking strategy parameters: {e}")
            return NOT_GIVEN
    return NOT_GIVEN


def _upload_single_file(
    file_path: str,
    vector_store_id: str,
    client: OpenAI,
    chunking_strategy: StaticFileChunkingStrategyObjectParam | NotGiven = NOT_GIVEN,
):
    try:
        created_file, file_name = _create_file(file_path=file_path, client=client)
        attach_response = client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=created_file.id,
            chunking_strategy=chunking_strategy,
        )
        return {"file": file_name, "status": "success"}
    except Exception as e:
        logger.exception(f"Error with {file_path}: {str(e)}")
        return {"file": file_path, "status": "failed", "error": str(e)}


def upload_pdf_files_to_vector_store(
    vector_store_id: str,
    list_file: List[str],
    client: OpenAI,
    chunking_strategy: StaticFileChunkingStrategyObjectParam | NotGiven = NOT_GIVEN,
):
    stats = {
        "total_files": len(list_file),
        "successful_uploads": 0,
        "failed_uploads": 0,
        "errors": [],
    }
    logger.info(f"{len(list_file)} PDF files to process. Uploading in parallel...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                _upload_single_file,
                file_path,
                vector_store_id,
                client,
                chunking_strategy,
            ): file_path
            for file_path in list_file
        }
        for future in tqdm(iterable=as_completed(futures), total=len(list_file)):
            result = future.result()
            if result["status"] == "success":
                stats["successful_uploads"] += 1
            else:
                stats["failed_uploads"] += 1
                stats["errors"].append(result)
    return stats


def create_vector_store(store_name: str) -> dict:
    try:
        vector_store = client.vector_stores.create(name=store_name)
        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed,
        }
        print("Vector store created:", details)
        return details
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}


# Execution
# store_name = "openai_blog_store"
# list_file = ["https://cdn.openai.com/API/docs/deep_research_blog.pdf"]
# vector_store_details = create_vector_store(store_name=store_name)
# custom_strategy = _custom_chunk_strategy(chunk_overlap=128, max_chunk_size=512)
# upload_pdf_files_to_vector_store(
#     vector_store_id=vector_store_details["id"],
#     list_file=list_file,
#     client=client,
#     chunking_strategy=custom_strategy,
# )

list_store = client.vector_stores.list()
all_stores = list(client.vector_stores.list(limit=100))
for store in all_stores:
    client.vector_stores.delete(vector_store_id=store.id)

# list_store_file = client.vector_stores.files.list(
#     vector_store_id=vector_store_details["id"]
# )

print(f"List Store: {list_store}")
# print(f"\nList file of store: {list_store_file}")
