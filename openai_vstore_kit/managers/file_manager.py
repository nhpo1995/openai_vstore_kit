from typing import Dict, List, Literal, Optional
from openai import OpenAI, NOT_GIVEN, NotGiven
from openai.types import (
    FileChunkingStrategyParam,
    FilePurpose,
    FileObject,
    StaticFileChunkingStrategyObjectParam,
)
from loguru import logger
import requests
from io import BytesIO


class FileManager:
    """Manages files within a specific Vector Store."""

    def __init__(self, client: OpenAI, store_id: str):
        if not store_id:
            raise ValueError("store_id cannot be empty when initializing FileManager.")
        self.client = client
        self.store_id = store_id

    def create_file_object(
        self,
        file_path: str,
        purpose: FilePurpose = "assistants",
    ):
        if file_path.startswith("http://") or file_path.startswith("https://"):
            # Download the file content from the URL
            response = requests.get(file_path)
            file_content = BytesIO(response.content)
            file_name = file_path.split("/")[-1]
            file_tuple = (file_name, file_content)
            file_response = self.client.files.create(file=file_tuple, purpose=purpose)
        else:
            # Handle local file path
            with open(file_path, "rb") as file_content:
                file_response = self.client.files.create(
                    file=file_content, purpose=purpose
                )
        return file_response

    def custom_chunk_strategy(
        self,
        max_chunk_size: int = 800,
        chunk_overlap: int = 400,
    ) -> StaticFileChunkingStrategyObjectParam:
        """Create a custom chunking strategy object

        Args:
            max_chunk_size (int): The maximum number of tokens in each chunk. The default value is `800`. The minimum value is `100` and the maximum value is `4096`.
            chunk_overlap (int): The number of tokens that overlap between chunks. The default value is `400`.
            - Overlap cannot be more than half of chunk size!
        Returns:
            StaticFileChunkingStrategyObjectParam: FileChunkingStrategyParam
        """
        custom_chunking_strategy: StaticFileChunkingStrategyObjectParam = {
            "type": "static",
            "static": {
                "max_chunk_size_tokens": max_chunk_size,
                "chunk_overlap_tokens": chunk_overlap,
            },
        }
        return custom_chunking_strategy

    def add(
        self,
        file_object: FileObject,
        chunking_strategy: FileChunkingStrategyParam | NotGiven = NOT_GIVEN,
        attributes: Dict = {},
    ) -> Optional[str]:
        """
        Uploads a file and attaches it to the vector store.
        Returns the Vector Store File ID if successful.
        """
        logger.info(
            f"Adding file '{file_object.filename}' to vector store {self.store_id}..."
        )
        try:
            logger.info(
                f"File uploaded with ID: {file_object.id}. Attaching to vector store..."
            )
            vector_store_file = self.client.vector_stores.files.create_and_poll(
                vector_store_id=self.store_id,
                file_id=file_object.id,
                chunking_strategy=chunking_strategy,
                attributes=(
                    {"file_name": file_object.filename.lower()}.update(attributes)
                    if attributes
                    else {"file_name": file_object.filename.lower()}
                ),
            )
            if vector_store_file:
                logger.success(
                    f"Successfully attached file {vector_store_file.id} to store {self.store_id}."
                )
                return vector_store_file.id
            return None
        except Exception as e:
            logger.error(f"Failed to add file '{file_object.filename}': {e}")
            return None

    def list(self, limit: int = 100) -> List[dict]:
        """Lists all files in the vector store."""
        logger.info(f"Fetching all files from vector store {self.store_id}...")
        files_as_dicts: List[dict] = []
        try:
            after = None
            while True:
                resp = self.client.vector_stores.files.list(
                    vector_store_id=self.store_id, limit=limit, after=after  # type: ignore
                )
                page = [f.model_dump() for f in resp.data]
                files_as_dicts.extend(page)
                if getattr(resp, "has_more", False):
                    # last_id có thể có sẵn; fallback lấy id phần tử cuối
                    after = getattr(resp, "last_id", None) or (
                        resp.data[-1].id if resp.data else None
                    )
                    if not after:
                        break
                else:
                    break
                logger.info(f"Found {len(files_as_dicts)} files.")
            return files_as_dicts
        except Exception as e:
            logger.error(f"Failed to list files for store {self.store_id}: {e}")
            return []

    def find_id_by_name(self, file_name: str) -> str:
        name_lc = file_name.lower()
        for file in self.list():
            attribute = file.get("attributes", "")
            if attribute:
                if str(attribute.get("file_name", "")).strip().lower == name_lc:
                    return str(file.get("id"))
        return ""

    def update_attributes(self, attribute: Dict, file_id: str):
        """Update the attributes to an created file"""
        self.client.vector_stores.files.update(
            vector_store_id=self.store_id, file_id=file_id, attributes=attribute
        )

    def delete(self, file_id: str) -> bool:
        """Deletes a file from the vector store."""
        logger.info(
            f"Attempting to delete file {file_id} from store {self.store_id}..."
        )
        try:
            response = self.client.vector_stores.files.delete(
                vector_store_id=self.store_id, file_id=file_id
            )
            if response.deleted:
                logger.success(f"Successfully deleted file {file_id}.")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file {file_id}: {e}")
            return False

    def semantic_retrieve(query: str):
        