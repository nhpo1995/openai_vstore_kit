from typing import Dict, List, Optional, Union
from openai import OpenAI, NOT_GIVEN, NotGiven
from openai.types import (
    FileChunkingStrategyParam,
    FilePurpose,
    FileObject,
    StaticFileChunkingStrategyObjectParam,
)
from loguru import logger
from openai.types.responses.response import Response
from openai.types.responses.response_output_item import ResponseOutputItem
import requests
from io import BytesIO
import os
from openai_vstore_toolkit._exceptions import DuplicateFileNameError


class FileService:
    """Manage files within a specific Vector Store.
    Thin wrapper around the OpenAI VectorStoreFile API.
    Exposes minimal methods for higher layers (endpoints) to call.
    """

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
        """Create an OpenAI file object from a local path or a URL.
        Ref:
        Args:
            file_path: Local file path or http(s) URL.
            purpose: OpenAI file purpose (default: "assistants").

        Returns:
            The created OpenAI FileObject.

        Raises:
            Exception: If upload fails.
        """
        try:
            if file_path.startswith("http://") or file_path.startswith("https://"):
                # Download the file content from the URL
                response = requests.get(file_path)
                file_content = BytesIO(response.content)
                file_name = file_path.split("/")[-1]
                file_tuple = (file_name, file_content)
                file_response = self.client.files.create(
                    file=file_tuple, purpose=purpose
                )
            else:
                # Handle local file path
                with open(file_path, mode="rb") as file_content:
                    file_name = os.path.basename(file_path)
                    file_response = self.client.files.create(
                        file=(file_name, file_content), purpose=purpose
                    )
            return file_response
        except Exception as e:
            logger.error(f"Failed to create file object from '{file_path}': {e}")
            raise

    def custom_chunk_strategy(
        self,
        max_chunk_size: int = 800,
        chunk_overlap: int = 400,
    ) -> StaticFileChunkingStrategyObjectParam:
        """Build a static chunking strategy object.

        Args:
            max_chunk_size: Max tokens per chunk (min 100, max 4096).
            chunk_overlap: Overlap tokens between chunks (must be ≤ half of chunk size).

        Returns:
            StaticFileChunkingStrategyObjectParam compatible dict.
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
        """Attach a file to the vector store (create_and_poll). Return vector store file ID.

        Args:
            file_object: The created FileObject to attach.
            chunking_strategy: Chunking strategy for indexing.
            attributes: Additional attributes stored on the vector store file.

        Returns:
            Vector store file ID if successful.

        Raises:
            Exception: If the operation fails (duplicate or API error).
        """
        logger.info(
            f"Adding file '{file_object.filename}' to vector store {self.store_id}..."
        )
        try:
            file_id = self.find_id_by_name(file_object.filename)
            if file_id:
                raise DuplicateFileNameError("File name is already exist in the store!")
            logger.info(
                f"File uploaded with ID: {file_object.id}. Attaching to vector store..."
            )
            base_attrs = {"file_name": file_object.filename.lower()}
            merged_attrs = {**base_attrs, **(attributes or {})}
            vector_store_file = self.client.vector_stores.files.create_and_poll(
                vector_store_id=self.store_id,
                file_id=file_object.id,
                chunking_strategy=chunking_strategy,
                attributes=merged_attrs,
            )
            if vector_store_file:
                logger.success(
                    f"Successfully attached file {vector_store_file.id} to store {self.store_id}."
                )
                return vector_store_file.id
            return None
        except Exception as e:
            logger.error(f"Failed to add file '{file_object.filename}': {e}")
            raise

    def list(self, limit: int = 100) -> List[dict]:
        """List all files in the vector store (auto pagination).

        Args:
            limit: Page size for listing.

        Returns:
            A list of file dicts.

        Raises:
            Exception: If the API call fails.
        """
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
                    # last_id may be present; fallback to last element id
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
            raise

    def find_id_by_name(self, file_name: str) -> str:
        """Find a vector store file ID by its stored attribute 'file_name' (case-insensitive).

        Args:
            file_name: The filename to search for.

        Returns:
            The vector store file ID if found, otherwise empty string.
        """
        name_lc = file_name.lower()
        for file in self.list():
            attributes = file.get("attributes", "")
            if attributes is not None:
                logger.debug(f"attributes found:\n{attributes}")
                file_name = str(attributes.get("file_name", "")).strip().lower()
                logger.debug(
                    f"attribute's file_name: {file_name} vs search_name: {name_lc}"
                )
                if file_name == name_lc:
                    logger.debug(
                        f"found file name! {file_name} match search name {name_lc}"
                    )
                    return str(object=file.get("id"))
        return ""

    def update_attributes(self, attribute: Dict, file_id: str) -> bool:
        """Update attributes of an attached vector store file.

        Args:
            attribute: Key-value attributes to set.
            file_id: The vector store file ID.

        Returns:
            True if attributes applied and verified, otherwise False.

        Raises:
            Exception: If the API call fails.
        """
        try:
            result = self.client.vector_stores.files.update(
                vector_store_id=self.store_id, file_id=file_id, attributes=attribute
            )
            if result and isinstance(result.attributes, dict):
                if all(
                    key in result.attributes and result.attributes[key] == val
                    for key, val in attribute.items()
                ):
                    logger.success(f"Successfully updated file {file_id}.")
                    return True
            return False
        except Exception as e:
            logger.exception(f"Failed to update file {file_id}: {e}")
            raise

    def delete(self, file_id: str) -> bool:
        """Delete (detach) a file from the vector store.

        Args:
            file_id: The vector store file ID.

        Returns:
            True if deleted, otherwise False.

        Raises:
            Exception: If the API call fails.
        """
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
            raise

    def semantic_retrieve(
        self, query: str, model: str = "gpt-4o-mini", top_k: int = 10
    ) -> Union[Response, str]:
        """Run a file-search-powered response limited to this store.

        Args:
            query: User query.
            model: Model name.
            top_k: Max number of file search results.

        Returns:
            OpenAI Response object or a formatted string with references and answer.

        Raises:
            Exception: If the API call fails.
        """
        try:
            response = self.client.responses.create(
                model=model,
                instructions="You are a helpful Vietnamese assistant who always responds in fluent, natural Vietnamese.\n\nOnly answer questions using information returned by the file_search tool.\n\nIf file_search returns no result or lacks enough information, reply only: 'No Answer.\n\nDo not guess or use outside knowledge.\n\nIf the file_search result includes a relevant table, you may include it in Markdown format if it helps clarify your answer.\n\nKeep your answers accurate, concise, and clearly structured in Vietnamese. Prioritize clarity and usefulness for the reader.",
                input=query,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [self.store_id],
                        "max_num_results": top_k,
                    }
                ],
                include=["file_search_call.results"],
            )
            # logger.debug(f"Response:\n{response}")
            sources = set()
            for output in response.output:
                if output.type == "file_search_call" and output.results:
                    for result in output.results:
                        sources.add(result.filename)
            sources_text = f"References:\n{'\n\n'.join(sources)}"
            return f"{sources_text}\n\nanswer:\n{response.output_text}"
        except Exception as e:
            logger.error(f"Failed to semantic_retrieve on store {self.store_id}: {e}")
            raise
