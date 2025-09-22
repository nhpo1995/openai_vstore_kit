import re
from typing import Dict, List, Optional, Union
from openai import OpenAI, NOT_GIVEN, NotGiven
from openai.types import (
    FileChunkingStrategyParam,
    FilePurpose,
    FileObject,
    StaticFileChunkingStrategyObjectParam,
)
from loguru import logger
from openai.types.responses import (
    Response,
    ResponseOutputItem,
    ResponseFileSearchToolCall,
)
from openai.types.responses.response_file_search_tool_call import Result
import requests
from io import BytesIO
import mimetypes
import magic
import os
import urllib.parse
from openai_vstore_toolkit._exceptions import (
    DuplicateFileNameError,
    FileExtensionError,
    FileExtensionError,
)
from openai_vstore_toolkit._models import ResultDetail, FileSearchResponse, FileDetail
from openai_vstore_toolkit._helpers import Helpers


class FileService:
    """Manage files within a specific Vector Store.
    Thin wrapper around the OpenAI VectorStoreFile API.
    Exposes minimal methods for higher layers (endpoints) to call.
    """

    def __init__(self, client: OpenAI, store_id: str):
        if not store_id:
            raise ValueError("store_id cannot be empty when initializing FileManager.")
        self._client = client
        self._store_id = store_id
        self._helper = Helpers()

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
            file_details = self._helper.get_file_detail(file_paths=[file_path])
            file_detail = file_details[0]
            logger.debug(f"file_detail: {file_detail}")
            if not file_detail:
                raise FileExtensionError("Failed to get file detail.")
            file_name = file_detail.file_name
            file_content = file_detail.content
            file_response = self._client.files.create(
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
            f"Adding file '{file_object.filename}' to vector store {self._store_id}..."
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
            vector_store_file = self._client.vector_stores.files.create_and_poll(
                vector_store_id=self._store_id,
                file_id=file_object.id,
                chunking_strategy=chunking_strategy,
                attributes=merged_attrs,
            )
            if vector_store_file:
                logger.success(
                    f"Successfully attached file {vector_store_file.id} to store {self._store_id}."
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
        logger.info(f"Fetching all files from vector store {self._store_id}...")
        files_as_dicts: List[dict] = []
        try:
            after = NotGiven = NOT_GIVEN
            while True:
                resp = self._client.vector_stores.files.list(
                    vector_store_id=self._store_id, limit=limit, after=after
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
            logger.error(f"Failed to list files for store {self._store_id}: {e}")
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
            result = self._client.vector_stores.files.update(
                vector_store_id=self._store_id, file_id=file_id, attributes=attribute
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
            f"Attempting to delete file {file_id} from store {self._store_id}..."
        )
        try:
            response = self._client.vector_stores.files.delete(
                vector_store_id=self._store_id, file_id=file_id
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
            response = self._client.responses.create(
                model=model,
                instructions="You are a helpful RAG assistant who always responds in fluent, natural Vietnamese and natural English.\n\nOnly answer questions using information returned by the file_search tool.\n\nIf file_search returns no result or lacks enough information, reply only: 'No answer.\n\nDo not guess or use outside knowledge.\n\nIf the file_search result includes a relevant table, you may include it in Markdown format if it helps clarify your answer.\n\nKeep your answers accurate, concise, and clearly structured in the language of user. Prioritize clarity and usefulness for the reader.",
                input=query,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [self._store_id],
                        "max_num_results": top_k,
                    }
                ],
                include=["file_search_call.results"],
            )
            # logger.debug(f"Response:\n{response}")
            return FileService._final_answer_with_guardrails(response=response)
        except Exception as e:
            logger.error(f"Failed to semantic_retrieve on store {self._store_id}: {e}")
            raise

    @staticmethod
    def extract_sources(response) -> List[FileSearchResponse]:
        """
        Extract referenced filenames from a Response (file_search results).

        Args:
            response: The Response object to inspect.

        Returns:
            List[FileSearchResponse]: A List of FileSearchResponse objects.
            FileSearchResponse{
                file_id: The unique ID of the file
                detail: List[ResultDetail] {
                    attributes: file's attributes
                    quote: content of the file
                    score: Optional[float] = None
                }
            }

        Raises:
            None
        """
        try:
            keys: set = set()
            sources: List[FileSearchResponse] = []
            for output in getattr(response, "output", []) or []:
                if (
                    isinstance(output, ResponseFileSearchToolCall)
                    and getattr(output, "results", None)
                    and output.results
                ):
                    for result in output.results:
                        file_id = result.file_id
                        if file_id not in keys:
                            keys.add(result.file_id)
                            result_detail = ResultDetail(
                                attributes=result.attributes,
                                file_name=result.filename,
                                quote=result.text,
                                score=result.score,
                            )
                            search_response = FileSearchResponse(
                                file_id=file_id, details=[result_detail]
                            )
                            sources.append(search_response)
                        else:
                            result_detail = ResultDetail(
                                attributes=result.attributes,
                                file_name=result.filename,
                                quote=result.text,
                                score=result.score,
                            )
                            for source in sources:
                                if source.file_id == file_id:
                                    source.details.append(result_detail)
            return sources

        except Exception:
            return []

    @staticmethod
    def _final_answer_with_guardrails(response: Response) -> str:
        """
        Compose a final answer string with basic guardrails and references.

        Args:
            response: The Response object returned by the API.

        Returns:
            str: A formatted string that includes referenced sources (if any) and
            the model's textual output.
        """
        sources = FileService.extract_sources(response=response)
        quotes = ""
        for source in sources:
            quotes += f"File name: {source.file_id}\n"
            for d in source.details:
                quotes += f"Quote: {d.quote}\n"
        if quotes == "":
            return "No answer"
        text = response.output_text
        if text == "" or text == "No answer":
            return "No answer"
        return f"References:\n{quotes}\n\nAI:{text}"
