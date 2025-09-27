from typing import Dict, List, Optional

from loguru import logger
from openai import NOT_GIVEN, NotGiven, OpenAI
from openai.types import (
    FileChunkingStrategyParam,
    FileObject,
    FilePurpose,
    StaticFileChunkingStrategyObjectParam,
)
from openai.types.responses import (
    Response,
    ResponseFileSearchToolCall,
)

from openai_vstore_toolkit.utils import (
    DuplicateFileNameError,
    FileExtensionError,
    FileSearchResponse,
    Helper,
)


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
        self._helper = Helper()

    def custom_chunk_strategy(
        self,
        max_chunk_size: int = 800,
        chunk_overlap: int = 400,
    ) -> StaticFileChunkingStrategyObjectParam:
        """
        Create a static chunking strategy dictionary.
        Args:
            max_chunk_size (int): Maximum chunk size in tokens (100-4096).
            chunk_overlap (int): Overlap size in tokens (0 to max_chunk_size/2).
        Returns:
            StaticFileChunkingStrategyObjectParam: The chunking strategy dict.
        Raises:
            ValueError: If parameters are out of valid ranges.
        """
        if max_chunk_size < 100 or max_chunk_size > 4096:
            raise ValueError("max_chunk_size must be in [100, 4096].")
        if chunk_overlap < 0 or chunk_overlap > max_chunk_size // 2:
            raise ValueError("chunk_overlap must be <= half of max_chunk_size.")

        return {
            "type": "static",
            "static": {
                "max_chunk_size_tokens": max_chunk_size,
                "chunk_overlap_tokens": chunk_overlap,
            },
        }

    def _prepare_and_create_file_object(
        self, path: str, purpose: FilePurpose = "assistants"
    ) -> FileObject:
        """
        Prepare and create a FileObject from a staged path.
        Args:
            path (str): The staged file path.
            purpose (FilePurpose): The purpose for the file upload.
        Returns:
            FileObject: The created FileObject.
        Raises:
            FileExtensionError: If file detail extraction fails.
        """
        details = self._helper.get_file_detail(file_paths=[path])
        if not details or not details[0]:
            raise FileExtensionError(
                f"Failed to get file detail for staged path: {path}"
            )
        fd = details[0]
        try:
            fd.content.seek(0)
        except Exception:
            pass
        return self._client.files.create(
            file=(fd.file_name, fd.content, fd.mime_type),
            purpose=purpose,
        )

    def create_file_object(
        self, file_path: str, purpose: FilePurpose = "assistants"
    ) -> FileObject:
        """
        Create a FileObject directly from a local or url file path.
        Args:
            file_path (str): The local file path.
            purpose (FilePurpose): The purpose for the file upload.
        Returns:
            FileObject: The created FileObject.
        Raises:
            FileExtensionError: If file detail extraction fails.
        """
        try:
            details = self._helper.get_file_detail(file_paths=[file_path])
            if not details or not details[0]:
                raise FileExtensionError("Failed to get file detail.")
            fd = details[0]
            logger.debug(f"file_detail: {fd}")
            return self._client.files.create(
                file=(fd.file_name, fd.content, fd.mime_type),
                purpose=purpose,
            )
        except Exception as e:
            logger.error(f"Failed to create file object from '{file_path}': {e}")
            raise

    def add(
        self,
        file_object: FileObject,
        chunking_strategy: FileChunkingStrategyParam | NotGiven = NOT_GIVEN,
        attributes: Dict = {},
    ) -> Optional[str]:
        """
        Attach an existing uploaded file to the Vector Store.
        Prefer ingest_path(...) in most cases to avoid "File type not supported".
        """
        logger.info(
            f"Adding file '{file_object.filename}' to vector store {self._store_id}..."
        )
        try:
            file_id = self.find_id_by_name(file_object.filename)
            if file_id:
                raise DuplicateFileNameError("File name is already exist in the store!")

            merged_attrs = {
                "file_name": file_object.filename.lower(),
                **(attributes or {}),
            }
            vs_file = self._client.vector_stores.files.create_and_poll(
                vector_store_id=self._store_id,
                file_id=file_object.id,
                chunking_strategy=chunking_strategy,
                attributes=merged_attrs,
            )
            if vs_file:
                logger.success(
                    f"Successfully attached file {vs_file.id} to store {self._store_id}."
                )
                return vs_file.id
            return None
        except Exception as e:
            msg = str(e)
            if "File type not supported" in msg:
                logger.error(
                    "Vector Store rejected this file type at ATTACH step. "
                    "Use FileService.ingest_path(...) to stage/convert into supported formats first."
                )
            logger.error(f"Failed to add file '{file_object.filename}': {e}")
            raise

    def list(self, limit: int = 100) -> List[dict]:
        """
        List all files already attached to this Vector Store (auto-pagination).
        """
        logger.info(f"Fetching all files from vector store {self._store_id}...")
        files_as_dicts: List[dict] = []
        try:
            after = NOT_GIVEN
            while True:
                resp = self._client.vector_stores.files.list(
                    vector_store_id=self._store_id, limit=limit, after=after
                )
                page = [f.model_dump() for f in resp.data]
                files_as_dicts.extend(page)
                if getattr(resp, "has_more", False):
                    after = getattr(resp, "last_id", None) or (
                        resp.data[-1].id if resp.data else NOT_GIVEN
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
        """
        Find a vector store file by attribute 'file_name' (case-insensitive).
        """
        name_lc = file_name.lower()
        for file in self.list():
            attributes = file.get("attributes", {})
            if attributes is not None:
                stored = str(attributes.get("file_name", "")).strip().lower()
                if stored == name_lc:
                    return str(file.get("id", ""))
        return ""

    def update_attributes(self, attribute: Dict, file_id: str) -> bool:
        """
        Update attributes of an attached vector store file and verify.
        """
        try:
            result = self._client.vector_stores.files.update(
                vector_store_id=self._store_id, file_id=file_id, attributes=attribute
            )
            if result and isinstance(result.attributes, dict):
                ok = all(result.attributes.get(k) == v for k, v in attribute.items())
                if ok:
                    logger.success(f"Successfully updated file {file_id}.")
                    return True
            return False
        except Exception as e:
            logger.exception(f"Failed to update file {file_id}: {e}")
            raise

    def delete(self, file_id: str) -> bool:
        """
        Detach a file from the Vector Store.
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

    # ----------------- File-search powered retrieval -----------------
    def semantic_retrieve(
        self, query: str, model: str = "gpt-4o-mini", top_k: int = 10
    ) -> str:
        """
        Run a file_search-powered response limited to this vector store.
        """
        try:
            response = self._client.responses.create(
                model=model,
                instructions=(
                    "You are a helpful RAG assistant who always responds in fluent, natural Vietnamese and natural English.\n"
                    "Only answer questions using information returned by the file_search tool.\n"
                    "If file_search returns no result or lacks enough information, reply only: 'No answer.\n"
                    "Do not guess or use outside knowledge.\n"
                    "If the file_search result includes a relevant table, you may include it in Markdown.\n"
                    "Keep answers accurate, concise, clearly structured."
                ),
                input=query,
                tools=[
                    {
                        "type": "file_search",
                        "vector_store_ids": [self._store_id],
                        "max_num_results": top_k,
                        "ranking_options": {
                            "score_threshold": 0.7,
                            "ranker": "default-2024-11-15",
                        },
                    }
                ],
                include=["file_search_call.results"],
                tool_choice="auto",
            )
            logger.debug(f"Response:\n{response}")
            return FileService._final_answer_with_guardrails(response=response)
        except Exception as e:
            logger.error(f"Failed to semantic_retrieve on store {self._store_id}: {e}")
            raise

    @staticmethod
    def extract_sources(response) -> List[FileSearchResponse]:
        """
        Extract file search results from the Response object.
        Args:
            response (Response): The response object from the API call.
        Returns:
            List[FileSearchResponse]: A list of FileSearchResponse objects.
            FileSearchResponse{
                file_id (str | None): The unique ID of the file.
                filename (str | None): The name of the file.
                details (List[Result]): List of search results with details.
            }
        Raises:
            None
        """

        try:
            seen = set()
            out: List[FileSearchResponse] = []
            for output in getattr(response, "output", []) or []:
                if (
                    isinstance(output, ResponseFileSearchToolCall)
                    and output.results is not None
                ):
                    for result in output.results:
                        fid = result.file_id
                        if fid not in seen:
                            seen.add(fid)
                            out.append(
                                FileSearchResponse(
                                    file_id=fid,
                                    filename=result.filename,
                                    details=(
                                        [result] if result is not None else [result]
                                    ),
                                )
                            )
                        else:
                            for s in out:
                                if s.file_id == fid:
                                    s.details.append(result)
            return out
        except Exception:
            return []

    @staticmethod
    def _final_answer_with_guardrails(response: Response) -> str:
        """
        Compose a final answer string with basic guardrails and references.
        """
        sources = FileService.extract_sources(response=response)
        quotes = ""
        for source in sources:
            quotes += (
                f"## File name: {source.filename} | File ID: {source.file_id}\n\n---\n"
            )
            for inx, d in enumerate(source.details):
                snippet = (d.text or "")[:100].strip()
                quotes += f"\n### Quote {inx + 1} | Score: {getattr(d, 'score', None)}\n\n**first 100 token Content**:\n\n{snippet}...\n"
        if quotes == "":
            return "No answer"
        text = response.output_text
        if text in ("", "No answer"):
            return "No answer"
        return f"References:\n{quotes}\n---\n\n## AI_answer:\n{text}"
