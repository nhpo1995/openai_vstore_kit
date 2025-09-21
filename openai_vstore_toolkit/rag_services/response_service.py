from typing import Optional, List
from openai import OpenAI, NOT_GIVEN, NotGiven
from openai.types.responses.response import Response
from openai.types.shared_params import Metadata
from openai.types.responses.file_search_tool_param import (
    FileSearchToolParam,
    RankingOptions,
)
from loguru import logger


class ResponseRAGService:
    """
    Responses API wrapper dedicated to file_search (RAG) within one conversation.

    Args:
        client: OpenAI client instance.
        conversation_id: Conversation ID to scope/continue responses.
        vector_store_ids: Vector store IDs used by file_search tool.

    Raises:
        ValueError: If required parameters are missing or empty.
    """

    def __init__(
        self, client: OpenAI, conversation_id: str, vector_store_ids: List[str]
    ):
        """
        Initialize the service with a client, conversation context, and vector stores.

        Args:
            client: OpenAI client.
            conversation_id: Conversation ID to continue.
            vector_store_ids: Vector store IDs used by file_search.

        Raises:
            ValueError: If `conversation_id` or `vector_store_ids` are empty.
        """
        if not conversation_id:
            raise ValueError("conversation_id must not be empty.")
        if not vector_store_ids:
            raise ValueError("vector_store_ids must not be empty.")
        self._client = client
        self._conversation_id = conversation_id
        self._vector_store_ids = vector_store_ids

    def create(
        self,
        *,
        model: str,
        input_text: str,
        top_k: int | None,
        instructions: Optional[str] | NotGiven = NOT_GIVEN,
        metadata: Metadata | NotGiven = NOT_GIVEN,
        score_threshold: float | None,
    ) -> Response:
        """
        Create a RAG response tied to this conversation using the file_search tool.

        Ref:
            https://platform.openai.com/docs/api-reference/responses/create

        Args:
            model: Model name to use for the response.
            input_text: The user input text for generation.
            top_k: Max number of retrieval results (file_search) to use; if None, SDK default applies.
            instructions: Optional system/instructions message passed to the model.
            metadata: Optional metadata to attach to the response.
            score_threshold: Optional retrieval score threshold for ranking.

        Returns:
            Response: The created OpenAI Response object.

        Raises:
            Exception: If the API call fails for any reason.
        """
        try:
            logger.info("Creating RAG response...")
            file_search_tool = FileSearchToolParam(
                type="file_search",
                vector_store_ids=self._vector_store_ids,
            )
            if score_threshold:
                ranker = RankingOptions(ranker="auto", score_threshold=score_threshold)
                file_search_tool["ranking_options"] = ranker
            if top_k:
                file_search_tool["max_num_results"] = top_k

            response = self._client.responses.create(
                model=model,
                input=input_text,
                conversation={"id": self._conversation_id},
                tools=[file_search_tool],
                instructions=instructions,
                metadata=metadata,
            )
            return response
        except Exception as e:
            logger.error(f"Failed to create RAG response: {e}")
            raise

    def get(self, response_id: str):
        """
        Retrieve a single response by its ID.

        Ref:
            https://platform.openai.com/docs/api-reference/responses/retrieve

        Args:
            response_id: The response identifier.

        Returns:
            Response: The retrieved response object.

        Raises:
            Exception: If the API call fails for any reason.
        """
        try:
            return self._client.responses.retrieve(response_id)
        except Exception as e:
            logger.error(f"Failed to retrieve response {response_id}: {e}")
            raise

    def cancel(self, response_id: str) -> bool:
        """
        Cancel a running response.

        Ref:
            https://platform.openai.com/docs/api-reference/responses/cancel

        Args:
            response_id: The response identifier to cancel.

        Returns:
            bool: True if the response is reported as cancelled; otherwise False.

        Raises:
            Exception: If the API call fails.
        """
        try:
            result = self._client.responses.cancel(response_id)
            ok = bool(getattr(result, "cancelled", True))
            logger.info(f"Cancelled response {response_id}: {ok}")
            return ok
        except Exception as e:
            logger.error(f"Failed to cancel response {response_id}: {e}")
            raise

    @staticmethod
    def extract_sources(response) -> List[str]:
        """
        Extract referenced filenames from a Response (file_search results).

        Args:
            response: The Response object to inspect.

        Returns:
            List[str]: A sorted list of unique filenames referenced in file_search outputs.
            Returns an empty list on parsing errors.

        Raises:
            None
        """
        try:
            sources = set()
            for output in getattr(response, "output", []) or []:
                if getattr(output, "type", None) == "file_search_call" and getattr(
                    output, "results", None
                ):
                    for res in output.results:
                        fname = getattr(res, "filename", None)
                        if fname:
                            sources.add(fname)
            return sorted(sources)
        except Exception:
            return []

    def _final_answer_with_guardrails(self, response: Response) -> str:
        """
        Compose a final answer string with basic guardrails and references.

        Args:
            response: The Response object returned by the API.

        Returns:
            str: A formatted string that includes referenced sources (if any) and
            the model's textual output.
        """
        sources = ResponseRAGService.extract_sources(response=response)
        text = response.output_text
        if sources is not []:
            return "No answer"
        else:
            return f"References:\n{'\n'.join(sources)}\n\nAI:{text}"
