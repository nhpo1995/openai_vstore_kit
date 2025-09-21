from typing import Literal, Optional, Dict, List
from openai import OpenAI, NotGiven, NOT_GIVEN
from openai.types.conversations import Conversation
from loguru import logger


class ConversationService:
    """Manage Conversation
    Thin wrapper around the OpenAI Conversations API.
    Exposes minimal methods for higher layers (endpoints) to call.
    """

    def __init__(self, client: OpenAI):
        self._client = client

    # --- Create ---
    def create(self, metadata: Optional[Dict] = None) -> str:
        """
        Create a conversation.
        Ref: https://platform.openai.com/docs/api-reference/conversations/create
        Args:
            metadata: Optional metadata to attach.

        Returns:
            The conversation ID.

        Raises:
            Exception: If the API call fails.
        """
        try:
            conv = self._client.conversations.create(metadata=metadata or {})
            logger.info(f"Created conversation: {conv.id}")
            return conv.id
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise

    # --- Retrieve ---
    def get(self, conversation_id: str) -> Conversation:
        """
        Retrieve a conversation.
        Ref: https://platform.openai.com/docs/api-reference/conversations/retrieve
        Args:
            conversation_id: ID of the conversation.

        Returns:
            Conversation object.
            Conversation{
                "id": "conv_123",
                "object": "conversation",
                "created_at": 1741900000,
                "metadata": {"topic": "project-x"}
            }
        Raises:
            Exception: If the API call fails.
        """
        try:
            conv = self._client.conversations.retrieve(conversation_id)
            return conv
        except Exception as e:
            logger.error(f"Failed to retrieve conversation {conversation_id}: {e}")
            raise

    # --- Update Metadata ---
    def update(self, conversation_id: str, metadata: Dict[str, str]) -> Conversation:
        """
        Update metadata for a conversation.
        Ref: https://platform.openai.com/docs/api-reference/conversations/update
        Args:
            conversation_id: ID of the conversation.
            metadata: Metadata of the conversation.

        Returns:
            A conversation that is updated.
            Conversation{
                "id": "conv_123",
                "object": "conversation",
                "created_at": 1741900000,
                "metadata": {"topic": "project-x"}(Updated metadata)
            }

        Raises:
            Exception: If the API call fails.
        """
        try:
            conv = self._client.conversations.update(
                conversation_id=conversation_id, metadata=metadata
            )
            return conv
        except Exception as e:
            logger.error(
                f"Failed to update metadata: {metadata} for conversation {conversation_id}: {e}"
            )
            raise

    # --- List Items (timeline) ---
    def list_items(
        self,
        conversation_id: str,
        *,
        limit: int = 50,
        after: str | NotGiven = NOT_GIVEN,
        order: Literal["asc", "desc"] | NotGiven = NOT_GIVEN,
    ) -> List[Dict]:
        """
        List timeline items (user/assistant/tool) of a conversation.
        Ref: https://platform.openai.com/docs/api-reference/conversations/list-items
        Args:
            conversation_id: ID of the conversation.
            limit: Max items per page.  Limit can range between 1 and 100, and the default is 20.
            after: Cursor for pagination.

        Returns:
            A list of items as dicts.

        Raises:
            Exception: If the API call fails.
        """
        try:
            resp = self._client.conversations.items.list(
                conversation_id=conversation_id,
                limit=limit,
                after=after,
                order=order,
            )
            return [item.model_dump() for item in resp.data]
        except Exception as e:
            logger.error(
                f"Failed to list items for conversation {conversation_id}: {e}"
            )
            raise

    # --- Delete ---
    def delete(self, conversation_id: str) -> bool:
        """
        Delete a conversation.
        Ref: https://platform.openai.com/docs/api-reference/conversations/delete
        Args:
            conversation_id: ID of the conversation.

        Returns:
            True if deleted.

        Raises:
            Exception: If the API call fails.
        """
        try:
            result = self._client.conversations.delete(conversation_id)
            ok = bool(getattr(result, "deleted", False))
            logger.info(f"Deleted conversation {conversation_id}: {ok}")
            return ok
        except Exception as e:
            logger.error(f"Failed to delete conversation {conversation_id}: {e}")
            raise
