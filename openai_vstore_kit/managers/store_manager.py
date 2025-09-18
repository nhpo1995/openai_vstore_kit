from typing import Any, Dict, List, Optional
from openai import OpenAI
from loguru import logger


class StoreManager:
    """Manages the lifecycle of Vector Stores."""

    def __init__(self, client: OpenAI):
        self.client = client

    def create(self, store_name: str) -> Optional[str]:
        """Creates a new vector store and returns its ID."""
        logger.info(f"Creating a new vector store named '{store_name}'...")
        try:
            vector_store = self.client.vector_stores.create(name=store_name)
            logger.success(
                f"Successfully created vector store '{store_name}' with ID: {vector_store.id}"
            )
            return vector_store.id
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            return None

    def get(self, store_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin chi tiết của một vector store và trả về dưới dạng dictionary.
        Lý do: Cung cấp sự linh hoạt khi cần thiết.
        """
        logger.info(f"Fetching details for vector store {store_id}...")
        try:
            vector_store = self.client.vector_stores.retrieve(vector_store_id=store_id)
            return vector_store.model_dump()
        except Exception as e:
            logger.error(f"Failed to retrieve vector store {store_id}: {e}")
            return None

    def list(self) -> List[dict]:
        """
        Lists all vector stores, handling pagination automatically.
        Ref: https://platform.openai.com/docs/api-reference/vector-stores/list
        """
        logger.info("Fetching all vector stores...")
        try:
            # The SDK's iterator handles pagination automatically.
            # We use list() to consume the iterator and fetch all pages.
            all_stores = list(self.client.vector_stores.list(limit=100))
            stores_as_dicts = [store.model_dump() for store in all_stores]
            logger.info(f"Found {len(stores_as_dicts)} vector stores.")
            return stores_as_dicts
        except Exception as e:
            logger.error(f"Failed to list vector stores: {e}")
            return []

    def delete(self, store_id: str) -> bool:
        """Deletes a vector store by its ID. Returns True if successful."""
        logger.info(f"Attempting to delete vector store with ID: {store_id}...")
        try:
            response = self.client.vector_stores.delete(vector_store_id=store_id)
            if response.deleted:
                logger.success(f"Successfully deleted vector store {store_id}.")
                return True
            else:
                logger.warning(
                    f"Delete request for store {store_id} did not confirm deletion."
                )
                return False
        except Exception as e:
            logger.error(f"Failed to delete vector store {store_id}: {e}")
            return False
