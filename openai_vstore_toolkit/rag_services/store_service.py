from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI

from openai_vstore_toolkit.utils import Helper


class StoreService:
    """Manage the lifecycle of Vector Stores using the OpenAI API.
    Thin wrapper around the OpenAI VectorStore API.
    Exposes minimal methods for higher layers (endpoints) to call.
    """

    def __init__(self, client: OpenAI):
        self._client = client
        self._helper = Helper()

    def get_or_create(self, store_name: str) -> str:
        """Get or create a vector store by name.

        If the store exists, return its ID.
        If not, create a new store and return its ID.

        Args:
            store_name: Name of the store.

        Returns:
            The vector store ID.

        Raises:
            Exception: If the API call fails.
        """
        name = self._helper.standardize_store_name(store_name)
        existed_id = self.find_id_by_name(name)
        if existed_id:
            logger.info(f"Reusing existing vector store '{name}' with ID: {existed_id}")
            return existed_id
        return self.create(name)

    def find_id_by_name(self, store_name: str) -> Optional[str]:
        """Find a vector store ID by its name (case-insensitive).

        Args:
            store_name: Name of the store.

        Returns:
            The store ID if found, otherwise None.
        """
        name_lc = store_name.strip().lower()
        for store in self.list_store():
            if str(store.get("name", "")).strip().lower() == name_lc:
                return store.get("id")
        return None

    def create(self, store_name: str) -> str:
        """Create a new vector store.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/create

        Args:
            store_name: Name of the store.

        Returns:
            The new store ID.

        Raises:
            Exception: If the API call fails.
        """
        logger.info(f"Creating a new vector store named '{store_name}'...")
        try:
            store_name = self._helper.standardize_store_name(store_name)
            vector_store = self._client.vector_stores.create(name=store_name)
            logger.success(
                f"Successfully created vector store '{store_name}' with ID: {vector_store.id}"
            )
            return vector_store.id
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    def get(self, store_id: str) -> Dict[str, Any]:
        """Retrieve vector store information as dictionary.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/retrieve

        Args:
            store_id: ID of the vector store.

        Returns:
            The store details as a dict.

        Raises:
            Exception: If the API call fails.
        """
        logger.info(f"Fetching details for vector store {store_id}...")
        try:
            vector_store = self._client.vector_stores.retrieve(vector_store_id=store_id)
            return vector_store.model_dump()
        except Exception as e:
            logger.error(f"Failed to retrieve vector store {store_id}: {e}")
            raise

    def list_store(self) -> List[Dict[Any, Any]]:
        """List all vector stores with automatic pagination.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/list

        Returns:
            A list of vector store dicts.

        Raises:
            Exception: If the API call fails.
        """
        logger.info("Fetching all vector stores...")
        stores_as_dicts: List[Dict[Any, Any]] = []
        try:
            after = None
            while True:
                resp = self._client.vector_stores.list(limit=100, after=after)  # type: ignore
                page = [vs.model_dump() for vs in resp.data]
                stores_as_dicts.extend(page)
                if getattr(resp, "has_more", False):
                    after = getattr(resp, "last_id", None) or (
                        resp.data[-1].id if resp.data else None
                    )
                    if not after:
                        break
                else:
                    break
            logger.info(f"Found {len(stores_as_dicts)} vector stores.")
            return stores_as_dicts
        except Exception as e:
            logger.error(f"Failed to list vector stores: {e}")
            raise

    def _list_store_id(self) -> List[Any]:
        """Helper: Return list of all vector store IDs."""
        all_stores = self.list_store()
        return [s.get("id") for s in all_stores] if all_stores else []

    def delete(self, store_id: str) -> bool:
        """Delete a vector store by its ID.

        Ref: https://platform.openai.com/docs/api-reference/vector-stores/delete

        Args:
            store_id: ID of the vector store.

        Returns:
            True if the store was deleted, False otherwise.

        Raises:
            Exception: If the API call fails.
        """
        logger.info(f"Attempting to delete vector store with ID: {store_id}...")
        try:
            response = self._client.vector_stores.delete(vector_store_id=store_id)
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
            raise
