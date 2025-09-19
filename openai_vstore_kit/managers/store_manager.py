from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI
from loguru import logger


class StoreManager:
    """Manages the lifecycle of Vector Stores."""

    def __init__(self, client: OpenAI):
        self.client = client
        self.store_ids = self._list_store_id()

    def get_or_create(self, store_name: str) -> Optional[str]:
        """
        Idempotent by store's name:
        - If store'name is already exist -> fallback to old id
        - If store'name isn't exist -> create new store then return its ID
        """
        name = store_name.strip()
        existed_id = self.find_id_by_name(name)
        if existed_id:
            logger.info(f"Reusing existing vector store '{name}' with ID: {existed_id}")
            return existed_id
        return self.create(name)

    def find_id_by_name(self, store_name: str) -> Optional[str]:
        """Tìm ID store theo tên (case-insensitive)."""
        """Get store'id by store's name

        Returns:
            String: store's id 
        """
        name_lc = store_name.strip().lower()
        for store in self.list_store():
            if str(store.get("name", "")).strip().lower() == name_lc:
                return store.get("id")
        return None

    def create(self, store_name: str) -> Optional[str]:
        """Creates a new vector store and returns its ID.

        Returns:
            String: Store's id
        """
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
        """Get Store information as Dictionary
            Ref: https://platform.openai.com/docs/api-reference/vector-stores/list
        Returns:
            Dictionary:
                {
                    "id": "vs_abc123",
                    "object": "vector_store",
                    "created_at": 1699061776,
                    "name": "Support FAQ",
                    "bytes": 139920,
                    "file_counts": {
                        "in_progress": 0,
                        "completed": 3,
                        "failed": 0,
                        "cancelled": 0,
                        "total": 3
                    }
                }
        """
        logger.info(f"Fetching details for vector store {store_id}...")
        try:
            vector_store = self.client.vector_stores.retrieve(vector_store_id=store_id)
            return vector_store.model_dump()
        except Exception as e:
            logger.error(f"Failed to retrieve vector store {store_id}: {e}")
            return None

    def list_store(self) -> List[dict]:
        """
        Lists all vector stores, handling pagination automatically.
        Ref: https://platform.openai.com/docs/api-reference/vector-stores/list

        Returns:
            List[dict]: List of store detail in JSON
            [
                {
                "id": "vs_abc123",
                "object": "vector_store",
                "created_at": 1699061776,
                "name": "Support FAQ",
                "bytes": 139920,
                "file_counts": {
                        "in_progress": 0,
                        "completed": 3,
                        "failed": 0,
                        "cancelled": 0,
                        "total": 3
                    }
                },
            ]
        """
        logger.info("Fetching all vector stores...")
        stores_as_dicts: List[dict] = []
        try:
            after = None
            while True:
                resp = self.client.vector_stores.list(limit=100, after=after)  # type: ignore
                # resp.data: List[VectorStore]
                page = [vs.model_dump() for vs in resp.data]
                stores_as_dicts.extend(page)
                if getattr(resp, "has_more", False):
                    # last_id có thể có sẵn; fallback lấy id phần tử cuối
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
            return []

    def _list_store_id(self) -> List[Any]:
        all_stores = self.list_store()
        if all_stores:
            return [s.get("id") for s in all_stores]
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
        
    def attach_to_assistant(self, assistant_id: str, list_store_id:List[str]):
        """
        Gắn vector store vào Assistant (beta.assistants API).
        """
        return self.client.beta.assistants.update(
            assistant_id=assistant_id,
            tool_resources={"file_search": {"vector_store_ids": list_store_id}},
        )
