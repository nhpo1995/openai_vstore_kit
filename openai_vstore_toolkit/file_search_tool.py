from langchain_core.tools import tool
from openai import OpenAI

from openai_vstore_toolkit.rag_services import FileService, StoreService

client = OpenAI()


@tool
def file_search_tool(query: str) -> str:
    """Search for files in the vector store.

    Args:
        query (str): The search query.

    Returns:
        str: The search results including quotes with score.
    """
    store_Service = StoreService(client=client)
    current_store_id = store_Service._list_store_id()[0]
    file_service = FileService(client=client, store_id=current_store_id)
    results = file_service.semantic_retrieve(query)
    return results
