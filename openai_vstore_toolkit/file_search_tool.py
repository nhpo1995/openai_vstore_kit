from openai_vstore_toolkit.rag_services.file_service import FileService
from langchain_core.tools import tool


@tool
def file_search_tool(query: str) -> str:
    """Search for files in the vector store.

    Args:
        query (str): The search query.

    Returns:
        str: The search results including quotes with score.
    """
    file_service = FileService()
    results = file_service.semantic_retrieve(query)
    return results
