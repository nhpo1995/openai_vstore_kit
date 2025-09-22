from typing import List
from openai.types.responses.response import Response


def format_results(results):
    formatted_results = ""
    for result in results.data:
        formatted_result = (
            f"<result file_id='{result.file_id}' file_name='{result.file_name}'>"
        )
        for part in result.content:
            formatted_result += f"<content>{part.text}</content>"
        formatted_results += formatted_result + "</result>"
    return f"<sources>{formatted_results}</sources>"


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


def final_answer_with_guardrails(self, response: Response) -> str:
    """
    Compose a final answer string with basic guardrails and references.

    Args:
        response: The Response object returned by the API.

    Returns:
        str: A formatted string that includes referenced sources (if any) and
        the model's textual output.
    """
    sources = extract_sources(response=response)
    text = response.output_text
    if sources is not []:
        return "No answer"
    else:
        return f"References:\n{'\n'.join(sources)}\n\nAI:{text}"
