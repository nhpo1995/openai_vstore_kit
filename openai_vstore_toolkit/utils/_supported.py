"""
Single source of truth for the formats directly indexable by the OpenAI File Search tool.
If OpenAI updates support, update here and the rest of the pipeline will respect it.
"""

SUPPORTED_EXT = {
    ".c",
    ".cpp",
    ".cs",
    ".css",
    ".doc",
    ".docx",
    ".go",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".pdf",
    ".php",
    ".pptx",
    ".py",
    ".rb",
    ".sh",
    ".tex",
    ".ts",
    ".txt",
}

SUPPORTED_MIME = {
    "text/x-c",
    "text/x-c++",
    "text/x-csharp",
    "text/css",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/x-golang",
    "text/html",
    "text/x-java",
    "text/javascript",
    "application/json",
    "text/markdown",
    "application/pdf",
    "text/x-php",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/x-python",
    "text/x-script.python",
    "text/x-ruby",
    "application/x-sh",
    "text/x-tex",
    "application/typescript",
    "text/plain",
}

# Convenience set used by stager for final outputs
INDEXABLE_EXT = {
    ".pdf",
    ".doc",
    ".docx",
    ".pptx",
    ".txt",
    ".md",
    ".html",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".rb",
    ".php",
    ".tex",
    ".json",
}

"""
Single source of truth for the formats directly indexable by the OpenAI File Search tool.
If OpenAI updates support, update here and the rest of the pipeline will respect it.
"""

SUPPORTED_EXT = {
    ".c",
    ".cpp",
    ".cs",
    ".css",
    ".doc",
    ".docx",
    ".go",
    ".html",
    ".java",
    ".js",
    ".json",
    ".md",
    ".pdf",
    ".php",
    ".pptx",
    ".py",
    ".rb",
    ".sh",
    ".tex",
    ".ts",
    ".txt",
}

SUPPORTED_MIME = {
    "text/x-c",
    "text/x-c++",
    "text/x-csharp",
    "text/css",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/x-golang",
    "text/html",
    "text/x-java",
    "text/javascript",
    "application/json",
    "text/markdown",
    "application/pdf",
    "text/x-php",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/x-python",
    "text/x-script.python",
    "text/x-ruby",
    "application/x-sh",
    "text/x-tex",
    "application/typescript",
    "text/plain",
}

# Convenience set used by stager for final outputs
INDEXABLE_EXT = {
    ".pdf",
    ".doc",
    ".docx",
    ".pptx",
    ".txt",
    ".md",
    ".html",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".cs",
    ".rb",
    ".php",
    ".tex",
    ".json",
}


# ----------------- Helper functions -----------------
def is_supported_ext(ext: str) -> bool:
    """
    Check if a file extension is directly supported by File Search.
    Args:
        ext: File extension, e.g. ".pdf"
    """
    if not ext.startswith("."):
        ext = "." + ext
    return ext.lower() in SUPPORTED_EXT


def is_supported_mime(mime: str) -> bool:
    """
    Check if a MIME type is directly supported by File Search.
    Args:
        mime: MIME type string, e.g. "application/pdf"
    """
    return mime.lower() in SUPPORTED_MIME


def is_indexable_ext(ext: str) -> bool:
    """
    Check if a file extension is considered indexable in our staging pipeline.
    Args:
        ext: File extension, e.g. ".pdf"
    """
    if not ext.startswith("."):
        ext = "." + ext
    return ext.lower() in INDEXABLE_EXT
