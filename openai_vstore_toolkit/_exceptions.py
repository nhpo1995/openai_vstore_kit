class DuplicateFileNameError(Exception):
    """Catch in case the filename are already exist in an specific store"""

    pass


class FileProcessingError(Exception):
    """Base exception for errors raised during file processing."""

    pass


class FileExtensionError(FileProcessingError):
    """Raised when a file's extension cannot be determined from its MIME type."""

    pass
