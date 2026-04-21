class ParseFailedError(Exception):
    """Raised when VLM returns malformed output after all retry attempts."""


class UnsupportedModelError(ValueError):
    """Raised when an unknown model_id is passed to LLMClientFactory."""
