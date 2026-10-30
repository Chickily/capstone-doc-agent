import re
from pathlib import Path
from .config import settings

ALLOWED_EXT = {".pdf", ".txt", ".csv", ".xlsx"}
INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|prompts)",
    r"disregard .{0,40}instructions",
    r"reveal .{0,20}(system )?prompt",
]
NO_EVIDENCE_MSG = (
    "I couldn't find relevant information in the uploaded documents to answer "
    "that. Try rephrasing, or upload a document that covers this topic."
)


class ValidationError(ValueError):
    """Raised for bad user input; message is safe to show to the user."""


def validate_upload(filename: str, size_bytes: int) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ValidationError(f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXT)}")
    if size_bytes > settings.max_file_mb * 1024 * 1024:
        raise ValidationError(f"'{filename}' exceeds {settings.max_file_mb} MB.")
    if size_bytes == 0:
        raise ValidationError(f"'{filename}' is empty.")


def validate_question(question: str) -> str:
    q = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", question or "").strip()
    if not q:
        raise ValidationError("Please enter a question.")
    if len(q) > settings.max_question_chars:
        raise ValidationError(f"Question too long (max {settings.max_question_chars} characters).")
    if any(re.search(p, q, re.IGNORECASE) for p in INJECTION_PATTERNS):
        raise ValidationError("That request looks like a prompt-injection attempt, so I can't process it.")
    return q


def friendly_error(e: Exception) -> str:
    name = type(e).__name__
    if name == "AuthenticationError":
        return "The API key was rejected. Check the key in the sidebar or your secrets."
    if name == "RateLimitError":
        return "Rate limit reached. Wait a moment and try again."
    if name == "NotFoundError":
        return "Model not found. Check MODEL_NAME in your .env or secrets (copy the exact name from your provider)."
    if name in ("BadRequestError", "PermissionDeniedError"):
        return "The AI service rejected the request. Check that the API key matches LLM_PROVIDER and that MODEL_NAME is valid."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Couldn't reach the AI service. Check your connection and retry."
    return f"Something went wrong ({name}). Please try again."
