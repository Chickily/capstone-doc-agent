import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # reads .env locally; does nothing on Streamlit Cloud


def get_secret(name: str, default=None):
    """Look in environment variables first, then st.secrets (Streamlit Cloud)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:  # no secrets file present
        return default


PROVIDER = (get_secret("LLM_PROVIDER", "gemini") or "gemini").strip().lower()
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "gemini": "gemini-flash-latest"}
KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


def _pick_model() -> str:
    model = (get_secret("MODEL_NAME") or "").strip()
    default = DEFAULT_MODELS.get(PROVIDER, "")
    # guard against a model name left over from the other provider
    if PROVIDER == "gemini" and model.lower().startswith("claude"):
        return default
    if PROVIDER == "anthropic" and model.lower().startswith("gemini"):
        return default
    return model or default


@dataclass(frozen=True)
class Settings:
    provider: str = PROVIDER
    model: str = _pick_model()
    api_key_name: str = KEY_NAMES.get(PROVIDER, "GEMINI_API_KEY")
    chunk_size: int = 800          # characters per chunk
    chunk_overlap: int = 150
    top_k: int = 5
    max_distance: float = 0.85     # cosine distance cutoff; tune for your data
    max_file_mb: int = 10
    max_question_chars: int = 2000
    max_table_rows: int = 5000
    max_agent_steps: int = 5
    max_questions_shared_key: int = 25   # per session, when using the app owner's key


settings = Settings()
