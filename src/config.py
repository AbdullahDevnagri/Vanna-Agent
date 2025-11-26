import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_SRC_DIR = os.path.dirname(__file__)
_DATA_DIR = os.path.join(_SRC_DIR, "data")


def _resolve_path(env_value: Optional[str], default_filename: str) -> str:
    """
    Resolve paths for DuckDB and Chroma so they always point to files under src/data
    unless explicitly overridden via env vars.
    """
    if env_value:
        return os.path.abspath(env_value)
    return os.path.join(_DATA_DIR, default_filename)


class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    VANNA_API_KEY = os.getenv("VANNA_API_KEY")
    VANNA_MODEL = os.getenv("VANNA_MODEL", "gpt-4o")
    DUCKDB_PATH = _resolve_path(os.getenv("DUCKDB_PATH"), "your_database.duckdb")
    CHROMA_PERSIST_DIR = _resolve_path(os.getenv("CHROMA_PERSIST_DIR"), "chroma_db")
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))


# Ensure data directories exist so ingestion + vector store can write immediately.
os.makedirs(os.path.dirname(Settings.DUCKDB_PATH), exist_ok=True)
os.makedirs(Settings.CHROMA_PERSIST_DIR, exist_ok=True)


settings = Settings()
