import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    VANNA_API_KEY = os.getenv("VANNA_API_KEY")
    VANNA_MODEL = os.getenv("VANNA_MODEL", "gpt-4o")
    DUCKDB_PATH = os.getenv("DUCKDB_PATH", "./data/your_database.duckdb")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", 5000))

settings = Settings()
