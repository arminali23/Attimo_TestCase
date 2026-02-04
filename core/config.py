import os 
from dotenv import load_dotenv

load_dotenv()

def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default
    
    
class Settings:
    
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
    CHROMA_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma").strip()
    COLLECTION: str = os.getenv("CHROMA_COLLECTION", "docs").strip()
    TOP_K: int = _get_int("TOP_K", 4)
    MAX_CONTEXT_CHARS: int = _get_int("MAX_CONTEXT_CHARS", 12000)
    LLM_TIMEOUT_SECONDS: int = _get_int("LLM_TIMEOUT_SECONDS", 4)
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").strip()

settings = Settings()