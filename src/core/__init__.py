# Core module exports
from .config import (
    NVIDIA_API_KEY,
    NVIDIA_API_KEY_2,
    DEEPSEEK_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    CFG, crawl, chunk, embed, vector, qa, scoring,
    EMBED_BATCH_SIZE, CACHE_DIR, PERSIST_DIR,
    BASE_DIR,
)
from .logging import get_logger

__all__ = [
    "NVIDIA_API_KEY", "NVIDIA_API_KEY_2", "DEEPSEEK_API_KEY", "GOOGLE_API_KEY",
    "QDRANT_URL", "QDRANT_API_KEY",
    "CFG", "crawl", "chunk", "embed", "vector", "qa", "scoring",
    "EMBED_BATCH_SIZE", "CACHE_DIR", "PERSIST_DIR", "BASE_DIR",
    "get_logger",
]