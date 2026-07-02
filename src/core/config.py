"""Core configuration loader — reads .env.local + config.yaml."""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent.parent

# Load .env.local
load_dotenv(BASE_DIR / ".env.local")

# ── Secrets (from .env.local) ──────────────────────────────────────────────
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_API_KEY_2 = os.getenv("NVIDIA_API_KEY_2", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ── Config.yaml tunables ───────────────────────────────────────────────────
CONFIG_FILE = BASE_DIR / "config.yaml"

def load_config() -> dict:
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()

# ── Convenience accessors ───────────────────────────────────────────────────
def crawl():
    return CFG.get("crawl", {})

def chunk():
    return CFG.get("chunk", {})

def embed():
    return CFG.get("embed", {})

def vector():
    return CFG.get("vector", {})

def qa():
    return CFG.get("qa", {})

def scoring():
    return CFG.get("scoring", {})

# ── Derived constants ────────────────────────────────────────────────────────
EMBED_BATCH_SIZE = embed().get("batch_size", 100)
CACHE_DIR = BASE_DIR / embed().get("cache_dir", "data/.embed_cache")
PERSIST_DIR = BASE_DIR / vector().get("persist_directory", "data/chromadb")

# Ensure data dirs exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIR.mkdir(parents=True, exist_ok=True)