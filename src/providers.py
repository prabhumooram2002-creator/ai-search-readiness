"""Provider abstraction — the single seam between the pipeline and any vendor.

No pipeline module hardcodes a vendor. Everything goes through:

    llm_complete(prompt, *, json=False, temperature=0, ...)  -> str   # LLM_PROVIDER
    embed_texts(list[str]) -> list[vec1024]                           # EMBED_PROVIDER

Reranker, NLI, NER, and relation extraction are ALWAYS local (no provider switch),
so they live in their own modules, not here.

Defaults are fully local and require zero API keys:
    LLM_PROVIDER=ollama    LLM_MODEL=qwen2.5:7b-instruct
    EMBED_PROVIDER=local   EMBED_MODEL=BAAI/bge-m3   (1024-dim)

If a provider is selected but its key is absent, we log a warning and fall back to
the local default. Heavy imports (ollama, sentence-transformers) are lazy so that
importing this module is cheap and never fails when models aren't pulled yet.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from dotenv import load_dotenv

from .core.logging import get_logger

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent

# Load .env then .env.local (local overrides), matching core.config's convention.
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.local", override=True)

# ── Provider flags (defaults fully local) ───────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct").strip()
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local").strip().lower()
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3").strip()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "").strip()

EMBED_DIM = 1024  # BGE-M3 == NVIDIA nv-embedqa-e5-v5 == 1024; Kuzu FLOAT[1024] schema

# ── Lazy singletons ─────────────────────────────────────────────────────────
_st_model = None       # sentence-transformers BGE-M3
_st_model_lock = threading.Lock()
_ollama_ok: bool | None = None


# ─────────────────────────────────────────────────────────────────────────────
# LLM completion
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_llm_provider() -> str:
    """Pick the effective LLM provider, honoring fallback-to-local on missing key."""
    provider = LLM_PROVIDER
    if provider == "deepseek" and not DEEPSEEK_API_KEY:
        logger.warning("LLM_PROVIDER=deepseek but DEEPSEEK_API_KEY is unset -> falling back to local ollama")
        provider = "ollama"
    return provider


def _ollama_complete(prompt: str, *, system: str | None, json: bool,
                     temperature: float, max_tokens: int | None) -> str:
    import ollama  # lazy
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    options: dict = {"temperature": temperature}
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    resp = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        format="json" if json else None,
        options=options,
    )
    return resp["message"]["content"]


def _deepseek_complete(prompt: str, *, system: str | None, json: bool,
                       temperature: float, max_tokens: int | None) -> str:
    import httpx  # lazy
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload: dict = {
        "model": os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if json:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    with httpx.Client(timeout=120.0) as client:
        resp = client.post("https://api.deepseek.com/v1/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def llm_complete(prompt: str, *, json: bool = False, temperature: float = 0,
                 system: str | None = None, max_tokens: int | None = None) -> str:
    """Generate a completion via the configured LLM provider. Returns raw text.

    When json=True the provider is asked to emit a single JSON object; callers
    still json.loads() the returned string (we keep one return type: str).
    """
    provider = _resolve_llm_provider()
    if provider == "deepseek":
        return _deepseek_complete(prompt, system=system, json=json,
                                  temperature=temperature, max_tokens=max_tokens)
    # default: local ollama
    return _ollama_complete(prompt, system=system, json=json,
                            temperature=temperature, max_tokens=max_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_embed_provider() -> str:
    provider = EMBED_PROVIDER
    if provider == "nvidia" and not NVIDIA_API_KEY:
        logger.warning("EMBED_PROVIDER=nvidia but NVIDIA_API_KEY is unset -> falling back to local BGE-M3")
        provider = "local"
    return provider


def _get_st_model():
    global _st_model
    if _st_model is None:
        with _st_model_lock:
            if _st_model is None:
                from sentence_transformers import SentenceTransformer  # lazy (pulls torch)
                logger.info(f"Loading local embedder {EMBED_MODEL} (first load downloads the model)...")
                _st_model = SentenceTransformer(EMBED_MODEL)
    return _st_model


def _local_embed(texts: list[str]) -> list[list[float]]:
    model = _get_st_model()
    vecs = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vecs]


def _nvidia_embed(texts: list[str]) -> list[list[float]]:
    # Reuse the repo's existing, retry-hardened NVIDIA path.
    import asyncio
    from .embed.embedder import _nvidia_embed as _nv_async  # noqa
    model = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
    return asyncio.run(_nv_async(texts, model))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts -> list of 1024-dim vectors, via the configured embed provider."""
    if not texts:
        return []
    provider = _resolve_embed_provider()
    if provider == "nvidia":
        return _nvidia_embed(texts)
    return _local_embed(texts)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics (used by pre-stage verification)
# ─────────────────────────────────────────────────────────────────────────────
def provider_info() -> dict:
    return {
        "llm_provider": LLM_PROVIDER,
        "llm_provider_effective": _resolve_llm_provider(),
        "llm_model": LLM_MODEL,
        "embed_provider": EMBED_PROVIDER,
        "embed_provider_effective": _resolve_embed_provider(),
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "deepseek_key_present": bool(DEEPSEEK_API_KEY),
        "nvidia_key_present": bool(NVIDIA_API_KEY),
    }
