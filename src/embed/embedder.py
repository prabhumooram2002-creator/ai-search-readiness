"""Embedding client — NVIDIA NIM (free) primary, DeepSeek + Google fallbacks.
   Uses content-hash caching so re-runs don't waste API calls.
   Retry with exponential backoff per Section 8 of the spec.
"""
import asyncio
import hashlib
import json
from typing import Optional
import httpx

from ..core.config import (
    NVIDIA_API_KEY, NVIDIA_API_KEY_2,
    DEEPSEEK_API_KEY, GOOGLE_API_KEY,
    embed as embed_cfg,
    CACHE_DIR,
)
from ..core.logging import get_logger

logger = get_logger(__name__)

NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
DEEPSEEK_EMBED_URL = "https://api.deepseek.com/v1/embeddings"
GOOGLE_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# ─── Cache ────────────────────────────────────────────────────────────────────
def _embed_tag() -> str:
    """Model-aware cache namespace so local (1024-dim) and nvidia (other dim)
    embeddings never collide in the on-disk cache."""
    from .. import providers
    if providers._resolve_embed_provider() == "local":
        return providers.EMBED_MODEL.replace("/", "_")
    return "nvidia"

def _cache_path(content_hash: str) -> str:
    return str(CACHE_DIR / f"{_embed_tag()}__{content_hash}.json")

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]

def _cache_get(content_hash: str) -> Optional[dict]:
    try:
        with open(_cache_path(content_hash)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def _cache_set(content_hash: str, data: dict) -> None:
    with open(_cache_path(content_hash), "w") as f:
        json.dump(data, f)

# ─── NVIDIA Embedding ─────────────────────────────────────────────────────────
# NVIDIA NIM nvidia/nv-embed-v1 has a 4096-token limit per input
# At ~3 chars/token, that's ~8000 chars — use 8000 to be safe
MAX_CHARS = 4_000

def _truncate(text: str, max_chars: int = MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


async def _nvidia_embed_with_retry(texts: list[str], model: str = "nvidia/nv-embed-v1") -> list[list[float]]:
    """Call NVIDIA NIM with exponential backoff on 400/429/5xx."""
    import asyncio
    keys = [k for k in [NVIDIA_API_KEY, NVIDIA_API_KEY_2] if k]
    last_exc = None

    for attempt in range(4):  # 4 attempts total
        for key_i, key in enumerate(keys):
            headers = {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            }
            payload = {
                "input": [_truncate(t) for t in texts],
                "model": model,
                "encoding_format": "float",
            }
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(NVIDIA_EMBED_URL, headers=headers, json=payload)
                    if resp.status_code == 400:
                        body = resp.text[:200]
                        logger.warning(f"key{key_i+1} 400 (attempt {attempt+1}/4): {body}")
                        last_exc = RuntimeError(f"400: {body}")
                        await asyncio.sleep((attempt + 1) * 2)
                        continue  # try next key
                    elif resp.status_code in (401, 403):
                        logger.warning(f"Auth error key{key_i+1}, trying next")
                        continue
                    elif resp.status_code == 429:
                        logger.warning(f"Rate-limited key{key_i+1}, backing off")
                        await asyncio.sleep((attempt + 1) * 3)
                        last_exc = RuntimeError("429")
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    return [item["embedding"] for item in data["data"]]
            except httpx.HTTPStatusError as e:
                if e.response.status_code not in (400, 429, 401, 403):
                    raise
                last_exc = e
                continue
            except httpx.RequestError as e:
                logger.warning(f"Request error key{key_i+1}: {e}")
                last_exc = e
                continue
        # All keys exhausted on this attempt
        if attempt < 3:
            wait = (attempt + 1) * 4
            logger.warning(f"Attempt {attempt+1}/4 failed, retrying in {wait}s...")
            await asyncio.sleep(wait)

    raise RuntimeError(f"NVIDIA NIM embed failed after 4 attempts: {last_exc}")


async def _nvidia_embed(texts: list[str], model: str = "nvidia/nv-embed-v1") -> list[list[float]]:
    """Public wrapper — delegates to retry version."""
    return await _nvidia_embed_with_retry(texts, model)

# ─── DeepSeek Embedding (fallback) ───────────────────────────────────────────
async def _deepseek_embed(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek API key not set")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"input": texts, "model": model}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(DEEPSEEK_EMBED_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]


# ─── Google Gemini Embedding (fallback) ──────────────────────────────────────
async def _google_embed(texts: list[str], model: str = "text-embedding-004") -> list[list[float]]:
    """Call Google Gemini embeddings API."""
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google API key not set")

    # Gemini batch embed: POST /models/{model}:batchEmbedContents
    # Request format: {"requests": [{"model": "models/text-embedding-004", "content": {"parts": [{"text": "..."}]}}]}
    url = f"{GOOGLE_EMBED_URL}/{model}:batchEmbedContents?key={GOOGLE_API_KEY}"
    payload = {
        "requests": [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}}
            for t in texts
        ]
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code == 400:
            logger.warning(f"Google 400: {resp.text[:200]}")
        resp.raise_for_status()
        data = resp.json()
        # Response: {"embeddings": [{"values": [float, ...]}, ...]}
        emb_list = data.get("embeddings", [])
        return [emb["values"] for emb in emb_list]

# ─── Main embed_chunks ────────────────────────────────────────────────────────
async def embed_chunks(chunk_texts: list[str], use_cache: bool = True) -> list[list[float]]:
    """
    Embed a list of chunk texts using true batch API calls.
    Cache hit → return cached immediately (no API call).
    Cache miss → call NVIDIA → on failure call DeepSeek.

    Performance: batches ALL uncached chunks into single API calls via asyncio.gather.
    """
    import os
    cfg = embed_cfg()
    nvidia_model = cfg.get("nvidia_embed_model", "nvidia/nv-embed-v1")
    deepseek_model = cfg.get("deepseek_embed_model", "text-embedding-3-small")
    google_model = cfg.get("google_embed_model", "gemini-embedding-001")
    batch_size = cfg.get("batch_size", 100)
    provider_priority = cfg.get("provider_priority", ["nvidia", "deepseek", "google"])

    # ── Phase 1: resolve cache hits/misses ──────────────────────────────────
    results: list[list[float]] = [None] * len(chunk_texts)
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []

    for i, text in enumerate(chunk_texts):
        content_hash = _content_hash(text)
        if use_cache:
            cached = _cache_get(content_hash)
            if cached:
                results[i] = cached["embedding"]
                continue
        uncached_indices.append(i)
        uncached_texts.append(text)

    logger.info(f"Embed: {len(chunk_texts) - len(uncached_indices)} cache hits, "
                f"{len(uncached_texts)} need API calls")

    if not uncached_texts:
        return results

    # ── Phase 2: batch all uncached into API calls via asyncio.gather ──────
    async def call_provider(texts: list[str]) -> list[list[float]]:
        """Route to the configured embed provider. Local BGE-M3 is the default
        (no API key); nvidia/deepseek/google only when EMBED_PROVIDER selects them."""
        from .. import providers
        if providers._resolve_embed_provider() == "local":
            # CPU-bound sentence-transformers call — run off the event loop.
            return await asyncio.to_thread(providers.embed_texts, texts)
        last_exc = None
        for provider in provider_priority:
            try:
                if provider == "nvidia" and NVIDIA_API_KEY:
                    return await _nvidia_embed(texts, nvidia_model)
                elif provider == "deepseek" and DEEPSEEK_API_KEY:
                    return await _deepseek_embed(texts, deepseek_model)
                elif provider == "google" and GOOGLE_API_KEY:
                    return await _google_embed(texts, google_model)
            except Exception as e:
                logger.warning(f"Embed provider {provider} failed: {e}")
                last_exc = e
                continue
        raise RuntimeError(f"All embedding providers failed: {last_exc}")

    # Process in batches with asyncio.gather for true parallelism
    tasks = []
    for i in range(0, len(uncached_texts), batch_size):
        tasks.append(call_provider(uncached_texts[i:i+batch_size]))

    batch_results: list[list[list[float]]] = await asyncio.gather(*tasks)

    # Flatten and cache results
    emb_idx = 0
    for batch_embs in batch_results:
        for emb in batch_embs:
            orig_i = uncached_indices[emb_idx]
            content_hash = _content_hash(chunk_texts[orig_i])
            if use_cache:
                _cache_set(content_hash, {"embedding": emb, "provider": "nvidia"})
            results[orig_i] = emb
            emb_idx += 1

    logger.info(f"Embedded {len(uncached_texts)} chunks in {len(tasks)} batch calls")
    return results


def embed_chunks_sync(chunk_texts: list[str], use_cache: bool = True) -> list[list[float]]:
    import asyncio, concurrent.futures
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(embed_chunks(chunk_texts, use_cache))
    else:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, embed_chunks(chunk_texts, use_cache))
            return future.result()