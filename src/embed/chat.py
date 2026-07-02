"""NVIDIA NIM chat completion — for Q&A answer generation.
   
   Primary: NVIDIA NIM (free tier)
   Fallback: DeepSeek
   Per-spec Section 0: retry transient errors, fall through to DeepSeek on auth/quota.
"""
import json
import hashlib
import httpx

from ..core.config import (
    NVIDIA_API_KEY, NVIDIA_API_KEY_2,
    DEEPSEEK_API_KEY, GOOGLE_API_KEY,
    qa as qa_cfg,
)
from ..core.logging import get_logger

logger = get_logger(__name__)

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"


async def _nvidia_chat(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Call NVIDIA NIM chat endpoint. Tries both keys for resilience."""
    keys = [k for k in [NVIDIA_API_KEY, NVIDIA_API_KEY_2] if k]
    last_exc = None

    for key in keys:
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                resp = await client.post(NVIDIA_CHAT_URL, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403, 404):
                    logger.warning(f"NVIDIA key {key[:12]}... error ({e.response.status_code}), trying next")
                    last_exc = e
                    continue
                elif e.response.status_code == 429:
                    logger.warning(f"NVIDIA rate-limited, trying next key")
                    last_exc = e
                    continue
                else:
                    raise
            except Exception as e:
                last_exc = e
                continue

    raise RuntimeError(f"NVIDIA chat failed on all {len(keys)} keys: {last_exc}")


async def _deepseek_chat(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek API key not set")
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(DEEPSEEK_CHAT_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def chat_complete(
    prompt: str,
    system_prompt: str = "",
    context_chunks: list[str] = None,
    use_cache: bool = True,
) -> str:
    """
    Generate a chat completion.
    Caches by prompt hash.
    """
    cfg = qa_cfg()
    provider_priority = cfg.get("provider_priority", ["nvidia", "deepseek"])
    nvidia_model = cfg.get("nvidia_chat_model", "nvidia/llama-3.3-nemotron-super-49b-v1")
    deepseek_model = cfg.get("deepseek_chat_model", "deepseek-chat")
    max_tokens = cfg.get("max_tokens", 512)
    temperature = cfg.get("temperature", 0.3)

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Build context block from retrieved chunks
    if context_chunks:
        context_text = "\n\n".join(
            f"[Source {i+1}]: {c}" for i, c in enumerate(context_chunks)
        )
        messages.append({
            "role": "system",
            "content": f"You are a helpful assistant answering based ONLY on the provided sources. "
                      f"If the answer is not in the sources, say you don't know.\n\nSources:\n{context_text}"
        })

    messages.append({"role": "user", "content": prompt})

    # Cache key
    cache_key = hashlib.sha256(
        json.dumps({"messages": messages, "model": nvidia_model}, sort_keys=True).encode()
    ).hexdigest()[:32]

    cache_path = f"data/.chat_cache/{cache_key}.json"
    import os
    from pathlib import Path
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
            logger.debug(f"Chat cache hit: {cache_key[:8]}...")
            return cached["response"]

    last_error = None
    for provider in provider_priority:
        if provider == "nvidia" and NVIDIA_API_KEY:
            try:
                response = await _nvidia_chat(
                    messages, model=nvidia_model,
                    max_tokens=max_tokens, temperature=temperature,
                )
                with open(cache_path, "w") as f:
                    json.dump({"response": response}, f)
                return response
            except Exception as e:
                logger.warning(f"NVIDIA chat failed: {e}")
                last_error = e
                continue

        elif provider == "deepseek" and DEEPSEEK_API_KEY:
            try:
                response = await _deepseek_chat(
                    messages, model=deepseek_model,
                    max_tokens=max_tokens, temperature=temperature,
                )
                with open(cache_path, "w") as f:
                    json.dump({"response": response}, f)
                return response
            except Exception as e:
                logger.warning(f"DeepSeek chat failed: {e}")
                last_error = e
                continue

    raise RuntimeError(f"All chat providers failed. Last error: {last_error}")