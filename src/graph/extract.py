"""Entity and relationship extraction from chunks using an LLM.

   Provider: LOCAL Ollama by default (via ``providers.llm_complete`` — zero keys,
   routes on LLM_PROVIDER). NVIDIA NIM / DeepSeek remain optional fallbacks that
   only engage when a key is configured and the local path fails.
   Caches by chunk content hash — re-runs don't re-extract.
   Entity/relationship label sets are domain-agnostic (no brand coupling).
"""
import json
import hashlib
import asyncio
from dataclasses import dataclass, field
from typing import Optional
import httpx

from ..core.config import (
    NVIDIA_API_KEY, NVIDIA_API_KEY_2, DEEPSEEK_API_KEY,
)
from .. import providers
from ..core.logging import get_logger

logger = get_logger(__name__)

NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"

# Domain-agnostic label sets (no brand/vertical coupling). Override per project
# by editing these lists; the extractor passes them into the prompt.
ENTITY_TYPES = [
    "Person", "Organization", "Product", "Service", "Feature", "Technology",
    "Topic", "Concept", "Location", "Attribute", "Date", "Price",
]

RELATIONSHIP_TYPES = [
    "relatedTo", "partOf", "hasFeature", "offeredBy", "locatedIn",
    "contains", "supports", "comparesTo", "uses", "targets",
]

EXTRACTION_PROMPT = """You are a domain-agnostic entity and relationship extraction system for a website content audit.

Extract entities and relationships from the content below. Follow the schema exactly.

ENTITY SCHEMA:
- id: string (lowercase slug of entity name, e.g. "acme-corp", "api-gateway")
- type: one of: {entity_types}
- name: string (human-readable name)
- synonyms: list of alternate names/variations
- source_chunk_ids: list of chunk IDs this entity was extracted from

RELATIONSHIP SCHEMA:
- source_id: string (the id of the source entity)
- target_id: string (the id of the target entity)
- type: one of: {relationship_types}
- evidence_chunk_id: string (the chunk ID containing the evidence)
- confidence: float 0-1

RULES:
- Only extract entities genuinely mentioned in the content.
- Only extract relationships clearly supported by the text.
- id must be a unique, stable lowercase slug.
- confidence >= 0.5; use 1.0 only for very clear relationships.
- Use only the provided entity and relationship types.
- source_chunk_ids should reference the provided chunk_id.
- If the content has no clear entities, return empty lists (do not invent).

Return ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "entities": [
    {{
      "id": "entity-slug",
      "type": "EntityType",
      "name": "Entity Name",
      "synonyms": ["alt name"],
      "source_chunk_ids": ["chunk_id_placeholder"]
    }}
  ],
  "relationships": [
    {{
      "source_id": "entity-slug",
      "target_id": "other-entity-slug",
      "type": "relationshipType",
      "evidence_chunk_id": "chunk_id_placeholder",
      "confidence": 0.8
    }}
  ]
}}

Content (chunk_id: {chunk_id}):
---
{content}
---

JSON OUTPUT:"""


def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:32]


async def _nvidia_extract(
    messages: list[dict],
    model: str = "nvidia/llama-3.3-nemotron-super-49b-v1",
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> str:
    """Extract entities via NVIDIA NIM — tries both keys."""
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
        async with httpx.AsyncClient(timeout=90.0) as client:
            try:
                resp = await client.post(NVIDIA_CHAT_URL, headers=headers, json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (401, 403, 404):
                    last_exc = e
                    continue
                elif e.response.status_code == 429:
                    last_exc = e
                    continue
                else:
                    raise
            except Exception as e:
                last_exc = e
                continue

    raise RuntimeError(f"NVIDIA extraction failed on all keys: {last_exc}")


async def _deepseek_extract(
    messages: list[dict],
    model: str = "deepseek-chat",
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("DeepSeek API key not set for fallback")
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
        return resp.json()["choices"][0]["message"]["content"]


async def _local_extract(
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.1,
) -> str:
    """Extract via the LOCAL LLM (Ollama) through the provider abstraction.

    Routes on LLM_PROVIDER (default ``ollama``) and requires no API key, so the
    knowledge graph builds fully offline. ``providers.llm_complete`` is sync, so
    we run it in the default executor to avoid blocking the event loop.
    """
    system = next((m["content"] for m in messages if m.get("role") == "system"), None)
    user = next((m["content"] for m in messages if m.get("role") == "user"), "")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: providers.llm_complete(
            user, system=system, json=True,
            temperature=temperature, max_tokens=max_tokens,
        ),
    )


def _parse_extraction(raw: str) -> dict:
    """Parse LLM JSON output, stripping markdown code blocks."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    # Strip leading/trailing braces if incomplete
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the JSON portion
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse extraction JSON: {text[:200]}")
        return {"entities": [], "relationships": []}


async def extract_from_chunks(
    chunks: list,
    use_cache: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Extract entities and relationships from a list of chunks.
    Uses content-hash caching — re-runs skip already-processed chunks.
    
    Returns: (entities, relationships)
    """
    all_entities: list[dict] = []
    all_relationships: list[dict] = []
    
    # Process one chunk at a time (larger batches exceed token limits for nutrition tables)
    batch_size = 1
    cache_dir = "data/.extract_cache"
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Build prompt for this batch
        chunk_contexts = "\n\n".join(
            f"[chunk {j} of {i+batch_size}]\n{c['content'][:800]}"
            for j, c in enumerate(batch, start=i+1)
        )
        
        # Check cache per chunk
        batch_entities: list[dict] = []
        batch_rels: list[dict] = []
        skip_batch = False
        
        for c in batch:
            h = _chunk_hash(c['content'])
            cache_path = f"{cache_dir}/{h}.json"
            import os
            os.makedirs(cache_dir, exist_ok=True)
            
            if use_cache and os.path.exists(cache_path):
                with open(cache_path) as f:
                    cached = json.load(f)
                    batch_entities.extend(cached.get("entities", []))
                    batch_rels.extend(cached.get("relationships", []))
            else:
                skip_batch = True  # Need to call LLM for this chunk
        
        if not skip_batch:
            all_entities.extend(batch_entities)
            all_relationships.extend(batch_rels)
            logger.info(f"Extraction cache hit for batch {i//batch_size + 1}")
            continue
        
        # Build messages
        prompt_content = EXTRACTION_PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            relationship_types=", ".join(RELATIONSHIP_TYPES),
            chunk_id=f"chunks {i+1}-{i+len(batch)}",
            content=chunk_contexts,
        )
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise entity extraction system. Output ONLY valid JSON "
                    "matching the exact schema provided. No markdown, no explanation."
                )
            },
            {"role": "user", "content": prompt_content},
        ]
        
        # LOCAL-FIRST: Ollama via providers.llm_complete (zero keys, default).
        # NVIDIA/DeepSeek engage only if a key is configured and local fails.
        raw = None
        try:
            raw = await _local_extract(messages)
        except Exception as e:
            logger.warning(f"Local extraction failed: {e}; trying cloud fallback if keys present")
            for provider in ["nvidia", "deepseek"]:
                try:
                    if provider == "nvidia" and NVIDIA_API_KEY:
                        raw = await _nvidia_extract(messages)
                        break
                    elif provider == "deepseek" and DEEPSEEK_API_KEY:
                        raw = await _deepseek_extract(messages)
                        break
                except Exception as e2:
                    logger.warning(f"Extraction provider {provider} failed: {e2}")
                    continue

        if raw is None:
            logger.error(f"All extraction providers failed for batch {i//batch_size + 1}")
            continue
        
        result = _parse_extraction(raw)
        
        # Fix chunk_ids in extracted entities to point to actual chunk IDs
        # Use integer index as chunk id for tracking
        chunk_ids_in_batch = [str(i + batch.index(c)) for c in batch]
        
        for ent in result.get("entities", []):
            ent["source_chunk_ids"] = chunk_ids_in_batch
        
        for rel in result.get("relationships", []):
            rel["evidence_chunk_id"] = chunk_ids_in_batch[0]
        
        # Cache per chunk
        for c in batch:
            h = _chunk_hash(c['content'])
            os.makedirs(cache_dir, exist_ok=True)
            # Save with all entities from this batch tagged to this chunk
            chunk_idx = i + batch.index(c)
            chunk_cached = {
                "entities": [
                    e for e in result.get("entities", [])
                    if str(chunk_idx) in e.get("source_chunk_ids", [])
                ],
                "relationships": result.get("relationships", []),
            }
            with open(f"{cache_dir}/{h}.json", "w") as f:
                json.dump(chunk_cached, f)
        
        all_entities.extend(result.get("entities", []))
        all_relationships.extend(result.get("relationships", []))
        logger.info(
            f"Extracted batch {i//batch_size + 1}: "
            f"{len(result.get('entities', []))} entities, "
            f"{len(result.get('relationships', []))} relationships"
        )
    
    # Deduplicate entities by id
    seen = set()
    unique_entities = []
    for e in all_entities:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique_entities.append(e)

    # ── Merge entities with same normalized name and compatible types ──
    # e.g. "Noodles" (id=noodles, type=Category) and "Noodles" (id=noodles-collection, type=Category)
    # should merge into one entity
    merged = {}
    for e in unique_entities:
        name_key = e["name"].strip().lower()
        if name_key in merged:
            existing = merged[name_key]
            # Merge source_chunk_ids
            existing_chunks = set(existing.get("source_chunk_ids", []))
            new_chunks = set(e.get("source_chunk_ids", []))
            existing["source_chunk_ids"] = list(existing_chunks | new_chunks)
            # Keep the shorter/more canonical id (prefer shorter slug)
            if len(e["id"]) < len(existing["id"]):
                existing["id"] = e["id"]
            # Keep the more specific type if compatible
            if e["type"] and existing["type"] and e["type"] != existing["type"]:
                # If one is Category and other is Product, keep Product
                type_priority = {"Product": 5, "Brand": 4, "Category": 1, "Ingredient": 3,
                                 "Nutrient": 3, "Attribute": 2, "Organization": 3}
                if type_priority.get(e["type"], 0) > type_priority.get(existing["type"], 0):
                    existing["type"] = e["type"]
        else:
            merged[name_key] = dict(e)

    merged_entities = list(merged.values())
    dedup_count = len(unique_entities) - len(merged_entities)
    if dedup_count:
        logger.info(f"Entity merge: merged {dedup_count} duplicate entities by name")

    logger.info(
        f"Extraction complete: {len(merged_entities)} unique entities, "
        f"{len(all_relationships)} relationships"
    )
    return merged_entities, all_relationships