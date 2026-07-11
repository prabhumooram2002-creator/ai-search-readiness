"""Node (B) — local graph-extraction path.

Proves src/graph/extract.py extracts entities via the LOCAL provider seam
(providers.llm_complete, JSON mode) with NO API key set. The LLM is stubbed so
the test is fast, deterministic, and offline — it verifies the wiring, not the
model. (An end-to-end run against real Ollama on example.com is logged
separately in BUILD_LOG.md.)
"""
import sys
import asyncio
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.graph import extract as ex


def test_local_extraction_routes_through_providers_without_key(monkeypatch):
    calls = {}

    def fake_llm_complete(prompt, *, system=None, json=False, temperature=0, max_tokens=None):
        calls["called"] = True
        calls["json"] = json
        calls["has_system"] = bool(system)
        return (
            '{"entities":[{"id":"acme","type":"Organization","name":"Acme",'
            '"synonyms":[],"source_chunk_ids":["0"]}],"relationships":[]}'
        )

    # Route the local path to our stub; guarantee zero keys so no cloud path can run.
    monkeypatch.setattr(ex.providers, "llm_complete", fake_llm_complete)
    monkeypatch.setattr(ex, "NVIDIA_API_KEY", "")
    monkeypatch.setattr(ex, "NVIDIA_API_KEY_2", "")
    monkeypatch.setattr(ex, "DEEPSEEK_API_KEY", "")

    chunks = [{"content": "Acme Corp builds widgets.", "chunk_id": "c0"}]
    entities, relationships = asyncio.run(ex.extract_from_chunks(chunks, use_cache=False))

    assert calls.get("called") is True          # local provider path was taken
    assert calls.get("json") is True            # asked for JSON mode
    assert calls.get("has_system") is True       # system prompt threaded through
    assert any(e["name"] == "Acme" for e in entities)
    assert entities[0]["type"] == "Organization"


def test_entity_label_set_is_domain_agnostic():
    # De-branding: no health/nutrition/brand-specific types leaked into the schema.
    banned = {"Disease", "HealthCondition", "Nutrient", "Ingredient", "Brand", "FAQ"}
    assert banned.isdisjoint(set(ex.ENTITY_TYPES)), ex.ENTITY_TYPES
    assert "Organization" in ex.ENTITY_TYPES and "Product" in ex.ENTITY_TYPES
