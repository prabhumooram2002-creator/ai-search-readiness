"""Layer 1 step 3 — Entity Extraction via GLiNER (CLAUDE.md).

Local zero-shot NER over chunks with a configurable label set. No API keys,
no per-chunk LLM calls — batched local inference, deterministic.

- Model: ``urchade/gliner_large-v2.1`` (override with env ``GLINER_MODEL``).
- Labels: configurable per call; defaults per CLAUDE.md
  (person, organization, product, feature, price, date, location).
- Cross-check: spaCy ``en_core_web_trf`` — chunks where the two extractors
  disagree by more than ``DISAGREEMENT_THRESHOLD`` (Jaccard distance on
  normalized entity texts, heuristic) are flagged for review.
- Trace: when a ``Trace`` is passed, emits one StepTrace named
  ``entity_extraction`` with outputs + heuristic scores.

Entity record shape is compatible with the graph upsert path
(``{id, type, name, synonyms, source_chunk_ids}``) plus char offsets and the
GLiNER confidence score.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from .core.logging import get_logger

logger = get_logger(__name__)

GLINER_MODEL = os.getenv("GLINER_MODEL", "urchade/gliner_large-v2.1")

# CLAUDE.md Layer 1 step 3 default label set (configurable per call).
DEFAULT_LABELS = [
    "person", "organization", "product", "feature", "price", "date", "location",
]

DEFAULT_THRESHOLD = 0.4          # GLiNER confidence floor
DISAGREEMENT_THRESHOLD = 0.2     # >20% disagreement with spaCy -> flag chunk

# spaCy label -> our label space (only comparable labels are cross-checked)
_SPACY_LABEL_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "PRODUCT": "product",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "DATE": "date",
    "MONEY": "price",
}

# ── Lazy singletons (heavy imports stay off the module import path) ─────────
_gliner_model = None
_spacy_nlp = None


def _get_gliner():
    global _gliner_model
    if _gliner_model is None:
        # Windows without Developer Mode cannot create symlinks; force the HF
        # cache into copy mode or the first download dies with WinError 1314.
        # Patch the already-imported constant too, since huggingface_hub reads
        # the env var only at import time.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        try:
            from huggingface_hub import constants as _hfc
            _hfc.HF_HUB_DISABLE_SYMLINKS = True
        except Exception:
            pass
        from gliner import GLiNER  # lazy (pulls torch)
        logger.info(f"Loading GLiNER {GLINER_MODEL} (first load downloads the model)...")
        _gliner_model = GLiNER.from_pretrained(GLINER_MODEL)
    return _gliner_model


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy  # lazy
        logger.info("Loading spaCy en_core_web_trf for NER cross-check...")
        _spacy_nlp = spacy.load("en_core_web_trf")
    return _spacy_nlp


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "entity"


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_entities(
    chunks: list[dict],
    labels: Optional[list[str]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    trace=None,
) -> list[dict]:
    """Run GLiNER over chunks -> deduplicated entity records.

    chunks: [{"content": str, "chunk_id": str}, ...]
    Returns entities: {id, type, name, synonyms, source_chunk_ids,
                       score (max GLiNER conf, heuristic), mentions[
                       {chunk_id, char_start, char_end, score}]}
    """
    labels = labels or DEFAULT_LABELS
    model = _get_gliner()

    step_cm = trace.start_step(
        "entity_extraction", n_chunks=len(chunks), labels=labels,
        threshold=threshold, model=GLINER_MODEL,
    ) if trace is not None else None
    st = step_cm.__enter__() if step_cm is not None else None

    try:
        merged: dict[tuple[str, str], dict] = {}
        raw_count = 0
        for c in chunks:
            text = c["content"]
            chunk_id = str(c.get("chunk_id", ""))
            preds = model.predict_entities(text, labels, threshold=threshold)
            raw_count += len(preds)
            for p in preds:
                name = p["text"].strip()
                label = str(p["label"]).lower()
                key = (_slug(name), label)
                mention = {
                    "chunk_id": chunk_id,
                    "char_start": int(p.get("start", -1)),
                    "char_end": int(p.get("end", -1)),
                    "score": round(float(p.get("score", 0.0)), 4),
                }
                if key in merged:
                    e = merged[key]
                    if chunk_id not in e["source_chunk_ids"]:
                        e["source_chunk_ids"].append(chunk_id)
                    if name not in e["synonyms"] and name != e["name"]:
                        e["synonyms"].append(name)
                    e["score"] = max(e["score"], mention["score"])  # heuristic
                    e["mentions"].append(mention)
                else:
                    merged[key] = {
                        "id": _slug(name),
                        "type": label.capitalize(),
                        "name": name,
                        "synonyms": [],
                        "source_chunk_ids": [chunk_id],
                        "score": mention["score"],  # heuristic confidence
                        "mentions": [mention],
                    }

        entities = list(merged.values())
        logger.info(
            f"GLiNER: {raw_count} raw mentions -> {len(entities)} unique entities "
            f"across {len(chunks)} chunks (threshold={threshold}, heuristic scores)"
        )
        if st is not None:
            st.outputs["entities"] = entities
            st.outputs["n_raw_mentions"] = raw_count
            st.scores["n_entities"] = float(len(entities))  # heuristic
        return entities
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# spaCy cross-check (CLAUDE.md: flag >20% disagreement for review)
# ─────────────────────────────────────────────────────────────────────────────
def crosscheck_with_spacy(
    chunks: list[dict],
    gliner_entities: list[dict],
    disagreement_threshold: float = DISAGREEMENT_THRESHOLD,
) -> dict:
    """Compare GLiNER output against spaCy en_core_web_trf per chunk.

    Disagreement per chunk = Jaccard distance between the two extractors'
    normalized entity-text sets (comparable labels only). Heuristic metric.
    Returns {"per_chunk": [{chunk_id, gliner, spacy, disagreement, flagged}],
             "mean_disagreement": float, "flagged_chunk_ids": [...]}.
    """
    nlp = _get_spacy()

    by_chunk: dict[str, set[str]] = {}
    for e in gliner_entities:
        for m in e["mentions"]:
            by_chunk.setdefault(m["chunk_id"], set()).add(e["name"].lower())

    per_chunk = []
    for c in chunks:
        chunk_id = str(c.get("chunk_id", ""))
        doc = nlp(c["content"])
        spacy_set = {
            ent.text.strip().lower()
            for ent in doc.ents if ent.label_ in _SPACY_LABEL_MAP
        }
        gliner_set = by_chunk.get(chunk_id, set())
        union = gliner_set | spacy_set
        if not union:
            disagreement = 0.0  # both empty -> agreement
        else:
            disagreement = 1.0 - len(gliner_set & spacy_set) / len(union)
        per_chunk.append({
            "chunk_id": chunk_id,
            "gliner": sorted(gliner_set),
            "spacy": sorted(spacy_set),
            "disagreement": round(disagreement, 3),  # heuristic
            "flagged": disagreement > disagreement_threshold,
        })

    mean = sum(p["disagreement"] for p in per_chunk) / len(per_chunk) if per_chunk else 0.0
    flagged = [p["chunk_id"] for p in per_chunk if p["flagged"]]
    if flagged:
        logger.warning(
            f"NER cross-check: {len(flagged)}/{len(per_chunk)} chunks exceed "
            f"{disagreement_threshold:.0%} GLiNER/spaCy disagreement (heuristic) -> review"
        )
    return {
        "per_chunk": per_chunk,
        "mean_disagreement": round(mean, 3),
        "flagged_chunk_ids": flagged,
    }
