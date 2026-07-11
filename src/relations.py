"""Layer 1 step 4 — Relation Extraction via GLiREL (CLAUDE.md).

Local zero-shot relation extraction over the entity pairs produced by
Layer 1 step 3 (``src/ner.py``). No API keys, no LLM calls.

- Model: ``jackboyla/glirel-large-v0`` (override with env ``GLIREL_MODEL``).
- Input: chunks + step-3 entity records (with per-mention char offsets).
- Output triplets: ``{subject_entity_id, relation, object_entity_id,
  source_chunk_id, score}``.
- Verification rule (enforced here, per CLAUDE.md): both endpoints of every
  triplet must exist in the step-3 entities for that chunk. Anything else is
  dropped — with the reason recorded on the StepTrace at drop time.
- Trace: emits one StepTrace named ``relation_extraction`` when a Trace is
  passed, including on error.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from .core.logging import get_logger

logger = get_logger(__name__)

GLIREL_MODEL = os.getenv("GLIREL_MODEL", "jackboyla/glirel-large-v0")

# Default zero-shot relation label set (configurable per call). Natural-language
# labels — that's what GLiREL is trained to match against.
DEFAULT_RELATION_LABELS = [
    "founded by", "headquartered in", "located in", "part of",
    "competes with", "develops", "offers", "works for", "uses",
]

DEFAULT_THRESHOLD = 0.5  # GLiREL confidence floor (heuristic)

_glirel_model = None


def _get_glirel():
    global _glirel_model
    if _glirel_model is None:
        # Same Windows fix as src/ner.py: force HF cache copy mode or the first
        # download dies with WinError 1314 (symlinks need Developer Mode).
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        try:
            from huggingface_hub import constants as _hfc
            _hfc.HF_HUB_DISABLE_SYMLINKS = True
        except Exception:
            pass
        from glirel import GLiREL  # lazy (pulls torch)
        # glirel 1.2.1 (latest) requires keyword-only `proxies`/`resume_download`
        # in _from_pretrained, but huggingface_hub >= 1.x stopped forwarding
        # them -> TypeError. Shim defaults in until glirel ships a fix.
        _orig = GLiREL._from_pretrained.__func__

        def _shim(cls, *args, proxies=None, resume_download=False, **kw):
            return _orig(cls, *args, proxies=proxies,
                         resume_download=resume_download, **kw)

        GLiREL._from_pretrained = classmethod(_shim)
        logger.info(f"Loading GLiREL {GLIREL_MODEL} (first load downloads the model)...")
        _glirel_model = GLiREL.from_pretrained(GLIREL_MODEL)
    return _glirel_model


# ─────────────────────────────────────────────────────────────────────────────
# Tokenization with char-offset tracking (whitespace words; GLiREL wants words)
# ─────────────────────────────────────────────────────────────────────────────
def _tokenize_with_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens, offsets = [], []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        offsets.append((m.start(), m.end()))
    return tokens, offsets


def _char_span_to_token_span(
    char_start: int, char_end: int, offsets: list[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    """Map a char span to an inclusive (start_tok, end_tok); None if unmappable."""
    start_tok = end_tok = None
    for i, (s, e) in enumerate(offsets):
        if start_tok is None and e > char_start:
            start_tok = i
        if s < char_end:
            end_tok = i
    if start_tok is None or end_tok is None or end_tok < start_tok:
        return None
    return start_tok, end_tok


# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_relations(
    chunks: list[dict],
    entities: list[dict],
    labels: Optional[list[str]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int = 1,
    trace=None,
) -> list[dict]:
    """Run GLiREL over step-3 entity pairs per chunk -> validated triplets.

    chunks:   [{"content": str, "chunk_id": str}, ...]
    entities: step-3 records (``src.ner.extract_entities`` output) — must carry
              ``mentions`` with char offsets.
    Returns:  [{"subject_entity_id", "relation", "object_entity_id",
                "source_chunk_id", "score"}] deduplicated, best score kept.
    """
    labels = labels or DEFAULT_RELATION_LABELS
    model = _get_glirel()

    step_cm = trace.start_step(
        "relation_extraction", n_chunks=len(chunks), n_entities=len(entities),
        labels=labels, threshold=threshold, model=GLIREL_MODEL,
    ) if trace is not None else None
    st = step_cm.__enter__() if step_cm is not None else None

    try:
        # Group step-3 mentions per chunk
        mentions_by_chunk: dict[str, list[tuple[dict, dict]]] = {}
        for e in entities:
            for m in e.get("mentions", []):
                mentions_by_chunk.setdefault(str(m["chunk_id"]), []).append((e, m))

        triplets: dict[tuple, dict] = {}
        raw_count = 0

        for c in chunks:
            chunk_id = str(c.get("chunk_id", ""))
            pairs = mentions_by_chunk.get(chunk_id, [])
            if len(pairs) < 2:
                continue  # relations need at least two entities in the chunk

            tokens, offsets = _tokenize_with_offsets(c["content"])
            ner_spans = []
            span_to_entity: dict[int, str] = {}  # start-token -> entity_id
            valid_ids: set[str] = set()
            for e, m in pairs:
                span = _char_span_to_token_span(m["char_start"], m["char_end"], offsets)
                if span is None:
                    if st is not None:
                        st.drop({"entity_id": e["id"], "chunk_id": chunk_id},
                                "mention char span not mappable to tokens")
                    continue
                t0, t1 = span
                ner_spans.append([t0, t1, e["type"], e["name"]])
                span_to_entity[t0] = e["id"]
                valid_ids.add(e["id"])

            if len(ner_spans) < 2:
                continue

            preds = model.predict_relations(
                tokens, labels, threshold=threshold, ner=ner_spans, top_k=top_k,
            )
            raw_count += len(preds)

            for p in preds:
                subj = span_to_entity.get(int(p["head_pos"][0]))
                obj = span_to_entity.get(int(p["tail_pos"][0]))
                score = round(float(p.get("score", 0.0)), 4)
                # CLAUDE.md verification: endpoints must be step-3 entities for
                # THIS chunk — otherwise drop, with reason logged at drop time.
                if subj is None or obj is None or subj not in valid_ids or obj not in valid_ids:
                    if st is not None:
                        st.drop(
                            {"head": p.get("head_text"), "tail": p.get("tail_text"),
                             "label": p.get("label"), "chunk_id": chunk_id},
                            "endpoint not in step-3 entities for this chunk",
                        )
                    continue
                if subj == obj:
                    if st is not None:
                        st.drop({"entity_id": subj, "label": p.get("label"),
                                 "chunk_id": chunk_id}, "self-relation")
                    continue
                key = (subj, str(p["label"]), obj, chunk_id)
                if key not in triplets or triplets[key]["score"] < score:
                    triplets[key] = {
                        "subject_entity_id": subj,
                        "relation": str(p["label"]),
                        "object_entity_id": obj,
                        "source_chunk_id": chunk_id,
                        "score": score,  # heuristic confidence
                    }

        out = sorted(triplets.values(), key=lambda t: -t["score"])
        logger.info(
            f"GLiREL: {raw_count} raw predictions -> {len(out)} validated triplets "
            f"(threshold={threshold}, heuristic scores)"
        )
        if st is not None:
            st.outputs["triplets"] = out
            st.outputs["n_raw_predictions"] = raw_count
            st.scores["n_triplets"] = float(len(out))  # heuristic
        return out
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
