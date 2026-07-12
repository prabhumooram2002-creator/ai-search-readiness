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

DEFAULT_THRESHOLD = 0.5  # GLiREL raw prediction threshold (heuristic)
# Post-filter precision floor applied to DEDUPED edges (backlog: GLiREL
# over-generates — 84 raw predictions from 3 sentences). Env-tunable.
# NOTE (measured): the floor alone was insufficient — it cut only 23% of edges
# and left high-scoring nonsense (e.g. "San Francisco founded-by Dario Amodei"
# 0.87). Type constraints below are what actually recover precision.
import os
PRECISION_FLOOR = float(os.getenv("GLIREL_PRECISION_FLOOR", "0.6"))

# Per-relation allowed (head_types, tail_types). None = any type allowed.
# Types are the step-3 GLiNER entity types (case-insensitive). Relations not
# listed are unconstrained. SYMMETRIC relations also canonicalize direction.
RELATION_TYPE_CONSTRAINTS: dict[str, tuple[Optional[set], Optional[set]]] = {
    "founded by":       ({"organization"}, {"person", "organization"}),
    "headquartered in": ({"organization", "product"}, {"location"}),
    "located in":       ({"organization", "person"}, {"location"}),  # not products
    "works for":        ({"person"}, {"organization"}),
    "develops":         ({"organization"}, {"product", "service", "feature"}),
    "offers":           ({"organization"}, {"product", "service", "feature"}),
    "competes with":    ({"organization", "product"}, {"organization", "product"}),
    "part of":          (None, None),
    "uses":             (None, None),
}
SYMMETRIC_RELATIONS = {"competes with", "related to"}


def _type_ok(relation: str, head_type: str, tail_type: str) -> bool:
    """Does this triplet satisfy the relation's type constraints?"""
    ht, tt = RELATION_TYPE_CONSTRAINTS.get(relation, (None, None))
    h, t = (head_type or "").lower(), (tail_type or "").lower()
    return (ht is None or h in ht) and (tt is None or t in tt)

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
        # Group step-3 mentions per chunk + id -> type for constraint checks
        mentions_by_chunk: dict[str, list[tuple[dict, dict]]] = {}
        id_to_type: dict[str, str] = {}
        for e in entities:
            id_to_type[e["id"]] = e.get("type", "")
            for m in e.get("mentions", []):
                mentions_by_chunk.setdefault(str(m["chunk_id"]), []).append((e, m))

        triplets: dict[tuple, dict] = {}
        raw_count = 0
        n_type_dropped = 0

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
                relation = str(p["label"])
                # Type constraint: the relation's head/tail entity types must
                # match (kills "San Francisco founded-by Dario", wrong-direction
                # "Dario founded-by Anthropic", "Claude located-in ...", etc.)
                if not _type_ok(relation, id_to_type.get(subj, ""),
                                id_to_type.get(obj, "")):
                    n_type_dropped += 1
                    if st is not None:
                        st.drop({"triplet": (subj, relation, obj),
                                 "head_type": id_to_type.get(subj),
                                 "tail_type": id_to_type.get(obj)},
                                "type constraint violated")
                    continue
                # Symmetric relations: canonicalize direction so A<->B is one edge.
                if relation in SYMMETRIC_RELATIONS and subj > obj:
                    subj, obj = obj, subj
                # Dedup identical (subject, relation, object) ACROSS chunks into
                # ONE edge with a support_count; keep best score, gather chunks.
                key = (subj, relation, obj)
                t = triplets.get(key)
                if t is None:
                    triplets[key] = {
                        "subject_entity_id": subj,
                        "relation": relation,
                        "object_entity_id": obj,
                        "source_chunk_id": chunk_id,
                        "source_chunk_ids": [chunk_id],
                        "support_count": 1,
                        "score": score,  # heuristic confidence
                    }
                else:
                    t["support_count"] += 1
                    if chunk_id not in t["source_chunk_ids"]:
                        t["source_chunk_ids"].append(chunk_id)
                    t["score"] = max(t["score"], score)

        deduped = list(triplets.values())
        # Precision floor: drop low-confidence edges (over-generation). An edge
        # seen in >=2 chunks (corroborated) is kept one notch lower.
        kept = []
        for t in deduped:
            floor = PRECISION_FLOOR - (0.1 if t["support_count"] >= 2 else 0.0)
            if t["score"] >= floor:
                kept.append(t)
            elif st is not None:
                st.drop({"triplet": (t["subject_entity_id"], t["relation"],
                                     t["object_entity_id"]), "score": t["score"]},
                        f"below precision floor {floor:.2f}")

        out = sorted(kept, key=lambda t: -t["score"])
        logger.info(
            f"GLiREL: {raw_count} raw -> {n_type_dropped} type-violations dropped "
            f"-> {len(deduped)} deduped -> {len(out)} kept "
            f"(precision_floor={PRECISION_FLOOR}, heuristic scores)"
        )
        if st is not None:
            st.outputs["triplets"] = out
            st.outputs["n_raw_predictions"] = raw_count
            st.outputs["n_type_dropped"] = n_type_dropped
            st.outputs["n_deduped"] = len(deduped)
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
