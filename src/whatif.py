"""PHASE 7 — Pre-publish what-if scoring (v3 brief).

Run a DRAFT through chunking -> structural scorer -> retrieval simulation
against a TEMPORARY in-memory overlay (draft chunks added to the candidate set,
NEVER persisted to the real Chroma/Kuzu stores). Reports predicted
cluster-coverage delta per target query, structure flags, and which fan-out
sub-intents the draft newly covers.

Coverage is predicted with the same signals as the Layer-2 retriever
(rank_bm25 0.4 + BGE cosine 0.6, fan-out weights) WITHOUT loading the
reranker/NLI/synthesis models — coverage doesn't need them, and this keeps
what-if fast and side-effect-free.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

from . import providers
from .core.logging import get_logger

logger = get_logger(__name__)

BM25_WEIGHT, EMBED_WEIGHT = 0.4, 0.6
DEAD_END_FLOOR = 0.25


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def load_draft(path: str | Path) -> str:
    """Read a draft: .md/.txt directly, .docx via python-docx if installed."""
    p = Path(path)
    if p.suffix.lower() == ".docx":
        try:
            import docx  # optional
            return "\n\n".join(par.text for par in docx.Document(str(p)).paragraphs)
        except ImportError:
            raise RuntimeError("python-docx not installed — convert the draft to "
                               ".md/.txt or `pip install python-docx`")
    return p.read_text(encoding="utf-8-sig", errors="replace")


def _hash_paths(paths: list[str]) -> dict[str, Optional[str]]:
    """Hash persistent stores so we can prove the overlay never wrote to them."""
    out = {}
    for pth in paths:
        p = Path(pth)
        if not p.exists():
            out[pth] = None
        elif p.is_file():
            out[pth] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        else:
            h = hashlib.sha256()
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    h.update(f.name.encode())
                    h.update(f.read_bytes())
            out[pth] = h.hexdigest()[:16]
    return out


def _weighted_coverage(query: str, chunks: list[dict], embeddings: list[list[float]],
                       fanout: list[dict]) -> dict:
    """Predicted weighted cluster coverage for one query over a candidate set."""
    from rank_bm25 import BM25Okapi
    ids = [str(c["chunk_id"]) for c in chunks]
    bm25 = BM25Okapi([_tok(c["content"]) for c in chunks])
    emb_by_id = {ids[i]: embeddings[i] for i in range(min(len(ids), len(embeddings)))}
    subs = [query] + [d["sub_query"] for d in fanout]
    weights = {d["sub_query"]: d["weight"] for d in fanout}
    qvecs = providers.embed_texts([q[:512] for q in subs])

    covered_w = total_w = 0.0
    covered = 0
    for qi, q in enumerate(subs):
        if q == query:
            continue
        bscores = bm25.get_scores(_tok(q))
        bmax = max(bscores) if len(bscores) and max(bscores) > 0 else 1.0
        qv = qvecs[qi]
        best = 0.0
        for ci, cid in enumerate(ids):
            ev = emb_by_id.get(cid)
            cos = max(0.0, sum(a * b for a, b in zip(qv, ev))) if ev else 0.0
            combined = BM25_WEIGHT * (bscores[ci] / bmax) + EMBED_WEIGHT * cos
            best = max(best, combined)
        w = weights.get(q, 0.0)
        total_w += w
        if best >= DEAD_END_FLOOR:
            covered_w += w
            covered += 1
    return {"weighted_coverage": round(covered_w / total_w, 4) if total_w else 0.0,
            "sub_intents_covered": covered, "sub_intents_total": len(fanout)}


def whatif(draft_path: str, target_queries: list[str],
           base_chunks: list[dict], base_embeddings: list[list[float]],
           persist_paths: Optional[list[str]] = None,
           trace=None) -> dict:
    """Score a draft against target queries via a non-persisting overlay.

    base_chunks: [{"chunk_id","content","url"?}] currently indexed.
    Returns per-query coverage before/after + delta, draft structure flags, and
    a store-integrity check (hashes identical before/after).
    """
    from .chunking2 import semantic_chunks
    from .structure import score_chunks
    from .fanout import fanout_distribution

    step_cm = trace.start_step("whatif", draft=draft_path,
                               n_targets=len(target_queries)) if trace else None
    st = step_cm.__enter__() if step_cm else None
    try:
        persist_paths = persist_paths or []
        before = _hash_paths(persist_paths)

        # draft -> chunks (in memory) -> structural flags
        text = load_draft(draft_path)
        draft_chunks_raw = semantic_chunks(text, url=f"draft://{Path(draft_path).name}")
        draft_chunks = [{"chunk_id": c["id"], "content": c["text"],
                         "url": c["page_url"], "heading_path": c["heading_path"]}
                        for c in draft_chunks_raw]
        struct = score_chunks(draft_chunks_raw)
        draft_embs = providers.embed_texts([c["content"][:512] for c in draft_chunks])

        overlay_chunks = base_chunks + draft_chunks
        overlay_embs = list(base_embeddings) + list(draft_embs)

        per_query = []
        for q in target_queries:
            fanout = fanout_distribution(q, use_cache=True)
            cov_before = _weighted_coverage(q, base_chunks, base_embeddings, fanout)
            cov_after = _weighted_coverage(q, overlay_chunks, overlay_embs, fanout)
            delta = round((cov_after["weighted_coverage"] or 0) -
                          (cov_before["weighted_coverage"] or 0), 4)
            per_query.append({
                "query": q,
                "coverage_before": cov_before["weighted_coverage"],
                "coverage_after": cov_after["weighted_coverage"],
                "coverage_delta": delta,
                "sub_intents_covered_before": cov_before["sub_intents_covered"],
                "sub_intents_covered_after": cov_after["sub_intents_covered"],
                "newly_covered": cov_after["sub_intents_covered"] -
                                 cov_before["sub_intents_covered"]})

        after = _hash_paths(persist_paths)
        stores_unchanged = before == after

        result = {
            "draft": draft_path, "n_draft_chunks": len(draft_chunks),
            "structure_flags": [f for r in struct for f in r["flags"]],
            "mean_structure_score": round(
                sum(r["structure_score"] for r in struct) / len(struct), 4)
                if struct else None,
            "per_query": per_query,
            "stores_unchanged": stores_unchanged,     # overlay never persisted
            "store_hashes_before": before, "store_hashes_after": after,
        }
        logger.info(f"[whatif] {len(draft_chunks)} draft chunks; stores_unchanged="
                    f"{stores_unchanged}; deltas="
                    f"{[(p['query'][:30], p['coverage_delta']) for p in per_query]}")
        if st is not None:
            st.outputs.update({k: result[k] for k in
                               ("n_draft_chunks", "per_query", "stores_unchanged")})
            st.scores["max_coverage_delta"] = max(
                (p["coverage_delta"] for p in per_query), default=0.0)
        return result
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
