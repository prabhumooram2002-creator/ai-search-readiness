"""Phase 3c — Stochastic fan-out (v3 brief).

Real answer engines explode one query into 8-15+ sub-queries and only ~27% are
stable across runs. A fixed 3-5 facet expansion under-models that. This node
runs the expansion prompt N times at temperature 0.8 (the ONE sanctioned
temperature-0 exception — logged), deduplicates near-identical sub-queries by
embedding cosine, and frequency-weights the survivors into a
``fanout_distribution[{sub_query, weight}]``.

The distribution is cached per query (it costs N LLM calls); invalidated only
when the query set changes. Coverage/dead-ends downstream are scored per
sub-query and aggregated by weight.

Emits a StepTrace named ``fanout`` when a Trace is passed.
"""
from __future__ import annotations

import hashlib
import json as _json
from pathlib import Path
from typing import Optional

from . import providers
from .core.config import BASE_DIR
from .core.logging import get_logger

logger = get_logger(__name__)

N_SAMPLES = 12
FANOUT_TEMPERATURE = 0.8       # sanctioned exception to temp-0 (logged)
# Cluster PARAPHRASES into shared sub-intents. 0.92 was far too strict for real
# BGE-M3 embeddings of verbose LLM sub-questions (paraphrases sit ~0.75-0.88),
# so nothing merged -> 140+ singleton intents -> unstable top-5. 0.80 groups
# paraphrases of the same intent while keeping distinct intents apart.
MERGE_COSINE = 0.80
CACHE_DIR = BASE_DIR / "data" / "fanout_cache"

FANOUT_PROMPT = (
    "A user asks an AI assistant: \"{query}\"\n"
    "List the 4-7 MAIN sub-questions the assistant must resolve to answer well. "
    "Keep each short (one facet each); cover the key angles, not edge cases.\n"
    'Return ONLY JSON: {{"sub_queries": ["...", "..."]}}')


def _cache_path(query: str) -> Path:
    h = hashlib.sha256(query.encode()).hexdigest()[:20]
    return CACHE_DIR / f"{h}.json"


def _json_obj(raw: str) -> dict:
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    try:
        return _json.loads(t.strip())
    except _json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}") + 1
        if 0 <= s < e:
            try:
                return _json.loads(t[s:e])
            except _json.JSONDecodeError:
                pass
    return {}


def _dedup_weight(samples: list[str]) -> list[dict]:
    """Cluster near-identical sub-queries by embedding cosine; weight by
    frequency (share of the N samples that landed in each cluster)."""
    import numpy as np
    if not samples:
        return []
    vecs = providers.embed_texts([s[:256] for s in samples])
    clusters: list[dict] = []   # {rep, vec, members}
    for s, v in zip(samples, vecs):
        v = np.asarray(v)
        best, best_sim = None, -1.0
        for c in clusters:
            sim = float(v.dot(c["vec"]) /
                        ((np.linalg.norm(v) * np.linalg.norm(c["vec"])) or 1.0))
            if sim > best_sim:
                best, best_sim = c, sim
        if best is not None and best_sim >= MERGE_COSINE:
            best["members"].append(s)
        else:
            clusters.append({"rep": s, "vec": v, "members": [s]})
    total = len(samples)
    dist = [{"sub_query": c["rep"],
             "weight": round(len(c["members"]) / total, 4),
             "count": len(c["members"])} for c in clusters]
    dist.sort(key=lambda d: -d["weight"])
    return dist


def fanout_distribution(
    query: str,
    n_samples: int = N_SAMPLES,
    use_cache: bool = True,
    trace=None,
) -> list[dict]:
    """Return the weighted sub-query distribution for one query (cached)."""
    step_cm = trace.start_step("fanout", query=query, n_samples=n_samples,
                               temperature=FANOUT_TEMPERATURE) if trace else None
    st = step_cm.__enter__() if step_cm else None
    try:
        cache = _cache_path(query)
        if use_cache and cache.exists():
            dist = _json.loads(cache.read_text(encoding="utf-8"))
            if st is not None:
                st.note("served from cache")
                st.outputs["fanout_distribution"] = dist
                st.scores["n_sub_intents"] = float(len(dist))
            return dist

        logger.info(f"[fanout] sampling expansion x{n_samples} at "
                    f"temperature={FANOUT_TEMPERATURE} (sanctioned temp-0 exception)")
        samples: list[str] = []
        for _ in range(n_samples):
            raw = providers.llm_complete(
                FANOUT_PROMPT.format(query=query), json=True,
                temperature=FANOUT_TEMPERATURE,
                system="You expand search queries. Output ONLY valid JSON.")
            for sq in _json_obj(raw).get("sub_queries", []):
                sq = str(sq).strip()
                if sq:
                    samples.append(sq)

        dist = _dedup_weight(samples)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(_json.dumps(dist, indent=2), encoding="utf-8")
        if st is not None:
            st.outputs["fanout_distribution"] = dist
            st.outputs["n_raw_samples"] = len(samples)
            st.scores["n_sub_intents"] = float(len(dist))
        return dist
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)


def intent_stability(dist_a: list[dict], dist_b: list[dict],
                     k: int = 5, threshold: float = 0.8) -> float:
    """Semantic top-k stability between two fan-out runs.

    Sub-intents are semantic clusters, so two independent runs express the same
    intent with different wording — literal string Jaccard understates
    stability. This matches each of A's top-k intents to B's top-k by embedding
    cosine (>= threshold) and returns matched / k in [0,1].
    """
    import numpy as np
    a = [d["sub_query"] for d in dist_a[:k]]
    b = [d["sub_query"] for d in dist_b[:k]]
    if not a or not b:
        return 0.0
    va = providers.embed_texts(a)
    vb = providers.embed_texts(b)
    used, matched = set(), 0
    for x in va:
        x = np.asarray(x)
        best_j, best = -1, -1.0
        for j, y in enumerate(vb):
            if j in used:
                continue
            y = np.asarray(y)
            sim = float(x.dot(y) / ((np.linalg.norm(x) * np.linalg.norm(y)) or 1.0))
            if sim > best:
                best, best_j = sim, j
        if best >= threshold and best_j >= 0:
            used.add(best_j)
            matched += 1
    return round(matched / k, 4)


def invalidate(query: str) -> None:
    p = _cache_path(query)
    if p.exists():
        p.unlink()
