"""Layer 1 step 7 — Topic Clustering via HDBSCAN + local-LLM labels (CLAUDE.md).

Embed every chunk (``providers.embed_texts`` -> BGE-M3, 1024-dim), cluster
site-wide with HDBSCAN, label each cluster with the local LLM from its
representative chunks. No BERTopic. Fully local, zero keys.

- Output: ``topic{id, label, chunk_ids[]}``; noise chunks reported separately.
- Verification rules (enforced here, per CLAUDE.md):
  * noise ratio must be <= 15% — otherwise re-run with adjusted
    ``min_cluster_size`` (up to MAX_CLUSTER_ATTEMPTS, best attempt kept);
  * generic labels ("General"/"Misc"/...) are rejected and re-asked once with
    a stricter prompt; a still-generic label is kept but flagged in the Trace.
- Trace: emits one StepTrace named ``topic_clustering`` when a Trace is
  passed, including on error.
"""
from __future__ import annotations

import json as _json
from typing import Optional

from . import providers
from .core.logging import get_logger

logger = get_logger(__name__)

MAX_NOISE_RATIO = 0.15          # CLAUDE.md: <=15% noise
MAX_CLUSTER_ATTEMPTS = 3
REPRESENTATIVE_CHUNKS = 3       # per cluster, for labeling
EXCERPT_CHARS = 300
# Truncate chunk text before embedding: clustering needs the gist, not the
# whole chunk, and untruncated long chunks OOM the CPU (attention memory grows
# quadratically with sequence length — an 8 GB alloc failure on real input,
# see BUILD_LOG step-7).
EMBED_CHARS = 2000

GENERIC_LABELS = {
    "general", "misc", "miscellaneous", "other", "others", "content",
    "information", "topics", "various", "overview", "unknown",
}

LABEL_PROMPT = """These text excerpts all belong to one topic cluster from a website.

Excerpts:
{excerpts}

Name the SPECIFIC topic they share, in 2-5 words. Be concrete (e.g.
"Python error handling", not "Programming" or "General").

Return ONLY valid JSON: {{"label": "<2-5 word specific topic label>"}}"""

LABEL_RETRY_SUFFIX = """

IMPORTANT: your previous answer was too generic. Labels like "General",
"Misc", "Content", "Information", "Overview" are FORBIDDEN. Name the concrete
subject matter of the excerpts."""


def _cluster(embeddings: list[list[float]], min_cluster_size: int) -> list[int]:
    """HDBSCAN labels (-1 = noise). Isolated for testability."""
    import hdbscan  # lazy
    import numpy as np
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, metric="euclidean",
    )
    return [int(x) for x in clusterer.fit_predict(np.asarray(embeddings))]


def _label_cluster(excerpts: list[str], retry: bool = False) -> str:
    prompt = LABEL_PROMPT.format(
        excerpts="\n---\n".join(e[:EXCERPT_CHARS] for e in excerpts))
    if retry:
        prompt += LABEL_RETRY_SUFFIX
    raw = providers.llm_complete(
        prompt, json=True, temperature=0,
        system="You label topic clusters. Output ONLY valid JSON.",
    )
    try:
        data = _json.loads(raw.strip())
        return str(data.get("label", "")).strip()
    except _json.JSONDecodeError:
        logger.warning(f"Topic label JSON unparseable: {raw[:120]}")
        return ""


def _representatives(indices: list[int], embeddings, k: int = REPRESENTATIVE_CHUNKS) -> list[int]:
    """Indices of the k members closest to the cluster centroid."""
    import numpy as np
    vecs = np.asarray([embeddings[i] for i in indices])
    centroid = vecs.mean(axis=0)
    dists = ((vecs - centroid) ** 2).sum(axis=1)
    order = dists.argsort()[:k]
    return [indices[int(i)] for i in order]


def cluster_topics(
    chunks: list[dict],
    min_cluster_size: Optional[int] = None,
    trace=None,
) -> list[dict]:
    """Cluster chunks into topics and label them.

    chunks: [{"content": str, "chunk_id": str}, ...]
    Returns: [{"id": "topic_N", "label": str, "chunk_ids": [...]}]
    (noise chunk ids + ratios live on the StepTrace outputs).
    """
    mcs = min_cluster_size or max(2, len(chunks) // 20)

    step_cm = trace.start_step(
        "topic_clustering", n_chunks=len(chunks), min_cluster_size=mcs,
        embed_model=providers.EMBED_MODEL,
        llm=f"{providers.LLM_PROVIDER}:{providers.LLM_MODEL}",
    ) if trace is not None else None
    st = step_cm.__enter__() if step_cm is not None else None

    try:
        if len(chunks) < 4:
            if st is not None:
                st.note(f"only {len(chunks)} chunks — too few to cluster; no topics emitted")
            logger.warning(f"Topic clustering skipped: {len(chunks)} chunks (<4)")
            if st is not None:
                st.outputs["topics"] = []
                st.scores["n_topics"] = 0.0
            return []

        embeddings = providers.embed_texts([c["content"][:EMBED_CHARS] for c in chunks])

        # Cluster; re-run with smaller min_cluster_size while noise > 15%.
        best_labels: list[int] = []
        best_noise = 1.0
        attempt_mcs = mcs
        for attempt in range(1, MAX_CLUSTER_ATTEMPTS + 1):
            labels = _cluster(embeddings, attempt_mcs)
            noise = sum(1 for l in labels if l == -1) / len(labels)
            if st is not None:
                st.note(f"attempt {attempt}: min_cluster_size={attempt_mcs} "
                        f"noise={noise:.0%} (heuristic)")
            if noise < best_noise or not best_labels:
                best_labels, best_noise = labels, noise
            if noise <= MAX_NOISE_RATIO:
                break
            if attempt_mcs <= 2:
                break
            # spec: adjust and re-run — halve, don't nibble (mcs-1 provably
            # went nowhere on 675 real chunks: 33->32->31 stayed at 100% noise)
            attempt_mcs = max(2, attempt_mcs // 2)
        labels = best_labels

        # Group members per cluster
        members: dict[int, list[int]] = {}
        for i, l in enumerate(labels):
            if l >= 0:
                members.setdefault(l, []).append(i)
        noise_ids = [chunks[i]["chunk_id"] for i, l in enumerate(labels) if l == -1]

        # Label clusters with the local LLM; reject generic labels once.
        topics: list[dict] = []
        generic_kept: list[str] = []
        for cluster_id in sorted(members):
            idxs = members[cluster_id]
            reps = _representatives(idxs, embeddings)
            excerpts = [chunks[i]["content"] for i in reps]
            label = _label_cluster(excerpts)
            if not label or label.lower() in GENERIC_LABELS:
                if st is not None:
                    st.note(f"cluster {cluster_id}: generic/empty label {label!r} rejected -> retry")
                label = _label_cluster(excerpts, retry=True)
            if not label or label.lower() in GENERIC_LABELS:
                generic_kept.append(f"topic_{cluster_id}:{label!r}")
            topics.append({
                "id": f"topic_{cluster_id}",
                "label": label or f"unlabeled cluster {cluster_id}",
                "chunk_ids": [chunks[i]["chunk_id"] for i in idxs],
            })

        if generic_kept and st is not None:
            st.note(f"STILL-GENERIC labels kept for review: {generic_kept}")
        logger.info(
            f"Topics: {len(topics)} clusters from {len(chunks)} chunks, "
            f"noise={best_noise:.0%} (limit {MAX_NOISE_RATIO:.0%}), "
            f"labels={[t['label'] for t in topics]}"
        )

        if st is not None:
            st.outputs["topics"] = topics
            st.outputs["noise_chunk_ids"] = noise_ids
            st.scores["n_topics"] = float(len(topics))       # heuristic
            st.scores["noise_ratio"] = round(best_noise, 4)  # heuristic
        return topics
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
