"""PHASE 10a — The impact chain (v3 brief).

Chains machinery that already exists: for every Layer-3 recommendation,
Phase 8 (``src/fixes.py``) already generated a fix artifact -> run that
artifact through Phase 7's what-if overlay (``src/whatif.py``) -> attach the
**predicted coverage delta** to the recommendation.

Every action then reads: "Do this -> your coverage on '<query>' goes
0.31 -> 0.74 (predicted, simulated)."

Content-type fixes (FAQ drafts for dead ends, block rewrites for structure
flags) go through the real what-if overlay. Technical fixes (JSON-LD,
llms.txt) don't add retrievable prose, so they get a qualitative impact note
instead of a fabricated coverage number — never estimate what the machinery
can't actually measure.

Cached per (finding_id, fix_content_hash) so repeat runs only recompute when
either the finding or its fix artifact actually changed.
"""
from __future__ import annotations

import hashlib
import json as _json
from pathlib import Path
from typing import Optional

from .core.config import BASE_DIR
from .core.logging import get_logger

logger = get_logger(__name__)

IMPACT_CACHE = BASE_DIR / "data" / "impact_cache.json"
PREDICTED_LABEL = "predicted (simulated)"

# Fix types with real prose content -> go through the what-if overlay.
_CONTENT_FIX_TYPES = {"faq_draft", "block_rewrite"}
# Fix types that are structural/technical -> qualitative note, no coverage delta.
_TECHNICAL_FIX_TYPES = {"jsonld", "llms_txt"}

_TECHNICAL_NOTES = {
    "jsonld": "resolves missing/invalid structured data on this page — "
              "no direct content-coverage delta (structural fix)",
    "llms_txt": "publishes a machine-readable site index for AI crawlers — "
               "no direct content-coverage delta (technical fix)",
}


def _fix_content_hash(path: str) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"


def _load_cache(cache_path: Path) -> dict:
    try:
        return _json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, _json.JSONDecodeError):
        return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(_json.dumps(cache, indent=2), encoding="utf-8")


def attach_predicted_impact(
    recommendations: list[dict],
    fix_manifest: list[dict],
    query: str,
    base_chunks: list[dict],
    base_embeddings: list[list[float]],
    persist_paths: Optional[list[str]] = None,
    cache_path: Optional[Path] = None,
    trace=None,
) -> list[dict]:
    """Attach ``predicted_impact`` to every recommendation that has a matching
    Phase 8 fix artifact. Recommendations with no fix artifact (nothing was
    generated for them, e.g. ``--fixes`` wasn't passed) are returned
    unchanged — 10a only chains what already exists, it never invents a fix.
    """
    from .whatif import whatif

    cache_path = cache_path or IMPACT_CACHE
    cache = _load_cache(cache_path)
    by_finding = {m["finding_id"]: m for m in fix_manifest}

    step_cm = trace.start_step("impact_chain", query=query,
                               n_recommendations=len(recommendations)) if trace else None
    st = step_cm.__enter__() if step_cm is not None else None
    n_computed = n_cached = n_skipped = 0

    try:
        for rec in recommendations:
            fid = rec.get("finding_id")
            fix = by_finding.get(fid)
            if fix is None:
                n_skipped += 1
                continue

            fix_hash = _fix_content_hash(fix["path"])
            cache_key = f"{fid}|{fix_hash}"

            if cache_key in cache:
                rec["predicted_impact"] = cache[cache_key]
                n_cached += 1
                continue

            if fix["type"] in _TECHNICAL_FIX_TYPES:
                impact = {"kind": "technical", "label": PREDICTED_LABEL,
                          "note": _TECHNICAL_NOTES.get(fix["type"], "technical fix")}
            elif fix["type"] in _CONTENT_FIX_TYPES and fix.get("valid", True):
                try:
                    res = whatif(fix["path"], [query], base_chunks, base_embeddings,
                                 persist_paths=persist_paths, trace=None)
                    pq = res["per_query"][0]
                    impact = {"kind": "coverage", "label": PREDICTED_LABEL,
                              "coverage_before": pq["coverage_before"],
                              "coverage_after": pq["coverage_after"],
                              "delta": pq["coverage_delta"],
                              "newly_covered_sub_intents": pq["newly_covered"]}
                except Exception as exc:  # what-if is best-effort; never break the report
                    logger.warning(f"[impact] what-if failed for {fid}: {exc}")
                    impact = {"kind": "unknown", "label": PREDICTED_LABEL,
                              "note": "could not be simulated"}
            else:
                impact = {"kind": "unknown", "label": PREDICTED_LABEL,
                          "note": "fix artifact invalid or type not recognized"}

            rec["predicted_impact"] = impact
            cache[cache_key] = impact
            n_computed += 1

        _save_cache(cache_path, cache)
        logger.info(f"[impact] {n_computed} computed, {n_cached} cache hits, "
                    f"{n_skipped} recommendations with no fix artifact")
        if st is not None:
            st.outputs["n_computed"] = n_computed
            st.outputs["n_cached"] = n_cached
            st.outputs["n_skipped"] = n_skipped
            st.scores["n_with_impact"] = float(n_computed + n_cached)
        return recommendations
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
