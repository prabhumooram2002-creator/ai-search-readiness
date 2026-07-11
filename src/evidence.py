"""Layer 1 step 6 — Evidence Extraction via NLI (CLAUDE.md).

For each step-5 claim, build a ±2-sentence evidence span around its source
sentence and score it with the local cross-encoder NLI model
(``cross-encoder/nli-deberta-v3-base``): entailment / neutral / contradiction
plus confidence. Fully local, zero keys.

- Output: ``evidence{claim_id, evidence_text, entailment_label,
  entailment_confidence}`` (+ claim text and chunk id for downstream use).
- Verification rule (enforced here, per CLAUDE.md): a claim that CONTRADICTS
  its own source context is flagged and re-run once against the full chunk
  text; if it still contradicts, it stays flagged for review.
- Trace: emits one StepTrace named ``evidence_extraction`` when a Trace is
  passed, including on error.
"""
from __future__ import annotations

import math
import os
from typing import Optional

from .claims import segment_sentences
from .core.logging import get_logger

logger = get_logger(__name__)

NLI_MODEL = os.getenv("NLI_MODEL", "cross-encoder/nli-deberta-v3-base")
CONTEXT_SENTENCES = 2  # ±2 per CLAUDE.md

_nli_model = None


def _get_nli():
    global _nli_model
    if _nli_model is None:
        # Same Windows fix as ner.py/relations.py: HF cache copy mode.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        try:
            from huggingface_hub import constants as _hfc
            _hfc.HF_HUB_DISABLE_SYMLINKS = True
        except Exception:
            pass
        from sentence_transformers import CrossEncoder  # lazy (pulls torch)
        logger.info(f"Loading NLI cross-encoder {NLI_MODEL} (first load downloads the model)...")
        _nli_model = CrossEncoder(NLI_MODEL)
    return _nli_model


def _id2label(model) -> dict[int, str]:
    """Read the label order from the model config — never hardcode it."""
    raw = getattr(getattr(model, "model", model).config, "id2label", None) or {}
    labels = {int(k): str(v).lower() for k, v in raw.items()}
    return labels or {0: "contradiction", 1: "entailment", 2: "neutral"}


def _softmax(logits) -> list[float]:
    exps = [math.exp(float(x)) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _score_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """NLI-score (premise, hypothesis) pairs -> [(label, confidence)]."""
    model = _get_nli()
    labels = _id2label(model)
    logits = model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
    out = []
    for row in logits:
        probs = _softmax(row)
        best = max(range(len(probs)), key=lambda i: probs[i])
        out.append((labels.get(best, str(best)), round(probs[best], 4)))
    return out


def _evidence_span(sentences: list[dict], number: int, k: int = CONTEXT_SENTENCES) -> str:
    """±k sentences around 1-based sentence `number` (inclusive of it)."""
    idx = next((i for i, s in enumerate(sentences) if s["number"] == number), None)
    if idx is None:
        return ""
    lo, hi = max(0, idx - k), min(len(sentences), idx + k + 1)
    return " ".join(s["text"] for s in sentences[lo:hi])


def extract_evidence(chunks: list[dict], claims: list[dict], trace=None) -> list[dict]:
    """Score each claim against its ±2-sentence source context.

    chunks: [{"content": str, "chunk_id": str}, ...]
    claims: step-5 records (``src.claims.extract_claims`` output).
    Returns evidence records; contradiction-flagged claims are re-run once
    against the full chunk text (flag preserved either way in
    ``contradiction_flags`` on the StepTrace and in the record itself).
    """
    step_cm = trace.start_step(
        "evidence_extraction", n_claims=len(claims), model=NLI_MODEL,
        context_sentences=CONTEXT_SENTENCES,
    ) if trace is not None else None
    st = step_cm.__enter__() if step_cm is not None else None

    try:
        chunk_by_id = {str(c.get("chunk_id", "")): c for c in chunks}
        sentences_cache: dict[str, list[dict]] = {}

        # Validate claims and build rows (sentence premise + ±2-span premise)
        rows: list[dict] = []
        for cl in claims:
            chunk_id = cl["source_chunk_id"]
            chunk = chunk_by_id.get(chunk_id)
            if chunk is None:
                if st is not None:
                    st.drop({"claim_id": cl["id"]}, "source chunk not provided")
                continue
            if chunk_id not in sentences_cache:
                sentences_cache[chunk_id] = segment_sentences(chunk["content"], chunk_id)
            try:
                number = int(str(cl["source_sentence_id"]).rsplit(":s", 1)[1])
            except (IndexError, ValueError):
                if st is not None:
                    st.drop({"claim_id": cl["id"]}, "unparseable source_sentence_id")
                continue
            sentences = sentences_cache[chunk_id]
            span = _evidence_span(sentences, number)
            if not span:
                if st is not None:
                    st.drop({"claim_id": cl["id"]}, "source sentence not found in chunk")
                continue
            sent_text = next(s["text"] for s in sentences if s["number"] == number)
            rows.append({"claim": cl, "evidence_text": span, "sentence_text": sent_text})

        # Pass 1 — premise = the exact source sentence. This cross-encoder is
        # trained on short premises; multi-sentence spans demonstrably turn
        # true entailments into "neutral" (see BUILD_LOG step-6 diagnostic).
        res1 = _score_pairs([(r["sentence_text"], r["claim"]["text"]) for r in rows]) if rows else []

        # Pass 2 — non-entailed claims retry against the ±2 span (context may
        # be needed to resolve references that span sentences).
        retry_idx = [i for i, (l, _c) in enumerate(res1) if l != "entailment"]
        res2_by_idx: dict[int, tuple[str, float]] = {}
        if retry_idx:
            res2 = _score_pairs([(rows[i]["evidence_text"], rows[i]["claim"]["text"])
                                 for i in retry_idx])
            res2_by_idx = dict(zip(retry_idx, res2))

        evidence: list[dict] = []
        retry_rows: list[dict] = []
        retry_pairs: list[tuple[str, str]] = []
        for i, (row, (l1, c1)) in enumerate(zip(rows, res1)):
            if l1 == "entailment":
                label, conf, premise = l1, c1, "source_sentence"
            else:
                l2, c2 = res2_by_idx[i]
                if l2 == "entailment":
                    label, conf, premise = l2, c2, "evidence_span"
                elif "contradiction" in (l1, l2):
                    label = "contradiction"
                    conf = max(c for l, c in ((l1, c1), (l2, c2)) if l == "contradiction")
                    premise = "source_sentence" if l1 == "contradiction" else "evidence_span"
                else:
                    label, conf, premise = l2, c2, "evidence_span"
            rec = {
                "claim_id": row["claim"]["id"],
                "claim_text": row["claim"]["text"],
                "source_chunk_id": row["claim"]["source_chunk_id"],
                "evidence_text": row["evidence_text"],
                "entailment_label": label,
                "entailment_confidence": conf,
                "premise_used": premise,
                "contradiction_flagged": False,
            }
            if label == "contradiction":
                chunk = chunk_by_id[row["claim"]["source_chunk_id"]]
                retry_rows.append(rec)
                retry_pairs.append((chunk["content"], row["claim"]["text"]))
            evidence.append(rec)

        if retry_pairs:
            logger.warning(
                f"NLI: {len(retry_pairs)} claim(s) contradict their own source "
                f"context -> re-running against full chunk text"
            )
            for rec, (label2, conf2) in zip(retry_rows, _score_pairs(retry_pairs)):
                rec["contradiction_flagged"] = True  # flag survives the re-run
                rec["rerun_label"] = label2
                rec["rerun_confidence"] = conf2
                if label2 != "contradiction":
                    # wider context resolved it; keep the better read but stay flagged
                    rec["entailment_label"] = label2
                    rec["entailment_confidence"] = conf2

        counts: dict[str, int] = {}
        for rec in evidence:
            counts[rec["entailment_label"]] = counts.get(rec["entailment_label"], 0) + 1
        flagged = [r["claim_id"] for r in evidence if r["contradiction_flagged"]]
        logger.info(f"Evidence: {len(evidence)} records, labels={counts}, flagged={len(flagged)}")

        if st is not None:
            st.outputs["evidence"] = evidence
            st.outputs["label_counts"] = counts
            st.outputs["contradiction_flags"] = flagged
            st.scores["n_evidence"] = float(len(evidence))            # heuristic
            st.scores["n_contradiction_flags"] = float(len(flagged))  # heuristic
        return evidence
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
