"""Layer 1 steps 1-2 — formal chunking nodes (v3 Phase 3a).

Step 1 — Sentence segmentation: pysbd with char offsets preserved, plus an
  abbreviation-merge post-pass (pysbd splits "Est. 2021" mid-number).
Step 2 — Semantic chunking: split at markdown headings first; within an
  oversized section, sub-split where consecutive-sentence cosine similarity
  (all-MiniLM-L6-v2, local) drops below a threshold; chunks <= ~500 tokens;
  heading_path + char range preserved on every chunk.

Both emit a StepTrace when a Trace is passed.
"""
from __future__ import annotations

import hashlib
import re
from typing import Optional

from .core.logging import get_logger

logger = get_logger(__name__)

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIM_THRESHOLD = 0.5          # cosine-drop sub-split boundary
MAX_TOKENS = 500
CHARS_PER_TOKEN = 4          # rough token estimate without a tokenizer
MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN

# Abbreviations pysbd tends to break on when followed by a number/lowercase.
_ABBREV = {
    "est", "no", "vol", "pp", "inc", "ltd", "co", "corp", "etc", "eg", "ie",
    "vs", "dr", "mr", "mrs", "ms", "jr", "sr", "st", "ave", "approx", "fig",
    "al", "dept", "univ", "mt", "gen", "rev", "ch", "sec", "art", "para",
}

_segmenter = None
_minilm = None


def _get_segmenter():
    global _segmenter
    if _segmenter is None:
        import pysbd
        _segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)
    return _segmenter


def _get_minilm():
    global _minilm
    if _minilm is None:
        import os
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading {MINILM_MODEL} for semantic chunking...")
        _minilm = SentenceTransformer(MINILM_MODEL)
    return _minilm


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — segmentation with abbreviation merge
# ─────────────────────────────────────────────────────────────────────────────
def _should_merge(prev_text: str, next_text: str) -> bool:
    """True if pysbd wrongly split an abbreviation (prev) from its continuation."""
    m = re.search(r"([A-Za-z]+)\.[\"')\s]*$", prev_text)
    if not m:
        return False
    word, first = m.group(1), next_text.lstrip()[:1]
    if not first:
        return False
    if word.lower() in _ABBREV and (first.isdigit() or first.islower()):
        return True
    # generic: short Capitalized token + period, continuation starts with a digit
    if len(word) <= 4 and word[0].isupper() and first.isdigit():
        return True
    return False


def segment_sentences(text: str, chunk_id: str = "", trace=None) -> list[dict]:
    """pysbd sentences with char offsets + abbreviation merge.

    Returns [{"sentence_id", "number", "text", "char_start", "char_end"}].
    """
    spans = _get_segmenter().segment(text)
    merged: list[dict] = []
    for sp in spans:
        if merged and _should_merge(merged[-1]["_raw"], sp.sent):
            merged[-1]["_raw"] += sp.sent
            merged[-1]["end"] = int(sp.end)
        else:
            merged.append({"_raw": sp.sent, "start": int(sp.start),
                           "end": int(sp.end)})

    out = []
    n = 0
    for m in merged:
        stripped = m["_raw"].strip()
        if not stripped:
            continue
        n += 1
        out.append({
            "sentence_id": f"{chunk_id}:s{n}" if chunk_id else f"s{n}",
            "number": n, "text": stripped,
            "char_start": m["start"], "char_end": m["end"],
        })
    if trace is not None:
        with trace.start_step("sentence_segmentation", chunk_id=chunk_id,
                              model="pysbd") as st:
            st.outputs["n_sentences"] = len(out)
            st.scores["n_sentences"] = float(len(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — semantic chunking (heading-first + cosine-drop sub-split)
# ─────────────────────────────────────────────────────────────────────────────
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _split_sections(content: str) -> list[dict]:
    """Split markdown into sections by heading, tracking heading_path + offsets."""
    sections: list[dict] = []
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    body_start = 0
    pos = 0

    def flush():
        if body and "".join(body).strip():
            sections.append({"heading_path": [h[1] for h in stack],
                             "text": "".join(body),
                             "char_start": body_start})

    for line in content.splitlines(keepends=True):
        m = _HEADING.match(line)
        if m:
            flush()
            body = []
            level, htext = len(m.group(1)), m.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, htext))
            body_start = pos + len(line)
        else:
            if not body:
                body_start = pos
            body.append(line)
        pos += len(line)
    flush()
    return sections


def _token_est(text: str) -> int:
    return max(len(text) // CHARS_PER_TOKEN, len(text.split()))


def _cosine(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    d = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(a.dot(b) / d)


def semantic_chunks(
    content: str,
    url: str = "",
    sim_threshold: float = SIM_THRESHOLD,
    max_tokens: int = MAX_TOKENS,
    trace=None,
) -> list[dict]:
    """Split a page into semantic chunks. Every chunk carries heading_path and
    an exact (char_start, char_end) range into ``content``.

    Returns [{"id", "page_url", "heading_path", "text", "char_start",
              "char_end", "token_estimate"}].
    """
    step_cm = trace.start_step("semantic_chunking", url=url,
                               sim_threshold=sim_threshold,
                               max_tokens=max_tokens) if trace else None
    st = step_cm.__enter__() if step_cm else None
    try:
        chunks: list[dict] = []
        model = None
        for sec in _split_sections(content):
            sents = segment_sentences(sec["text"])
            if not sents:
                continue
            # offsets are relative to the section body -> shift to page offsets
            base = sec["char_start"]
            embs = None
            if len(sents) > 1:
                model = model or _get_minilm()
                embs = model.encode([s["text"] for s in sents],
                                    normalize_embeddings=True,
                                    show_progress_bar=False)

            cur: list[dict] = []
            cur_tokens = 0
            for i, s in enumerate(sents):
                tok = _token_est(s["text"])
                boundary = False
                if cur:
                    if cur_tokens + tok > max_tokens:
                        boundary = True
                    elif embs is not None and _cosine(embs[i], embs[i - 1]) < sim_threshold:
                        boundary = True
                if boundary:
                    chunks.append(_mk_chunk(cur, sec, base, url, cur_tokens))
                    cur, cur_tokens = [], 0
                cur.append(s)
                cur_tokens += tok
            if cur:
                chunks.append(_mk_chunk(cur, sec, base, url, cur_tokens))

        # renumber ids stable per page
        for idx, c in enumerate(chunks):
            c["id"] = _chunk_id(url, idx, c["text"])
        logger.info(f"Semantic chunking: {len(chunks)} chunks from "
                    f"{len(_split_sections(content))} sections "
                    f"(<= {max_tokens} tokens, sim_threshold={sim_threshold})")
        if st is not None:
            st.outputs["n_chunks"] = len(chunks)
            st.outputs["max_token_estimate"] = max(
                (c["token_estimate"] for c in chunks), default=0)
            st.scores["n_chunks"] = float(len(chunks))
        return chunks
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)


def _mk_chunk(sents: list[dict], sec: dict, base: int, url: str,
             tokens: int) -> dict:
    # token_estimate = the SAME running sum used for the budget decision, so a
    # chunk that passed the <=max_tokens gate never reports over it.
    text = " ".join(s["text"] for s in sents)
    return {
        "id": "", "page_url": url, "heading_path": list(sec["heading_path"]),
        "text": text,
        "char_start": base + sents[0]["char_start"],
        "char_end": base + sents[-1]["char_end"],
        "token_estimate": tokens,
    }


def _chunk_id(url: str, index: int, text: str) -> str:
    h = hashlib.sha256(f"{url}|{index}|{text[:80]}".encode()).hexdigest()[:16]
    return f"chunk_{h}"
