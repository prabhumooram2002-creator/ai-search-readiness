"""Universal Input Layer — Phase 6.
Any input format (file, pasted text, interactive) normalizes to
the same internal representation before hitting the graph pipeline.

Brand / domain lexicon is NOT hardcoded here. It is loaded from
``config/lexicon.yaml`` (empty by default), so the shipped input layer has zero
brand coupling and degrades to generic, brand-agnostic behavior. Populate that
file per project to re-enable synonym normalization and rule-based entity
extraction. Override the path with the ``LEXICON_PATH`` env var.
"""
import os
import json
import re
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import asyncio

from ..core.logging import get_logger

logger = get_logger(__name__)

# ── Intent types ──────────────────────────────────────────────────────────

INTENT_TYPES = ["informational", "comparison", "transactional", "diagnostic"]

# ── Externalized domain lexicon (config/lexicon.yaml, EMPTY by default) ─────

try:
    import yaml as _yaml
except Exception:  # pragma: no cover - yaml is a core dep, guard anyway
    _yaml = None

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_lexicon() -> dict:
    """Load the domain lexicon from config/lexicon.yaml.

    Every section defaults to empty, so a missing/empty file yields a fully
    brand-agnostic lexicon. Never raises — a malformed file logs a warning and
    falls back to empty.
    """
    path = Path(os.getenv("LEXICON_PATH", str(_REPO_ROOT / "config" / "lexicon.yaml")))
    data: dict = {}
    if _yaml is not None and path.exists():
        try:
            data = _yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning(f"lexicon load failed ({path}): {e}; using empty lexicon")
            data = {}
    return {
        "synonyms": dict(data.get("synonyms") or {}),
        "slang": dict(data.get("slang") or {}),
        "noise_patterns": list(data.get("noise_patterns") or []),
        "brands": dict(data.get("brands") or {}),
        "product_types": dict(data.get("product_types") or {}),
        "ingredients": dict(data.get("ingredients") or {}),
        "attributes": dict(data.get("attributes") or {}),
        "hero_entities": [str(e).lower() for e in (data.get("hero_entities") or [])],
        "default_root": data.get("default_root"),
        "default_brand": data.get("default_brand"),
    }


_LEX = _load_lexicon()


def hero_entities() -> set[str]:
    """Lowercased entity names that receive a business-value boost (empty by default)."""
    return set(_LEX["hero_entities"])


@dataclass
class NormalizedQuery:
    """Canonical internal representation of one parsed input unit."""
    original: str               # exactly as provided
    canonical: str             # after semantic normalization
    intent: str                # informational | comparison | transactional | diagnostic
    detected_entities: list[str]  # entities extracted from the query text
    source: str                # file | text | interactive
    volume: Optional[float] = None  # from CSV column if available
    line_number: int = 0       # for error reporting


# ── File parsing ───────────────────────────────────────────────────────────

def parse_file(path: str | Path) -> list[NormalizedQuery]:
    """Auto-detect format and parse file into list[NormalizedQuery]."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".txt", ".tsv"):
        return _parse_txt(path)
    elif suffix in (".csv",):
        return _parse_csv(path)
    elif suffix in (".json",):
        return _parse_json(path)
    elif suffix in (".xlsx", ".xls"):
        return _parse_xlsx(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _auto_decode(path: Path) -> str:
    """Read file with auto-detection of UTF-8, UTF-16, or raw bytes."""
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le")
    if raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16-be")
    return raw.decode("utf-8-sig", errors="replace")


def _parse_txt(path: Path) -> list[NormalizedQuery]:
    """One phrase per line — no header, no volume."""
    queries = []
    lines = _auto_decode(path).splitlines()
    # Skip first line if it looks like a header (contains 'keyword' in lowercase, short, multiple tabs)
    start = 0
    if lines and lines[0].lower().strip() in ("keyword", "﻿keyword", "keyword\t\t\t"):
        start = 1
    for i, line in enumerate(lines[start:], start + 1):
        line = line.strip()
        if not line:
            continue
        queries.append(NormalizedQuery(
            original=line,
            canonical=_semantic_normalize(line),
            intent=_detect_intent(line),
            detected_entities=_extract_entities_from_query(line),
            source=f"file:{path.name}",
            line_number=i,
        ))
    return queries


def _parse_csv(path: Path) -> list[NormalizedQuery]:
    """CSV with arbitrary columns — detect query column automatically."""
    queries = []
    text = _auto_decode(path)
    try:
        delimiter = "\t" if "\t" in text.splitlines()[0] else ","
        reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
        rows = list(reader)
    except Exception:
        # Fall back to treating first column as keyword
        reader = None
        rows = []

    if reader and rows:
        # Find the best column for query text
        col = _detect_query_column(reader.fieldnames or [], rows)
        if col is None:
            col = list(rows[0].keys())[0]  # fallback: first column
        vol_col = next((k for k in rows[0].keys() if "vol" in k.lower()), None)
    else:
        # Tab-separated simple keyword list (like the test file)
        col = "keyword"
        rows = [{"keyword": line.strip()} for line in text.splitlines() if line.strip()]
        vol_col = None

    for i, row in enumerate(rows, 1):
        raw = row.get(col, "").strip()
        if not raw:
            continue
        vol = float(row[vol_col]) if vol_col and vol_col in row and row[vol_col] else None
        queries.append(NormalizedQuery(
            original=raw,
            canonical=_semantic_normalize(raw),
            intent=_detect_intent(raw),
            detected_entities=_extract_entities_from_query(raw),
            source=f"file:{path.name}",
            volume=vol,
            line_number=i,
        ))

    logger.info(f"Parsed CSV: {len(queries)} rows, query_col={col}, vol_col={vol_col}")
    return queries


def _parse_json(path: Path) -> list[NormalizedQuery]:
    """JSON array of strings OR array of {query, source, volume?} objects."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    queries = []
    i = 0

    if not data:
        return queries

    # Auto-detect: list of strings or list of dicts
    if isinstance(data[0], str):
        for i, item in enumerate(data, 1):
            queries.append(NormalizedQuery(
                original=item,
                canonical=_semantic_normalize(item),
                intent=_detect_intent(item),
                detected_entities=_extract_entities_from_query(item),
                source=f"file:{path.name}",
                line_number=i,
            ))
    else:
        for i, item in enumerate(data, 1):
            raw = item.get("query", "") or item.get("text", "") or item.get("keyword", "")
            if not raw:
                continue
            queries.append(NormalizedQuery(
                original=raw,
                canonical=_semantic_normalize(raw),
                intent=_detect_intent(raw),
                detected_entities=_extract_entities_from_query(raw),
                source=item.get("source", f"file:{path.name}"),
                volume=item.get("volume"),
                line_number=i,
            ))

    return queries


def _parse_xlsx(path: Path) -> list[NormalizedQuery]:
    """Excel — requires openpyxl. Falls back gracefully if missing."""
    try:
        import openpyxl
    except ImportError:
        logger.warning("openpyxl not installed — cannot parse .xlsx files")
        return []

    queries = []
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    # First row = header
    headers = [str(h).strip() if h else f"col_{j}" for j, h in enumerate(rows[0])]
    data_rows = rows[1:]

    col = _detect_query_column(headers, [dict(zip(headers, r)) for r in data_rows[:10]])
    if col is None:
        col = headers[0]

    vol_col = next((h for h in headers if "vol" in h.lower()), None)

    for i, row in enumerate(data_rows, 2):
        d = dict(zip(headers, row))
        raw = str(d.get(col, "")).strip()
        if not raw or raw == "None":
            continue
        vol = float(d[vol_col]) if vol_col and d.get(vol_col) not in (None, "None") else None
        queries.append(NormalizedQuery(
            original=raw,
            canonical=_semantic_normalize(raw),
            intent=_detect_intent(raw),
            detected_entities=_extract_entities_from_query(raw),
            source=f"file:{path.name}",
            volume=vol,
            line_number=i,
        ))

    wb.close()
    logger.info(f"Parsed XLSX: {len(queries)} rows")
    return queries


def _detect_query_column(columns: list[str], sample_rows: list[dict]) -> str | None:
    """Pick the best column for query text from a header list."""
    score = {}
    for col in columns:
        cl = col.lower().strip()
        for row in sample_rows[:10]:
            val = str(row.get(col, "")).strip().lower()
            if not val:
                continue
            # Good indicators
            if any(k in cl for k in ["query", "keyword", "phrase", "search", "term"]):
                score[col] = score.get(col, 0) + 10
            elif len(val) > 15 and len(val) < 200:
                score[col] = score.get(col, 0) + 3
            elif len(val.split()) <= 10:
                score[col] = score.get(col, 0) + 1

    if not score:
        return None
    return max(score, key=score.get)


# ── Segmentation for unstructured text ─────────────────────────────────────

def segment_text(text: str) -> list[NormalizedQuery]:
    """Split pasted blocks (Reddit threads, reviews, etc.) into discrete queries."""
    queries = []

    # Split on sentence-like boundaries
    segments = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    if len(segments) == 1:
        # No sentence breaks — split on newlines
        segments = [s.strip() for s in text.splitlines() if s.strip()]

    for i, seg in enumerate(segments, 1):
        seg = seg.strip()
        if len(seg) < 8:
            continue  # skip fragments
        queries.append(NormalizedQuery(
            original=seg,
            canonical=_semantic_normalize(seg),
            intent=_detect_intent(seg),
            detected_entities=_extract_entities_from_query(seg),
            source="pasted_text",
            line_number=i,
        ))

    return queries


# ── Semantic normalization ─────────────────────────────────────────────────

def _semantic_normalize(text: str) -> str:
    """Strip noise, resolve synonyms, canonicalize phrasing.

    All noise patterns / slang / synonyms come from the externalized lexicon
    (empty by default → only generic cleanup: URL/markdown strip + whitespace).
    """
    text = text.lower().strip()

    # Remove URL fragments / markdown
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # md links → text

    # Remove configured noise patterns
    for pattern in _LEX["noise_patterns"]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Slang resolution
    for slang, replacement in _LEX["slang"].items():
        text = text.replace(slang, replacement)

    # Synonym resolution (longest phrase first)
    for phrase, canonical in sorted(_LEX["synonyms"].items(), key=lambda x: -len(x[0])):
        if phrase in text:
            text = text.replace(phrase, str(canonical))

    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── Intent detection ───────────────────────────────────────────────────────

def _detect_intent(text: str) -> str:
    """Classify query intent type using generic, brand-agnostic signals."""
    text_lower = text.lower()

    if any(k in text_lower for k in ["compare", "vs ", "versus", "difference between", "better", "best", "top ", "reviews"]):
        return "comparison"
    if any(k in text_lower for k in ["buy", "order", "price", "cost", "where to buy", "shop", "purchase", "coupon", "discount"]):
        return "transactional"
    if any(k in text_lower for k in ["why", "because", "reason", "cause", "how does", "works", "ingredients", "nutrit"]):
        return "diagnostic"
    return "informational"


# ── Entity extraction from query text ───────────────────────────────────────

def _extract_entities_from_query(text: str) -> list[str]:
    """Rule-based entity extraction driven by the externalized lexicon.

    With the default (empty) lexicon this returns [] — no brand assumptions.
    """
    text_lower = text.lower()
    entities: list[str] = []

    for substr, entity in _LEX["brands"].items():
        if substr in text_lower:
            entities.append(entity)
    # product types then ingredients (longest keyword first for specificity)
    for section in ("product_types", "ingredients"):
        for kw, entity in sorted(_LEX[section].items(), key=lambda x: -len(x[0])):
            if kw in text_lower:
                entities.append(entity)

    # Deduplicate preserving order
    seen = set()
    out = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            out.append(e)

    return out


# ── Expected entity chain from query ──────────────────────────────────────

def parse_expected_chain(nq: NormalizedQuery, graph=None) -> list[str]:
    """Return the expected entity reasoning chain for a NormalizedQuery.

    Chain inference is lexicon-driven. With the default (empty) lexicon there
    are no product/brand defaults, so this returns the detected entities as-is
    (possibly fewer than 2) and logs an incomplete-chain warning rather than
    inventing brand nodes.
    """
    entities = list(nq.detected_entities)
    root = _LEX["default_root"]
    brand = _LEX["default_brand"]
    product_kws = list(_LEX["product_types"].keys())

    has_product = any(any(pk in e.lower() for pk in product_kws) for e in entities) if product_kws else False
    has_brand = bool(brand) and any(brand.lower() == e.lower() for e in entities)

    if has_brand and not has_product:
        inferred = _infer_product_type(nq.canonical)
        if inferred and inferred not in entities:
            entities.append(inferred)

    if has_brand and has_product:
        attr = _infer_attribute(nq.canonical)
        if attr and attr not in entities:
            entities.append(attr)

    # -- Try to reach 2 distinct entities via attributes -------------------
    if len(entities) < 2:
        attr = _infer_attribute(nq.canonical)
        if attr and attr not in entities:
            entities.append(attr)

    if len(entities) < 2:
        second = _infer_attribute(nq.original)
        if second and second not in entities:
            entities.append(second)

    # -- Lexicon-driven fallback (only when a default_root is configured) ---
    if len(entities) < 2 and root:
        if entities:
            if entities[0].lower() != root.lower():
                entities.append(root)
            elif brand:
                entities.insert(0, brand)
        else:
            entities = [nq.canonical.capitalize(), root] if nq.canonical else [root]

    # -- Validation: reject consecutive-duplicate chains --------------------
    if len(entities) >= 2:
        if entities[0].lower() == entities[1].lower() and root:
            entities.insert(1, root)
        deduped = [entities[0]]
        for e in entities[1:]:
            if e.lower() != deduped[-1].lower():
                deduped.append(e)
        entities = deduped

    if len(entities) < 2:
        logger.warning(
            f"chain_extraction_incomplete: query='{nq.original}' "
            f"entities={list(nq.detected_entities)} (empty/insufficient lexicon)"
        )

    return entities


def _infer_product_type(canonical: str) -> str | None:
    """Longest-match product type from the lexicon, else the configured default_root (or None)."""
    for kw, entity in sorted(_LEX["product_types"].items(), key=lambda x: -len(x[0])):
        if kw in canonical:
            return entity
    return _LEX["default_root"]


def _infer_attribute(canonical: str) -> str | None:
    """First matching attribute from the lexicon (None if unmatched / empty lexicon)."""
    for kw, attr in _LEX["attributes"].items():
        if kw in canonical:
            return attr
    return None
