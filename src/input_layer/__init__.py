"""Universal Input Layer — Phase 6.
Any input format (file, pasted text, interactive) normalizes to
the same internal representation before hitting the graph pipeline.
"""
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

# ── Semantic normalization map ─────────────────────────────────────────────

_SYNONYMS = {
    "wickedgud": "WickedGüd", "wicked gud": "WickedGüd", "wicked good": "WickedGüd",
    "atta": "whole wheat", "atta maggi": "atta noodles",
    "maggie": "maggi",
    "paasata": "pasta", "pastaa": "pasta",
    "no maida": "zero maida", "zero-maida": "zero maida",
    "no palm oil": "zero palm oil", "zero-palm-oil": "zero palm oil",
    "high protein": "high protein",
    "healthy noodles": "healthy noodles", "healthier noodles": "healthy noodles",
    "protein pasta": "protein pasta", "protein noodles": "protein noodles",
    "instant noodles": "instant noodles", "instant ramen": "instant ramen",
    "veg": "vegetarian", "vegetarian": "vegetarian",
    "cup noodles": "cup noodles", "cup ramen": "cup ramen",
    "packet noodles": "packet noodles", "pack noodles": "packet noodles",
    "ramen noodles": "ramen", "noodle ramen": "ramen",
    "spam": "spicy", "spicy": "spicy",
    "cheapest": "affordable", "price": "price",
}

_SLANG = {
    " gud ": " good ", "gud ": "good ", " gud": " good",
    " tho ": " though ", "tho ": "though ",
    " tho": " though",
    " n ": " and ",
}

_NOISE_PATTERNS = [
    r"^(what is|what are|how to|how do|where can|where is|which is|who makes|why is|can i|is there)\s+",
    r"\s*(review|reviews|price|prices|buy|order|shop|cost|buy online)\s*$",
    r"^(best|top|cheap|affordable)\s+(?=noodles|pasta|ramen|maggi)",
]


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
    if lines and lines[0].lower().strip() in ("keyword", "\ufeffkeyword", "keyword\t\t\t"):
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
    """Strip noise, resolve synonyms, canonicalize phrasing."""
    text = text.lower().strip()

    # Remove URL fragments / markdown
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # md links → text

    # Remove leading noise patterns
    for pattern in _NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Slang resolution
    for slang, replacement in _SLANG.items():
        text = text.replace(slang, replacement)

    # Synonym resolution
    for phrase, canonical in sorted(_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if phrase in text:
            text = text.replace(phrase, canonical)

    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ── Intent detection ───────────────────────────────────────────────────────

def _detect_intent(text: str) -> str:
    """Classify query intent type."""
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
    """Fast rule-based entity extraction from query string (no LLM needed)."""
    text_lower = text.lower()
    entities = []

    # Brand — normalize all variants to match graph's stored name
    brand_text = text_lower.replace("ü", "u").replace("ue", "u")
    if any(k in brand_text for k in ["wickedgud", "wicked gud", "wicked good"]):
        entities.append("WickedGüd")
    if "maggi" in text_lower:
        entities.append("Maggi")
    if "samyang" in text_lower:
        entities.append("Samyang")
    if "nissin" in text_lower:
        entities.append("Nissin")
    if "top ramen" in text_lower:
        entities.append("TopRamen")

    # Product types
    for kw, entity in [
        ("noodles", "Noodles"), ("ramen", "Ramen"), ("pasta", "Pasta"),
        ("spaghetti", "Spaghetti"), ("macaroni", "Macaroni"),
        ("wraps", "Wraps"), ("combos", "Combos"), ("byob", "BYOB"),
        ("atta noodles", "Atta Noodles"), ("millet noodles", "Millet Noodles"),
        ("quinoa noodles", "Quinoa Noodles"), ("rice noodles", "Rice Noodles"),
        ("whole wheat pasta", "Whole Wheat Pasta"), ("protein pasta", "Protein Pasta"),
        ("instant noodles", "Instant Noodles"), ("instant ramen", "Instant Ramen"),
        ("cup noodles", "Cup Noodles"), ("packet noodles", "Packet Noodles"),
    ]:
        if kw in text_lower:
            entities.append(entity)

    # Ingredients / health
    for kw, entity in [
        ("maida", "Zero Maida"), ("atta", "Whole Wheat"), ("millet", "Millet"),
        ("quinoa", "Quinoa"), ("rice flour", "Rice Flour"),
        ("protein", "High Protein"), ("gluten free", "Gluten Free"),
        ("no palm oil", "Zero Palm Oil"), ("zero oil", "Zero Palm Oil"),
        ("high fiber", "High Fiber"), ("no artificial", "No Artificial Colors"),
    ]:
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
    """
    Given a NormalizedQuery, return the expected entity reasoning chain.
    Uses detected_entities + heuristics to build an ordered list.
    Guarantees at least 2 distinct entities.

    E.g. "healthy noodles for diabetics" -> ["Healthy Noodles", "Diabetes-Safe", "Low Glycemic"]
    E.g. "wickedgud noodles price"       -> ["WickedGud", "Noodles", "Price"]
    E.g. "maggi calories"                -> ["Maggi", "Calories"]
    """
    entities = list(nq.detected_entities)

    # Heuristic: add inferred intermediate nodes
    has_product = any(e.lower() in ["noodles", "pasta", "ramen", "spaghetti", "wraps", "combos", "byob"] for e in entities)
    has_brand = any(b in entities for b in ["WickedGüd", "WickedGud", "Wicked Gud", "wickedgud"])

    if has_brand and not has_product:
        # "wickedgud noodles" -> brand is start, product inferred
        inferred = _infer_product_type(nq.canonical)
        if inferred and inferred not in entities:
            entities.append(inferred)

    if has_brand and has_product:
        # Add attribute if query mentions health/price/compare
        attr = _infer_attribute(nq.canonical)
        if attr and attr not in entities:
            entities.append(attr)

    # -- ENSURE at least 2 DISTINCT entities -------------------------------
    if len(entities) < 2:
        # Try extracting an attribute from the canonical text
        attr = _infer_attribute(nq.canonical)
        if attr and attr not in entities:
            entities.append(attr)

    if len(entities) < 2:
        # Try looking for a second concept in the original query text
        # (e.g. "maggi calories" -> "calories" maps to "Calories")
        second = _infer_attribute(nq.original)
        if second and second not in entities:
            entities.append(second)

    if len(entities) < 2:
        # Fallback: link the query entity to the site root category
        root = "Noodles"
        if entities:
            # If single entity is not the root, add root
            if entities[0].lower() != root.lower():
                entities.append(root)
            else:
                # If single entity IS the root, add "WickedGud" as brand context
                entities.insert(0, "WickedGud")
        else:
            # No entities at all — use query canonical as first node + root
            entities = [nq.canonical.capitalize(), root]

    # -- Validation: reject same-entity chains ------------------------------
    if len(entities) >= 2:
        if entities[0].lower() == entities[1].lower():
            # Duplicate entities — insert root between them
            entities.insert(1, "Noodles")
        # Remove consecutive duplicates while preserving order
        deduped = [entities[0]]
        for e in entities[1:]:
            if e.lower() != deduped[-1].lower():
                deduped.append(e)
        entities = deduped

    # Log if chain collapsed to < 2 after dedup (should be rare)
    if len(entities) < 2:
        logger = __import__('logging').getLogger(__name__)
        logger.warning(f"chain_extraction_failed: query='{nq.original}' entities={list(nq.detected_entities)}")

    return entities


def _infer_product_type(canonical: str) -> str:
    # Multi-word product types
    for kw, entity in [
        ("millet noodles", "Millet Noodles"),
        ("quinoa noodles", "Quinoa Noodles"),
        ("rice noodles", "Rice Noodles"),
        ("atta noodles", "Atta Noodles"),
        ("whole wheat pasta", "Whole Wheat Pasta"),
        ("protein pasta", "Protein Pasta"),
        ("instant noodles", "Instant Noodles"),
        ("instant ramen", "Instant Ramen"),
        ("cup noodles", "Cup Noodles"),
        ("packet noodles", "Packet Noodles"),
        ("healthy noodles", "Healthy Noodles"),
        ("protein noodles", "Protein Noodles"),
        ("korean noodles", "Korean Noodles"),
        ("spicy noodles", "Spicy Noodles"),
        ("spicy ramen", "Spicy Ramen"),
        ("ramen noodles", "Ramen"),
        ("noodle ramen", "Ramen"),
        ("noodles ramen", "Ramen"),
        ("noodles packet", "Packet Noodles"),
        ("maggi noodles", "Maggi Noodles"),
        ("maggi pasta", "Maggi Pasta"),
        ("maggi calories", "Maggi"),
    ]:
        if kw in canonical:
            return entity
    return "Noodles"  # default


def _infer_attribute(canonical: str) -> str | None:
    for kw, attr in [
        ("calori", "Calories"), ("price", "Price"), ("cost", "Price"),
        ("protein", "High Protein"), ("fiber", "High Fiber"),
        ("maida", "Zero Maida"), ("palm oil", "Zero Palm Oil"),
        ("artificial", "No Artificial Colors"), ("trans fat", "No Trans Fat"),
        ("gluten", "Gluten Free"), ("diabet", "Diabetes-Safe"),
        ("low gly", "Low Glycemic"), ("spicy", "Spicy"),
        ("vegan", "Vegan"), ("veg", "Vegetarian"), ("organic", "Organic"),
        ("compare", "Comparison"), ("review", "Reviews"),
    ]:
        if kw in canonical:
            return attr
    return None