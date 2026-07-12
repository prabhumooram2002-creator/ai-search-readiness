"""PHASE 8 — Fix generation (v3 brief): emit patches, not advice.

Upgrades Layer-3 recommendations from prose into ready-to-use artifacts, ALL
clearly labeled *draft — human review required, never auto-publish*:

- missing/invalid schema  -> a complete JSON-LD block for the page (validated
  structurally against schema.org requirements before emitting)
- site-level               -> `llms.txt` from the KG (top pages, topics,
  canonical descriptions) — only real URLs
- retrieval dead end       -> local-LLM-drafted FAQ/section for the missing
  sub-intent, seeded with the source claims it should contain
- structure flags          -> concrete block rewrite (table for numeric prose,
  split paragraphs)

Artifacts are written to ``fixes/<run>/`` and each maps back to a specific
report finding by id (returned manifest).
"""
from __future__ import annotations

import json as _json
import re
from pathlib import Path
from typing import Optional

from . import providers
from .core.config import BASE_DIR
from .core.logging import get_logger

logger = get_logger(__name__)

FIXES_DIR = BASE_DIR / "fixes"
DRAFT_BANNER = "draft — human review required, never auto-publish"

# schema.org type -> required properties (pragmatic validation subset)
_REQUIRED = {
    "WebPage": ["name", "url"],
    "Article": ["headline", "url"],
    "FAQPage": ["mainEntity"],
    "Organization": ["name", "url"],
}


# ─────────────────────────────────────────────────────────────────────────────
# JSON-LD
# ─────────────────────────────────────────────────────────────────────────────
def _is_question(text: str) -> bool:
    return text.strip().endswith("?")


def generate_jsonld(page: dict) -> dict:
    """Build a schema.org JSON-LD block for a page.

    page: {"url", "title", "headings"?: [{level,text}], "content"?}. FAQ-shaped
    pages (question headings) become FAQPage; otherwise Article/WebPage.
    """
    url = page["url"]
    title = page.get("title") or url
    desc = (page.get("content") or "").strip().replace("\n", " ")[:200]
    q_headings = [h["text"] for h in page.get("headings", []) if _is_question(h["text"])]

    if len(q_headings) >= 2:
        block = {
            "@context": "https://schema.org", "@type": "FAQPage", "url": url,
            "name": title,
            "mainEntity": [
                {"@type": "Question", "name": q,
                 "acceptedAnswer": {"@type": "Answer",
                                    "text": "<answer drawn from the section — review>"}}
                for q in q_headings[:10]],
        }
    else:
        block = {
            "@context": "https://schema.org", "@type": "Article",
            "headline": title[:110], "url": url, "name": title,
            "description": desc or title,
        }
    return block


def validate_jsonld(block: dict) -> tuple[bool, list[str]]:
    """Structural validation against schema.org requirements (not a network
    call). Returns (ok, errors)."""
    errors = []
    if str(block.get("@context", "")).rstrip("/") not in (
            "https://schema.org", "http://schema.org"):
        errors.append("@context must be https://schema.org")
    typ = block.get("@type")
    if not typ:
        errors.append("@type missing")
    for prop in _REQUIRED.get(typ, []):
        if not block.get(prop):
            errors.append(f"{typ} requires '{prop}'")
    if typ == "FAQPage":
        for i, qa in enumerate(block.get("mainEntity", []) or []):
            if qa.get("@type") != "Question" or not qa.get("name"):
                errors.append(f"mainEntity[{i}] must be a Question with a name")
            if not qa.get("acceptedAnswer", {}).get("text"):
                errors.append(f"mainEntity[{i}] missing acceptedAnswer.text")
    try:
        _json.dumps(block)
    except (TypeError, ValueError) as e:
        errors.append(f"not JSON-serializable: {e}")
    return (not errors, errors)


# ─────────────────────────────────────────────────────────────────────────────
# llms.txt
# ─────────────────────────────────────────────────────────────────────────────
def generate_llms_txt(site_name: str, pages: list[dict], topics: list[dict],
                      description: str = "") -> str:
    """Build an llms.txt from real pages + topics. Only real URLs are listed."""
    real = [p for p in pages if str(p.get("url", "")).startswith("http")]
    real.sort(key=lambda p: -(p.get("pagerank") or 0))
    lines = [f"# {site_name}", ""]
    if description:
        lines += [f"> {description}", ""]
    lines.append(f"<!-- {DRAFT_BANNER} -->\n")
    lines.append("## Pages")
    for p in real[:50]:
        d = (p.get("description") or p.get("title") or "").strip().replace("\n", " ")[:120]
        lines.append(f"- [{p.get('title') or p['url']}]({p['url']})"
                     + (f": {d}" if d else ""))
    if topics:
        lines += ["", "## Topics"]
        lines += [f"- {t['label']}" for t in topics if t.get("label")]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# FAQ / section draft for a dead-end sub-intent (local LLM)
# ─────────────────────────────────────────────────────────────────────────────
def generate_faq(sub_intent: str, source_claims: Optional[list[str]] = None) -> str:
    claims = "\n".join(f"- {c}" for c in (source_claims or [])) or "(none on file)"
    raw = providers.llm_complete(
        f"Draft a short FAQ section that answers this user sub-intent for a "
        f"website: \"{sub_intent}\". Ground it ONLY in these known facts:\n"
        f"{claims}\nIf facts are missing, write a bracketed [TODO: confirm ...].\n"
        'Return ONLY JSON: {"heading": "...", "qa": [{"q":"...","a":"..."}]}',
        json=True, temperature=0,
        system="You draft factual FAQ content. Output ONLY valid JSON.")
    try:
        data = _json.loads(raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
    except _json.JSONDecodeError:
        data = {"heading": sub_intent, "qa": []}
    md = [f"<!-- {DRAFT_BANNER} -->", f"## {data.get('heading', sub_intent)}", ""]
    for qa in data.get("qa", []):
        md += [f"**{qa.get('q','')}**", "", qa.get("a", ""), ""]
    return "\n".join(md)


# ─────────────────────────────────────────────────────────────────────────────
# Structure rewrite (local LLM)
# ─────────────────────────────────────────────────────────────────────────────
def rewrite_block(text: str, flags: list[str]) -> str:
    raw = providers.llm_complete(
        f"Rewrite the content block below to fix these structural issues: "
        f"{', '.join(flags)}. If it contains a numeric series, output a "
        f"markdown table; if a paragraph is too long, split into single-claim "
        f"paragraphs. Preserve all facts; invent nothing.\n\n{text[:2000]}\n\n"
        "Return ONLY the rewritten markdown.",
        temperature=0,
        system="You restructure content without changing its facts.")
    return f"<!-- {DRAFT_BANNER} -->\n\n{raw.strip()}\n"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration — emit files mapped to findings
# ─────────────────────────────────────────────────────────────────────────────
def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:40] or "fix"


def generate_fixes(run_id: int, *, pages_without_schema: list[dict] | None = None,
                   site: dict | None = None, dead_ends: list[dict] | None = None,
                   structure_flagged: list[dict] | None = None,
                   fixes_dir: Optional[Path] = None, use_llm: bool = True,
                   trace=None) -> dict:
    """Write fix artifacts to fixes/<run>/ and return a manifest mapping each
    finding id -> {type, path, valid?}."""
    root = (fixes_dir or FIXES_DIR) / f"run_{run_id:04d}"
    root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    step_cm = trace.start_step("fix_generation", run_id=run_id) if trace else None
    st = step_cm.__enter__() if step_cm else None
    try:
        # 1. JSON-LD per page missing schema (validated before emit)
        for pg in (pages_without_schema or []):
            block = generate_jsonld(pg)
            ok, errs = validate_jsonld(block)
            fid = f"schema:{_slug(pg['url'])}"
            path = root / f"schema_{_slug(pg['url'])}.jsonld"
            path.write_text(_json.dumps(block, indent=2), encoding="utf-8")
            manifest.append({"finding_id": fid, "type": "jsonld",
                             "path": str(path), "valid": ok, "errors": errs})

        # 2. site-level llms.txt
        if site:
            txt = generate_llms_txt(site.get("name", "Site"), site.get("pages", []),
                                    site.get("topics", []), site.get("description", ""))
            path = root / "llms.txt"
            path.write_text(txt, encoding="utf-8")
            manifest.append({"finding_id": "site:llms.txt", "type": "llms_txt",
                             "path": str(path), "valid": True})

        # 3. FAQ drafts for retrieval dead ends
        for de in (dead_ends or []):
            sub = de.get("sub_query") or de.get("query", "")
            content = (generate_faq(sub, de.get("source_claims"))
                       if use_llm else f"<!-- {DRAFT_BANNER} -->\n## {sub}\n")
            fid = f"deadend:{_slug(sub)}"
            path = root / f"faq_{_slug(sub)}.md"
            path.write_text(content, encoding="utf-8")
            manifest.append({"finding_id": fid, "type": "faq_draft",
                             "path": str(path), "valid": True})

        # 4. block rewrites for structure flags
        for ch in (structure_flagged or []):
            flags = [f["flag"] for f in ch.get("flags", [])]
            content = (rewrite_block(ch.get("text", ""), flags)
                       if use_llm else f"<!-- {DRAFT_BANNER} -->\n")
            fid = f"structure:{ch.get('chunk_id')}"
            path = root / f"rewrite_{_slug(ch.get('chunk_id'))}.md"
            path.write_text(content, encoding="utf-8")
            manifest.append({"finding_id": fid, "type": "block_rewrite",
                             "path": str(path), "valid": True, "flags": flags})

        (root / "manifest.json").write_text(
            _json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"[fixes] wrote {len(manifest)} artifacts -> {root}")
        if st is not None:
            st.outputs["manifest"] = manifest
            st.outputs["n_fixes"] = len(manifest)
            st.scores["n_invalid_jsonld"] = float(
                sum(1 for m in manifest if m["type"] == "jsonld" and not m["valid"]))
        return {"run_dir": str(root), "manifest": manifest}
    except BaseException as exc:
        if step_cm is not None:
            step_cm.__exit__(type(exc), exc, exc.__traceback__)
            step_cm = None
        raise
    finally:
        if step_cm is not None:
            step_cm.__exit__(None, None, None)
