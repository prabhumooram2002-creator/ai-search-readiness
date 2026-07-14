"""Section 1 — Website Identity [NEW MODULE] (INTELLIGENCE_REPORT_BRIEF.md).

Extracts the site's SELF-DECLARED identity from data already crawled — no
new network calls, no external lookups on the default path. "Does ChatGPT
know this brand?" is answered ONLY from calibration-panel data when present
(labeled observed); otherwise the report prints "unknown — run a
calibration panel", never a guess.
"""
from __future__ import annotations

import re

_IDENTITY_TYPES = {"Organization", "LocalBusiness", "Person", "Corporation"}
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}")


def _flatten_jsonld(block) -> list[dict]:
    """@graph-wrapped or single blocks -> a flat list of dicts."""
    if isinstance(block, list):
        out = []
        for b in block:
            out.extend(_flatten_jsonld(b))
        return out
    if isinstance(block, dict):
        if "@graph" in block:
            return _flatten_jsonld(block["@graph"])
        return [block]
    return []


def extract_identity_blocks(l0_pages: list[dict]) -> list[dict]:
    """Every Organization/LocalBusiness/Person JSON-LD block found site-wide,
    tagged with the page it came from."""
    found = []
    for p in l0_pages:
        for raw in p.get("schema_jsonld", []) or []:
            for block in _flatten_jsonld(raw):
                if block.get("@type") in _IDENTITY_TYPES:
                    found.append({"url": p["url"], "block": block})
    return found


def contact_signals(l0_pages: list[dict]) -> dict:
    """Emails/phones found in raw page text, for the contact-consistency
    check (also reused by Section 17's trust checklist)."""
    by_page: dict[str, dict] = {}
    for p in l0_pages:
        text = p.get("content") or p.get("markdown") or ""
        emails = sorted(set(_EMAIL_RE.findall(text)))
        phones = sorted(set(m.strip() for m in _PHONE_RE.findall(text) if len(m.strip()) >= 7))
        if emails or phones:
            by_page[p["url"]] = {"emails": emails, "phones": phones}
    return by_page


def brand_aliases(entities: list[dict], org_type: str = "Organization") -> dict:
    """Distinct surface forms of the org entity from the entity store —
    flagged as an identity ambiguity when multiple forms exist with no
    schema (sameAs) tying them together."""
    orgs = [e for e in entities if e.get("type", "").lower() == org_type.lower()]
    names = sorted({e["name"] for e in orgs})
    return {"forms": names, "n_forms": len(names),
           "ambiguous": len(names) > 1}


def hreflang_and_lang(l0_pages: list[dict]) -> dict:
    langs = sorted({p.get("html_lang") for p in l0_pages if p.get("html_lang")})
    hreflangs = sorted({h for p in l0_pages for h in (p.get("hreflang") or [])})
    return {"html_lang_values": langs, "hreflang_set": hreflangs}


def build_identity_report(l0_pages: list[dict], entities: list[dict],
                          calibration: dict | None = None) -> dict:
    blocks = extract_identity_blocks(l0_pages)
    contacts = contact_signals(l0_pages)
    aliases = brand_aliases(entities)
    lang = hreflang_and_lang(l0_pages)

    # sameAs declared by the site itself only — never fetched
    same_as = sorted({url for b in blocks for url in (b["block"].get("sameAs") or [])
                     if isinstance(b["block"].get("sameAs"), list)})

    # contact consistency: same email/phone set on every page that has one
    contact_sets = [tuple(sorted(v["emails"] + v["phones"])) for v in contacts.values()]
    contact_consistent = len(set(contact_sets)) <= 1 if contact_sets else None

    conflicts = []
    if aliases["ambiguous"] and not same_as:
        conflicts.append(
            f"{aliases['n_forms']} distinct brand names used "
            f"({', '.join(aliases['forms'])}) with no schema (sameAs) connecting them")
    if contact_consistent is False:
        conflicts.append("contact details (email/phone) differ across pages")

    if calibration:
        brand_awareness = {"known": calibration.get("brand_known"), "label": "observed"}
    else:
        brand_awareness = {"known": "unknown — run a calibration panel", "label": "observed"}

    return {
        "identity_blocks": blocks,
        "n_identity_blocks": len(blocks),
        "declared_same_as": same_as,
        "brand_aliases": aliases,
        "contact_consistent": contact_consistent,
        "contacts_by_page": contacts,
        "language": lang,
        "conflicts": conflicts,
        "ai_brand_awareness": brand_awareness,
    }
