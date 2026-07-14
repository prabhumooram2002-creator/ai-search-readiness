"""Section 17 — Trust Intelligence [RULE-BASED checks, heuristic]

EEAT-style checklist. Every row carries its evidence location or "missing"
— never a bare pass/fail with no trace.
"""
from __future__ import annotations

import re

_BYLINE = re.compile(r"\bby\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)+|\bauthor\b", re.I)
_POLICY_HINTS = {"privacy": "privacy policy", "refund": "refund/return policy",
                 "return": "refund/return policy", "about": "about page",
                 "contact": "contact page", "terms": "terms of service"}

TRUST_FORMULA = "heuristic: mean of 7 EEAT checklist items, each 0 or 1"


def _policy_pages_present(l0_pages: list[dict]) -> dict:
    found: dict[str, str] = {}
    for p in l0_pages:
        url_l = p["url"].lower()
        for hint, label in _POLICY_HINTS.items():
            if hint in url_l and label not in found:
                found[label] = p["url"]
    return found


def _review_schema_present(l0_pages: list[dict]) -> list[str]:
    urls = []
    for p in l0_pages:
        for block in p.get("schema_jsonld", []) or []:
            t = (block.get("@type") if isinstance(block, dict) else None) or ""
            if "review" in str(t).lower() or "aggregaterating" in str(t).lower():
                urls.append(p["url"])
    return urls


def _bylines_present(l0_pages: list[dict]) -> list[str]:
    return [p["url"] for p in l0_pages
           if _BYLINE.search((p.get("content") or "")[:4000])]


def trust_checklist(l0_pages: list[dict], entities: list[dict],
                    contacts_by_page: dict, org_schema_complete: bool) -> dict:
    bylines = _bylines_present(l0_pages)
    reviews = _review_schema_present(l0_pages)
    certs_awards = [e for e in entities if e.get("type", "").lower() in ("award", "certification")]
    policy_pages = _policy_pages_present(l0_pages)
    contact_consistent = len({tuple(sorted(v.get("emails", []) + v.get("phones", [])))
                             for v in contacts_by_page.values()}) <= 1

    items = {
        "author_bylines_present": {"pass": bool(bylines), "evidence": bylines[:5] or "missing"},
        "review_rating_schema": {"pass": bool(reviews), "evidence": reviews[:5] or "missing"},
        "certifications_or_awards": {"pass": bool(certs_awards),
                                    "evidence": [e["name"] for e in certs_awards[:5]] or "missing"},
        "contact_consistency": {"pass": contact_consistent,
                               "evidence": "consistent" if contact_consistent else "differs across pages"},
        "policy_pages_present": {"pass": len(policy_pages) >= 2, "evidence": policy_pages or "missing"},
        "org_schema_complete": {"pass": org_schema_complete,
                               "evidence": "present" if org_schema_complete else "missing"},
    }
    score = round(sum(1 for v in items.values() if v["pass"]) / len(items), 4)
    return {"checklist": items, "score": {"value": score, "label": "heuristic",
                                          "formula": TRUST_FORMULA}}
