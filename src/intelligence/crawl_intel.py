"""Section 8 — Crawl Intelligence [EXISTS + Phase 1c manifest]"""
from __future__ import annotations

from collections import Counter


def crawl_manifest(sitemap_stats: dict, l0_pages: list[dict], crawl_results: list) -> dict:
    """Coverage manifest: discovered vs crawled vs skipped; depth distribution;
    dead pages; duplicate URLs (near-dup markdown hash); sitemap coverage."""
    depth_dist = Counter(getattr(r, "depth", None) for r in crawl_results
                        if getattr(r, "success", False))
    dead = [getattr(r, "url", None) for r in crawl_results
           if getattr(r, "status", 200) == 404]
    hash_seen: dict[str, list[str]] = {}
    for p in l0_pages:
        h = p.get("markdown_hash") or p.get("content_hash")
        if h:
            hash_seen.setdefault(h, []).append(p["url"])
    duplicates = [{"hash": h, "urls": urls} for h, urls in hash_seen.items() if len(urls) > 1]

    return {
        "sitemap_urls": sitemap_stats.get("sitemap_urls", 0),
        "sitemap_captured": sitemap_stats.get("captured", 0),
        "sitemap_capture_rate": sitemap_stats.get("capture_rate"),
        "pages_crawled": len(l0_pages),
        "depth_distribution": dict(depth_dist),
        "dead_pages": [u for u in dead if u],
        "duplicate_url_groups": duplicates,
        "canonical_conflicts": {"available": False,
                               "reason": "canonical tag extraction not stored per-page by "
                                        "the current crawler — needs a Layer 0 field addition"},
    }
