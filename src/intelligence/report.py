"""Phase 12 — assembles all 19 sections into intelligence.json and renders
intelligence.html (self-contained, sidebar nav, client-side search, no
external CDN — opens straight from disk, same convention as Phase 10's
report.html)."""
from __future__ import annotations

import html as _html
import json as _json
from pathlib import Path

from . import (backlinks, chunks_intel, citation_readiness, content_intel,
              crawl_intel, crawlability, entities as entities_mod, keywords,
              kg_quality, query_sim, reasoning_path, recommendations,
              relationships, topics_intel, trust)
from .. import claim_intel, identity as identity_mod


def _claims_and_evidence_by_chunk(kg) -> tuple[dict, dict, list[dict]]:
    """Derive claims-by-chunk and evidence-by-chunk straight from the KG —
    this data isn't retained on run()'s in-memory ctx after indexing."""
    res = kg._exec(
        "MATCH (c:Chunk)-[r:SupportsClaim]->(cl:Claim) "
        "RETURN c.id, cl.id, cl.text, r.entailment_label, r.entailment_confidence")
    claims_by_chunk: dict[str, list[dict]] = {}
    evidence_by_chunk: dict[str, list[dict]] = {}
    all_claims: dict[str, dict] = {}
    while res.has_next():
        cid, clid, text, label, conf = res.get_next()
        claims_by_chunk.setdefault(cid, []).append({"id": clid, "text": text})
        evidence_by_chunk.setdefault(cid, []).append(
            {"claim_id": clid, "nli_label": label, "nli_conf": conf})
        all_claims[clid] = {"id": clid, "text": text, "source_chunk_id": cid}
    return claims_by_chunk, evidence_by_chunk, list(all_claims.values())


def _topic_by_chunk(kg) -> dict[str, str]:
    res = kg._exec("MATCH (c:Chunk)-[:BelongsToTopic]->(t:Topic) RETURN c.id, t.label")
    out = {}
    while res.has_next():
        cid, label = res.get_next()
        out[cid] = label
    return out


def build_intelligence_report(
    ctx: dict, traces: list, explanations: list[dict], struct: list[dict],
    fx: dict | None, query_weights: dict, skipped_queries: list[dict] | None,
    calibration: dict | None = None,
) -> dict:
    """Assembles all 19 sections from data run()'s ctx/traces/struct already
    hold plus direct KG queries for what indexing doesn't retain in memory."""
    kg = ctx["kg"]
    l0_pages = ctx["l0_pages"]
    chunks = ctx["chunks"]
    page_signals = ctx["page_signals"]
    struct_by_chunk = {s["chunk_id"]: s for s in struct}
    manifest = (fx or {}).get("manifest", [])

    ents = entities_mod.entity_stats(kg, l0_pages)
    claims_by_chunk, evidence_by_chunk, all_claims = _claims_and_evidence_by_chunk(kg)
    topic_by_chunk = _topic_by_chunk(kg)

    # Section 1 — Identity
    identity_report = identity_mod.build_identity_report(l0_pages, ents["all"], calibration)

    # Section 2 — already computed (ents)

    # Section 3 — Relationships
    triplets = relationships.relationship_triplets(kg)
    rel_findings = relationships.relationship_findings(triplets)
    top_entities_svg = relationships.render_top_entities_svg(kg)

    # Section 4 — Topics
    topics = topics_intel.topic_stats(kg, page_signals)

    # Section 5 — Chunks
    chunk_dossiers = chunks_intel.chunk_dossiers(
        chunks, ents["all"], claims_by_chunk, evidence_by_chunk,
        struct_by_chunk, topic_by_chunk, traces)

    # Section 6 — Keywords (GSC/GKP import is opt-in; empty without files)
    kw_report = {"gsc_imported": False, "gkp_imported": False,
                "note": "import via run_audit.py import-queries --gkp / a GSC export "
                       "to populate this section", "excluded_metrics": keywords.EXCLUDED_METRICS}

    # Section 7 — Backlinks
    backlink_report = backlinks.backlink_report(kg, l0_pages)

    # Section 8 — Crawl
    crawl_report = crawl_intel.crawl_manifest(
        ctx["site_report"].get("sitemap", {}), l0_pages, [])

    # Section 9 — Crawlability
    crawlability_report = crawlability.crawlability_report(
        ctx.get("invisibility") or {}, l0_pages, struct_by_chunk, manifest)

    # Section 10 — KG Quality
    kgq = kg_quality.kg_quality_metrics(kg, ents["all"], len(all_claims))

    # Section 11/12 — Claim Intelligence (site-wide, reuses the NLI model already loaded)
    from .. import providers
    from ..evidence import _score_pairs
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    chunk_embeddings = dict(zip((c["chunk_id"] for c in chunks), ctx["embeddings"]))
    claim_texts = [c["text"] for c in all_claims]
    claim_vecs = providers.embed_texts([t[:512] for t in claim_texts]) if claim_texts else []
    claim_embeddings = {c["id"]: v for c, v in zip(all_claims, claim_vecs)}
    claim_records = claim_intel.cross_page_evidence(
        all_claims, chunks_by_id, claim_embeddings, chunk_embeddings, _score_pairs)
    claim_ranking = claim_intel.rank_claims(claim_records)

    # Section 13 — Query Simulation (full)
    full_queries = query_sim.full_query_dossiers(traces, skipped_queries)

    # Section 14 — Reasoning Path
    narratives = reasoning_path.render_all(traces)

    # Section 16 — Content Intelligence
    content_rows = []
    for p in l0_pages:
        page_chunks = [c for c in chunks if c.get("url") == p["url"]]
        n_claims = sum(len(claims_by_chunk.get(c["chunk_id"], [])) for c in page_chunks)
        n_entities = sum(1 for e in ents["all"] if p["url"] in {m["url"] for m in e["mentions"]})
        scores = [struct_by_chunk.get(c["chunk_id"], {}).get("structure_score")
                 for c in page_chunks if c["chunk_id"] in struct_by_chunk]
        mean_struct = round(sum(scores) / len(scores), 4) if scores else None
        content_rows.append(content_intel.page_content_intel(
            p["url"], p.get("content", ""), n_claims, n_entities, mean_struct,
            unsupported_rate=None, sub_intents_covered=0, sub_intents_total=0,
            freshness=page_signals.get(p["url"], {}).get("temporal_freshness")))

    # Section 17 — Trust
    contacts = identity_mod.contact_signals(l0_pages)
    org_schema_complete = bool(identity_report["identity_blocks"])
    trust_report = trust.trust_checklist(l0_pages, ents["all"], contacts, org_schema_complete)

    # Section 18 — Citation Readiness
    inv = ctx.get("invisibility") or {}
    invis_by_url = {wp["url"]: wp["mean_invisible_ratio"] for wp in inv.get("worst_pages", [])}
    citation_rows = citation_readiness.citation_readiness_table([
        {"url": p["url"],
         "structure_score": next((s.get("structure_score") for cid, s in struct_by_chunk.items()
                                  if any(c["chunk_id"] == cid and c["url"] == p["url"] for c in chunks)), None),
         "evidence_strength": None, "invisible_ratio": invis_by_url.get(p["url"]),
         "freshness": page_signals.get(p["url"], {}).get("temporal_freshness"),
         "authority": page_signals.get(p["url"], {}).get("source_authority"),
         "simulated_retrieval_success": any(p["url"] in {c.get("page_url") for c in t.citations}
                                           for t in traces),
         }
        for p in l0_pages])

    # Section 19 — Recommendations
    all_recs = [r for e in explanations for r in e.get("recommendations", [])]
    causal_chains = recommendations.build_causal_chains(all_recs)

    return {
        "url": ctx["site_report"]["url"],
        "section_1_identity": identity_report,
        "section_2_entities": ents,
        "section_3_relationships": {"triplets": triplets, "findings": rel_findings,
                                    "top_entities_svg": top_entities_svg},
        "section_4_topics": topics,
        "section_5_chunks": chunk_dossiers,
        "section_6_keywords": kw_report,
        "section_7_backlinks": backlink_report,
        "section_8_crawl": crawl_report,
        "section_9_crawlability": crawlability_report,
        "section_10_kg_quality": kgq,
        "section_11_12_claims": {"records": claim_records, "ranking": claim_ranking},
        "section_13_query_simulation": full_queries,
        "section_14_reasoning_paths": narratives,
        "section_15_competitor": {"note": "requires an explicit competitor URL "
                                         "via the competitor CLI subcommand — not run this session"},
        "section_16_content": content_rows,
        "section_17_trust": trust_report,
        "section_18_citation_readiness": citation_rows,
        "section_19_recommendations": causal_chains,
    }
