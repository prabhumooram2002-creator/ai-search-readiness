#!/usr/bin/env python
"""run_audit.py — plain synchronous, fully-traced audit pipeline (CLAUDE.md).

Layers 1->3, fully local, ZERO API keys:

  index:  crawl -> chunk -> BGE-M3 embed -> Chroma store -> GLiNER entities ->
          GLiREL relations -> qwen claims -> NLI evidence -> HDBSCAN topics ->
          heuristic signals -> Kùzu knowledge graph
  query:  Layer 2 simulator (intent -> expansion -> hybrid retrieve -> rerank
          -> graph traversal -> evidence validation -> contradiction ->
          confidence -> synthesis -> citations), one Trace per query
  report: Layer 3 explainability (pure introspection over the Trace)

Queries are never hardcoded — only --queries (file) or --query (repeatable).

Usage:
    python run_audit.py --url https://example.com --queries queries.txt
    python run_audit.py --url https://example.com --query "..." --max-pages 8
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from src.core.logging import get_logger
from src.trace import Trace
from src import providers

logger = get_logger("run_audit")


def load_queries(path: str | None, inline: list[str] | None) -> list[str]:
    queries: list[str] = list(inline or [])
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Queries file not found: {path}")
        suffix = p.suffix.lower()
        if suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                queries += [q if isinstance(q, str) else q.get("query", "") for q in data]
            elif isinstance(data, dict) and "queries" in data:
                queries += list(data["queries"])
        elif suffix == ".csv":
            import csv
            with p.open(newline="", encoding="utf-8") as f:
                for row in csv.reader(f):
                    if row and row[0].strip():
                        queries += [row[0].strip()]
        else:
            queries += [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    queries = [q for q in (q.strip() for q in queries) if q]
    if not queries:
        raise ValueError("No queries provided. Pass --queries <file> or one or "
                         "more --query. There are no hardcoded brand defaults.")
    return queries


def _unload_extraction_models() -> None:
    """Free GLiNER/GLiREL/spaCy before Layer 2 — the full local model set does
    not fit in RAM at once on this class of machine (see BUILD_LOG OOM)."""
    from src import ner, relations
    ner._gliner_model = None
    ner._spacy_nlp = None
    relations._glirel_model = None
    gc.collect()


def build_index(url: str, max_pages: int | None, fresh: bool,
                claims_cap: int, kg_path: str | None = None) -> dict:
    """Layer 1: crawl -> ... -> knowledge graph. Returns site-level context."""
    from src.crawl import crawl_site_sync
    from src.chunk import chunk_pages
    from src.embed import embed_chunks_sync
    from src.vector import VectorStore
    from src.ner import extract_entities, crosscheck_with_spacy
    from src.relations import extract_relations
    from src.claims import extract_claims
    from src.evidence import extract_evidence
    from src.topics import cluster_topics
    from src.signals import score_pages
    from src.kg import KnowledgeGraph

    if max_pages is not None:
        from src.core import config as _cfg
        _cfg.CFG.setdefault("crawl", {})["max_pages"] = max_pages

    from src.layer0 import run_layer0, fetch_sitemap

    site_trace = Trace(query="__site_index__")

    # LAYER 0: sitemap-FIRST seeding — sitemap URLs join the crawl queue
    sitemap = fetch_sitemap(url)
    logger.info(f"[index] crawling {url} "
                f"(sitemap seeded: {len(sitemap)} URLs)")
    crawl_results = crawl_site_sync(url, seed_urls=[s["url"] for s in sitemap])
    ok = [r for r in crawl_results if getattr(r, "success", False)]
    if not ok:
        raise RuntimeError(f"No pages could be crawled from {url}")
    logger.info(f"[index] {len(ok)}/{len(crawl_results)} pages crawled")

    # LAYER 0 enrichment: JSON-LD/headings/links, 4-bot re-fetch ->
    # access_gaps, robots cross-check, PDFs
    l0 = run_layer0(url, crawl_results, trace=site_trace, sitemap=sitemap)

    chunk_objs = chunk_pages(crawl_results)
    chunks = [{"content": c.content, "chunk_id": c.chunk_id, "url": c.url}
              for c in chunk_objs]
    logger.info(f"[index] {len(chunks)} chunks")

    # step 9 — embeddings: Chroma + (below) Kùzu Chunk nodes
    vs = VectorStore()
    if fresh:
        vs.reset()
    embeddings = embed_chunks_sync([c["content"] for c in chunks], use_cache=True)
    vs.add_chunks(chunk_objs, embeddings)
    chroma_parity = vs.count() == len(chunks)
    logger.info(f"[index] vectors: chroma={vs.count()} chunks={len(chunks)} "
                f"parity={'OK' if chroma_parity else 'MISMATCH'}")

    # steps 3-4 — GLiNER entities + GLiREL relations (local zero-shot)
    entities = extract_entities(chunks, trace=site_trace)
    xcheck = crosscheck_with_spacy(chunks[:20], entities)
    relationships = extract_relations(chunks, entities, trace=site_trace)

    # steps 5-6 — claims (local LLM; capped for CPU runtime) + NLI evidence
    claim_chunks = chunks[:claims_cap]
    if len(chunks) > claims_cap:
        logger.warning(f"[index] claims capped to first {claims_cap}/{len(chunks)} "
                       f"chunks (CPU runtime); raise --claims-cap to widen")
    claims = extract_claims(claim_chunks, trace=site_trace)
    evidence = extract_evidence(claim_chunks, claims, trace=site_trace)

    # step 7 — topics
    topics = cluster_topics(chunks, trace=site_trace)

    # steps 11-14 — heuristic signals, now fed real Layer-0 enrichment
    # (JSON-LD presence, anchored internal links)
    inbound: dict[str, int] = {}
    for p in l0["pages"]:
        for link in p["internal_links"]:
            inbound[link["dest_url"]] = inbound.get(link["dest_url"], 0) + 1
    pages = [{"url": p["url"], "title": p["title"], "content": p["content"],
              "schema_jsonld": p["schema_jsonld"],
              "internal_inbound": inbound.get(p["url"], 0),
              "internal_links": p["internal_links"]}
             for p in l0["pages"]]
    page_signals = score_pages(pages)

    # step 8 (+9) — knowledge graph with embeddings on Chunk nodes
    kg = KnowledgeGraph(kg_path)
    kg_stats = kg.build(chunks, entities, relationships, claims, evidence,
                        topics, pages=pages, embeddings=embeddings,
                        trace=site_trace)

    # Phase 2 — capture baseline state so the next run can be incremental
    from src.incremental import IncrementalState, baseline_capture
    _state = IncrementalState()
    baseline_capture(_state, crawl_results, chunk_objs, l0_pages=l0["pages"])
    _state.close()

    _unload_extraction_models()

    return {
        "site_report": {
            "url": url, "pages_crawled": len(ok), "pages_total": len(crawl_results),
            "chunks": len(chunks), "vector_count": vs.count(),
            "chroma_kuzu_parity": chroma_parity,
            "embed_dim": len(embeddings[0]) if embeddings else None,
            "n_entities": len(entities), "n_relationships": len(relationships),
            "n_claims": len(claims), "n_topics": len(topics),
            "ner_crosscheck_mean_disagreement": xcheck["mean_disagreement"],
            "kg": kg_stats,
            "sitemap": l0["sitemap_stats"], "n_pdfs": len(l0["pdfs"]),
            "robots_conflicts": sum(len(p["robots_conflict"]) for p in l0["pages"]),
        },
        "chunks": chunks, "embeddings": embeddings, "kg": kg,
        "page_signals": page_signals, "site_trace": site_trace,
        "l0_pages": l0["pages"],
    }


def build_index_incremental(url: str, claims_cap: int) -> dict:
    """Phase 2: incremental index update (4-level skip chain), then rebuild
    the query context from stored state. Falls back to full when no baseline."""
    import json as _json
    from src.incremental import IncrementalState, incremental_update
    from src.embed import embed_chunks_sync
    from src.vector import VectorStore
    from src.ner import extract_entities
    from src.relations import extract_relations
    from src.claims import extract_claims
    from src.evidence import extract_evidence
    from src.topics import cluster_topics
    from src.signals import score_pages
    from src.kg import KnowledgeGraph
    from src.layer0 import invisibility_score  # noqa: F401 (used by run())

    state = IncrementalState()
    if not state.has_baseline():
        logger.warning("[incremental] no baseline — running a FULL build instead")
        state.close()
        return build_index(url, None, fresh=False, claims_cap=claims_cap)

    site_trace = Trace(query="__site_index__")
    kg = KnowledgeGraph()
    vs = VectorStore()

    def process(new_chunk_objs):
        cd = [{"chunk_id": c.chunk_id, "content": c.content, "url": c.url}
              for c in new_chunk_objs]
        embs = embed_chunks_sync([c.content for c in new_chunk_objs], use_cache=True)
        vs.add_chunks(new_chunk_objs, embs)
        ents = extract_entities(cd, trace=site_trace)
        rels = extract_relations(cd, ents, trace=site_trace)
        cls = extract_claims(cd[:claims_cap], trace=site_trace)
        ev = extract_evidence(cd[:claims_cap], cls, trace=site_trace)
        kg.build(cd, ents, rels, cls, ev, embeddings=embs, trace=site_trace)

    stats = incremental_update(url, state, kg, vs, process, trace=site_trace)

    # rebuild query context from state
    chunks = state.active_chunks()
    embeddings = embed_chunks_sync([c["content"] for c in chunks], use_cache=True)
    topics = cluster_topics(chunks, trace=site_trace) if stats["rerun_topics"] else []

    pages, inbound = [], {}
    for row in state.all_pages():
        links = _json.loads(row["internal_links_json"] or "[]")
        for l in links:
            inbound[l.get("dest_url", "")] = inbound.get(l.get("dest_url", ""), 0) + 1
        pages.append({
            "url": row["url"], "title": row["title"],
            "content": " ".join(c["content"] for c in chunks
                                if c["url"] == row["url"])[:8000],
            "internal_links": links,
            "access_gaps": _json.loads(row["access_gaps_json"] or "[]"),
        })
    for p in pages:
        p["internal_inbound"] = inbound.get(p["url"], 0)
    page_signals = score_pages(pages)
    _unload_extraction_models()

    return {
        "site_report": {
            "url": url, "mode": "incremental", **stats,
            "chunks": len(chunks), "vector_count": vs.count(),
            "chroma_kuzu_parity": vs.count() == len(chunks),
            "embed_dim": len(embeddings[0]) if embeddings else None,
            "n_topics": len(topics), "kg": kg.verify(),
        },
        "chunks": chunks, "embeddings": embeddings, "kg": kg,
        "page_signals": page_signals, "site_trace": site_trace,
        "l0_pages": pages,
    }


def write_reports(ctx: dict, traces: list[Trace], explanations: list[dict],
                  out_dir: Path) -> Path:
    from src.explain import render_markdown
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "invisibility": ctx.get("invisibility"),   # headline metric, measured
        "site": ctx["site_report"],
        "provider": providers.provider_info(),
        "site_trace": ctx["site_trace"].to_dict(),
        "queries": [t.to_dict() for t in traces],
        "explanations": explanations,
    }
    json_path = out_dir / "audit_report.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str),
                         encoding="utf-8")

    sr = ctx["site_report"]
    inv = ctx.get("invisibility") or {}
    lines = [f"# AI-Search Readiness Audit — {sr['url']}", ""]
    if inv:
        lines += [f"## {inv['headline']}",
                  f"(AI Invisibility Score: {inv['site_invisibility']:.0%} — "
                  f"{inv['label']}; weighting: {inv['weighting']})", ""]
        for wp in inv["worst_pages"][:3]:
            if wp["mean_invisible_ratio"] > 0:
                lines.append(f"- {wp['url']}: {wp['mean_invisible_ratio']:.0%} invisible"
                             f" — e.g. missing: "
                             f"{(wp['worst_missing_blocks'] or ['—'])[0][:90]!r}")
        lines.append("")
    lines += [
             f"- Mode: {sr.get('mode', 'full')}  "
             f"Pages: {sr.get('pages_crawled', sr.get('pages_total', '?'))}  "
             f"Chunks: {sr['chunks']}  Embed dim: {sr['embed_dim']}",
             f"- Entities: {sr.get('n_entities', '-')}  "
             f"Relations: {sr.get('n_relationships', '-')}  "
             f"Claims: {sr.get('n_claims', '-')}  Topics: {sr.get('n_topics', 0)}",
             f"- KG: {sr['kg']['nodes']} | orphan_chunks={sr['kg']['orphan_chunks']} "
             f"claims_without_support={sr['kg']['claims_without_support']}",
             f"- Provider: {providers.provider_info()['llm_provider_effective']}"
             f" / {providers.provider_info()['embed_provider_effective']} (zero keys required)",
             ""]
    for rows in explanations:
        lines.append(render_markdown(rows))
        lines.append("")
    md_path = out_dir / "audit_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path


def run(url: str, queries: list[str], out_dir: Path, max_pages: int | None,
        fresh: bool, claims_cap: int = 25, mode: str = "full",
        emit_fixes: bool = False) -> dict:
    from src.simulator import run_query
    from src.explain import explain_query

    from src.layer0 import invisibility_score

    if mode == "incremental":
        ctx = build_index_incremental(url, claims_cap)
    else:
        ctx = build_index(url, max_pages, fresh, claims_cap)
    traces, explanations = [], []
    for q in queries:
        t = run_query(q, ctx["chunks"], ctx["embeddings"], kg=ctx["kg"],
                      page_signals=ctx["page_signals"])
        traces.append(t)
        explanations.append(explain_query(t, kg=ctx["kg"]))
    # AI Invisibility Score (measured) — weighted by PageRank and by the pages
    # that actually won retrievals for this query set
    winner_urls = {c["page_url"] for t in traces for c in t.citations if c["page_url"]}
    ctx["invisibility"] = invisibility_score(ctx["l0_pages"], winner_urls)

    # Phase 3b structural scores + Phase 5 snapshot persistence
    from src.structure import score_chunks
    from src.snapshots import build_snapshot, save_snapshot
    pages_with_schema = {p["url"] for p in ctx["l0_pages"]
                         if p.get("schema_jsonld")}
    struct = score_chunks(
        [{"id": c["chunk_id"], "text": c["content"], "page_url": c["url"],
          "heading_path": c.get("heading_path", [])} for c in ctx["chunks"]],
        pages_with_schema=pages_with_schema)
    snap = build_snapshot(url, ctx["invisibility"], struct,
                          ctx["page_signals"], traces)
    run_id = save_snapshot(snap)
    ctx["snapshot_run_id"] = run_id

    # Phase 8 — emit ready-to-use fix artifacts (draft, human-review) if asked
    if emit_fixes:
        from src.fixes import generate_fixes
        pages_no_schema = [p for p in ctx["l0_pages"] if not p.get("schema_jsonld")]
        dead_ends = [{"sub_query": de.get("sub_query", de) if isinstance(de, dict) else de}
                     for t in traces for de in t.retrieval_dead_ends][:5]
        flagged = [{"chunk_id": s["chunk_id"],
                    "text": next((c["content"] for c in ctx["chunks"]
                                  if c["chunk_id"] == s["chunk_id"]), ""),
                    "flags": s["flags"]} for s in struct if s["flags"]][:5]
        site = {"name": url, "description": "",
                "pages": [{"url": p["url"], "title": p.get("title", ""),
                           "pagerank": 0} for p in ctx["l0_pages"]],
                "topics": []}
        fx = generate_fixes(run_id, pages_without_schema=pages_no_schema[:5],
                            site=site, dead_ends=dead_ends, structure_flagged=flagged,
                            trace=ctx["site_trace"])
        ctx["fixes_dir"] = fx["run_dir"]
        logger.info(f"[fixes] {len(fx['manifest'])} artifacts -> {fx['run_dir']}")

    json_path = write_reports(ctx, traces, explanations, out_dir)
    logger.info(f"[report] wrote {json_path}")
    return {"site_report": ctx["site_report"], "traces": traces,
            "explanations": explanations, "report_path": str(json_path)}


def cmd_import_queries(argv: list[str]) -> None:
    """`run_audit.py import-queries --gkp <csv> [--paa <file>] --out <json>`."""
    from src.demand import build_query_set
    ap = argparse.ArgumentParser(prog="run_audit.py import-queries",
                                 description="Ingest real query demand (Phase 4).")
    ap.add_argument("--gkp", help="Google Keyword Planner CSV/TSV export")
    ap.add_argument("--paa", help="People-Also-Ask paste file (one per line)")
    ap.add_argument("--max-variants", type=int, default=3)
    ap.add_argument("--out", default="data/queries_imported.json")
    args = ap.parse_args(argv)
    if not args.gkp and not args.paa:
        ap.error("provide --gkp and/or --paa")
    queries = build_query_set(args.gkp, args.paa, args.max_variants)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(queries, indent=2), encoding="utf-8")
    seeds = len({q["seed"] for q in queries if q.get("seed")})
    print(f"Imported {len(queries)} queries from {seeds} seeds "
          f"(ordered by volume weight) -> {args.out}")


def cmd_diff(argv: list[str]) -> int:
    """`run_audit.py diff [--from N --to M]` — regressions since a prior run.
    Non-zero exit when a regression exceeds threshold (cron/CI usable)."""
    from src.snapshots import (list_snapshots, load_snapshot, diff_snapshots,
                               render_diff)
    ap = argparse.ArgumentParser(prog="run_audit.py diff")
    ap.add_argument("--from", dest="frm", type=int)
    ap.add_argument("--to", dest="to", type=int)
    args = ap.parse_args(argv)
    runs = list_snapshots()
    if len(runs) < 2 and not (args.frm and args.to):
        print("Need at least two snapshots to diff.")
        return 2
    frm = args.frm or runs[-2]
    to = args.to or runs[-1]
    d = diff_snapshots(load_snapshot(frm), load_snapshot(to))
    print(render_diff(d))
    return 1 if d["ci_status"] == "fail" else 0


def cmd_calibrate(argv: list[str]) -> int:
    """`run_audit.py calibrate --run N [--template]` — compare the simulator's
    winning pages against real ChatGPT/Perplexity/Gemini citations (manual
    panel). Observed numbers are reported separately from simulated."""
    from src.calibration import (generate_templates, load_run, compute_agreement)
    from src.snapshots import list_snapshots, load_snapshot
    ap = argparse.ArgumentParser(prog="run_audit.py calibrate")
    ap.add_argument("--run", type=int, help="snapshot run id (default: latest)")
    ap.add_argument("--template", action="store_true",
                    help="write blank paste-templates for the top-20 queries")
    args = ap.parse_args(argv)
    runs = list_snapshots()
    if not runs:
        print("No snapshots yet — run an audit first.")
        return 2
    run_id = args.run or runs[-1]
    snap = load_snapshot(run_id)
    if args.template:
        qs = [{"query": q, "weight": v.get("weight", 0)}
              for q, v in snap["queries"].items()]
        root = generate_templates(run_id, qs)
        print(f"Wrote calibration templates -> {root}\n"
              f"Fill each engine's answer + cited URLs, then run "
              f"`run_audit.py calibrate --run {run_id}`.")
        return 0
    calibration = load_run(run_id)
    if not calibration:
        print(f"No filled calibration files for run {run_id}. "
              f"Run with --template first.")
        return 2
    wins = {q: v.get("winning_pages", []) for q, v in snap["queries"].items()}
    ag = compute_agreement(calibration, wins)
    print(f"OBSERVED citation agreement (run {run_id}) — reported separately "
          f"from simulated confidence:")
    print(f"  overall: {ag['overall_agreement']}  per-engine: "
          f"{ag['per_engine_agreement']}")
    print(f"  {ag['note']}")
    return 0


def cmd_whatif(argv: list[str]) -> int:
    """`run_audit.py whatif --draft <md|txt|docx> --query "..." [...]` — predict
    a draft's coverage impact against a non-persisting overlay index."""
    from src.whatif import whatif
    from src.incremental import IncrementalState
    from src.embed import embed_chunks_sync
    ap = argparse.ArgumentParser(prog="run_audit.py whatif")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--query", action="append", required=True,
                    help="target query (repeatable)")
    args = ap.parse_args(argv)

    state = IncrementalState()
    base_chunks = state.active_chunks()
    state.close()
    if not base_chunks:
        print("No baseline index — run an audit first."); return 2
    base_embs = embed_chunks_sync([c["content"] for c in base_chunks], use_cache=True)
    persist = [str(BASE_DIR / "data" / "chromadb"),
               str(BASE_DIR / "data" / "kg.kuzu")]
    res = whatif(args.draft, args.query, base_chunks, base_embs, persist_paths=persist)
    print(f"What-if for {args.draft} ({res['n_draft_chunks']} draft chunks, "
          f"mean structure {res['mean_structure_score']}):")
    for pq in res["per_query"]:
        print(f"  {pq['query'][:50]!r}: coverage {pq['coverage_before']} -> "
              f"{pq['coverage_after']} (delta {pq['coverage_delta']:+}, "
              f"+{pq['newly_covered']} sub-intents)")
    print(f"  persistent stores unchanged: {res['stores_unchanged']}")
    if res["structure_flags"]:
        print(f"  draft structure flags: "
              f"{sorted({f['flag'] for f in res['structure_flags']})}")
    return 0


from src.core.config import BASE_DIR  # noqa: E402 (used by cmd_whatif)


def main() -> None:
    # Lightweight subcommand dispatch (keeps `run_audit.py --url ...` working;
    # Phase 4+ add import-queries / diff / whatif / calibrate / import-logs).
    if len(sys.argv) > 1 and sys.argv[1] == "import-queries":
        cmd_import_queries(sys.argv[2:])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "diff":
        raise SystemExit(cmd_diff(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        raise SystemExit(cmd_calibrate(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "whatif":
        raise SystemExit(cmd_whatif(sys.argv[2:]))

    ap = argparse.ArgumentParser(
        description="Traced, local-first AI-search readiness audit (Layers 1-3).")
    ap.add_argument("--url", required=True, help="Root URL to audit")
    ap.add_argument("--queries", help="Path to a queries file (.txt/.csv/.json)")
    ap.add_argument("--query", action="append", help="Inline query (repeatable)")
    ap.add_argument("--max-pages", type=int, default=None, help="Cap pages crawled")
    ap.add_argument("--claims-cap", type=int, default=25,
                    help="Max chunks to run LLM claim extraction on (CPU cost)")
    ap.add_argument("--out", default="data/reports", help="Report output directory")
    ap.add_argument("--no-fresh", action="store_true",
                    help="Do not reset the vector store before indexing")
    ap.add_argument("--incremental", action="store_true",
                    help="Incremental update (default once a baseline exists)")
    ap.add_argument("--full", action="store_true",
                    help="Force a full rebuild even when a baseline exists")
    ap.add_argument("--fixes", action="store_true",
                    help="Emit ready-to-use fix artifacts (draft) to fixes/<run>/")
    args = ap.parse_args()

    # Phase 2 mode resolution: incremental is the default once a baseline
    # exists; --full always forces a rebuild.
    from src.incremental import IncrementalState
    _s = IncrementalState()
    has_baseline = _s.has_baseline()
    _s.close()
    if args.full:
        mode = "full"
    elif args.incremental or has_baseline:
        mode = "incremental"
    else:
        mode = "full"

    queries = load_queries(args.queries, args.query)
    logger.info(f"Auditing {args.url} with {len(queries)} queries mode={mode} "
                f"(LLM={providers.LLM_PROVIDER}:{providers.LLM_MODEL}, "
                f"EMBED={providers.EMBED_PROVIDER}:{providers.EMBED_MODEL})")
    result = run(args.url, queries, Path(args.out), args.max_pages,
                 fresh=not args.no_fresh, claims_cap=args.claims_cap, mode=mode,
                 emit_fixes=args.fixes)
    print(f"\nReport written to {result['report_path']}")


if __name__ == "__main__":
    main()
