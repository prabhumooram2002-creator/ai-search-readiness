"""Main CLI -- audit <url>, ask <q>, status, graph, report, reset commands.
   
   Phase 1: Crawl -> Chunk -> Embed -> ChromaDB -> Vector Q&A
   Phase 2: Entity Extraction -> Kuzu Graph
   Phase 3: Hybrid Retrieval + Broken Path Detection
   Phase 4: Coverage Scoring
   Phase 5: HTML Report
"""
import asyncio
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.core.logging import get_logger
from src.crawl import crawl_site_sync, CrawlResult
from src.chunk import chunk_pages
from src.embed import embed_chunks
from src.vector import VectorStore
from src.qa import QAPipeline
from src.graph import get_graph, reset_graph, extract_from_chunks
from src.input_layer import parse_file, segment_text
from src.input_layer.batch_pipeline import audit_file, audit_text, print_batch_summary, run_batch_audit

logger = get_logger(__name__)

def print_banner():
    print(r"""
+--------------------------------------------------------------+
|       AI Search Readiness Platform -- All Phases              |
|  Crawl -> Chunk -> Embed -> Vector Q&A -> Knowledge Graph     |
+--------------------------------------------------------------+
""")


def print_indexing_summary(crawl_results: list, chunks_count: int,
                            entity_count: int = 0, rel_count: int = 0,
                            graph_available: bool = False):
    success = [r for r in crawl_results if r.success]
    start_url = crawl_results[0].url if crawl_results else "N/A"

    graph_line = ""
    if graph_available and entity_count > 0:
        graph_line = (
            f"|  Entities:        {entity_count:>6}                         |\n"
            f"|  Relationships:   {rel_count:>6}                         |"
        )

    print(f"""
+==========================================================+
|                  INDEXING SUMMARY                         |
+==========================================================+
|  Pages crawled:     {len(success):>6} / {len(crawl_results):<6} successful
|  Chunks created:    {chunks_count:>6}
{graph_line if graph_line else "|                                                          |"}
+==========================================================+

[OK]  Phase 1 indexing complete!
{"[OK]  Phase 2 entity extraction complete!" if entity_count > 0 else "[WARN]   Phase 2 skipped (graph not available)"}

Start URL: {start_url}

Commands:
  audit <url>   -- Run full audit (Phase 1 + Phase 2)
  ask <question> -- Ask a question about the indexed site
  status        -- Show indexing stats
  graph         -- Show entity/relationship summary
  report        -- Generate HTML report (Phase 5)
  reset         -- Clear all data and start over
  exit          -- End session
""")


async def run_audit(start_url: str) -> dict:
    """Run Phase 1 + Phase 2 audit pipeline. Returns stats dict."""
    logger.info(f"Starting audit: {start_url}")
    stats = {
        "crawl_results": [], "chunks": [], "entities": [],
        "relationships": [], "graph_available": False
    }

    # -- Phase 1: Crawl -------------------------------------------------
    print(f"\n [Phase 1] Crawling {start_url} ...")
    crawl_results = crawl_site_sync(start_url)
    success = [r for r in crawl_results if r.success]
    if not success:
        print(" No pages could be crawled. Check URL and try again.")
        sys.exit(1)
    print(f"   [OK]  {len(success)} pages crawled")
    stats["crawl_results"] = crawl_results

    # -- Phase 1: Chunk -------------------------------------------------
    print(f"\n[PACK] [Phase 1] Chunking {len(success)} pages ...")
    all_chunks = chunk_pages(crawl_results)
    print(f"   [OK]  {len(all_chunks)} chunks created")
    stats["chunks"] = all_chunks

    # -- Phase 1: Embed -------------------------------------------------
    print(f"\n[BRAIN] [Phase 1] Embedding {len(all_chunks)} chunks (NVIDIA NIM free tier) ...")
    chunk_texts = [c.content for c in all_chunks]
    embeddings = await embed_chunks(chunk_texts, use_cache=True)
    print(f"   [OK]  {len(embeddings)} embeddings generated")
    stats["embeddings"] = embeddings

    # -- Phase 1: Store -------------------------------------------------
    print(f"\n[STORE] [Phase 1] Storing in ChromaDB ...")
    vs = VectorStore()
    vs.add_chunks(all_chunks, embeddings)
    print(f"   [OK]  {vs.count()} total chunks in vector store")
    stats["vector_count"] = vs.count()

    # -- Phase 2: Entity Extraction + Kuzu Graph ------------------------
    graph_available = False
    entity_count = 0
    rel_count = 0

    try:
        print(f"\n[LINK] [Phase 2] Connecting to Kuzu graph ...")
        graph = get_graph()
        graph.verify_connection()
        graph_available = True
        stats["graph_available"] = True
        print(f"   [OK]  Kuzu graph connected")

        print(f"\n[BUILD]  [Phase 2] Extracting entities + relationships ...")
        entities, relationships = await extract_from_chunks(all_chunks, use_cache=True)
        print(f"   [OK]  {len(entities)} entities, {len(relationships)} relationships extracted")

        if entities or relationships:
            print(f"\n [Phase 2] Writing to Kuzu ...")
            graph.upsert_entities(entities)
            graph.upsert_relationships(relationships)

            g_stats = graph.get_stats()
            entity_count = g_stats["nodes"]
            rel_count = g_stats["relationships"]
            print(f"   [OK]  {entity_count} entities, {rel_count} relationships in graph")

            stats["entities"] = entities
            stats["relationships"] = relationships

    except Exception as e:
        print(f"\n[WARN]   [Phase 2] Graph not available: {e}")
        print(f"   Skipping entity extraction -- graph DB may still be initializing.")

    print_indexing_summary(
        crawl_results, len(all_chunks),
        entity_count, rel_count, graph_available
    )

    return stats


def interactive_loop():
    """Interactive Q&A loop after audit completes."""
    vs = VectorStore()
    qa = QAPipeline(vs)

    if vs.count() == 0:
        print(" No data indexed. Run `audit <url>` first.")
        return

    print("\n[CHAT] Ask me anything about the indexed site.\n")

    while True:
        try:
            user_input = input(" You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n Exiting.")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ("exit", "quit", "q"):
            print(" Goodbye!")
            break

        if cmd == "status":
            vs = VectorStore()
            print(f"\n[DATA] Vector store: {vs.count()} chunks\n")
            try:
                graph = get_graph()
                gs = graph.get_stats()
                print(f"[DATA] Graph: {gs['nodes']} entities, {gs['relationships']} relationships\n")
            except Exception as e:
                print(f"[DATA] Graph: not available ({e})\n")
            continue

        if cmd == "graph":
            try:
                graph = get_graph()
                gs = graph.get_stats()
                print(f"\n  Graph Stats:")
                print(f"   Entities:      {gs['nodes']}")
                print(f"   Relationships: {gs['relationships']}")
                print()
            except Exception as e:
                print(f" Graph: {e}\n")
            continue

        if cmd == "reset":
            vs = VectorStore()
            vs.reset()
            try:
                reset_graph()
                print("  Vector store + graph cleared.\n")
            except Exception:
                print("  Vector store cleared (graph not available).\n")
            continue

        if cmd == "report":
            print("[DATA] Generating HTML report ...")
            try:
                from src.report import generate_report
                path = generate_report()
                print(f"   [OK]  Report saved to: {path}\n")
            except Exception as e:
                print(f"   [WARN]   Report generation failed: {e}\n")
            continue

        question = user_input

        # Run Q&A
        result = qa.ask(question)

        print(f"\n Answer:\n{result.answer}\n")
        print(f" Sources:")
        for i, src in enumerate(result.sources[:3], 1):
            print(f"  [{i}] {src.title or src.url} (distance: {src.distance:.4f})")

        if result.graph_path:
            print(f"\n  Graph path: {result.graph_path}")
        if result.missing_relationship:
            print(f"[WARN]   Missing: {result.missing_relationship}")
            print(f" Fix: {result.suggested_fix}")
        if result.verdict:
            print(f"\n Verdict: {result.verdict} (scores: {result.scores})")
        print()


def main():
    print_banner()

    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python -m src.main audit <url>   -- Full audit (Phase 1 + Phase 2)")
        print("  python -m src.main ask <q>      -- Ask a question")
        print("  python -m src.main status        -- Show stats")
        print("  python -m src.main graph         -- Show graph summary")
        print("  python -m src.main report       -- Generate HTML report")
        print("  python -m src.main reset          -- Clear all data")
        sys.exit(1)

    cmd = args[0].lower()

    if cmd == "audit":
        if len(args) < 2:
            print("Usage: audit <url>")
            sys.exit(1)
        url = args[1]
        stats = asyncio.run(run_audit(url))
        interactive_loop()

    elif cmd == "ask":
        vs = VectorStore()
        if vs.count() == 0:
            print(" No data indexed. Run `audit <url>` first.")
            sys.exit(1)
        question = " ".join(args[1:]) if len(args) > 1 else input(" Question: ").strip()
        qa = QAPipeline(vs)
        result = qa.ask(question)
        print(f"\n Answer:\n{result.answer}\n")
        for i, src in enumerate(result.sources[:3], 1):
            print(f"  [{i}] {src.title or src.url}")

    elif cmd == "import":
        if len(args) < 2:
            print("Usage: import <path-to-file.csv|.txt|.json|.xlsx>")
            sys.exit(1)
        file_path = args[1]
        results = asyncio.run(audit_file(file_path))
        print_batch_summary(results)
        print(f"\n[STORE] Run `gap-report` to generate the full HTML report.\n")

    elif cmd == "analyze":
        text = " ".join(args[1:]).strip()
        if not text:
            print("Usage: analyze <pasted text block>")
            sys.exit(1)
        results = asyncio.run(audit_text(text))
        print_batch_summary(results)

    elif cmd == "gap-report":
        from src.report import generate_gap_report
        path = generate_gap_report()
        print(f"[OK]  Gap Report: {path}")

    elif cmd == "status":
        vs = VectorStore()
        print(f"  ChromaDB: {vs.count()} chunks")
        try:
            graph = get_graph()
            gs = graph.get_stats()
            print(f"[DATA] Graph: {gs['nodes']} entities, {gs['relationships']} relationships")
        except Exception as e:
            print(f"[DATA] Graph: not configured ({e})")

    elif cmd == "graph":
        try:
            graph = get_graph()
            gs = graph.get_stats()
            print(f"  Graph:")
            print(f"   Entities:      {gs['nodes']}")
            print(f"   Relationships: {gs['relationships']}")
        except Exception as e:
            print(f" {e}")

    elif cmd == "report":
        print("Generating HTML report ...")
        try:
            from src.report import generate_report
            path = generate_report()
            print(f"[OK]  Report: {path}")
        except Exception as e:
            print(f" {e}")

    elif cmd == "reset":
        vs = VectorStore()
        vs.reset()
        try:
            reset_graph()
        except Exception:
            pass
        print("  Reset complete.")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()