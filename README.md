# AI Search Readiness Platform

Crawl websites, build knowledge graphs, detect semantic gaps, and simulate AI
answer-engine behavior — all from your terminal, **fully local and offline, with
zero API keys required**.

## Quick Start (local-first, no keys)

```bash
# 1. Clone + install
git clone <repo-url>
cd knowledge-graph
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Pull the local models (one time; runs offline afterwards)
#    - Install Ollama:  https://ollama.com/download
ollama pull qwen2.5:7b-instruct        # local generation
python -m spacy download en_core_web_trf
python -m playwright install chromium
# BGE-M3 (embeddings) and the reranker/NER/relation models download on first use.

# 3. Configure (optional — defaults are fully local)
cp .env.example .env                   # runs verbatim with NO keys

# 4. Run a traced audit
python run_audit.py --url https://example.com --queries queries.txt
#   queries.txt = one query per line (also .csv / .json). No brand defaults.
```

Everything runs on CPU if you have no GPU (slower, but no keys and no network).

## Providers (local by default, external optional)

Set in `.env` (see `.env.example`):

| Flag | Default (local) | Opt-in alternative |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | `ollama` / `qwen2.5:7b-instruct` | `deepseek` (needs `DEEPSEEK_API_KEY`) |
| `EMBED_PROVIDER` / `EMBED_MODEL` | `local` / `BAAI/bge-m3` (1024-dim) | `nvidia` (needs `NVIDIA_API_KEY`) |

If you select an external provider but leave its key blank, the tool logs a
warning and falls back to the local default. Reranking, NLI, NER, and relation
extraction are always local. Vector store = embedded ChromaDB; knowledge graph =
embedded KuzuDB (both free, no external service).

## What It Does

| Step | Description |
|------|-------------|
| Crawl | Discover and fetch pages, extract links + metadata (+ bot-access diff) |
| Chunk | Split content into semantic units (sections, FAQ, claims) |
| Embed | Local BGE-M3 vector embeddings (1024-dim) |
| Vector Store | ChromaDB for semantic search |
| Knowledge Graph | KuzuDB — entities, topics, relationships |
| Hybrid Retrieval | BM25 + vector + graph traversal |
| Coverage Scoring | heuristic metrics (labeled "heuristic" throughout) |
| AI Simulation | Traced reasoning: retrieval → evidence → confidence → answer |
| Explainability | Per-query report introspected from the Trace (no new inference) |

## CLI Commands

```bash
python run_audit.py --url <url> --queries <file>   # traced, local-first audit (default entrypoint)

python -m src.main audit <url>       # legacy full crawl + chunk + embed + graph
python -m src.main ask <question>    # ask about indexed content
python -m src.main analyze <text>    # analyze pasted text
python -m src.main import <file>     # import query file (.csv/.txt/.json/.xlsx)
python -m src.main report            # generate HTML report
python -m src.main gap-report        # generate gap analysis HTML report
python -m src.main status            # show indexing stats
python -m src.main graph             # show entity/relationship summary
python -m src.main reset             # clear all data
```

## Optional: Hermes orchestration

The durable Hermes workflow engine is **optional and off the default path**
(quarantined under `src/optional/hermes/`). The default audit is a plain
synchronous, fully-traced pipeline (`run_audit.py`). Import Hermes explicitly
only if you want async multi-scan queuing:

```python
from src.optional.hermes import create_api
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) for local generation
- ~12–14 GB disk for the local model set; 4 GB+ RAM (8 GB+ GPU optional, speeds up)
- No API keys required

## License

MIT
