# AI Search Readiness Platform

Crawl websites, build knowledge graphs, detect semantic gaps, and simulate AI answer engine behavior — all from your terminal.

## Quick Start

```bash
# 1. Clone + install
git clone <repo-url>
cd knowledge-graph
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Set API keys
cp .env.example .env
# Edit .env — you need at least NVIDIA_API_KEY (free tier)

# 3. Full brand audit
python -m src.main audit https://yourbrand.com

# 4. Or via Hermes orchestration
python -c "
from src.orchestrator import create_api
api = create_api()
api.create_scan(
    project_id='my_brand',
    target_url='https://yourbrand.com',
    queries=['healthy alternatives for fitness freaks', 'best product for weight loss'],
)
api.start_worker()
"
```

## What It Does

| Step | Description |
|------|-------------|
| Crawl | Discover and fetch pages, extract links + metadata |
| Chunk | Split content into semantic units (sections, FAQ, claims) |
| Embed | Generate vector embeddings (NVIDIA NIM free tier) |
| Vector Store | ChromaDB for semantic search |
| Knowledge Graph | KuzuDB — entities, topics, relationships |
| Hybrid Retrieval | Broken path detection + GraphRAG |
| Coverage Scoring | 6-metric score (semantic, entity, relationship, evidence, KG completeness, overall) |
| Semantic Repair | Page-level optimization — edit vs create, suggested slugs, internal links |
| AI Simulation | Citation probability, confidence, competitor comparison |
| Hermes Orchestration | Durable workflow engine — state machine, retries, events, artifacts |

## CLI Commands

```bash
python -m src.main audit <url>       # Full crawl + chunk + embed + graph
python -m src.main ask <question>    # Ask about indexed content
python -m src.main analyze <text>    # Analyze pasted text
python -m src.main import <file>     # Import query file (.csv/.txt/.json/.xlsx)
python -m src.main report            # Generate HTML report
python -m src.main gap-report        # Generate gap analysis HTML report
python -m src.main status            # Show indexing stats
python -m src.main graph             # Show entity/relationship summary
python -m src.main reset             # Clear all data
```

## Hermes Orchestration API

```python
from src.orchestrator import create_api

api = create_api()

# Start a scan
scan = api.create_scan(
    project_id="my_brand",
    target_url="https://example.com",
    queries=["query 1", "query 2"],
)
api.start_scan(scan["scan_id"])
api.start_worker()

# Check status
status = api.get_scan_status(scan["scan_id"])
# status: created → queued → running → completed

# Get results
recs = api.get_recommendations(scan["scan_id"])
events = api.get_events(scan["scan_id"])
sims = api.get_simulation_results(scan["scan_id"])

# Cancel or retry
api.cancel_scan(scan["scan_id"])
api.retry_scan(scan["scan_id"])
```

## Requirements

- Python 3.10+
- NVIDIA API key (free tier: https://build.nvidia.com)
- 2GB+ RAM for ChromaDB + KuzuDB

## License

MIT
