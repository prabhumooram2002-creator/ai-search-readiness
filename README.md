# AI Search Readiness Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#license)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)

Crawl a website, build a knowledge graph of what it actually says, and
**simulate how an AI answer engine (ChatGPT, Claude, Perplexity, Gemini)
would answer real questions about it** — then show exactly where the
content falls short. Every score is traced back to the specific evidence
that produced it; nothing is a black box.

Runs **fully local and offline by default, with zero API keys required.**
Cloud providers (NVIDIA, DeepSeek) are opt-in swaps for the same pipeline,
not a separate mode.

## Why this exists

Search is moving from "10 blue links" to AI-generated answers. A page can
rank #1 in Google and still never be cited by ChatGPT — because AI crawlers
don't render JavaScript, because the page never states the fact plainly
enough to be extracted as a claim, or because a competitor's content simply
answers the question more directly. This tool measures that gap directly:
it indexes a site the way a retrieval-augmented answer engine would, asks it
real questions, and reports — per query, per page, with citations — what's
missing.

## Architecture

The pipeline is four layers, each one narrower and more inferential than
the last. Every node in every layer emits a `StepTrace` (inputs, outputs,
and — on Layer 2 — a heuristic score), so any number in the final report
can be traced back to the exact evidence that produced it.

```
Layer 0  Crawl & Access           →  Layer 1  Index & Knowledge Graph
─────────────────────────           ──────────────────────────────────
Sitemap-first BFS crawl             Chunk → Embed → Vector store
4-bot UA re-fetch                   Entity extraction (NER)
  (GPTBot/ClaudeBot/                Relation extraction
   PerplexityBot/Google-Extended)   Claim extraction + NLI evidence
robots.txt cross-check              Topic clustering
AI Invisibility Score               Knowledge graph (Kùzu)
      │                                        │
      ▼                                        ▼
Layer 2  Reasoning Simulator        Layer 3  Explainability Report
──────────────────────────          ──────────────────────────────
Per query, 10 traced nodes:         Gap report: per-query citability,
intent → expansion → hybrid         missing claims, worst pages, and
retrieval → rerank → graph          concrete fix drafts — all pure
traversal → NLI evidence →          introspection of the Trace, no
contradiction check → confidence    new inference at report time
→ answer synthesis → citations
```

## How it works — the tools, and what each one does

Nothing here is a monolithic "AI does everything" black box. Each stage is
a small, purpose-built, mostly-local model or library, chosen so the whole
pipeline can run on a laptop with no API keys:

| Stage | Tool | What it does | Why this one |
|---|---|---|---|
| Crawling | [`crawl4ai`](https://github.com/unclecode/crawl4ai) | Async, JS-rendering-aware site crawler; extracts markdown, JSON-LD, headings, and internal/external links | Renders JS like a browser, so we can diff that against what AI crawlers (which mostly don't) actually see |
| PDF extraction | `pymupdf` | Pulls text + tables out of linked PDFs | Fast, no external service |
| Sentence segmentation | `pysbd` | Splits page text into sentences with exact character offsets preserved | Every downstream claim traces back to one real sentence, not an approximate span |
| Embeddings | `sentence-transformers` (**BAAI/bge-m3**, 1024-dim, local) or NVIDIA NIM (`nv-embedqa-e5-v5`, opt-in cloud) | Turns chunks into vectors for semantic search | BGE-M3 is a strong, fully local, multilingual embedder with no API key; the NVIDIA path is a drop-in swap when local RAM is tight |
| Vector store | `ChromaDB` (embedded) | Stores chunk vectors, does nearest-neighbor retrieval | Zero-config, no external service, runs in-process |
| Entity extraction (NER) | [`GLiNER`](https://github.com/urchade/GLiNER) (`gliner_small`/`large-v2.1`) | Zero-shot named entity recognition — finds people, organizations, products, prices, dates with no fine-tuning needed | Runs entirely on CPU, no fixed label schema, small enough to fit alongside everything else |
| NER cross-check | `spaCy` (`en_core_web_trf` or lighter `en_core_web_sm`) | Independent second opinion on entities, flags chunks where the two extractors disagree >20% | A single extractor's errors go unnoticed; disagreement is a real QA signal |
| Relation extraction | [`GLiREL`](https://github.com/jackboyla/GLiREL) (default) or an LLM-prompted fallback | Links entity pairs with typed relations ("founded by", "offers", "competes with") | Zero-shot like GLiNER, same reasoning — see [Notes on the local model stack](#notes-on-the-local-model-stack) below for when to use the LLM fallback instead |
| Claim extraction | Local LLM via [`Ollama`](https://ollama.com/) (`qwen2.5:7b-instruct` default) | Extracts discrete, atomic factual claims from each chunk, tagged to their exact source sentence | Structured-JSON prompting at temperature 0 for determinism; fully local, no API key |
| Evidence / contradiction check | `cross-encoder/nli-deberta-v3-base` | Natural-language-inference model — does the source sentence actually entail the claim? | Catches hallucinated claims before they ever reach a report |
| Topic clustering | `HDBSCAN` + `scikit-learn` over chunk embeddings | Groups chunks into topics, with generic-label rejection and automatic re-clustering on high noise | Density-based clustering needs no predetermined number of topics |
| Knowledge graph | [`KuzuDB`](https://kuzudb.com/) (embedded) | Stores Page/Chunk/Entity/Claim/Topic nodes and their relationships for graph traversal at query time | Embedded, typed schema (catches dimension/type bugs at write time), no server to run |
| Hybrid retrieval | `rank-bm25` (lexical) + BGE-M3 (semantic), blended 0.4/0.6 | Combines keyword and meaning-based search for each simulated query | Neither alone is enough — BM25 catches exact terms, embeddings catch paraphrase |
| Reranking | `BAAI/bge-reranker-v2-m3` | Re-scores the top retrieved chunks against the query for precision | Retrieval casts a wide net; reranking narrows it the way a real answer engine would |
| Fuzzy matching | `rapidfuzz` | String-similarity utilities used in entity/query dedup | Fast, C-accelerated |
| Answer simulation | Local LLM (`Ollama`), temperature 0 | Synthesizes an answer strictly from retrieved context — never outside knowledge — then selects per-sentence citations via NLI | This is the actual "would an AI cite this site" simulation, and it's fully traced |

Every heuristic number in the output is explicitly labeled `heuristic` —
simulated scores are never presented as if they were observed production
data.

### Notes on the local model stack

The full local model set (BGE-M3 + GLiNER + spaCy-transformer + GLiREL +
NLI cross-encoder + reranker) is **~6–8 GB combined** and was built assuming
a machine with headroom for that. On a lighter machine, three env-var
swaps keep everything working without a rewrite:

| Constraint | Swap |
|---|---|
| Low RAM for embeddings | `EMBED_PROVIDER=nvidia` — routes embeddings through the NVIDIA NIM free tier instead of loading BGE-M3 locally |
| Low RAM for NER | `GLINER_MODEL=urchade/gliner_small-v2.1` (~400MB) instead of the ~1.8GB `large` default |
| Low RAM for relation extraction | `RELATION_PROVIDER=llm` — extracts relations via the same LLM already used for claims, instead of loading the ~1.8GB GLiREL model. `SKIP_RELATIONS=1` is a last-resort bypass that returns no relationship edges at all. |

## Quick Start

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency
management (a lockfile is committed for reproducible installs).

```bash
# 1. Clone + install
git clone https://github.com/prabhumooram2002-creator/ai-search-readiness.git
cd ai-search-readiness
uv sync

# 2. Pull the local models (one time; runs offline afterwards)
#    - Install Ollama: https://ollama.com/download
ollama pull qwen2.5:7b-instruct        # local generation
uv run python -m spacy download en_core_web_trf
uv run playwright install chromium
# BGE-M3, the reranker, GLiNER, GLiREL, and the NLI model all download on first use.

# 3. Configure (optional — defaults are fully local)
cp .env.example .env                   # runs verbatim with NO keys

# 4. Run a traced audit
uv run python run_audit.py --url https://example.com --queries queries.txt
#   queries.txt = one query per line (also .csv / .json). No brand defaults.
```

No `uv`? `pip install -e .` against `pyproject.toml` works too — there's no
separate `requirements.txt` to keep in sync.

Everything runs on CPU if you have no GPU (slower, but no keys and no
network required beyond the initial model downloads).

## Configuration — providers (local by default, external optional)

Set in `.env` (see `.env.example`):

| Env var | Default (local) | Opt-in alternative |
|---|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | `ollama` / `qwen2.5:7b-instruct` | `deepseek` (needs `DEEPSEEK_API_KEY`) |
| `EMBED_PROVIDER` / `EMBED_MODEL` | `local` / `BAAI/bge-m3` (1024-dim) | `nvidia` (needs `NVIDIA_API_KEY`; also 1024-dim, see note below) |
| `GLINER_MODEL` | `urchade/gliner_large-v2.1` | `urchade/gliner_small-v2.1` (lighter) |
| `SPACY_MODEL` | `en_core_web_trf` | `en_core_web_sm` (lighter, ~13MB) |
| `RELATION_PROVIDER` | `glirel` | `llm` (routes through the configured LLM instead) |
| `SKIP_RELATIONS` | unset | `1` — skip relation extraction entirely |

If you select an external provider but leave its key blank, the tool logs a
warning and falls back to the local default. Reranking, NLI, NER, and (by
default) relation extraction are always local. Vector store = embedded
ChromaDB; knowledge graph = embedded KùzuDB — both free, no external
service.

> **If you switch `NVIDIA_EMBED_MODEL`,** the on-disk embed cache is tagged
> by model name specifically so different-dimension vectors never collide
> in the same cache bucket (this bit us once — see the git log around
> `Fix embedding dimension mismatch`).

## What each layer produces

| Layer | Output |
|---|---|
| 0 — Crawl & Access | Full crawl (markdown, JSON-LD, links) + per-page 4-bot access diff + the **AI Invisibility Score**: what % of answer-capable content never reaches GPTBot/ClaudeBot/PerplexityBot because it's JS-only |
| 1 — Index & KG | Chunks, embeddings, entities, relationships, claims + NLI-verified evidence, topics, and a queryable knowledge graph — zero orphan chunks, zero unsupported claims by construction |
| 2 — Reasoning Simulator | Per query: a full traced simulation of retrieval → reranking → graph traversal → evidence check → contradiction check → confidence score → synthesized answer → per-sentence citations |
| 3 — Explainability Report | The gap report: which queries the site can/can't answer, which claims are missing, which pages are weakest, and draft fixes — ready to hand to a content team |

## CLI reference

```bash
# Primary entrypoint — traced, local-first audit (Layers 0-3 in one run)
python run_audit.py --url <url> --queries <file> [options]

  --queries FILE        one query per line (.txt/.csv/.json)
  --query "text"         inline query, repeatable
  --max-pages N          cap pages crawled
  --claims-cap N          max chunks to run LLM claim extraction on (CPU cost)
  --out DIR              report output directory
  --no-fresh              don't reset the vector store before indexing
  --incremental           incremental update (default once a baseline exists)
  --full                  force a full rebuild even with a baseline
  --fixes                 emit ready-to-use fix drafts to fixes/<run>/
```

```bash
# Legacy interactive CLI (older, non-traced entrypoint)
python -m src.main audit <url>       # full crawl + chunk + embed + graph
python -m src.main ask <question>    # ask about indexed content
python -m src.main analyze <text>    # analyze pasted text
python -m src.main import <file>     # import a query file (.csv/.txt/.json/.xlsx)
python -m src.main report            # generate HTML report
python -m src.main gap-report        # generate gap-analysis HTML report
python -m src.main status            # show indexing stats
python -m src.main graph             # show entity/relationship summary
python -m src.main reset             # clear all data
```

## Optional: Hermes orchestration

The durable Hermes workflow engine is **optional and off the default path**
(quarantined under `src/optional/hermes/`). The default audit is a plain
synchronous, fully-traced pipeline (`run_audit.py`). Import Hermes
explicitly only if you want async multi-scan queuing:

```python
from src.optional.hermes import create_api
```

## Project structure

```
run_audit.py          Primary entrypoint — traced Layers 0-3 audit
config.yaml            Runtime config (crawl limits, provider models, batch sizes)
src/
  crawl/               Layer 0 — sitemap-first crawler
  layer0.py             4-bot re-fetch, robots cross-check, AI Invisibility Score
  chunk/, chunking2.py  Layer 1 steps 1-2 — sentence + semantic chunking
  embed/                Embedding client (NVIDIA/DeepSeek/local) + on-disk cache
  ner.py                Layer 1 step 3 — GLiNER entities + spaCy cross-check
  relations.py          Layer 1 step 4 — GLiREL / LLM relation extraction
  claims.py             Layer 1 step 5 — local-LLM claim extraction
  evidence.py           Layer 1 step 6 — NLI evidence + contradiction check
  topics.py             Layer 1 step 7 — HDBSCAN topic clustering
  kg.py                 Layer 1 steps 8-9 — KùzuDB knowledge graph
  simulator.py           Layer 2 — the 10-node reasoning simulator
  report/                Layer 3 — HTML report + gap-report generation
  fixes.py               Ready-to-use fix draft generation
  providers.py           The one seam between the pipeline and any vendor
  trace.py               StepTrace backbone — every node's audit trail
  incremental.py         4-level skip chain for incremental re-crawls
tests/                  Real test suite (root-level test_*.py are gitignored scratch scripts)
docs/engineering-spec.md  Full engineering spec
BUILD_LOG.md            Phase-by-phase build log with proofs for every claim
CLAUDE.md               The spec-driven build brief this project was built from
```

## Testing

```bash
uv run pytest tests/
```

## Design principles

- **Fully local by default, zero API keys.** External providers are opt-in
  swaps for the same interface, never a separate code path.
- **Every heuristic is labeled `heuristic`.** Simulated scores and observed
  production data are never conflated.
- **Every node is traced.** Inputs, outputs, and drop reasons are recorded
  on a `StepTrace` before the next node runs — any report number can be
  traced back to its evidence.
- **Small, reviewable commits.** Check for an existing MIT/Apache/BSD
  package before writing a new module.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) for local generation
- ~6–8 GB disk for the full local model set; 8 GB+ RAM recommended (see
  [Notes on the local model stack](#notes-on-the-local-model-stack) for
  lighter-machine swaps)
- No API keys required

## License

MIT
