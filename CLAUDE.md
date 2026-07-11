# CLAUDE.md — AI-Search Readiness Engine (v3 continuation brief)

## Read this first: where the build stands

This brief **supersedes v2** and assumes its build state. Do NOT rebuild what is proven.

**Done and proven (do not touch except where a phase below explicitly says so):**
- Layer −1 remediation (de-branding, Hermes quarantined, keys optional, Trace backbone)
- Layer 1 steps 3–9 + 11–14 (GLiNER, GLiREL, local-LLM claims, NLI evidence, HDBSCAN topics, Kùzu KG with zero orphans, BGE-M3 embeddings with Chroma parity, heuristic signals)
- Layer 2: all 10 reasoning-simulator nodes with StepTraces
- Layer 3: explainability report, pure introspection
- `run_audit.py --url … --queries …` runs Layers 1→3 fully local, zero keys, 5.5 min end-to-end
- 48 tests passing, 1 skipped; BUILD_LOG complete

**Not done (carried into phases below):**
Layer 0 upgrades (4-bot UA re-fetch, cloaking check, sitemap seeding, PDFs), formal chunking nodes 1–2 (pysbd "Est. 2021" bug), node2vec, Common Crawl backlinks, topic-noise criterion, GLiREL over-generation, rerank recall benchmark, competitor comparison, legacy subcommand tracing. **And nothing is committed.**

**Hard constraints (unchanged from v2):**
- Fully local, zero API keys on the default path. LLM = Ollama `qwen2.5:7b-instruct` (NOT Kimi — no 7B Kimi exists; do not swap). Embeddings = BGE-M3 (1024-dim).
- Every heuristic score labeled "heuristic". Simulated vs observed numbers never conflated.
- Every node: StepTrace + verification before the next node + BUILD_LOG entry.
- Small reviewable commits. Check PyPI/GitHub for an existing MIT/Apache/BSD package before writing a module.

**Build order is the phase order below. Stop after each phase's verification and show the result.**

---

## PHASE 0 — Commit the work (do this before ANY new code)

Create branch `build/v2-layers-1-3`. Commit the existing working tree as small logical commits (one module or checkpoint each): remediation, providers, trace, each Layer-1 node group, Layer-2 simulator, Layer-3 report, tests, BUILD_LOG. Nothing in this brief starts until `git log` shows the history and the working tree is clean.

**Verification:** `git status` clean; `git log --oneline` shows ≥8 meaningful commits; tests still pass from a fresh checkout of the branch.

---

## PHASE 1 — Layer 0 completion + the AI Invisibility Score (highest value)

Build the v2 Layer 0 spec as designed, then add one new headline metric on top.

### 1a. The v2 crawler upgrades (unchanged spec, summarized)
- Sitemap-first seeding (`sitemap.xml` + nested indexes, capture `lastmod`)
- crawl4ai async BFS, same-domain, max-depth config; per page: raw HTML, markdown, JSON-LD, headings w/ level, internal/external links w/ anchor text
- Linked PDFs via pymupdf (text + tables)
- **4-bot UA re-fetch**: every page re-fetched as `GPTBot`, `ClaudeBot`, `PerplexityBot`, `Google-Extended`; diff text blocks vs browser-rendered → `access_gaps[{bot_name, missing_content[]}]`
- robots.txt parse + declared-vs-actual cross-check → `robots_conflict` (cloaking flag)

### 1b. NEW — AI Invisibility Score (the headline number)
Rationale: as of 2026, no major AI crawler renders JavaScript (Gemini via Googlebot is the lone exception). JS-only content is invisible to ChatGPT/Claude/Perplexity. Turn the access-gap diff into ONE site-level number.

- Per page: `invisible_ratio(bot) = chars(missing_content) / chars(rendered_text)`
- Site-level: weight each page by (a) internal PageRank over the InternalLink graph and (b) whether the page holds chunks that win retrievals for the query set
- Output at the very top of every report: **"X% of your answer-capable content never reaches GPTBot/ClaudeBot/PerplexityBot"** with the top-10 worst pages and the exact missing blocks
- Label: measured (it is a real diff, not a heuristic)

**Verification:** ≥95% sitemap URLs captured; `access_gaps` present (even if empty) for all 4 bots on every page; on a known JS-heavy test site the score is materially >0 and the listed missing blocks actually appear only after JS rendering (manually confirm 5 samples); on a static test site the score is ≈0.

---

## PHASE 2 — Incremental crawl & data freshness (websites change; never rebuild)

New module `src/incremental.py` + `crawl_runs` tracking. Four-level skip chain — each level prevents the more expensive one below it:

1. **Conditional GET.** Store `ETag`/`Last-Modified` per URL on first crawl; send `If-None-Match`/`If-Modified-Since` on re-crawl. `304` → skip page entirely. Check sitemap `lastmod` first to prioritize fetch order.
2. **Page hash.** Hash extracted **markdown** (never raw HTML — ads/timestamps churn it). Unchanged hash → skip processing even if headers lied.
3. **Chunk-level diff.** Changed page → re-chunk → hash each chunk → only changed/new chunks go through embeddings, GLiNER, GLiREL, claims, NLI. Deleted chunks → tombstone.
4. **Surgical graph update.** Everything already traces to `chunk_id` — use it: delete that chunk's MentionsEntity/SupportsClaim edges, Claims, and Chroma vector; insert replacements. NO full KG rebuild. Re-run topic clustering (and node2vec once it exists) only when >15% of chunks changed in a run.

Also:
- `crawl_runs` table: run_id, timestamp, pages_fetched/304/changed, chunks_changed, durations
- Adaptive revisit: pages changed in recent runs get short revisit intervals; static pages long ones
- CLI: `run_audit.py --incremental` (default once a baseline exists); `--full` forces rebuild

**Verification:** run full audit on a test site; touch 1 page; `--incremental` re-processes exactly that page's changed chunks (assert counts in trace); Chroma count == Chunk count after update; zero orphan nodes; incremental run ≥5x faster than full run on the test site.

---

## PHASE 3 — Precision upgrades to the simulator

### 3a. Formal chunking nodes (closes the v2 gap, prerequisite for 3b)
Steps 1–2 as real traced nodes: pysbd segmentation with char offsets (add regression test for the known "Est. 2021" mid-number split — post-process merge if pysbd still splits it); MiniLM semantic chunking (heading-first, cosine-drop sub-split at 0.5, chunks ≤500 tokens, heading_path + char range preserved).

**Verification:** the "Est. 2021" case passes; 10 sampled pages show no mid-abbreviation splits; every chunk traceable to heading path + char range.

### 3b. NEW — Structural scorer (GEO-SFE-style, deterministic, no ML)
Research basis: content structure predicts AI citation independently of content (one claim per block, tables over inline data, comparison grids → ~17% citation lift; schema-tagged pages cited ~2.3x more).

New node after chunking, pure rules, per chunk:
- `claims_per_block` (target 1; flag >2 per paragraph)
- `paragraph_length` (flag >120 words)
- `tabular_data` (numeric series in prose that should be a table → flag)
- `heading_question_alignment` (does the heading match interrogative/query form of the section content)
- `schema_presence` (page-level: relevant JSON-LD types present and valid)
- Output `structure_score` 0–1 per chunk + concrete flag list. Label: heuristic.
- Feed flags into Layer-3 recommendations with exact locations ("split the 400-word paragraph at /pricing §2 into 3 claim-blocks; move the 5-row comparison into a table").

**Verification:** hand-check 10 chunks — every flag points at a real structural issue; a deliberately well-structured test page scores >0.8, a wall-of-text page <0.4.

### 3c. NEW — Stochastic fan-out (replaces static query expansion)
Rationale: real engines explode 1 query into 8–15+ sub-queries and only ~27% of sub-queries are stable across runs. A fixed 3–5 facet expansion under-models this.

- Run the existing expansion prompt N=12 times at `temperature=0.8` (this is the ONE sanctioned exception to temperature-0; log it)
- Deduplicate near-identical sub-queries (embedding cosine >0.92 → merge), frequency-weight the survivors → `fanout_distribution[{sub_query, weight}]`
- Retrieval + coverage now scored per sub-query, aggregated by weight → **cluster coverage**: "covers 4 of 11 probable sub-intents (weighted 0.38)"
- Retrieval dead ends become per-sub-intent, weighted — a dead end on a 0.9-weight sub-intent outranks one on a 0.1
- Cache the distribution per query (it's 12 LLM calls); invalidate only when the query set changes

**Verification:** for 5 test queries, the distribution is stable across two generation rounds (Jaccard of top-5 sub-queries ≥0.6); coverage numbers change sensibly when a covering chunk is deleted (re-run shows the drop).

---

## PHASE 4 — Query demand ingestion (the queries must be real demand)

The audit is only as good as its queries. No paid APIs — file-based imports.

- `run_audit.py import-queries --gkp <csv>`: parse a Google Keyword Planner export (user downloads it free from their GKP account); keep keyword + volume bucket
- **Conversationalize**: GKP keywords are Google-style, AI prompts are longer and constraint-loaded. For each seed keyword, generate 2–3 conversational question variants with the local LLM (json mode, temp 0.7), e.g. "best crm small business" → "I run a 5-person agency, which CRM won't overwhelm us?" Tag each query `{seed, variant_type, volume_bucket}`
- Optional `--paa <file>`: paste-in file of People-Also-Ask questions (manually collected, free) merged into the set
- Volume bucket becomes a query weight in reporting: dead ends on high-volume queries surface first

**Verification:** importing a sample GKP CSV yields a queries file where every seed has ≥2 conversational variants; end-to-end audit runs off the imported set; report orders gaps by volume weight.

---

## PHASE 5 — Snapshots & regression alerts (turns the audit into a monitor)

- After every run, persist a `snapshot`: per-page invisibility ratio, per-chunk structure_score, per-query cluster coverage + confidence breakdown, site signal scores
- `run_audit.py diff --from <run> --to <run>` (default: last two): report ONLY regressions and improvements, each traced to cause — "6 pages lost bot-visibility (new template renders pricing via JS)", "query Q3 coverage dropped 0.4 → chunk c_812 was deleted"
- Severity = magnitude × query weight (from Phase 4) × page PageRank
- Exit code non-zero on regression above threshold → cron/CI usable

**Verification:** run baseline; deliberately break one thing (hide a section behind JS on the test site, or delete a covering chunk); diff flags exactly that regression with the correct cause and nothing else.

---

## PHASE 6 — Manual calibration panel ($0, replaces API probes)

Simulated citation probability is untrusted until checked against reality. No API budget → manual monthly panel.

- `calibration/` folder: user manually asks the top-20 weighted queries in free ChatGPT, Perplexity, and Gemini web UIs, pastes each answer + its cited URLs into `calibration/<run>/<query_id>.md` (define a trivial paste template)
- `run_audit.py calibrate`: parse the files, extract cited domains/URLs, compute per-query: did the simulator's top-5 chunks' pages appear in real citations? → agreement rate per engine + overall
- Store as `citation_probability_observed`; report it strictly separately from simulated (v2 rule stands)
- If overall agreement <50% after 2 panels: tune retriever weights (BM25/embed split, rerank cutoff) against the observed set as a labeled benchmark — log every tuning change

**Verification:** parser correctly extracts citations from 3 sample pasted answers per engine; report shows simulated and observed side by side, never merged; agreement math hand-checked on one query.

---

## PHASE 7 — Pre-publish what-if scoring

- `run_audit.py whatif --draft <md|docx> --queries <ids>`: run the draft through chunking → structural scorer → retrieval simulation **against a temporary overlay index** (draft chunks added to Chroma in-memory, never persisted to the real KG)
- Report: predicted cluster-coverage delta per target query, structure flags, missing entities vs the fan-out distribution — "this draft covers 4 of 9 sub-intents; add a comparison table and a dated pricing claim"

**Verification:** overlay leaves the persistent stores byte-identical (hash Chroma + Kùzu before/after); a draft written to answer a known dead-end query measurably raises that query's predicted coverage.

---

## PHASE 8 — Fix generation (emit patches, not advice)

Upgrade Layer-3 recommendations from prose to ready-to-use artifacts, all clearly labeled **draft — human review required, never auto-publish** (v2 working agreement stands):

- Missing/invalid schema → emit the complete JSON-LD block for that page (validate it against schema.org before emitting)
- Site-level → generate `llms.txt` from the KG (top pages, topics, canonical descriptions)
- Retrieval dead end → local-LLM-drafted FAQ/section outline for the missing sub-intent, with the source claims it should contain
- Structure flags → concrete rewrite of the flagged block (table version of the numeric prose, split paragraphs)
- Output to `fixes/<run>/` as files, referenced from the report

**Verification:** emitted JSON-LD passes schema validation; llms.txt lists only real URLs; every fix file maps to a specific report finding by id.

---

## PHASE 9 — Server log ingestion (the wedge no competitor has)

- `run_audit.py import-logs <access.log ...>`: parse common/combined log formats; filter AI bot UAs (GPTBot, ClaudeBot, PerplexityBot, OAI-SearchBot, Claude-SearchBot, Google-Extended, Meta-ExternalAgent); join hits to Page nodes
- Report: crawl-demand map — which pages bots actually visit vs which pages hold retrieval-winning chunks. The killer sentence: "AI bots hit 400 pages last month and never touched the 12 pages that answer your top queries"
- Cross-reference with `access_gaps`: bot visited AND content invisible = highest-priority fix

**Verification:** parse a sample log with known bot lines → correct per-bot counts; pages absent from logs but present in KG appear in the "never visited" list; malformed lines skipped with count, not crash.

---

## BACKLOG (from v2, do after Phase 5 or when blocking) — full build specs

1. **node2vec** (populate the existing `Entity.graph_embedding FLOAT[128]` column). Export the Entity–RelatesTo–Entity subgraph to networkx; run node2vec with dim=128, walk_length=30, num_walks=200; write vectors back to Kùzu. Re-run inside Phase 2's ">15% chunks changed" trigger. *Verify:* cosine similarity of ≤2-hop entity pairs measurably > random pairs.
2. **Common Crawl backlinks** (fills the empty ExternalDomain/ExternalPage tables). Query Common Crawl's host-level Web Graph (free, no key) for inbound links per Page; fetch linking pages, extract anchor text + context snippet + dofollow; create ExternalLinksTo edges; `topical_relevance` = cosine(context-sentence BGE-M3 embedding, linked-Chunk embedding). This is the ONE network-dependent module — put it behind `--enrich` so the default path stays offline; citation/authority signals keep their "degraded" label until it has run. *Verify:* known-linked pages have ≥1 edge; zero edges with null context_snippet.
3. **GLiREL over-generation fix.** At index time: drop relations below a confidence floor (start 0.5, tune on 30 hand-checked triplets), dedup identical (subject, relation, object) across chunks into one edge with a `support_count` property, and reject triplets whose endpoints aren't step-3 entities. *Verify:* on 30 sampled triplets, precision ≥0.8 after the floor; edge count drops materially with no loss on the hand-checked true positives.
4. **Rerank recall benchmark.** Reuse the Phase-6 calibration queries as the hand-labeled set (label which chunks truly answer each). Measure top-5 recall of the reranker; target >80%; tune the 0.4/0.6 BM25/embed split and rerank cutoff if under. Log every weight change.
5. **Topic noise ≤15%.** Re-run HDBSCAN on the full corpus (not 12 chunks); grid-search `min_cluster_size` ∈ {3,5,8,12}; reject generic labels and re-label with the local LLM. *Verify:* ≤15% noise on the full corpus.
6. **Competitor comparison.** Run Layers 0–2 on competitor domains with a separate Kùzu DB + Chroma collection per domain (never merged); per query, diff cluster coverage, structure scores, and invisibility — output "they cover sub-intents you dead-end on" with their winning chunk as evidence. Cheap once Phase 2 exists (incremental keeps competitor KGs fresh). *Verify:* diff on a query both sites answer shows both sides' winning chunks with scores.
7. **Legacy src.main subcommands:** thread the Trace through them or delete them; no untraced code path survives on the default CLI.

---

## Deliverables checklist (v3)

1. Clean committed history on `build/v2-layers-1-3` + feature branches per phase
2. `run_audit.py` full/incremental/diff/whatif/calibrate/import-queries/import-logs — all local, zero keys
3. Every report leads with the AI Invisibility Score and orders findings by (query weight × severity)
4. Simulated vs observed citation numbers separated everywhere; heuristics labeled
5. BUILD_LOG.md updated per node: library chosen, why, verification result, failures hit
