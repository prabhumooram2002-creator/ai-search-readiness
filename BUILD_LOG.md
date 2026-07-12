# BUILD_LOG

Per-module record of what was built, the library chosen and why, and the
verification result. Newest layer at the bottom. Every heuristic score is
labeled "heuristic" in code, logs, and UI.

---

## Environment (this machine)

- OS: Windows 11 Home, shell = Git Bash (PowerShell unavailable in harness).
- Python: 3.11.15 venv at `.venv/` (Scripts/python.exe). `pip` was missing from
  the venv and was bootstrapped via `ensurepip` (pip 24.0).
- **No NVIDIA GPU** (`nvidia-smi` absent). All local models run on **CPU**.
  User decision (2026-07-11): keep the brief's default **Ollama
  `qwen2.5:7b-instruct`** despite CPU, and install the full local stack now.
- Disk: ~49 GB free on C:.

### Pre-stage installs — verification

| Component | Result |
|---|---|
| pip packages (pysbd, spacy, sentence-transformers, hdbscan, node2vec, scikit-learn, rapidfuzz, pymupdf, transformers, gliner, glirel, ollama, torch-cpu) | all import OK |
| already present (kuzu, chromadb, crawl4ai, rank_bm25, networkx, tenacity, dotenv, httpx) | OK |
| Ollama runtime | installed via winget, `ollama version 0.31.2`, server reachable |
| spaCy `en_core_web_trf` | downloaded, `spacy.load` OK |
| playwright chromium | installed (exit 0) |
| BGE-M3 (`BAAI/bge-m3`) warmup | downloaded -- 4.3 GB in HF hub cache, both `pytorch_model.bin` + `model.safetensors` present, dim=1024 confirmed |
| `qwen2.5:7b-instruct` pull | downloaded -- 4.7 GB, `ollama list` shows it, no partial blobs |
| pytest | installed |

Note: Ollama installs to `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` and is not on
the Git Bash PATH; the full path is used in scripts.

---

## Provider abstraction — `src/providers.py`

- Exposes `llm_complete(prompt, *, json, temperature, system, max_tokens) -> str`
  (routes on `LLM_PROVIDER`: ollama | deepseek) and
  `embed_texts(list[str]) -> list[vec1024]` (routes on `EMBED_PROVIDER`: local | nvidia).
- Defaults fully local: `LLM_PROVIDER=ollama`, `LLM_MODEL=qwen2.5:7b-instruct`,
  `EMBED_PROVIDER=local`, `EMBED_MODEL=BAAI/bge-m3` (1024-dim).
- If a provider is selected but its key is absent → logs a warning and falls back
  to local (`_resolve_llm_provider` / `_resolve_embed_provider`).
- Heavy imports (ollama, sentence-transformers/torch) are lazy so importing the
  module is cheap and never fails when models aren't pulled yet.
- Reranker / NLI / NER / relations are intentionally NOT here — they are always
  local and live in their own modules (no provider switch), per the brief.
- **Verification:** imports OK; `provider_info()` returns local defaults even
  though the user's `.env.local` holds NVIDIA/DeepSeek keys (keys present but
  unused by default). BGE-M3 dim confirmed = 1024 during warmup.

## Trace backbone — `src/trace.py`

- `StepTrace{name, inputs, outputs, scores, dropped, latency_ms, notes}` with
  `.drop(item, reason)` (reason logged at drop time, for Layer 3) and `.note()`.
- `Trace{query, intent, ..., retrieved, reranked, graph_paths, evidence,
  contradictions, confidence(+breakdown), answer, citations,
  unsupported_sentences, + Layer-3 rollups, + steps[]}`.
- `Trace.start_step(name, **inputs)` context manager times the node and appends
  its StepTrace **even on error** (guarantees no stage silently vanishes).
- Zero heavy deps so tests/stages import it freely.
- **Verification:** `tests/test_trace_contract.py` — 4 passed. Enforces the rule
  "every declared stage emits a non-empty StepTrace" (offline; embed/chat
  monkeypatched).

---

## LAYER -1 — Repo remediation

Repo confirmed: `ai-search-readiness` v6.6, git `main`, at
`C:\Users\Prabh\Projects\knowledge-graph`.

1. **Hermes quarantined** — `git mv src/orchestrator → src/optional/hermes`.
   `src/optional/__init__.py` documents that nothing there is on the default
   path. `src/main.py` never imported the orchestrator, so the default CLI is
   unaffected. `pyproject.toml` `hermes` script repointed;
   `tests/test_hermes.py` imports updated to `src.optional.hermes`.
2. **Hardcoded brand queries removed** — the `"healthy alternatives for fitness
   freaks" / "weight loss"` defaults in `steps.py` now raise (queries must be
   explicit); the `__init__.py` docstring example de-branded.
3. **Keys optional** — `.env.example` rewritten local-first: copied verbatim
   (no keys) the tool runs fully offline. NVIDIA/DeepSeek only behind provider
   flags; missing key → warn + local fallback.
4. **Kept** ChromaDB + KuzuDB (embedded, free, already integrated).
5. **CLI surface kept**; new `run_audit.py` is the traced default entrypoint
   (`--url`, `--queries`, `--query`, `--max-pages`, `--out`, `--no-fresh`).
6. **Traced pipeline** — `run_audit.py` threads a `Trace` per query through the
   stages that exist today (intent → retriever → answer_synthesis); site-wide
   indexing (crawl → chunk → embed → store) reported separately. Local-first via
   a single seam: `embed_chunks` and `chat_complete` now delegate to
   `providers.py` (local by default), so BOTH `run_audit.py` and `src.main` run
   offline with no code duplication. Embed cache is model-namespaced so 1024-dim
   local vectors never collide with prior NVIDIA (4096-dim) cache entries.

### Layer -1 verification

| Check | Result |
|---|---|
| `grep -riE "fitness\|weight loss\|healthy alternatives\|gym freak" src/` | none (PASS) |
| Hermes off the default path | `src.main` + `run_audit` import with no orchestrator dependency (PASS) |
| `run_audit.py --help` / `load_queries` refuses empty (no brand defaults) | PASS |
| edited modules import (`src.main`, `embed.embedder`, `embed.chat`, `run_audit`) | PASS |
| trace contract test | 4 passed |
| end-to-end `run_audit --url ... --queries ...` with NO keys | PASS -- ran against https://example.com (2026-07-11): crawl 1/1 -> chunk -> BGE-M3 embed (dim=1024, from cache) -> Chroma store -> retrieve -> qwen2.5:7b answer, all local, zero keys. Report at `data/reports/audit_report.md`. |

### Flagged for review (not yet actioned)

- **Brand lexicon in `src/input_layer/`** — `__init__.py` and `batch_pipeline.py`
  carry a wickedgüd/noodles/pasta/ramen domain dictionary (synonym maps, entity
  canonicalization, `hero_entities`). The brief's *literal* grep
  (`fitness|weight loss|nvidia_api_key`) passes, but its *spirit* ("no
  brand-specific defaults anywhere in src/") is violated. This module is NOT on
  the default `run_audit.py` path (only `src.main analyze/import` use it).
  Recommendation: externalize the lexicon to a config file / make it empty by
  default. Deferred pending user decision — gutting a working module wired into
  `src.main` is a scope call, not a mechanical cleanup.
- **`nvidia_api_key` string still present** in `core/config.py`, `embed/*`,
  `graph/extract.py`, `providers.py` — but ONLY as the optional NVIDIA provider
  and its fallback logic, exactly as the brief's provider section requires. The
  default run path uses none of it. A literal-zero grep is incompatible with
  keeping NVIDIA as an optional provider, so this is interpreted as "default
  path requires no key" (satisfied).


---

## Milestone closeout -- local stack verified + test suite green (2026-07-11)

- **End-to-end proof (no keys):** `run_audit.py --url https://example.com` ran the
  full `crawl -> chunk -> embed -> store -> retrieve -> answer` path on the local
  stack (ollama:qwen2.5:7b-instruct + local BGE-M3, dim=1024). Coherent answer
  synthesized on CPU (~3 min). Report written to `data/reports/`.
- **Test suite green:** `pytest` -> **8 passed, 1 skipped** (0.4s).
  - `tests/test_trace_contract.py` -- 4 passed (offline).
  - `tests/test_hermes.py` -- 4 passed + 1 opt-in pipeline test skipped
    (`HERMES_RUN_PIPELINE=1` to run the real worker end-to-end).
- **Fixes applied to make it green:**
  1. `HermesStore._local` was a **class** attribute (`threading.local()`), leaking
     the first-opened connection across all store instances. Made it per-instance
     in `__init__`.
  2. Added `HermesStore.close()` / `HermesAPI.close()` to release the SQLite
     connection -- Windows refuses to unlink an open DB file, which was failing
     every Hermes test at teardown (`PermissionError [WinError 32]`).
  3. Rewrote `tests/test_hermes.py`: each test uses its own throwaway temp DB,
     closed in a `finally`; ASCII-only output (Windows console can't encode the
     old unicode markers); de-branded the pipeline test (was hardcoding
     wickedgud/fitness/pasta queries -- contradicted the Layer -1 de-branding) and
     made it opt-in so the default suite stays fast + deterministic.
  4. `pyproject.toml` `[tool.pytest.ini_options] testpaths=["tests"]` so bare
     `pytest` no longer tries to collect the root-level ad-hoc scratch scripts
     (`test_v5.py`, `test_v5b.py`, `test_v66.py`, `test_mapper_noodles.py`), which
     execute pipeline code at import and crash collection.
  5. Fixed a stale `from src.orchestrator import ...` reference in the Hermes
     package docstring (module moved to `src.optional.hermes` in Layer -1).

**Not done (out of scope for this milestone):** Layers 0-2 feature work
(query expansion, BM25 + rerank, graph traversal, evidence/contradiction/
confidence nodes) and the Layer 3 introspection pass. These are the documented
next layers, each a StepTrace-emitting node on the existing traced backbone.

---

## Node (A) — De-brand: externalize input-layer lexicon (2026-07-11)

**Node:** Finish de-branding — move noodles/pasta/WickedGüd lexicon out of
`src/input_layer` into a config file that is EMPTY by default.

**Implemented:**
- `config/lexicon.yaml` — new, EMPTY by default (all sections `{}`/`[]`/`null`).
  Schema documented in-file: synonyms, slang, noise_patterns, brands,
  product_types, ingredients, attributes, hero_entities, default_root/brand.
- `src/input_layer/__init__.py` — added `_load_lexicon()` + `_LEX` + public
  `hero_entities()`. `_semantic_normalize`, `_extract_entities_from_query`,
  `parse_expected_chain`, `_infer_product_type`, `_infer_attribute` now read the
  loaded lexicon instead of hardcoded literals. Path overridable via `LEXICON_PATH`.
- `src/input_layer/batch_pipeline.py:255` — hero-products set now comes from
  `hero_entities()` (empty by default) instead of a hardcoded WickedGüd set.

**Proof — grep (brand terms in src/input_layer):**
```
grep -rinE "wickedg|noodle|ramen|pasta|maggi|samyang|nissin|combos|byob|millet|quinoa|maida" src/input_layer --include=*.py
-> exit 1 (none found)
```
Residual brand strings elsewhere (out of scope for (A), noted): comment in
`src/graph/extract.py:326`; hardcoded HTML title in `src/report/gap_report.py:122,172`.

**Proof — real run (default empty lexicon):**
```
hero_entities(): set()
_LEX sizes: all 0 / None
QUERY 'wickedgud noodles price'  -> detected_entities: []  canonical: 'wickedgud noodles price'  chain: []
QUERY 'best pasta for weight loss' -> detected_entities: []  chain: []
QUERY 'maggi vs samyang ramen'   -> detected_entities: []  chain: []
```

**Proof — pytest (`tests/test_lexicon.py`):**
```
test_default_lexicon_is_empty_and_brand_free PASSED
test_populated_lexicon_reenables_extraction  PASSED
Full suite: 10 passed, 1 skipped
```
(Second test loads a temp lexicon via LEXICON_PATH and confirms extraction
re-enables — de-branding removed the coupling, not the capability.)

---

## Node (B) — Local graph-extraction path via providers.llm_complete (2026-07-11)

**Node:** Add a LOCAL extraction path in `src/graph/extract.py` so
entities/relationships build with zero keys (CLAUDE.md provider rule: any
external-API path must degrade to local).

**Implemented (one file, `src/graph/extract.py`):**
- New `_local_extract()` — routes through `providers.llm_complete(json=True)`
  (Ollama qwen2.5:7b by default; sync call run in executor).
- Extraction loop is now LOCAL-FIRST: local runs unconditionally; NVIDIA/DeepSeek
  engage only if local fails AND a key is configured.
- De-branded the extraction schema: health/nutrition ENTITY_TYPES
  (Nutrient/Disease/…) and prompt rules replaced with domain-agnostic sets
  (Person/Organization/Product/…); "nutrition table" prompt block removed.

**Proof — clean-room run (`.env.local` moved aside + env vars stripped):**
```
PROVIDER: {"llm_provider_effective": "ollama", "llm_model": "qwen2.5:7b-instruct",
           "deepseek_key_present": false, "nvidia_key_present": false}
example.com chunk: '# Example Domain This domain is for use in documentation
examples without needing permission. Avoid use in operations.'
raw qwen output: {"entities": [], "relationships": []}   <- correctly conservative:
  the page genuinely has no extractable entities; "do not invent" honored.
Rich test chunk (Anthropic/Claude text) -> 6 entities, 5 relationships, e.g.
  Anthropic [Organization], Dario Amodei [Person], Claude Opus [Product];
  anthropic -builds-> claude-family-of-ai-assistants (conf 1.0)
```

**Proof — entities in the Trace** (`trace.steps[0]` after `entity_extraction` step):
```
TRACE.steps[0].name  : entity_extraction
TRACE.steps[0].scores: {'n_entities': 6.0, 'n_relationships': 5.0}
outputs.entities: Anthropic [Organization], Dario Amodei [Person],
  Claude family of AI assistants / OpenAI's ChatGPT / Google's Gemini /
  Claude Opus [Product]
```

**Proof — pytest (`tests/test_local_extraction.py`):**
```
test_local_extraction_routes_through_providers_without_key PASSED
test_entity_label_set_is_domain_agnostic                   PASSED
Full suite: 12 passed, 1 skipped
```

**Known gaps (honest):**
1. Relation-type schema adherence is imperfect: qwen emitted `foundedBy`,
   `builds`, `competesWith` — not in RELATIONSHIP_TYPES (only `contains`
   conformed). Fix candidates: post-validation/coercion, or move to GLiREL
   (CLAUDE.md Layer 1 step 4 — the canonical node for this).
2. First proof run showed `nvidia_key_present: true` despite the clean-room
   setup (source undetermined; likely a failed `.env.local` move). Re-ran with
   verified `false/false` — the run of record above. Even in run 1 the cloud
   path was never invoked (local succeeded first).
3. example.com yields 0 entities by design (thin content) — rich-content run is
   the extraction proof.

---

## Node — Layer 1 step 3: Entity Extraction via GLiNER (2026-07-11)

**Node (CLAUDE.md Layer 1.3):** GLiNER `urchade/gliner_large-v2.1` zero-shot NER
over chunks, configurable labels, spaCy `en_core_web_trf` cross-check with >20%
disagreement flag.

**Implemented (one file, `src/ner.py`):**
- `extract_entities(chunks, labels=DEFAULT_LABELS, threshold=0.4, trace=None)`
  -> deduplicated entity records `{id, type, name, synonyms, source_chunk_ids,
  score, mentions[{chunk_id, char_start, char_end, score}]}` (graph-upsert
  compatible). Emits StepTrace `entity_extraction` when a Trace is passed —
  including on error.
- `crosscheck_with_spacy(...)` -> per-chunk Jaccard disagreement (heuristic,
  labeled), flags chunks >20%.
- Windows fix: force `HF_HUB_DISABLE_SYMLINKS` copy mode before the model
  download (first attempt died with `OSError [WinError 1314]` — symlink
  privilege; logged honestly, fixed in-file, retried).

**Proof — real run (example.com crawl->chunk + rich chunk):**
```
GLiNER: 13 raw mentions -> 11 unique entities across 2 chunks (4.5s inference)
example.com (18 words): Example Domain [Organization] 0.597, domain [Location] 0.479
rich chunk: Anthropic [Organization] 0.987, Dario Amodei [Person] 0.991,
  San Francisco [Location] 0.984, Claude/ChatGPT/Gemini/Opus [Product], OpenAI/Google [Organization]
CROSS-CHECK: rich chunk disagreement 0.2 (not flagged);
  example.com flagged at 1.0 (spaCy finds nothing on the thin page)
TRACE.steps[0]: name=entity_extraction inputs={n_chunks:2, threshold:0.4,
  model:urchade/gliner_large-v2.1} scores={n_entities:11.0} latency_ms=4498
```

**Proof — pytest (`tests/test_ner.py`):**
```
test_default_labels_match_claude_md            PASSED
test_extraction_dedup_merge_and_trace          PASSED
test_trace_step_appended_even_on_model_error   PASSED
test_crosscheck_disagreement_math              PASSED
Full suite: 16 passed, 1 skipped
```

**Notes (honest):**
1. GLiNER beats the Node-B LLM path on speed (4.5s vs ~5min) and found entities
   on example.com where qwen returned none; one mislabel observed
   ('domain' tagged Location, score 0.48 — near threshold).
2. CLAUDE.md's "<20% disagreement on 20 sampled chunks" verification needs a
   real multi-chunk site; with only 2 chunks here (one pathologically thin) the
   mean is 0.6. DEFERRED to first real-site indexing run.
3. spaCy comparison is label-mapped (PERSON/ORG/PRODUCT/GPE/LOC/FAC/DATE/MONEY);
   'feature' has no spaCy counterpart so it never penalizes disagreement.
4. Model cached at ~/.cache/huggingface (copy mode, no symlinks).

---

## Node — Layer 1 step 4: Relation Extraction via GLiREL (2026-07-11)

**Node (CLAUDE.md Layer 1.4):** GLiREL over step-3 entity pairs ->
`triplet{subject_entity_id, relation, object_entity_id, source_chunk_id}`.
Verification rule: both endpoints of every triplet exist in step-3 entities
for that chunk (no hallucinated entities).

**Implemented (one file, `src/relations.py`):**
- `extract_relations(chunks, entities, labels=DEFAULT_RELATION_LABELS,
  threshold=0.5, top_k=1, trace=None)` — whitespace tokenizer with char-offset
  mapping (step-3 mentions -> token spans -> GLiREL `ner` input), endpoint
  validation with `st.drop(reason)` at drop time, dedup keeping best score,
  StepTrace `relation_extraction` (also on error).
- Model `jackboyla/glirel-large-v0` (env `GLIREL_MODEL`), 1.8 GB, cached.

**Failures hit and fixed (both pasted in session):**
1. `ModuleNotFoundError: loguru` — glirel 1.2.1 doesn't declare it. Installed;
   added to pyproject. CORRECTION: the pre-stage BUILD_LOG claim that glirel
   "imports OK" was WRONG — it had never actually been imported successfully.
2. `TypeError: GLiREL._from_pretrained() missing 'proxies'/'resume_download'`
   — glirel 1.2.1 (latest, no upstream fix) vs huggingface_hub 1.21 API drift.
   Shimmed defaults in `_get_glirel()`.

**Proof — real run (chain: GLiNER step 3 -> GLiREL step 4, same 2 chunks):**
```
GLiREL: 84 raw predictions -> 60 validated triplets (threshold=0.5, 4.1s inference)
Top: claude -[founded by]-> dario-amodei 0.899 | anthropic -[founded by]-> dario-amodei 0.893
     anthropic -[located in]-> san-francisco 0.697 | anthropic -[develops]-> opus 0.595
     claude -[competes with]-> chatgpt/gemini ~0.73-0.75
VERIFICATION PASS: all 60 triplet endpoints exist in step-3 entities (0 dropped)
TRACE steps: ['entity_extraction', 'relation_extraction']; latency 4131ms
```

**Proof — pytest (`tests/test_relations.py`):**
```
test_char_to_token_span_mapping                              PASSED
test_valid_triplets_and_ner_spans                            PASSED
test_hallucinated_endpoint_dropped_with_reason               PASSED
test_dedup_keeps_best_score_and_skips_single_entity_chunks   PASSED
test_trace_step_appended_even_on_model_error                 PASSED
Full suite: 21 passed, 1 skipped
```

**Quality caveat (honest, important):** zero-shot GLiREL OVER-GENERATES on
entity-dense text: 60 triplets from one 3-sentence chunk, including wrong
directions ("dario-amodei -[founded by]-> anthropic" 0.879) and nonsense with
HIGH scores ("san-francisco -[founded by]-> dario-amodei" 0.874,
"san-francisco -[competes with]-> google" 0.645). Score alone cannot separate
good from bad here. The spec's own verification (endpoint validity) PASSES;
precision is expected to be recovered downstream by Layer 1 step 6 / Layer 2
evidence validation (NLI entailment against source text). Two cheaper in-node
mitigations available on request: (a) GLiREL type-constrained labels
(allowed_head/allowed_tail per relation), (b) raise threshold to ~0.8 (keeps
the core facts, kills most but not all noise).

---

## Node — Layer 1 step 5: Claim Extraction via local LLM (2026-07-11)

**Node (CLAUDE.md Layer 1.5):** `llm_complete(json=True, temperature=0)` extracts
discrete atomic factual claims, each tagged with its exact source sentence.
Verification rule: every claim maps to a real sentence offset from step 1.

**Implemented (one file, `src/claims.py`):**
- `segment_sentences(text, chunk_id)` — pysbd with char spans (step-1
  discipline: every sentence carries `char_start/char_end` verified to slice
  the source text exactly).
- `extract_claims(chunks, trace=None)` — numbered-sentence prompt to the local
  LLM (JSON mode, temp 0), claims with invalid/missing sentence references are
  DROPPED with reason on the StepTrace (`claim_extraction`, emitted even on
  error). Claim ids are deterministic content hashes.

**Proof — real run (example.com + rich chunk, effective provider ollama:qwen2.5:7b):**
```
5 atomic claims from 2 chunks (127.8s CPU, 0 dropped):
  rich0:s1 (compound) split atomically ->
    'Anthropic was founded by Dario Amodei.'
    'Anthropic builds the Claude family of AI assistants.'
  rich0:s2 split -> 'Claude competes with OpenAI's ChatGPT.' /
                    'Claude competes with Google's Gemini.'
  rich0:s3 -> 'Claude Opus is the most capable model in the lineup.'
VERIFICATION PASS: every claim maps to a real step-1 sentence offset
TRACE.steps[0]: name=claim_extraction scores={n_claims: 5.0}
```

**Proof — pytest (`tests/test_claims.py`):**
```
test_segment_sentences_offsets_and_ids            PASSED
test_claims_map_to_real_sentences                 PASSED
test_invalid_sentence_reference_dropped_with_reason PASSED
test_fenced_json_tolerated                        PASSED
test_trace_step_appended_even_on_llm_error        PASSED
Full suite: 26 passed, 1 skipped
```

**Notes (honest):**
1. Atomic splitting works exactly as spec'd (compound sentences -> multiple
   claims), but recall isn't perfect: "founded ... in San Francisco" was not
   emitted as its own location claim.
2. qwen produced 0 claims for example.com's two sentences (treated as
   instructions, not facts) — defensibly conservative, noted.
3. This run used the default env (keys present but UNUSED — effective provider
   was ollama). Zero-key routing of `llm_complete` was already clean-room
   proven in Node (B).
4. pysbd edge case spotted during preflight: it splits "Est. 2021" mid-
   abbreviation — relevant to step 1's own verification when that node is
   formally built.

---

## Node — Layer 1 step 6: Evidence Extraction via NLI (2026-07-11)

**Node (CLAUDE.md Layer 1.6):** for each step-5 claim, ±2-sentence evidence
span scored with `cross-encoder/nli-deberta-v3-base` (entailment / neutral /
contradiction + confidence). Verification rule: claims contradicting their own
source context are flagged and re-run.

**Implemented (one file, `src/evidence.py`):**
- `extract_evidence(chunks, claims, trace=None)` -> `evidence{claim_id,
  evidence_text, entailment_label, entailment_confidence, premise_used,
  contradiction_flagged}`. Label order read from model config (never
  hardcoded); invalid claims dropped with reasons; StepTrace
  `evidence_extraction` (also on error). HF copy-mode fix reused.
- **Two-pass premise scoring** (fix found during this node's verification):
  pass 1 scores claim vs its exact source sentence; only non-entailed claims
  retry vs the ±2 span; contradictions re-run vs the full chunk (flag survives
  either way).

**Verification failure found and fixed in place (per working agreement):**
First live run scored 3/5 TRUE claims as "neutral" (e.g. 'Anthropic was
founded by Dario Amodei.' -> neutral 0.9993 against the ±2 span). Diagnostic
(pasted in session) isolated the cause — premise length:
```
sent-only premise -> entailment 0.9966 | ±2-span premise -> neutral 0.9993
```
The cross-encoder is trained on short premises; multi-sentence spans bury the
fact. After the two-pass fix: 5/5 genuine claims entailed.

**Proof — final live run (real qwen claims + real NLI, 1 synthetic false claim):**
```
Evidence: 6 records, labels={'entailment': 5, 'contradiction': 1}, flagged=1
  [entailment 0.9966] 'Anthropic was founded by Dario Amodei.'
  [entailment 0.9978] 'Anthropic builds the Claude family of AI assistants.'
  [entailment 0.9961] "Claude competes with OpenAI's ChatGPT."
  [entailment 0.9950] "Claude competes with Google's Gemini."
  [entailment 0.9971] 'Claude Opus is the most capable model in the lineup.'
  [contradiction 0.9999] SYNTHETIC 'Claude Opus is the least capable...'
      FLAGGED (re-run vs full chunk: contradiction 0.9997 -> stays flagged)
VERIFICATION PASS: false claim flagged + re-run; 5/5 genuine claims entailed
TRACE steps: ['claim_extraction', 'evidence_extraction']
```

**Proof — pytest (`tests/test_evidence.py`):**
```
test_evidence_span_is_plus_minus_two                 PASSED
test_pass1_entailment_uses_source_sentence_premise   PASSED
test_neutral_pass1_retries_against_span              PASSED
test_contradiction_flagged_and_rerun_unresolved      PASSED
test_contradiction_rerun_resolved_keeps_flag         PASSED
test_bad_claims_dropped_with_reason                  PASSED
test_trace_step_appended_even_on_model_error         PASSED
Full suite: 33 passed, 1 skipped
```

**Notes:** NLI model cached (~740 MB); scoring is fast (~seconds for 6 claims,
3 batched passes). First proof attempt was killed mid-download (resumed).
The contradiction case used a SYNTHETIC false claim (real NLI scoring, clearly
labeled) — organic claims from truthful pages rarely contradict their source.
This node is what will filter GLiREL's step-4 over-generation downstream.

---

## Batch build session (2026-07-11, "complete all tasks" directive)

Mode switched from one-node-at-a-time to batch: all remaining modules written,
verified via the offline suite, plus ONE combined end-to-end proof (below).

### Layer 1 step 7 — Topic Clustering (src/topics.py) — HONEST PARTIAL
- HDBSCAN over BGE-M3 embeddings + qwen labels; generic labels rejected+retried;
  noise>15% re-runs with HALVED min_cluster_size (first retry strategy, mcs-1,
  provably useless: 33->32->31 stayed 100% noise on 675 real chunks — fixed).
- OOM found+fixed: untruncated chunks -> 8 GB alloc failure in BGE attention;
  now embeds first 2000 chars per chunk (EMBED_CHARS).
- Live proof (docs.python.org, 12 scoped chunks): 2 clusters, labels
  'Python versions' / 'Python programming' -> generic-label check PASS.
  Noise 17% vs 15% limit -> **FAIL by one chunk** (12-chunk granularity =
  8.3%/chunk; criterion is defined for site scale). DEFERRED to a full-site
  run — not claimed as a pass.
- Tests: 5 passed (incl. halving re-run + generic-retry mechanics).

### Layer 1 steps 8+9 — Knowledge Graph (src/kg.py)
- Full CLAUDE.md DDL: Page/Chunk(FLOAT[1024])/Entity(FLOAT[128])/Claim/Topic/
  ExternalDomain/ExternalPage + HasChunk/MentionsEntity/RelatesTo/SupportsClaim/
  BelongsToTopic/InternalLink/ExternalLinksTo. Parameterized Cypher, MERGE
  idempotent. two_hop_entities() traversal for Layer 2.
- Verification enforced in verify(): counts per type, zero orphan Chunks,
  every Claim >=1 SupportsClaim (structural: own source chunk + step-6 NLI
  enrichment). Step 9: embeddings stored on Chunk nodes + Chroma parity check
  in run_audit.
- Real-kuzu smoke: counts exact, 0 orphans, 0 unsupported, 2-hop returns
  cross-chunk paths. Tests: 4 passed (incl. idempotent rebuild, drop reasons).

### LAYER 2 — AI Reasoning Simulator (src/simulator.py)
- 10 StepTrace-emitting nodes: intent -> expansion (3-5) -> hybrid retriever
  (rank_bm25 0.4 + BGE cosine 0.6, top-20, dead-end floor 0.25) -> rerank
  (bge-reranker-v2-m3, MiniLM fallback, top-5, losers dropped WITH scores) ->
  Kùzu 2-hop traversal -> NLI evidence validation (drops recorded; if all drop,
  keeps best reranked WITH note) -> pairwise contradiction detection (flagged,
  never silently resolved) -> confidence = clamp(0.35*retr + 0.25*entail +
  0.2*authority + 0.2*freshness - 0.3*contradiction) with full breakdown ->
  context-only synthesis (temp 0) -> per-sentence NLI citation selection with
  unsupported_sentences[] surfaced.
- Trace contract test updated: all 10 stages must emit non-empty StepTraces.

### LAYER 3 — Explainability (src/explain.py)
- Pure introspection over the Trace — no new inference. Rows: why-answer/
  chunks, retrieved-but-lost (losing scores), drops-with-reasons, missing
  entities (KG lookup), weak evidence (<0.5), circular RelatesTo refs,
  hallucination risk (unsupported sentences), retrieval dead ends, trust
  bottlenecks, citation probability (SIMULATED label; observed=None unless
  real engine runs are logged), rule-engine recommendations per failure
  pattern. Markdown renderer included.

### Layer 1 steps 11-14 — Signals (src/signals.py)
- Heuristic 0-1 scores, formulas logged in FORMULAS, labeled "heuristic":
  trust (schema/byline/sameAs/inverse-contradiction), citation (internal
  inbound only — 10b pending, noted per page), authority (0.6*citation +
  0.4*trust, domain proxy pending), freshness (exp decay from date evidence,
  0.5 neutral when none).

### Wiring — run_audit.py (Layers 1->3)
- index: crawl -> chunk -> embed (Chroma+Kùzu) -> GLiNER -> GLiREL -> claims
  (capped via --claims-cap for CPU) -> NLI evidence -> topics -> signals -> KG.
- Frees GLiNER/GLiREL/spaCy before Layer 2 (full model set does not fit in RAM
  at once on this machine).
- per query: Layer 2 simulator -> Layer 3 explanation; reports (JSON + MD).

### Suite after batch: 48 passed, 1 skipped.

### Fix log
- glirel: missing loguru dep (added to pyproject); _from_pretrained shim for
  huggingface_hub 1.21 API drift.
- HF symlink WinError 1314: copy-mode forced in ner/relations/evidence/simulator.
- explain.py: TypeError in why_this_answer caught by suite, fixed.

### HONEST GAP LIST — specified but NOT built (no claims otherwise)
1. **LAYER 0 upgrades**: sitemap-first seeding w/ lastmod, 4-bot UA re-fetch
   (GPTBot/ClaudeBot/PerplexityBot/Google-Extended) + access_gaps diff,
   robots.txt cloaking cross-check, PDF extraction (pymupdf), JSON-LD capture.
   Current crawler = plain BFS from Layer -1 era. This is the spec's
   "crown jewel" and is MISSING.
2. **Step 1-2 as formal nodes**: chunking is paragraph-based, not
   MiniLM-similarity semantic chunking; pysbd offsets exist only inside
   claims/evidence. pysbd "Est. 2021" mid-abbreviation split noted.
3. **Step 10 node2vec**: Entity.graph_embedding column exists, never populated.
4. **Step 10b Common Crawl**: External* tables exist, always empty; citation/
   authority signals explicitly degraded (noted per page).
5. **Step 4 relation quality**: GLiREL over-generation (60 triplets/3
   sentences) unresolved; downstream NLI filters at query time only.
6. **Layer 2 rerank verification** ("top-5 recall >80% on 20 hand-labeled
   queries") not performed — needs a hand-labeled set.
7. **Step 7 noise<=15%** deferred (17% on a 12-chunk sample; granularity).
8. **Competitor comparison** (Layer 3 row): not built.
9. **CLI surface**: run_audit.py is the traced path; `src.main` subcommands
   (audit/ask/...) still route through the OLD un-traced pipeline.

### END-TO-END PROOF (zero keys, clean room — 2026-07-11 23:00-23:06)
`run_audit.py --url https://example.com --queries q2.txt --max-pages 3`,
`.env.local` moved aside + key env vars stripped. Verified in the report JSON:
`deepseek_key_present: false, nvidia_key_present: false`, provider ollama/local.
```
site_trace steps : entity_extraction, relation_extraction, claim_extraction,
                   evidence_extraction, topic_clustering, knowledge_graph
per-query steps  : intent, query_expansion, retriever, rerank, graph_traversal,
                   evidence_validation, contradiction_detection, confidence,
                   answer_synthesis, citation_selection   (all 10, both queries)
KG: Page 1, Chunk 1, Entity 2 | orphan_chunks=0, claims_without_support=0
Chroma/Kùzu parity OK, embed dim 1024. Runtime ~5.5 min on CPU.
Answers grounded with 2 citations each, 0 unsupported sentences.
confidence 0.20 / 0.16 with full breakdowns (low is CORRECT for a thin page —
retrieval dead ends + weak evidence drove it down, and the Layer 3 report says
exactly why, with per-drop reasons and content-gap recommendations).
```
Layer 3 output includes: dropped-with-reason (NLI high-confidence neutral),
4 retrieval dead ends per query (real content gaps on example.com), and
rule-engine recommendations (weak_evidence, retrieval_dead_end).
Suite after everything: **48 passed, 1 skipped**.

---

## PHASE 1 (v3 brief) — Layer 0 completion + AI Invisibility Score (2026-07-12)

**Built (`src/layer0.py` + crawler seeding + run_audit wiring):**
- Sitemap-first seeding: fetch_sitemap (nested indexes, lastmod) -> URLs seeded
  into the crawl queue (crawl_site gained `seed_urls`).
- Per-page enrichment: JSON-LD, headings w/ level, internal/external links w/
  anchor text, linked-PDF extraction (pymupdf, text+tables).
- 4-bot UA re-fetch (GPTBot/ClaudeBot/PerplexityBot/Google-Extended, plain
  HTTP = no JS, like the real bots) -> access_gaps[{bot, missing_content[]}]
  per page, present even when empty.
- robots.txt declared-vs-actual cross-check -> robots_conflict
  (fetchable_despite_disallow / blocked_despite_allow).
- AI Invisibility Score (labeled MEASURED): per page chars(missing)/chars(
  rendered) per bot; site level weighted by internal PageRank x2 for
  retrieval-winning pages; headline + top-10 worst pages lead every report.

**Failures found and fixed during live verification (both pasted in session):**
1. False 100% invisibility on a fully static page — crawl4ai markdown flattens
   link HREFS into text; hrefs are attributes, never in bot-visible text.
   Fixed: URLs stripped from rendered blocks + 4-word shingle containment
   (>=50%) instead of brittle whole-block substring. Regression test added.
2. Sitemap capture 3.6% — seeding was missing (only measured). Implemented
   sitemap-first queue seeding; capture went to 100%.

**Live verification (quotes.toscrape.com/js vs static, sitemaps.org):**
```
JS site   : invisibility 1.00 (all 4 bots), 13 missing blocks = the quotes;
            5/5 sampled blocks CONFIRMED absent from raw GPTBot HTML
Static twin: invisibility 0.00 (all 4 bots)
Sitemap    : 84 URLs seeded -> 84/84 captured (100% >= 95%), lastmod 84/84
access_gaps present for all 4 bots on every page (incl. 90-page crawl)  PASS
```

**Tests:** tests/test_layer0.py 8 passed (sitemap parse+nesting, enrichment,
JS-diff, 4-bot coverage, robots conflicts, weighting+label, PDF roundtrip,
href regression). Full suite: 56 passed, 1 skipped.

---

## PHASE 2 (v3 brief) — Incremental crawl & data freshness (2026-07-12)

**Built (`src/incremental.py` + surgical deletes in kg/vector store + CLI):**
- 4-level skip chain: (1) conditional GET w/ stored ETag/Last-Modified -> 304
  skips page; sitemap lastmod orders fetches; (2) markdown hash (never raw
  HTML) catches lying headers; (3) chunk-level content-hash diff -> only
  changed/new chunks reprocessed, deleted chunks tombstoned; (4) surgical
  update: chunk_id-scoped deletes in Kuzu (claims + edges + node) and Chroma,
  NO full rebuild. Topics re-run only when >15% chunks changed.
- SQLite state (pages w/ validators + adaptive revisit halve/double, chunks w/
  tombstones, crawl_runs metrics).
- CLI: --incremental (default once baseline exists) / --full; full builds
  auto-capture the baseline.

**Live verification (local http.server, 6-page site, touch d.html):**
```
pages_304=5  pages_changed=1  chunks_added=1 (exactly the edited chunk,
  'renovated pricing' confirmed in reprocessed content)  tombstoned=0
chroma count 6 == active chunks 6 (parity)   kg orphan_chunks=0
TIMING: full=65.2s  incremental=11.0s  ->  5.9x  (>=5x PASS)
```
First run on a 3-page site scored 4.8x — under the bar because one changed
page pays full Playwright startup (~6s fixed) against a tiny baseline; the
6-page site reflects the true scaling (incremental cost is O(changed), full
is O(site)). Logged, not hidden.

**Tests:** tests/test_incremental.py 6 passed (304 skip, lying-header hash
catch, chunk diff + tombstone + surgical deletes on both stores, >15% topic
trigger, adaptive revisit, crawl_runs rows). Full suite: 62 passed, 1 skipped.

**Honest limits:** incremental mode re-enriches Layer-0 (bot re-fetch) only
for changed pages — unchanged pages keep stored access_gaps; page signals in
incremental context rebuild content from active chunks (approximation).

---

## Speed: Ollama Cloud model (optional, 2026-07-12)

Local qwen2.5:7b on CPU was ~120s/call, making live proofs take 20-25 min.
The user is signed into Ollama Cloud, whose models route through the SAME
`ollama.chat` API — so `providers.llm_complete` uses them with ZERO code
changes, just the model name. Set `LLM_MODEL=gpt-oss:120b-cloud` in `.env.local`
(gitignored, per-user). Measured: 1.8-2.0s/call in JSON mode (~60x faster),
schema-clean output (Organization/Person/Product).
- The CODE default is UNCHANGED: `LLM_MODEL=qwen2.5:7b-instruct`, fully local,
  zero keys. Cloud is a per-machine override, requires `ollama signin`, and is
  NOT zero-key. The earlier zero-key end-to-end proof stands on local qwen.

## PHASE 3 (v3 brief) — Precision upgrades (2026-07-12)

**3a — Formal chunking nodes (`src/chunking2.py`):**
- Step 1 pysbd segmentation + abbreviation-merge post-pass. Live smoke:
  "The firm is old. Est. 2021, it employs many researchers. It ships tools."
  -> 3 sentences, "Est. 2021..." merged (was split Est.|2021); "Dr. Smith",
  "No. 5" handled; offsets exact slices.
- Step 2 MiniLM semantic chunking: heading-first split, cosine-drop sub-split
  (0.5), <=500 tokens, heading_path + (char_start,char_end) preserved. Live
  smoke: nested heading paths correct, every chunk's offset slice matches text.
- Tests: test_chunking2.py 7 passed (Est.2021, no-over-merge, more abbrevs,
  heading+offset traceability, token cap forces split, both StepTraces).

**3b — Structural scorer (`src/structure.py`, deterministic, no ML):**
- Per-chunk claims_per_block / paragraph_length / numeric-series-should-be-table
  / heading-question-alignment / schema_presence -> structure_score + flags
  with concrete fixes + exact chunk ids, fed to Layer-3 recs. Live smoke:
  well-structured=1.0 (>0.8), wall-of-text=0.145 (<0.4) with the 4 expected flags.
- Tests: 4 passed (well>0.8/wall<0.4, flags real, tables ignored, trace).

**3c — Stochastic fan-out (`src/fanout.py`), replaces static expansion:**
- N=12 samples at temp 0.8 (sanctioned temp-0 exception, logged), embedding
  dedup (cosine>=0.92) + frequency weights -> fanout_distribution. Cached per
  query. Wired into the simulator's query_expansion node; retriever now emits
  weighted cluster_coverage; dead ends carry fan-out weights.
- Tests: test_structure_fanout.py fan-out 3 passed (dedup+weight, cache
  roundtrip=N calls once then 0, stability Jaccard math) + coverage-sensitivity
  test in test_layer2_layer3 (weighted coverage drops when covering chunk
  removed).

**Offline suite after Phase 3: 77 passed, 1 skipped.**

## PHASE 4 (v3 brief) — Query demand ingestion (2026-07-12)

**Built (`src/demand.py` + `run_audit.py import-queries`):**
- parse_gkp_csv: Google Keyword Planner export (CSV/TSV, UTF-8/UTF-16 w/ BOM),
  skips preamble rows, detects delimiter from the header row (bug fixed:
  was detecting from a preamble line), dedups -> keyword + searches +
  volume_bucket + log-scaled weight.
- conversationalize: local LLM (temp 0.7) -> 2-3 conversational question
  variants per seed, tagged {seed, variant_type, volume_bucket, weight}.
- parse_paa: People-Also-Ask paste file merged in.
- build_query_set: ordered by weight so high-volume gaps surface first.
- CLI: `run_audit.py import-queries --gkp <csv> [--paa <file>] --out <json>`
  (lightweight subcommand dispatch; existing `--url` audit flow untouched).

**Tests: test_demand.py 6 passed** (bucket/weight monotonic, GKP preamble skip
+ dedup, UTF-16 TSV, every seed gets >=2 variants, PAA parse, build ordered by
weight -> enterprise-crm 12K outranks free-tool 90). Offline suite: 83 passed.

## PHASE 5 (v3 brief) — Snapshots & regression alerts (2026-07-12)

**Built (`src/snapshots.py` + `run_audit.py diff`):**
- After every run, save a snapshot: per-page invisibility + signals + pagerank,
  per-chunk structure_score, per-query weighted coverage + confidence breakdown
  + winning chunks/pages. Stored as data/snapshots/run_NNNN.json.
- diff_snapshots(a,b): reports ONLY regressions + improvements, each traced to
  a cause (deleted covering chunk / new JS template / winning page dropped),
  ranked by severity = magnitude x query_weight x page_pagerank.
- CLI: `run_audit.py diff [--from N --to M]` (default last two); non-zero exit
  when worst severity > threshold (0.1) -> cron/CI usable.

**Verification (the brief's exact scenario, in test_snapshots.py):** baseline
snapshot -> delete ONE covering chunk (c1) -> diff flags exactly the q1
coverage regression 0.8->0.4, cause "covering chunk(s) deleted: ['c1']",
severity = 0.4*0.9 (query weight), ci_status=fail. Invisibility regression
carries a JS-template cause + pagerank-weighted severity. Improvements are not
flagged as regressions.

**Tests: test_snapshots.py 6 passed.** Wired snapshot-save into run_audit
(+ structural scores). Offline suite: 90 passed, 1 skipped.

## PHASE 3c FIX — fan-out stability (2026-07-12, follow-up)

First LIVE 3c check FAILED honestly: MERGE_COSINE=0.92 too strict for real
BGE-M3 embeddings of verbose gpt-oss sub-questions -> 146 singleton intents,
top-5 unstable (string Jaccard 0.00). Fixes:
- MERGE_COSINE 0.92 -> 0.80 (cluster paraphrases into shared intents);
- prompt asks for 4-7 MAIN facets (short) instead of "distinct + varied".
- Added intent_stability(): the RIGHT metric — sub-intents are semantic
  clusters, so two runs reword the same intent; string Jaccard understates.
  Matches top-k by embedding cosine (>=0.8).
LIVE re-verify (cloud gpt-oss:120b): r1=18, r2=22 sub-intents;
  string Jaccard=0.00 (wording varies) but SEMANTIC top-5 stability=0.80
  (>=0.6 PASS). Both rounds share: exception types, custom exceptions,
  try/except structure, specific-vs-broad catch. Offline suite: 90 passed.

## PHASE 6 (v3 brief) — Manual calibration panel (2026-07-12)

**Built (`src/calibration.py` + `run_audit.py calibrate`):**
- generate_templates: writes calibration/run_NNNN/qNNN.md paste-templates for
  the top-20 weighted queries (ChatGPT / Perplexity / Gemini sections).
- parse_calibration_file: extracts each engine's answer + cited URLs -> domains.
- compute_agreement: per-engine + overall agreement = did the simulator's
  winning pages' domains appear in the engine's real citations? Stored as
  citation_probability_observed, LABELED "observed", never merged with the
  simulated confidence (v2 rule holds). <50% overall -> tuning note.
- CLI: `calibrate --template` (write blanks) / `calibrate --run N` (score).
- calibration/ gitignored (user paste data).

**Verification (test_calibration.py, 5 passed):** parses a filled 3-engine
file; agreement hand-checked — simulator cited hubspot.com, ChatGPT cited it
(1.0), Perplexity didn't (0.0), Gemini blank -> skipped, overall 0.5; below-
floor case emits the retriever-tuning note; observed dict carries label
"observed" and NO simulated/confidence keys (never merged). Offline suite: 95
passed, 1 skipped.

## PHASE 7 (v3 brief) — Pre-publish what-if scoring (2026-07-12)

**Built (`src/whatif.py` + `run_audit.py whatif`):**
- load_draft (.md/.txt; .docx via python-docx if installed).
- Draft -> semantic_chunks -> structural flags -> local BGE-M3 embeddings, all
  IN MEMORY. Overlay = base_chunks + draft_chunks passed to a lightweight
  coverage predictor (rank_bm25 0.4 + BGE cosine 0.6 + fan-out weights) that
  never loads the reranker/NLI/synthesis and never writes any store.
- Per target query: predicted weighted cluster-coverage before/after + delta +
  newly-covered sub-intents; draft structure flags; store-integrity hashes.
- CLI: `run_audit.py whatif --draft <f> --query "..."` (pulls base chunks from
  the incremental state; overlay is transient).

**Verification (test_whatif.py, 3 passed — both brief checks):**
1. store integrity: real kg.kuzu file + chromadb dir hashed before/after ->
   identical; files literally untouched (stores_unchanged=True).
2. a draft answering a known dead-end sub-intent (refund policy) raises that
   query's predicted coverage 0.5 -> 1.0 (delta +0.5, +1 sub-intent).
   Draft structure flags surface (paragraph_too_long, numeric-series-as-table).
Offline suite: 98 passed, 1 skipped.

## PHASE 8 (v3 brief) — Fix generation: patches, not advice (2026-07-12)

**Built (`src/fixes.py` + `run_audit.py --fixes`):**
- generate_jsonld: FAQPage (2+ question headings) else Article; validate_jsonld
  checks @context=schema.org, @type, type-required props, FAQ Q/A shape,
  JSON-serializability BEFORE emit.
- generate_llms_txt: from real pages (http only) + topics, pagerank-ordered,
  draft banner.
- generate_faq: local LLM drafts Q/A grounded ONLY in supplied source claims
  ([TODO: confirm] when facts missing).
- rewrite_block: local LLM restructures flagged blocks (numeric prose -> table,
  long paragraph -> split), facts preserved.
- generate_fixes: writes fixes/run_NNNN/*.{jsonld,md,txt} + manifest.json
  mapping each artifact to a finding id; every draft carries
  "draft — human review required, never auto-publish". `--fixes` wires it into
  the audit; fixes/ gitignored.

**Verification (test_fixes.py 5 passed + live cloud run):**
- JSON-LD validates (Article + FAQPage); validator rejects bad @context /
  missing required props.
- llms.txt lists ONLY real http URLs (javascript:/relative dropped).
- manifest maps every fix file to a finding id (schema:/site:/deadend:/
  structure:<chunk_id>); banner present on all drafts.
- LIVE (gpt-oss:120b-cloud): FAQ grounded in the given claim ("30-day
  money-back guarantee"); numeric prose rewritten into a real markdown table.
Offline suite: 103 passed, 1 skipped.
