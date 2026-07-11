"""Layer 1 step 8 — Knowledge Graph in KùzuDB (CLAUDE.md schema, kept).

Embedded, free, local. Builds the spec's graph from the Layer-1 node outputs:

  Nodes:  Page, Chunk (embedding FLOAT[1024]), Entity (graph_embedding
          FLOAT[128]), Claim, Topic, ExternalDomain, ExternalPage
  Rels:   HasChunk, MentionsEntity, RelatesTo, SupportsClaim, BelongsToTopic,
          InternalLink, ExternalLinksTo

Dimension stays 1024 (BGE-M3). ExternalDomain/ExternalPage/ExternalLinksTo are
created but stay empty until step 10b (Common Crawl enrichment) is built.

Verification rules (enforced by ``verify()``, per CLAUDE.md):
  * node/edge counts per type reported;
  * zero orphan Chunks (every Chunk reachable via HasChunk);
  * every Claim has >= 1 SupportsClaim edge (guaranteed structurally: each
    claim's own source chunk supports it, enriched with step-6 NLI labels).

Emits one StepTrace named ``knowledge_graph`` when a Trace is passed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .core.config import BASE_DIR
from .core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DB = str(BASE_DIR / "data" / "kg.kuzu")

_DDL = [
    # ── Node tables ─────────────────────────────────────────────────────────
    "CREATE NODE TABLE IF NOT EXISTS Page(url STRING, title STRING, PRIMARY KEY(url))",
    "CREATE NODE TABLE IF NOT EXISTS Chunk(id STRING, url STRING, text STRING, "
    "embedding FLOAT[1024], PRIMARY KEY(id))",
    "CREATE NODE TABLE IF NOT EXISTS Entity(id STRING, name STRING, type STRING, "
    "synonyms STRING, graph_embedding FLOAT[128], PRIMARY KEY(id))",
    "CREATE NODE TABLE IF NOT EXISTS Claim(id STRING, text STRING, "
    "source_sentence_id STRING, source_chunk_id STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE IF NOT EXISTS Topic(id STRING, label STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE IF NOT EXISTS ExternalDomain(host STRING, PRIMARY KEY(host))",
    "CREATE NODE TABLE IF NOT EXISTS ExternalPage(url STRING, PRIMARY KEY(url))",
    # ── Rel tables ──────────────────────────────────────────────────────────
    "CREATE REL TABLE IF NOT EXISTS HasChunk(FROM Page TO Chunk)",
    "CREATE REL TABLE IF NOT EXISTS MentionsEntity(FROM Chunk TO Entity, score DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS RelatesTo(FROM Entity TO Entity, "
    "relation STRING, score DOUBLE, source_chunk_id STRING)",
    "CREATE REL TABLE IF NOT EXISTS SupportsClaim(FROM Chunk TO Claim, "
    "entailment_label STRING, entailment_confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS BelongsToTopic(FROM Chunk TO Topic)",
    "CREATE REL TABLE IF NOT EXISTS InternalLink(FROM Page TO Page, anchor_text STRING)",
    "CREATE REL TABLE IF NOT EXISTS ExternalLinksTo(FROM Page TO ExternalPage, "
    "anchor_text STRING, context_snippet STRING, dofollow BOOLEAN, "
    "topical_relevance DOUBLE)",
]


class KnowledgeGraph:
    """Kùzu-backed knowledge graph for one audited site."""

    def __init__(self, db_path: Optional[str] = None):
        import kuzu  # lazy
        path = db_path or DEFAULT_DB
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(path)
        self._conn = kuzu.Connection(self._db)
        for stmt in _DDL:
            self._conn.execute(stmt)

    def close(self) -> None:
        self._conn.close()
        self._db.close()

    # ── helpers ─────────────────────────────────────────────────────────────
    def _exec(self, query: str, params: dict | None = None):
        return self._conn.execute(query, parameters=params or {})

    def _count(self, pattern: str) -> int:
        res = self._exec(f"MATCH {pattern} RETURN count(*) AS c")
        row = res.get_next()
        return int(row[0]) if row else 0

    # ── build ───────────────────────────────────────────────────────────────
    def build(
        self,
        chunks: list[dict],
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
        claims: list[dict] | None = None,
        evidence: list[dict] | None = None,
        topics: list[dict] | None = None,
        pages: list[dict] | None = None,
        embeddings: list[list[float]] | None = None,
        trace=None,
    ) -> dict:
        """Upsert Layer-1 outputs into the graph. Returns ``verify()`` stats.

        chunks:        [{"chunk_id", "content", "url"?}]
        entities:      step-3 records (src.ner) — mentions carry chunk_id+score
        relationships: step-4 triplets (src.relations)
        claims:        step-5 records (src.claims)
        evidence:      step-6 records (src.evidence) — enrich SupportsClaim
        topics:        step-7 records (src.topics)
        pages:         optional [{"url", "title", "internal_links": [...]}]
        embeddings:    optional 1024-dim vector per chunk (step 9)
        """
        step_cm = trace.start_step(
            "knowledge_graph",
            n_chunks=len(chunks), n_entities=len(entities or []),
            n_relationships=len(relationships or []), n_claims=len(claims or []),
            n_topics=len(topics or []),
        ) if trace is not None else None
        st = step_cm.__enter__() if step_cm is not None else None

        try:
            chunk_ids = set()
            # Pages (explicit, or derived from chunk urls)
            page_urls = {p["url"] for p in (pages or [])}
            for c in chunks:
                if c.get("url"):
                    page_urls.add(c["url"])
            for url in page_urls:
                title = next((p.get("title", "") for p in (pages or [])
                              if p["url"] == url), "")
                self._exec("MERGE (p:Page {url: $url}) SET p.title = $title",
                           {"url": url, "title": title})

            # Chunks + HasChunk (+ optional step-9 embeddings)
            for i, c in enumerate(chunks):
                cid = str(c["chunk_id"])
                chunk_ids.add(cid)
                url = c.get("url", "")
                self._exec(
                    "MERGE (c:Chunk {id: $id}) SET c.url = $url, c.text = $text",
                    {"id": cid, "url": url, "text": c["content"][:4000]})
                if embeddings is not None and i < len(embeddings) and embeddings[i]:
                    self._exec(
                        "MATCH (c:Chunk {id: $id}) SET c.embedding = $vec",
                        {"id": cid, "vec": [float(x) for x in embeddings[i]]})
                if url:
                    self._exec(
                        "MATCH (p:Page {url: $url}), (c:Chunk {id: $id}) "
                        "MERGE (p)-[:HasChunk]->(c)", {"url": url, "id": cid})

            # Entities + MentionsEntity
            for e in entities or []:
                self._exec(
                    "MERGE (e:Entity {id: $id}) SET e.name = $name, "
                    "e.type = $type, e.synonyms = $syn",
                    {"id": e["id"], "name": e["name"], "type": e.get("type", ""),
                     "syn": ",".join(e.get("synonyms", []))})
                for m in e.get("mentions", []) or [
                        {"chunk_id": cid, "score": e.get("score", 0.0)}
                        for cid in e.get("source_chunk_ids", [])]:
                    if str(m["chunk_id"]) not in chunk_ids:
                        if st is not None:
                            st.drop({"entity": e["id"], "chunk": m["chunk_id"]},
                                    "mention chunk not in graph")
                        continue
                    self._exec(
                        "MATCH (c:Chunk {id: $cid}), (e:Entity {id: $eid}) "
                        "MERGE (c)-[r:MentionsEntity]->(e) SET r.score = $s",
                        {"cid": str(m["chunk_id"]), "eid": e["id"],
                         "s": float(m.get("score", 0.0))})

            # RelatesTo (step-4 triplets)
            for t in relationships or []:
                self._exec(
                    "MATCH (a:Entity {id: $s}), (b:Entity {id: $o}) "
                    "MERGE (a)-[r:RelatesTo {relation: $rel}]->(b) "
                    "SET r.score = $score, r.source_chunk_id = $chunk",
                    {"s": t["subject_entity_id"], "o": t["object_entity_id"],
                     "rel": t["relation"], "score": float(t.get("score", 0.0)),
                     "chunk": str(t.get("source_chunk_id", ""))})

            # Claims + SupportsClaim (own source chunk always supports;
            # step-6 evidence enriches the edge with the NLI verdict)
            ev_by_claim = {e["claim_id"]: e for e in (evidence or [])}
            for cl in claims or []:
                self._exec(
                    "MERGE (cl:Claim {id: $id}) SET cl.text = $text, "
                    "cl.source_sentence_id = $sid, cl.source_chunk_id = $cid",
                    {"id": cl["id"], "text": cl["text"],
                     "sid": cl["source_sentence_id"],
                     "cid": str(cl["source_chunk_id"])})
                ev = ev_by_claim.get(cl["id"], {})
                if str(cl["source_chunk_id"]) in chunk_ids:
                    self._exec(
                        "MATCH (c:Chunk {id: $cid}), (cl:Claim {id: $clid}) "
                        "MERGE (c)-[r:SupportsClaim]->(cl) "
                        "SET r.entailment_label = $lbl, r.entailment_confidence = $conf",
                        {"cid": str(cl["source_chunk_id"]), "clid": cl["id"],
                         "lbl": ev.get("entailment_label", "unscored"),
                         "conf": float(ev.get("entailment_confidence", 0.0))})
                elif st is not None:
                    st.drop({"claim": cl["id"]}, "claim source chunk not in graph")

            # Topics + BelongsToTopic (step 7)
            for tp in topics or []:
                self._exec("MERGE (t:Topic {id: $id}) SET t.label = $label",
                           {"id": tp["id"], "label": tp["label"]})
                for cid in tp.get("chunk_ids", []):
                    if str(cid) not in chunk_ids:
                        continue
                    self._exec(
                        "MATCH (c:Chunk {id: $cid}), (t:Topic {id: $tid}) "
                        "MERGE (c)-[:BelongsToTopic]->(t)",
                        {"cid": str(cid), "tid": tp["id"]})

            # InternalLink (from crawl records, when provided)
            for p in pages or []:
                for link in p.get("internal_links", []) or []:
                    dest = link.get("dest_url", "")
                    if dest not in page_urls:
                        continue
                    self._exec(
                        "MATCH (a:Page {url: $src}), (b:Page {url: $dst}) "
                        "MERGE (a)-[l:InternalLink]->(b) SET l.anchor_text = $txt",
                        {"src": p["url"], "dst": dest,
                         "txt": link.get("anchor_text", "")})

            stats = self.verify()
            logger.info(f"KG build: {stats}")
            if st is not None:
                st.outputs["stats"] = stats
                st.scores["n_nodes"] = float(sum(
                    stats["nodes"].values()))  # heuristic
                st.scores["orphan_chunks"] = float(stats["orphan_chunks"])
                st.scores["claims_without_support"] = float(
                    stats["claims_without_support"])
            return stats
        except BaseException as exc:
            if step_cm is not None:
                step_cm.__exit__(type(exc), exc, exc.__traceback__)
                step_cm = None
            raise
        finally:
            if step_cm is not None:
                step_cm.__exit__(None, None, None)

    # ── verification (CLAUDE.md step-8 rules) ───────────────────────────────
    def verify(self) -> dict:
        nodes = {t: self._count(f"(n:{t})")
                 for t in ["Page", "Chunk", "Entity", "Claim", "Topic"]}
        rels = {r: self._count(f"()-[e:{r}]->()")
                for r in ["HasChunk", "MentionsEntity", "RelatesTo",
                          "SupportsClaim", "BelongsToTopic", "InternalLink"]}
        orphan = self._count("(c:Chunk) WHERE NOT EXISTS "
                             "{ MATCH (:Page)-[:HasChunk]->(c) }")
        no_support = self._count("(cl:Claim) WHERE NOT EXISTS "
                                 "{ MATCH (:Chunk)-[:SupportsClaim]->(cl) }")
        return {
            "nodes": nodes,
            "rels": rels,
            "orphan_chunks": orphan,
            "claims_without_support": no_support,
        }

    # ── traversal used by Layer 2 ───────────────────────────────────────────
    def two_hop_entities(self, entity_ids: list[str]) -> list[dict]:
        """RelatesTo/MentionsEntity 2-hop neighborhood -> supporting chunks."""
        out = []
        for eid in entity_ids:
            res = self._exec(
                "MATCH (a:Entity {id: $id})-[:RelatesTo*1..2]-(b:Entity)"
                "<-[:MentionsEntity]-(c:Chunk) "
                "RETURN DISTINCT b.id, b.name, c.id LIMIT 50", {"id": eid})
            while res.has_next():
                row = res.get_next()
                out.append({"via_entity_id": row[0], "via_entity_name": row[1],
                            "chunk_id": row[2], "seed_entity_id": eid})
        return out
