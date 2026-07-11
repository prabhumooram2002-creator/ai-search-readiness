"""Layer 1 step 8/9 — Kùzu knowledge graph (src/kg.py). Real kuzu, temp DBs."""
import sys
import shutil
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import pytest
from src.kg import KnowledgeGraph
from src.trace import Trace


@pytest.fixture()
def kg():
    tmp = tempfile.mkdtemp(prefix="kg_test_")
    g = KnowledgeGraph(tmp + "/t.kuzu")
    yield g
    g.close()
    shutil.rmtree(tmp, ignore_errors=True)


CHUNKS = [
    {"chunk_id": "c1", "content": "Acme builds widgets.", "url": "https://a.com/p1"},
    {"chunk_id": "c2", "content": "Widgets are great.", "url": "https://a.com/p2"},
]
ENTS = [
    {"id": "acme", "name": "Acme", "type": "Organization", "synonyms": [],
     "source_chunk_ids": ["c1"], "score": 0.9,
     "mentions": [{"chunk_id": "c1", "score": 0.9}]},
    {"id": "widget", "name": "Widget", "type": "Product", "synonyms": ["widgets"],
     "source_chunk_ids": ["c1", "c2"], "score": 0.8,
     "mentions": [{"chunk_id": "c1", "score": 0.8}, {"chunk_id": "c2", "score": 0.7}]},
]
RELS = [{"subject_entity_id": "acme", "relation": "develops",
         "object_entity_id": "widget", "source_chunk_id": "c1", "score": 0.9}]
CLAIMS = [{"id": "cl1", "text": "Acme builds widgets", "source_sentence_id": "c1:s1",
           "source_chunk_id": "c1"}]
EVID = [{"claim_id": "cl1", "entailment_label": "entailment",
         "entailment_confidence": 0.99}]
TOPICS = [{"id": "topic_0", "label": "Widget manufacturing", "chunk_ids": ["c1", "c2"]}]


def test_full_build_counts_and_invariants(kg):
    trace = Trace(query="q")
    stats = kg.build(CHUNKS, ENTS, RELS, CLAIMS, EVID, TOPICS,
                     embeddings=[[0.1] * 1024, [0.2] * 1024], trace=trace)
    assert stats["nodes"] == {"Page": 2, "Chunk": 2, "Entity": 2, "Claim": 1, "Topic": 1}
    assert stats["rels"]["HasChunk"] == 2
    assert stats["rels"]["MentionsEntity"] == 3
    assert stats["rels"]["RelatesTo"] == 1
    assert stats["rels"]["SupportsClaim"] == 1
    assert stats["rels"]["BelongsToTopic"] == 2
    # CLAUDE.md step-8 verification invariants
    assert stats["orphan_chunks"] == 0
    assert stats["claims_without_support"] == 0
    assert trace.step_names() == ["knowledge_graph"]
    assert trace.steps[0].scores["orphan_chunks"] == 0.0


def test_idempotent_rebuild(kg):
    kg.build(CHUNKS, ENTS, RELS, CLAIMS, EVID, TOPICS)
    stats = kg.build(CHUNKS, ENTS, RELS, CLAIMS, EVID, TOPICS)  # MERGE, no dupes
    assert stats["nodes"]["Entity"] == 2
    assert stats["rels"]["MentionsEntity"] == 3
    assert stats["rels"]["RelatesTo"] == 1


def test_two_hop_traversal(kg):
    kg.build(CHUNKS, ENTS, RELS, CLAIMS, EVID, TOPICS)
    hops = kg.two_hop_entities(["acme"])
    assert any(h["chunk_id"] == "c2" for h in hops)  # acme->widget<-c2


def test_bad_references_dropped_with_reason(kg):
    trace = Trace(query="q")
    bad_ents = [{**ENTS[0], "mentions": [{"chunk_id": "ghost", "score": 0.5}]}]
    bad_claims = [{**CLAIMS[0], "source_chunk_id": "ghost"}]
    stats = kg.build(CHUNKS, bad_ents, [], bad_claims, [], [], trace=trace)
    reasons = {d["reason"] for d in trace.steps[0].dropped}
    assert reasons == {"mention chunk not in graph", "claim source chunk not in graph"}
    # claim node exists but unsupported -> invariant catches it
    assert stats["claims_without_support"] == 1
