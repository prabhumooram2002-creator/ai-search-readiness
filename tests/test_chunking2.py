"""Phase 3a — formal chunking nodes (src/chunking2.py)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import chunking2 as ck
from src.trace import Trace


def test_est_2021_abbreviation_merge():
    sents = ck.segment_sentences(
        "The firm is old. Est. 2021, it employs many researchers. It ships tools.")
    texts = [s["text"] for s in sents]
    assert "Est. 2021, it employs many researchers." in texts   # merged
    assert "Est." not in texts and "2021, it employs many researchers." not in texts
    # offsets remain exact slices of the source
    src = "The firm is old. Est. 2021, it employs many researchers. It ships tools."
    for s in sents:
        assert src[s["char_start"]:s["char_end"]].strip() == s["text"]


def test_no_over_merge_of_real_sentences():
    sents = ck.segment_sentences("The cost is low. Sales grew. Profit rose.")
    assert [s["text"] for s in sents] == \
        ["The cost is low.", "Sales grew.", "Profit rose."]


def test_more_abbreviations():
    # "No. 5" and "vol. 3" continuations merge; a normal sentence boundary stays
    sents = ck.segment_sentences("It is item No. 5, in stock. Ships today.")
    assert sents[0]["text"] == "It is item No. 5, in stock."
    assert sents[1]["text"] == "Ships today."


def test_semantic_chunks_heading_path_and_offsets():
    md = ("# Guide\n\nIntro paragraph about the product for teams to read here.\n\n"
          "## Pricing\n\nPricing starts at ten dollars per month for basic tier.\n")
    chunks = ck.semantic_chunks(md, url="http://x/g")
    assert chunks, "expected chunks"
    for c in chunks:
        assert c["id"].startswith("chunk_")
        assert c["page_url"] == "http://x/g"
        assert c["token_estimate"] <= ck.MAX_TOKENS
        # exact traceability: the offset slice contains the chunk's first words
        sl = md[c["char_start"]:c["char_end"]]
        assert c["text"].split()[0] in sl
    paths = [tuple(c["heading_path"]) for c in chunks]
    assert ("Guide",) in paths
    assert ("Guide", "Pricing") in paths     # nested heading path preserved


def test_token_cap_forces_split():
    # one heading, many sentences exceeding the token budget -> multiple chunks
    body = " ".join(f"Sentence number {i} has several words in it for size." * 1
                    for i in range(60))
    md = f"# Big\n\n{body}\n"
    chunks = ck.semantic_chunks(md, url="http://x/big", max_tokens=60)
    assert len(chunks) >= 2
    assert all(c["token_estimate"] <= 60 for c in chunks)


def test_segmentation_emits_steptrace():
    trace = Trace(query="q")
    ck.segment_sentences("One. Two. Three.", "c", trace=trace)
    assert trace.step_names() == ["sentence_segmentation"]
    assert trace.steps[0].scores["n_sentences"] == 3.0


def test_semantic_chunking_emits_steptrace():
    trace = Trace(query="q")
    ck.semantic_chunks("# H\n\nA paragraph with enough words to be a chunk here.\n",
                       url="http://x", trace=trace)
    assert trace.step_names() == ["semantic_chunking"]
    assert trace.steps[0].outputs["n_chunks"] >= 1
