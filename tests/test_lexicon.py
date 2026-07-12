"""Node (A) — de-branding: the input-layer lexicon is externalized to
config/lexicon.yaml and EMPTY by default.

Proves two things:
1. With the shipped (empty) lexicon, the input layer makes no brand assumptions.
2. The mechanism still works — a populated lexicon re-enables extraction — so
   de-branding removed the coupling, not the capability.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import input_layer as il


def test_default_lexicon_is_empty_and_brand_free():
    lex = il._load_lexicon()  # reads config/lexicon.yaml (empty by default)
    assert lex["synonyms"] == {}
    assert lex["brands"] == {}
    assert lex["product_types"] == {}
    assert lex["ingredients"] == {}
    assert lex["hero_entities"] == []
    assert lex["default_root"] is None

    # A previously brand-coupled query now yields nothing brand-specific.
    nq = il.segment_text("wickedgud noodles price")[0]
    assert nq.detected_entities == []
    assert nq.canonical == "wickedgud noodles price"  # no synonym rewrite
    assert il.parse_expected_chain(nq) == []
    assert il.hero_entities() == set()


def test_populated_lexicon_reenables_extraction(tmp_path, monkeypatch):
    cfg = tmp_path / "lex.yaml"
    cfg.write_text(
        "brands:\n"
        "  acme: Acme\n"
        "product_types:\n"
        "  widget: Widget\n"
        "attributes:\n"
        "  price: Price\n"
        "hero_entities: [widget]\n"
        "default_root: Widget\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LEXICON_PATH", str(cfg))
    lex = il._load_lexicon()
    monkeypatch.setattr(il, "_LEX", lex)

    nq = il.segment_text("acme widget price")[0]
    assert "Acme" in nq.detected_entities
    assert "Widget" in nq.detected_entities
    assert il.hero_entities() == {"widget"}
    # chain reaches >= 2 entities when a lexicon is present
    chain = il.parse_expected_chain(nq)
    assert len(chain) >= 2
