"""Section 1 — Website Identity (src/identity.py)."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src import identity as idm


def test_extract_identity_blocks_flattens_graph_wrapper():
    pages = [{"url": "u1", "schema_jsonld": [
        {"@graph": [{"@type": "Organization", "name": "Acme"},
                   {"@type": "WebPage", "name": "irrelevant"}]}]}]
    blocks = idm.extract_identity_blocks(pages)
    assert len(blocks) == 1
    assert blocks[0]["block"]["name"] == "Acme"


def test_contact_signals_finds_email_and_phone():
    pages = [{"url": "u1", "content": "Contact us at hello@acme.com or 555-123-4567"}]
    out = idm.contact_signals(pages)
    assert out["u1"]["emails"] == ["hello@acme.com"]
    assert out["u1"]["phones"]


def test_brand_aliases_flags_ambiguity_with_multiple_forms():
    entities = [{"name": "Acme", "type": "Organization"},
               {"name": "Acme Inc", "type": "Organization"},
               {"name": "Widget", "type": "Product"}]
    out = idm.brand_aliases(entities)
    assert out["n_forms"] == 2
    assert out["ambiguous"] is True


def test_brand_aliases_single_form_not_ambiguous():
    entities = [{"name": "Acme", "type": "Organization"}]
    out = idm.brand_aliases(entities)
    assert out["ambiguous"] is False


def test_build_identity_report_flags_conflict_without_sameas():
    pages = [{"url": "u1", "schema_jsonld": [{"@type": "Organization", "name": "Acme"}],
             "content": "hello@acme.com"},
            {"url": "u2", "schema_jsonld": [], "content": "different@acme.com"}]
    entities = [{"name": "Acme", "type": "Organization"},
               {"name": "Acme Inc", "type": "Organization"}]
    out = idm.build_identity_report(pages, entities)
    assert any("distinct brand names" in c for c in out["conflicts"])
    assert out["ai_brand_awareness"]["known"] == "unknown — run a calibration panel"


def test_build_identity_report_uses_calibration_when_present():
    out = idm.build_identity_report([], [], calibration={"brand_known": True})
    assert out["ai_brand_awareness"]["known"] is True
    assert out["ai_brand_awareness"]["label"] == "observed"
