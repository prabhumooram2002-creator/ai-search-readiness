"""PHASE 12 — Website Intelligence Report (INTELLIGENCE_REPORT_BRIEF.md).

Nineteen sections reconstructing a site's "knowledge model" from data the
pipeline already produces (Layers 0-3 + Phases 4-10). This package is a
REPORTING layer — it reads the KG/stores/traces and derives cheap stats
(networkx, cosine, textstat); new inference only where a module's own
docstring says NEW MODULE.

Every section function returns a JSON-serializable dict; report.py
assembles all of them into intelligence.json and renders intelligence.html.
"""
