"""
v5 Page-Level Optimization Mapper — resolves chain nodes to concrete page URLs
and decides edit-vs-create with suggested slugs and internal link targets.
"""
from typing import Optional
from dataclasses import dataclass, field
from src.chunk.chunking import Chunk
from src.vector.store import VectorStore
from src.graph.kuzu_graph import get_graph
from src.core.logging import get_logger

logger = get_logger(__name__)


def _strip_boilerplate_chunks(chunks: list, all_chunks: list) -> list:
    """Filter to keep only non-boilerplate chunks (content > 200 chars, not nav/footer)."""
    return [c for c in chunks if len(c.content) > 200]


@dataclass
class PageResolution:
    """One chain node resolved to its supporting page URL(s)."""
    entity_name: str
    status: str  # "found" | "missing"
    supporting_pages: list[str] = field(default_factory=list)
    chunk_count: int = 0


@dataclass
class PageOptimizationResult:
    """Full resolution output for one query."""
    chain_resolutions: list[PageResolution] = field(default_factory=list)
    recommendation_type: str = ""  # "edit_existing_page" | "create_new_page"
    reason: str = ""
    suggested_new_page: Optional[dict] = None
    suggested_edits: list[dict] = field(default_factory=list)


class PageMapper:
    """
    Resolves every entity in the expected chain back to supporting page URLs
    via source_chunk_ids, then applies the decision rule (edit vs. create).
    """

    def __init__(self, chunks: list[Chunk]):
        # Build index -> chunk lookup
        self.chunks = chunks
        self.idx_to_chunk = {i: c for i, c in enumerate(chunks)}

    def resolve_entity(self, entity_name: str, source_chunk_ids_str: str) -> PageResolution:
        """Resolve a single entity's source_chunk_ids to page URLs."""
        urls = []
        has_chunks = False
        if source_chunk_ids_str and source_chunk_ids_str != "None":
            parts = source_chunk_ids_str.split(",")
            has_chunks = len([p for p in parts if p.strip()]) > 0
            for p in parts:
                p = p.strip()
                if p:
                    try:
                        idx = int(p)
                        if idx in self.idx_to_chunk:
                            url = self.idx_to_chunk[idx].url
                            if url and url not in urls:
                                urls.append(url)
                    except ValueError:
                        pass
        # Deduplicate URLs
        unique_urls = list(dict.fromkeys(urls))
        # Status depends on whether entity has ANY source chunks (even if we can't resolve URLs)
        status = "found" if has_chunks else "missing"
        return PageResolution(
            entity_name=entity_name,
            status=status,
            supporting_pages=unique_urls,
            chunk_count=len(parts) if source_chunk_ids_str and source_chunk_ids_str != "None" else 0,
        )

    def resolve_chain(self, entity_chain: list[str], graph=None) -> list[PageResolution]:
        """Resolve a full entity chain to page URLs using graph data."""
        if graph is None:
            graph = get_graph()
        resolutions = []
        for entity in entity_chain:
            # Query graph for source_chunk_ids
            result = graph._conn.execute(
                "MATCH (e:Entity) WHERE e.name = '%s' RETURN e.source_chunk_ids"
                % entity.replace("'", "''")
            )
            df = result.get_as_df()
            source_ids = ""
            if len(df):
                source_ids = str(df.iloc[0].get("e.source_chunk_ids", ""))
            resolution = self.resolve_entity(entity, source_ids)
            resolutions.append(resolution)
        return resolutions

    def decide_edit_or_create(self, resolutions: list[PageResolution]) -> PageOptimizationResult:
        """
        Decision rule (Section 3.2):
        - 3+ missing entities forming a sub-topic -> create_new_page
        - 1-2 missing entities that are narrow -> edit closest page
        - 1 missing relationship between 2 existing entities -> edit more central page
        """
        missing = [r for r in resolutions if r.status == "missing"]
        found = [r for r in resolutions if r.status == "found"]

        result = PageOptimizationResult(chain_resolutions=resolutions)

        if len(missing) >= 3:
            # Whole absent content pillar
            result.recommendation_type = "create_new_page"
            missing_names = [m.entity_name for m in missing]
            result.reason = (
                f"{len(missing)} missing entities forming a whole absent content pillar "
                f"({', '.join(missing_names)}) — too large for a single-page edit"
            )

            # Build suggested new page
            # Use the first found entity's first page as the should_internally_link_to base
            internal_links = []
            for f_ent in found:
                internal_links.extend(f_ent.supporting_pages[:2])

            result.suggested_new_page = {
                "title": " & ".join(missing_names[:3]) + " — Content Guide",
                "type": "blog",
                "suggested_slug": "/blogs/blog/" + "-".join(m.lower().replace(" ", "-") for m in missing_names[:5]),
                "must_establish_entities": missing_names,
                "must_establish_relationships": [
                    {"source": missing_names[i], "target": missing_names[i+1], "type": "relatedTo"}
                    for i in range(len(missing_names) - 1)
                ],
                "should_internally_link_to": list(dict.fromkeys(internal_links)),
            }

        elif len(missing) == 1 and found:
            # Single narrow missing entity -> edit closest existing page
            result.recommendation_type = "edit_existing_page"
            result.reason = (
                f"Single missing entity '{missing[0].entity_name}' — can be added to "
                f"an existing page"
            )
            # Suggest editing the closest found entity's first page
            closest = found[0]
            result.suggested_edits = [{
                "target_url": closest.supporting_pages[0] if closest.supporting_pages else "",
                "action": "New FAQ / New Paragraph",
                "entity_to_add": missing[0].entity_name,
            }]

        elif not missing and found:
            # All entities found — relationship gap
            result.recommendation_type = "edit_existing_page"
            result.reason = "All entities exist — strengthen relationships between existing pages"
            if len(found) >= 2:
                result.suggested_edits = [{
                    "target_url": found[0].supporting_pages[0] if found[0].supporting_pages else "",
                    "action": "New Internal Link",
                    "link_from": found[0].entity_name,
                    "link_to": found[1].entity_name,
                }]

        else:
            # Fallback
            result.recommendation_type = "create_new_page"
            result.reason = f"Partial content coverage ({len(found)} found, {len(missing)} missing)"

        return result
