"""Kuzu graph DB — free, local, zero infrastructure.
   
   Node table: Entity(id PK, name, type, synonyms, source_chunk_ids)
   Rel table:  HasRelationship(FROM Entity TO Entity, rel_type, evidence_chunk_id, confidence)
"""
import os
from typing import Optional
from ..core.config import BASE_DIR

import kuzu


def _db_path():
    path = BASE_DIR / "data" / "kuzu_graph.kuzu"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


class GraphClient:
    """Sync Kuzu client. Stores data in data/kuzu_graph.kuzu."""

    def __init__(self, db_path: Optional[str] = None):
        path = db_path or _db_path()
        self._db = kuzu.Database(path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            "CREATE NODE TABLE IF NOT EXISTS Entity("
            "id STRING, name STRING, type STRING, synonyms STRING, "
            "source_chunk_ids STRING, PRIMARY KEY(id))"
        )
        self._conn.execute(
            "CREATE REL TABLE IF NOT EXISTS HasRelationship("
            "FROM Entity TO Entity, "
            "rel_type STRING, evidence_chunk_id STRING, confidence DOUBLE)"
        )

    def verify_connection(self) -> None:
        self._conn.execute("RETURN 1")

    def get_stats(self) -> dict:
        n = self._conn.execute("MATCH (e:Entity) RETURN count(e) AS cnt").get_as_df()
        r = self._conn.execute(
            "MATCH ()-[r:HasRelationship]->() RETURN count(r) AS cnt"
        ).get_as_df()
        return {
            "nodes": int(n.iloc[0]["cnt"]) if len(n) else 0,
            "relationships": int(r.iloc[0]["cnt"]) if len(r) else 0,
        }

    def upsert_entities(self, entities: list[dict]) -> None:
        for e in entities:
            eid = e["id"].replace("'", "''")
            name = e["name"].replace("'", "''")
            etype = e.get("type", "Topic").replace("'", "''")
            synonyms = ",".join(e.get("synonyms", [])).replace("'", "''")
            chunks = ",".join(e.get("source_chunk_ids", [])).replace("'", "''")

            cypher = (
                "MERGE (ent:Entity {id: '%s'}) "
                "SET ent.name = '%s', ent.type = '%s', "
                "ent.synonyms = '%s', ent.source_chunk_ids = '%s'"
                % (eid, name, etype, synonyms, chunks)
            )
            try:
                self._conn.execute(cypher)
            except Exception:
                pass  # Skip on any error

    def upsert_relationships(self, relationships: list[dict]) -> None:
        for rel in relationships:
            src = rel["source_id"].replace("'", "''")
            tgt = rel["target_id"].replace("'", "''")
            rtype = rel["type"].upper().replace("'", "''")
            evidence = rel.get("evidence_chunk_id", "").replace("'", "''")
            confidence = float(rel.get("confidence", 0.5))

            cypher = (
                "MATCH (s:Entity), (t:Entity) "
                "WHERE s.id = '%s' AND t.id = '%s' "
                "CREATE (s)-[:HasRelationship {"
                "rel_type: '%s', evidence_chunk_id: '%s', confidence: %s"
                "}]->(t)"
                % (src, tgt, rtype, evidence, confidence)
            )
            try:
                self._conn.execute(cypher)
            except Exception:
                pass

    def find_paths(self, entity1_name: str, entity2_name: str, max_hops: int = 3) -> list[dict]:
        n1 = entity1_name.replace("'", "''")
        n2 = entity2_name.replace("'", "''")
        cypher = (
            "MATCH p = (a:Entity)-[:HasRelationship*1..%d]->(b:Entity) "
            "WHERE a.name = '%s' AND b.name = '%s' RETURN p LIMIT 5"
            % (max_hops, n1, n2)
        )
        try:
            result = self._conn.execute(cypher)
            return [{"path": str(row["p"])} for _, row in result.get_as_df().iterrows()]
        except Exception:
            return []

    def query_entities_by_type(self, entity_type: str) -> list[dict]:
        etype = entity_type.replace("'", "''")
        try:
            result = self._conn.execute(
                "MATCH (e:Entity) WHERE e.type = '%s' RETURN e.id, e.name" % etype
            )
            return result.get_as_df().to_dict("records")
        except Exception:
            return []

    def query_node(self, name: str) -> Optional[dict]:
        """Check if a node exists by name. Also checks synonyms. Returns node data or None."""
        safe = name.replace("'", "''")
        try:
            # Exact match first
            result = self._conn.execute(
                "MATCH (e:Entity) WHERE e.name = '%s' RETURN e.id, e.name, e.type, e.synonyms LIMIT 1" % safe
            )
            df = result.get_as_df()
            if len(df):
                row = df.iloc[0]
                return {
                    "id": row["e.id"],
                    "name": row["e.name"],
                    "type": row.get("e.type", ""),
                    "schema_type": row.get("e.type", ""),
                }
            # Fallback: check CONTAINS and synonyms
            result = self._conn.execute(
                "MATCH (e:Entity) WHERE CONTAINS(lower(e.synonyms), '%s') "
                "OR CONTAINS(lower(e.name), '%s') "
                "RETURN e.id, e.name, e.type, e.synonyms LIMIT 1"
                % (safe.lower(), safe.lower())
            )
            df = result.get_as_df()
            if len(df):
                row = df.iloc[0]
                return {
                    "id": row["e.id"],
                    "name": row["e.name"],
                    "type": row.get("e.type", ""),
                    "schema_type": row.get("e.type", ""),
                }
        except Exception:
            pass
        return None

    def query_related(self, entity_name: str, rel_type: Optional[str] = None) -> list[dict]:
        name = entity_name.replace("'", "''")
        if rel_type:
            rtype = rel_type.upper().replace("'", "''")
            cypher = (
                "MATCH (e:Entity)-[r:HasRelationship]->(t:Entity) "
                "WHERE e.name = '%s' AND r.rel_type = '%s' "
                "RETURN t.name AS name, t.type AS type, r.rel_type AS rel_type, r.confidence AS confidence"
                % (name, rtype)
            )
        else:
            cypher = (
                "MATCH (e:Entity)-[r:HasRelationship]->(t:Entity) "
                "WHERE e.name = '%s' "
                "RETURN t.name AS name, t.type AS type, r.rel_type AS rel_type, r.confidence AS confidence"
                % name
            )
        try:
            result = self._conn.execute(cypher)
            return result.get_as_df().to_dict("records")
        except Exception:
            return []

    def close(self) -> None:
        self._conn.close()


# ── Singleton ──────────────────────────────────────────────────────────────

_graph_client: Optional[GraphClient] = None


def get_graph() -> GraphClient:
    global _graph_client
    if _graph_client is None:
        _graph_client = GraphClient()
    return _graph_client


def reset_graph() -> None:
    global _graph_client
    if _graph_client is not None:
        try:
            _graph_client.close()
        except Exception:
            pass
        _graph_client = None
    path = _db_path()
    if os.path.exists(path):
        os.remove(path)