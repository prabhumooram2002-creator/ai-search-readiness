"""Neo4j Aura client — connects to cloud graph DB, handles schema creation."""
import json
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import ServiceUnavailable, AuthError

from ..core.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str = "neo4j"
    connect_timeout: float = 30.0
    max_connection_lifetime: float = 3600.0


class Neo4jClient:
    """
    Async Neo4j Aura client with idempotent schema init and connection verification.
    
    Handles Aura's TLS requirements automatically — driver negotiates TLS on
    neo4j+s:// URIs without extra config in neo4j>=5.
    
    Connection failures raise RuntimeError with a clear message (pipeline handles this
    gracefully and continues without Neo4j if unavailable).
    """

    def __init__(self, config: Neo4jConfig = None):
        if config is None:
            self.cfg = Neo4jConfig(
                uri=NEO4J_URI or "neo4j+s://localhost",
                user=NEO4J_USER or "neo4j",
                password=NEO4J_PASSWORD or "",
                database=NEO4J_DATABASE or "neo4j",
            )
        else:
            self.cfg = config
        
        self._driver: Optional[AsyncDriver] = None

    async def _get_driver(self) -> AsyncDriver:
        """Lazily create driver, caching it for reuse."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self.cfg.uri,
                auth=(self.cfg.user, self.cfg.password),
                max_connection_lifetime=self.cfg.max_connection_lifetime,
            )
        return self._driver

    async def verify_connection(self) -> None:
        """
        Verify connectivity. Raises RuntimeError with clear message on failure.
        The pipeline catches this and continues without Neo4j.
        """
        try:
            driver = await self._get_driver()
            async with driver.session(database=self.cfg.database) as session:
                # Test with a simple query
                result = await session.run("RETURN 1 AS n")
                val = await result.single()
                if val is None or val[0] != 1:
                    raise RuntimeError("Unexpected response from Neo4j")
        except AuthError as e:
            raise RuntimeError(
                f"Neo4j authentication failed. Check NEO4J_USER/NEO4J_PASSWORD. {e}"
            )
        except ServiceUnavailable as e:
            # Aura-specific: instance might still be starting or IP not whitelisted
            msg = str(e)
            if "routing information" in msg.lower() or "connection failed" in msg.lower():
                raise RuntimeError(
                    f"Cannot connect to Neo4j Aura at {self.cfg.uri}. "
                    f"Instance may still be starting (wait 60s) or requires IP whitelist. "
                    f"From your Aura console, check the instance status and ensure your current "
                    f"IP is allowed. Full error: {e}"
                )
            raise RuntimeError(f"Neo4j unavailable at {self.cfg.uri}: {e}")
        except Exception as e:
            raise RuntimeError(f"Neo4j connection error at {self.cfg.uri}: {e}")

    async def init_schema(self) -> None:
        """Create constraints and indexes — idempotent."""
        driver = await self._get_driver()
        async with driver.session(database=self.cfg.database) as session:
            # Entity constraint
            await session.run("""
                CREATE CONSTRAINT entity_id IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """)
            # Entity name index
            await session.run("""
                CREATE INDEX entity_name IF NOT EXISTS
                FOR (e:Entity) ON (e.name)
            """)
            # Relationship confidence index
            await session.run("""
                CREATE INDEX rel_confidence IF NOT EXISTS
                FOR ()-[r]-() ON (r.confidence)
            """)
        logger.info("Neo4j schema initialized")

    async def upsert_entities_and_relationships(
        self,
        entities: list[dict],
        relationships: list[dict],
    ) -> None:
        """
        Upsert all entities and relationships in batch.
        Uses MERGE for idempotency — safe to re-run.
        """
        if not entities and not relationships:
            logger.info("No entities or relationships to upsert")
            return

        driver = await self._get_driver()
        async with driver.session(database=self.cfg.database) as session:

            # ── Upsert entities ────────────────────────────────────────────
            for ent in entities:
                ent_id = ent.get("id") or ent.get("name", "").lower().replace(" ", "-")
                ent_type = ent.get("type", "Topic")
                name = ent.get("name", ent_id)
                synonyms = json.dumps(ent.get("synonyms", []))
                chunk_ids = json.dumps(ent.get("source_chunk_ids", []))

                await session.run(f"""
                    MERGE (e:Entity {{id: $id}})
                    SET e.name = $name,
                        e.type = $type,
                        e.synonyms = $synonyms,
                        e.source_chunk_ids = $chunk_ids
                """,
                    id=ent_id,
                    name=name,
                    type=ent_type,
                    synonyms=synonyms,
                    chunk_ids=chunk_ids,
                )

            logger.info(f"Upserted {len(entities)} entities")

            # ── Upsert relationships ─────────────────────────────────────
            for rel in relationships:
                src_id = rel.get("source_id", "")
                tgt_id = rel.get("target_id", "")
                rel_type = rel.get("type", "relatedTo").upper()
                evidence = rel.get("evidence_chunk_id", "")
                confidence = float(rel.get("confidence", 0.5))

                if not src_id or not tgt_id:
                    continue

                # Build dynamic relationship query (type is a parameter here — safe)
                await session.run(f"""
                    MATCH (src:Entity {{id: $src_id}})
                    MATCH (tgt:Entity {{id: $tgt_id}})
                    MERGE (src)-[r:{rel_type}]->(tgt)
                    SET r.evidence_chunk_id = $evidence,
                        r.confidence = $confidence
                """,
                    src_id=src_id,
                    tgt_id=tgt_id,
                    evidence=evidence,
                    confidence=confidence,
                )

            logger.info(f"Upserted {len(relationships)} relationships")

    async def get_stats(self) -> dict:
        """Return entity/relationship counts + type distributions."""
        driver = await self._get_driver()
        async with driver.session(database=self.cfg.database) as session:
            # Entity count
            ent_result = await session.run("MATCH (e:Entity) RETURN count(e) AS cnt")
            ent_count = (await ent_result.single())[0]

            # Relationship count
            rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            rel_count = (await rel_result.single())[0]

            # Entity type distribution
            ent_types_result = await session.run("""
                MATCH (e:Entity) RETURN e.type AS type, count(e) AS cnt ORDER BY cnt DESC
            """)
            ent_types = {row["type"]: row["cnt"] async for row in ent_types_result}

            # Relationship type distribution
            rel_types_result = await session.run("""
                MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS cnt ORDER BY cnt DESC
            """)
            rel_types = {row["type"]: row["cnt"] async for row in rel_types_result}

            return {
                "entities": ent_count,
                "relationships": rel_count,
                "entity_types": ent_types,
                "relationship_types": rel_types,
            }

    async def get_entity_by_name(self, name: str) -> Optional[dict]:
        """Look up entity by name (fuzzy)."""
        driver = await self._get_driver()
        async with driver.session(database=self.cfg.database) as session:
            result = await session.run(
                "MATCH (e:Entity) WHERE e.name = $name OR $name IN e.synonyms RETURN e",
                name=name,
            )
            row = await result.single()
            return dict(row["e"]) if row else None

    async def find_path(
        self, src_name: str, tgt_name: str, max_hops: int = 2
    ) -> list[dict]:
        """
        Find paths between two entities. Returns list of path dicts.
        Empty list = no path found.
        """
        driver = await self._get_driver()
        async with driver.session(database=self.cfg.database) as session:
            result = await session.run(f"""
                MATCH path = (a:Entity)-[r*1..{max_hops}]-(b:Entity)
                WHERE a.name = $src AND b.name = $tgt
                RETURN path, length(path) AS hops
                ORDER BY hops
                LIMIT 5
            """, src=src_name, tgt=tgt_name)
            
            paths = []
            async for row in result:
                nodes = [dict(n) for n in row["path"].nodes]
                rels = [dict(r) for r in row["path"].relationships]
                paths.append({"nodes": nodes, "relationships": rels, "hops": row["hops"]})
            return paths

    async def clear_all(self) -> None:
        """Delete all nodes and relationships — use with caution."""
        driver = await self._get_driver()
        async with driver.session(database=self.cfg.database) as session:
            await session.run("MATCH (n) DETACH DELETE n")
        logger.warning("Cleared all nodes and relationships from Neo4j")

    async def close(self) -> None:
        """Close driver if open."""
        if self._driver:
            try:
                await self._driver.close()
            except Exception:
                pass
            self._driver = None