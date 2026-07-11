"""ChromaDB vector store — persists embeddings, supports similarity search."""
import chromadb
from chromadb.config import Settings
from pathlib import Path
from dataclasses import dataclass
from ..core.config import vector, PERSIST_DIR, CFG
from ..core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class SearchResult:
    chunk_id: str
    url: str
    title: str
    content: str
    distance: float
    index: int


class VectorStore:
    """ChromaDB-backed vector store with collection management."""
    
    def __init__(self, collection_name: str = None):
        cfg = vector()
        self.collection_name = collection_name or cfg.get("collection_name", "ai_search_chunks")
        self.persist_dir = PERSIST_DIR
        
        # Init ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AI Search Readiness Platform chunks"},
        )
        
        logger.info(f"VectorStore ready: collection={self.collection_name}, "
                    f"count={self.collection.count()}, persist_dir={self.persist_dir}")
    
    def add_chunks(self, chunks: list, embeddings: list[list[float]]) -> None:
        """
        Add chunk texts + embeddings to the collection.
        IDs are chunk_ids (content-hash based, stable for re-runs).
        """
        if not chunks or not embeddings:
            return
        
        ids = [c.chunk_id for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {
                "url": c.url,
                "title": c.title or "",
                "chunk_index": c.index,
            }
            for c in chunks
        ]
        
        # Check which IDs already exist (idempotent — skip if already stored)
        existing = self.collection.get(ids=ids)
        existing_ids = set(existing["ids"])
        
        new_ids = []
        new_docs = []
        new_metas = []
        new_embs = []
        
        for i, chunk_id in enumerate(ids):
            if chunk_id not in existing_ids:
                new_ids.append(chunk_id)
                new_docs.append(documents[i])
                new_metas.append(metadatas[i])
                new_embs.append(embeddings[i])
        
        if new_ids:
            self.collection.add(
                ids=new_ids,
                documents=new_docs,
                metadatas=new_metas,
                embeddings=new_embs,
            )
            logger.info(f"Added {len(new_ids)} new chunks to vector store "
                        f"(skipped {len(ids) - len(new_ids)} already-stored)")
        else:
            logger.info("All chunks already in vector store — skipping")
    
    def delete_ids(self, chunk_ids: list[str]) -> None:
        """Surgical delete (Phase 2 incremental): remove specific chunk vectors."""
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"Vector store: deleted {len(chunk_ids)} chunk vectors")

    def search(self, query_embedding: list[float], top_k: int = None) -> list[SearchResult]:
        """Vector similarity search — returns top_k closest chunks."""
        cfg = vector()
        k = top_k or cfg.get("top_k", 5)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        
        search_results: list[SearchResult] = []
        for i in range(len(results["ids"][0])):
            search_results.append(SearchResult(
                chunk_id=results["ids"][0][i],
                url=results["metadatas"][0][i].get("url", ""),
                title=results["metadatas"][0][i].get("title", ""),
                content=results["documents"][0][i],
                distance=results["distances"][0][i],
                index=i,
            ))
        
        return search_results
    
    def count(self) -> int:
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all chunks from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)
        logger.info("Vector store cleared")
    
    def reset(self) -> None:
        """Delete and recreate — use after switching sites."""
        self.clear()