"""Q&A pipeline — vector search + LLM answer + Phase 3 hybrid + Phase 4 scoring."""
from dataclasses import dataclass, field
from typing import Optional
from ..core.config import vector, qa as qa_cfg
from ..core.logging import get_logger
from ..embed.embedder import embed_chunks
from ..embed.chat import chat_complete
from ..vector.store import VectorStore, SearchResult
from ..graph.neo4j import Neo4jClient
from ..retrieve.hybrid import broken_path_detect, hybrid_retrieve
from ..score.coverage import compute_coverage_score, generate_recommendations

logger = get_logger(__name__)

@dataclass
class AnswerResult:
    question: str
    answer: str
    sources: list[SearchResult] = field(default_factory=list)
    intent: str = ""
    entities_found: str = ""
    graph_path: str = ""
    missing_relationship: str = ""
    reason: str = ""
    suggested_fix: str = ""
    verdict: str = ""
    scores: dict = None


class QAPipeline:
    """
    Full pipeline: Phase 1 (vector Q&A) + Phase 3 (hybrid) + Phase 4 (scoring).
    Phase 2 (Neo4j) used automatically when available.
    """
    
    def __init__(self, vector_store: VectorStore, neo4j: Neo4jClient = None):
        self.vector_store = vector_store
        self.neo4j = neo4j
    
    def retrieve(self, question_embedding: list[float], top_k: int = None) -> list[SearchResult]:
        return self.vector_store.search(question_embedding, top_k=top_k)
    
    async def embed_question_async(self, question: str) -> list[float]:
        cfg = vector()
        embs = await embed_chunks([question], use_cache=True)
        return embs[0] if embs else []
    
    async def answer_async(self, question: str, sources: list[SearchResult]) -> str:
        if not sources:
            return "No relevant content found in the indexed pages."
        
        context = "\n\n".join(
            f"[Source {i+1} — {s.title or s.url}]: {s.content}"
            for i, s in enumerate(sources[:3])
        )
        
        system_prompt = (
            "You are an AI search auditor. Answer based ONLY on the provided sources. "
            "Cite which source your answer comes from. If uncertain, say you don't know."
        )
        
        try:
            response = await chat_complete(
                prompt=(
                    f"Based ONLY on the sources provided, answer the question.\n"
                    f"Question: {question}\n"
                    f"If the answer is not in the sources, say you don't know."
                ),
                system_prompt=system_prompt,
                context_chunks=[s.content for s in sources[:3]],
            )
            return response
        except Exception as e:
            logger.error(f"LLM answer generation failed: {e}")
            best = sources[0].content[:300] if sources else ""
            return f"LLM generation failed. Best match: {best}..."

    async def ask_async(
        self,
        question: str,
        entity_chain: list[str] = None,
    ) -> AnswerResult:
        """
        Full Q&A with hybrid retrieval, broken path detection, and scoring.
        
        If neo4j is available, performs:
        - Phase 3: Graph path traversal + broken path detection
        - Phase 4: Coverage scoring + recommendations
        """
        logger.info(f"Q&A: {question}")
        
        # ── Phase 1: Embed + Vector Retrieve ────────────────────────────
        q_emb = await self.embed_question_async(question)
        
        if self.neo4j:
            # ── Phase 3: Hybrid retrieval ─────────────────────────────────
            try:
                sources = await hybrid_retrieve(
                    question_embedding=q_emb,
                    neo4j=self.neo4j,
                    vector_store=self.vector_store,
                    top_k=5,
                )
            except Exception as e:
                logger.warning(f"Hybrid retrieval failed: {e}")
                sources = self.retrieve(q_emb)
        else:
            sources = self.retrieve(q_emb)
        
        logger.info(f"Retrieved {len(sources)} chunks")
        
        # ── Generate LLM answer ────────────────────────────────────────────
        answer = await self.answer_async(question, sources)
        
        result = AnswerResult(
            question=question,
            answer=answer,
            sources=sources,
        )
        
        # ── Phase 3: Broken path detection ───────────────────────────────
        if self.neo4j and entity_chain:
            try:
                path_result = await broken_path_detect(question, entity_chain, self.neo4j)
                result.graph_path = " → ".join(path_result.full_path) if path_result.full_path else ""
                result.missing_relationship = path_result.missing_rel or ""
                result.suggested_fix = path_result.suggested_fix or ""
                
                if path_result.break_point:
                    result.reason = f"Break point: {path_result.break_point}"
                
                # ── Phase 4: Coverage scoring ───────────────────────────────
                score = await compute_coverage_score(
                    question=question,
                    question_embedding=q_emb,
                    entity_chain=entity_chain,
                    graph_edge_count=len(path_result.full_path) - 1 if path_result.full_path else 0,
                    total_expected_edges=len(entity_chain) - 1,
                    evidence_confidences=[0.7] * max(0, len(path_result.full_path) - 1),
                    vector_store=self.vector_store,
                    neo4j=self.neo4j,
                )
                result.scores = score.to_dict()
                result.verdict = score.verdict
                result.entities_found = ", ".join(path_result.entities_found) if path_result.entities_found else ""
                
            except Exception as e:
                logger.warning(f"Graph/score phase failed: {e}")
        
        return result
    
    def ask(self, question: str, entity_chain: list[str] = None) -> AnswerResult:
        """Sync wrapper with nested loop handling."""
        import asyncio, concurrent.futures
        
        async def _run():
            return await self.ask_async(question, entity_chain)
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())
        else:
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run())
                return future.result()