"""Semantic chunking — splits content into overlapping chunks for embedding."""
import hashlib
import re
from dataclasses import dataclass
from typing import Optional
from ..core.config import chunk as chunk_cfg
from ..core.logging import get_logger

logger = get_logger(__name__)

@dataclass
class Chunk:
    chunk_id: str     # hash of url + chunk_index (stable across re-runs)
    url: str
    title: str
    index: int        # chunk number within this page
    content: str
    word_count: int
    char_count: int


def _make_chunk_id(url: str, chunk_index: int) -> str:
    """Stable ID from URL + index — ensures re-runs skip already-embedded chunks."""
    raw = f"{url}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitting on common punctuation."""
    # Split on sentence boundaries: . ! ? followed by space/capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_page(result) -> list[Chunk]:
    """
    Split a CrawlResult into semantic chunks.
    Accepts either a CrawlResult object or a dict with keys: content, title, url, success.
    """
    cfg = chunk_cfg()
    chunk_size = cfg.get("chunk_size", 512)    # words
    chunk_overlap = cfg.get("chunk_overlap", 64)  # words overlap
    min_len = cfg.get("min_chunk_length", 50)

    # Handle both objects and dicts
    if isinstance(result, dict):
        content = result.get("content", "") or ""
        title = result.get("title", "") or ""
        url = result.get("url", "")
        if not result.get("success", True):
            return []
    else:
        content = result.content or ""
        title = result.title or ""
        url = result.url
    
    if not content or len(content) < min_len:
        return []
    
    # Paragraph split — handle both \n\n (markdown paragraphs) and
    # single \n (Wikipedia-style lines, bulleted lists)
    # First try double newlines, then fall back to single
    raw_paragraphs = [p.strip() for p in re.split(r'\n{2,}', content) if p.strip()]
    if len(raw_paragraphs) <= 2 and len(content) > 5000:
        # Fallback: content didn't split well, try single newlines
        raw_paragraphs = [p.strip() for p in re.split(r'\n(?=\S)', content) if p.strip()]
    # Remove very short lines (nav menus, breadcrumbs) — keep lines >= 40 chars
    paragraphs = [p for p in raw_paragraphs if len(p) >= 40]
    if not paragraphs:
        return []

    chunks: list[Chunk] = []
    chunk_index = 0
    
    current_words: list[str] = []
    current_word_count = 0
    
    for para in paragraphs:
        para_words = para.split()
        para_word_count = len(para_words)
        
        # If this paragraph alone exceeds chunk_size, split it further
        if para_word_count > chunk_size:
            # Flush current chunk first
            if current_words:
                text = " ".join(current_words)
                chunks.append(Chunk(
                    chunk_id=_make_chunk_id(url, chunk_index),
                    url=url,
                    title=title,
                    index=chunk_index,
                    content=text,
                    word_count=len(current_words),
                    char_count=len(text),
                ))
                chunk_index += 1
                current_words = []
                current_word_count = 0
            
            # Split long paragraph by sentences
            sentences = _split_into_sentences(para)
            for sent in sentences:
                sent_words = sent.split()
                if not current_words:
                    current_words = sent_words
                    current_word_count = len(sent_words)
                elif current_word_count + len(sent_words) <= chunk_size + 100:
                    # Allow slight overshoot for sentence integrity
                    current_words.extend(sent_words)
                    current_word_count += len(sent_words)
                else:
                    # Emit current chunk
                    text = " ".join(current_words)
                    if len(text) >= min_len:
                        chunks.append(Chunk(
                            chunk_id=_make_chunk_id(url, chunk_index),
                            url=url,
                            title=title,
                            index=chunk_index,
                            content=text,
                            word_count=len(current_words),
                            char_count=len(text),
                        ))
                        chunk_index += 1
                    # Start new chunk with overlap
                    overlap_words = current_words[-chunk_overlap:] if chunk_overlap > 0 else []
                    current_words = overlap_words + sent_words
                    current_word_count = len(current_words)
        else:
            # Normal paragraph — add to current chunk
            if current_word_count + para_word_count <= chunk_size:
                current_words.extend(para_words)
                current_word_count += para_word_count
            else:
                # Flush and start new chunk
                text = " ".join(current_words)
                if len(text) >= min_len:
                    chunks.append(Chunk(
                        chunk_id=_make_chunk_id(url, chunk_index),
                        url=url,
                        title=title,
                        index=chunk_index,
                        content=text,
                        word_count=len(current_words),
                        char_count=len(text),
                    ))
                    chunk_index += 1
                # Overlap — carry last N words into next chunk
                overlap_words = current_words[-chunk_overlap:] if chunk_overlap > 0 else []
                current_words = overlap_words + para_words
                current_word_count = len(current_words)
    
    # Flush remaining
    if current_words:
        text = " ".join(current_words)
        if len(text) >= min_len:
            chunks.append(Chunk(
                chunk_id=_make_chunk_id(url, chunk_index),
                url=url,
                title=title,
                index=chunk_index,
                content=text,
                word_count=len(current_words),
                char_count=len(text),
            ))
    
    logger.debug(f"Chunked {url}: {len(chunks)} chunks")
    return chunks


def chunk_pages(crawl_results: list) -> list[Chunk]:
    """Chunk all crawl results into a flat list of chunks."""
    all_chunks: list[Chunk] = []
    for result in crawl_results:
        if isinstance(result, dict):
            success = result.get("success", True)
        else:
            success = result.success
        if success:
            all_chunks.extend(chunk_page(result))
    logger.info(f"Chunking complete: {len(all_chunks)} total chunks from {len(crawl_results)} pages")

    # ── Boilerplate dedup ────────────────────────────────────────────
    # Detect and mark chunks that appear near-identically on >50% of pages.
    # These are site furniture (nav/footer/announcement bars) that should
    # not be sent to entity extraction N times.
    if crawl_results:
        total_pages = len(crawl_results)
        threshold = max(3, int(total_pages * 0.5))  # >50% of pages
        # Build a content-hash → page-URL mapping
        from collections import Counter
        hash_page_counter = Counter()
        chunk_map = {}
        normalized_chunks = []
        clean_chunks = []

        for i, chunk in enumerate(all_chunks):
            # Normalize: lowercase, strip whitespace, collapse whitespace
            normalized = " ".join(chunk.content.lower().split())
            content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]
            # Track unique URLs per hash
            if content_hash not in chunk_map:
                chunk_map[content_hash] = {"count": 0, "urls": set(), "first_index": i}
            chunk_map[content_hash]["urls"].add(chunk.url)
            normalized_chunks.append((i, content_hash))

        # Classify each hash
        boilerplate_hashes = set()
        for h, info in chunk_map.items():
            unique_pages = len(info["urls"])
            if unique_pages > threshold:
                boilerplate_hashes.add(h)
                logger.debug(f"Boilerplate detected: hash={h} appears on {unique_pages}/{total_pages} pages")

        # Filter chunks: keep boilerplate only ONCE (first occurrence)
        seen_boilerplate = set()
        for i, chunk in enumerate(all_chunks):
            h = normalized_chunks[i][1]
            if h in boilerplate_hashes:
                if h not in seen_boilerplate:
                    seen_boilerplate.add(h)
                    clean_chunks.append(chunk)
                # else: skip duplicate boilerplate
            else:
                clean_chunks.append(chunk)

        removed = len(all_chunks) - len(clean_chunks)
        if removed:
            logger.info(f"Boilerplate dedup: removed {removed} duplicate boilerplate chunks ({len(boilerplate_hashes)} unique boilerplate patterns)")
        return clean_chunks

    return all_chunks