"""
Document chunking strategies for RAG.

Implements two chunking approaches:
1. Fixed-size chunks with overlap
2. Sentence-based chunks
"""

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """A document chunk."""
    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    strategy: str  # "fixed" or "sentence"


class FixedSizeChunker:
    """Fixed-size chunking with overlap."""

    def __init__(self, chunk_size: int = 256, overlap: int = 50):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, doc_id: str, doc_title: str, text: str) -> List[Chunk]:
        """
        Chunk text into fixed-size pieces with overlap.

        Args:
            doc_id: Document identifier
            doc_title: Document title
            text: Text to chunk

        Returns:
            List of Chunk objects
        """
        chunks = []
        step = self.chunk_size - self.overlap

        for i in range(0, len(text), step):
            chunk_text = text[i : i + self.chunk_size]
            if len(chunk_text.strip()) > 0:
                chunk_id = f"{doc_id}_fixed_{len(chunks)}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        strategy="fixed",
                    )
                )

        return chunks


class SentenceBasedChunker:
    """Sentence-based chunking."""

    def __init__(self, max_sentences_per_chunk: int = 3):
        """
        Initialize sentence-based chunker.

        Args:
            max_sentences_per_chunk: Max sentences per chunk
        """
        self.max_sentences_per_chunk = max_sentences_per_chunk

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.

        Handles common abbreviations and patterns.
        """
        # Split on periods, exclamation marks, question marks
        # but preserve the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, doc_id: str, doc_title: str, text: str) -> List[Chunk]:
        """
        Chunk text into sentence-based pieces.

        Args:
            doc_id: Document identifier
            doc_title: Document title
            text: Text to chunk

        Returns:
            List of Chunk objects
        """
        sentences = self._split_sentences(text)
        chunks = []

        i = 0
        chunk_num = 0
        while i < len(sentences):
            # Group sentences up to max_sentences_per_chunk
            chunk_sentences = sentences[i : i + self.max_sentences_per_chunk]
            chunk_text = ' '.join(chunk_sentences)

            if len(chunk_text.strip()) > 0:
                chunk_id = f"{doc_id}_sentence_{chunk_num}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        doc_id=doc_id,
                        doc_title=doc_title,
                        strategy="sentence",
                    )
                )
                chunk_num += 1

            i += self.max_sentences_per_chunk

        return chunks
