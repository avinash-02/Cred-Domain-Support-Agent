"""
RAG retrieval pipeline.

Orchestrates chunking, embedding, and vector store indexing.
Provides retrieval interface for the multi-agent system.
Supports both in-memory and ChromaDB backends via factory pattern.
"""

import json
from typing import List, Dict, Literal
from pathlib import Path

from src.chunking.strategies import FixedSizeChunker, SentenceBasedChunker, Chunk
from src.embeddings.generator import EmbeddingGenerator
from src.vectordb.factory import get_vector_store


class RAGPipeline:
    """End-to-end RAG pipeline for indexing and retrieval."""

    def __init__(
        self,
        knowledge_base_path: str = "data/raw/knowledge_base_docs.json",
        backend: Literal["memory", "chromadb"] = "memory",
        chromadb_dir: str = "./chromadb_data",
        chunk_size: int = 256,
        chunk_overlap: int = 50,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize RAG pipeline.

        Args:
            knowledge_base_path: Path to knowledge base JSON
            backend: "memory" (default, no dependencies) or "chromadb" (persistent)
            chromadb_dir: ChromaDB persistence directory (ignored for memory backend)
            chunk_size: Fixed chunk size in characters
            chunk_overlap: Overlap between fixed chunks
            embedding_model: SentenceTransformer model name
        """
        self.kb_path = knowledge_base_path
        self.backend = backend
        self.fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        self.sentence_chunker = SentenceBasedChunker(max_sentences_per_chunk=3)
        self.embedding_gen = EmbeddingGenerator(model_name=embedding_model)

        # Create vector store using factory
        self.vectordb = get_vector_store(backend=backend, persist_dir=chromadb_dir)

        self.fixed_collection_name = "banking_docs_fixed_chunks"
        self.sentence_collection_name = "banking_docs_sentence_chunks"

    def load_knowledge_base(self) -> List[Dict]:
        """Load knowledge base documents from JSON."""
        with open(self.kb_path, 'r') as f:
            data = json.load(f)
        return data.get('documents', [])

    def index_documents(self) -> Dict:
        """
        Index all knowledge base documents using both chunking strategies.

        Returns:
            Statistics about indexing process
        """
        documents = self.load_knowledge_base()

        fixed_chunks = []
        sentence_chunks = []

        # Chunk documents with both strategies
        for doc in documents:
            doc_id = doc['id']
            doc_title = doc['title']
            content = doc['content']

            # Fixed-size chunks
            fixed = self.fixed_chunker.chunk(doc_id, doc_title, content)
            fixed_chunks.extend(fixed)

            # Sentence-based chunks
            sentence = self.sentence_chunker.chunk(doc_id, doc_title, content)
            sentence_chunks.extend(sentence)

        # Embed fixed-size chunks
        fixed_texts = [c.text for c in fixed_chunks]
        fixed_embeddings = self.embedding_gen.embed(fixed_texts)
        fixed_metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "strategy": c.strategy,
            }
            for c in fixed_chunks
        ]

        # Upsert to fixed collection
        self.vectordb.upsert_chunks(
            collection_name=self.fixed_collection_name,
            chunk_ids=[c.chunk_id for c in fixed_chunks],
            embeddings=fixed_embeddings,
            texts=fixed_texts,
            metadatas=fixed_metadatas,
        )

        # Embed sentence-based chunks
        sentence_texts = [c.text for c in sentence_chunks]
        sentence_embeddings = self.embedding_gen.embed(sentence_texts)
        sentence_metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "strategy": c.strategy,
            }
            for c in sentence_chunks
        ]

        # Upsert to sentence collection
        self.vectordb.upsert_chunks(
            collection_name=self.sentence_collection_name,
            chunk_ids=[c.chunk_id for c in sentence_chunks],
            embeddings=sentence_embeddings,
            texts=sentence_texts,
            metadatas=sentence_metadatas,
        )

        return {
            "total_documents": len(documents),
            "fixed_chunks_count": len(fixed_chunks),
            "sentence_chunks_count": len(sentence_chunks),
            "embedding_dimension": self.embedding_gen.get_embedding_dimension(),
        }

    def retrieve(
        self,
        query: str,
        strategy: str = "fixed",
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query string
            strategy: "fixed" or "sentence" (which collection to search)
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with scores
        """
        # Embed the query
        query_embedding = self.embedding_gen.embed([query])[0]

        # Select collection
        collection_name = (
            self.fixed_collection_name if strategy == "fixed"
            else self.sentence_collection_name
        )

        # Query
        results = self.vectordb.query_collection(
            collection_name=collection_name,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # Format results
        retrieved = []
        for i in range(len(results['ids'][0])):
            retrieved.append({
                "chunk_id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "distance": results['distances'][0][i],
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
            })

        return retrieved

    def get_stats(self) -> Dict:
        """Get statistics for both collections."""
        return {
            "fixed": self.vectordb.get_collection_stats(self.fixed_collection_name),
            "sentence": self.vectordb.get_collection_stats(self.sentence_collection_name),
        }
