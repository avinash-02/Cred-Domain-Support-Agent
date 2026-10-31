"""
ChromaDB vector store indexing for RAG.

Creates and maintains separate collections for each chunking strategy.
"""

from typing import List, Dict
from pathlib import Path

# Disable telemetry before any chromadb imports
import sys
import os
os.environ['CHROMA_TELEMETRY'] = '0'

# Try importing chromadb with telemetry disabled
try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"Warning: ChromaDB import issue: {e}")
    # Fallback: create a mock store for now
    chromadb = None


class ChromaDBStore:
    """ChromaDB vector store manager."""

    def __init__(self, persist_dir: str = "./chromadb_data"):
        """
        Initialize ChromaDB store with persistence.

        Args:
            persist_dir: Directory for persistent storage
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Use persistent client
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

    def get_or_create_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Get or create a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            ChromaDB collection object
        """
        # Get or create with default settings
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine distance for embeddings
        )
        return collection

    def upsert_chunks(
        self,
        collection_name: str,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict],
    ) -> None:
        """
        Upsert chunks into a collection.

        Args:
            collection_name: Name of the collection
            chunk_ids: List of chunk identifiers
            embeddings: List of embedding vectors
            texts: List of chunk texts
            metadatas: List of metadata dicts for each chunk
        """
        collection = self.get_or_create_collection(collection_name)

        collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query_collection(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> Dict:
        """
        Query a collection for similar chunks.

        Args:
            collection_name: Name of the collection
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            Query results with distances and documents
        """
        collection = self.get_or_create_collection(collection_name)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        return results

    def get_collection_stats(self, collection_name: str) -> Dict:
        """
        Get statistics about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Dictionary with collection stats
        """
        collection = self.get_or_create_collection(collection_name)
        count = collection.count()

        return {
            "name": collection_name,
            "total_chunks": count,
        }
