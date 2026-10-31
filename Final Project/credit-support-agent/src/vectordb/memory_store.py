"""
In-memory vector store using NumPy for similarity search.

Lightweight, dependency-free alternative to ChromaDB.
Uses cosine similarity for retrieval.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class MemoryCollection:
    """In-memory collection of vectors and metadata."""
    name: str
    embeddings: List[np.ndarray] = field(default_factory=list)
    ids: List[str] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)
    metadatas: List[Dict] = field(default_factory=list)

    def count(self) -> int:
        """Return number of items in collection."""
        return len(self.ids)


class MemoryVectorStore:
    """In-memory vector store using NumPy."""

    def __init__(self):
        """Initialize in-memory store."""
        self.collections: Dict[str, MemoryCollection] = {}

    def get_or_create_collection(self, collection_name: str) -> MemoryCollection:
        """
        Get or create a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            MemoryCollection object
        """
        if collection_name not in self.collections:
            self.collections[collection_name] = MemoryCollection(name=collection_name)
        return self.collections[collection_name]

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
            embeddings: List of embedding vectors (lists of floats)
            texts: List of chunk texts
            metadatas: List of metadata dicts for each chunk
        """
        collection = self.get_or_create_collection(collection_name)

        # Convert embeddings to numpy arrays
        embeddings_np = [np.array(e, dtype=np.float32) for e in embeddings]

        # Upsert or append
        for chunk_id, embedding, text, metadata in zip(
            chunk_ids, embeddings_np, texts, metadatas
        ):
            # Check if ID already exists (update case)
            if chunk_id in collection.ids:
                idx = collection.ids.index(chunk_id)
                collection.embeddings[idx] = embedding
                collection.documents[idx] = text
                collection.metadatas[idx] = metadata
            else:
                # Append new
                collection.ids.append(chunk_id)
                collection.embeddings.append(embedding)
                collection.documents.append(text)
                collection.metadatas.append(metadata)

    def _cosine_similarity(
        self, query: np.ndarray, embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and embeddings.

        Args:
            query: Query embedding
            embeddings: List of embeddings to compare against

        Returns:
            Array of similarities (higher = more similar)
        """
        if not embeddings:
            return np.array([])

        # Stack embeddings into matrix
        embeddings_matrix = np.stack(embeddings, axis=0)

        # Normalize
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        embeddings_norm = embeddings_matrix / (
            np.linalg.norm(embeddings_matrix, axis=1, keepdims=True) + 1e-8
        )

        # Cosine similarity (dot product of normalized vectors)
        similarities = np.dot(embeddings_norm, query_norm)

        # Convert to distances (lower = more similar for consistency with ChromaDB)
        distances = 1 - similarities

        return distances

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
            Query results matching ChromaDB format
        """
        collection = self.get_or_create_collection(collection_name)

        if not collection.ids:
            # Empty collection
            return {
                'ids': [[]],
                'documents': [[]],
                'distances': [[]],
                'metadatas': [[]],
            }

        # Compute distances
        query_np = np.array(query_embedding, dtype=np.float32)
        distances = self._cosine_similarity(query_np, collection.embeddings)

        # Get top-k
        top_indices = np.argsort(distances)[:top_k]

        # Format results to match ChromaDB format
        results = {
            'ids': [[collection.ids[i] for i in top_indices]],
            'documents': [[collection.documents[i] for i in top_indices]],
            'distances': [[distances[i] for i in top_indices]],
            'metadatas': [[collection.metadatas[i] for i in top_indices]],
        }

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

        return {
            "name": collection_name,
            "total_chunks": collection.count(),
        }
