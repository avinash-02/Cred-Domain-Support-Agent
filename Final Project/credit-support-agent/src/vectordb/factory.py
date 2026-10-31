"""
Vector store factory for switching between implementations.

Allows runtime selection of in-memory or ChromaDB backend.
"""

from typing import Union, Literal


def get_vector_store(
    backend: Literal["memory", "chromadb"] = "memory",
    persist_dir: str = "./chromadb_data",
):
    """
    Factory function to create a vector store instance.

    Args:
        backend: "memory" for in-memory NumPy store (default, no dependencies)
                 "chromadb" for persistent ChromaDB store
        persist_dir: Directory for ChromaDB persistence (ignored for memory backend)

    Returns:
        Vector store instance with identical interface

    Raises:
        ValueError: If backend is not recognized
    """
    if backend == "memory":
        from src.vectordb.memory_store import MemoryVectorStore
        return MemoryVectorStore()

    elif backend == "chromadb":
        from src.vectordb.chromadb_store import ChromaDBStore
        return ChromaDBStore(persist_dir=persist_dir)

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Choose 'memory' or 'chromadb'"
        )
