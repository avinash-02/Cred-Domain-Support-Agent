"""
Embedding generation using SentenceTransformers.

Uses the free, local all-MiniLM-L6-v2 model for efficient embeddings.
"""

from typing import List
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.

        Args:
            model_name: HuggingFace model identifier
                       (default: all-MiniLM-L6-v2 - lightweight, fast, free)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (each is a list of floats)
        """
        embeddings = self.model.encode(texts)
        # Convert numpy array to list of lists
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """Return the dimension of generated embeddings."""
        return self.embedding_dim
