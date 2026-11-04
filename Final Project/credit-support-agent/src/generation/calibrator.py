"""
Calibration script for grounded generation threshold.

Measures top-1 cosine similarity for in-scope and out-of-scope queries
to empirically determine the "I don't know" threshold.
"""

from src.retrieval.rag_pipeline import RAGPipeline
from typing import List, Dict


class GroundednessCalibrator:
    """Calibrate groundedness threshold empirically."""

    def __init__(self, rag_pipeline: RAGPipeline):
        """Initialize calibrator with RAG pipeline."""
        self.rag = rag_pipeline

    def measure_similarity(
        self,
        query: str,
        strategy: str = "fixed",
        top_k: int = 1,
    ) -> float:
        """
        Measure top-1 cosine similarity for a query.

        Args:
            query: Query string
            strategy: "fixed" or "sentence"
            top_k: Number of results to retrieve

        Returns:
            Top-1 distance (cosine distance, lower = more similar)
        """
        results = self.rag.retrieve(query, strategy=strategy, top_k=top_k)

        if not results:
            return 1.0  # Worst case: no results

        # Return top-1 distance (lower is better)
        return results[0]["distance"]

    def calibrate(
        self,
        in_scope_queries: List[str],
        out_of_scope_queries: List[str],
        strategy: str = "fixed",
    ) -> Dict:
        """
        Calibrate threshold by measuring in-scope vs out-of-scope queries.

        Args:
            in_scope_queries: List of queries within banking domain
            out_of_scope_queries: List of queries outside banking domain
            strategy: "fixed" or "sentence"

        Returns:
            Calibration report with measured values and recommended threshold
        """
        print("=" * 70)
        print("GROUNDEDNESS CALIBRATION")
        print("=" * 70)

        # Measure in-scope queries
        print(f"\nMeasuring {len(in_scope_queries)} IN-SCOPE queries:")
        print("-" * 70)
        in_scope_distances = []
        for i, query in enumerate(in_scope_queries, 1):
            distance = self.measure_similarity(query, strategy=strategy, top_k=1)
            in_scope_distances.append(distance)
            print(f"{i}. Query: {query}")
            print(f"   Top-1 Distance: {distance:.4f}")
            print()

        # Measure out-of-scope queries
        print(f"\nMeasuring {len(out_of_scope_queries)} OUT-OF-SCOPE queries:")
        print("-" * 70)
        out_of_scope_distances = []
        for i, query in enumerate(out_of_scope_queries, 1):
            distance = self.measure_similarity(query, strategy=strategy, top_k=1)
            out_of_scope_distances.append(distance)
            print(f"{i}. Query: {query}")
            print(f"   Top-1 Distance: {distance:.4f}")
            print()

        # Calculate statistics
        in_scope_min = min(in_scope_distances)
        in_scope_max = max(in_scope_distances)
        in_scope_avg = sum(in_scope_distances) / len(in_scope_distances)

        out_of_scope_min = min(out_of_scope_distances)
        out_of_scope_max = max(out_of_scope_distances)
        out_of_scope_avg = sum(out_of_scope_distances) / len(out_of_scope_distances)

        # Find threshold: midpoint between max in-scope and min out-of-scope
        # (or use max in-scope if out-of-scope is lower)
        if out_of_scope_min > in_scope_max:
            # Clear separation: use midpoint
            threshold = (in_scope_max + out_of_scope_min) / 2
            separation = "CLEAR"
        else:
            # Overlap: use conservative approach (higher threshold)
            threshold = in_scope_max
            separation = "OVERLAPPING"

        # Print report
        print("\n" + "=" * 70)
        print("CALIBRATION RESULTS")
        print("=" * 70)

        print(f"\nIN-SCOPE Statistics (lower = better match):")
        print(f"  Min distance: {in_scope_min:.4f}")
        print(f"  Max distance: {in_scope_max:.4f}")
        print(f"  Avg distance: {in_scope_avg:.4f}")

        print(f"\nOUT-OF-SCOPE Statistics (higher = better rejection):")
        print(f"  Min distance: {out_of_scope_min:.4f}")
        print(f"  Max distance: {out_of_scope_max:.4f}")
        print(f"  Avg distance: {out_of_scope_avg:.4f}")

        print(f"\nSeparation: {separation}")
        print(f"Recommended Threshold: {threshold:.4f}")
        print(f"  (Respond with 'I don't know' if top-1 distance > {threshold:.4f})")

        print("\n" + "=" * 70)

        return {
            "in_scope_distances": in_scope_distances,
            "out_of_scope_distances": out_of_scope_distances,
            "in_scope_stats": {
                "min": in_scope_min,
                "max": in_scope_max,
                "avg": in_scope_avg,
            },
            "out_of_scope_stats": {
                "min": out_of_scope_min,
                "max": out_of_scope_max,
                "avg": out_of_scope_avg,
            },
            "threshold": threshold,
            "separation": separation,
        }


if __name__ == "__main__":
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline(backend="memory")

    print("Indexing documents...")
    rag.index_documents()

    # Define test queries
    in_scope_queries = [
        "What is the minimum income requirement for a home loan?",
        "How is EMI calculated?",
        "What documents are needed for KYC?",
        "What is CIBIL score and why does it matter?",
        "What are the penalties for prepayment?",
    ]

    out_of_scope_queries = [
        "What is the capital of France?",
        "How do I cook pasta?",
    ]

    # Run calibration
    calibrator = GroundednessCalibrator(rag)
    results = calibrator.calibrate(
        in_scope_queries=in_scope_queries,
        out_of_scope_queries=out_of_scope_queries,
        strategy="fixed",
    )
