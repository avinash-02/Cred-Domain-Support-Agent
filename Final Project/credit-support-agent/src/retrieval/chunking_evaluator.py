"""
Chunking strategy evaluation and comparison.

Evaluates precision and recall at document level for fixed vs sentence-based chunking.
"""

import sys
sys.path.insert(0, '.')

from src.retrieval.rag_pipeline import RAGPipeline
from typing import Set, List, Dict, Tuple


class ChunkingStrategyEvaluator:
    """Evaluate and compare chunking strategies."""

    def __init__(self, rag_pipeline: RAGPipeline):
        """Initialize with RAG pipeline."""
        self.rag = rag_pipeline

    def _get_relevant_documents(self, query: str) -> Set[str]:
        """
        Manually identify truly relevant documents for a query.
        Based on domain knowledge of banking policies.
        """
        query_lower = query.lower()

        # Define ground truth document relevance
        relevance_map = {
            "income requirement": {"doc_001", "doc_007"},  # Eligibility, Interest rates
            "emi calculated": {"doc_002"},  # EMI Calculation Rules
            "documents needed": {"doc_004"},  # KYC Requirements
            "cibil score": {"doc_010"},  # Credit Score Impact
            "penalties prepayment": {"doc_008"},  # Prepayment Penalty
        }

        for keyword, docs in relevance_map.items():
            if keyword in query_lower:
                return docs

        return set()

    def _extract_doc_id(self, chunk_id: str) -> str:
        """Extract parent document ID from chunk ID."""
        # Chunk IDs are like: doc_001_fixed_0 or doc_001_sentence_0
        return chunk_id.rsplit('_', 2)[0]

    def evaluate_query(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict:
        """
        Evaluate a single query across both strategies.

        Returns precision and recall metrics.
        """
        # Get ground truth relevant documents
        relevant_docs = self._get_relevant_documents(query)

        if not relevant_docs:
            return None

        # Retrieve from fixed chunking
        fixed_chunks = self.rag.retrieve(query, strategy="fixed", top_k=top_k)
        fixed_retrieved_docs = set(
            self._extract_doc_id(c["chunk_id"]) for c in fixed_chunks
        )

        # Retrieve from sentence-based chunking
        sentence_chunks = self.rag.retrieve(query, strategy="sentence", top_k=top_k)
        sentence_retrieved_docs = set(
            self._extract_doc_id(c["chunk_id"]) for c in sentence_chunks
        )

        # Compute precision and recall for fixed
        fixed_tp = len(fixed_retrieved_docs & relevant_docs)
        fixed_fp = len(fixed_retrieved_docs - relevant_docs)
        fixed_fn = len(relevant_docs - fixed_retrieved_docs)

        fixed_precision = fixed_tp / len(fixed_retrieved_docs) if fixed_retrieved_docs else 0
        fixed_recall = fixed_tp / len(relevant_docs) if relevant_docs else 0

        # Compute precision and recall for sentence
        sentence_tp = len(sentence_retrieved_docs & relevant_docs)
        sentence_fp = len(sentence_retrieved_docs - relevant_docs)
        sentence_fn = len(relevant_docs - sentence_retrieved_docs)

        sentence_precision = sentence_tp / len(sentence_retrieved_docs) if sentence_retrieved_docs else 0
        sentence_recall = sentence_tp / len(relevant_docs) if relevant_docs else 0

        return {
            "query": query,
            "relevant_docs": relevant_docs,
            "fixed": {
                "retrieved_docs": fixed_retrieved_docs,
                "true_positives": fixed_tp,
                "false_positives": fixed_fp,
                "false_negatives": fixed_fn,
                "precision": fixed_precision,
                "recall": fixed_recall,
                "f1": 2 * (fixed_precision * fixed_recall) / (fixed_precision + fixed_recall) if (fixed_precision + fixed_recall) > 0 else 0,
            },
            "sentence": {
                "retrieved_docs": sentence_retrieved_docs,
                "true_positives": sentence_tp,
                "false_positives": sentence_fp,
                "false_negatives": sentence_fn,
                "precision": sentence_precision,
                "recall": sentence_recall,
                "f1": 2 * (sentence_precision * sentence_recall) / (sentence_precision + sentence_recall) if (sentence_precision + sentence_recall) > 0 else 0,
            }
        }

    def evaluate_all(self, queries: List[str], top_k: int = 5) -> Tuple[List[Dict], Dict]:
        """
        Evaluate all queries.

        Returns list of per-query results and aggregate statistics.
        """
        results = []

        for query in queries:
            result = self.evaluate_query(query, top_k=top_k)
            if result:
                results.append(result)

        # Aggregate statistics
        fixed_precisions = [r["fixed"]["precision"] for r in results]
        fixed_recalls = [r["fixed"]["recall"] for r in results]
        fixed_f1s = [r["fixed"]["f1"] for r in results]

        sentence_precisions = [r["sentence"]["precision"] for r in results]
        sentence_recalls = [r["sentence"]["recall"] for r in results]
        sentence_f1s = [r["sentence"]["f1"] for r in results]

        aggregate = {
            "fixed": {
                "avg_precision": sum(fixed_precisions) / len(fixed_precisions) if fixed_precisions else 0,
                "avg_recall": sum(fixed_recalls) / len(fixed_recalls) if fixed_recalls else 0,
                "avg_f1": sum(fixed_f1s) / len(fixed_f1s) if fixed_f1s else 0,
            },
            "sentence": {
                "avg_precision": sum(sentence_precisions) / len(sentence_precisions) if sentence_precisions else 0,
                "avg_recall": sum(sentence_recalls) / len(sentence_recalls) if sentence_recalls else 0,
                "avg_f1": sum(sentence_f1s) / len(sentence_f1s) if sentence_f1s else 0,
            }
        }

        return results, aggregate


def run_evaluation():
    """Run chunking strategy evaluation."""

    print("\n" + "=" * 90)
    print("CHUNKING STRATEGY EVALUATION")
    print("=" * 90)

    # Initialize
    print("\nInitializing RAG pipeline...")
    rag = RAGPipeline(backend="memory")

    print("Indexing documents...")
    rag.index_documents()

    evaluator = ChunkingStrategyEvaluator(rag)

    # Test queries (from Task 4)
    queries = [
        "What is the minimum income requirement for a home loan?",
        "How is EMI calculated?",
        "What documents are needed for KYC?",
        "What is CIBIL score and why does it matter?",
        "What are the penalties for prepayment?",
    ]

    # Evaluate
    results, aggregate = evaluator.evaluate_all(queries, top_k=5)

    # Print per-query results
    print("\n" + "=" * 90)
    print("PER-QUERY EVALUATION")
    print("=" * 90)

    for i, result in enumerate(results, 1):
        query = result["query"]
        relevant = result["relevant_docs"]

        fixed = result["fixed"]
        sentence = result["sentence"]

        print(f"\n{i}. Query: {query}")
        print(f"   Ground Truth Documents: {relevant}")

        print(f"\n   FIXED-SIZE CHUNKING:")
        print(f"     Retrieved Docs: {fixed['retrieved_docs']}")
        print(f"     TP={fixed['true_positives']}, FP={fixed['false_positives']}, FN={fixed['false_negatives']}")
        print(f"     Precision = TP/(TP+FP) = {fixed['true_positives']}/({fixed['true_positives']}+{fixed['false_positives']}) = {fixed['precision']:.3f}")
        print(f"     Recall    = TP/(TP+FN) = {fixed['true_positives']}/({fixed['true_positives']}+{fixed['false_negatives']}) = {fixed['recall']:.3f}")
        print(f"     F1-Score  = {fixed['f1']:.3f}")

        print(f"\n   SENTENCE-BASED CHUNKING:")
        print(f"     Retrieved Docs: {sentence['retrieved_docs']}")
        print(f"     TP={sentence['true_positives']}, FP={sentence['false_positives']}, FN={sentence['false_negatives']}")
        print(f"     Precision = TP/(TP+FP) = {sentence['true_positives']}/({sentence['true_positives']}+{sentence['false_positives']}) = {sentence['precision']:.3f}")
        print(f"     Recall    = TP/(TP+FN) = {sentence['true_positives']}/({sentence['true_positives']}+{sentence['false_negatives']}) = {sentence['recall']:.3f}")
        print(f"     F1-Score  = {sentence['f1']:.3f}")

        # Winner for this query
        if fixed['f1'] > sentence['f1']:
            winner = "FIXED-SIZE"
            diff = fixed['f1'] - sentence['f1']
        elif sentence['f1'] > fixed['f1']:
            winner = "SENTENCE-BASED"
            diff = sentence['f1'] - fixed['f1']
        else:
            winner = "TIE"
            diff = 0

        print(f"\n   Winner: {winner} (F1 diff: {diff:+.3f})")

    # Print aggregate results
    print("\n" + "=" * 90)
    print("AGGREGATE RESULTS (5 Queries)")
    print("=" * 90)

    print(f"\nFIXED-SIZE CHUNKING:")
    print(f"  Average Precision: {aggregate['fixed']['avg_precision']:.3f}")
    print(f"  Average Recall:    {aggregate['fixed']['avg_recall']:.3f}")
    print(f"  Average F1-Score:  {aggregate['fixed']['avg_f1']:.3f}")

    print(f"\nSENTENCE-BASED CHUNKING:")
    print(f"  Average Precision: {aggregate['sentence']['avg_precision']:.3f}")
    print(f"  Average Recall:    {aggregate['sentence']['avg_recall']:.3f}")
    print(f"  Average F1-Score:  {aggregate['sentence']['avg_f1']:.3f}")

    # Recommendation
    print("\n" + "=" * 90)
    print("RECOMMENDATION")
    print("=" * 90)

    if aggregate['fixed']['avg_f1'] > aggregate['sentence']['avg_f1']:
        winner = "FIXED-SIZE CHUNKING"
        winner_score = aggregate['fixed']['avg_f1']
        loser_score = aggregate['sentence']['avg_f1']
    else:
        winner = "SENTENCE-BASED CHUNKING"
        winner_score = aggregate['sentence']['avg_f1']
        loser_score = aggregate['fixed']['avg_f1']

    print(f"\nDeploy: {winner}")
    print(f"  Winner F1-Score: {winner_score:.3f}")
    print(f"  Loser F1-Score:  {loser_score:.3f}")
    print(f"  Advantage:       {winner_score - loser_score:+.3f}")

    # Write recommendation text
    if winner == "FIXED-SIZE CHUNKING":
        rec_text = (
            f"Deploy fixed-size chunking (F1={winner_score:.3f}) over sentence-based (F1={loser_score:.3f}). "
            f"Fixed-size chunks provide more consistent retrieval performance with uniform context windows "
            f"and better precision across diverse banking queries. The {(winner_score - loser_score)*100:.1f}% "
            f"F1-score advantage justifies the trade-off against sentence boundaries."
        )
    else:
        rec_text = (
            f"Deploy sentence-based chunking (F1={winner_score:.3f}) over fixed-size (F1={loser_score:.3f}). "
            f"Sentence-based chunks preserve semantic boundaries and achieve {(winner_score - loser_score)*100:.1f}% "
            f"higher F1-score, providing more contextually coherent retrieval for banking policy queries."
        )

    print(f"\n{rec_text}")
    print("=" * 90 + "\n")

    return results, aggregate, rec_text


if __name__ == "__main__":
    run_evaluation()
