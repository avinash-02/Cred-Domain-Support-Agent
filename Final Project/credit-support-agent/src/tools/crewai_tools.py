"""
CrewAI tools for the credit support agent crew.

Wraps RAG retrieval and loan lookup as tools with explicit Pydantic schemas.
In CrewAI 1.15.35, schemas are defined via function parameter type hints.
"""

import sys
sys.path.insert(0, '.')

from typing import Optional, Annotated
from pydantic import BaseModel, Field
from crewai.tools import tool

from src.retrieval.rag_pipeline import RAGPipeline
from src.tools.loan_lookup_tool import check_loan_application_status


# ============================================================================
# Tool Argument Schemas
# ============================================================================

class RAGQuerySchema(BaseModel):
    """Schema for RAG retrieval query."""
    query: str = Field(
        ...,
        description="Banking policy query to search in knowledge base",
        example="What are the prepayment penalty rules?"
    )
    strategy: str = Field(
        default="fixed",
        description="Chunking strategy: 'fixed' or 'sentence'",
        enum=["fixed", "sentence"]
    )
    top_k: int = Field(
        default=5,
        description="Number of documents to retrieve",
        ge=1,
        le=10
    )


class LoanLookupSchema(BaseModel):
    """Schema for loan application status lookup."""
    record_id: str = Field(
        ...,
        description="Loan application record ID (e.g., LOAN_0001)",
        pattern="^LOAN_\\d{4}$",
        example="LOAN_0001"
    )


# ============================================================================
# RAG Tool
# ============================================================================

_rag_pipeline = None

def get_rag_pipeline() -> RAGPipeline:
    """Get or initialize RAG pipeline (singleton)."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(backend="chromadb")
        _rag_pipeline.index_documents()
    return _rag_pipeline


@tool
def retrieve_banking_policy(
    query: Annotated[str, "Banking policy query to search in knowledge base"],
    strategy: Annotated[str, "Chunking strategy: 'fixed' or 'sentence'"] = "fixed",
    top_k: Annotated[int, "Number of documents to retrieve"] = 5
) -> str:
    """
    Retrieve relevant banking policy documents from the knowledge base.

    This tool searches the indexed banking policies using the recommended
    fixed-size chunking strategy and returns the top-k most relevant
    documents based on semantic similarity.

    Args:
        query: Banking policy question (e.g., "What are prepayment penalties?")
        strategy: Chunking strategy to use ('fixed' or 'sentence')
        top_k: Number of results to return

    Returns:
        Formatted string with retrieved documents and source citations
    """
    rag = get_rag_pipeline()

    try:
        results = rag.retrieve(query, strategy=strategy, top_k=top_k)

        if not results:
            return "No relevant banking policies found for this query."

        # Format results
        output = f"Retrieved {len(results)} relevant banking policy documents:\n\n"

        for i, result in enumerate(results, 1):
            doc_title = result["metadata"].get("doc_title", "Unknown")
            text_preview = result["text"][:150] + "..." if len(result["text"]) > 150 else result["text"]
            distance = result["distance"]

            output += f"{i}. {doc_title}\n"
            output += f"   Relevance (distance): {distance:.4f}\n"
            output += f"   Content: {text_preview}\n\n"

        return output

    except Exception as e:
        return f"Error retrieving banking policies: {str(e)}"


# ============================================================================
# Loan Lookup Tool
# ============================================================================

@tool
def check_application_status(
    record_id: Annotated[str, "Loan application record ID (format: LOAN_XXXX)"]
) -> str:
    """
    Check the status of a loan application and get escalation score.

    This tool retrieves the current status, loan amount, and computed
    escalation score for a given application. Escalation score combines
    fraud risk and recency signals to prioritize cases for review.

    Args:
        record_id: Loan application record ID (format: LOAN_XXXX)

    Returns:
        Formatted string with application details and escalation status
    """
    try:
        result = check_loan_application_status(record_id)

        if "error" in result:
            return f"Error: {result['error']}"

        # Format result
        output = f"Application Status: {record_id}\n"
        output += f"  Status: {result['status']}\n"
        output += f"  Loan Amount: INR {result['loan_amount_inr']:,}\n"
        output += f"  Days Since Created: {result['days_since_created']}\n"
        output += f"  Fraud Flagged: {result['flagged_for_fraud_review']}\n"
        output += f"  Escalation Score: {result['escalation_score']:.4f}\n"
        output += f"  Escalation Recommended: {'YES' if result['escalation_recommended'] else 'NO'}\n"

        return output

    except Exception as e:
        return f"Error checking application status: {str(e)}"


# ============================================================================
# Tool Registry
# ============================================================================

CREW_TOOLS = [
    retrieve_banking_policy,
    check_application_status,
]


if __name__ == "__main__":
    # Quick test of tool schemas
    print("Tool Definitions:")
    print(f"retrieve_banking_policy: {retrieve_banking_policy.description}")
    print(f"check_application_status: {check_application_status.description}")

