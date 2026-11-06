"""
Structured output schema for crew responses.

Every crew response must conform to AgentResponse BaseModel for validation.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Unified schema that every crew response must conform to."""

    query: str = Field(
        description="The user query processed by the crew"
    )
    intent: Literal["policy_inquiry", "application_status", "mixed", "general"] = Field(
        description="Classified query intent based on query content"
    )
    answer: str = Field(
        description="The final synthesized answer from the crew"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Chunk IDs, document references, loan record IDs, or tool names used"
    )
    escalation_required: bool = Field(
        default=False,
        description="True if escalation threshold exceeded or fraud flagged"
    )
    escalation_reason: Optional[str] = Field(
        default=None,
        description="Reason for escalation if required (e.g., fraud flag, high recency score)"
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence or groundedness indicator (1.0 = full confidence)"
    )

    class Config:
        """Pydantic config."""
        frozen = False
        json_schema_extra = {
            "example": {
                "query": "What is the status of application LOAN_0001?",
                "intent": "application_status",
                "answer": "Application Status: LOAN_0001\n  Status: under_review\n  Escalation Recommended: YES",
                "sources": ["LOAN_0001", "check_application_status"],
                "escalation_required": True,
                "escalation_reason": "Fraud flagged and escalation score (0.6400) exceeds threshold (0.60)",
                "confidence_score": 1.0,
            }
        }


def classify_intent(query: str) -> Literal["policy_inquiry", "application_status", "mixed", "general"]:
    """Classify query intent based on content."""
    query_lower = query.lower()

    # Check for mixed intent
    has_policy = any(w in query_lower for w in ["penalty", "prepayment", "eligibility", "emi", "cibil", "kyc", "policy", "requirement", "criteria"])
    has_lookup = any(w in query_lower for w in ["status", "application", "loan_", "record", "escalation", "fraud"])

    if has_policy and has_lookup:
        return "mixed"
    elif has_lookup:
        return "application_status"
    elif has_policy:
        return "policy_inquiry"
    else:
        return "general"


def format_crew_response_as_structured(
    query: str,
    crew_answer: str,
    sources: List[str] = None,
    escalation_required: bool = False,
    escalation_reason: Optional[str] = None,
    confidence_score: float = 1.0,
) -> AgentResponse:
    """
    Format crew output into validated AgentResponse schema.

    Args:
        query: Original user query
        crew_answer: Raw answer from crew.kickoff()
        sources: List of sources/tools/records used
        escalation_required: Whether escalation is triggered
        escalation_reason: Reason for escalation if any
        confidence_score: Confidence indicator

    Returns:
        Validated AgentResponse instance
    """
    intent = classify_intent(query)

    if sources is None:
        sources = []

    response = AgentResponse(
        query=query,
        intent=intent,
        answer=crew_answer,
        sources=sources,
        escalation_required=escalation_required,
        escalation_reason=escalation_reason,
        confidence_score=confidence_score,
    )

    # Validate against schema
    return AgentResponse.model_validate(response.model_dump())
