"""
Pydantic request and response schemas for FastAPI endpoints.

Defines:
- AskRequest: Query input with optional session_id
- AddDocumentRequest: Document ingestion input
- AddDocumentResponse: Ingestion confirmation
- HealthResponse: Service health status
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request model for POST /ask endpoint."""
    query: str = Field(..., description="User query to process", example="What is the status of LOAN_0001?")
    session_id: Optional[str] = Field(None, description="Optional session ID for multi-turn conversation", example="session_alpha")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the prepayment penalty rules?",
                "session_id": "session_alpha"
            }
        }


class AddDocumentRequest(BaseModel):
    """Request model for POST /add-document endpoint."""
    text: str = Field(..., description="Document text content", example="Prepayment penalties apply after 2 years...")
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional document metadata",
        example={"category": "prepayment", "source": "policy_manual"}
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Prepayment penalties apply after 2 years of the loan origination date.",
                "metadata": {"category": "prepayment", "source": "policy_manual"}
            }
        }


class AddDocumentResponse(BaseModel):
    """Response model for POST /add-document endpoint."""
    status: str = Field(..., description="Ingestion status", example="success")
    chunks_added: int = Field(..., description="Number of chunks created", example=3)
    message: str = Field(..., description="Confirmation message", example="Document ingested successfully")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "chunks_added": 3,
                "message": "Document ingested successfully into ChromaDB"
            }
        }


class HealthResponse(BaseModel):
    """Response model for GET /health endpoint."""
    status: str = Field(..., description="Service status", example="healthy")
    rag_index_size: int = Field(..., description="Number of documents in RAG index", example=14)
    service_version: str = Field(..., description="API version", example="1.0.0")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "rag_index_size": 14,
                "service_version": "1.0.0"
            }
        }


class WebSocketMessage(BaseModel):
    """WebSocket message model for bi-directional communication."""
    type: str = Field(..., description="Message type: 'user_query' or 'agent_response'", example="user_query")
    content: str = Field(..., description="Message content", example="What is the status of LOAN_0001?")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn conversation", example="session_alpha")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "user_query",
                "content": "What is the status of LOAN_0001?",
                "session_id": "session_alpha"
            }
        }
