"""
FastAPI server for Credit Support Agent.

Exposes:
- POST /ask: Query the guarded crew pipeline
- POST /add-document: Ingest documents into RAG index
- GET /health: Service health status
- WebSocket /ws/chat: Real-time multi-turn conversation with session state
"""

import os
import sys
sys.path.insert(0, '.')

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

import logging
import re
from typing import Optional
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from src.observability.logging import StructuredLoggingMiddleware

from src.api.schemas import (
    AskRequest,
    AddDocumentRequest,
    AddDocumentResponse,
    HealthResponse,
    WebSocketMessage,
)
from src.orchestration.crew import run_crew_async
from src.orchestration.structured_output import AgentResponse
from src.guardrails.pipeline_guardrails import apply_input_guardrails, apply_output_guardrails, sanitize_for_logging, mask_pii_for_logging
from src.retrieval.rag_pipeline import RAGPipeline
from src.memory.session_memory import get_session_history, SESSION_STORE
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================================
# Logging Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Global RAG Pipeline — lazily initialised on first use so that both
# uvicorn and the TestClient (which skips lifespan by default) share the
# same indexed pipeline without import-time side effects.
# ============================================================================

rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    """Get or lazily initialise the RAG pipeline."""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline(backend="chromadb")
        rag_pipeline.index_documents()
        logger.info("RAG pipeline lazily initialised")
    return rag_pipeline


def _get_rag_doc_count() -> int:
    """Return total chunks across both collections."""
    stats = get_rag_pipeline().get_stats()
    fixed_count = stats.get("fixed", {}).get("total_chunks", 0)
    sentence_count = stats.get("sentence", {}).get("total_chunks", 0)
    return fixed_count + sentence_count


# ============================================================================
# FastAPI Application
# ============================================================================

from src.observability.logging import StructuredLoggingMiddleware

app = FastAPI(
    title="Credit Support Agent API",
    description="Multi-agent credit support system with RAG, session memory, and guardrails",
    version="1.0.0",
)

# Add structured logging middleware
app.add_middleware(StructuredLoggingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HTTP Endpoints
# ============================================================================

@app.post("/ask", response_model=AgentResponse)
async def ask(request: AskRequest) -> AgentResponse:
    """
    Query the guarded crew pipeline.

    Accepts a user query and optional session_id, executes the guarded crew
    pipeline, and returns the validated AgentResponse.
    """
    try:
        logger.info(f"Received query: {sanitize_for_logging(request.query[:100])}... (session: {request.session_id})")

        # Apply input guardrails
        sanitized_query, injection_rejection = apply_input_guardrails(request.query)

        # If injection detected, return rejection immediately
        if injection_rejection is not None:
            logger.warning(f"Prompt injection detected in query (session: {sanitize_for_logging(request.session_id)})")
            return injection_rejection

        # If session_id provided, store the query in session memory
        if request.session_id:
            session = get_session_history(request.session_id)
            session.add_message(HumanMessage(content=sanitized_query))
            logger.info(f"Query added to session {sanitize_for_logging(request.session_id)}")

        # Run the crew pipeline using async wrapper
        response = await run_crew_async(sanitized_query)

        # Apply output guardrails (groundedness check)
        rag = get_rag_pipeline()
        retrieved_contexts = rag.retrieve(sanitized_query, strategy="fixed", top_k=5)
        response = apply_output_guardrails(response, retrieved_contexts)

        # Store response in session memory
        if request.session_id:
            session = get_session_history(request.session_id)
            session.add_message(AIMessage(content=response.answer))
            logger.info(f"Response added to session {sanitize_for_logging(request.session_id)}")

        logger.info(f"Query processed successfully. Intent: {response.intent}, session: {sanitize_for_logging(request.session_id)}")
        return response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/add-document", response_model=AddDocumentResponse)
async def add_document(request: AddDocumentRequest) -> AddDocumentResponse:
    """
    Ingest a document into the RAG index.

    Accepts document payload, chunks and embeds it into our ChromaDB RAG
    index, and returns ingestion confirmation.
    """
    try:
        logger.info(f"Received document ingestion request (text length: {len(request.text)})")

        rag = get_rag_pipeline()

        # Chunk the document using the pipeline's own fixed chunker
        from src.chunking.strategies import FixedSizeChunker
        chunker = FixedSizeChunker(chunk_size=256, overlap=50)

        # FixedSizeChunker.chunk expects (doc_id, doc_title, content)
        doc_id = f"api_doc_{_get_rag_doc_count() + 1}"
        doc_title = (request.metadata or {}).get("title", "API Ingested Document")
        # Mask text before processing/logging
        sanitized_text, _ = mask_pii_for_logging(request.text)
        chunks = chunker.chunk(doc_id, doc_title, sanitized_text)

        # Embed chunks
        chunk_texts = [c.text for c in chunks]
        embeddings = rag.embedding_gen.embed(chunk_texts)
        metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_title": c.doc_title,
                "strategy": c.strategy,
                "source": "api_ingestion",
                **(request.metadata or {}),
            }
            for c in chunks
        ]

        # Upsert into the fixed-chunk collection
        rag.vectordb.upsert_chunks(
            collection_name=rag.fixed_collection_name,
            chunk_ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            texts=chunk_texts,
            metadatas=metadatas,
        )

        chunks_added = len(chunks)
        logger.info(f"Document ingested successfully. Chunks added: {chunks_added}")

        return AddDocumentResponse(
            status="success",
            chunks_added=chunks_added,
            message=f"Document ingested successfully into ChromaDB ({chunks_added} chunks)",
        )

    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Service health status."""
    try:
        doc_count = _get_rag_doc_count()
        logger.info(f"Health check: {doc_count} chunks in index")

        return HealthResponse(
            status="healthy",
            rag_index_size=doc_count,
            service_version="1.0.0",
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    Real-time multi-turn conversation with session state.

    Maintains session state across messages within a connection using LangChain
    InMemoryChatMessageHistory. Handles graceful disconnection.
    """
    await websocket.accept()
    session_id = None

    try:
        logger.info("WebSocket client connected")

        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = WebSocketMessage.model_validate_json(data)
            except Exception as e:
                logger.error(f"Invalid message format: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "error": f"Invalid message format: {str(e)}"
                })
                continue

            # Extract session_id if provided
            if message.session_id:
                session_id = message.session_id

            logger.info(f"WebSocket message received (type: {message.type}, session: {sanitize_for_logging(session_id)})")

            # Process user query
            if message.type == "user_query":
                try:
                    # Apply input guardrails
                    sanitized_query, injection_rejection = apply_input_guardrails(message.content)

                    if injection_rejection is not None:
                        logger.warning("Prompt injection detected in WebSocket message")
                        await websocket.send_json({
                            "type": "agent_response",
                            "content": injection_rejection.answer,
                            "intent": injection_rejection.intent,
                            "confidence_score": injection_rejection.confidence_score,
                        })
                        continue

                    # Store in session memory
                    if session_id:
                        session = get_session_history(session_id)
                        session.add_message(HumanMessage(content=sanitized_query))

                    # Run crew pipeline
                    response = await run_crew_async(sanitized_query)

                    # Apply output guardrails
                    rag = get_rag_pipeline()
                    retrieved_contexts = rag.retrieve(sanitized_query, strategy="fixed", top_k=5)
                    response = apply_output_guardrails(response, retrieved_contexts)

                    # Store response in session memory
                    if session_id:
                        session = get_session_history(session_id)
                        session.add_message(AIMessage(content=response.answer))

                    # Send response back to client
                    await websocket.send_json({
                        "type": "agent_response",
                        "content": response.answer,
                        "intent": response.intent,
                        "confidence_score": response.confidence_score,
                        "escalation_required": response.escalation_required,
                        "escalation_reason": response.escalation_reason,
                        "session_id": session_id,
                    })

                    logger.info(f"WebSocket response sent (intent: {response.intent}, session: {sanitize_for_logging(session_id)})")

                except Exception as e:
                    logger.error(f"Error processing WebSocket query: {str(e)}", exc_info=True)
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Query processing failed: {str(e)}",
                    })

            else:
                logger.warning(f"Unknown message type: {message.type}")
                await websocket.send_json({
                    "type": "error",
                    "error": f"Unknown message type: {message.type}",
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected (session: {sanitize_for_logging(session_id)})")
        # Client disconnected gracefully — no action needed

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "error": f"Server error: {str(e)}",
            })
        except Exception:
            pass
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
