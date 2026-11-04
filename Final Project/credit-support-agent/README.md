# Credit Support Agent — Banking RAG + CrewAI Multi-Agent System

A comprehensive full-stack banking agent combining RAG retrieval, multi-agent orchestration, and FastAPI backend with structured evaluation.

## Project Structure

```
credit-support-agent/
├── .env.example                # Environment variables (CREWAI_DISABLE_TELEMETRY, etc.)
├── .gitignore                  # Git ignore patterns
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration for models, thresholds, costs
├── main.py                     # CLI entry point
│
├── data/
│   ├── raw/
│   │   └── knowledge_base_docs.json    # 12+ policy documents + guardrail test cases
│   └── generated/
│       └── loan_applications.json      # Generated dataset (≥40 records)
│
├── logs/
│   └── app.log                 # JSON-Lines audit log
│
├── src/
│   ├── ingestion/              # Data loading and generation
│   ├── chunking/               # Document chunking strategies
│   ├── embeddings/             # Embedding models
│   ├── vectordb/               # Vector database (ChromaDB)
│   ├── retrieval/              # Retrieval pipelines
│   ├── llm/                    # MockLLM implementation
│   ├── tools/                  # CrewAI tools
│   ├── agents/                 # CrewAI agents
│   ├── orchestration/          # Crew orchestration and memory
│   ├── guardrails/             # PII masking, injection detection, groundedness
│   ├── review/                 # AutoGen review stage
│   ├── governance/             # Autonomy, budgeting, caching
│   ├── prompts/                # Prompt templates
│   ├── schemas/                # Pydantic models
│   ├── api/                    # FastAPI routes and WebSocket
│   └── utils/                  # Logging, metrics, helpers
│
├── tests/
│   ├── test_dataset.py
│   ├── test_chunking_eval.py
│   ├── test_guardrails.py
│   ├── test_autogen_review.py
│   └── test_api.py
│
└── transcripts/                # Submission proof files
    ├── part1_calibration.txt
    ├── part1_precision_recall.txt
    ├── part2_multiturn.txt
    ├── part3_eval_15_queries.json
    └── part4_review_cases.txt
```

## Implementation Parts

### Part 1: RAG Foundation
- Dataset generation (≥40 seeded loan applications)
- Knowledge base loading (14 documents covering banking policies)
- Dual chunking strategies (fixed-size + sentence-based)
- ChromaDB collections with embeddings
- Retrieval evaluation (Precision/Recall metrics)
- **Grounded generation with empirically calibrated threshold**

### Groundedness Calibration Results

**Threshold Determination**:
Measured top-1 cosine distance for in-scope and out-of-scope queries to empirically determine the "I don't know" threshold.

**In-Scope Queries (Banking Domain)**:
| Query | Top-1 Distance |
|-------|-----------------|
| What is the minimum income requirement for a home loan? | 0.3245 |
| How is EMI calculated? | 0.3127 |
| What documents are needed for KYC? | 0.5356 |
| What is CIBIL score and why does it matter? | 0.5890 |
| What are the penalties for prepayment? | 0.3562 |

**In-Scope Statistics**: Min=0.3127, Max=0.5890, Avg=0.4236

**Out-of-Scope Queries (Non-Banking)**:
| Query | Top-1 Distance |
|-------|-----------------|
| What is the capital of France? | 0.8752 |
| How do I cook pasta? | 0.9082 |

**Out-of-Scope Statistics**: Min=0.8752, Max=0.9082, Avg=0.8917

**Separation**: CLEAR (gap of 0.2862 between max in-scope and min out-of-scope)

**Chosen Threshold**: **0.7321**
- Midpoint between in-scope max (0.5890) and out-of-scope min (0.8752)
- **Decision rule**: If top-1 distance > 0.7321 → respond "I don't know" | else → generate grounded answer
- Rationale: Empirically calibrated for this specific knowledge base; avoids untested preset values (0.5/0.6/0.7)

### Part 2: CrewAI Multi-Agent System
- MockLLM (BaseLLM subclass, deterministic, zero network calls)
- 3+ agents: Retrieval, Lookup, Composer
- Structured Pydantic outputs
- Multi-turn conversation memory
- 3 Guardrails: PII masking, prompt injection, groundedness validation

### Part 3: FastAPI Backend
- 2+ HTTP endpoints + 1 WebSocket endpoint
- Structured JSON-Lines logging with trace IDs
- 4-metric evaluation (Accuracy, Grounding, Completeness, Safety)
- 15 test queries with evaluation results

### Part 4: Production Hardening
- AutoGen 2-agent review (approve/revise)
- Least autonomy enforcement
- Cost budget capping
- Response caching with cache-hit demonstration

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Generate dataset
python main.py generate-dataset

# Run RAG evaluation
python main.py eval-rag

# Start API server
python main.py start-api

# Run tests
pytest tests/
```

## Key Files

- `src/llm/mock_llm_client.py` — MockLLM implementation
- `src/ingestion/dataset_generator.py` — Deterministic loan application generator
- `src/guardrails/` — Input/output validation
- `src/api/routes.py` — FastAPI endpoints
- `src/utils/metrics.py` — Evaluation scoring

## Reproduction

All results are deterministic and reproducible using:
- `MOCK_LLM_SEED=42` for seeded randomness
- Fixed ChromaDB collections per chunking strategy
- Deterministic evaluation metrics

See `transcripts/` for submission proof of all 4 parts.
