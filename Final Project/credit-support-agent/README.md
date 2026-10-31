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
