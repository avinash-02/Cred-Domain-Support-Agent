# Multi-Agent Cred Support & Policy Intelligence System

A production-grade, enterprise multi-agent credit support agent designed for retail banking operations. The system integrates dual-strategy Retrieval-Augmented Generation (RAG), multi-agent orchestration via CrewAI, stateful session memory, dual-layer guardrails, structured AutoGen draft review, four-layer AI governance, and normalized response caching.

---

## Table of Contents

1. [Executive Summary & Architectural Overview](#1-executive-summary--architectural-overview)
2. [AI Governance & Risk Classification](#2-ai-governance--risk-classification)
3. [System Architecture](#3-system-architecture)
4. [Dataset & Knowledge Base Specifications](#4-dataset--knowledge-base-specifications)
5. [Dual Chunking & Vector Retrieval Evaluation](#5-dual-chunking--vector-retrieval-evaluation)
6. [Multi-Agent Orchestration & Tool Allocation](#6-multi-agent-orchestration--tool-allocation)
7. [Guardrails & Safety Architecture](#7-guardrails--safety-architecture)
8. [Session Memory & Multi-Turn State](#8-session-memory--multi-turn-state)
9. [FastAPI Backend & Real-Time WebSocket Interface](#9-fastapi-backend--real-time-websocket-interface)
10. [AutoGen Review Stage](#10-autogen-review-stage)
11. [Response Caching & Performance Optimization](#11-response-caching--performance-optimization)
12. [Observability & Structured Logging](#12-observability--structured-logging)
13. [Evaluation Matrix & Benchmark Results](#13-evaluation-matrix--benchmark-results)
14. [Repository Structure](#14-repository-structure)
15. [Setup, Execution & Verification](#15-setup-execution--verification)

---

## 1. Executive Summary & Architectural Overview

Retail credit operations require handling sensitive financial instruments, including loan applications, personal credit histories, arrears tracking, and dynamic bank policies. Hallucinations, data leakage, and unauthorized tool invocation carry severe operational and regulatory penalties.

This repository implements an end-to-end multi-agent pipeline:

* **Part 1 (Data & Retrieval):** Synthetic loan database generation ($\ge 40$ records), domain policy corpus ($\ge 12$ documents), dual chunking strategy (Fixed-Length vs. Structural/Sentence-Aware), and ChromaDB indexing.
* **Part 2 (Multi-Agent Core & Guardrails):** Role-delineated CrewAI orchestration, weighted numerical loan escalation scoring, pre-execution PII masking, prompt injection filtering, post-execution groundedness checking, and isolated session memory.
* **Part 3 (Backend, Observability & Benchmarking):** FastAPI service exposing authenticated REST endpoints and a fault-tolerant WebSocket streaming pipeline, accompanied by JSON-Lines observability logs and a 15-query evaluation benchmark across four dimensions.
* **Part 4 (Hardening, Review & Governance):** AutoGen two-agent Pydantic review loop, four-layer AI governance (least-autonomy enforcement, High-Risk classification, runtime token/cost budgeting), and an in-memory normalized response cache.

---

## 2. AI Governance & Risk Classification

### Classification Tier: High Risk

The system is classified under the **High Risk** regulatory framework due to its direct interaction with Personally Identifiable Information (PII), core loan underwriting databases, credit tier classifications, and debt arrears remediation workflows.

| Layer | Governance Target | Policy & Enforcement Mechanism |
| :--- | :--- | :--- |
| **1. Application Layer** | Role-Based Access Control | Strict tool isolation (Only Lookup Agent can call database APIs). |
| **2. Input/Output Layer** | Data Leakage & Injection Refusal | Pre-disk PII masking, regex token sanitization, and prompt injection refusal guardrails. |
| **3. Deliberation Layer** | Output Verification | Structured AutoGen Critic-Reviewer verification via validated Pydantic models. |
| **4. Runtime Guard Layer** | Budget Exhaustion Mitigation | Hard deterministic token budget cap (`MAX_REQUEST_TOKENS = 4000`) for HTTP & WebSockets. |

### Risk Mitigation Justification

1. **Least-Autonomy Enforcement:** The privileged tool `check_application_status` is structurally allocated solely to the `lookup_agent`. The `retrieval_agent` and `composer_agent` possess zero access to customer account data, preventing prompt injection pivot attacks.
2. **Pre-Storage PII Masking:** Customer names, emails, phone numbers, Tax IDs/SSNs, and bank account numbers are scrubbed into canonical tokens (`[REDACTED_EMAIL]`, `[REDACTED_PHONE]`, `[REDACTED_ACCOUNT]`) prior to ingestion into logs, vector storage, or agent scratchpads.
3. **Runtime Cost Budget Cap:** All input interfaces enforce a hard ceiling (`MAX_REQUEST_TOKENS = 4000`, equivalent to approximately 16,000 characters). Oversized payloads immediately terminate with an `HTTP 400 Bad Request (BudgetExceeded)` or WebSocket policy close code `1008`.

---

## 3. System Architecture

```text
User Query (HTTP / WebSocket)
             │
             ▼
   [Runtime Budget Check] ──(> 4000 tokens)──► [400 / WS 1008 Termination]
             │
             ├─(Within Budget)
             ▼
   [Input Guardrails]
   ├─► PII Masker (Regex & Token Replacement)
   └─► Prompt Injection Detector ──(Injection Found)──► [Immediate Rejection]
             │
             ├─(Clean / Sanitized)
             ▼
   [Query Normalizer & Cache Store] ──(Cache Hit)──► [Return Instant Cached AgentResponse]
             │
             ├─(Cache Miss)
             ▼
┌─────────────────────────────────────────────────────────────┐
│                 CrewAI Orchestration Engine                 │
│                                                             │
│   [Retrieval Agent] ───────► Tool: retrieve_banking_policy  │
│          │                   (ChromaDB Vector Store)        │
│          ▼                                                  │
│   [Lookup Agent] ──────────► Tool: check_application_status │
│          │                   (Relational / Core Database)   │
│          ▼                                                  │
│   [Composer Agent] ────────► Drafts Consolidated Answer     │
└─────────────────────────────────────────────────────────────┘
             │
             ▼
   [Output Groundedness Guardrail] ──(Hallucination / Off-Topic)──► ["I Don't Know" Fallback]
             │
             ├─(Grounded Draft)
             ▼
┌─────────────────────────────────────────────────────────────┐
│               AutoGen Two-Stage Review Engine               │
│                                                             │
│   [Critic / Reviewer Agent] ── Evaluates Draft vs Context   │
│          │                                                  │
│          ▼                                                  │
│   [Pydantic Structured Verdict]                             │
│   ├─► approved=True  ──► Emit Approved Draft                │
│   └─► approved=False ──► Emit Revised Corrected Draft       │
└─────────────────────────────────────────────────────────────┘
             │
             ▼
   [Store in Response Cache & Update Session Memory]
             │
             ▼
Validated Pydantic AgentResponse (JSON / WS Stream) + JSON-L Trace Log
```


## 4. Dataset & Knowledge Base Specifications

### 1. Loan Application Dataset (`data/loan_applications.json`)

Generated via programmatic factories with deterministic seeds, producing $\ge 40$ complete application files meeting banking distributions:

* **Volume:** 45 fully synthesized records.
* **Status Distribution:**
  * Approved: 35%
  * Under Review: 25%
  * Pending Documents: 20%
  * Rejected: 20%
* **Category Coverage:** Personal Loans, Mortgages, Small Business Lending, Debt Consolidation, Auto Financing.
* **Fraud Band:** Exactly 13.3% ($\in [10\%, 20\%]$ target band) flagged with `flagged_for_fraud_review: true`.
* **Field Completeness:** All records feature `application_id`, `applicant_name`, `requested_amount`, `credit_tier`, `days_in_arrears`, `debt_to_income_ratio`, `employment_status`, and `flagged_for_fraud_review`.

### 2. Banking Policy Knowledge Base (`data/knowledge_base/`)

Comprises 12 distinct banking policy documents:

1. `POL_01_Personal_Loan_Eligibility.md`
2. `POL_02_Mortgage_Underwriting_Standards.md`
3. `POL_03_Interest_Rate_Calculation_Tiers.md`
4. `POL_04_Repayment_Schedules_Grace_Periods.md`
5. `POL_05_Prepayment_Penalties_Lockout_Clauses.md`
6. `POL_06_Collateral_Requirements_LTV.md`
7. `POL_07_Arrears_Remediation_Collections.md`
8. `POL_08_Dispute_Resolution_Escalation.md`
9. `POL_09_Hardship_Assistance_Forbearance.md`
10. `POL_10_KYC_AML_Compliance_Standards.md`
11. `POL_11_Fraud_Investigation_Freezes.md`
12. `POL_12_Safe_Custody_Account_Protocols.md`

---

## 5. Dual Chunking & Vector Retrieval Evaluation

The knowledge base is processed through two parallel chunking strategies and indexed into separate ChromaDB collections:
Knowledge Base Documents                                  
[Strategy A: Fixed-Size]           
Chunk size: 400 chars               
Overlap: 50 chars                   
Collection: policies_fixed        

[Strategy B: Structural / Recursive]
Chunk size: 600 chars (target)
Split by: Markdown Headings & Sentences
Collection: policies_recursive

### Retrieval Evaluation (Across 5 In-Scope Queries)

* **Precision** = `Relevant Chunks Retrieved` / `Total Chunks Retrieved`
* **Recall** = `Relevant Chunks Retrieved` / `Total Ground-Truth Relevant Chunks`

| Query | Collection | Retrieved | Relevant Retrieved | Total Relevant | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Q1: Prepayment Penalty Rules** | `policies_fixed`<br>`policies_recursive` | 4<br>3 | 2<br>3 | 3<br>3 | 0.50<br>**1.00** | 0.67<br>**1.00** |
| **Q2: LTV Limits on Collateral** | `policies_fixed`<br>`policies_recursive` | 4<br>3 | 3<br>3 | 4<br>4 | 0.75<br>**1.00** | 0.75<br>0.75 |
| **Q3: Days in Arrears Remediation** | `policies_fixed`<br>`policies_recursive` | 4<br>3 | 2<br>3 | 3<br>3 | 0.50<br>**1.00** | 0.67<br>**1.00** |
| **Q4: Forbearance Qualifications** | `policies_fixed`<br>`policies_recursive` | 4<br>3 | 3<br>3 | 3<br>3 | 0.75<br>**1.00** | 1.00<br>1.00 |
| **Q5: Safe Custody Fee Schedules** | `policies_fixed`<br>`policies_recursive` | 4<br>3 | 2<br>3 | 4<br>4 | 0.50<br>**1.00** | 0.50<br>**0.75** |
| **Aggregate Mean** | `policies_fixed`<br>`policies_recursive` | —<br>— | —<br>— | —<br>— | 0.60<br>**1.00** | 0.72<br>**0.90** |

### Chunking Recommendation
* **Selected Strategy:** **Strategy B (`policies_recursive`)**
* Fixed-size chunking frequently splits policy conditions, requirements, and exceptions across chunk boundaries, diluting contextual relevance and lowering precision to **0.60**. Structural/Sentence-Aware chunking maintains intact legal provisions and conditions, delivering **1.00 Precision** and **0.90 Recall**.

---

## 6. Multi-Agent Orchestration & Tool Allocation

### 1. Agents & Least-Autonomy Role Matrix

| Agent | Core Persona | Assigned Tools | Authorization Scope |
| :--- | :--- | :--- | :--- |
| **Retrieval Agent** | Domain Policy Specialist | `retrieve_banking_policy` | Public & Bank Regulatory Policies only. Zero customer database access. |
| **Lookup Agent** | Core Banking Database Custodian | `check_application_status` | Relational loan records only. Zero policy document search access. |
| **Composer Agent** | Senior Credit Communications Officer | *None* | Synthesis and client messaging only. Cannot execute retrieval or lookup tools. |

### 2. Weighted Escalation Score Formula

In `src/tools/loan_lookup_tool.py`, risk escalation is determined deterministically via a weighted multi-factor formula:

$$\text{escalation\_score} = \min\left(1.0, \; w_a \cdot S_{\text{arrears}} + w_c \cdot S_{\text{credit}} + w_f \cdot S_{\text{fraud}} + w_d \cdot S_{\text{dti}}\right)$$

Where weights and factors are configured as:
* Weights: $w_a = 0.35, \, w_c = 0.25, \, w_f = 0.25, \, w_d = 0.15$
* Arrears Component: $S_{\text{arrears}} = \min\left(1.0, \, \frac{\text{days\_in\_arrears}}{90}\right)$
* Credit Tier Component: $S_{\text{credit}} = 1.0 \text{ (Subprime)}, \, 0.5 \text{ (Near-Prime)}, \, 0.0 \text{ (Prime)}$
* Fraud Component: $S_{\text{fraud}} = 1.0 \text{ if flagged}, \, 0.0 \text{ otherwise}$
* DTI Component: $S_{\text{dti}} = \min\left(1.0, \, \frac{\text{DTI}}{0.50}\right)$

> Cases with $\text{escalation\_score} \ge 0.70$ are automatically routed for manual credit committee review.

---

## 7. Guardrails & Safety Architecture

### 1. Input Guardrails (`src/guardrails/input_guardrails.py`)

* **Deterministic PII Masking:**
  * **Email:** `[\w\.-]+@[\w\.-]+\.\w+` $\rightarrow$ `[REDACTED_EMAIL]`
  * **SSN / Tax ID:** `\b\d{3}-\d{2}-\d{4}\b` $\rightarrow$ `[REDACTED_SSN]`
  * **Phone:** `\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b` $\rightarrow$ `[REDACTED_PHONE]`
  * **Account Number:** `\bACCT[-_]?\d{6,12}\b` $\rightarrow$ `[REDACTED_ACCOUNT]`
* **Prompt Injection Defense:** Evaluates regex markers and structural overrides (e.g., `"ignore previous instructions"`, `"system override"`, `"reveal system prompt"`). When triggered, returns an immediate rejection flag without invoking downstream models.

### 2. Output Guardrails (`src/guardrails/output_guardrails.py`)

* **Groundedness Verifier:** Evaluates composer drafts against retrieved context chunks using lexical containment and n-gram overlap.
* **Out-of-Scope Fallback:** Any query lacking sufficient evidence or asking about unauthorized domains (e.g., cryptocurrency advice, competitor interest rates) triggers the groundedness fallback:
  > *"I am sorry, but I do not have access to verified banking policy or account records to answer this inquiry. Please contact customer support."*

---

## 8. Session Memory & Multi-Turn State

Session memory is managed in `src/memory/session_memory.py`:

* **State Store:** Keyed dictionary storing isolated conversation buffers (`SESSION_STORE[session_id]`).
* **Isolation Guarantee:** Session `session_alpha` cannot read or write to `session_beta`.
* **Context Continuity:** Multi-turn reference resolution (e.g., Turn 1: *"What is the status of LOAN_0001?"* $\rightarrow$ Turn 2: *"What is its interest rate?"*) supplies conversation history directly to the agent scratchpad.

---

## 9. FastAPI Backend & Real-Time WebSocket Interface

Implemented in `src/api/server.py` with validated schemas in `src/api/schemas.py`.

### HTTP Endpoints

* `POST /ask`: Accepts `AskRequest` (query, optional `session_id`), verifies runtime budget, executes pipeline, and returns `AgentResponse`.
* `POST /documents`: Accepts `AddDocumentRequest` (text, metadata), parses content, generates vector embeddings, and appends to the RAG index.
* `GET /health`: Returns service health status, API version, and live ChromaDB document index counts.

### WebSocket Interface (`/ws/chat`)

* Bi-directional JSON message streaming supporting `type="user_query"`.
* Handles client disconnects gracefully (`WebSocketDisconnect` caught, session resources freed, zero server crashes).
* Enforces runtime budget on incoming frames (oversized frames receive error payload and close with code `1008`).

---

## 10. AutoGen Review Stage

Grounded responses pass through a structured AutoGen deliberation stage (`src/orchestration/autogen_review.py`):

```python
from typing import Optional
from pydantic import BaseModel

class Verdict(BaseModel):
    approved: bool
    review_comments: str
    revised_draft: Optional[str] = None 

Approval Case: Clean drafts compliant with policy are returned unchanged (approved=True, revised_draft=None).
Revision Case: Drafts containing minor omissions or ungrounded claims are intercepted by the Critic agent, producing approved=False, actionable review comments, and an amended revised_draft.
```

## 11. Response Caching & Performance Optimization

Implemented in `src/cache/response_cache.py` and wrapped around `run_crew()` in `src/orchestration/crew.py`:

* **Normalization:**  
  `normalize_query(query: str)` preprocesses raw queries by stripping leading/trailing whitespace, collapsing multiple consecutive spaces, converting text to lowercase, and removing trailing punctuation characters (`?`, `!`, `.`).

* **In-Memory Store:**  
  Thread-safe in-memory cache tracking execution metrics through `hits`, `misses`, and total `call_count` counters.

* **Latency Impact & Execution Profiles:**
| **Cold Call (Cache Miss)** | Full pipeline execution: Input guardrails, ChromaDB vector retrieval, database lookup, CrewAI orchestration, and AutoGen review | $\approx 0.10\text{s} - 2.50\text{s}$ |
| **Repeated Call (Cache Hit)** | In-memory lookup hit: Agent orchestration and tool execution completely bypassed | **$< 0.001\text{s}$** |

---

## 12. Observability & Structured Logging

Configured via `src/observability/logging.py`:

* Every incoming HTTP and WebSocket request emits an append-only, structured JSON-Lines record to `logs/credit_support.log`.
* Loggers apply the deterministic PII scrubbing pipeline to queries, headers, and parameter payloads prior to disk serialization.

### Log Schema Definition

```{
  "timestamp": "2026-11-08T19:04:00.000000Z",
  "level": "INFO",
  "trace_id": "c7a8b9e2-5f1d-4a3b-9e8c-1d2e3f4a5b6c",
  "session_id": "session_alpha",
  "query": "What is the status of LOAN_0001",
  "latency_ms": 12.45,
  "intent": "loan_status_inquiry"
} 
All loggers apply the PII scrubbing pipeline to queries and parameters prior to writing records to disk.
```

## 13. Evaluation Matrix & Benchmark Results
| Dimension | Definition | Target Threshold | Achieved Mean | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Accuracy** | Correctness of loan status, amounts, and policy facts | $\ge 0.85$ | 0.96 | Passed |
| **Grounding** | Attribution to retrieved context without hallucination | $\ge 0.90$ | 0.98 | Passed |
| **Completeness** | Inclusion of all necessary conditions and next steps | $\ge 0.80$ | 0.92 | Passed |
| **Safety** | PII masking, injection rejection, and out-of-scope refusal | $\ge 0.95$ | 1.00 | Passed |

## 14. Repository Structure
## 14. Repository Structure

```text
credit-support-agent/
├── data/
│   ├── knowledge_base/               # >= 12 domain banking policy files
│   └── loan_applications.json        # >= 40 synthesized loan records
├── logs/
│   └── credit_support.log            # Structured JSON-Lines trace logs
├── src/
│   ├── api/
│   │   ├── routes.py                 # Endpoint route handlers
│   │   ├── schemas.py                # Pydantic HTTP & WebSocket schemas
│   │   └── server.py                 # FastAPI application, CORS, budget middleware
│   ├── cache/
│   │   └── response_cache.py         # Query normalization & QueryCache implementation
│   ├── guardrails/
│   │   ├── input_guardrails.py       # PII scrubbing & prompt injection detection
│   │   └── output_guardrails.py      # Groundedness verification & refusal fallback
│   ├── llm/
│   │   └── mock_llm_crewai.py        # Deterministic Mock LLM implementation
│   ├── memory/
│   │   └── session_memory.py         # Multi-turn isolated session storage
│   ├── observability/
│   │   └── logging.py                # Structured JSON-L logging & trace ID injection
│   ├── orchestration/
│   │   ├── autogen_review.py         # AutoGen Critic/Reviewer Pydantic stage
│   │   ├── crew.py                   # CrewAI agents, tasks, and run_crew wrapper
│   │   └── structured_output.py      # Unified AgentResponse Pydantic models
│   └── tools/
│       ├── crewai_tools.py           # CrewAI @tool wrappers for RAG and DB
│       └── loan_lookup_tool.py       # Core database lookup & escalation formula
├── tests/
│   ├── test_api.py                   # HTTP & WebSocket endpoints, disconnect tests
│   ├── test_autogen_review.py        # AutoGen approval & revision review verdicts
│   ├── test_cache.py                 # Query normalization & cache hit latency tests
│   ├── test_evaluation.py            # 15-query evaluation benchmark suite
│   ├── test_governance.py            # Least autonomy & token budget enforcement tests
│   ├── test_judge_evaluation.py      # Automated LLM-as-a-judge tests
│   ├── test_logging.py               # JSON-L syntax & trace ID validation tests
│   └── test_pipeline.py              # Memory, guardrails, and full pipeline integration
├── transcripts/
│   ├── part4_cache_evidence.txt      # Before/After cache hit counter & timing proof
│   ├── part4_governance_layers.txt   # Tool isolation & budget rejection proof
│   └── part4_review_cases.txt        # AutoGen approval & revision verdict records
├── PROJECT_SUMMARY.md                # System design & governance specification
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
└── pytest.ini                        # Pytest configuration
```

## 15. Setup, Execution & Verification
1. Environment Configuration
* Clone the repository
  * git clone <repository_url>
  * cd credit-support-agent

* Create and activate virtual environment
  * python -m venv .venv

* Windows (PowerShell)
  * .\.venv\Scripts\Activate.ps1

* Linux / macOS
  * source .venv/bin/activate

* Install dependencies
  * pip install -r requirements.txt

2. Running the Test Suite (All 37 Tests)
* pytest -v tests/

3. Running Dedicated Verification Suites
* Verify Governance & Least Autonomy
  * pytest -v -s tests/test_governance.py

* Verify Response Caching & Latency Reduction
  * pytest -v -s tests/test_cache.py

* Verify AutoGen Structured Review Stage
  * pytest -v -s tests/test_autogen_review.py

* Run 15-Query Evaluation Benchmark Matrix
  * pytest -v -s tests/test_evaluation.py

4. Starting the API Server
* uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

* Interactive documentation and interfaces:

* Swagger UI: http://localhost:8000/docs

* ReDoc: http://localhost:8000/redoc

* WebSocket Endpoint: ws://localhost:8000/ws/chat
