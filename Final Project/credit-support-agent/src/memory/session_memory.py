"""
Session-based conversation memory using LangChain's InMemoryChatMessageHistory.

Maintains conversation history across turns within a single process run.
Uses RunnableWithMessageHistory for proper session state management.
"""

import os
import re
from typing import Dict, Any, Optional
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

import sys
sys.path.insert(0, '.')

# Session registry: in-process storage only (does not survive restart)
SESSION_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Retrieve or create in-memory chat message history for a given session."""
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return SESSION_STORE[session_id]


def extract_record_id(text: str) -> Optional[str]:
    """Extract LOAN_XXXX pattern from text."""
    match = re.search(r"\b(LOAN_\d{4})\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def get_last_referenced_loan_id(history: Any) -> Optional[str]:
    """Find the most recent loan ID in conversation history."""
    # history can be a list of messages or InMemoryChatMessageHistory
    messages = history if isinstance(history, list) else (history.messages if history else [])
    for msg in reversed(messages):
        rec_id = extract_record_id(msg.content)
        if rec_id:
            return rec_id
    return None


def orchestrator_step(inputs: Dict[str, Any]) -> str:
    """
    Core step invoked by RunnableWithMessageHistory.

    Demonstrates stateful orchestration:
    1. Look for explicit ID in query
    2. If absent, resolve from session memory
    3. Handle status/escalation inquiries
    4. Fall back to general policy
    """
    user_query = inputs.get("query", "")
    history = inputs.get("history")

    # 1. Look for explicit ID in query
    record_id = extract_record_id(user_query)

    # 2. If absent, resolve from session memory
    if not record_id and history:
        record_id = get_last_referenced_loan_id(history)
        if record_id:
            print(f"   [Memory] Resolved loan ID from history: {record_id}")

    # 3. Handle status / escalation inquiries
    if any(w in user_query.lower() for w in ["status", "escalat", "recommend", "score"]):
        if not record_id:
            return "Error: No loan application ID provided or found in session context. Please specify a loan ID (e.g., LOAN_0001)."

        from src.tools.loan_lookup_tool import check_loan_application_status
        res = check_loan_application_status(record_id)
        if "error" in res:
            return f"Application {record_id} not found."

        rec_str = "YES" if res["escalation_recommended"] else "NO"
        return (
            f"Application Details for {record_id}:\n"
            f"  Status: {res['status']}\n"
            f"  Loan Amount: INR {res['loan_amount_inr']:,}\n"
            f"  Days Since Created: {res['days_since_created']}\n"
            f"  Fraud Flagged: {res['flagged_for_fraud_review']}\n"
            f"  Escalation Score: {res['escalation_score']:.4f}\n"
            f"  Escalation Recommended: {rec_str}"
        )

    # General / Policy fallback
    return "Policy details retrieved: Please refer to standard bank policy guidelines."


# Build the stateful runnable
runnable = RunnableLambda(orchestrator_step)

conversational_agent = RunnableWithMessageHistory(
    runnable,
    get_session_history=get_session_history,
    input_messages_key="query",
    history_messages_key="history",
)


if __name__ == "__main__":
    # Test the session memory directly
    print("\n" + "=" * 80)
    print("SESSION MEMORY TEST")
    print("=" * 80)

    # Create and test session_alpha
    session_alpha = get_session_history("session_alpha")
    print(f"\n✓ Session 'session_alpha' created")
    print(f"  Initial message count: {len(session_alpha.messages)}")

    # Add messages
    from langchain_core.messages import HumanMessage, AIMessage
    session_alpha.add_message(HumanMessage(content="What is the status of LOAN_0001?"))
    session_alpha.add_message(AIMessage(content="Status retrieved..."))

    print(f"  After 2 messages: {len(session_alpha.messages)}")
    print(f"  Messages: {[m.type for m in session_alpha.messages]}")

    # Test fresh session
    session_beta = get_session_history("session_beta")
    print(f"\n✓ Session 'session_beta' created (fresh)")
    print(f"  Initial message count: {len(session_beta.messages)}")

    print("\n" + "=" * 80 + "\n")
