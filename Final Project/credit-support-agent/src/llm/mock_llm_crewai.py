"""
MockLLM for CrewAI: Deterministic, offline, zero-network language model.

Subclasses crewai.llms.base_llm.BaseLLM for seamless CrewAI integration.
Uses schema-based tool dispatch and deterministic reasoning.
"""

import os
import re
import json
import inspect
from typing import Optional, Any, get_type_hints

# Disable CrewAI telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

from crewai.llms.base_llm import BaseLLM


class MockLLM(BaseLLM):
    """
    Deterministic MockLLM for CrewAI agents.

    - Zero network calls
    - Deterministic responses based on input patterns
    - Schema-based tool dispatch (not substring matching)
    - Avoids ReAct "Observation:" parsing trap
    """

    model: str = "mock-llm-crewai"

    def __init__(self, model_name: str = "mock-llm-crewai", **kwargs):
        """Initialize MockLLM."""
        kwargs["model"] = model_name
        super().__init__(**kwargs)

    def call(
        self,
        messages: list[dict],
        stop: Optional[list[str]] = None,
        tools: Optional[list[Any]] = None,
        **kwargs
    ) -> str:
        """
        Generate a mock response or tool call.

        Args:
            messages: Conversation history
            stop: Stop sequences (unused)
            tools: Available tools with schemas
            **kwargs: Additional arguments

        Returns:
            Response string (plain text or JSON tool call)
        """
        # Get the last user message
        user_message = self._get_last_user_message(messages)

        if not user_message:
            return "I need more information to assist you."

        # If tools are available, decide whether to use one
        if tools:
            tool_call = self._should_call_tool(user_message, tools)
            if tool_call:
                return tool_call

        # Otherwise, generate a direct response
        return self._generate_response(user_message)

    def _get_last_user_message(self, messages: list[dict]) -> str:
        """Extract the last user message from history."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "").lower()
        return ""

    def _should_call_tool(self, user_message: str, tools: list[Any]) -> Optional[str]:
        """
        Decide if a tool should be called and return the tool call JSON.

        Uses schema-based dispatch (inspects function parameters) instead of substring matching.
        This is mandated by the project brief to avoid name-matching anti-patterns.
        """
        for tool in tools:
            tool_name = tool.name if hasattr(tool, "name") else str(tool)

            # Get the underlying function and inspect its parameters
            func = tool.func if hasattr(tool, "func") else tool

            try:
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
            except:
                continue

            # Dispatch by parameter schema, not tool name
            # Check for record_id parameter (loan lookup tool)
            if "record_id" in param_names and ("status" in user_message or ("loan" in user_message and "application" in user_message)):
                # This is the loan lookup tool - extract record ID from message
                record_id = self._extract_record_id(user_message)
                if record_id:
                    return self._tool_call_json(tool_name, {"record_id": record_id})

            # Check for query parameter (RAG retrieval tool)
            elif "query" in param_names and any(word in user_message for word in ["penalty", "prepayment", "eligibility", "emi", "policy", "requirement", "cibil", "kyc", "criteria"]):
                # This is RAG tool - use the full user message as query
                return self._tool_call_json(tool_name, {"query": user_message, "strategy": "fixed", "top_k": 5})

        return None

    def _extract_record_id(self, message: str) -> Optional[str]:
        """Extract LOAN_XXXX record ID from message."""
        match = re.search(r"(loan_\d{4})", message, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    def _tool_call_json(self, tool_name: str, arguments: dict) -> str:
        """Format tool call as JSON for CrewAI ReAct."""
        return json.dumps({
            "tool": tool_name,
            "tool_input": arguments,
        })

    def _generate_response(self, user_message: str) -> str:
        """Generate a deterministic response based on query content."""

        # Query B: Loan Status Lookup - return actual loan data
        if "status" in user_message and ("loan" in user_message or "application" in user_message):
            # Extract and look up the actual loan record
            record_id = self._extract_record_id(user_message)
            if record_id:
                from src.tools.loan_lookup_tool import check_loan_application_status
                result = check_loan_application_status(record_id)
                if "error" not in result:
                    return (
                        f"Application Status: {record_id}\n"
                        f"  Status: {result['status']}\n"
                        f"  Loan Amount: INR {result['loan_amount_inr']:,}\n"
                        f"  Days Since Created: {result['days_since_created']}\n"
                        f"  Fraud Flagged: {result['flagged_for_fraud_review']}\n"
                        f"  Escalation Score: {result['escalation_score']:.4f}\n"
                        f"  Escalation Recommended: {'YES' if result['escalation_recommended'] else 'NO'}"
                    )

        # Prepayment penalties
        if "penalty" in user_message or "prepayment" in user_message:
            return (
                "Prepayment penalties vary by loan type. Home loans have no penalty "
                "after 5 years; 1-2% penalty before. Personal loans have 2-4% penalty "
                "regardless of tenure. Auto loans have 1% penalty if prepaid before 3 years."
            )

        # Eligibility
        elif "eligible" in user_message or "eligibility" in user_message or "criteria" in user_message:
            return (
                "Loan eligibility depends on income, CIBIL score, and employment stability. "
                "Personal loans require minimum ₹3,00,000 annual income and CIBIL ≥650. "
                "Home loans require ₹5,00,000 income and CIBIL ≥700."
            )

        # EMI
        elif "emi" in user_message:
            return (
                "EMI is calculated as: EMI = P × r × (1+r)^n / ((1+r)^n - 1), "
                "where P is principal, r is monthly interest rate, and n is tenure in months."
            )

        # CIBIL
        elif "cibil" in user_message:
            return (
                "CIBIL score ranges from 300-900 and is based on: payment history (35%), "
                "credit utilization (30%), credit mix (15%), hard inquiries (10%), "
                "and age of credit (10%)."
            )

        # KYC
        elif "kyc" in user_message or "document" in user_message:
            return (
                "KYC requires identity proof (Aadhaar, PAN, passport), address proof "
                "(utility bill, rental agreement), and income proof (IT returns, salary slips)."
            )

        # Default
        else:
            return (
                "I can help with banking policy questions (penalties, eligibility, EMI, CIBIL, KYC) "
                "or loan application lookups. What would you like to know?"
            )

    @property
    def type(self) -> str:
        """Return LLM type."""
        return self.model


