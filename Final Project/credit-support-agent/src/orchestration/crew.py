"""
CrewAI crew for multi-agent credit support orchestration.

Defines three agents:
1. Retrieval Agent - Uses RAG tool for banking policy queries
2. Lookup Agent - Uses loan lookup tool for application status
3. Response Composer - Synthesizes outputs into coherent answer
"""

import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import sys
sys.path.insert(0, '.')

from crewai import Agent, Crew, Task
from src.llm.mock_llm_crewai import MockLLM
from src.tools.crewai_tools import retrieve_banking_policy, check_application_status
from src.orchestration.structured_output import AgentResponse, format_crew_response_as_structured


# ============================================================================
# Initialize MockLLM
# ============================================================================

mock_llm = MockLLM(model_name="mock-llm-crewai")


# ============================================================================
# Define Agents
# ============================================================================

retrieval_agent = Agent(
    role="Banking Policy Retrieval Specialist",
    goal="Search and retrieve relevant banking policies from the knowledge base to answer policy questions",
    backstory=(
        "You are an expert at finding relevant banking and credit policy information. "
        "You use the retrieve_banking_policy tool to search the knowledge base for policies "
        "related to loan eligibility, EMI calculations, prepayment penalties, KYC requirements, "
        "credit scores, and other banking topics."
    ),
    tools=[retrieve_banking_policy],
    llm=mock_llm,
    verbose=True,
)

lookup_agent = Agent(
    role="Loan Application Lookup Specialist",
    goal="Look up loan application details and provide status, amount, and escalation information",
    backstory=(
        "You are an expert at retrieving loan application information. "
        "You use the check_application_status tool to look up application records by ID, "
        "providing status, loan amount, fraud flags, and escalation scores."
    ),
    tools=[check_application_status],
    llm=mock_llm,
    verbose=True,
)

composer_agent = Agent(
    role="Response Composer",
    goal="Synthesize information from policy retrieval and application lookup into coherent, user-friendly answers",
    backstory=(
        "You are an expert at combining information from multiple sources. "
        "You take results from the Retrieval Specialist and Lookup Specialist, "
        "synthesize them, and provide clear, actionable answers to user queries."
    ),
    tools=[],  # Composer doesn't use tools directly - it works with other agents' outputs
    llm=mock_llm,
    verbose=True,
)


# ============================================================================
# Define Tasks
# ============================================================================

def create_crew(query: str):
    """
    Create and run a crew for a given query.

    The crew dynamically selects which agents to use based on query type:
    - Pure policy queries: Retrieval Agent
    - Pure lookup queries: Lookup Agent
    - Mixed queries: Both agents + Composer

    Args:
        query: User query

    Returns:
        Crew instance ready to execute
    """

    # Task 1: Retrieval (if query contains policy keywords)
    retrieval_task = Task(
        description=(
            f"Search the banking policy knowledge base for information relevant to this query: '{query}'\n"
            "Use the retrieve_banking_policy tool to find relevant documents. "
            "Provide the top 5 most relevant policies with their content summaries."
        ),
        agent=retrieval_agent,
        expected_output="List of relevant banking policies with document titles and content excerpts",
    )

    # Task 2: Lookup (if query contains application/status keywords)
    lookup_task = Task(
        description=(
            f"Look up loan application information for this query: '{query}'\n"
            "Use the check_application_status tool to retrieve application details. "
            "Provide the status, loan amount, fraud flag, and escalation score."
        ),
        agent=lookup_agent,
        expected_output="Loan application details including status, amount, and escalation score",
    )

    # Task 3: Compose (synthesize all information)
    compose_task = Task(
        description=(
            f"Synthesize all retrieved information to answer the user's query: '{query}'\n"
            "Combine policy information (from Retrieval Specialist) and application details (from Lookup Specialist) "
            "into one coherent, user-friendly answer. Provide actionable recommendations."
        ),
        agent=composer_agent,
        expected_output="A clear, comprehensive answer synthesizing all information sources",
    )

    # Determine which tasks to include based on query content
    tasks = []
    query_lower = query.lower()

    # Check for policy keywords
    has_policy_keywords = any(word in query_lower for word in [
        "penalty", "prepayment", "eligibility", "emi", "cibil", "kyc",
        "requirement", "policy", "interest", "fee", "charge"
    ])

    # Check for lookup keywords
    has_lookup_keywords = any(word in query_lower for word in [
        "status", "application", "loan_", "record", "escalation", "fraud"
    ])

    if has_policy_keywords:
        tasks.append(retrieval_task)

    if has_lookup_keywords:
        tasks.append(lookup_task)

    # Always include composer if we have multiple tasks or need synthesis
    if len(tasks) > 1 or (has_policy_keywords and has_lookup_keywords):
        tasks.append(compose_task)

    # If no tasks were selected, run both
    if not tasks:
        tasks = [retrieval_task, lookup_task, compose_task]

    # Create and return crew
    crew = Crew(
        agents=[retrieval_agent, lookup_agent, composer_agent],
        tasks=tasks,
        verbose=True,
        max_iter=10,  # Max iterations per task
    )

    return crew


# ============================================================================
# Structured Output Wrapper
# ============================================================================

def run_crew(query: str) -> AgentResponse:
    """
    Execute crew on query and return validated AgentResponse.

    This is the primary function for running the crew. It wraps create_crew()
    and crew.kickoff() to ensure all responses are validated against the
    unified AgentResponse Pydantic schema.

    Args:
        query: User query to process

    Returns:
        AgentResponse: Validated structured response with schema enforcement
    """
    import re

    # Create and execute crew
    crew = create_crew(query)
    raw_answer = crew.kickoff()
    crew_answer_str = str(raw_answer)

    # Extract escalation and source information
    answer_lower = crew_answer_str.lower()
    escalation_required = "escalation recommended: yes" in answer_lower
    escalation_reason = None
    sources = []

    # Extract loan ID if present
    loan_match = re.search(r'\b(LOAN_\d{4})\b', crew_answer_str)
    if loan_match:
        sources.append(loan_match.group(1))

    # Extract escalation reason if escalation is recommended
    if escalation_required:
        if "fraud" in answer_lower:
            escalation_reason = "Fraud flagged"
        elif "escalation score" in answer_lower:
            # Extract score
            score_match = re.search(r'escalation score: ([\d.]+)', answer_lower)
            if score_match:
                score = float(score_match.group(1))
                escalation_reason = f"Escalation score ({score:.4f}) exceeds threshold (0.60)"

    # Add tool/intent-specific sources
    if "prepayment" in query.lower() or "penalty" in query.lower():
        sources.append("retrieve_banking_policy")
    if "status" in query.lower() and "loan" in query.lower():
        sources.append("check_application_status")
    if "eligibility" in query.lower() or "criteria" in query.lower():
        sources.append("retrieve_banking_policy")

    # Format as structured output and validate
    response = format_crew_response_as_structured(
        query=query,
        crew_answer=crew_answer_str,
        sources=list(set(sources)),  # Remove duplicates
        escalation_required=escalation_required,
        escalation_reason=escalation_reason,
        confidence_score=1.0,
    )

    # Validate against schema
    return AgentResponse.model_validate(response.model_dump())

def run_demonstration():
    """Run demonstration on all three query types."""

    print("\n" + "=" * 80)
    print("CREWAI MULTI-AGENT ORCHESTRATION DEMONSTRATION")
    print("=" * 80)

    test_queries = [
        ("Query A (RAG Only)", "What are the prepayment penalty rules?"),
        ("Query B (Lookup Only)", "What is the status of application LOAN_0001?"),
        ("Query C (Mixed)", "What is the status of LOAN_0001, and what is the policy for loan eligibility?"),
    ]

    for query_type, query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"{query_type}")
        print(f"Query: {query}")
        print("=" * 80)

        crew = create_crew(query)
        result = crew.kickoff()

        print(f"\nFinal Answer:\n{result}")
        print("\n")


if __name__ == "__main__":
    run_demonstration()
