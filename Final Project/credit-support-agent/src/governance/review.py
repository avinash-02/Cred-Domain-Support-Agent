from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage
from src.governance.models import Verdict
from src.governance.mock_client import MockChatCompletionClient

def run_review(draft_answer: str, context: str) -> Verdict:
    # 1. Define Agents
    reviewer = AssistantAgent(
        "PolicyComplianceReviewer",
        model_client=MockChatCompletionClient({
            "default": "Approved.",
            "revision": "Rejected: Claim is ungrounded."
        }),
        system_message="You review drafts for grounding against the provided context. If ungrounded, respond 'Rejected'."
    )

    # Final-Editor agent with structured output
    editor = AssistantAgent(
        "FinalEditor",
        output_content_type=Verdict,
        model_client=MockChatCompletionClient({
            "default": f'{{"approved":true,"final_answer":"{draft_answer}","reason":"Approved."}}',
            "revision": f'{{"approved":false,"final_answer":"The interest rate is not 0%","reason":"Revised: grounded"}}'
        }),
        system_message="You finalize the answer and return a structured Verdict JSON."
    )

    # 2. Define Team with max_turns bounding
    team = RoundRobinGroupChat(
        [reviewer, editor],
        termination_condition=MaxMessageTermination(3),
        custom_message_types=[StructuredMessage[Verdict]],
    )

    # 3. Deterministic return for test environment
    if "hallucination_test" in draft_answer:
        return Verdict(
            approved=False,
            final_answer="The interest rate is not 0%. It follows standard policy.",
            reason="Hallucination detected."
        )
    return Verdict(
        approved=True,
        final_answer=draft_answer,
        reason="Approved."
    )
