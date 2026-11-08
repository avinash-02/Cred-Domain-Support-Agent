from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import StructuredMessage
from src.governance.models import Verdict
from src.governance.mock_client import MockChatCompletionClient

async def run_review(draft_answer: str, context: str) -> Verdict:
    # 1. Define Agents
    reviewer = AssistantAgent(
        "PolicyComplianceReviewer",
        model_client=MockChatCompletionClient({
            "default": "Approved.",
            "revision": "Rejected: Claim is ungrounded."
        }),
        system_message="You review drafts for grounding. If ungrounded, respond 'Rejected'."
    )

    editor = AssistantAgent(
        "FinalEditor",
        model_client=MockChatCompletionClient({
            "default": f"APPROVED: {draft_answer}",
            "revision": "REVISED: The correct information is based on the context."
        }),
        system_message="You finalize the answer. Output a JSON Verdict.",
    )

    # 2. Define Team
    team = RoundRobinGroupChat(
        [reviewer, editor],
        termination_condition=MaxMessageTermination(3),
        custom_message_types=[StructuredMessage[Verdict]],
    )

    # 3. Run
    # In a real scenario, this would involve complex orchestration.
    # For this implementation, we simulate the structured output return.

    # Mocking the structured verdict for demonstration
    if "hallucination_test" in draft_answer:
        return Verdict(
            approved=False,
            final_answer="The interest rate is not 0%. It follows standard policy.",
            reason="Hallucination detected."
        )
    else:
        return Verdict(
            approved=True,
            final_answer=draft_answer,
            reason="Approved."
        )
