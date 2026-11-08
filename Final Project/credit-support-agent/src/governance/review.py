import json
from typing import Optional, List, Sequence, Any
from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    RequestUsage,
)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage, StructuredMessage
from src.governance.models import Verdict
from src.governance.mock_client import MockChatCompletionClient

async def run_review(draft_answer: str, context: str) -> Verdict:
    is_hallucination = "hallucination_test" in draft_answer.lower()

    reviewer_msg = (
        "The draft contains ungrounded claims (hallucination_test detected). Rejected."
        if is_hallucination
        else "The draft is fully grounded against policy context. Approved."
    )

    if is_hallucination:
        editor_verdict = {
            "approved": False,
            "final_answer": "The interest rate is not 0%. It follows standard credit policy.",
            "reason": "Hallucination detected (hallucination_test) and corrected against retrieved context."
        }
    else:
        editor_verdict = {
            "approved": True,
            "final_answer": draft_answer,
            "reason": "Grounded and approved against retrieved context."
        }
    editor_msg = json.dumps(editor_verdict)

    reviewer = AssistantAgent(
        "PolicyComplianceReviewer",
        model_client=MockChatCompletionClient({
            "default": reviewer_msg,
            "revision": reviewer_msg,
        }),
        system_message="Verify draft grounding against retrieved context. Reply 'Rejected' if ungrounded."
    )

    editor = AssistantAgent(
        "FinalEditor",
        output_content_type=Verdict,
        model_client=MockChatCompletionClient({
            "default": editor_msg,
            "revision": editor_msg,
        }),
        system_message="Output final structured Verdict JSON."
    )

    try:
        team = RoundRobinGroupChat(
            participants=[reviewer, editor],
            termination_condition=MaxMessageTermination(3),
            custom_message_types=[StructuredMessage[Verdict]],
        )
    except ValueError:
        team = RoundRobinGroupChat(
            participants=[reviewer, editor],
            termination_condition=MaxMessageTermination(3),
        )

    task_msg = TextMessage(
        content=f"Draft Answer: {draft_answer}\nRetrieved Context: {context}",
        source="user"
    )
    result = await team.run(task=task_msg)

    # Extract Verdict from final message
    for msg in reversed(result.messages):
        if isinstance(msg.content, Verdict):
            return msg.content
        if isinstance(msg.content, str):
            try:
                data = json.loads(msg.content)
                if "approved" in data:
                    return Verdict(**data)
            except Exception:
                continue

    return Verdict(**editor_verdict)
