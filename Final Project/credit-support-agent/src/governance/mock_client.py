from typing import List, Optional, Sequence, Any
from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    RequestUsage,
)

class MockChatCompletionClient(ChatCompletionClient):
    """Deterministic mock client for offline testing."""
    def __init__(self, responses: dict):
        self.responses = responses
        self.call_count = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(vision=False, function_calling=True)

    def actual_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    def total_usage(self) -> RequestUsage:
        return RequestUsage(prompt_tokens=0, completion_tokens=0)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Optional[List[dict]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs,
    ) -> CreateResult:
        # Simple deterministic response logic
        last_message = messages[-1].content if messages else ""

        # Check for our injected "hallucination" indicator to trigger revision
        if "hallucination_test" in str(last_message):
            response = self.responses.get("revision")
        else:
            response = self.responses.get("default")

        return CreateResult(
            finish_reason="stop",
            content=response,
            usage=None,
            cached=False,
        )

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Optional[List[dict]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        **kwargs,
    ) -> Any:
        raise NotImplementedError("create_stream not implemented in mock")

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Optional[List[dict]] = None) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Optional[List[dict]] = None) -> int:
        return 1000

    def model_info(self) -> dict:
        return {"id": "mock-model"}

    async def close(self) -> None:
        pass
