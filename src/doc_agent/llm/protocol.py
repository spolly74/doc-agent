"""Protocol definitions for LLM providers."""

from typing import AsyncIterator, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """A message in an LLM conversation."""

    role: str  # "user", "assistant", "system"
    content: str

    @classmethod
    def user(cls, content: str) -> "LLMMessage":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "LLMMessage":
        """Create an assistant message."""
        return cls(role="assistant", content=content)

    @classmethod
    def system(cls, content: str) -> "LLMMessage":
        """Create a system message."""
        return cls(role="system", content=content)


class LLMUsage(BaseModel):
    """Token usage information from an LLM call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str
    model: str
    usage: LLMUsage = Field(default_factory=LLMUsage)
    stop_reason: Optional[str] = None
    raw_response: Optional[dict] = None  # For debugging

    @property
    def was_truncated(self) -> bool:
        """Whether the response was truncated due to max_tokens."""
        return self.stop_reason == "max_tokens"


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    This enables swapping between different LLM backends (Claude, OpenAI, etc.)
    without changing the evaluation logic.
    """

    @property
    def model_name(self) -> str:
        """Return the model identifier being used."""
        ...

    async def complete(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send messages and get a completion.

        Args:
            messages: List of conversation messages.
            system: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature (0.0 = deterministic).

        Returns:
            LLMResponse with the completion.
        """
        ...

    async def complete_structured(
        self,
        messages: list[LLMMessage],
        response_model: type[BaseModel],
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> BaseModel:
        """Get a structured (JSON) response parsed into a Pydantic model.

        Args:
            messages: List of conversation messages.
            response_model: Pydantic model class to parse response into.
            system: Optional system prompt.
            max_tokens: Maximum tokens in response.

        Returns:
            Parsed Pydantic model instance.
        """
        ...

    async def stream(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]:
        """Stream response tokens.

        Args:
            messages: List of conversation messages.
            system: Optional system prompt.
            max_tokens: Maximum tokens in response.
            temperature: Sampling temperature.

        Yields:
            Response tokens as they are generated.
        """
        ...
