"""Claude/Anthropic LLM provider implementation."""

import json
import logging
from typing import AsyncIterator, Optional

from anthropic import AsyncAnthropic, APIConnectionError, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from doc_agent.llm.protocol import LLMMessage, LLMResponse, LLMUsage

logger = logging.getLogger("doc_agent.llm.claude")


class ClaudeProvider:
    """Claude/Anthropic implementation of the LLM provider protocol."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        api_key: Optional[str] = None,
    ):
        """Initialize the Claude provider.

        Args:
            model: Claude model ID to use.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay between retries (exponential backoff).
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
        """
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model

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
            temperature: Sampling temperature.

        Returns:
            LLMResponse with the completion.
        """
        return await self._complete_with_retry(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        retry=retry_if_exception_type((APIConnectionError, RateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _complete_with_retry(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Complete with retry logic for transient errors."""
        # Convert messages to Anthropic format
        anthropic_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role != "system"  # System messages handled separately
        ]

        # Build request kwargs
        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }

        if system:
            kwargs["system"] = system

        # Make API call
        response = await self._client.messages.create(**kwargs)

        # Extract text content
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return LLMResponse(
            content=content,
            model=response.model,
            usage=LLMUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            ),
            stop_reason=response.stop_reason,
        )

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

        Raises:
            ValidationError: If response cannot be parsed into the model.
        """
        # Build schema instruction
        schema = response_model.model_json_schema()
        schema_instruction = f"""
You must respond with valid JSON that matches this schema:

```json
{json.dumps(schema, indent=2)}
```

Respond ONLY with the JSON object. Do not include any other text, markdown formatting, or code blocks.
"""

        # Combine with existing system prompt
        full_system = schema_instruction
        if system:
            full_system = f"{system}\n\n{schema_instruction}"

        # Get completion
        response = await self.complete(
            messages=messages,
            system=full_system,
            max_tokens=max_tokens,
            temperature=0.0,  # Always use 0 for structured output
        )

        # Parse JSON from response
        content = response.content.strip()

        # Handle potential markdown code block wrapping
        if content.startswith("```"):
            # Extract content between code blocks
            lines = content.split("\n")
            # Skip first line (```json or ```) and last line (```)
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(content)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {content[:500]}")
            raise ValidationError.from_exception_data(
                title=response_model.__name__,
                line_errors=[
                    {
                        "type": "json_invalid",
                        "loc": (),
                        "msg": f"Invalid JSON: {e}",
                        "input": content,
                    }
                ],
            )

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
        # Convert messages to Anthropic format
        anthropic_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role != "system"
        ]

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }

        if system:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        This is an approximation - Claude uses a custom tokenizer.
        Rough estimate: ~4 characters per token for English text.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        # Simple approximation
        return len(text) // 4
