"""Ollama LLM provider implementation for local models."""

import json
import logging
from typing import AsyncIterator, Optional

import httpx
from pydantic import BaseModel, ValidationError

from doc_agent.llm.protocol import LLMMessage, LLMResponse, LLMUsage

logger = logging.getLogger("doc_agent.llm.ollama")

# Default Ollama API endpoint
DEFAULT_OLLAMA_URL = "http://localhost:11434"


class OllamaProvider:
    """Ollama implementation of the LLM provider protocol.

    Use this for local LLM inference with models like:
    - llama3.2 (8B, good for evaluation tasks)
    - mistral (7B, fast and capable)
    - codellama (for code-heavy docs)
    - qwen2.5 (good at structured output)
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = 300.0,  # 5 minutes for large docs
    ):
        """Initialize the Ollama provider.

        Args:
            model: Ollama model name (e.g., "llama3.2", "mistral", "qwen2.5").
            base_url: Ollama API base URL.
            timeout: Request timeout in seconds.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return f"ollama/{self._model}"

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
        # Build Ollama messages format
        ollama_messages = []

        if system:
            ollama_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "system" and not system:
                ollama_messages.append({"role": "system", "content": msg.content})
            elif msg.role != "system":
                ollama_messages.append({"role": msg.role, "content": msg.content})

        # Make API call
        try:
            response = await self._client.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data.get("message", {}).get("content", "")

            # Extract usage if available
            usage = LLMUsage(
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
            )

            return LLMResponse(
                content=content,
                model=self._model,
                usage=usage,
                stop_reason=data.get("done_reason", "stop"),
                raw_response=data,
            )

        except httpx.ConnectError:
            logger.error(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Make sure Ollama is running: `ollama serve`"
            )
            raise
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API error: {e.response.text}")
            raise

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
            temperature=0.0,
        )

        # Parse JSON from response
        content = response.content.strip()

        # Handle potential markdown code block wrapping
        if content.startswith("```"):
            lines = content.split("\n")
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
        # Build Ollama messages format
        ollama_messages = []

        if system:
            ollama_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role == "system" and not system:
                ollama_messages.append({"role": "system", "content": msg.content})
            elif msg.role != "system":
                ollama_messages.append({"role": msg.role, "content": msg.content})

        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": ollama_messages,
                "stream": True,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            },
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if content := data.get("message", {}).get("content", ""):
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def count_tokens(self, text: str) -> int:
        """Estimate token count for text.

        This is a rough approximation. Actual tokenization varies by model.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    async def list_models(self) -> list[str]:
        """List available Ollama models.

        Returns:
            List of model names.
        """
        try:
            response = await self._client.get(f"{self._base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
