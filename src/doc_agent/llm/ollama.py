"""Ollama LLM provider implementation for local models."""

import json
import logging
from typing import AsyncIterator, Optional

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from doc_agent.llm.protocol import LLMMessage, LLMResponse, LLMUsage

logger = logging.getLogger("doc_agent.llm.ollama")

# Default Ollama API endpoint
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Default timeout: 10 minutes for local inference on large docs
DEFAULT_OLLAMA_TIMEOUT = 600.0


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
        timeout: float = DEFAULT_OLLAMA_TIMEOUT,
    ):
        """Initialize the Ollama provider.

        Args:
            model: Ollama model name (e.g., "llama3.2", "mistral", "qwen2.5").
            base_url: Ollama API base URL.
            timeout: Request timeout in seconds (default 10 minutes).
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
        return await self._complete_with_retry(
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=10, max=60),
        retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _complete_with_retry(
        self,
        messages: list[LLMMessage],
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send messages with retry logic for timeouts."""
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
        # Build a simpler schema instruction with example format
        # Smaller models work better with examples than formal JSON schemas
        schema = response_model.model_json_schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # Build example JSON structure
        example_obj = {}
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            if field_type == "integer":
                example_obj[field_name] = 0
            elif field_type == "number":
                example_obj[field_name] = 0.0
            elif field_type == "boolean":
                example_obj[field_name] = True
            elif field_type == "array":
                example_obj[field_name] = []
            elif field_type == "object":
                example_obj[field_name] = {}
            else:
                example_obj[field_name] = "..."

        schema_instruction = f"""IMPORTANT: You must respond with ONLY a valid JSON object in this exact format:

{json.dumps(example_obj, indent=2)}

Required fields: {", ".join(required) if required else "all fields shown above"}

Do NOT include any explanation, markdown, or code blocks. Output ONLY the JSON object."""

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

        # Try to extract JSON from various formats
        json_content = self._extract_json(content)

        try:
            data = json.loads(json_content)
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

    def _extract_json(self, content: str) -> str:
        """Extract JSON from model response, handling various formats.

        Args:
            content: Raw response content from the model.

        Returns:
            Extracted JSON string.
        """
        import re

        # Try direct JSON first
        content = content.strip()
        if content.startswith("{") and content.endswith("}"):
            return content

        # Handle markdown code blocks (```json ... ```)
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", content, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Find JSON object anywhere in the response
        # Match from first { to last } that creates valid JSON structure
        brace_match = re.search(r"\{.*\}", content, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(0)
            # Validate it's actually JSON by trying to parse
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        # Fallback: return original content
        return content

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
