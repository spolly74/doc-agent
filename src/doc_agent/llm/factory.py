"""Factory function for creating LLM providers."""

from typing import Optional

from doc_agent.config.models import LLMConfig, LLMProviderType
from doc_agent.llm.claude import ClaudeProvider
from doc_agent.llm.ollama import OllamaProvider
from doc_agent.llm.protocol import LLMProvider


def create_llm_provider(
    config: Optional[LLMConfig] = None,
    provider_type: Optional[LLMProviderType] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """Create an LLM provider based on configuration.

    Args:
        config: LLMConfig object with provider settings.
        provider_type: Override provider type (defaults to config or Claude).
        model: Override model name.
        api_key: Override API key (only used for Claude).

    Returns:
        LLMProvider instance.

    Raises:
        ValueError: If provider type is not supported.
    """
    # Use defaults if no config provided
    if config is None:
        config = LLMConfig()

    # Apply overrides
    effective_provider = provider_type or config.provider
    effective_model = model or config.model

    # Create provider based on type
    if effective_provider == LLMProviderType.CLAUDE:
        return ClaudeProvider(
            model=effective_model,
            max_retries=config.retry_attempts,
            retry_delay=config.retry_delay_seconds,
            api_key=api_key,
        )
    elif effective_provider == LLMProviderType.OLLAMA:
        # For Ollama, use a sensible default model if Claude model specified
        if effective_model.startswith("claude"):
            effective_model = "llama3.2"  # Default Ollama model
        return OllamaProvider(
            model=effective_model,
            base_url=config.ollama_base_url,
            timeout=config.ollama_timeout,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {effective_provider}")


def get_available_providers() -> list[str]:
    """Get list of available provider types.

    Returns:
        List of provider type names.
    """
    return [p.value for p in LLMProviderType]
