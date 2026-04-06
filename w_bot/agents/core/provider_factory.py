from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI


def build_langchain_llm(
    settings: Any,
    *,
    model_name: str,
    streaming: bool = False,
) -> "ChatOpenAI":
    """Build a LangChain ChatOpenAI instance with proper configuration."""
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": settings.llm_api_key,
        "base_url": settings.llm_base_url,
        "temperature": settings.llm_temperature,
        "streaming": streaming,
    }
    if settings.llm_extra_headers:
        kwargs["default_headers"] = settings.llm_extra_headers
    return ChatOpenAI(**kwargs)


# LLMClient and build_llm_client are kept for backwards compatibility
# but use lazy imports to avoid dependency issues
def _get_openai_compat_provider_lite():
    from .openai_compat_provider_lite import OpenAICompatProviderLite
    return OpenAICompatProviderLite


def _get_llm_client_class():
    from dataclasses import dataclass
    OpenAICompatProviderLite = _get_openai_compat_provider_lite()

    @dataclass(frozen=True)
    class LLMClient:
        provider: OpenAICompatProviderLite
        model_name: str
        temperature: float
        reasoning_effort: str | None = None
        max_tokens: int = 4096

    return LLMClient


def build_llm_client(settings: Any, *, model_name: str) -> Any:
    """Build an LLM client with OpenAI-compatible provider."""
    OpenAICompatProviderLite = _get_openai_compat_provider_lite()
    LLMClient = _get_llm_client_class()

    provider = OpenAICompatProviderLite(
        api_key=settings.llm_api_key,
        api_base=settings.llm_base_url,
        default_model=model_name,
        extra_headers=settings.llm_extra_headers or None,
    )
    return LLMClient(
        provider=provider,
        model_name=model_name,
        temperature=float(settings.llm_temperature),
    )
