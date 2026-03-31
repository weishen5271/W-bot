from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .openai_compat_provider_lite import OpenAICompatProviderLite


@dataclass(frozen=True)
class LLMClient:
    provider: OpenAICompatProviderLite
    model_name: str
    temperature: float
    reasoning_effort: str | None = None
    max_tokens: int = 4096


def build_llm_client(settings: Any, *, model_name: str) -> LLMClient:
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
