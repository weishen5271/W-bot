from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_image_input: bool = False
    supports_audio_input: bool = False
    supports_video_input: bool = False
    supports_document_input: bool = False
    max_image_bytes: int = 8 * 1024 * 1024
    max_audio_seconds: int = 15 * 60
    max_video_frames: int = 12


def resolve_provider_capabilities(*, model_name: str) -> ProviderCapabilities:
    """处理resolve/provider/capabilities相关逻辑并返回结果。
    
    Args:
        model_name: 当前使用的模型名称。
    """

    normalized = model_name.lower().strip()

    # Qwen family: only explicit VL/vision/omni variants are treated as image-capable.
    if "qwen" in normalized:
        supports_vision = any(key in normalized for key in ("vl", "vision", "omni"))
        return ProviderCapabilities(
            supports_image_input=supports_vision,
            supports_audio_input=False,
            supports_video_input=False,
            supports_document_input=False,
            max_image_bytes=8 * 1024 * 1024,
            max_audio_seconds=10 * 60,
            max_video_frames=8,
        )

    # MiniMax family: enable vision only for explicit multimodal variants.
    if "minimax" in normalized:
        supports_vision = any(key in normalized for key in ("vision", "vl", "omni", "m1", "v2"))
        return ProviderCapabilities(
            supports_image_input=supports_vision,
            supports_audio_input=False,
            supports_video_input=False,
            supports_document_input=False,
            max_image_bytes=10 * 1024 * 1024,
            max_audio_seconds=10 * 60,
            max_video_frames=8,
        )

    # OpenAI modern multimodal family best-effort baseline.
    if any(key in normalized for key in ("gpt-4o", "gpt-5", "o1", "o3", "o4")):
        return ProviderCapabilities(
            supports_image_input=True,
            supports_audio_input=True,
            supports_video_input=False,
            supports_document_input=False,
            max_image_bytes=20 * 1024 * 1024,
            max_audio_seconds=20 * 60,
            max_video_frames=12,
        )

    # Unknown models default to text-only for safety.
    return ProviderCapabilities(
        supports_image_input=False,
        supports_audio_input=False,
        supports_video_input=False,
        supports_document_input=False,
    )
