from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agents.providers.capabilities import ProviderCapabilities

from .models import ArtifactRef, CapabilityDecision, MediaItem, NormalizedUserContent
from .pipeline import read_text_excerpt, to_data_url


@dataclass(frozen=True)
class MultimodalRuntimeConfig:
    enabled: bool
    max_file_bytes: int
    max_total_bytes_per_turn: int
    max_files_per_turn: int
    audio_mode: str
    video_keyframe_interval_sec: int
    video_max_frames: int
    document_max_chars: int


class MultimodalNormalizer:
    def __init__(
        self,
        *,
        cfg: MultimodalRuntimeConfig,
        capabilities: ProviderCapabilities,
    ) -> None:
        """初始化对象并保存运行所需依赖。
        
        Args:
            cfg: 多模态配置对象。
            capabilities: 模型能力集合，用于判断是否支持特性。
        """
        self._cfg = cfg
        self._capabilities = capabilities

    def normalize(
        self,
        *,
        text: str,
        media: list[MediaItem],
        compact_media: bool,
    ) -> NormalizedUserContent:
        """将输入标准化为统一结构。
        
        Args:
            text: 待处理文本。
            media: 媒体项列表。
            compact_media: 是否使用精简媒体表示。
        """
        output = NormalizedUserContent()

        if text.strip():
            output.blocks.append({"type": "text", "text": text.strip()})

        if not self._cfg.enabled or not media:
            return output

        limited_media = media[: self._cfg.max_files_per_turn]
        total_bytes = 0
        for item in limited_media:
            total_bytes += item.size_bytes
            if item.size_bytes > self._cfg.max_file_bytes:
                output.blocks.append(
                    {
                        "type": "text",
                        "text": f"[附件忽略] {item.id}: 文件过大({item.size_bytes} bytes)",
                    }
                )
                output.decisions.append(
                    CapabilityDecision(
                        media_id=item.id,
                        use_native=False,
                        reason="file_too_large",
                    )
                )
                continue

            if total_bytes > self._cfg.max_total_bytes_per_turn:
                output.blocks.append(
                    {
                        "type": "text",
                        "text": f"[附件忽略] {item.id}: 超过单轮总大小限制",
                    }
                )
                output.decisions.append(
                    CapabilityDecision(
                        media_id=item.id,
                        use_native=False,
                        reason="total_size_exceeded",
                    )
                )
                continue

            self._normalize_item(item=item, compact_media=compact_media, out=output)

        return output

    def _normalize_item(
        self,
        *,
        item: MediaItem,
        compact_media: bool,
        out: NormalizedUserContent,
    ) -> None:
        """将输入标准化为统一结构。
        
        Args:
            item: 当前处理的单个元素。
            compact_media: 是否使用精简媒体表示。
            out: 结果输出列表，用于收集归一化后的块。
        """
        kind = item.kind

        if kind == "image":
            if self._capabilities.supports_image_input and item.size_bytes <= self._capabilities.max_image_bytes:
                if compact_media:
                    out.blocks.append(
                        {
                            "type": "text",
                            "text": f"[历史图片] {item.path}",
                        }
                    )
                    out.artifacts.append(ArtifactRef(media_id=item.id, kind="placeholder", value=item.path))
                    out.decisions.append(
                        CapabilityDecision(media_id=item.id, use_native=False, reason="history_compacted")
                    )
                    return

                out.blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": to_data_url(path=item.path, mime=item.mime)},
                    }
                )
                out.decisions.append(
                    CapabilityDecision(media_id=item.id, use_native=True, reason="native_image")
                )
                return

            out.blocks.append(
                {
                    "type": "text",
                    "text": f"[图片附件] {item.path}",
                }
            )
            out.decisions.append(
                CapabilityDecision(media_id=item.id, use_native=False, reason="image_not_supported")
            )
            return

        if kind == "audio":
            if self._cfg.audio_mode in {"auto", "native"} and self._capabilities.supports_audio_input:
                # Many OpenAI-compatible providers still vary on audio block schema; keep
                # a stable textual fallback while preserving artifact reference.
                out.blocks.append(
                    {
                        "type": "text",
                        "text": f"[音频附件待转写] {item.path}",
                    }
                )
                out.artifacts.append(ArtifactRef(media_id=item.id, kind="audio", value=item.path))
                out.decisions.append(
                    CapabilityDecision(media_id=item.id, use_native=False, reason="audio_schema_guard")
                )
                return

            out.blocks.append({"type": "text", "text": f"[音频附件] {item.path}"})
            out.artifacts.append(ArtifactRef(media_id=item.id, kind="audio", value=item.path))
            out.decisions.append(CapabilityDecision(media_id=item.id, use_native=False, reason="audio_fallback"))
            return

        if kind == "video":
            out.blocks.append(
                {
                    "type": "text",
                    "text": f"[视频附件] {item.path} (暂未启用关键帧提取)",
                }
            )
            out.artifacts.append(ArtifactRef(media_id=item.id, kind="video", value=item.path))
            out.decisions.append(CapabilityDecision(media_id=item.id, use_native=False, reason="video_fallback"))
            return

        if kind == "document":
            excerpt = ""
            try:
                if Path(item.path).suffix.lower() in {".txt", ".md", ".json", ".csv", ".log", ".yml", ".yaml"}:
                    excerpt = read_text_excerpt(path=item.path, max_chars=self._cfg.document_max_chars)
            except Exception:
                excerpt = ""

            if excerpt:
                out.blocks.append({"type": "text", "text": f"[文档摘录 {item.path}]\n{excerpt}"})
                out.artifacts.append(ArtifactRef(media_id=item.id, kind="document_excerpt", value=excerpt))
                out.decisions.append(
                    CapabilityDecision(media_id=item.id, use_native=False, reason="document_text_extracted")
                )
                return

            out.blocks.append({"type": "text", "text": f"[文档附件] {item.path}"})
            out.artifacts.append(ArtifactRef(media_id=item.id, kind="document", value=item.path))
            out.decisions.append(CapabilityDecision(media_id=item.id, use_native=False, reason="document_fallback"))
            return

        out.blocks.append({"type": "text", "text": f"[附件] {item.path}"})
        out.decisions.append(CapabilityDecision(media_id=item.id, use_native=False, reason="unsupported_kind"))


def parse_human_payload(
    content: Any,
    *,
    additional_kwargs: dict[str, Any] | None = None,
) -> tuple[str, list[MediaItem], bool]:
    """解析输入并返回结构化结果。
    
    Args:
        content: 消息内容主体。
        additional_kwargs: 附加关键字参数字典。
    """

    from .pipeline import to_media_item

    extra_media = []
    if isinstance(additional_kwargs, dict):
        maybe_media = additional_kwargs.get("media")
        if isinstance(maybe_media, list):
            extra_media = maybe_media

    if isinstance(content, str):
        media_items: list[MediaItem] = []
        for raw in extra_media:
            if isinstance(raw, dict):
                media = to_media_item(raw)
                if media is not None:
                    media_items.append(media)
        return content, media_items, False

    if isinstance(content, list):
        # Already provider-facing blocks.
        return "", [], True

    if not isinstance(content, dict):
        return str(content), [], False

    text = str(content.get("text") or "")
    media_raw = content.get("media")
    media_items: list[MediaItem] = []
    if isinstance(media_raw, list):
        for raw in media_raw:
            if isinstance(raw, dict):
                media = to_media_item(raw)
                if media is not None:
                    media_items.append(media)
    return text, media_items, False
