"""Streaming utility functions for LLM interaction."""

import json
import time
from collections import defaultdict
from typing import Any, Callable

from langchain_core.messages import AIMessage, AnyMessage

from .logging_config import get_logger

logger = get_logger(__name__)


def _invoke_llm_with_optional_stream(
    *,
    llm: Any,
    messages: list[AnyMessage],
    token_callback: Callable[[Any], None] | None,
    debug_callback: Callable[[str], None] | None = None,
) -> AIMessage:
    if token_callback is None:
        return llm.invoke(messages)

    direct_stream_result = _invoke_openai_compatible_direct_stream(
        llm=llm,
        messages=messages,
        token_callback=token_callback,
        debug_callback=debug_callback,
    )
    if direct_stream_result is not None:
        return direct_stream_result

    if debug_callback is not None:
        debug_callback("completion_start")
    merged: AIMessage | None = None
    chunk_count = 0
    chunk_text_count = 0
    merged_emitted_text = ""
    merged_emitted_reasoning = ""
    reasoning_started = False
    answer_started = False
    emitted_any = False
    for chunk in llm.stream(messages):
        chunk_count += 1
        merged = chunk if merged is None else (merged + chunk)
        reasoning_text = _extract_stream_chunk_reasoning(chunk)
        if not reasoning_text:
            merged_reasoning = _to_stream_reasoning_content(getattr(merged, "content", ""))
            if not merged_reasoning:
                merged_reasoning = _to_stream_reasoning_content(getattr(merged, "additional_kwargs", None))
            if merged_reasoning and merged_reasoning.startswith(merged_emitted_reasoning):
                reasoning_text = merged_reasoning[len(merged_emitted_reasoning):]
                merged_emitted_reasoning = merged_reasoning
        if reasoning_text:
            if not reasoning_started:
                reasoning_started = True
            token_callback({"kind": "reasoning", "text": reasoning_text})
            if debug_callback is not None and not emitted_any:
                debug_callback("first_token_emitted")
            emitted_any = True
        text = _extract_stream_chunk_text(chunk)
        text_source = "chunk"
        if not text:
            # Nanobot-style behavior: recover incremental text from merged
            # stream state when provider-specific chunk payload has no direct
            # text field.
            merged_text = _to_stream_text_content(getattr(merged, "content", ""))
            if merged_text and merged_text.startswith(merged_emitted_text):
                text = merged_text[len(merged_emitted_text):]
                merged_emitted_text = merged_text
                text_source = "merged"
        if text:
            if reasoning_started and not answer_started:
                answer_started = True
            logger.debug(
                "Stream emit: chunk=%d text_source=%s text=%r merged_emitted_text=%r",
                chunk_count, text_source, text[:50] if text else "", merged_emitted_text[:50] if merged_emitted_text else ""
            )
            token_callback({"kind": "answer", "text": text})
            chunk_text_count += 1
            if debug_callback is not None and not emitted_any:
                debug_callback("first_token_emitted")
            emitted_any = True
        merged_text = _to_stream_text_content(getattr(merged, "content", ""))
        if merged_text and merged_text.startswith(merged_emitted_text):
            merged_emitted_text = merged_text
        merged_reasoning = _to_stream_reasoning_content(getattr(merged, "content", ""))
        if not merged_reasoning:
            merged_reasoning = _to_stream_reasoning_content(getattr(merged, "additional_kwargs", None))
        if merged_reasoning and merged_reasoning.startswith(merged_emitted_reasoning):
            merged_emitted_reasoning = merged_reasoning

    if merged is None:
        return AIMessage(content="")
    if not emitted_any:
        # Non-streaming provider behavior: emit the final answer right after
        # this completion call returns, instead of waiting for the full turn.
        final_text = _to_stream_text_content(getattr(merged, "content", "")).strip()
        if final_text:
            token_callback({"kind": "answer", "text": final_text})
            if debug_callback is not None:
                debug_callback("fallback_emit_final_text")
    if debug_callback is not None:
        debug_callback(
            f"completion_end chunks={chunk_count} chunks_with_text={chunk_text_count} emitted_any={emitted_any}"
        )
    logger.info(
        "LLM stream stats: chunks=%s, chunks_with_text=%s, emitted_any=%s",
        chunk_count,
        chunk_text_count,
        emitted_any,
    )
    return merged


def _invoke_openai_compatible_direct_stream(
    *,
    llm: Any,
    messages: list[AnyMessage],
    token_callback: Callable[[Any], None],
    debug_callback: Callable[[str], None] | None = None,
) -> AIMessage | None:
    chat_model = getattr(llm, "bound", llm)
    binding_kwargs = getattr(llm, "kwargs", {}) if hasattr(llm, "bound") else {}
    root_client = getattr(chat_model, "root_client", None)
    build_payload = getattr(chat_model, "_get_request_payload", None)
    if root_client is None or not callable(build_payload):
        return None

    setup_started_at = time.perf_counter()
    try:
        payload_started_at = time.perf_counter()
        payload = build_payload(messages, **binding_kwargs)
        payload_ready_at = time.perf_counter()
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        request_started_at = time.perf_counter()
        stream = root_client.chat.completions.create(**payload)
        request_ready_at = time.perf_counter()
    except Exception:
        logger.debug("Direct OpenAI-compatible stream setup failed", exc_info=True)
        return None

    message_count = len(messages)
    tool_count = len(payload.get("tools") or []) if isinstance(payload, dict) else 0
    payload_bytes = 0
    try:
        payload_bytes = len(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        logger.debug("Failed to estimate direct stream payload size", exc_info=True)
    logger.info(
        "Direct stream setup timing: messages=%s tools=%s payload_bytes=%s build_payload_ms=%.1f create_stream_ms=%.1f total_setup_ms=%.1f",
        message_count,
        tool_count,
        payload_bytes,
        (payload_ready_at - payload_started_at) * 1000,
        (request_ready_at - request_started_at) * 1000,
        (request_ready_at - setup_started_at) * 1000,
    )

    if debug_callback is not None:
        debug_callback("completion_start_direct")

    content_parts: list[str] = []
    tool_buffers: dict[int, dict[str, str]] = defaultdict(lambda: {"id": "", "name": "", "arguments": ""})
    emitted_any = False
    chunk_count = 0
    text_chunk_count = 0
    first_chunk_at: float | None = None
    first_text_at: float | None = None
    reasoning_started = False
    answer_started = False
    try:
        for chunk in stream:
            chunk_count += 1
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
                logger.info(
                    "Direct stream first chunk latency: %.1f ms",
                    (first_chunk_at - request_started_at) * 1000,
                )
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            reasoning = _to_stream_reasoning_content(delta)
            if reasoning:
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                    logger.info(
                        "Direct stream first text latency: %.1f ms",
                        (first_text_at - request_started_at) * 1000,
                    )
                if not reasoning_started:
                    reasoning_started = True
                token_callback({"kind": "reasoning", "text": reasoning})
                if debug_callback is not None and not emitted_any:
                    debug_callback("first_token_emitted_direct")
                emitted_any = True
            text = getattr(delta, "content", None)
            if isinstance(text, str) and text:
                if first_text_at is None:
                    first_text_at = time.perf_counter()
                    logger.info(
                        "Direct stream first text latency: %.1f ms",
                        (first_text_at - request_started_at) * 1000,
                    )
                if reasoning_started and not answer_started:
                    answer_started = True
                logger.debug(
                    "Direct stream emit: chunk=%d text=%r content_parts_len=%d",
                    chunk_count, text[:50] if text else "", len(content_parts)
                )
                token_callback({"kind": "answer", "text": text})
                content_parts.append(text)
                text_chunk_count += 1
                if debug_callback is not None and not emitted_any:
                    debug_callback("first_token_emitted_direct")
                emitted_any = True

            for tc in getattr(delta, "tool_calls", None) or []:
                index = getattr(tc, "index", 0) or 0
                entry = tool_buffers[index]
                tc_id = getattr(tc, "id", None)
                if isinstance(tc_id, str) and tc_id:
                    entry["id"] = tc_id
                function = getattr(tc, "function", None)
                if function is None:
                    continue
                fn_name = getattr(function, "name", None)
                if isinstance(fn_name, str) and fn_name:
                    entry["name"] = fn_name
                fn_args = getattr(function, "arguments", None)
                if isinstance(fn_args, str) and fn_args:
                    entry["arguments"] += fn_args
    except Exception:
        logger.debug("Direct OpenAI-compatible stream iteration failed", exc_info=True)
        return None

    tool_calls: list[dict[str, Any]] = []
    for idx in sorted(tool_buffers):
        item = tool_buffers[idx]
        if not item["name"]:
            continue
        args_payload = item["arguments"].strip()
        try:
            args = json.loads(args_payload) if args_payload else {}
        except Exception:
            logger.debug("Failed to parse streamed tool arguments: %s", args_payload, exc_info=True)
            args = {}
        tool_calls.append(
            {
                "id": item["id"] or f"tool_call_{idx}",
                "name": item["name"],
                "args": args if isinstance(args, dict) else {},
                "type": "tool_call",
            }
        )

    if debug_callback is not None:
        debug_callback(
            f"completion_end_direct chunks={chunk_count} chunks_with_text={text_chunk_count} emitted_any={emitted_any}"
        )
    logger.info(
        "Direct OpenAI-compatible stream stats: chunks=%s, chunks_with_text=%s, emitted_any=%s, tool_calls=%s, first_chunk_ms=%s, first_text_ms=%s",
        chunk_count,
        text_chunk_count,
        emitted_any,
        len(tool_calls),
        f"{(first_chunk_at - request_started_at) * 1000:.1f}" if first_chunk_at is not None else "-",
        f"{(first_text_at - request_started_at) * 1000:.1f}" if first_text_at is not None else "-",
    )
    return AIMessage(content="".join(content_parts), tool_calls=tool_calls)


def _extract_stream_chunk_text(chunk: Any) -> str:
    # Preferred path: LangChain message chunks usually implement text().
    try:
        text_method = getattr(chunk, "text", None)
        if callable(text_method):
            value = text_method()
            if isinstance(value, str) and value:
                return value
    except Exception:
        logger.debug("chunk.text() extraction failed", exc_info=True)

    content = getattr(chunk, "content", "")
    text = _to_stream_text_content(content)
    if text:
        return text

    additional = getattr(chunk, "additional_kwargs", None)
    if additional is not None:
        text = _to_stream_text_content(additional)
        if text:
            return text
    return ""


def _extract_stream_chunk_reasoning(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    reasoning = _to_stream_reasoning_content(content)
    if reasoning:
        return reasoning

    additional = getattr(chunk, "additional_kwargs", None)
    if additional is not None:
        reasoning = _to_stream_reasoning_content(additional)
        if reasoning:
            return reasoning
    return ""


def _to_stream_text_content(content: Any) -> str:
    """Extract only text-like stream payload fields, avoiding metadata noise."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        lines: list[str] = []
        for item in content:
            extracted = _to_stream_text_content(item)
            if extracted:
                lines.append(extracted)
        return "\n".join(lines)
    if isinstance(content, dict):
        block_type = str(content.get("type") or "").lower()
        if block_type in {"text", "text_delta", "output_text"}:
            block_text = content.get("text")
            if isinstance(block_text, str):
                return block_text
        text = content.get("text")
        if isinstance(text, str):
            return text
        delta = content.get("delta")
        if isinstance(delta, str):
            return delta
        if isinstance(delta, (dict, list)):
            delta_text = _to_stream_text_content(delta)
            if delta_text:
                return delta_text
        output_text = content.get("output_text")
        if isinstance(output_text, str):
            return output_text
        choices = content.get("choices")
        if isinstance(choices, list):
            lines: list[str] = []
            for choice in choices:
                extracted = _to_stream_text_content(choice)
                if extracted:
                    lines.append(extracted)
            if lines:
                return "\n".join(lines)
        message = content.get("message")
        if isinstance(message, (dict, list, str)):
            message_text = _to_stream_text_content(message)
            if message_text:
                return message_text
        nested = content.get("content")
        if nested is not None and nested is not content:
            nested_text = _to_stream_text_content(nested)
            if nested_text:
                return nested_text
        return ""
    return ""


def _to_stream_reasoning_content(content: Any) -> str:
    if isinstance(content, str):
        return ""
    if isinstance(content, list):
        lines: list[str] = []
        for item in content:
            extracted = _to_stream_reasoning_content(item)
            if extracted:
                lines.append(extracted)
        return "\n".join(lines)
    if isinstance(content, dict):
        reasoning = content.get("reasoning_content")
        if isinstance(reasoning, str):
            return reasoning
        delta = content.get("delta")
        if isinstance(delta, (dict, list)):
            delta_reasoning = _to_stream_reasoning_content(delta)
            if delta_reasoning:
                return delta_reasoning
        nested = content.get("content")
        if nested is not None and nested is not content:
            nested_reasoning = _to_stream_reasoning_content(nested)
            if nested_reasoning:
                return nested_reasoning
        choices = content.get("choices")
        if isinstance(choices, list):
            lines: list[str] = []
            for choice in choices:
                extracted = _to_stream_reasoning_content(choice)
                if extracted:
                    lines.append(extracted)
            if lines:
                return "\n".join(lines)
        return ""
    reasoning = getattr(content, "reasoning_content", None)
    if isinstance(reasoning, str):
        return reasoning
    delta = getattr(content, "delta", None)
    if isinstance(delta, (dict, list)):
        delta_reasoning = _to_stream_reasoning_content(delta)
        if delta_reasoning:
            return delta_reasoning
    nested = getattr(content, "content", None)
    if nested is not None and nested is not content:
        nested_reasoning = _to_stream_reasoning_content(nested)
        if nested_reasoning:
            return nested_reasoning
    return ""
