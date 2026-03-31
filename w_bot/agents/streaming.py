from __future__ import annotations


class StreamTextAssembler:
    def __init__(self) -> None:
        self._text = ""

    @property
    def text(self) -> str:
        return self._text

    def consume(self, incoming_text: str | None) -> str:
        payload = incoming_text or ""
        if not payload:
            return ""
        if not self._text:
            self._text = payload
            return payload
        if payload.startswith(self._text):
            delta = payload[len(self._text):]
            self._text = payload
            return delta
        if payload in self._text:
            return ""
        overlap_limit = min(len(self._text), len(payload))
        for size in range(overlap_limit, 0, -1):
            if self._text.endswith(payload[:size]):
                delta = payload[size:]
                self._text += delta
                return delta
        self._text += payload
        return payload

