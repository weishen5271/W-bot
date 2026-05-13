from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import HumanMessage

from w_bot.agents.core.model_routing import ModelRouter


class FakeTool:
    name = "search"


def test_model_router_binds_tools_for_text_route() -> None:
    llm = MagicMock()
    bound = MagicMock()
    llm.bind_tools.return_value = bound
    router = ModelRouter(llm_text=llm, tools_by_name={"search": FakeTool()})

    selection = router.select(history=[HumanMessage(content="hi")], messages=[HumanMessage(content="hi")])

    assert selection.llm is bound
    assert selection.route == "text"
    assert selection.tool_names == ("search",)
    llm.bind_tools.assert_called_once()


def test_model_router_uses_audio_route_when_available(tmp_path) -> None:
    text_llm = MagicMock()
    audio_llm = MagicMock()
    audio_file = tmp_path / "a.wav"
    audio_file.write_bytes(b"audio")
    router = ModelRouter(llm_text=text_llm, llm_audio=audio_llm)

    selection = router.select(
        history=[
            HumanMessage(
                content="听一下",
                additional_kwargs={
                    "media": [
                        {
                            "id": "a1",
                            "path": str(audio_file),
                            "mime": "audio/wav",
                            "kind": "audio",
                            "size_bytes": 12,
                            "sha256": "abc",
                        }
                    ]
                },
            )
        ],
        messages=[HumanMessage(content="听一下")],
    )

    assert selection.llm is audio_llm
    assert selection.route == "audio"


def test_model_router_falls_back_to_text_without_audio_model(tmp_path) -> None:
    text_llm = MagicMock()
    audio_file = tmp_path / "a.wav"
    audio_file.write_bytes(b"audio")
    router = ModelRouter(llm_text=text_llm)

    selection = router.select(
        history=[
            HumanMessage(
                content="听一下",
                additional_kwargs={
                    "media": [
                        {
                            "id": "a1",
                            "path": str(audio_file),
                            "mime": "audio/wav",
                            "kind": "audio",
                            "size_bytes": 12,
                            "sha256": "abc",
                        }
                    ]
                },
            )
        ],
        messages=[HumanMessage(content="听一下")],
    )

    assert selection.llm is text_llm
    assert selection.route == "text"
