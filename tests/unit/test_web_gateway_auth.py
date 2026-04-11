from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from w_bot.agents.core.escalation import EscalationManager
from w_bot.channels.web.gateway import TokenPrincipal, WebAuthConfig, _build_app


def _build_test_client(tmp_path: Path) -> tuple[TestClient, EscalationManager]:
    escalation_manager = EscalationManager(str(tmp_path / "escalations.json"))
    auth_config = WebAuthConfig(
        enabled=True,
        proxy_user_header="X-Forwarded-User",
        proxy_roles_header="X-Forwarded-Roles",
        approver_roles=("admin", "approver"),
        session_binding_file_path=str(tmp_path / "web_sessions.json"),
        bearer_tokens=(
            TokenPrincipal(token="user-token", user_id="alice", roles=("user",)),
            TokenPrincipal(token="admin-token", user_id="bob", roles=("admin", "approver")),
        ),
    )

    class DummyGraph:
        def __init__(self) -> None:
            self.history: dict[str, list[object]] = {}

        def get_state(self, config: dict[str, object]) -> SimpleNamespace:
            session_id = str(config["configurable"]["thread_id"])
            return SimpleNamespace(values={"messages": list(self.history.get(session_id, []))})

        def invoke(self, inputs: dict[str, object], config: dict[str, object]) -> dict[str, object]:
            session_id = str(config["configurable"]["thread_id"])
            history = list(self.history.get(session_id, []))
            history.extend(inputs["messages"])
            history.append(AIMessage(content="secured-reply"))
            self.history[session_id] = history
            return {"messages": history}

    app = _build_app(
        graph=DummyGraph(),
        thread_prefix="web",
        expose_step_logs=False,
        recursion_limit=5,
        escalation_manager=escalation_manager,
        auth_config=auth_config,
    )
    return TestClient(app), escalation_manager


def test_web_api_requires_authentication(tmp_path: Path) -> None:
    client, _ = _build_test_client(tmp_path)

    response = client.post("/api/session/new")

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required"


def test_session_is_bound_to_authenticated_user(tmp_path: Path) -> None:
    client, _ = _build_test_client(tmp_path)
    alice_headers = {"Authorization": "Bearer user-token"}
    bob_headers = {"Authorization": "Bearer admin-token"}

    session_response = client.post("/api/session/new", headers=alice_headers)
    session_id = session_response.json()["session_id"]

    chat_response = client.post(
        "/api/chat",
        headers=alice_headers,
        json={"message": "hello", "session_id": session_id},
    )
    assert chat_response.status_code == 200
    assert chat_response.json()["reply"] == "secured-reply"

    forbidden_response = client.get(
        "/api/history",
        headers=bob_headers,
        params={"session_id": session_id},
    )
    assert forbidden_response.status_code == 403
    assert "does not belong" in forbidden_response.json()["detail"]


def test_proxy_header_authentication_is_supported(tmp_path: Path) -> None:
    client, _ = _build_test_client(tmp_path)

    session_response = client.post("/api/session/new", headers={"X-Forwarded-User": "proxy-user"})
    assert session_response.status_code == 200

    session_id = session_response.json()["session_id"]
    history_response = client.get(
        "/api/history",
        headers={"X-Forwarded-User": "proxy-user", "X-Forwarded-Roles": "user"},
        params={"session_id": session_id},
    )
    assert history_response.status_code == 200
    assert history_response.json()["session_id"] == session_id


def test_escalation_approval_requires_approver_role_and_records_audit(tmp_path: Path) -> None:
    client, escalation_manager = _build_test_client(tmp_path)
    alice_headers = {"Authorization": "Bearer user-token"}
    bob_headers = {"Authorization": "Bearer admin-token"}

    session_id = client.post("/api/session/new", headers=alice_headers).json()["session_id"]
    escalation_request = escalation_manager.create_request(
        session_id=session_id,
        command="git push origin main",
        working_dir=str(tmp_path),
        justification="publish changes",
        prefix_rule=["git", "push"],
        risk_type="network",
    )

    forbidden_response = client.post(
        "/api/escalations/approve",
        headers=alice_headers,
        json={"session_id": session_id, "request_id": escalation_request.id, "reason": "looks good"},
    )
    assert forbidden_response.status_code == 403

    approved_response = client.post(
        "/api/escalations/approve",
        headers=bob_headers,
        json={"session_id": session_id, "request_id": escalation_request.id, "reason": "looks good"},
    )
    assert approved_response.status_code == 200
    approved_payload = approved_response.json()
    assert approved_payload["status"] == "approved"
    assert approved_payload["approved_by"] == "bob"
    assert approved_payload["approval_reason"] == "looks good"


def test_history_endpoint_returns_existing_messages(tmp_path: Path) -> None:
    client, _ = _build_test_client(tmp_path)
    headers = {"Authorization": "Bearer user-token"}
    session_id = client.post("/api/session/new", headers=headers).json()["session_id"]

    client.post(
        "/api/chat",
        headers=headers,
        json={"message": "hello", "session_id": session_id},
    )
    history_response = client.get("/api/history", headers=headers, params={"session_id": session_id})

    assert history_response.status_code == 200
    messages = history_response.json()["messages"]
    assert [item["role"] for item in messages] == ["human", "thought"]
    assert messages[0]["content"] == "hello"
