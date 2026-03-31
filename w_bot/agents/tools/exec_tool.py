from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

from ..logging_config import get_logger
from .common import safe_completed_text

logger = get_logger(__name__)


def build_exec_tool(*, workspace_root: Path, sandbox_root: Path) -> StructuredTool:
    def _exec(command: str, timeout_sec: int = 30) -> str:
        normalized = (command or "").strip()
        if not normalized:
            return "Command is empty"
        if looks_like_python_source(normalized):
            logger.info("Executing inline Python via exec tool, code_len=%s", len(normalized))
            return run_python_in_local_sandbox(code=normalized, sandbox_root=sandbox_root)

        try:
            completed = subprocess.run(
                normalized,
                shell=True,
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=max(1, timeout_sec),
            )
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as exc:
            return f"Command failed: {type(exc).__name__}: {exc}"

        chunks: list[str] = [f"exit_code={completed.returncode}"]
        stdout = safe_completed_text(completed.stdout)
        stderr = safe_completed_text(completed.stderr)
        if stdout:
            chunks.append(f"stdout:\n{stdout}")
        if stderr:
            chunks.append(f"stderr:\n{stderr}")
        if len(chunks) == 1:
            chunks.append("No output")
        return "\n\n".join(chunks)

    return StructuredTool.from_function(
        func=_exec,
        name="exec",
        description=(
            "Execute a shell command in the local workspace and return its output. "
            "If the input is raw Python source code instead of a shell command, "
            "run it in the local Python sandbox."
        ),
    )


def run_python_in_local_sandbox(*, code: str, sandbox_root: Path, timeout_sec: int = 15) -> str:
    sandbox_root.mkdir(parents=True, exist_ok=True)

    script_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".py",
            prefix="snippet_",
            dir=str(sandbox_root),
            delete=False,
        ) as script_file:
            script_file.write(code)
            script_path = Path(script_file.name)

        completed = subprocess.run(
            [sys.executable, "-I", str(script_path)],
            cwd=str(sandbox_root),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=max(1, timeout_sec),
            env=sandbox_env(sandbox_root),
            preexec_fn=sandbox_preexec_fn(),
        )
    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout_sec}s"
    except Exception as exc:
        logger.exception("Local sandbox execution failed")
        return f"Execution failed: {type(exc).__name__}: {exc}"
    finally:
        if script_path and script_path.exists():
            try:
                script_path.unlink()
            except OSError:
                logger.warning("Failed to remove sandbox script: %s", script_path)

    chunks: list[str] = [f"exit_code={completed.returncode}"]
    stdout = safe_completed_text(completed.stdout)
    stderr = safe_completed_text(completed.stderr)
    if stdout:
        chunks.append(f"stdout:\n{stdout}")
    if stderr:
        chunks.append(f"stderr:\n{stderr}")
    if len(chunks) == 1:
        chunks.append("No output")
    return "\n\n".join(chunks)


def looks_like_python_source(command: str) -> bool:
    snippet = (command or "").lstrip()
    if not snippet:
        return False
    if snippet.startswith(("python ", "python3 ", "uv run ", "bash ", "sh ", "zsh ")):
        return False
    indicators = ("\n", "import ", "from ", "print(", "def ", "class ", "for ", "while ", "if ", " = ")
    return any(token in snippet for token in indicators)


def sandbox_env(sandbox_root: Path) -> dict[str, str]:
    allowed = ["PATH", "LANG", "LC_ALL", "TZ"]
    env = {k: v for k, v in os.environ.items() if k in allowed}
    env["HOME"] = str(sandbox_root)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def sandbox_preexec_fn() -> Any | None:
    try:
        import resource
    except ImportError:
        return None

    def _set_limits() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        except Exception:
            pass

    return _set_limits
