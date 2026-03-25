from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from ..memory import LongTermMemoryStore
from ..logging_config import get_logger

try:
    # Newer E2B SDK
    from e2b_code_interpreter import Sandbox
except ImportError:  # pragma: no cover - compatibility fallback
    # Legacy E2B SDK
    from e2b import Sandbox

logger = get_logger(__name__)


def build_tools(memory_store: LongTermMemoryStore, user_id: str, e2b_api_key: str) -> list[Any]:
    logger.info("Building tools for user_id=%s", user_id)

    @tool
    def execute_python(code: str) -> str:
        """在 E2B 沙箱执行 Python 代码。输入必须是完整可执行 Python 代码字符串。"""

        logger.info("Executing Python code in E2B sandbox, code_len=%s", len(code))
        sandbox = Sandbox(api_key=e2b_api_key)
        execution = sandbox.run_code(code)

        outputs: list[str] = []
        for item in execution.logs.stdout:
            outputs.append(item)
        for item in execution.logs.stderr:
            outputs.append(f"[stderr] {item}")

        if execution.error:
            outputs.append(f"[error] {execution.error.name}: {execution.error.value}")
            logger.warning("E2B execution returned error: %s", execution.error.name)

        logger.info("E2B execution finished, output_lines=%s", len(outputs))
        return "\n".join(outputs) if outputs else "执行完成，无输出。"

    @tool
    def save_memory(text: str, memory_type: str = "experience") -> str:
        """将重要信息写入长期记忆库。text 为记忆文本，memory_type 常用值是 experience 或 preference。"""

        logger.info("Tool save_memory called: type=%s, text_len=%s", memory_type, len(text))
        doc_id = memory_store.save(user_id=user_id, text=text, memory_type=memory_type)
        if not doc_id:
            logger.warning("Tool save_memory skipped: memory backend unavailable")
            return "长期记忆服务当前不可用，已跳过写入。"
        logger.info("Tool save_memory success: id=%s", doc_id)
        return f"已写入长期记忆，id={doc_id}"

    return [execute_python, save_memory]
