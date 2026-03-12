"""
sandbox.py  –  Secure code execution tool for NEXUS agents.

Two modes:
  • USE_DOCKER_SANDBOX=false  →  subprocess with timeout (default, easy setup)
  • USE_DOCKER_SANDBOX=true   →  Docker container with memory cap (production)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import os

from langchain_core.tools import tool
from config.settings import SANDBOX_TIMEOUT, USE_DOCKER_SANDBOX, SANDBOX_MEMORY_LIMIT


# ── helpers ───────────────────────────────────────────────────────────────────

def _strip_markdown_fences(code: str) -> str:
    """Remove ```python ... ``` wrappers if the LLM added them."""
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _run_subprocess(code: str, language: str) -> dict:
    """Run code in a local subprocess with a hard timeout."""
    clean_code = _strip_markdown_fences(code)

    suffix = ".py" if language == "python" else ".js"

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=suffix,
        delete=False,
    ) as f:
        f.write(clean_code)
        tmp_path = f.name

    try:
        interpreter = sys.executable if language == "python" else "node"

        result = subprocess.run(
            [interpreter, tmp_path],
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT,
        )
        return {
            "stdout":    result.stdout.strip(),
            "stderr":    result.stderr.strip(),
            "exit_code": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "stdout":    "",
            "stderr":    f"⏱ Execution timed out after {SANDBOX_TIMEOUT}s",
            "exit_code": -1,
        }
    except FileNotFoundError:
        interpreter = sys.executable if language == "python" else "node"
        return {
            "stdout":    "",
            "stderr":    f"Interpreter '{interpreter}' not found",
            "exit_code": -1,
        }
    finally:
        os.unlink(tmp_path)


def _run_docker(code: str, language: str) -> dict:
    """Run code inside a disposable Docker container."""
    clean_code = _strip_markdown_fences(code)

    if language == "python":
        image    = "python:3.12-slim"
        cmd_flag = "python3 -c"
    else:
        image    = "node:20-slim"
        cmd_flag = "node -e"

    # escape double quotes in code so shell doesn't break
    escaped = clean_code.replace('"', '\\"')

    docker_cmd = [
        "docker", "run", "--rm",
        "--memory", SANDBOX_MEMORY_LIMIT,
        "--network", "none",
        "--cpus", "0.5",
        image,
        "sh", "-c", f'{cmd_flag} "{escaped}"',
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=SANDBOX_TIMEOUT + 10,
        )
        return {
            "stdout":    result.stdout.strip(),
            "stderr":    result.stderr.strip(),
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout":    "",
            "stderr":    f"⏱ Docker execution timed out after {SANDBOX_TIMEOUT}s",
            "exit_code": -1,
        }


# ── LangChain tool ────────────────────────────────────────────────────────────

@tool
def execute_code(code: str, language: str = "python") -> str:
    """
    Execute code in a secure sandbox and return the output.

    Args:
        code:     The source code to run (Python or JavaScript).
        language: 'python' (default) or 'javascript'.

    Returns:
        A formatted string with stdout, stderr, and exit status.
    """
    language = language.lower().strip()

    if language == "js":
        language = "javascript"

    if language not in ("python", "javascript"):
        return "❌ Unsupported language. Use 'python' or 'javascript'."

    runner = _run_docker if USE_DOCKER_SANDBOX else _run_subprocess
    result = runner(code, language)

    lines = []

    if result["stdout"]:
        lines.append(f"📤 STDOUT:\n{result['stdout']}")

    if result["stderr"]:
        lines.append(f"⚠️  STDERR:\n{result['stderr']}")

    status = "✅" if result["exit_code"] == 0 else "❌"
    lines.append(f"{status} Exit code: {result['exit_code']}")

    return "\n\n".join(lines)