
from __future__ import annotations
import os
import subprocess
import sys
import tempfile

from langchain_core.tools import tool
from config.settings import SANDBOX_MEMORY_LIMIT, SANDBOX_TIMEOUT, USE_DOCKER_SANDBOX


#  Helpers 

def _strip_markdown_fences(code: str) -> str:
   
    lines = code.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _run_subprocess(code: str, language: str) -> dict:
    clean_code = _strip_markdown_fences(code)
    suffix     = ".py" if language == "python" else ".js"

    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
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
        return {"stdout": "", "stderr": f"⏱ Timed out after {SANDBOX_TIMEOUT}s", "exit_code": -1}
    except FileNotFoundError:
        interp = sys.executable if language == "python" else "node"
        return {"stdout": "", "stderr": f"Interpreter '{interp}' not found", "exit_code": -1}
    finally:
        os.unlink(tmp_path)


def _run_docker(code: str, language: str) -> dict:
    clean_code  = _strip_markdown_fences(code)
    image       = "python:3.12-slim" if language == "python" else "node:20-slim"
    interpreter = "python3"          if language == "python" else "node"

    docker_cmd = [
        "docker", "run", "--rm", "-i",
        "--memory", SANDBOX_MEMORY_LIMIT,
        "--network", "none",
        "--cpus", "0.5",
        image, interpreter,
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            input=clean_code,
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
        return {"stdout": "", "stderr": f"⏱ Docker timed out after {SANDBOX_TIMEOUT}s", "exit_code": -1}


#  LangChain tool 

@tool
def execute_code(code: str, language: str = "python") -> str:
    """
    Execute code in a secure sandbox and return formatted output.

    Args:
        code:     Source code to run (Python or JavaScript).
        language: 'python' (default) or 'javascript'.

    Returns:
        Formatted string with stdout, stderr, and exit status.
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
