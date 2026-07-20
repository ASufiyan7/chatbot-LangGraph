"""
settings.py — All environment variables in one place.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL    = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "deepseek-coder:6.7b")

# ── Coder provider ────────────────────────────────────────────────────────────
# "groq"   → fast cloud inference, recommended for CPU-only machines (default)
# "ollama" → local inference, only use if you have a dedicated GPU
CODER_PROVIDER = os.getenv("CODER_PROVIDER", "groq")

# ── Execution Sandbox ─────────────────────────────────────────────────────────
SANDBOX_TIMEOUT      = int(os.getenv("SANDBOX_TIMEOUT", "15"))
USE_DOCKER_SANDBOX   = os.getenv("USE_DOCKER_SANDBOX", "false").lower() == "true"
SANDBOX_MEMORY_LIMIT = os.getenv("SANDBOX_MEMORY_LIMIT", "128m")

# ── Memory (ChromaDB) ─────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
MEMORY_COLLECTION  = "nexus_memory"
MAX_MEMORY_RESULTS = 3

# ── Quality Gate ──────────────────────────────────────────────────────────────
QUALITY_PASS_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.70"))
MAX_REVISION_LOOPS     = int(os.getenv("MAX_REVISION_LOOPS", "2"))

# ── Coder self-fix ────────────────────────────────────────────────────────────
MAX_EXEC_RETRIES = int(os.getenv("MAX_EXEC_RETRIES", "2"))

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))