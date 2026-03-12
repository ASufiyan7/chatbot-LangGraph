"""
settings.py  –  All environment variables in one place
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# ── Execution Sandbox ─────────────────────────────────────────────────────────
SANDBOX_TIMEOUT      = int(os.getenv("SANDBOX_TIMEOUT", "15"))
SANDBOX_MEMORY_LIMIT = os.getenv("SANDBOX_MEMORY_LIMIT", "128m")
USE_DOCKER_SANDBOX   = os.getenv("USE_DOCKER_SANDBOX", "false").lower() == "true"

# ── Memory (ChromaDB) ─────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
MEMORY_COLLECTION  = "nexus_memory"
MAX_MEMORY_RESULTS = 3

# ── Critic / Quality Gate ─────────────────────────────────────────────────────
CRITIC_PASS_THRESHOLD = float(os.getenv("CRITIC_THRESHOLD", "0.70"))
MAX_REVISION_LOOPS    = int(os.getenv("MAX_REVISION_LOOPS", "2"))

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))