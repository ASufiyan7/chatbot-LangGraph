"""
memory.py  –  NEXUS episodic memory layer (ChromaDB-backed).

Agents call:
  • save_memory(session_id, task, code, outcome, score)  →  persists a coding episode
  • recall_memory(query, n)                              →  retrieves relevant past episodes
"""
from __future__ import annotations

import hashlib
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

from config.settings import CHROMA_PERSIST_DIR, MEMORY_COLLECTION, MAX_MEMORY_RESULTS


# ── client singleton ──────────────────────────────────────────────────────────

_client     = None
_collection = None


def _get_collection():
    global _client, _collection

    if _collection is not None:
        return _collection

    if not CHROMA_AVAILABLE:
        return None

    _client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    _collection = _client.get_or_create_collection(
        name=MEMORY_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    return _collection


# ── public API ────────────────────────────────────────────────────────────────

def save_memory(
    session_id: str,
    task:       str,
    code:       str,
    outcome:    str,
    score:      float = 0.0,
) -> None:
    """
    Persist a completed coding episode so future sessions can learn from it.

    Args:
        session_id: Unique thread/session identifier.
        task:       The original user request.
        code:       Final code produced.
        outcome:    Execution result or review summary.
        score:      Critic overall score (0-1).
    """
    col = _get_collection()
    if col is None:
        return

    # deterministic ID so re-running same task just updates the record
    doc_id = hashlib.sha256(
        f"{session_id}:{task}".encode()
    ).hexdigest()[:16]

    document = (
        f"TASK: {task}\n\n"
        f"CODE:\n{code}\n\n"
        f"OUTCOME: {outcome}"
    )

    metadata = {
        "session_id": session_id,
        "timestamp":  datetime.utcnow().isoformat(),
        "score":      score,
        "language":   _detect_language(code),
    }

    col.upsert(
        ids=[doc_id],
        documents=[document],
        metadatas=[metadata],
    )


def recall_memory(query: str, n: int = MAX_MEMORY_RESULTS) -> str:
    """
    Retrieve the most relevant past coding episodes for a given query.

    Returns:
        Formatted string ready to inject into a system prompt,
        or empty string if nothing relevant is found.
    """
    col = _get_collection()
    if col is None:
        return ""

    try:
        count = col.count()
        if count == 0:
            return ""

        results = col.query(
            query_texts=[query],
            n_results=min(n, count),
        )
    except Exception:
        return ""

    docs      = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not docs:
        return ""

    snippets = []
    for doc, meta in zip(docs, metadatas):
        ts    = meta.get("timestamp", "")[:10]   # just the date
        score = meta.get("score", 0)
        snippets.append(
            f"[Past Episode | {ts} | Score: {score:.2f}]\n{doc}"
        )

    return "\n\n---\n\n".join(snippets)


# ── internal helper ───────────────────────────────────────────────────────────

def _detect_language(code: str) -> str:
    code_lower = code.lower()
    if "def " in code_lower or "import " in code_lower:
        return "python"
    if "function " in code_lower or "const " in code_lower:
        return "javascript"
    return "unknown"