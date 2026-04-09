# NEXUS v2 — Autonomous Multi-Agent AI Coding System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F55036?style=for-the-badge)
![Ollama](https://img.shields.io/badge/Ollama-Local-black?style=for-the-badge)

**Give NEXUS a task. It plans, codes, runs, fixes, reviews, and delivers — autonomously.**

[Features](#features) • [Architecture](#architecture) • [Quickstart](#quickstart) • [API](#api-endpoints) • [Docker](#docker-deployment) • [Config](#configuration)

</div>

---

## What Is NEXUS?

NEXUS is an **autonomous multi-agent software engineering system** built on LangGraph. You send it a task — write code, debug a function, review a class, explain an algorithm — and a pipeline of four specialized AI agents handles the entire job without any further input from you.

Unlike a chatbot that just generates text, **NEXUS actually executes the code it writes**. If execution fails, it reads the error and fixes itself. If the code doesn't meet the quality threshold, it revises and retries. The whole process runs on **free-tier AI services** — no paid API required.

```
User: "Write a binary search tree with insert, delete, and inorder traversal in Python"

NEXUS:
  [Orchestrator]  → Classifies task, writes step-by-step plan, queries memory
  [Coder]         → Writes code, runs it in sandbox, self-fixes on error
  [Quality Gate]  → Reviews correctness/security/style, scores 0–1, routes or retries
  [Responder]     → Delivers clean final answer with explanation and usage examples

Result: Working, tested, quality-checked code. Fully autonomous.
```

---

## Features

- **4-Agent LangGraph Pipeline** — Orchestrator → Coder+Executor → Quality Gate → Responder
- **Live Code Execution** — Runs code in a subprocess sandbox; self-corrects on failure (up to 2 retries)
- **Quality Gate** — Scores code across correctness, security, and style; rejects below 0.70
- **ChromaDB Vector Memory** — Remembers past successful solutions; injects relevant context into new tasks
- **Token Bucket Rate Limiter** — Per-provider burst-tolerant rate limiting (no more hardcoded sleeps)
- **WebSocket Streaming** — Watch agent events in real time as the pipeline executes
- **Docker Deployment** — One command to start everything including local Ollama LLM
- **100% Free-Tier** — Groq (cloud, fast) + Ollama (local, free) + ChromaDB (local)

---

## Architecture

### The 4-Agent Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                         NEXUS Pipeline                          │
│                                                                 │
│  User Input                                                     │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────┐    code task    ┌─────────────┐               │
│  │ Orchestrator│ ──────────────► │   Coder +   │               │
│  │  (Groq 70B) │                 │  Executor   │               │
│  │             │  simple query   │ (Ollama 8B) │               │
│  │ • Classify  │ ──────┐         │             │               │
│  │ • Plan      │       │         │ • Write     │               │
│  │ • Memory    │       │         │ • Execute   │               │
│  └─────────────┘       │         │ • Self-fix  │               │
│                         │         └──────┬──────┘               │
│                         │                │ code + result         │
│                         │                ▼                       │
│                         │         ┌─────────────┐               │
│                         │         │ Quality Gate│               │
│                         │         │  (Groq 70B) │               │
│                         │         │             │  score < 0.70 │
│                         │         │ • Review    │ ──────────────┘
│                         │         │ • Score     │  (revision loop,
│                         │         │ • Route     │   max 2x)
│                         │         └──────┬──────┘
│                         │                │ score ≥ 0.70
│                         │                ▼
│                         │         ┌─────────────┐
│                         └────────►│  Responder  │
│                                   │  (Groq 70B) │
│                                   │             │
│                                   │ • Explain   │
│                                   │ • Format    │
│                                   │ • Deliver   │
│                                   └──────┬──────┘
│                                          │
│                                          ▼
│                                     Final Answer
└─────────────────────────────────────────────────────────────────┘
```

### Agent Breakdown

| Agent | LLM | Role |
|---|---|---|
| **Orchestrator** | Groq · llama-3.3-70b | Classifies task, writes plan, queries ChromaDB memory |
| **Coder + Executor** | Ollama · llama3.1:8b (local) | Writes code, runs in sandbox, self-fixes on error |
| **Quality Gate** | Groq · llama-3.3-70b | Reviews + scores code (correctness/security/style); routes or retries |
| **Responder** | Groq · llama-3.3-70b | Synthesizes clean final answer with explanation and usage |

### Task Routing

| Task Type | Example | Pipeline |
|---|---|---|
| `code_gen` | "Write a sorting algorithm" | Orchestrator → Coder → Quality Gate → Responder |
| `debug` | "Fix this broken function" | Orchestrator → Coder → Quality Gate → Responder |
| `review` | "Review my Python class" | Orchestrator → Coder → Quality Gate → Responder |
| `explain` | "Explain what this code does" | Orchestrator → Responder (no execution) |
| `direct` | "What is Big O notation?" | Orchestrator → Responder (no execution) |

### Quality Scoring

The Quality Gate scores every code submission on three dimensions:

```
Overall Score = 0.5 × correctness + 0.3 × security + 0.2 × style

Score ≥ 0.70  →  Pass → Responder (solution saved to ChromaDB memory)
Score < 0.70  →  Fail → Back to Coder with feedback (max 2 revision loops)
```

---

## Project Structure

```
nexus_v2/
├── main.py                  # FastAPI server — HTTP + WebSocket endpoints
├── agents/
│   ├── orchestrator.py      # Agent 1 — classify, plan, memory query
│   ├── coder.py             # Agent 2 — code generation + sandbox execution
│   ├── quality_gate.py      # Agent 3 — review, score, route
│   └── responder.py         # Agent 4 — final answer synthesis
├── graph/
│   ├── nexus_graph.py       # LangGraph definition — nodes, edges, routers
│   └── state.py             # NexusState TypedDict — shared pipeline state
├── config/
│   ├── llm_factory.py       # get_llm(), token bucket rate limiter, backoff
│   └── settings.py          # Environment variable definitions
├── tools/
│   └── sandbox.py           # Code execution — subprocess + Docker modes
├── memory/
│   └── memory.py            # ChromaDB save/recall wrappers
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quickstart

### Option A — Docker (Recommended)

The easiest way. Starts NEXUS + Ollama together.

```bash
# 1. Clone
git clone https://github.com/ASufiyan7/nexus-v2
cd nexus-v2

# 2. Set up environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free at console.groq.com)

# 3. Build and start (first run downloads llama3.1:8b — ~4GB, one time only)
docker compose up --build

# 4. Test it
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Write a binary search in Python", "thread_id": "t1"}'
```

### Option B — Local (Without Docker)

```bash
# 1. Clone and install
git clone https://github.com/ASufiyan7/nexus-v2
cd nexus-v2
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Install and start Ollama
# Download from https://ollama.com, then:
ollama pull llama3.1:8b
ollama serve                    # keep this running in Terminal 1

# 3. Set up environment
cp .env.example .env
# Add your GROQ_API_KEY

# 4. Start NEXUS (Terminal 2)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card)
3. Create an API key
4. Paste it into your `.env` file

---

## API Endpoints

Full interactive docs available at `http://localhost:8000/docs` (Swagger UI).

### `POST /chat`

Run the full agent pipeline on a task.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Write a Python function to detect if a linked list has a cycle",
    "thread_id": "session-1"
  }'
```

**Response:**
```json
{
  "response": "Here is a cycle detection function using Floyd's algorithm...",
  "generated_code": "def has_cycle(head):\n    slow, fast = head, head\n    ...",
  "quality_score": { "overall": 0.87, "correctness": 0.90, "security": 0.85, "style": 0.82 },
  "plan": "1. Use two-pointer approach...",
  "execution_result": "Test passed: cycle detected correctly"
}
```

### `WS /ws/chat/{thread_id}`

WebSocket endpoint for streaming. Watch each agent fire in real time.

```python
import asyncio, websockets, json

async def stream():
    async with websockets.connect("ws://localhost:8000/ws/chat/t1") as ws:
        await ws.send(json.dumps({"message": "Write a quicksort"}))
        async for msg in ws:
            event = json.loads(msg)
            print(f"[{event['type']}] {event.get('node', '')} {event.get('data', '')}")

asyncio.run(stream())
```

**Event types:** `node_start`, `node_end`, `token`, `done`, `error`

### `GET /graph/schema`

Returns the agent graph structure for frontend visualization.

### `GET /health`

```json
{
  "status": "ok",
  "agents": ["orchestrator", "coder", "quality_gate", "responder"],
  "providers": { "planning": "groq", "coding": "ollama", "review": "groq" }
}
```

---

## Docker Deployment

```bash
# Start (background)
docker compose up -d

# Watch logs
docker compose logs -f nexus

# Stop (keeps ChromaDB memory)
docker compose down

# Stop and wipe memory
docker compose down -v

# Rebuild after code changes
docker compose up --build
```

**What persists across restarts:** ChromaDB memory (named Docker volume), downloaded Ollama model weights.

---

## Configuration

All settings via `.env`. No code changes needed.

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | **required** | Free at [console.groq.com](https://console.groq.com) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Local model — `codellama:7b` is a good alternative |
| `QUALITY_THRESHOLD` | `0.70` | Minimum score to pass quality gate (0.0–1.0) |
| `MAX_REVISION_LOOPS` | `2` | Max times coder is sent back for revision |
| `MAX_EXEC_RETRIES` | `2` | Max self-fix attempts on execution failure |
| `SANDBOX_TIMEOUT` | `15` | Seconds before sandbox execution is killed |
| `USE_DOCKER_SANDBOX` | `false` | Enable Docker-isolated execution (Docker-in-Docker) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `GEMINI_API_KEY` | optional | Fallback provider (not required) |

---

## Key Design Decisions

**Why 4 agents instead of 6?**
The original design had 6 agents. The Debugger was merged into Coder (same model, same job). Reviewer + Critic were merged into Quality Gate (one Groq call instead of two). Result: fewer graph nodes, half the API calls per task, identical output quality.

**Why Ollama for code generation?**
Code gen is the most token-heavy step (2,000–4,000 tokens per task). Running it locally means Groq's free tier quota is reserved for planning and quality review — the steps that genuinely need fast cloud inference.

**Why a token bucket instead of `sleep()`?**
A token bucket is self-calibrating: it only waits when capacity is exhausted and allows bursting when it isn't. Separate buckets per provider mean a Groq wait never blocks an Ollama call. The original `time.sleep(2)` was a band-aid; the bucket is a proper solution.

**Why manual JSON parsing on Quality Gate?**
Groq's structured output via Pydantic schemas had intermittent failures. The Quality Gate now prompts for a plain JSON block and parses manually, with a safe fallback score (0.80) if parsing fails — keeping the pipeline alive instead of crashing.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Framework | LangGraph 0.2+ |
| LLM (Cloud) | Groq — llama-3.3-70b-versatile |
| LLM (Local) | Ollama — llama3.1:8b |
| API Server | FastAPI + Uvicorn |
| Real-time | WebSocket streaming |
| Vector Memory | ChromaDB (persistent) |
| Containerization | Docker + Docker Compose |
| Language | Python 3.11+ |

---

## Author

**Sufiyan Ansari** — AI & Data Science, SCET Surat

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/sufiyan-ansari-98056628a/)
[![GitHub](https://img.shields.io/badge/GitHub-ASufiyan7-181717?style=flat&logo=github)](https://github.com/ASufiyan7)

---

<div align="center">
<sub>Built with LangGraph · Groq · Ollama · FastAPI · ChromaDB · Docker</sub>
</div>
