"""
app.py  –  NEXUS Streamlit Chatbot (Refactored)
A clean, native Streamlit chat UI for the NEXUS FastAPI backend.
"""

import streamlit as st
import requests
import uuid
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
NEXUS_URL = "http://localhost:8000"
# Increased timeout to 300 seconds to give the local Ollama model time to self-correct
TIMEOUT   = 300  

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS — AI Coding Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "server_status" not in st.session_state:
    st.session_state.server_status = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def check_server():
    try:
        r = requests.get(f"{NEXUS_URL}/health", timeout=3)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def send_to_nexus(message: str) -> dict | None:
    """Send a message to NEXUS with a unique thread_id."""
    thread_id = f"ui_{uuid.uuid4().hex[:12]}"
    try:
        r = requests.post(
            f"{NEXUS_URL}/chat",
            json={"message": message, "thread_id": thread_id},
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
        else:
            return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The local Ollama model is taking a while to generate and test code."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to NEXUS backend. Is it running on localhost:8000?"}
    except Exception as e:
        return {"error": str(e)}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 NEXUS")
    st.markdown("**Autonomous Multi-Agent AI**")
    st.markdown("---")

    # Server status check
    if st.button("🔌 Check server", use_container_width=True):
        st.session_state.server_status = check_server()

    status = st.session_state.server_status
    if status is None:
        st.caption("Click to check server")
    elif status:
        st.success(f"Online — v{status.get('version','?')}")
    else:
        st.error("Server not reachable")

    st.markdown("---")
    NEXUS_URL = st.text_input("Backend URL", value=NEXUS_URL)

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.caption("NEXUS v2 · LangGraph + FastAPI\n\nGroq · Ollama · Gemini")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("NEXUS")
st.markdown("##### Autonomous Multi-Agent AI Coding System")
st.divider()

# ── Render Chat History ───────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.info("👋 Welcome to NEXUS! Ask me to write code, debug errors, or explain concepts.")

for entry in st.session_state.chat_history:
    # User Message
    with st.chat_message("user", avatar="👤"):
        st.write(entry["question"])
    
    # Assistant Message
    with st.chat_message("assistant", avatar="🤖"):
        if entry.get("error"):
            st.error(entry["error"])
        elif entry.get("response"):
            resp = entry["response"]
            
            # 1. Show the clean, final synthesis from the Responder agent
            if resp.get("response"):
                st.markdown(resp["response"])
            
            # 2. Tuck the verbose agent pipelines into an expander to keep UI clean
            with st.expander("🛠️ View Agent & Execution Details"):
                
                if resp.get("plan"):
                    st.markdown("**📋 Orchestrator Plan**")
                    st.info(resp["plan"])
                
                if resp.get("generated_code"):
                    lang = resp.get("language", "python")
                    st.markdown(f"**💻 Generated {lang.capitalize()} Code**")
                    st.code(resp["generated_code"], language=lang)
                
                if resp.get("execution_result"):
                    ok = resp.get("execution_ok", False)
                    st.markdown("**📤 Sandbox Execution Output**")
                    if ok:
                        st.success(resp["execution_result"])
                    else:
                        st.error(resp["execution_result"])
                
                score = resp.get("critic_score")
                if score:
                    overall = score.get("overall", 0)
                    if overall >= 0.70:
                        st.success(f"**⚖️ Quality Gate Passed:** {overall:.0%} overall score")
                    else:
                        st.warning(f"**⚖️ Quality Gate Failed:** {overall:.0%} overall score (Required revision)")


# ── Input Area ────────────────────────────────────────────────────────────────
# Using native chat_input fixes the empty label warning and automatically clears upon sending
if prompt := st.chat_input("Ask anything — write code, debug, review..."):
    
    # 1. Immediately append and show user message
    st.session_state.chat_history.append({
        "question": prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "error": None,
        "response": None
    })
    
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)

    # 2. Show thinking spinner while backend does the heavy lifting
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("NEXUS is thinking... (Orchestrator → Coder → Quality Gate → Responder)"):
            result = send_to_nexus(prompt)
            
            # Update the latest history entry with the result
            if result and "error" in result:
                st.session_state.chat_history[-1]["error"] = result["error"]
                st.error(result["error"])
            else:
                st.session_state.chat_history[-1]["response"] = result
                st.rerun() # Refresh the UI to render the formatted assistant response