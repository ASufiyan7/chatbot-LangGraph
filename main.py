import os
from typing import List, TypedDict, Annotated, Literal

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

app = FastAPI(title="Step 5: Planner-Executor Agent")

# --- Model Setup ---
llm_engine = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1,
    huggingfacehub_api_token=HF_TOKEN,
)
chat_model = ChatHuggingFace(llm=llm_engine)

# --- Tools ---
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Gets the weather for a specific city."""
    return f"The weather in {city} is sunny and 25°C."

tools = [multiply, get_weather]
tool_node = ToolNode(tools)

# --- Updated State for Step 5 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    plan: List[str] # List of steps to follow
    current_step: int # Track where we are in the plan

# --- Nodes ---

def planner_node(state: AgentState):
    user_input = state["messages"][-1].content

    prompt = (
        "You are a strict planner.\n"
        "If the user request is invalid, ambiguous, or impossible, "
        "respond with exactly: INVALID_REQUEST.\n\n"
        "Otherwise, return a numbered step-by-step plan.\n\n"
        f"User request: {user_input}"
    )

    response = chat_model.invoke(prompt)

    if "INVALID_REQUEST" in response.content:
        return {
            "plan": ["Explain why the request is invalid."],
            "current_step": 0,
        }

    plan = [s.strip() for s in response.content.split("\n") if s.strip()]
    return {"plan": plan, "current_step": 0}

def executor_node(state: AgentState):
    idx = state["current_step"]
    plan = state["plan"]

    if idx >= len(plan):
        return {}

    step = plan[idx]

    # 🚫 Guard: block math tools if no numbers
    if "multiply" in step.lower() and not any(ch.isdigit() for ch in step):
        return {
            "messages": [
                SystemMessage(
                    content="This request cannot be completed because it does not contain valid numbers."
                )
            ],
            "current_step": len(plan),
        }

    system_msg = SystemMessage(
        content=f"Execute this step carefully:\n{step}"
    )

    llm_with_tools = chat_model.bind_tools(tools)

    response = llm_with_tools.invoke(
        [system_msg] + state["messages"]
    )

    return {
        "messages": [response],
        "current_step": idx + 1,
    }


def final_answer_node(state: AgentState):
    """Summarizes all executed steps into a final response."""
    print("--- LOG: Final Answer ---")
    response = chat_model.invoke(state["messages"] + [SystemMessage(content="Summarize the results into a final answer for the user.")])
    return {"messages": [response]}

# --- Conditional Logic ---
def should_continue(state: AgentState) -> Literal["execute", "finalize"]:
    # If we have steps left in the plan, keep executing
    if state["current_step"] < len(state["plan"]):
        return "execute"
    return "finalize"

# --- Graph Construction ---
builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.add_node("tools", tool_node)
builder.add_node("final_answer", final_answer_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "executor")

# The Executor loop: checks for tool calls
def executor_router(state: AgentState):
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "check_plan"

builder.add_conditional_edges("executor", executor_router, {"tools": "tools", "check_plan": "planner_update"})

# Logic to decide if we need to re-plan or finish
builder.add_node("planner_update", lambda x: x) # Placeholder for more complex re-planning logic
builder.add_conditional_edges("planner_update", should_continue, {"execute": "executor", "finalize": "final_answer"})

builder.add_edge("tools", "executor")
builder.add_edge("final_answer", END)

app_graph = builder.compile(checkpointer=InMemorySaver())

# FastAPI API Layer
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    config = {"configurable": {"thread_id": req.thread_id}}

    output = app_graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config,
    )

    return {"response": output["messages"][-1].content}

@app.get("/")
def root():
    return {"status": "LangGraph Planner-Executor Chatbot running"}