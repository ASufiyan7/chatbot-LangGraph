import os
from typing import List, TypedDict, Annotated, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, RemoveMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Environment and Configuration
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

app = FastAPI(title="LangGraph HuggingFace Chatbot")

# LLM Setup
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    temperature=0.1,
    huggingfacehub_api_token=HF_TOKEN,
)
chat_model = ChatHuggingFace(llm=llm)

# Tool Definition
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integers."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """Returns a mock weather report for a given city."""
    return f"The weather in {city} is sunny with a high of 25°C."

tools = [multiply, get_weather]
llm_tools = chat_model.bind_tools(tools)

# State Definition
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str

def summarize_conversation(state: ChatState) -> ChatState:
    """Checks history length and creates a summary if needed."""
    messages = state["messages"]

    if len(messages) > 10:
        return {"messages": [SystemMessage(content="Summarizing conversation...")], "summary": "Summary created."}
    
    existing_summary = state.get("summary", "")
    if existing_summary:
        summary_prompt = f"Extend the current summary: {existing_summary}\n\n With these new messages: {messages}\n"

    else:
        summary_prompt = f"Summarize the following conversation: {messages}\n"

    response = chat_model.invoke(summary_prompt)

    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]

    return {
        "summary": response.content,
        "messages": delete_messages
    }

# Graph Definition
def agent_node(state: ChatState):
    """The brain: Decides whether to talk to the user or call a tool."""
    messages = state["messages"]
    summary = state.get("summary", "")

    if summary:
        system_msg = SystemMessage(content=f"Conversation summary: {summary}")
        messages = [system_msg] + messages

    response = llm_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Graph Construction
graph = StateGraph(ChatState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("summarize", summarize_conversation)

graph.add_edge(START, "summarize")
graph.add_edge("summarize", "agent")

graph.add_conditional_edges(
    "agent",
    tools_condition,
)

graph.add_edge("tools", "agent")

memory = InMemorySaver()
app_graph = graph.compile(checkpointer=memory)

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
    return {"status": "LangGraph HuggingFace Chatbot is running."}