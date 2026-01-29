import os
from typing import List, TypedDict, Annotated

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_huggingface import HuggingFaceEndpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

app = FastAPI(title="LangGraph HuggingFace Chatbot")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512
)

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def thinking_node(state: ChatState) -> ChatState:
    print("AI is thinking...")
    system_instruction = SystemMessage(content="You are a helpful assistant.Be concise and clear in your responses.")
    return {"messages": [system_instruction]}

def generation_node(state: ChatState) -> ChatState:
    print("Generating AI response...")
    response = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=response)]}

graph = StateGraph(ChatState)

graph.add_node("thinker", thinking_node)
graph.add_node("generator", generation_node)

graph.add_edge(START, "thinker")
graph.add_edge("thinker", "generator")
graph.add_edge("generator", END)

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user"

class ChatResponse(BaseModel):
    response: str

memory = InMemorySaver()
app_graph = graph.compile(checkpointer=memory)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    config = {"configurable": {"thread_id": req.thread_id}}

    output = app_graph.invoke(
        {"messages": [HumanMessage(content=req.message)]},
        config=config
    )

    return {'response': output["messages"][-1].content}

@app.get("/")
def root():
    return {"status": "LangGraph + HuggingFace Chatbot API Running"}
