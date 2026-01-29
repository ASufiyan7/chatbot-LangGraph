import os
from typing import List, TypedDict, Annotated

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_huggingface import HuggingFaceEndpoint
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

app = FastAPI(title="LangGraph HuggingFace Chatbot API")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512
)

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    response = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=response)]}

graph = StateGraph(ChatState)

graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", "chat")

app_graph = graph.compile()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

memory = InMemorySaver()
app_graph = graph.compile(checkpointer=memory)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):

    config = {"configurable": {"thread_id": "1"}}

    input_data = {"messages": [HumanMessage(content=req.message)]}

    output = app_graph.invoke(input_data, config=config)

    ai_message = output["messages"][-1]

    return {"response": ai_message.content}

@app.get("/")
def root():
    return {"status": "LangGraph + HuggingFace Chatbot API Running"}
