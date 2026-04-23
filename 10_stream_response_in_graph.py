# ============================================================
# Streaming Response in LangGraph
# ============================================================

import asyncio
import operator
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()


# --- 1. STATE & GRAPH ---
class State(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)


def chatbot_node(state: State) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def dummy_node(state: State) -> State:
    return state


builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_node("dummy", dummy_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "dummy")
builder.add_edge("dummy", END)

graph = builder.compile()


# --- 2A. stream_mode="updates" — one event per node ---
print("=== Method 1: stream_mode='updates' ===")
for event in graph.stream(
    {
        "messages": [HumanMessage("List 3 benefits of LangGraph in one line each")],
    },
    stream_mode="updates",
):
    for node_name, output in event.items():
        print(f"  [{node_name}] {output['messages']}")

# --- 2B. stream_mode="values" — full state after each node ---
print("\n=== Method 2: stream_mode='values' ===")
for snapshot in graph.stream(
    {
        "messages": [HumanMessage("Say hello in 3 languages")],
    },
    stream_mode="values",
):
    print(f"  State has {len(snapshot['messages'])} message(s) now")
    print(snapshot["messages"])


# # --- 2C. astream_events — token-by-token (async) ---
async def token_stream():
    print("\n=== Method 3: Token-by-token streaming ===")
    print("🤖 Bot: ", end="", flush=True)
    async for event in graph.astream_events(
        {
            "messages": [HumanMessage("Count from 1 to 5 slowly, one per line")],
        },
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                print(chunk, end="", flush=True)
    print()


asyncio.run(token_stream())
