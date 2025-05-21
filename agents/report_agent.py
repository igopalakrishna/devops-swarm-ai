# agents/report_agent.py

from langchain_google_vertexai import ChatVertexAI
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

LOCATION = "us-central1"
MODEL = "gemini-2.5-pro-preview-05-06"

llm = ChatVertexAI(
    model=MODEL,
    location=LOCATION,
    temperature=0,
    max_tokens=4096,
    streaming=True
)

def report_agent(state: dict, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    messages = state["messages"]
    messages.insert(0, {"type": "system", "content": "You are a DevOps summarizer. Generate a final CI/PR/Quality report from all findings so far."})
    response = llm.invoke(messages, config)
    return {"messages": [response] if not isinstance(response, list) else response}
