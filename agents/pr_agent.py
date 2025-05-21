# agents/pr_agent.py
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from langchain_google_vertexai import ChatVertexAI

llm = ChatVertexAI(
    model="gemini-2.5-pro-preview-05-06",
    location="us-central1",
    temperature=0,
    max_tokens=4096,
    streaming=True
)

def pr_agent(state: dict, config: RunnableConfig) -> dict[str, list[BaseMessage]]:
    content = """
    You are a PR Analysis Agent. Given a Pull Request title and diff, generate a summary.
    Extract relevant insights that would be helpful for reviewers.
    """
    pr_message = state["messages"][-1]
    messages = [{"type": "system", "content": content}, pr_message]
    response = llm.invoke(messages, config)
    return {"messages": [response]}
