# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code="union-attr"
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_google_vertexai import ChatVertexAI

# Import all our custom tools/agents
from tools.pr_analyzer_tool import analyze_pr
from tools.quality_check_tool import quality_check
from tools.ci_health_check_tool import ci_health_check
from tools.report_tool import generate_ci_report

LOCATION = "us-central1"
LLM = "gemini-2.5-pro-preview-05-06"

# Tool list
tools = [
    analyze_pr,
    quality_check,
    ci_health_check,
    generate_ci_report,
]

# 2. Set up the language model
llm = ChatVertexAI(
    model=LLM,
    location=LOCATION,
    temperature=0,
    max_tokens=4096,
    streaming=True
).bind_tools(tools)

# 3. Define workflow components
def should_continue(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    return "dev_tools" if last_message.tool_calls else END

def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    system_message = (
        " You are a DevOps AI agent that coordinates MR review. "
        " Use tools to summarize PRs, review code, analyze CI logs, and create reports."
    )
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
    response = llm.invoke(messages_with_system, config)
    return {"messages": response if isinstance(response, list) else [response]}

# 4. Create the workflow graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("dev_tools", ToolNode(tools))
workflow.set_entry_point("agent")

# 5. Define graph edges
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("dev_tools", "agent")

# 6. Compile the workflow
agent = workflow.compile()
