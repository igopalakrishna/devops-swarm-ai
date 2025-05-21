# multi_agent_graph.py

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph, MessagesState
from langchain_google_vertexai import ChatVertexAI

from agents.pr_agent import pr_agent
from agents.quality_agent import quality_agent
from agents.ci_agent import ci_agent
from agents.report_agent import report_agent

# === Graph Construction ===
workflow = StateGraph(MessagesState)

# Add each agent node
workflow.add_node("pr_agent", pr_agent)
workflow.add_node("quality_agent", quality_agent)
workflow.add_node("ci_agent", ci_agent)
workflow.add_node("report_agent", report_agent)

# Define agent pipeline sequence
workflow.set_entry_point("pr_agent")
workflow.add_edge("pr_agent", "quality_agent")
workflow.add_edge("quality_agent", "ci_agent")
workflow.add_edge("ci_agent", "report_agent")
workflow.add_edge("report_agent", END)

# Compile graph
graph_agent = workflow.compile()


# # multi_agent_graph.py

# from langchain_core.messages import BaseMessage
# from langchain_core.runnables import RunnableConfig
# from langgraph.graph import END, StateGraph, MessagesState
# from langgraph.experimental.parallel import split
# from langchain_google_vertexai import ChatVertexAI

# from agents.pr_agent import pr_agent
# from agents.quality_agent import quality_agent
# from agents.ci_agent import ci_agent
# from agents.report_agent import report_agent

# # === Graph Construction ===
# workflow = StateGraph(MessagesState)

# # Add agent nodes
# workflow.add_node("pr_agent", pr_agent)
# workflow.add_node("quality_agent", quality_agent)
# workflow.add_node("ci_agent", ci_agent)
# workflow.add_node("report_agent", report_agent)

# # Start -> Parallel Branches -> Merge to Report
# workflow.add_conditional_edges("start", split(["pr_agent", "quality_agent", "ci_agent"]))

# # Each agent feeds into report
# workflow.add_edge("pr_agent", "report_agent")
# workflow.add_edge("quality_agent", "report_agent")
# workflow.add_edge("ci_agent", "report_agent")

# # Report ends the graph
# workflow.add_edge("report_agent", END)

# # Set entry and finish
# workflow.set_entry_point("start")
# workflow.set_finish_point("report_agent")

# # Compile graph
# graph_agent = workflow.compile()
