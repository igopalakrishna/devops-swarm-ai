from langgraph.graph import StateGraph
from langgraph.experimental.parallel import split
from langchain_core.messages import HumanMessage

# Import your agent nodes
from app.agents.pr_agent import pr_agent
from app.agents.ci_agent import ci_agent
from app.agents.quality_agent import quality_agent
from app.agents.report_agent import report_agent

# Define your input/output state structure
state = {"messages": list}

# Create the graph
workflow = StateGraph(state)

# Add agent nodes
workflow.add_node("pr_agent", pr_agent)
workflow.add_node("ci_agent", ci_agent)
workflow.add_node("quality_agent", quality_agent)
workflow.add_node("report_agent", report_agent)

# Parallel split: all three agents run concurrently
workflow.add_conditional_edges("start", split(["pr_agent", "ci_agent", "quality_agent"]))

# All three converge to the report agent
workflow.add_edge("pr_agent", "report_agent")
workflow.add_edge("ci_agent", "report_agent")
workflow.add_edge("quality_agent", "report_agent")

# Final output
workflow.set_entry_point("start")
workflow.set_finish_point("report_agent")

# Export this runnable
graph_agent = workflow.compile()
