# run_graph.py

from multi_agent_graph import graph_agent
from langchain_core.messages import HumanMessage

input_state = {
    "messages": [
        HumanMessage(content="""
        Please:
        - Summarize this PR
        - Analyze this code for quality issues
        - Diagnose these CI logs
        - Generate a final report combining all outputs.

        Title: Fix login bug
        Diff: def login(): pass
        CI Logs: ERROR: test_login failed.
        """)
    ]
}

for step in graph_agent.stream(input_state):
    print(step)
