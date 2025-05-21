# agents/ci_health_check_tool.py

# from langchain_core.tools import tool
from langchain.tools import tool

from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI once
vertexai_init(project="devops-swarm-ai-project", location="us-central1")

@tool
def ci_health_check(pipeline_logs: str, mr_id: str) -> dict:
    """
    Analyzes CI pipeline logs and provides feedback on failures and fixes.

    Args:
        pipeline_logs: The raw CI logs from the pipeline.
        mr_id: The merge request ID associated with these logs.

    Returns:
        A string containing the CI health report.
    """
    try:
        model = GenerativeModel("gemini-2.5-pro-preview-05-06")  
        prompt = (
            f"Given the following CI pipeline logs for MR ID {mr_id}, "
            "identify failed jobs, error messages, and suggest likely fixes.\n\n"
            f"Logs:\n{pipeline_logs}"
        )
        response = model.generate_content(prompt)
        feedback = response.text.strip()
        return {
            "ci_feedback": feedback,
            "messages": [{"type": "ai", "content": feedback}]
        }
    except Exception as e:
        error_msg = f"❌ CI health check failed: {e}"
        print("❌ Error in ci_health_check tool:", e)
        return {
            "ci_feedback": error_msg,
            "messages": [{"type": "ai", "content": error_msg}]
        }
