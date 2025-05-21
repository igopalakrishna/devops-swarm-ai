# agents/report_tool.py

# from langchain_core.tools import tool
from langchain.tools import tool


@tool
def generate_ci_report(summary: str, quality_issues: str, ci_feedback: str) -> dict:
    """
    Generates a full formatted Merge Request review report using outputs from previous agents.

    Args:
        summary: The summary of the MR.
        quality_issues: Issues found in code review.
        ci_feedback: Feedback from CI pipeline logs.

    Returns:
        A full human-readable report as a string.
    """
    try:
        report = (
            "**Merge Request Review Summary**\n\n"
            f"**PR Summary:**\n{summary}\n\n"
            f"**Code Quality Feedback:**\n{quality_issues}\n\n"
            f"**CI Health Report:**\n{ci_feedback}"
        )
        return {
            "report": report,
            "messages": [{"type": "ai", "content": report}]
        }
    except Exception as e:
        error_msg = f"❌ Report generation failed: {e}"
        return {
            "report": error_msg,
            "messages": [{"type": "ai", "content": error_msg}]
        }


