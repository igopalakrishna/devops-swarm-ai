# tools/pr_analyzer_tool.py
# from langchain_core.tools import tool

from langchain.tools import tool

from vertexai import generative_models, init as vertexai_init

# Initialize Vertex AI once globally
vertexai_init(project="devops-swarm-ai-project", location="us-central1")

@tool
def analyze_pr(title: str, mr_diff: str) -> dict:
    """
    Analyze the purpose of a GitLab Merge Request given its title and diff.

    Args:
        title: The title of the merge request.
        mr_diff: The code diff in the merge request.

    Returns:
        A dict with summary and messages (required by LangGraph).
    """
    try:
        model = generative_models.GenerativeModel("gemini-2.5-pro-preview-05-06")  
        prompt = f"Summarize the purpose of this GitLab Merge Request.\nTitle: {title}\nDiff:\n{mr_diff}"
        response = model.generate_content(prompt)
        summary = response.text.strip()
        return {
            "summary": summary,
            "messages": [{"type": "ai", "content": summary}]
        }
    except Exception as e:
        error_msg = f"❌ Failed to generate PR summary: {e}"
        print("Error in analyze_pr tool:", e)
        return {
            "summary": error_msg,
            "messages": [{"type": "ai", "content": error_msg}]
        }
