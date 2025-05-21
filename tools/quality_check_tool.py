# tools/quality_check_tool.py
# from langchain_core.tools import tool

from langchain.tools import tool

from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI
vertexai_init(project="devops-swarm-ai-project", location="us-central1")

@tool
def quality_check(code: str) -> dict:
    """
    Review the given code for logic flaws, performance issues, and dangerous patterns.

    Args:
        code: A string of Python code from the merge request diff.

    Returns:
        A review summary and messages (required for LangGraph).
    """
    try:
        model = GenerativeModel("gemini-2.5-pro-preview-05-06") 
        prompt = (
            "You are a code reviewer.\n"
            "Review the following code for logic flaws, performance issues, and dangerous patterns.\n"
            f"Code:\n{code}"
        )
        response = model.generate_content(prompt)
        result = response.text.strip()
        return {
            "review": result,
            "messages": [{"type": "ai", "content": result}]
        }
    except Exception as e:
        error_msg = f"❌ Review failed: {e}"
        print("❌ Error in quality_check tool:", e)
        return {
            "review": error_msg,
            "messages": [{"type": "ai", "content": error_msg}]
        }
