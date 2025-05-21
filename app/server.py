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

import logging
import os
from collections.abc import Generator

from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse
from google.cloud import logging as google_cloud_logging
from langchain_core.runnables import RunnableConfig
from traceloop.sdk import Instruments, Traceloop
from langchain_core.messages import HumanMessage

from app.agent import agent
from app.utils.tracing import CloudTraceLoggingSpanExporter
from app.utils.typing import Feedback, InputChat, Request, dumps, ensure_valid_config

# Initialize FastAPI app and logging
app = FastAPI(
    title="devops-swarm-ai",
    description="API for interacting with the Agent devops-swarm-ai",
)
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)

# Initialize Telemetry
try:
    Traceloop.init(
        app_name=app.title,
        disable_batch=False,
        exporter=CloudTraceLoggingSpanExporter(),
        instruments={Instruments.LANGCHAIN, Instruments.CREW},
    )
except Exception as e:
    logging.error("Failed to initialize Telemetry: %s", str(e))


def set_tracing_properties(config: RunnableConfig) -> None:
    """Sets tracing association properties for the current request.

    Args:
        config: Optional RunnableConfig containing request metadata
    """
    Traceloop.set_association_properties(
        {
            "log_type": "tracing",
            "run_id": str(config.get("run_id", "None")),
            "user_id": config["metadata"].pop("user_id", "None"),
            "session_id": config["metadata"].pop("session_id", "None"),
            "commit_sha": os.environ.get("COMMIT_SHA", "None"),
        }
    )


def stream_messages(
    input: InputChat,
    config: RunnableConfig | None = None,
) -> Generator[str, None, None]:
    """Stream events in response to an input chat.

    Args:
        input: The input chat messages
        config: Optional configuration for the runnable

    Yields:
        JSON serialized event data
    """
    config = ensure_valid_config(config=config)
    set_tracing_properties(config)
    input_dict = input.model_dump()

    for data in agent.stream(input_dict, config=config, stream_mode="messages"):
        yield dumps(data) + "\n"


# Routes
@app.get("/", response_class=RedirectResponse)
def redirect_root_to_docs() -> RedirectResponse:
    """Redirect the root URL to the API documentation."""
    return RedirectResponse(url="/docs")


@app.post("/stream_messages")
def stream_chat_events(request: Request) -> StreamingResponse:
    """Stream chat events in response to an input request.

    Args:
        request: The chat request containing input and config

    Returns:
        Streaming response of chat events
    """
    return StreamingResponse(
        stream_messages(input=request.input, config=request.config),
        media_type="text/event-stream",
    )


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}

# execute
@app.post("/execute")
def execute_agent(payload: dict):
    print("=== Received Payload ===")
    print(payload)

    raw_inputs = payload.get("inputs", {})

    try:
        # Format the dict into a readable prompt string
        prompt = (
            f"Title: {raw_inputs.get('title')}\n"
            f"MR Diff:\n{raw_inputs.get('mr_diff')}\n\n"
            f"Pipeline Logs:\n{raw_inputs.get('pipeline_logs')}\n"
            f"MR ID: {raw_inputs.get('mr_id')}"
        )

        wrapped_input = {
            "messages": [
                {"type": "human", "content": prompt}
            ]
        }

        result = agent.invoke(wrapped_input, config={"recursion_limit": 25})
        print("=== Agent Output ===")
        print(result)
        return result

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "messages": [{"type": "error", "content": str(e)}]
        }


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
