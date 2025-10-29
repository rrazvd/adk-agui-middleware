"""Minimal FastAPI app that exposes an SSE endpoint for AGUI, using
google.adk's LLM agent (LlmAgent) as the underlying agent.

This example shows the smallest practical integration of the middleware:
- Creates an `SSEService` with in-memory services.
- Registers a single POST endpoint that streams Server-Sent Events.
- Extracts a simple `user_id` from the `X-User-Id` header (defaults to "guest").
- Instantiates an `LlmAgent` with a model name from environment variables.

Run locally:
    uvicorn app:app --reload

Environment variables (optional):
- `ADK_MODEL_NAME` (default: `gemini-2.0-flash`)

Note:
- Ensure `google-adk` and its model provider dependencies are installed.
- If your ADK version expects different constructor arguments for `LlmAgent`,
  adjust `build_llm_agent()` accordingly (the function is tiny on purpose).
"""

from __future__ import annotations

import os
from typing import Any

from ag_ui.core import RunAgentInput
from fastapi import FastAPI, Request

from google.adk.sessions import DatabaseSessionService
#from google.adk.cli.fast_api import get_fast_api_app

from adk_agui_middleware import SSEService, register_agui_endpoint
from adk_agui_middleware.data_model.config import PathConfig
from adk_agui_middleware.data_model.context import ConfigContext
from adk_agui_middleware.data_model.config import RunnerConfig

def build_llm_agent() -> Any:
    """Create a simple LLM agent from google.adk."""
    from google.adk.agents.llm_agent import LlmAgent  # type: ignore

    # Get model name from environment (gemini-2.0-flash is the current working model)
    model_name = os.getenv("ADK_MODEL_NAME", "gemini-2.0-flash")

    # Require API key for authentication
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is required. Get your API key from: "
            "https://aistudio.google.com/app/apikey"
        )
    
    def get_items() -> list:
        """Returns a list of available items."""
        mocked_items = ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5']
        return mocked_items

    return LlmAgent(name="demo_agent", model=model_name, tools=[get_items], instruction="""
        You are a helpful assistant that provides information about available items.
        
        Always check the possibility of using rendering tools to display data visually to the user.
        You must ensure that the user has the most visual experience possible, using the available rendering tools.

        ** AVAILABLE ITEMS **
            - get_items: Get a list of available items.
                Use case example: "What items are available?";

                ALWAYS show data using the render_Items rendering tool.
                When the render_Items tool returns "success" it means the rendering was successful, 
                so do not present the data in plain text format, just indicate: "Here are the available items for you to choose from.".
    """)

async def extract_user_id(_: RunAgentInput, request: Request) -> str:
    """Get user id from header (falls back to "guest")."""
    return request.headers.get("X-User-Id", "guest")

# Instantiate the LLM agent from google.adk for processing user requests
agent: Any = build_llm_agent()

# Minimal in-memory configuration suitable for local development
# This uses default in-memory services for history, state, and session management

# Configure how the middleware extracts request context from incoming requests
config_context = ConfigContext(
    app_name="demo-app",  # Application identifier for logging and state management
    user_id=extract_user_id,  # Function to extract user ID from request
    # session_id defaults to a safe generator; you can also supply your own.
)

DATABASE_SERVICE_URI = "sqlite:///./sessions.db"
runner_config = RunnerConfig(session_service=DatabaseSessionService(DATABASE_SERVICE_URI))

# Build the SSE service that will run the agent and stream events to the client
# This service orchestrates the entire request-response pipeline
sse_service = SSEService(
    agent=agent,  # The LLM agent that processes user inputs
    config_context=config_context,  # Request context extraction configuration
    runner_config=runner_config
)

""" # Create the FastAPI app using ADK's helper to get all original SDK routes
app: FastAPI = get_fast_api_app(
    agents_dir="../",
    session_service_uri=DATABASE_SERVICE_URI,
    web=True,
) """

# Create the FastAPI app and register the SSE endpoint at /ag-ui
app = FastAPI(title="AGUI Minimal SSE")

# Register the main endpoint that accepts POST requests and streams SSE responses
register_agui_endpoint(
    app=app,
    sse_service=sse_service,
    path_config=PathConfig(
        agui_main_path="/ag-ui"
    ),  # Endpoint will be available at POST /ag-ui
)


if __name__ == "__main__":  # pragma: no cover - manual run helper
    import uvicorn

    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True)
