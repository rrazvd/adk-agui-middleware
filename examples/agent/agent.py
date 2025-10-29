from __future__ import annotations

from fastapi import FastAPI

from google.adk.agents import Agent
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.sessions import DatabaseSessionService
from ag_ui.core import RunAgentInput

from adk_agui_middleware import SSEService, register_agui_endpoint
from adk_agui_middleware.data_model.config import PathConfig
from adk_agui_middleware.data_model.context import ConfigContext
from adk_agui_middleware.data_model.config import RunnerConfig

def get_items() -> list:
    """Returns a list of available items."""
    mocked_items = ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5']
    return mocked_items

root_agent = Agent(
    name="GenericAgent",
    model="gemini-2.5-flash",
    instruction="""
        You are a helpful assistant that provides information about available items.
        
        Always check the possibility of using rendering tools to display data visually to the user.
        You must ensure that the user has the most visual experience possible, using the available rendering tools.

        ** AVAILABLE ITEMS **
            - get_items: Get a list of available items.
                Use case example: "What items are available?";

                ALWAYS show data using the render_Items rendering tool.
                When the render_Items tool returns "success" it means the rendering was successful, 
                so do not present the data in plain text format, just indicate: "Here are the available items for you to choose from.".
    """,
    tools=[get_items]
)

## USER CONTEXT:
#- User name: "{user_name}"

DATABASE_SERVICE_URI = "sqlite:///./sessions.db"

# Define custom extractor that uses state
async def user_id_extractor(input: RunAgentInput, _) -> str:
    if hasattr(input.state, 'get') and input.state.get("user_id"):
        return input.state["user_id"]
    return "anonymous"

# Minimal in-memory configuration suitable for local development
# This uses default in-memory services for history, state, and session management

# Configure how the middleware extracts request context from incoming requests
config_context = ConfigContext(
    app_name="agent",  # Application identifier for logging and state management
    user_id=user_id_extractor,  # Function to extract user ID from state
    # session_id defaults to a safe generator; you can also supply your own.
)

runner_config = RunnerConfig(session_service=DatabaseSessionService(DATABASE_SERVICE_URI))

# Build the SSE service that will run the agent and stream events to the client
# This service orchestrates the entire request-response pipeline
sse_service = SSEService(
    agent=root_agent,  # The LLM agent that processes user inputs
    config_context=config_context,  # Request context extraction configuration
    runner_config=runner_config
)

# Create the FastAPI app using ADK's helper to get all original SDK routes
app: FastAPI = get_fast_api_app(
    agents_dir="../",
    session_service_uri=DATABASE_SERVICE_URI,
    web=True,
)

# Register the main endpoint that accepts POST requests and streams SSE responses
register_agui_endpoint(
    app=app,
    sse_service=sse_service,
    path_config=PathConfig(
        agui_main_path="/ag-ui"
    ),  # Endpoint will be available at POST /ag-ui
)

if __name__ == "__main__":
    import os
    import uvicorn

    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY environment variable not set!")
        print("   Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("   Get a key from: https://makersuite.google.com/app/apikey")
        print()

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
