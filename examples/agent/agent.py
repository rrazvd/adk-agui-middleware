from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from google.adk.agents import Agent
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.sessions import DatabaseSessionService
from ag_ui.core import RunAgentInput

from adk_agui_middleware import SSEService
from adk_agui_middleware.endpoint import register_agui_endpoint
from adk_agui_middleware.data_model.config import RunnerConfig, PathConfig
from adk_agui_middleware.data_model.context import ConfigContext
from adk_agui_middleware.tools.frontend_tool import FrontendToolset


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

                ALWAYS show data using the render_ItemsList rendering tool.
                When the render_ItemsList tool returns "success" it means the rendering was successful,
                so do not present the data in plain text format, just indicate: "Here are the available items for you to choose from.".

        ## USER CONTEXT:
        - User name: "{user_name}"
    """,
    tools=[get_items, FrontendToolset()]
)

DATABASE_SERVICE_URI = "sqlite:///./sessions.db"

# Define custom extractor that uses state
async def user_id_extractor(agui_input: RunAgentInput, _: Request) -> str:
    if hasattr(agui_input.state, 'get') and agui_input.state.get("user_id"):
        return agui_input.state["user_id"]
    return "anonymous"

sse_service = SSEService(
    agent=root_agent,
    config_context=ConfigContext(
        app_name="agent",
        user_id=user_id_extractor,
    ),
    runner_config=RunnerConfig(
        session_service=DatabaseSessionService(DATABASE_SERVICE_URI),
    ),
)

app: FastAPI = get_fast_api_app(
    agents_dir="../",
    session_service_uri=DATABASE_SERVICE_URI,
    web=True,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3005"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_agui_endpoint(
    app,
    sse_service,
    path_config=PathConfig(agui_main_path="/ag-ui"),
)

if __name__ == "__main__":
    import os
    import uvicorn

    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY environment variable not set!")
        print("   Set it with: export GOOGLE_API_KEY='your-key-here'")
        print("   Get a key from: https://makersuite.google.com/app/apikey")
        print()

    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)