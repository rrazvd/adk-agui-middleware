# Copyright (C) 2025 Trend Micro Inc. All rights reserved.
"""Concrete implementation of Server-Sent Events service for AGUI middleware."""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, cast

from ag_ui.core import BaseEvent, RunAgentInput
from fastapi import Request
from google.adk import Runner
from google.adk.agents import BaseAgent
from sse_starlette import EventSourceResponse
from starlette.responses import StreamingResponse

from ..base_abc.handler import BaseInOutHandler
from ..base_abc.sse_service import BaseSSEService
from ..data_model.common import InputInfo
from ..data_model.config import RunnerConfig
from ..data_model.context import ConfigContext, HandlerContext
from ..data_model.queue import EventQueue
from ..data_model.session import SessionParameter
from ..event.error_event import AGUIErrorEvent
from ..event.event_translator import EventTranslator
from ..handler.agui_user import AGUIUserHandler
from ..handler.queue import QueueHandler
from ..handler.running import RunningHandler
from ..handler.session import SessionHandler
from ..handler.user_message import UserMessageHandler
from ..manager.session import SessionManager
from ..tools.shutdown import ShutdownHandler
from ..utils.convert.agui_event_to_sse import (
    convert_agui_event_to_sse,
    convert_agui_event_to_str_fake_sse,
)


class SSEService(BaseSSEService):
    """Concrete implementation of SSE service for handling AGUI agent interactions.

    Manages agent execution, session state, and event streaming for AGUI middleware.
    Coordinates between agent runners, session management, and event translation.
    """

    def __init__(
        self,
        agent: BaseAgent,
        config_context: ConfigContext,
        runner_config: RunnerConfig | None = None,
        handler_context: HandlerContext | None = None,
    ):
        """Initialize SSE service with agent and configuration.

        Args:
            :param agent: Base agent implementation for processing requests
            :param runner_config: Configuration for agent runners and services
            :param config_context: Configuration for extracting context from requests
            :param handler_context: Optional context containing event handlers for processing
        """
        self.agent = agent
        self.runner_config = runner_config or RunnerConfig()
        self.session_manager = SessionManager(
            session_service=self.runner_config.session_service
        )
        self.config_context = config_context
        self.handler_context = handler_context or HandlerContext()
        self.shutdown_handler = ShutdownHandler()
        self.session_lock_handler = self.handler_context.session_lock_handler(
            config_context.session_lock_config
        )

    async def _get_config_value(
        self, config_attr: str, agui_content: RunAgentInput, request: Request
    ) -> str:
        """Extract configuration value from context config.

        Handles both static string values and dynamic callable configurations
        that can extract values from the request context. This enables flexible
        multi-tenant configuration based on request characteristics.

        Args:
            :param config_attr: Name of the configuration attribute to retrieve
            :param agui_content: Input containing agent execution parameters
            :param request: HTTP request for context extraction

        Returns:
            Configuration value as string
        """
        value: Callable[[RunAgentInput, Request], Awaitable[str]] | str = getattr(
            self.config_context, config_attr
        )
        if callable(value):
            return await value(agui_content, request)
        return value

    async def _create_and_record_message(
        self, input_info: InputInfo
    ) -> BaseInOutHandler | None:
        """Create and record incoming message for audit and logging purposes.

        Creates an input/output record handler if configured and records the
        incoming AGUI content and request for audit trails and logging.

        Args:
            :param input_info: Processed request context for the current interaction

        Returns:
            BaseInOutHandler instance if configured, None otherwise
        """
        if self.handler_context.in_out_record_handler:
            in_out_record = self.handler_context.in_out_record_handler()
            await in_out_record.input_record(input_info)
            return in_out_record
        return None

    @staticmethod
    async def _record_output_message(
        inout_handler: BaseInOutHandler | None, output_data: BaseEvent
    ) -> BaseEvent:
        """Record and potentially transform outgoing message data.

        Records the output data through the configured handler and applies any
        transformations or modifications before sending to the client.

        Args:
            :param inout_handler: Optional handler for recording and transforming output
            :param output_data: AGUI BaseEvent to process

        Returns:
            BaseEvent (potentially modified by handler)
        """
        if inout_handler:
            await inout_handler.output_record(output_data)
            return await inout_handler.output_catch_and_change(output_data)
        return output_data

    async def extract_app_name(
        self, agui_content: RunAgentInput, request: Request
    ) -> str:
        """Extract application name from the request context.

        Uses the configured app_name extractor to determine the application
        name for the current request, enabling multi-tenant deployments.

        Args:
            :param agui_content: Input containing agent execution parameters
            :param request: HTTP request for context extraction

        Returns:
            Application name string
        """
        return await self._get_config_value("app_name", agui_content, request)

    async def extract_user_id(
        self, agui_content: RunAgentInput, request: Request
    ) -> str:
        """Extract user identifier from the request context.

        Uses the configured user_id extractor to determine the user
        identity for the current request, essential for session isolation.

        Args:
            :param agui_content: Input containing agent execution parameters
            :param request: HTTP request for context extraction

        Returns:
            User identifier string
        """
        return await self._get_config_value("user_id", agui_content, request)

    async def extract_session_id(
        self, agui_content: RunAgentInput, request: Request
    ) -> str:
        """Extract session identifier from the request context.

        Uses the configured session_id extractor to determine the session
        identity for the current request, enabling conversation persistence.

        Args:
            :param agui_content: Input containing agent execution parameters
            :param request: HTTP request for context extraction

        Returns:
            Session identifier string
        """
        return await self._get_config_value("session_id", agui_content, request)

    async def extract_initial_state(
        self, agui_content: RunAgentInput, request: Request
    ) -> dict[str, Any] | None:
        """Extract initial state dictionary from the request context.

        Uses the configured initial state extractor to determine any
        initial session state for new sessions, enabling context-aware initialization.

        Args:
            :param agui_content: Input containing agent execution parameters
            :param request: HTTP request for context extraction

        Returns:
            Dictionary containing initial state key-value pairs, or None
        """
        value = self.config_context.extract_initial_state
        if callable(value):
            return await value(agui_content, request)
        return value

    def _encode_event_to_sse(self, event: BaseEvent) -> dict[str, str] | str:
        """Handle event encoding with error recovery.

        Attempts to encode the event using the provided encoder, falling back
        to error event encoding if the primary encoding fails. This ensures
        that clients always receive valid SSE events even when encoding errors occur.

        Args:
            :param event: Base event to be encoded

        Returns:
            Encoded event dictionary in SSE format, either successful encoding or error event
        """
        converter = (
            convert_agui_event_to_sse
            if self.config_context.event_source_response_mode
            else convert_agui_event_to_str_fake_sse
        )
        try:
            return converter(event)
        except Exception as e:
            return converter(AGUIErrorEvent.create_encoding_error_event(e))

    async def _create_runner(self, app_name: str) -> Runner:
        """Create a fresh ADK Runner instance for the provided app name.

        Clones the agent to avoid cross-request state leakage and wires in
        session, artifact, memory, and credential services from the runner config.

        Args:
            :param app_name: Application name bound to the runner instance

        Returns:
            Configured Runner ready to execute agent operations
        """
        return Runner(
            app_name=app_name,
            agent=self.agent.model_copy(deep=True),
            session_service=self.session_manager.session_service,
            artifact_service=self.runner_config.get_artifact_service(),
            memory_service=self.runner_config.get_memory_service(),
            credential_service=self.runner_config.get_credential_service(),
            plugins=self.runner_config.plugins,
        )

    async def get_runner(
        self, agui_content: RunAgentInput, request: Request
    ) -> tuple[
        Callable[[], AsyncGenerator[BaseEvent]],
        InputInfo,
        BaseInOutHandler | None,
    ]:
        """Create a configured runner function for the given request.

        Extracts context from the request, creates necessary handlers and managers,
        and returns a runner function that generates agent events.

        Args:
            :param agui_content: Input containing agent execution parameters
            :param request: HTTP request containing client context

        Returns:
            Callable that returns an async generator of BaseEvent objects
        """

        input_info = InputInfo(
            agui_content=agui_content,
            request=request,
            app_name=await self.extract_app_name(agui_content, request),
            user_id=await self.extract_user_id(agui_content, request),
            session_id=await self.extract_session_id(agui_content, request),
            initial_state=await self.extract_initial_state(agui_content, request),
            event_queue=QueueHandler(
                EventQueue(
                    adk_event_queue=asyncio.Queue(),
                    agui_event_queue=asyncio.Queue(),
                )
            ),
        )

        async def runner() -> AsyncGenerator[BaseEvent]:
            """Internal runner function that executes the agent and yields events.

            Yields:
                BaseEvent objects representing agent execution events
            """
            if not await self.session_lock_handler.lock(input_info):
                yield await self.session_lock_handler.get_locked_message(input_info)
                return

            user_handler = AGUIUserHandler(
                running_handler=RunningHandler(
                    runner=await self._create_runner(input_info.app_name),
                    run_config=self.runner_config.run_config,
                    handler_context=self.handler_context,
                    input_info=input_info,
                    event_translator=EventTranslator(
                        retune_on_stream_complete=self.config_context.retune_on_stream_complete,
                        add_raw_event=self.config_context.is_add_adk_event_in_agui_event,
                    ),
                ),
                user_message_handler=UserMessageHandler(
                    agui_content=agui_content,
                    request=request,
                    initial_state=input_info.initial_state,
                    convert_run_agent_input=self.config_context.convert_run_agent_input,
                ),
                session_handler=SessionHandler(
                    session_manager=self.session_manager,
                    session_parameter=SessionParameter(
                        app_name=input_info.app_name,
                        user_id=input_info.user_id,
                        session_id=input_info.session_id,
                    ),
                ),
                queue_handler=input_info.event_queue,
            )
            async for event in user_handler.run():
                if self.config_context.auto_remove_agui_raw_event:
                    event.raw_event = None
                yield event

        in_out_record = await self._create_and_record_message(input_info)
        return runner, input_info, in_out_record

    async def event_generator(
        self,
        runner: Callable[[], AsyncGenerator[BaseEvent]],
        input_info: InputInfo,
        inout_handler: BaseInOutHandler | None = None,
    ) -> EventSourceResponse | StreamingResponse:
        """Generate encoded event strings from the agent runner.

        Executes the runner and encodes each event for SSE transmission,
        handling any errors that occur during execution or encoding.

        Args:
            :param runner: Callable that returns an async generator of events
            :param input_info: Input information containing session and context details
            :param inout_handler: Optional handler for input/output recording and transformation

        Yields:
            Encoded event dictionaries ready for SSE transmission
        """

        async def _generate() -> AsyncGenerator[dict[str, str] | str]:
            """Internal generator for processing events with error handling.

            Yields:
                Encoded event dictionaries or error events if exceptions occur
            """
            try:
                async for event in runner():
                    yield self._encode_event_to_sse(
                        await self._record_output_message(inout_handler, event)
                    )
            except Exception as e:
                yield self._encode_event_to_sse(
                    await self._record_output_message(
                        inout_handler, AGUIErrorEvent.create_agent_error_event(e)
                    )
                )
            finally:
                await self.session_lock_handler.unlock(input_info)

        if self.config_context.event_source_response_mode:
            return EventSourceResponse(_generate())
        return StreamingResponse(cast(AsyncGenerator[str], _generate()))
