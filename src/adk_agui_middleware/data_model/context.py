# Copyright (C) 2025 Trend Micro Inc. All rights reserved.
"""Configuration models for AGUI middleware context and runner setup."""

from collections.abc import Awaitable, Callable
from typing import Any

from ag_ui.core import RunAgentInput
from fastapi import Request
from pydantic import BaseModel, ConfigDict, Field

from ..base_abc.handler import (
    BaseADKEventHandler,
    BaseADKEventTimeoutHandler,
    BaseAGUIEventHandler,
    BaseAGUIStateSnapshotHandler,
    BaseInOutHandler,
    BaseTranslateHandler,
    SessionLockHandler,
)
from ..handler.session_lock_handler import DefaultSessionLockHandler
from ..tools.default_session_id import default_session_id
from .common import SessionLockConfig


def _get_default_session_lock_handler() -> type[SessionLockHandler]:
    """Return the default session lock handler implementation.

    Provides the default concrete ``SessionLockHandler`` used when no custom
    handler is supplied via configuration.

    :returns: ``DefaultSessionLockHandler`` class implementing the lock contract
    """
    return DefaultSessionLockHandler


class HandlerContext(BaseModel):
    """Context container for optional event and state handlers.

    Provides a configuration structure for customizing event processing
    behavior by injecting custom handlers at different stages of the pipeline.
    This enables extensible event processing through dependency injection.

    Attributes:
        session_lock_handler: Concrete SessionLockHandler class to enforce per-session mutual exclusion
        adk_event_handler: Optional handler for processing ADK events before translation
        adk_event_timeout_handler: Optional handler for managing ADK event timeouts
        agui_event_handler: Optional handler for processing AGUI events before transmission
        agui_state_snapshot_handler: Optional handler for processing state snapshots
        translate_handler: Optional handler for custom event translation logic
        in_out_record_handler: Optional handler for input/output recording and transformation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_lock_handler: type[SessionLockHandler] = Field(
        default_factory=_get_default_session_lock_handler
    )
    adk_event_handler: type[BaseADKEventHandler] | None = None
    adk_event_timeout_handler: type[BaseADKEventTimeoutHandler] | None = None
    agui_event_handler: type[BaseAGUIEventHandler] | None = None
    agui_state_snapshot_handler: type[BaseAGUIStateSnapshotHandler] | None = None
    translate_handler: type[BaseTranslateHandler] | None = None
    in_out_record_handler: type[BaseInOutHandler] | None = None


class ConfigContext(BaseModel):
    """Configuration for extracting context information from requests.

    Defines how to extract application name, user ID, session ID, and initial state
    from incoming requests. Each field can be either a static value or a callable
    that extracts the value dynamically from the request. This enables flexible
    multi-tenant configuration and request-specific context extraction.

    Attributes:
        app_name: Static application name or callable to extract from request context
        user_id: Static user ID or callable to extract from request context (required)
        session_id: Static session ID or callable to extract from request context
        extract_initial_state: Optional callable to extract initial session state from request
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    app_name: str | Callable[[RunAgentInput, Request], Awaitable[str]] = "default"
    user_id: str | Callable[[RunAgentInput, Request], Awaitable[str]]
    session_id: str | Callable[[RunAgentInput, Request], Awaitable[str]] = (
        default_session_id
    )
    extract_initial_state: (
        Callable[[RunAgentInput, Request], Awaitable[dict[str, Any]]] | None
    ) = None
    convert_run_agent_input: (
        Callable[[RunAgentInput, dict[str, str]], Awaitable[RunAgentInput]] | None
    ) = None
    session_lock_config: SessionLockConfig = SessionLockConfig()
    event_source_response_mode: bool = False
    retune_on_stream_complete: bool = False
    is_add_adk_event_in_agui_event: bool = False
    auto_remove_agui_raw_event: bool = False
