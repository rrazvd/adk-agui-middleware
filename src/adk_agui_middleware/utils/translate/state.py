# Copyright (C) 2025 Trend Micro Inc. All rights reserved.
"""Utility functions for creating state-related AGUI events."""

from typing import Any

from ag_ui.core import EventType, StateDeltaEvent, StateSnapshotEvent
from google.adk.events import Event


class StateEventUtil:
    """Utility class for creating state delta and snapshot events.

    Provides static methods for creating AGUI state events with proper typing
    and consistent format for state management operations.
    """

    @staticmethod
    def create_state_delta_event_with_json_patch(
        delta: list[dict[str, Any]], adk_event: Event | None = None
    ) -> StateDeltaEvent:
        """Create a state delta event with JSON Patch operations.

        Args:
            :param delta: List of JSON Patch operations to apply to state

        Returns:
            StateDeltaEvent containing the JSON Patch operations
        """
        return StateDeltaEvent(
            type=EventType.STATE_DELTA, delta=delta, raw_event=adk_event
        )

    @staticmethod
    def create_state_snapshot_event(
        state_snapshot: dict[str, Any], adk_event: Event | None = None
    ) -> StateSnapshotEvent:
        """Create a state snapshot event with complete state data.

        Args:
            :param state_snapshot: Complete state dictionary to snapshot

        Returns:
            StateSnapshotEvent containing the full state snapshot
        """
        return StateSnapshotEvent(
            type=EventType.STATE_SNAPSHOT, snapshot=state_snapshot, raw_event=adk_event
        )
