"""Simple event system for agent communication."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class EventTopic(ABC):
    """Abstract base class for event topics."""

    @abstractmethod
    async def publish(self, event_type: str, source: str, data: Dict[str, Any]) -> None:
        """Publish an event."""
        pass

    @abstractmethod
    async def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type."""
        pass


class SimpleEventTopic(EventTopic):
    """Simple in-memory event topic implementation."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}

    async def publish(self, event_type: str, source: str, data: Dict[str, Any]) -> None:
        """Publish an event to all subscribers."""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_type, source, data)
                    else:
                        callback(event_type, source, data)
                except Exception as e:
                    logger.error(f"Error in event callback for {event_type}: {e}")

    async def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)


class PhonyTopic(EventTopic):
    """No-op event topic for testing or when events are not needed."""

    async def publish(self, event_type: str, source: str, data: Dict[str, Any]) -> None:
        """Do nothing."""
        pass

    async def subscribe(self, event_type: str, callback: Callable) -> None:
        """Do nothing."""
        pass
