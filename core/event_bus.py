from __future__ import annotations

import asyncio
from typing import Generic, TypeVar

T = TypeVar("T")


class EventBus(Generic[T]):
    """Minimal fan-out pub/sub over asyncio queues. Subscribers must be
    registered before the first publish or they will miss prior events."""

    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue[T]] = []

    def subscribe(self, maxsize: int = 0) -> asyncio.Queue[T]:
        q: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(q)
        return q

    async def publish(self, event: T) -> None:
        for q in self._subscribers:
            await q.put(event)
