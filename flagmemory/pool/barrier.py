"""回收屏障：只查询事件，以调用级单调时钟截止时间限制轮询等待。

未完成和等待到期均保持可恢复；仅后端错误报告 FAULT。零等待不计超时。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from .types import InvalidArgument

__all__ = ["BarrierTicket", "BarrierOutcome", "RecycleBarrier", "DEFAULT_TIMEOUT_S"]
DEFAULT_TIMEOUT_S = 0.05
_POLL_INTERVAL_S = 0.001


def validate_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidArgument("等待时间必须是有限非负实数")
    try:
        budget = float(value)
    except (ValueError, OverflowError) as exc:
        raise InvalidArgument("等待时间必须是有限非负实数") from exc
    if not math.isfinite(budget) or budget < 0:
        raise InvalidArgument("等待时间必须是有限非负实数")
    return budget


@dataclass
class BarrierTicket:
    """相关流各一个事件；隔离时也保留凭据，不隐式丢弃资源。"""
    events: dict[int, Any] = field(default_factory=dict)
    confirmed: set[int] = field(default_factory=set)
    timeouts: int = 0

    @property
    def outstanding(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.events) - self.confirmed))

    @property
    def complete(self) -> bool:
        return not self.outstanding


class BarrierOutcome:
    DONE = "done"
    PENDING = "pending"
    FAULT = "fault"


class RecycleBarrier:
    def __init__(self, backend: Any, *, device_index: int,
                 timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self._backend = backend
        self._device_index = device_index
        self._timeout_s = validate_timeout(timeout_s)

    def budget(self, timeout_s: float | None) -> float:
        return self._timeout_s if timeout_s is None else validate_timeout(timeout_s)

    def take_ticket(self, streams: dict[int, Any], ticket: BarrierTicket | None = None) -> BarrierTicket:
        """调用方可先附着凭据，再填充；中途失败仍持有已取得事件。"""
        if ticket is None:
            ticket = BarrierTicket()
        for stream_id, stream in streams.items():
            event = self._backend.create_event(self._device_index)
            ticket.events[stream_id] = event
            self._backend.record_event(event, stream)
        return ticket

    def advance(self, ticket: BarrierTicket, timeout_s: float | None = None,
                *, deadline: float | None = None) -> str:
        budget = self.budget(timeout_s)
        if deadline is None:
            deadline = time.monotonic() + budget
        while True:
            for stream_id in ticket.outstanding:
                try:
                    if self._backend.query_event(ticket.events[stream_id]):
                        ticket.confirmed.add(stream_id)
                except Exception:
                    return BarrierOutcome.FAULT
            if ticket.complete:
                return BarrierOutcome.DONE
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if budget > 0:
                    ticket.timeouts += 1
                return BarrierOutcome.PENDING
            time.sleep(min(_POLL_INTERVAL_S, remaining))
