"""回收屏障：判「块上的操作跑完了没有」。三条不变量全落在这个文件里。

| 不变量 | 这里怎么落 |
|---|---|
| `I-2` 判完成不裸调厂商事件查询 | 只经 `DeviceBackend.query_event`（统一后端接口已修正的事件适配器）。寒武纪 MLU590 与平头哥 PPU 上，未 record 的事件原生返回「已完成」，裸调会把在用显存判成可回收 |
| `I-11` 屏障不许阻塞等待 | 只用 `query_event` 轮询 + 单调钟截止时间。**刻意不调 `synchronize_event`** —— 四家里只有昇腾是真有界，其余三家的「有界」是超时上报或轮询实现，所以这里不借用任何「有界同步」的承诺 |
| `I-11` 超时不等于失败 | 超时返回 `pending`，块留在待回收并计一次超时；只有超时次数越过阈值才转隔离 |

跨流：屏障要等**所属流与跨流借用集里所有流**都确认完成（设计文档 2.4）。所以一张凭据
上挂的是「每条相关流各一个事件」，全部完成才放行 —— 而不是只看所属流那一个。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["BarrierTicket", "BarrierOutcome", "RecycleBarrier", "DEFAULT_TIMEOUT_S"]

#: 屏障默认超时（秒）。超时不等于失败，只是「这一轮没确认完」。
DEFAULT_TIMEOUT_S = 0.05

#: 轮询间隔（秒）。小到不明显拖慢 drain，大到不把 CPU 烧在空转上。
_POLL_INTERVAL_S = 0.001


@dataclass
class BarrierTicket:
    """一块的屏障凭据：相关流各一个事件。全部完成才放行到可复用。"""

    #: 池内流标识 → 事件对象
    events: dict[int, Any] = field(default_factory=dict)
    #: 已确认完成的流标识（确认过就不再查，省掉重复查询）
    confirmed: set[int] = field(default_factory=set)
    #: 累计超时次数。越过阈值才转隔离（超时本身不是失败）
    timeouts: int = 0

    @property
    def outstanding(self) -> tuple[int, ...]:
        """还没确认完成的流标识。"""
        return tuple(sorted(set(self.events) - self.confirmed))

    @property
    def complete(self) -> bool:
        return not self.outstanding


class BarrierOutcome:
    """一次屏障推进的结果（三态，刻意不用布尔）。"""

    DONE = "done"          # 全部相关流已确认完成 ⇒ 可转可复用
    PENDING = "pending"    # 还有流没确认完 ⇒ 留在待回收（可恢复态）
    FAULT = "fault"        # 事件适配器报错 ⇒ 转隔离并上报


class RecycleBarrier:
    """屏障的实现：取凭据、轮询推进。不持有块，只处理凭据。"""

    def __init__(
        self,
        backend: Any,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        timeout_threshold: int = 3,
    ) -> None:
        self._backend = backend
        self._timeout_s = float(timeout_s)
        #: 同一块累计超时多少次就转隔离（设计文档 2.2「屏障超阈 → 隔离」）
        self.timeout_threshold = int(timeout_threshold)

    # —— 取凭据 ——

    def take_ticket(self, streams: dict[int, Any]) -> BarrierTicket:
        """为「所属流 + 跨流借用集」里每条流各取一个事件凭据。

        取不到事件就让异常往上走：宁可归还失败并上报，也不给一张空凭据 ——
        空凭据会让块立刻变成「可复用」，等于把还在用的显存交给下一个人。
        """
        ticket = BarrierTicket()
        for stream_id, stream in streams.items():
            event = self._backend.create_event(getattr(stream, "device_index", 0))
            self._backend.record_event(event, stream)
            ticket.events[stream_id] = event
        return ticket

    # —— 推进 ——

    def advance(self, ticket: BarrierTicket, timeout_s: float | None = None) -> str:
        """轮询推进一张凭据，返回 `BarrierOutcome` 的三态之一。

        实现上只做两件事：调 `query_event`、看单调钟。**不调 `synchronize_event`、
        不做任何阻塞等待**（`I-11`）。
        """
        if ticket.complete:
            return BarrierOutcome.DONE

        budget = self._timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + max(budget, 0.0)

        while True:
            for stream_id in ticket.outstanding:
                event = ticket.events[stream_id]
                try:
                    # I-2：只经事件适配器。未 record 的事件适配器会报错，
                    # 而不是像原生那样误报「已完成」。
                    if self._backend.query_event(event):
                        ticket.confirmed.add(stream_id)
                except Exception:
                    # 适配器报错属账本/设备异常，不是「还没完成」：交给上层转隔离。
                    return BarrierOutcome.FAULT

            if ticket.complete:
                return BarrierOutcome.DONE
            if time.monotonic() >= deadline:
                ticket.timeouts += 1
                # 超时不等于失败（I-11）：留在待回收，按可恢复态上报。
                return BarrierOutcome.PENDING
            time.sleep(_POLL_INTERVAL_S)

    def over_threshold(self, ticket: BarrierTicket) -> bool:
        """超时次数是否已越过转隔离的阈值。"""
        return ticket.timeouts >= self.timeout_threshold
