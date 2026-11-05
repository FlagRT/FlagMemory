"""流登记表：池见过的每条流都**强持有**到池销毁（不变量 `I-3`）。

`I-3` 的由来：**流 + 弱引用 + 被回收**这三件事同时成立，会导致进程退出期段错误，
平头哥 PPU 与昇腾 910C 两家在四家矩阵里中招，且是最难查的一类故障。所以本文件
刻意不 import `weakref`，也不提供任何「空闲久了就清理」的入口 —— 没有弱引用，
就没有那条故障路径。

另一条（设计文档 2.3 末段）：**池不改流优先级，也不靠流优先级表达回收次序。**
四家里流优先级的可设置性没有一家能全指望（一家丢参数、一家退化成单档），
回收次序由池自己的档位与用途标签决定（见 `types.RECLAIM_ORDER`）。
"""

from __future__ import annotations

from typing import Any

from .types import InvalidArgument

__all__ = ["StreamRegistry", "DEFAULT_STREAM_ID"]

#: 缺省流的池内标识。分配请求不给流时绑它（设计文档 2.3）。
DEFAULT_STREAM_ID = 0


class StreamRegistry:
    """池内流标识 ↔ 流对象。标识对外，流对象只在池内，且只强引用。"""

    def __init__(self, backend: Any, device_index: int) -> None:
        self._backend = backend
        self._device_index = device_index
        # 刻意是普通 dict：强引用，保持到池销毁（I-3）。不是 WeakValueDictionary。
        self._by_id: dict[int, Any] = {}
        self._next_id = DEFAULT_STREAM_ID + 1

    # —— 登记 ——

    def register(self, stream: Any) -> int:
        """登记一条流并强持有，返回池内流标识。已登记过的返回原标识。"""
        if stream is None:
            raise InvalidArgument("流对象不能是 None；不给流请省略参数，池会绑默认流。")
        self._validate(stream)
        existing = self.find(stream)
        if existing is not None:
            return existing
        stream_id = self._next_id
        self._next_id += 1
        self._by_id[stream_id] = stream
        return stream_id

    def register_default(self, stream: Any) -> int:
        """把某条流登记成默认流（建池时用当前流占住 `DEFAULT_STREAM_ID`）。"""
        if stream is None:
            raise InvalidArgument("默认流对象不能是 None。")
        self._validate(stream)
        self._by_id[DEFAULT_STREAM_ID] = stream
        return DEFAULT_STREAM_ID

    def _validate(self, stream: Any) -> None:
        try:
            self._backend.validate_stream_device(stream, self._device_index)
        except Exception as exc:
            raise InvalidArgument(f"无法确认流属于设备 {self._device_index}: {exc}") from exc

    # —— 查询 ——

    def find(self, stream: Any) -> int | None:
        """这条流登记过吗。各家流对象的 `==` 语义不一致，一律走后端的 `streams_equal`。"""
        for stream_id, known in self._by_id.items():
            if known is stream:
                return stream_id
            try:
                if self._backend.streams_equal(known, stream):
                    return stream_id
            except Exception:
                # 比较失败不等于不同：比不了就只认身份相等，别猜。
                continue
        return None

    def get(self, stream_id: int) -> Any:
        """按标识取流对象。未登记就报错，不返回 None 让调用方去猜。"""
        try:
            return self._by_id[stream_id]
        except KeyError:
            raise InvalidArgument(
                f"流标识 {stream_id} 未登记。请先 register_stream，池不替调用方猜流。"
            ) from None

    def has(self, stream_id: int) -> bool:
        return stream_id in self._by_id

    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_id))

    def __len__(self) -> int:
        return len(self._by_id)

    # —— 销毁 ——

    def release_all(self) -> int:
        """销毁池时才放开全部强引用，返回放开的条数。

        这是**唯一**的释放入口：没有超时清理、没有引用计数清理、没有弱引用回调。
        """
        count = len(self._by_id)
        self._by_id.clear()
        return count
