"""不变量 `I-1`..`I-12` 逐条钉一次。全部不依赖真卡。

唯一登记处 = `docs/design/显存池概念与设计基础.md` 第三节第 5 小节，命名空间 = 显存池组件
内部（不与总组编号体系撞号）。那份文件的原话：「违反任一条即为实现缺陷，单测与门禁各钉一次」
—— 门禁那一次是 `tools/check_layering.py`，单测这一次就是本文件。

有几条是「不许调某个东西」形态的纪律（`I-2` / `I-3` / `I-4` / `I-5` / `I-11`）。这类用两种
方式钉：**行为**（装一个被调就炸的探针方法，看池到底有没有碰）与**源码**（扫池本体的源码里
有没有那个调用）。只验行为会漏掉「某条少见分支里偷偷调了」，只扫源码又证明不了运行时行为，
所以两样都要。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from flagmemory.caps.capabilities import Capability, L0_CAPABILITIES, L1_CAPABILITIES
from flagmemory.caps.fake import FakeStream, MemoryHandle
from flagmemory.pool import (
    BlockState,
    CapacityExhausted,
    FormUnavailable,
    OwnershipViolation,
    PoolForm,
    Purpose,
    TierSpec,
    WorkloadType,
    create_pool,
)
from fakes import CompletingBackend, FakeBaseAllocator

POOL_DIR = Path(__file__).resolve().parent.parent.parent / "flagmemory" / "pool"


def _pool_identifiers() -> dict[str, set[str]]:
    """池本体每个文件里**真实出现在代码里**的标识符（文件名 → 名字集合）。

    用 `ast` 而不是文本查找，和 `tools/check_layering.py` 同一套做法 —— 原因也一样：
    这些禁令的名字必然出现在文档与注释里（正是在解释「为什么不许调」），文本查找会把
    说明文字误判成违规。只有属性名、变量名与 import 名算真实使用。
    """
    out: dict[str, set[str]] = {}
    for path in sorted(POOL_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    names.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names.add(node.module.split(".")[0])
                for alias in node.names:
                    names.add(alias.name)
        out[path.name] = names
    return out


class _Tripwire(CompletingBackend):
    """被调就炸的探针：池如果碰了这些口，用例立刻红。"""

    def create_context(self, device_index: int = 0):  # L2，池一律不用（I-4）
        raise AssertionError("池调了上下文创建：违反 I-4")

    def set_context(self, ctx):
        raise AssertionError("池调了上下文切换：违反 I-4")

    def destroy_context(self, ctx):
        raise AssertionError("池调了上下文销毁：违反 I-4")

    def rebuild_device(self, device_index: int = 0):  # L3，只作诊断增强（I-5）
        raise AssertionError("池调了设备级重建：违反 I-5")

    def synchronize_event(self, event, timeout_s: float) -> bool:  # I-11
        raise AssertionError("池在屏障里用了有界同步/阻塞等待：违反 I-11")

    def set_stream_priority(self, stream, priority: int) -> None:
        raise AssertionError("池改了流优先级：违反设计文档 2.3")


class _SilentFreeBackend(CompletingBackend):
    """二次释放**静默返回成功**的后端（= 昇腾那一家的裸释放行为）。

    这正是 `I-1` 要防的局面：指望厂商报错，在这一家上会完全失效。
    """

    def free_handle(self, handle: MemoryHandle) -> None:
        self._handles.pop(handle.handle_id, None)   # 不在表里也不报错，静默成功


class TestInvariantI1:
    """`I-1` 回收不依赖厂商报错：由池自己的账本拦截。"""

    def test_厂商静默成功时池照样拦下重复归还(self, tier_spec: TierSpec):
        backend = _SilentFreeBackend(
            name="silent-free", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
        )
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(view.block_id)
        with pytest.raises(OwnershipViolation):
            pool.free_block(view.block_id)
        assert pool._ledger._blocks[view.storage_id].state is BlockState.PENDING_RECYCLE

    def test_厂商静默成功时池照样拦下非本池块(self, tier_spec: TierSpec):
        backend = _SilentFreeBackend(
            name="silent-free", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
        )
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        with pytest.raises(OwnershipViolation):
            pool.free_block(999)


class TestInvariantI2:
    """`I-2` 判完成不裸调厂商事件查询：一律走统一后端接口的事件适配器。"""

    def test_屏障只经后端的事件口(self):
        identifiers = _pool_identifiers()
        # 判完成走的是统一后端接口那个口
        assert "query_event" in identifiers["barrier.py"]
        # 池本体里不出现任何厂商事件查询（那些名字只该出现在设备方向的适配层里）
        for name, names in identifiers.items():
            for vendor in (
                "aclrtQueryEvent",
                "cnrtQueryNotifier",
                "cudaEventQuery",
                "hgEventQuery",
            ):
                assert vendor not in names, f"{name} 里调了厂商事件查询 {vendor}：违反 I-2"

    def test_事件先record再查询(self, handle_pool, handle_backend_done):
        """未 record 的事件在寒武纪与平头哥上原生误报「已完成」，适配层会报错 ——
        池先 record 再查，所以这条路上不该出现 EventNotRecorded。"""
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        handle_pool.free_block(view.block_id)
        assert handle_pool.drain_stream().promoted_blocks == 1


class TestInvariantI3:
    """`I-3` 池持有流必须强持有，禁止弱引用自动清理。"""

    def test_池本体代码里没有弱引用(self):
        """流 + 弱引用 + 被回收 = 退出期段错误（平头哥与昇腾两家中招）。"""
        for name, names in _pool_identifiers().items():
            for forbidden in ("weakref", "WeakValueDictionary", "WeakSet", "ref"):
                assert forbidden not in names, f"{name} 里用了 {forbidden}：违反 I-3"

    def test_登记过的流一直取得到(self, handle_pool):
        stream = FakeStream(stream_id=51)
        stream_id = handle_pool.register_stream(stream)
        for _ in range(5):
            view = handle_pool.alloc_on_stream(1 << 20, stream_id=stream_id, purpose=Purpose.ACTIVATION)
            handle_pool.free_block(view.block_id)
            handle_pool.drain_stream()
            handle_pool.compact()
        # 块来了又走，流对象还在，且还是同一个对象（强引用，不是重建）
        assert handle_pool.register_stream(stream) == stream_id

    def test_同一条流重复登记返回同一标识(self, handle_pool):
        stream = FakeStream(stream_id=52)
        assert handle_pool.register_stream(stream) == handle_pool.register_stream(stream)

    def test_销毁时才放开流引用(self, handle_pool):
        handle_pool.register_stream(FakeStream(stream_id=53))
        handle_pool.destroy_pool()
        assert handle_pool.destroyed is True


class TestInvariantI4:
    """`I-4` 池绝不自建、切换或销毁设备上下文。"""

    def test_池不碰上下文生命周期(self, tier_spec: TierSpec):
        backend = _Tripwire(
            name="tripwire",
            declared=frozenset(
                L0_CAPABILITIES | L1_CAPABILITIES | {Capability.CONTEXT_LIFECYCLE}
            ),
        )
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(view.block_id)
        pool.drain_stream()
        pool.compact()
        pool.destroy_pool()   # 探针没炸 = 池一路没碰上下文

    def test_池本体代码里没有上下文写操作(self):
        for name, names in _pool_identifiers().items():
            for forbidden in ("create_context", "set_context", "destroy_context", "push_context"):
                assert forbidden not in names, f"{name} 里调了 {forbidden}：违反 I-4"


class TestInvariantI5:
    """`I-5` 设备级故障只上报不自救：压力回收只做进程内回收。"""

    def test_压力回收不做设备级重建(self, tier_spec: TierSpec):
        backend = _Tripwire(
            name="tripwire",
            declared=frozenset(
                L0_CAPABILITIES | L1_CAPABILITIES | {Capability.DEVICE_LEVEL_REBUILD}
            ),
        )
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(view.block_id)
        pool.drain_stream()
        result = pool.reclaim(target_bytes=1 << 20)
        assert result.released_bytes == 1 << 20   # 回收做了，重建探针没炸

    def test_池本体代码里没有设备重建调用(self):
        for name, names in _pool_identifiers().items():
            offenders = [n for n in names if "rebuild" in n.lower() or "reset_device" in n.lower()]
            assert not offenders, f"{name} 里调了 {offenders}：违反 I-5（只上报不自救）"


class TestInvariantI6:
    """`I-6` 水位与碎片取数是双指标，单指标不得作为池内决策的唯一输入。"""

    def test_水位与碎片两侧都给(self, handle_pool):
        handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        wm = handle_pool.query_watermark()
        assert wm.device_total > 0
        assert wm.pool_in_use > 0
        frag = handle_pool.query_fragmentation()
        assert frag.watermark is not None
        assert frag.device_free_ratio >= 0.0 and frag.engine_fragmentation >= 0.0

    def test_容量不足的错误里带两侧读数(self, l0_backend_done, tier_spec: TierSpec):
        pool = create_pool(
            l0_backend_done,
            form=PoolForm.TENSOR,
            base_allocator=FakeBaseAllocator(capacity_bytes=1 << 20),
            tier_spec=tier_spec,
        )
        pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        with pytest.raises(CapacityExhausted) as exc:
            pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        text = str(exc.value)
        assert "设备侧" in text and "引擎侧" in text


class TestInvariantI7:
    """`I-7` 预算仲裁必须区分负载类型。"""

    @pytest.mark.parametrize(
        ("workload", "keyword"),
        [
            (WorkloadType.EMBEDDING, "纯余量"),
            (WorkloadType.GENERATIVE, "并发容量"),
            (WorkloadType.MIXED, "混合负载"),
        ],
    )
    def test_三种负载给三种判读(self, handle_backend_done, tier_spec: TierSpec, workload, keyword):
        pool = create_pool(
            handle_backend_done, form=PoolForm.HANDLE, workload=workload, tier_spec=tier_spec
        )
        budget = pool.query_budget()
        assert budget.workload is workload
        assert keyword in budget.compressible_reading

    def test_判读不是裸数字(self, handle_pool):
        budget = handle_pool.query_budget()
        assert isinstance(budget.compressible_bytes, int)
        assert len(budget.compressible_reading) > 10   # 数字必须配判读文本一起给


class TestInvariantI8:
    """`I-8` 核心路径只许用 L0；L1 走分支；L2 / L3 一律不用。"""

    def test_只有L0的后端能跑完整生命周期(self, tensor_pool):
        """= 寒武纪那一档：L1 四项全未声明，核心路径照样走通。"""
        view = tensor_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        tensor_pool.free_block(view.block_id)
        assert tensor_pool.drain_stream().promoted_blocks == 1
        assert tensor_pool.reclaim(target_bytes=1 << 20).met_target is True
        assert tensor_pool.destroy_pool().leaked == ()

    def test_L2能力声明了也不进分支清单(self, all_caps_backend, tier_spec: TierSpec):
        pool = create_pool(all_caps_backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        branches = pool.query_capability().branches_enabled
        assert Capability.CONTEXT_LIFECYCLE not in branches


class TestInvariantI9:
    """`I-9` 池不假设拿到的是裸内存；底座形态差异只在能力层体现。"""

    def test_底座引用可以是任意不透明对象(self, l0_backend_done, tier_spec: TierSpec):
        class Opaque:
            """既不是指针也不是整数，池不该对它做任何解释。"""

        class OpaqueAllocator:
            def __init__(self) -> None:
                self.live: list[object] = []

            def allocate(self, nbytes: int, device_index: int = 0, *, stream: object) -> object:
                ref = Opaque()
                self.live.append(ref)
                return ref

            def release(self, base_ref: object) -> None:
                self.live.remove(base_ref)

        allocator = OpaqueAllocator()
        pool = create_pool(
            l0_backend_done,
            form=PoolForm.TENSOR,
            base_allocator=allocator,
            tier_spec=tier_spec,
        )
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert len(allocator.live) == 1
        pool.free_block(view.block_id)
        pool.drain_stream()
        pool.compact()
        assert allocator.live == []

    def test_对外快照不暴露底座引用(self, handle_pool):
        view = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert not hasattr(view, "base_ref")


class TestInvariantI10:
    """`I-10` 失败不静默退化：报错，或显式降级并上报降级态。"""

    def test_形态不可用时报错而不是换形态(self, l0_backend_done, tier_spec: TierSpec):
        with pytest.raises(FormUnavailable):
            create_pool(l0_backend_done, form=PoolForm.HANDLE, tier_spec=tier_spec)

    def test_自动形态选了句柄形态要进降级清单(self, handle_backend_done, tier_spec: TierSpec):
        pool = create_pool(handle_backend_done, form=PoolForm.AUTO, tier_spec=tier_spec)
        assert pool.form is PoolForm.HANDLE
        keys = {d.key for d in pool.query_capability().degradations}
        assert "auto_form_picked_handle" in keys

    def test_每条降级都带代价说明(self, tensor_pool):
        for notice in tensor_pool.query_capability().degradations:
            assert notice.action and notice.cost

    def test_空操作不报成成功(self, handle_pool):
        assert handle_pool.compact().improved is False
        assert handle_pool.reclaim(target_bytes=1 << 20).met_target is False

    def test_缺上下文观测时归属校验那项是未校验而不是通过(self, tensor_pool):
        view = tensor_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        verdict = tensor_pool.verify_block_ownership(view.block_id)
        assert verdict.context_ok is None            # 未校验
        assert verdict.degradations                  # 且说明了为什么没校验
        assert verdict.device_ok and verdict.ledger_ok

    def test_有上下文观测时三项齐全(self, all_caps_backend, tier_spec: TierSpec):
        pool = create_pool(all_caps_backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        verdict = pool.verify_block_ownership(view.block_id)
        assert verdict.context_ok is True
        assert verdict.degradations == ()
        assert verdict.ok is True


class TestInvariantI11:
    """`I-11` 回收屏障不许阻塞等待，只许超时 + 轮询；超时不等于失败。"""

    def test_屏障里不调有界同步(self, tier_spec: TierSpec):
        backend = _Tripwire(
            name="tripwire", declared=frozenset(L0_CAPABILITIES | L1_CAPABILITIES)
        )
        pool = create_pool(backend, form=PoolForm.HANDLE, tier_spec=tier_spec)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(view.block_id)
        pool.drain_stream()   # 探针没炸 = 没走 synchronize_event

    def test_池本体代码里不调有界同步(self):
        for name, names in _pool_identifiers().items():
            for forbidden in ("synchronize_event", "synchronize", "wait_event"):
                assert forbidden not in names, (
                    f"{name} 里调了 {forbidden}：屏障只许超时 + 轮询，不许阻塞等待（I-11）"
                )

    def test_超时是可恢复态不抛异常(self, pending_pool):
        view = pending_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pending_pool.free_block(view.block_id)
        result = pending_pool.drain_stream(timeout_s=0.001)   # 真正等待到期，仍可恢复
        assert result.recoverable_timeout is True


class TestInvariantI12:
    """`I-12` 权重与通信缓冲永不被池回收或复用；未声明按最保守处理。"""

    @pytest.mark.parametrize(
        "purpose", [Purpose.WEIGHT, Purpose.COMM_BUFFER, Purpose.UNDECLARED]
    )
    def test_这三类归还后不进复用池(self, handle_pool, purpose):
        view = handle_pool.alloc_on_stream(1 << 20, purpose=purpose)
        handle_pool.free_block(view.block_id)
        handle_pool.drain_stream()
        assert handle_pool.query_watermark().pool_reusable == 0
        frag = handle_pool.query_fragmentation()
        assert all(t.reusable_blocks == 0 for t in frag.tiers)

    @pytest.mark.parametrize(
        "purpose", [Purpose.WEIGHT, Purpose.COMM_BUFFER, Purpose.UNDECLARED]
    )
    def test_这三类不会被下一次分配复用(self, handle_pool, purpose):
        first = handle_pool.alloc_on_stream(1 << 20, purpose=purpose)
        handle_pool.free_block(first.block_id)
        handle_pool.drain_stream()
        second = handle_pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        assert second.block_id != first.block_id
        assert handle_pool.query_audit().reuse_hits == 0

    def test_压力回收动不到这三类(self, handle_pool):
        for purpose in (Purpose.WEIGHT, Purpose.COMM_BUFFER, Purpose.UNDECLARED):
            view = handle_pool.alloc_on_stream(1 << 20, purpose=purpose)
            handle_pool.free_block(view.block_id)
        handle_pool.drain_stream()
        result = handle_pool.reclaim(target_bytes=3 << 20)
        assert result.released_bytes == 0
        assert result.met_target is False
        assert "I-12" in result.note

    def test_复用次序表里只有三类可回收标签(self):
        from flagmemory.pool import RECLAIM_ORDER

        assert RECLAIM_ORDER == (Purpose.SCRATCH, Purpose.ACTIVATION, Purpose.KV_CACHE)
