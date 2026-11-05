"""建池换出回调（分层换出接线）的契约。全部不依赖真卡。

判据来源：`docs/design/分层存储概念与设计基础.md` 1.2 第 2 条（池到分层只经建池登记的
一个换出回调，不反向 import，不变量 `T-5`）与 6.2 的 `reclaim` 行；《显存池概念与设计
基础》3.3 压力回收行与 `I-10`（失败不静默退化：未登记回调仍报 `NotWired`、回调异常
原样上抛不吞）。
"""

from __future__ import annotations

import pytest

from flagmemory.pool import (
    InvalidArgument,
    NotWired,
    PoolForm,
    Purpose,
    WorkloadType,
    create_pool,
)


class _Recorder:
    """测试替身：记录每次收到的池实例，回报固定字节数。"""

    def __init__(self, returns):
        self.calls: list[object] = []
        self._returns = returns

    def __call__(self, pool):
        self.calls.append(pool)
        return self._returns


def _make_pool(handle_backend_done, tier_spec, offload_callback=None):
    return create_pool(
        handle_backend_done,
        form=PoolForm.HANDLE,
        workload=WorkloadType.GENERATIVE,
        tier_spec=tier_spec,
        offload_callback=offload_callback,
    )


class TestNormal:
    """正常：回调被调一次、回报字节计入结果、接线与否可观测、换出后回收照常。"""

    def test_登记回调后允许分层换出时回调被调一次且拿到池实例(
        self, handle_backend_done, tier_spec
    ):
        recorder = _Recorder(0)
        pool = _make_pool(handle_backend_done, tier_spec, offload_callback=recorder)
        pool.reclaim(target_bytes=1 << 20, allow_tier_offload=True)
        assert len(recorder.calls) == 1
        assert recorder.calls[0] is pool

    def test_回调回报的换出字节数计入结果(self, handle_backend_done, tier_spec):
        pool = _make_pool(handle_backend_done, tier_spec, offload_callback=_Recorder(3 << 20))
        result = pool.reclaim(target_bytes=1 << 20, allow_tier_offload=True)
        assert result.tier_offloaded_bytes == 3 << 20

    def test_能力视图显示换出回调接线与否(self, handle_backend_done, tier_spec):
        wired = _make_pool(handle_backend_done, tier_spec, offload_callback=_Recorder(0))
        unwired = _make_pool(handle_backend_done, tier_spec)
        assert wired.query_capability().tier_offload_wired is True
        assert unwired.query_capability().tier_offload_wired is False

    def test_回调用池公开操作换出后进程内回收照常走(self, handle_backend_done, tier_spec):
        """分层侧收到压力事件后自行换出（取块 → 搬运后归还 → 推进待回收），池照常回收。"""

        def offload(pool_self):
            view = pool_self.alloc_on_stream(1 << 20, purpose=Purpose.KV_CACHE)
            pool_self.free_block(view.block_id)
            pool_self.drain_stream()
            return 1 << 20

        pool = _make_pool(handle_backend_done, tier_spec, offload_callback=offload)
        result = pool.reclaim(target_bytes=1 << 20, allow_tier_offload=True)
        assert result.tier_offloaded_bytes == 1 << 20
        assert result.released_by_purpose == {Purpose.KV_CACHE: 1 << 20}
        assert result.released_bytes == 1 << 20
        assert result.met_target is True


class TestBoundary:
    """边界：None/0 如实计、默认不调回调、未登记仍 NotWired、目标达成合计判读。"""

    @pytest.mark.parametrize("returns", [None, 0])
    def test_回调无回报或回报零都按零计(self, handle_backend_done, tier_spec, returns):
        pool = _make_pool(handle_backend_done, tier_spec, offload_callback=_Recorder(returns))
        result = pool.reclaim(target_bytes=1 << 20, allow_tier_offload=True)
        assert result.tier_offloaded_bytes == 0

    def test_默认不允许分层换出时不调回调(self, handle_backend_done, tier_spec):
        recorder = _Recorder(1 << 20)
        pool = _make_pool(handle_backend_done, tier_spec, offload_callback=recorder)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(view.block_id)
        pool.drain_stream()
        result = pool.reclaim(target_bytes=1 << 20)
        assert recorder.calls == []
        assert result.tier_offloaded_bytes == 0
        assert result.released_bytes == 1 << 20
        assert result.met_target is True

    def test_未登记回调而允许分层换出仍报NotWired(self, handle_backend_done, tier_spec):
        pool = _make_pool(handle_backend_done, tier_spec)
        with pytest.raises(NotWired) as exc:
            pool.reclaim(target_bytes=1 << 20, allow_tier_offload=True)
        assert "没有接线" in str(exc.value)

    def test_目标达成按进程内与分层换出合计判读(self, handle_backend_done, tier_spec):
        """空池 + 回调换出正好达标：进程内释放为 0 也不谎报「未达目标」。"""
        pool = _make_pool(handle_backend_done, tier_spec, offload_callback=_Recorder(1 << 20))
        result = pool.reclaim(target_bytes=1 << 20, allow_tier_offload=True)
        assert result.released_bytes == 0
        assert result.tier_offloaded_bytes == 1 << 20
        assert result.met_target is True
        assert "分层换出" in result.note


class TestError:
    """异常：回调抛错原样上抛、账本不受破坏、不可调用对象建池即报错。"""

    def test_回调抛错原样上抛且池账本不受破坏(self, handle_backend_done, tier_spec):
        class TierSideBroken(Exception):
            pass

        def offload(pool_self):
            raise TierSideBroken("分层侧炸了（用例刻意造的）")

        pool = _make_pool(handle_backend_done, tier_spec, offload_callback=offload)
        view = pool.alloc_on_stream(1 << 20, purpose=Purpose.ACTIVATION)
        pool.free_block(view.block_id)
        pool.drain_stream()

        with pytest.raises(TierSideBroken):
            pool.reclaim(target_bytes=1 << 20, allow_tier_offload=True)

        # 回调炸在进程内回收之前：可复用块原封未动，池后续操作照常。
        assert pool.query_fragmentation().watermark.pool_reusable == 1 << 20
        after = pool.reclaim(target_bytes=1 << 20)
        assert after.released_bytes == 1 << 20
        assert after.met_target is True

    def test_登记不可调用的回调建池即报错(self, handle_backend_done, tier_spec):
        with pytest.raises(InvalidArgument):
            _make_pool(handle_backend_done, tier_spec, offload_callback=object())
