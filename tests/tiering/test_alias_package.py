"""兼容别名包 offload.* 的懒加载与弃用警告（第 1 件改造的成果，这里补测）。

本机与 CI 都没有 torch / vLLM，所以**取类必然抛 ModuleNotFoundError**。要断言的是
「警告先发出」与「异常类型对」，不是「取类成功」—— 取类成功只能在带卡机器的栈内验，
那部分在 test_offload_realcard.py 里，标了 realcard。
"""

from __future__ import annotations

import importlib
import importlib.util
import warnings

import pytest

ALIASES = {
    "offload.mlu_offload": ("MluSimpleCPUOffloadConnector", "flagmemory.tiering.offload.mlu"),
    "offload.npu_offload": ("NpuSimpleCPUOffloadConnector", "flagmemory.tiering.offload.npu"),
    "offload.ppu_offload": ("PpuSimpleCPUOffloadConnector", "flagmemory.tiering.offload.ppu"),
    "offload.xpu_offload": ("XpuSimpleCPUOffloadConnector", "flagmemory.tiering.offload.xpu"),
}

#: 宿主栈在不在。不在（本机 / CI）时取类应抛 ModuleNotFoundError；在（带卡机器）时应拿到类。
HOST_STACK_PRESENT = all(
    importlib.util.find_spec(name) is not None for name in ("torch", "vllm")
)


class TestNormal:
    """正常：别名包与三个别名子包本身可 import，且不触发栈依赖。"""

    def test_别名包可import(self):
        mod = importlib.import_module("offload")
        assert set(mod.__all__) == {"mlu_offload", "npu_offload", "ppu_offload", "xpu_offload"}

    @pytest.mark.parametrize("alias", sorted(ALIASES))
    def test_别名子包可import且不触发栈依赖(self, alias):
        # 这是别名包设计的要点：三套栈各绑一个宿主栈版本，无条件 import 会把另两套的
        # ImportError 带给调用方，所以加载必须推迟到取属性时。
        mod = importlib.import_module(alias)
        assert mod.__all__ == [ALIASES[alias][0]]

    @pytest.mark.parametrize("alias", sorted(ALIASES))
    def test_取类时先发DeprecationWarning(self, alias):
        expected_class, new_path = ALIASES[alias]
        mod = importlib.import_module(alias)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                obtained = getattr(mod, expected_class)
            except ModuleNotFoundError as exc:
                # 无宿主栈环境的预期路径：警告必须已经发出，异常类型必须是 ModuleNotFoundError。
                assert not HOST_STACK_PRESENT, f"宿主栈在位却报 {exc}"
                obtained = None
            else:
                assert HOST_STACK_PRESENT
                assert isinstance(obtained, type)
        messages = [str(w.message) for w in caught if w.category is DeprecationWarning]
        assert messages, f"{alias} 取类没发 DeprecationWarning（obtained={obtained!r}）"
        assert "0.3.0" in messages[0]
        assert new_path in messages[0]


class TestBoundary:
    """边界：取不存在的属性、取 dunder、取 __all__ 都不许走懒加载那条路。"""

    @pytest.mark.parametrize("alias", sorted(ALIASES))
    def test_取不存在的属性抛AttributeError(self, alias):
        mod = importlib.import_module(alias)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(AttributeError):
                mod.NoSuchThing
        # 取不存在的东西不该发弃用警告（否则日志会被噪声淹掉）
        assert [w for w in caught if w.category is DeprecationWarning] == []

    @pytest.mark.parametrize("alias", sorted(ALIASES))
    def test_dir给出导出名(self, alias):
        mod = importlib.import_module(alias)
        assert dir(mod) == [ALIASES[alias][0]]

    def test_取别名包的子包名不抛(self):
        mod = importlib.import_module("offload")
        for name in mod.__all__:
            assert importlib.import_module(f"offload.{name}") is not None


class TestError:
    """异常：无宿主栈环境下取类的失败点必须在取属性那一刻，而不是 import 别名包那一刻。"""

    def test_import别名包不抛框架缺失(self):
        # 若这里抛 ModuleNotFoundError('torch')，说明别名包退回了无条件 import，是回归。
        importlib.import_module("offload")
        importlib.import_module("offload.npu_offload")

    @pytest.mark.parametrize("alias", sorted(ALIASES))
    def test_无宿主栈时取类抛ModuleNotFoundError(self, alias):
        expected_class, _ = ALIASES[alias]
        mod = importlib.import_module(alias)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            if HOST_STACK_PRESENT:
                assert isinstance(getattr(mod, expected_class), type)
            else:
                with pytest.raises(ModuleNotFoundError):
                    getattr(mod, expected_class)
