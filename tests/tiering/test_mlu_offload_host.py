"""mlu 卸载子包的主机可跑契约（结构 / 参数面 / 设备面纪律）。

本机与 CI 没有 torch / vllm / torch_mlu（mlu 子包 import 即触发栈依赖），所以这里用
ast 读源码验契约，不 import 被测模块；「能 import + 取到类」的真契约在
test_offload_realcard.py 的 mlu 行（realcard 标记，推理腿镜像容器内跑）。

验什么：
- 结构契约：四件套齐全，导出类与 npu/xpu 同名族（MluSimpleCPUOffloadConnector）；
- 参数面契约：connector / worker 的 __init__ 形参与 npu/xpu 及上游 0.20.2 一致；
- 设备面纪律：子包源码零 torch.cuda 引用（torch_mlu 不补 torch.cuda 面，全走 torch.mlu.*）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PKG = (
    Path(__file__).resolve().parent.parent.parent
    / "flagmemory"
    / "tiering"
    / "offload"
    / "mlu"
)


def _src(name: str) -> str:
    path = PKG / name
    assert path.is_file(), f"missing {name} in offload/mlu/"
    return path.read_text(encoding="utf-8")


def _tree(name: str) -> ast.Module:
    return ast.parse(_src(name))


def _classes(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {
        n.name: n for n in tree.body if isinstance(n, ast.ClassDef)
    }


def _funcs(tree: ast.Module) -> set[str]:
    return {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}


def _init_params(cls: ast.ClassDef) -> tuple[str, ...]:
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            return tuple(a.arg for a in item.args.args)
    return ()


def _methods(cls: ast.ClassDef) -> set[str]:
    return {
        item.name
        for item in cls.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class TestNormal:
    """正常：四件套齐全、导出面与参数面同 npu/xpu 族。"""

    def test_四件套齐全(self):
        for name in ("__init__.py", "connector.py", "mlu_worker.py",
                     "mlu_copy_backend.py", "mlu_mem_ops.py"):
            assert (PKG / name).is_file(), f"missing offload/mlu/{name}"

    def test_包导出唯一connector类(self):
        tree = _tree("__init__.py")
        exports = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                exports = {e.value for e in node.value.elts}
        assert exports == {"MluSimpleCPUOffloadConnector"}
        # 导出行真实存在（from .connector import MluSimpleCPUOffloadConnector）
        assert "from .connector import MluSimpleCPUOffloadConnector" in _src("__init__.py")

    def test_connector参数面与上游一致(self):
        classes = _classes(_tree("connector.py"))
        assert "MluSimpleCPUOffloadConnector" in classes
        cls = classes["MluSimpleCPUOffloadConnector"]
        bases = [ast.unparse(b) for b in cls.bases]
        assert any("SimpleCPUOffloadConnector" in b for b in bases), bases
        assert _init_params(cls) == ("self", "vllm_config", "role", "kv_cache_config")

    def test_worker参数面与上游一致(self):
        classes = _classes(_tree("mlu_worker.py"))
        assert "MluSimpleCPUOffloadWorker" in classes
        cls = classes["MluSimpleCPUOffloadWorker"]
        bases = [ast.unparse(b) for b in cls.bases]
        assert any("SimpleCPUOffloadWorker" in b for b in bases), bases
        # npu/xpu 族形参（vllm_config, kv_cache_config, cpu_capacity_bytes）
        assert _init_params(cls) == ("self", "vllm_config", "kv_cache_config",
                                     "cpu_capacity_bytes")
        # 注册归一两件辅助函数同族
        assert {"normalize_kv_caches", "_spec_page_sizes"} <= _funcs(_tree("mlu_worker.py"))

    def test_copy_backend鸭子面(self):
        classes = _classes(_tree("mlu_copy_backend.py"))
        assert "MluCopyBackend" in classes
        # 上游 worker 驱动后端的三件方法 + 守护线程循环
        assert {"init", "launch_copy", "shutdown", "_copy_loop"} <= _methods(
            classes["MluCopyBackend"]
        )

    def test_mem_ops两件辅助(self):
        funcs = _funcs(_tree("mlu_mem_ops.py"))
        assert {"pin_or_fallback", "stream_priority_range"} <= funcs


def _torch_cuda_code_refs(tree: ast.Module) -> list[str]:
    """收集代码层 torch.cuda.* 属性引用（docstring/注释里的叙述不算）。"""
    refs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            text = ast.unparse(node)
            if text == "torch.cuda" or text.startswith("torch.cuda."):
                refs.append(text)
    return refs


class TestBoundary:
    """边界：设备面纪律——mlu 子包代码零 torch.cuda 引用（torch_mlu 不补 torch.cuda 面）。

    docstring/注释里对 stock 死点的叙述允许出现 torch.cuda 字样，算的是代码引用。
    """

    @pytest.mark.parametrize(
        "name",
        ["__init__.py", "connector.py", "mlu_worker.py", "mlu_copy_backend.py",
         "mlu_mem_ops.py"],
    )
    def test_代码零torch_cuda引用(self, name):
        refs = _torch_cuda_code_refs(_tree(name))
        assert refs == [], (
            f"offload/mlu/{name} 代码引用了 {refs}：该栈 torch.cuda 非兼容面"
            "（is_available()=False），设备 API 必须走 torch.mlu.*"
        )

    def test_设备面走torch_mlu(self):
        # worker / copy_backend 必须显式 import torch_mlu 并用 torch.mlu.*
        worker = _src("mlu_worker.py")
        backend = _src("mlu_copy_backend.py")
        assert "import torch_mlu" in worker and "torch.mlu.Stream" in worker
        assert "import torch_mlu" in backend and "torch.mlu.Event" in backend

    def test_块数按spec页推导不信任num_blocks(self):
        # npu/xpu 族要点：actual_blocks = storage.nbytes() // page_size_bytes
        assert "storage.nbytes() // page_size_bytes" in _src("mlu_worker.py")
