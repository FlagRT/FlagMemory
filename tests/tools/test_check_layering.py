"""分层门禁脚本本身的用例。不依赖真卡，也不依赖 pytest 之外的任何东西。

违规样本一律造在 pytest 的 tmp_path 里，**不往仓里写文件**。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GATE = REPO_ROOT / "tools" / "check_layering.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("check_layering_under_test", GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # 先登记进 sys.modules 再执行：模块里有 dataclass，dataclasses 要按 __module__ 回查命名空间。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    """按真实用法跑脚本（验的是退出码与人话输出，不只是函数返回值）。"""
    return subprocess.run(
        [sys.executable, str(GATE), str(root)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _make_tree(root: Path, files: dict[str, str]) -> Path:
    """在 tmp_path 里造一棵最小的 flagmemory 树。"""
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


CLEAN_TREE = {
    "flagmemory/__init__.py": '"""顶层。"""\n',
    "flagmemory/caps/__init__.py": "from .backend import Thing\n",
    "flagmemory/caps/backend.py": "from dataclasses import dataclass\n\n\n@dataclass\nclass Thing:\n    x: int\n",
    "flagmemory/pool/__init__.py": "from flagmemory.caps import Thing\n\n__all__ = ['Thing']\n",
}


class TestNormal:
    """正常：干净树通过，本仓当前也通过。"""

    def test_本仓干净树通过(self):
        result = _run(REPO_ROOT)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "门禁通过" in result.stdout
        assert "规则 1" in result.stdout and "规则 2" in result.stdout

    def test_最小干净树通过(self, tmp_path):
        root = _make_tree(tmp_path, CLEAN_TREE)
        assert gate.check(root) == []
        assert _run(root).returncode == 0

    def test_池向下依赖能力层是允许的(self, tmp_path):
        root = _make_tree(
            tmp_path,
            {**CLEAN_TREE, "flagmemory/pool/block.py": "from ..caps import Thing\n\nT = Thing\n"},
        )
        assert gate.check(root) == []

    def test_门禁自己不import被检的包(self):
        # 门禁必须是静态扫描：门禁机器上没有 torch，真 import 会把「违规」和「没装」混起来。
        source = GATE.read_text(encoding="utf-8")
        assert "import ast" in source
        assert "import torch" not in source


class TestBoundary:
    """边界：空树、无关文件、tiering 里的框架 import 不该被误判。"""

    def test_空的池与能力层通过(self, tmp_path):
        root = _make_tree(
            tmp_path,
            {"flagmemory/__init__.py": "", "flagmemory/pool/__init__.py": "", "flagmemory/caps/__init__.py": ""},
        )
        assert gate.check(root) == []

    def test_tiering里的框架import不算违规(self, tmp_path):
        # 规则 1 只管 pool/** 与 caps/**；适配层本来就该绑框架，误判它等于把改造做反。
        root = _make_tree(
            tmp_path,
            {
                **CLEAN_TREE,
                "flagmemory/tiering/__init__.py": "",
                "flagmemory/tiering/offload/__init__.py": "import torch\nimport vllm\n",
            },
        )
        assert gate.check(root) == []

    def test_能力层依赖tiering不算规则2违规(self, tmp_path):
        # 规则 2 只钉池本体（pool 不许向上）。caps 真要依赖 tiering 是另一个话题，
        # 这里只确认门禁的管辖范围没写宽。
        root = _make_tree(
            tmp_path,
            {**CLEAN_TREE, "flagmemory/caps/x.py": "from flagmemory import tiering\n"},
        )
        assert [v.rule for v in gate.check(root)] == []

    def test_名字像但不是框架的模块不误伤(self, tmp_path):
        root = _make_tree(
            tmp_path,
            {
                **CLEAN_TREE,
                "flagmemory/pool/x.py": "import torchless\nimport my_torch_helper\nfrom vllmish import a\n",
            },
        )
        assert gate.check(root) == []

    def test_没有flagmemory包时按用法错误退出(self, tmp_path):
        result = _run(tmp_path)
        assert result.returncode == 2
        assert "没找到 flagmemory 包" in result.stdout


class TestError:
    """异常：各种违规写法都要抓到，并以非零码退出。"""

    @pytest.mark.parametrize(
        "body,needle",
        [
            ("import torch\n", "import torch"),
            ("import torch_npu\n", "torch_npu"),
            ("from vllm.config import VllmConfig\n", "vllm"),
            ("import vllm_ascend\n", "vllm_ascend"),
            ("import vllm_fl\n", "vllm_fl"),
            ("import torch.nn as nn\n", "torch.nn"),
        ],
    )
    def test_池本体里的框架import被抓到(self, tmp_path, body, needle):
        root = _make_tree(tmp_path, {**CLEAN_TREE, "flagmemory/pool/bad.py": body})
        violations = gate.check(root)
        assert len(violations) == 1
        assert violations[0].rule == "规则 1"
        assert needle in violations[0].statement
        result = _run(root)
        assert result.returncode == 1
        assert "门禁不通过" in result.stdout
        assert "flagmemory" in result.stdout and "bad.py:1" in result.stdout.replace("\\", "/")

    def test_能力层里的框架import被抓到(self, tmp_path):
        root = _make_tree(tmp_path, {**CLEAN_TREE, "flagmemory/caps/bad.py": "import torch\n"})
        violations = gate.check(root)
        assert len(violations) == 1
        assert violations[0].rule == "规则 1"

    def test_函数内的延迟import被抓到(self, tmp_path):
        body = "def go():\n    import torch\n    return torch\n"
        root = _make_tree(tmp_path, {**CLEAN_TREE, "flagmemory/pool/bad.py": body})
        violations = gate.check(root)
        assert len(violations) == 1
        assert violations[0].lineno == 2

    def test_importlib字面量被抓到(self, tmp_path):
        body = (
            "from importlib import import_module\n"
            "\n"
            "def go():\n"
            '    return import_module("torch")\n'
        )
        root = _make_tree(tmp_path, {**CLEAN_TREE, "flagmemory/pool/bad.py": body})
        violations = gate.check(root)
        assert len(violations) == 1
        assert violations[0].lineno == 4
        assert "import_module" in violations[0].statement

    def test_dunder_import字面量被抓到(self, tmp_path):
        body = 'def go():\n    return __import__("vllm")\n'
        root = _make_tree(tmp_path, {**CLEAN_TREE, "flagmemory/pool/bad.py": body})
        assert len(gate.check(root)) == 1

    def test_池依赖tiering与profiling被抓到(self, tmp_path):
        root = _make_tree(
            tmp_path,
            {
                **CLEAN_TREE,
                "flagmemory/pool/bad.py": (
                    "from flagmemory.tiering.offload.npu import connector\n"
                    "import flagmemory.profiling\n"
                ),
            },
        )
        violations = gate.check(root)
        assert [v.rule for v in violations] == ["规则 2", "规则 2"]
        assert _run(root).returncode == 1

    def test_池用相对import绕上去也被抓到(self, tmp_path):
        root = _make_tree(
            tmp_path,
            {**CLEAN_TREE, "flagmemory/pool/bad.py": "from ..tiering import offload\n"},
        )
        violations = gate.check(root)
        assert len(violations) == 1
        assert violations[0].rule == "规则 2"

    def test_多处违规全部列出且逐条带行号(self, tmp_path):
        body = "import torch\nimport vllm\n\n\ndef go():\n    import torch_npu\n"
        root = _make_tree(tmp_path, {**CLEAN_TREE, "flagmemory/pool/bad.py": body})
        violations = gate.check(root)
        assert sorted(v.lineno for v in violations) == [1, 2, 6]
        result = _run(root)
        assert result.returncode == 1
        assert "发现 3 处违规" in result.stdout

    def test_语法坏掉的文件也拦住(self, tmp_path):
        root = _make_tree(tmp_path, {**CLEAN_TREE, "flagmemory/pool/bad.py": "def (\n"})
        violations = gate.check(root)
        assert len(violations) == 1
        assert violations[0].rule == "语法"
        assert _run(root).returncode == 1
