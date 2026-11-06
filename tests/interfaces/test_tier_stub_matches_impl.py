"""分层存储类型存根与实现必须一致。这条进 CI。全部不依赖真卡。

与 `test_stub_matches_impl.py`（池的 15 条）同一套做法：核对形参名、形参种类、
有无缺省值、有无返回标注；不比对标注的字面文本。存根一旦与实现漂移，CI 变红。

设计文档《分层存储概念与设计基础》第六节 6.1 的 10 条操作必须条条在册：
第 1 条（create_tier_manager）与第 2 条的模块级别名（destroy_tier_manager）是
模块级函数，2..10 的主体是 `TierManager` 的方法。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from flagmemory.tiering.manager import (
    TierManager,
    create_tier_manager,
    destroy_tier_manager,
)

STUB_PATH = (
    Path(__file__).resolve().parent.parent.parent / "flagmemory" / "interfaces" / "tier.pyi"
)

#: 设计文档第六节 6.1 的 10 条操作 + 两个辅助口。
TEN_OPERATIONS = (
    "create_tier_manager",
    "destroy_tier_manager",
    "destroy",
    "register_tier_backend",
    "classify_placement",
    "evict",
    "promote",
    "query_tier_watermark",
    "query_tier_capability",
    "query_tier_audit",
    "set_tier_policy",
    "attach_stream",
)


def _stub_tree() -> ast.Module:
    return ast.parse(STUB_PATH.read_text(encoding="utf-8"), filename=str(STUB_PATH))


def _shape_from_stub(node: ast.FunctionDef) -> tuple[tuple[tuple[str, str, bool], ...], bool]:
    args = node.args
    shape: list[tuple[str, str, bool]] = []
    positional = list(args.posonlyargs) + list(args.args)
    defaults_start = len(positional) - len(args.defaults)
    for i, arg in enumerate(positional):
        kind = "POSITIONAL_ONLY" if arg in args.posonlyargs else "POSITIONAL_OR_KEYWORD"
        shape.append((arg.arg, kind, i >= defaults_start))
    if args.vararg:
        shape.append((args.vararg.arg, "VAR_POSITIONAL", False))
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        shape.append((arg.arg, "KEYWORD_ONLY", default is not None))
    if args.kwarg:
        shape.append((args.kwarg.arg, "VAR_KEYWORD", False))
    return tuple(shape), node.returns is not None


def _shape_from_impl(func) -> tuple[tuple[tuple[str, str, bool], ...], bool]:
    signature = inspect.signature(func)
    shape = tuple(
        (name, param.kind.name, param.default is not inspect.Parameter.empty)
        for name, param in signature.parameters.items()
    )
    return shape, signature.return_annotation is not inspect.Signature.empty


def _stub_functions() -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for node in _stub_tree().body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef) and node.name == "TierManager":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if not any(
                        isinstance(d, ast.Name) and d.id == "property"
                        for d in item.decorator_list
                    ):
                        found[item.name] = item
    return found


def _impl_function(name: str):
    module_functions = {
        "create_tier_manager": create_tier_manager,
        "destroy_tier_manager": destroy_tier_manager,
    }
    return module_functions[name] if name in module_functions else getattr(TierManager, name)


class TestNormal:
    def test_存根文件在该在的地方(self):
        assert STUB_PATH.is_file()

    @pytest.mark.parametrize("operation", TEN_OPERATIONS)
    def test_十条操作都在存根里(self, operation: str):
        assert operation in _stub_functions(), f"存根里缺对外操作 {operation}"

    @pytest.mark.parametrize("operation", TEN_OPERATIONS)
    def test_形参形状与实现一致(self, operation: str):
        stub_shape, stub_has_return = _shape_from_stub(_stub_functions()[operation])
        impl_shape, impl_has_return = _shape_from_impl(_impl_function(operation))
        assert stub_shape == impl_shape, (
            f"{operation} 的形参与存根不一致：\n  存根 {stub_shape}\n  实现 {impl_shape}"
        )
        assert stub_has_return is impl_has_return is True, f"{operation} 缺返回标注"

    def test_存根里的属性与实现对得上(self):
        properties = {
            item.name
            for node in _stub_tree().body
            if isinstance(node, ast.ClassDef) and node.name == "TierManager"
            for item in node.body
            if isinstance(item, ast.FunctionDef)
            and any(isinstance(d, ast.Name) and d.id == "property" for d in item.decorator_list)
        }
        assert properties == {"destroyed"}
        assert isinstance(TierManager.destroyed, property)


class TestBoundary:
    def test_存根里没有实现里不存在的东西(self):
        for name in _stub_functions():
            if name in {"create_tier_manager", "destroy_tier_manager"}:
                continue
            assert hasattr(TierManager, name), f"存根里的 {name} 在实现里不存在"

    def test_仅关键字参数如实标在存根里(self):
        """换入换出的可选参数刻意都是仅关键字 —— 位置传参容易把字节数与条目写反。"""
        for operation in ("evict", "promote", "destroy"):
            shape, _ = _shape_from_stub(_stub_functions()[operation])
            kinds = [kind for name, kind, _ in shape if name != "self"]
            # 允许最前一个位置形参（evict 的 candidates / promote 的 entry_id），
            # 其余必须全是仅关键字
            assert "KEYWORD_ONLY" in kinds, f"{operation} 应有仅关键字参数"
            tail_kinds = set(kinds[kinds.index("KEYWORD_ONLY"):])
            assert tail_kinds == {"KEYWORD_ONLY"}, (
                f"{operation} 的可选参数应全是仅关键字，实际 {kinds}"
            )

    def test_创建的池与后端参数是位置参数(self):
        shape, _ = _shape_from_stub(_stub_functions()["create_tier_manager"])
        assert shape[0] == ("pool", "POSITIONAL_OR_KEYWORD", False)
        assert shape[1] == ("backend", "POSITIONAL_OR_KEYWORD", False)


class TestError:
    def test_存根是合法的python语法(self):
        _stub_tree()  # 解析失败会直接抛 SyntaxError

    def test_存根里没有实现体(self):
        for name, node in _stub_functions().items():
            statements = [s for s in node.body if not isinstance(s, (ast.Expr, ast.Pass))]
            assert not statements, f"存根里的 {name} 带了实现体"
