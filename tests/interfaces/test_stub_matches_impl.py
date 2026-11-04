"""类型存根与实现必须一致。这条进 CI。全部不依赖真卡。

为什么要这条用例：按设计文档第五节的现行方案，类型存根（`flagmemory/interfaces/pool.pyi`）
+ 接口文档共同充当招标口径里的「接口头文件与接口文档」。那一节给出的理由之一是
**可检验性** —— 「类型存根能被类型检查器与单测直接校验，属可持续门禁；手写 C 头与 Python
实现的一致性只能靠人眼」。这条用例就是把那句话变成红绿灯：存根一旦与实现漂移，CI 变红。

核对的是**形参名、形参种类（位置 / 仅关键字 / 可变）、有无缺省值、有无返回标注**，不比对标注
的字面文本 —— 字面比对会因为写法差异（`int | None` 与 `Optional[int]`）误报，而上面那四项
才是消费方真正依赖的契约面。

设计文档第四节那 15 条操作必须条条在册，少一条就红。
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from flagmemory.pool import BlockPool, create_pool

STUB_PATH = (
    Path(__file__).resolve().parent.parent.parent / "flagmemory" / "interfaces" / "pool.pyi"
)

#: 设计文档第四节 4.1 的 15 条对外操作。第 1 条是模块级函数，2..15 是 `BlockPool` 的方法。
FIFTEEN_OPERATIONS = (
    "create_pool",
    "destroy_pool",
    "alloc_on_stream",
    "free_block",
    "register_stream",
    "declare_cross_stream_use",
    "drain_stream",
    "query_watermark",
    "query_fragmentation",
    "compact",
    "reclaim",
    "verify_block_ownership",
    "query_budget",
    "query_capability",
    "query_audit",
)


def _stub_tree() -> ast.Module:
    return ast.parse(STUB_PATH.read_text(encoding="utf-8"), filename=str(STUB_PATH))


def _shape_from_stub(node: ast.FunctionDef) -> tuple[tuple[tuple[str, str, bool], ...], bool]:
    """从存根的函数定义里取出（形参形状, 有无返回标注）。"""
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
    """从实现的签名里取出同样形状的东西。"""
    signature = inspect.signature(func)
    shape = tuple(
        (name, param.kind.name, param.default is not inspect.Parameter.empty)
        for name, param in signature.parameters.items()
    )
    return shape, signature.return_annotation is not inspect.Signature.empty


def _stub_functions() -> dict[str, ast.FunctionDef]:
    """存根里 `BlockPool` 的方法 + 模块级函数，键是操作名。"""
    found: dict[str, ast.FunctionDef] = {}
    for node in _stub_tree().body:
        if isinstance(node, ast.FunctionDef):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef) and node.name == "BlockPool":
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    # 属性（@property）不是操作，单独验
                    if not any(
                        isinstance(d, ast.Name) and d.id == "property" for d in item.decorator_list
                    ):
                        found[item.name] = item
    return found


def _impl_function(name: str):
    return create_pool if name == "create_pool" else getattr(BlockPool, name)


class TestNormal:
    """正常：15 条操作条条在册，且每条的形参形状与实现一致。"""

    def test_存根文件在该在的地方(self):
        assert STUB_PATH.is_file()

    @pytest.mark.parametrize("operation", FIFTEEN_OPERATIONS)
    def test_十五条操作都在存根里(self, operation: str):
        assert operation in _stub_functions(), f"存根里缺了对外操作 {operation}"

    @pytest.mark.parametrize("operation", FIFTEEN_OPERATIONS)
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
            if isinstance(node, ast.ClassDef) and node.name == "BlockPool"
            for item in node.body
            if isinstance(item, ast.FunctionDef)
            and any(isinstance(d, ast.Name) and d.id == "property" for d in item.decorator_list)
        }
        assert properties == {"form", "workload", "tier_table", "native_stream_ordered", "destroyed"}
        for name in properties:
            assert isinstance(getattr(BlockPool, name), property), f"{name} 在实现里不是属性"


class TestBoundary:
    """边界：存根不许多出实现里没有的操作；仅关键字参数要如实标。"""

    def test_存根里没有实现里不存在的东西(self):
        for name in _stub_functions():
            if name == "create_pool":
                continue
            assert hasattr(BlockPool, name), f"存根里的 {name} 在实现里不存在"

    def test_仅关键字参数如实标在存根里(self):
        """这几条操作的参数刻意都是仅关键字 —— 位置传参容易把字节数与超时写反。"""
        for operation in ("drain_stream", "reclaim", "compact", "destroy_pool"):
            shape, _ = _shape_from_stub(_stub_functions()[operation])
            kinds = {kind for name, kind, _ in shape if name != "self"}
            assert kinds == {"KEYWORD_ONLY"}, f"{operation} 的参数应全是仅关键字，实际 {kinds}"

    def test_建池的后端参数是位置参数(self):
        shape, _ = _shape_from_stub(_stub_functions()["create_pool"])
        assert shape[0] == ("backend", "POSITIONAL_OR_KEYWORD", False)


class TestError:
    """异常：存根本身要能被解析；语法坏了立刻红。"""

    def test_存根是合法的python语法(self):
        _stub_tree()   # 解析失败会直接抛 SyntaxError

    def test_存根里没有实现体(self):
        """存根只放声明。出现真实现体说明有人把逻辑写进了 .pyi。"""
        for name, node in _stub_functions().items():
            statements = [s for s in node.body if not isinstance(s, (ast.Expr, ast.Pass))]
            assert not statements, f"存根里的 {name} 带了实现体"
