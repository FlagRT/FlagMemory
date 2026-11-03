#!/usr/bin/env python3
"""分层门禁：静态检查「池本体与能力层不碰框架、方向只许向下」。

依据 ADR-0014 §3.5 第 4 件改造（解栈耦合）与 `docs/refactor/01-包结构抬层.md` 第四节
钉下的依赖方向：

    interfaces ← pool ← caps          池本体与能力层都不碰框架
    tiering.offload → pool / caps     适配层可以向下用，反向不行

检两条规则：

  规则 1  flagmemory/pool/** 与 flagmemory/caps/** 里不许 import 推理框架与厂商运行时。
  规则 2  flagmemory/pool/** 不许 import flagmemory.tiering 与 flagmemory.profiling。

**用 ast 静态扫描，不靠 import**：本仓刻意零依赖，门禁机器上没有 torch / vLLM，
真 import 一定失败，分不清「违规」和「环境没装」。静态扫描还能抓到函数内的延迟
import 与 `importlib.import_module("…")` / `__import__("…")` 的字面量写法 —— 那正是
「把框架依赖藏起来」最常见的两种手法。

用法：

    python tools/check_layering.py            # 检当前仓
    python tools/check_layering.py <仓根>      # 检指定目录（单测用它造违规样本）

退出码：0 = 干净；1 = 有违规（逐条打印 文件:行号 + 违规 import）；2 = 用法/路径错误。
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

# —— 禁止出现在池本体与能力层的顶级模块名 ——
# 两类：推理/训练框架，以及各厂商随框架给的运行时扩展。
# 新增芯片时往这里加，不要往代码里加例外。
BANNED_ROOTS: frozenset[str] = frozenset(
    {
        # 推理 / 训练框架
        "torch",
        "torchvision",
        "vllm",
        "vllm_fl",
        "vllm_ascend",
        "vllm_xpu",
        # 厂商运行时扩展
        "torch_npu",
        "torch_mlu",
        "torch_xmlir",
        "acl",
        "acllite",
        "cnrt",
        "cndev",
        "bkcl",
        "hccl",
    }
)

#: 规则 1 管的包（相对仓根的路径前缀）
FRAMEWORK_FREE_PACKAGES: tuple[str, ...] = (
    "flagmemory/pool",
    "flagmemory/caps",
)

#: 规则 2：池本体不许向上/横向依赖的包
POOL_FORBIDDEN_PACKAGES: frozenset[str] = frozenset(
    {
        "flagmemory.tiering",
        "flagmemory.profiling",
    }
)

#: 规则 2 管的包
POOL_PACKAGES: tuple[str, ...] = ("flagmemory/pool",)

_DYNAMIC_IMPORT_CALLS = frozenset({"import_module", "__import__"})


@dataclass(frozen=True)
class Violation:
    path: Path
    lineno: int
    statement: str
    rule: str
    reason: str


def _top_level(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _module_package(repo_root: Path, path: Path) -> list[str]:
    """这个文件所在的包（点分段列表），用来解析相对 import。"""
    rel = path.relative_to(repo_root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts = parts[:-1]
    return parts


def _resolve_relative(package: list[str], level: int, module: str | None) -> str:
    """把 `from ..x import y` 解成绝对点分模块名。"""
    base = package[: len(package) - (level - 1)] if level > 1 else list(package)
    if module:
        base = base + module.split(".")
    return ".".join(base)


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _scan_file(repo_root: Path, path: Path, *, check_framework: bool, check_pool: bool):
    """扫一个文件，产出违规。"""
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:  # 语法坏了也算门禁不过，别放过去
        yield Violation(path, exc.lineno or 0, (exc.text or "").strip(), "语法", f"无法解析：{exc.msg}")
        return

    package = _module_package(repo_root, path)
    # 同一条 import 语句可能被判出两次（模块名本身 + `from x import 子模块` 的子名），
    # 一条语句只报一次，按（行号, 规则）去重。
    seen: set[tuple[int, str]] = set()

    def judge(dotted: str, lineno: int, statement: str):
        if check_framework and _top_level(dotted) in BANNED_ROOTS and (lineno, "规则 1") not in seen:
            seen.add((lineno, "规则 1"))
            yield Violation(
                path,
                lineno,
                statement,
                "规则 1",
                f"池本体与能力层不得依赖推理框架 / 厂商运行时（命中 `{_top_level(dotted)}`）",
            )
        if check_pool:
            for forbidden in sorted(POOL_FORBIDDEN_PACKAGES):
                if (dotted == forbidden or dotted.startswith(forbidden + ".")) and (
                    lineno,
                    "规则 2",
                ) not in seen:
                    seen.add((lineno, "规则 2"))
                    yield Violation(
                        path,
                        lineno,
                        statement,
                        "规则 2",
                        f"依赖方向只许向下，池本体不得依赖 `{forbidden}`",
                    )

    for node in ast.walk(tree):
        # import a.b / import a.b as c
        if isinstance(node, ast.Import):
            for alias in node.names:
                stmt = f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
                yield from judge(alias.name, node.lineno, stmt)

        # from a.b import c / from . import c / from ..x import c
        elif isinstance(node, ast.ImportFrom):
            names = ", ".join(a.name for a in node.names)
            dots = "." * node.level
            stmt = f"from {dots}{node.module or ''} import {names}"
            if node.level:
                dotted = _resolve_relative(package, node.level, node.module)
            else:
                dotted = node.module or ""
            if dotted:
                yield from judge(dotted, node.lineno, stmt)
            # `from flagmemory import tiering` 这种：模块名在 names 里
            for alias in node.names:
                child = f"{dotted}.{alias.name}" if dotted else alias.name
                yield from judge(child, node.lineno, stmt)

        # importlib.import_module("torch") / __import__("torch")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                callee = func.attr
            elif isinstance(func, ast.Name):
                callee = func.id
            else:
                callee = ""
            if callee in _DYNAMIC_IMPORT_CALLS and node.args:
                literal = _string_literal(node.args[0])
                if literal:
                    stmt = f"{callee}({literal!r})"
                    yield from judge(literal, node.lineno, stmt)


def _python_files(repo_root: Path, prefixes: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for prefix in prefixes:
        base = repo_root / Path(prefix)
        if not base.is_dir():
            continue
        found.extend(sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts))
    return found


def check(repo_root: Path) -> list[Violation]:
    framework_files = set(_python_files(repo_root, FRAMEWORK_FREE_PACKAGES))
    pool_files = set(_python_files(repo_root, POOL_PACKAGES))
    violations: list[Violation] = []
    for path in sorted(framework_files | pool_files):
        violations.extend(
            _scan_file(
                repo_root,
                path,
                check_framework=path in framework_files,
                check_pool=path in pool_files,
            )
        )
    return violations


def main(argv: list[str]) -> int:
    # Windows 控制台默认 GBK，中文输出会崩；门禁必须在任何控制台里都能跑。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):  # pragma: no cover - 非交互流
            pass

    if len(argv) > 2:
        print("用法：python tools/check_layering.py [仓根目录]")
        return 2
    repo_root = Path(argv[1]).resolve() if len(argv) == 2 else Path(__file__).resolve().parent.parent
    if not (repo_root / "flagmemory").is_dir():
        print(f"门禁没找到 flagmemory 包：{repo_root}")
        return 2

    scanned = len(set(_python_files(repo_root, FRAMEWORK_FREE_PACKAGES)) | set(_python_files(repo_root, POOL_PACKAGES)))
    violations = check(repo_root)

    print(f"分层门禁 · 仓根 {repo_root}")
    print(f"扫了 {scanned} 个文件（flagmemory/pool/** 与 flagmemory/caps/**，用 ast 静态扫描，不做 import）")

    if not violations:
        print("规则 1 池本体与能力层不依赖推理框架 / 厂商运行时：通过")
        print("规则 2 池本体不依赖 flagmemory.tiering / flagmemory.profiling：通过")
        print("门禁通过，没有发现违规 import。")
        return 0

    print(f"门禁不通过：发现 {len(violations)} 处违规。")
    for v in violations:
        try:
            shown = v.path.relative_to(repo_root)
        except ValueError:  # pragma: no cover
            shown = v.path
        print(f"  {shown}:{v.lineno}  [{v.rule}] {v.statement}")
        print(f"      原因：{v.reason}")
    print("修法：框架绑定一律下沉到 flagmemory/tiering/offload 的可选适配层；")
    print("      池本体只经 flagmemory.caps 的后端协议向下取能力。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
