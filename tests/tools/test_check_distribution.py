"""Exercise the distribution gate against real temporary archives."""

from __future__ import annotations

import io
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

GATE = Path(__file__).resolve().parents[2] / "tools" / "check_distribution.py"
STUB = "flagmemory/interfaces/pool.pyi"


def _archives(tmp_path, *, wheel_stub=True, sdist_stub=True, dependencies=()):
    wheel = tmp_path / "flagmemory-0.2.0-py3-none-any.whl"
    metadata = "Metadata-Version: 2.1\nName: flagmemory\nVersion: 0.2.0\n"
    metadata += "".join(f"Requires-Dist: {value}\n" for value in dependencies)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("flagmemory-0.2.0.dist-info/METADATA", metadata)
        if wheel_stub:
            archive.writestr(STUB, "# interface declaration\n")
    sdist = tmp_path / "flagmemory-0.2.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        name = STUB if sdist_stub else "README.md"
        body = b"# resource\n"
        member = tarfile.TarInfo("flagmemory-0.2.0/" + name)
        member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
    return wheel, sdist


def _run(wheel, sdist):
    return subprocess.run(
        [sys.executable, str(GATE), str(wheel), str(sdist)],
        capture_output=True, text=True, encoding="utf-8",
    )


@pytest.mark.parametrize(
    "dependencies",
    [
        (),
        ("pytest>=8; extra == 'dev'", "mypy>=1.8; extra == \"dev\""),
        ("pytest; 'dev' == extra",),
        ("pytest; extra != ''",),
        ("pytest; extra == 'dev' and python_version >= '3.10'",),
        ("pytest; (extra == 'dev' or extra == 'test') and sys_platform == 'linux'",),
    ],
)
def test_accepts_archives_with_only_extra_dependencies(tmp_path, dependencies):
    result = _run(*_archives(tmp_path, dependencies=dependencies))
    assert result.returncode == 0, result.stderr
    assert "distribution contract passed" in result.stdout


@pytest.mark.parametrize(
    "dependency",
    [
        "torch",
        "torch; sys_platform == 'linux'",
        "torch; python_version < '3.10'",
        "torch; extra != 'dev'",
        "torch; extra == 'dev' or sys_platform == 'linux'",
        "torch; extra == ''",
        "torch; extra in 'dev,test'",
        "torch; extra == 'dev' or (python_version < '3.10' and os_name == 'posix')",
    ],
)
def test_rejects_dependency_that_can_apply_without_an_extra(tmp_path, dependency):
    result = _run(*_archives(tmp_path, dependencies=(dependency,)))
    assert result.returncode != 0
    assert "ordinary runtime dependency" in result.stderr


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_rejects_missing_interface_resource(tmp_path, kind):
    artifacts = _archives(
        tmp_path, wheel_stub=kind != "wheel", sdist_stub=kind != "sdist",
    )
    result = _run(*artifacts)
    assert result.returncode != 0
    assert f"{kind} missing {STUB}" in result.stderr
