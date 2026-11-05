#!/usr/bin/env python3
"""Check interface resources and the absence of ordinary runtime dependencies.

Requires packaging in the development/build environment only. Non-extra marker
conditions are conservatively treated as potentially true on another host.
"""

from __future__ import annotations

import argparse
from email.parser import Parser
from pathlib import Path
import tarfile
import zipfile

from packaging.markers import Marker
from packaging.requirements import Requirement

STUB = "flagmemory/interfaces/pool.pyi"


def _may_apply_without_extra(markers: list) -> bool:
    """Evaluate marker AND/OR groups with extra='' and other atoms unknown.

    packaging exposes no public marker visitor. Keep its parsed-tree access in
    this helper; CLI regressions cover nesting, operands and membership syntax.
    Treating every non-extra atom as true can reject an impossible condition,
    but cannot let a platform-specific ordinary dependency through.
    """
    groups = [True]
    for item in markers:
        if isinstance(item, list):
            value = _may_apply_without_extra(item)
        elif isinstance(item, tuple):
            left, operator, right = item
            if left.serialize() == "extra" or right.serialize() == "extra":
                atom = " ".join(part.serialize() for part in item)
                value = Marker(atom).evaluate({"extra": ""})
            else:
                value = True
        elif item == "or":
            groups.append(True)
            continue
        elif item == "and":
            continue
        else:
            raise ValueError(f"unsupported marker node: {item!r}")
        groups[-1] = groups[-1] and value
    return any(groups)


def check(wheel: Path, sdist: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if STUB not in names or archive.getinfo(STUB).is_dir():
            raise ValueError(f"wheel missing {STUB}")
        metadata_names = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain exactly one METADATA")
        metadata = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
        for value in metadata.get_all("Requires-Dist", []):
            requirement = Requirement(value)
            if requirement.marker is None or _may_apply_without_extra(
                requirement.marker._markers
            ):
                raise ValueError(f"ordinary runtime dependency: {value}")
    with tarfile.open(sdist) as archive:
        if not any(
            member.isfile() and member.name.endswith("/" + STUB)
            for member in archive.getmembers()
        ):
            raise ValueError(f"sdist missing {STUB}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    args = parser.parse_args()
    try:
        check(args.wheel, args.sdist)
    except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"distribution contract failed: {exc}\n")
    print("distribution contract passed")


if __name__ == "__main__":
    main()
