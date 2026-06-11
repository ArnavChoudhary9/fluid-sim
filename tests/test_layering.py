"""Mechanically enforce the least-privilege layering rule.

No module under ``fluidsim.core`` may import pygame or any of the consumer
layers (render / interaction / app). If someone adds such an import, this test
fails — the architecture rule is checked, not just documented.
"""

from __future__ import annotations

import ast
from pathlib import Path

import fluidsim.core as core_pkg

FORBIDDEN_TOP_LEVEL = {"pygame"}
FORBIDDEN_SUBPACKAGES = {"render", "interaction", "app"}


def _imported_names(source: str) -> list[str]:
    """Return the dotted module names imported by a Python source string."""
    tree = ast.parse(source)
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # Resolve relative imports against the core package.
            prefix = "." * node.level
            names.append(f"{prefix}{node.module or ''}")
    return names


def test_core_does_not_import_ui_or_consumers() -> None:
    core_dir = Path(core_pkg.__file__).parent
    offenders: list[str] = []

    for path in core_dir.glob("*.py"):
        for name in _imported_names(path.read_text(encoding="utf-8")):
            top = name.lstrip(".").split(".")[0]
            if top in FORBIDDEN_TOP_LEVEL:
                offenders.append(f"{path.name}: imports {name!r}")
            # A relative import like '..render' resolves to a sibling subpackage.
            if name.startswith("..") and name.lstrip(".").split(".")[0] in FORBIDDEN_SUBPACKAGES:
                offenders.append(f"{path.name}: imports {name!r}")
            if name.startswith("fluidsim.") and name.split(".")[1] in FORBIDDEN_SUBPACKAGES:
                offenders.append(f"{path.name}: imports {name!r}")

    assert not offenders, "core layering violated:\n" + "\n".join(offenders)
