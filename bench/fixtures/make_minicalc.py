from __future__ import annotations

from pathlib import Path

OPS = 20
FNS = 8


def _op_source(m: int) -> str:
    lines = [f'"""minicalc.op_{m:02d}: arithmetic kernels (generated)."""', ""]
    for k in range(FNS):
        lines += [
            f"def f{k}(x: int) -> int:",
            f'    """Return x * {m + 1} + {k} (op {m}, fn {k})."""',
            "    if not isinstance(x, int):",
            '        raise TypeError("x must be int")',
            f"    y{k} = x * {m + 1}",
            f"    y{k} = y{k} + {k}",
            f"    return y{k}",
            "",
            "",
        ]
    return "\n".join(lines)


def _registry_source() -> str:
    lines = ['"""minicalc: registry of arithmetic kernels."""']
    for m in range(OPS):
        lines.append(f"from . import op_{m:02d}")
    lines.append("")
    lines.append("REGISTRY = {")
    for m in range(OPS):
        lines.append(f'    "op_{m:02d}": op_{m:02d},')
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _test_source(m: int) -> str:
    lines = [f"from minicalc import op_{m:02d}", ""]
    for k in range(FNS):
        lines.append(f"def test_f{k}():")
        lines.append(f"    assert op_{m:02d}.f{k}(3) == {3 * (m + 1) + k}")
        lines.append("")
    return "\n".join(lines)


def write_minicalc(dest: Path) -> Path:
    """Write the deterministic minicalc fixture into `dest` and return it."""
    dest = Path(dest)
    pkg = dest / "minicalc"
    tests = dest / "tests"
    pkg.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)

    (dest / "conftest.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text(
        '"""minicalc: generated arithmetic package."""\n'
        "from .registry import REGISTRY\n"
        '__all__ = ["REGISTRY"]\n',
        encoding="utf-8",
    )
    (pkg / "registry.py").write_text(_registry_source(), encoding="utf-8")
    for m in range(OPS):
        (pkg / f"op_{m:02d}.py").write_text(_op_source(m), encoding="utf-8")
        (tests / f"test_op_{m:02d}.py").write_text(_test_source(m), encoding="utf-8")
    return dest
