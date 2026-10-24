"""
Static check: every params_def that can reach QickProgram.reset() declares
every key reset() reads off cfg.expt.

reset() uses direct attribute access, per the root CLAUDE.md rule, so a
missing key is an AttributeError at compile time rather than a default. That
only fires once the program is built, i.e. partway into a run on hardware --
which is how `reset_latch` reached f0g1_reset.py's four params_def blocks and
crashed 32_ef_stark_shift_vs_f0g1_power.py after the sweep had started.

Walks the source rather than building anything, so it needs no hardware:

    python check_reset_keys.py
"""

import ast
import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Read off QickProgram.reset()/measure() -- the keys those bodies touch.
RESET_KEYS = ("reset", "reset_f", "reset_verify", "reset_latch",
              "reset_while_typical", "setup_reset",
              "threshold", "read_wait", "extra_delay")
# configure_reset() supplies these at runtime, so params_def need not carry them.
DERIVED = ("threshold", "read_wait", "extra_delay")


def dict_keys_near(tree, src):
    """Every string key of every dict literal in the file, with its line."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = {k.value for k in node.keys
                if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        if keys:
            out.append((node.lineno, keys))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "dict":
            keys = {kw.arg for kw in node.keywords if kw.arg}
            if keys:
                out.append((node.lineno, keys))
    return out


def main():
    required = [k for k in RESET_KEYS if k not in DERIVED]
    failures = 0
    for path in sorted((REPO / "experiments").rglob("*.py")):
        src = io.open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        rel = path.relative_to(REPO).as_posix()
        for lineno, keys in dict_keys_near(tree, src):
            # a params_def is any dict that already declares some reset key
            touched = keys & set(required)
            if not touched or touched == {"reset"}:
                continue
            missing = [k for k in required if k not in keys]
            if missing:
                print(f"  [FAIL] {rel}:{lineno}: declares {sorted(touched)} "
                      f"but is missing {missing}")
                failures += 1
            else:
                print(f"  [ok]   {rel}:{lineno}")
    print(f"\n{'ALL OK' if failures == 0 else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
