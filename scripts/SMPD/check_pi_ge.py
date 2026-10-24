"""
Static check: every Program that can measurement-reset declares the pulses
QickProgram.reset()/reset_while() play -- "pi_ge" always, "pi_ef" under reset_f.

QickProgram.reset() plays `self.pulse(ch=self.qubit_ch, name="pi_ge")`, so a
Program whose _initialize only declares pi_ge under some other condition
(checkEF, pulse_e, check_f) will fault the moment active_reset is turned on.
Building the Programs needs real QICK hardware objects, so this walks the
source instead: find each make_cfg_pulse(..., "pi_ge") and confirm the
enclosing `if` selects the active reset, in either spelling (or that there
is no enclosing `if` at all).
"""
import ast
import io
import sys
from pathlib import Path

REPO = Path(r"C:\_Lib\python\slab_qick_calib")

FILES = [
    "experiments/smpd/fwm_shots.py",
    "experiments/smpd/f0g1_reset.py",
    "experiments/single_qubit/qubit_spec_shot.py",
    "experiments/single_qubit/qubit_buffer_spec_shot.py",
    "experiments/single_qubit/FWM_sweeps.py",
    "experiments/single_qubit/stark_spectroscopy.py",
    "experiments/single_qubit/pulse_probe_spectroscopy.py",
    "experiments/single_qubit/resonator_spectroscopy.py",
    "experiments/single_qubit/single_shot.py",
    "experiments/single_qubit/rabi.py",
    "experiments/single_qubit/t1.py",
    "experiments/single_qubit/t2.py",
]


def pi_ge_calls(tree, src, pulse="pi_ge", needs=("active_reset",)):
    """
    Yield (lineno, guard_source) for every make_cfg_pulse(..., pulse).

    `needs` is a tuple of acceptable guard spellings: a declaration counts
    as covered if any of them appears in the test. Two are in play because
    the reset selector is mid-migration -- the un-migrated experiments read
    `cfg.expt.active_reset`, the migrated ones `reset_plays_pi_pulses(...)`.
    An equality test against one mode is not an accepted spelling; see main().
    """
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "make_cfg_pulse"):
            continue
        if not any(isinstance(a, ast.Constant) and a.value == pulse
                   for a in node.args):
            continue
        # Walk up to the nearest enclosing If, stopping at the function.
        enclosing, cur = None, node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.FunctionDef):
                break
            if isinstance(cur, ast.If):
                enclosing = cur
                break
        if enclosing is None:
            yield node.lineno, None
            continue

        # An `elif` is just an If inside the parent's orelse, so climb to the
        # top of the chain and consider every branch together: a declaration
        # in one branch covers active_reset if a sibling branch tests for it
        # and every branch declares pi_ge.
        root = enclosing
        while parents.get(root) is not None and isinstance(parents[root], ast.If)                 and root in parents[root].orelse:
            root = parents[root]

        chain, cur = [], root
        while True:
            chain.append(cur)
            if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                cur = cur.orelse[0]
            else:
                break

        def declares(body):
            return any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "make_cfg_pulse"
                and any(isinstance(a, ast.Constant) and a.value == pulse
                        for a in n.args)
                for stmt in body for n in ast.walk(stmt)
            )

        # ast.unparse, not get_source_segment: a guard split over two lines
        # would otherwise be truncated at the newline and read as uncovered.
        tests = [ast.unparse(c.test) for c in chain]
        covered = any(any(n in t for n in needs) for t in tests)
        all_branches = all(declares(c.body) for c in chain)
        if covered and all_branches and len(chain) > 1:
            yield node.lineno, "chain: " + " | ".join(
                " ".join((t or "").split()) for t in tests
            )
        else:
            yield node.lineno, ast.unparse(enclosing.test)


def check(pulse, needs, label):
    """Every declaration of `pulse` must be reachable whenever `needs` is set."""
    failures = 0
    print(f"--- {label} ---")
    for rel in FILES:
        path = REPO / rel
        src = io.open(path, encoding="utf-8").read()
        tree = ast.parse(src)
        found = list(pi_ge_calls(tree, src, pulse=pulse, needs=needs))
        if not found:
            # No declaration at all is only a problem if the module can set
            # `needs` -- i.e. if it declares the key in a params_def.
            if any(f'"{n}"' in src or f"{n}=" in src for n in needs):
                print(f"  [FAIL] {rel}: sets {needs[0]} but never declares {pulse}")
                failures += 1
            else:
                print(f"  [skip] {rel}: no {pulse}, and never sets {needs[0]}")
            continue
        for lineno, guard in found:
            if guard is None:
                print(f"  [ok]   {rel}:{lineno}: unconditional")
            elif any(n in guard for n in needs):
                print(f"  [ok]   {rel}:{lineno}: if {guard}")
            else:
                print(f"  [FAIL] {rel}:{lineno}: guarded by `{guard}` "
                      f"with none of {needs} -- reset would fault here")
                failures += 1
    return failures


def main():
    # Both spellings of "this sequence active-resets": the un-migrated
    # experiments read cfg.expt.active_reset, the migrated ones go through
    # reset_mode_of. The f0-g1 mode plays neither pi pulse, so it is not
    # listed -- a Program that only declares pi_ge for it would be wrong.
    failures = check(
        "pi_ge",
        # `reset_mode_of(...) == "active"` is deliberately NOT accepted. It
        # reads as a correct guard and is not: it excludes reset_mode="while",
        # which plays pi_ge through reset_while and so faulted script 26 with
        # "trying to play pulse pi_ge, but it hasn't been defined". The guards
        # go through reset_plays_pi_pulses, which covers both modes at once;
        # legacy classes still read cfg.expt.active_reset.
        ("active_reset", "reset_plays_pi_pulses"),
        "pi_ge under a measurement-based reset",
    )
    print()
    failures += check("pi_ef", ("reset_f",), "pi_ef under reset_f")
    print(f"\n{'ALL OK' if failures == 0 else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
