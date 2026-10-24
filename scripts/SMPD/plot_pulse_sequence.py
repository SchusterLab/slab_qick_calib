"""Draw the pulse sequence a numbered script would actually play.

    python plot_pulse_sequence.py 26
    python plot_pulse_sequence.py 35 26 --out seq.png
    python plot_pulse_sequence.py 35 --set readout_gap=43 --set gain_f0g1=0.74

Runs offline -- no hardware, no nameserver. It reads the script's own
constants and `params` dict, builds the experiment against a stub soc exactly
as `verify_active_reset.py` does, and draws the sequence from the resulting
`cfg.expt`. So the numbers on the figure are the numbers that would be played,
not a hand-copied snapshot -- the previous hand-made diagram went stale within
a day when F0G1_RESET_LENGTH changed from 10 to 1.56 us.

**Layout.** Pulses carry only an index; every name, duration, frequency and
gain goes in a table underneath. Trying to fit that text on or beside the box
is what makes a generic sequence plotter fall over: a 60 ns ramp and a 40 us
buffer window are three orders of magnitude apart, so any label long enough to
be useful either overflows a narrow box or collides with its neighbour. An
index plus a table is legible at any aspect ratio and needs no collision logic.

**Adding a script** means one entry in RECIPES: a function from `cfg.expt` to
a list of Seg. Anything not listed raises with the supported list rather than
drawing a misleading partial sequence.
"""

from __future__ import annotations

import argparse
import ast
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_active_reset import REPO, StubIM, StubSocCfg  # noqa: E402

import slab_qick_calib.experiments as meas  # noqa: E402

ROWS = ["qubit", "f0-g1", "buffer / pump", "readout"]
COLOR = {"qubit": "#C1440E", "f0-g1": "#4C6EF5",
         "buffer / pump": "#2F9E44", "readout": "#7048E8"}
INK, WAIT = "#33322E", "#BFBDB5"


@dataclass
class Seg:
    """One drawn element. `row=None` means a dead interval across all rows."""
    row: str | None
    t0: float
    dur: float
    name: str
    gain: float | None = None
    freq: float | None = None
    note: str = ""


def _ramp(c, which="f0g1"):
    return float(c.get(f"ramp_sigma_{which}", 0.0)) * \
        float(c.get(f"ramp_sigma_inc_{which}", 0))


def _reset_segs(c, t):
    """The inter-shot reset, whichever mode is selected."""
    from slab_qick_calib.experiments.general.qick_program import reset_mode_of
    mode = reset_mode_of(c)
    out = []
    if mode == "passive":
        out.append(Seg(None, t, float(c.final_delay), "final_delay (T1 wait)"))
        return out, t + float(c.final_delay)
    gap = float(c.read_wait) + float(c.extra_delay)
    out.append(Seg(None, t, gap, "ring-down")); t += gap
    if mode == "f0g1":
        dur = float(c.f0g1_reset_length) + float(
            c.get("f0g1_reset_ramp_sigma", 0)) * float(
            c.get("f0g1_reset_ramp_sigma_inc", 0))
        out += [Seg("f0-g1", t, dur, "f0g1_reset_pulse",
                    c.f0g1_reset_gain, c.f0g1_reset_freq),
                Seg("qubit", t, dur, "ef_reset_pulse",
                    c.f0g1_reset_gain_ef, c.f0g1_reset_freq_ef)]
        t += dur
        out.append(Seg(None, t, float(c.f0g1_reset_gap), "photon escape"))
        t += float(c.f0g1_reset_gap)
    else:
        rounds = int(c.reset)
        per = float(c.read_wait) + float(c.extra_delay)
        out.append(Seg(None, t, rounds * per + (rounds - 1) * float(
            c.readout_length), f"active reset, {rounds} rounds"))
        t += rounds * per + (rounds - 1) * float(c.readout_length)
    out.append(Seg(None, t, 1.0, "final_delay")); t += 1.0
    return out, t


def _recipe_fwm(c):
    """21 / 23 / 24 / 26 -- SMPDHistogramExperiment (the 'e' scan)."""
    segs, t = [], 0.0
    segs.append(Seg("buffer / pump", t, float(c.length_b), "buffer_pulse",
                    c.gain_b, c.frequency_b))
    segs.append(Seg("qubit", t, float(c.length_p), "qubit_pulse (FWM pump)",
                    c.gain_p, c.frequency_p))
    t += max(float(c.length_b), float(c.length_p))
    segs.append(Seg("readout", t, float(c.readout_length), "readout",
                    c.get("gain"), c.get("frequency")))
    t += float(c.readout_length)
    more, t = _reset_segs(c, t)
    return segs + more


def _recipe_reset_time(c):
    """35 -- ResetTimeScan, drawn at the longest reset time."""
    segs, t = [], 0.0
    dur = float(c.max_length) + _ramp(c, "f0g1")
    segs += [Seg("f0-g1", t, dur, "f0g1_pulse", c.gain_f0g1, c.frequency_f0g1,
                 "swept 0 -> max"),
             Seg("qubit", t, dur, "ef_pulse", c.gain_ef, c.frequency_ef,
                 "swept with it")]
    t += dur
    segs.append(Seg(None, t, float(c.readout_gap), "readout_gap"))
    t += float(c.readout_gap)
    segs.append(Seg("readout", t, float(c.readout_length), "readout"))
    t += float(c.readout_length)
    more, t = _reset_segs(c, t)
    return segs + more


def _recipe_f0g1_rabi(c):
    """33 / 34 -- F0G1Rabi, drawn at the longest drive."""
    segs, t = [], 0.0
    dur = float(c.max_length) + _ramp(c, "f0g1")
    segs.append(Seg("f0-g1", t, dur, "f0g1_pulse", c.gain_f0g1,
                    c.frequency_f0g1, "swept 0 -> max"))
    t += dur
    segs.append(Seg(None, t, float(c.readout_gap), "readout_gap"))
    t += float(c.readout_gap)
    segs.append(Seg("readout", t, float(c.readout_length), "readout"))
    t += float(c.readout_length)
    more, t = _reset_segs(c, t)
    return segs + more


def _recipe_single_shot(c):
    """12 -- HistogramExperiment, the |e> shot."""
    segs, t = [], 0.0
    segs.append(Seg("qubit", t, 0.06, "pi_ge (prep |e>)"))
    t += 0.06
    segs.append(Seg("readout", t, float(c.readout_length), "readout",
                    c.get("gain"), c.get("frequency")))
    t += float(c.readout_length)
    more, t = _reset_segs(c, t)
    return segs + more


RECIPES = {
    12: ("HistogramExperiment", _recipe_single_shot, {"check_f": True}),
    21: ("SMPDHistogramExperiment", _recipe_fwm, {}),
    23: ("SMPDHistogramExperiment", _recipe_fwm, {}),
    24: ("SMPDHistogramExperiment", _recipe_fwm, {}),
    26: ("SMPDHistogramExperiment", _recipe_fwm, {}),
    33: ("F0G1Rabi", _recipe_f0g1_rabi, {}),
    34: ("F0G1Rabi", _recipe_f0g1_rabi, {}),
    35: ("ResetTimeScan", _recipe_reset_time, {}),
}


_UNSET = object()


def _resolve(node, consts):
    """
    Evaluate a params value: a literal, a constant, or an arithmetic
    expression over them.

    literal_eval alone is not enough -- scripts legitimately write things like
    `"extra_delay": RESET_RINGDOWN - 0.1`, and silently dropping those made
    the diagram show a stale default while the script ran the new value. The
    eval is over the parsed constants only, with builtins removed, so it can
    read the script's own numbers and nothing else.
    """
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.Name):
        return consts.get(node.id, _UNSET)
    try:
        expr = ast.Expression(body=node)
        code = compile(ast.fix_missing_locations(expr), "<params>", "eval")
        return eval(code, {"__builtins__": {}}, dict(consts))
    except Exception:
        return _UNSET


def script_params(num):
    """
    The script's own constants and `params` dict, without executing it.

    Reads the literal assignments inside `if __name__ == "__main__"` and then
    the `params` dict, resolving Name references against them. Executing the
    script instead would try to reach the nameserver.
    """
    matches = sorted(HERE.glob(f"{num}_*.py")) or sorted(HERE.glob(f"{num}*.py"))
    if not matches:
        raise SystemExit(f"no script {num}_*.py in {HERE}")
    path = matches[0]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test).startswith(
                "__name__"):
            body = node.body
    consts, params = {}, {}
    for node in body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        if tgt.id == "params" and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                try:
                    key = ast.literal_eval(k)
                except Exception:
                    continue
                val = _resolve(v, consts)
                if val is not _UNSET:
                    params[key] = val
        else:
            try:
                consts[tgt.id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return path.name, consts, params


def build(num, cfg_path, overrides):
    name, consts, params = script_params(num)
    cls_name, recipe, extra = RECIPES[num]
    params = {**params, **overrides}
    qi = consts.get("qi", 1)
    d = {"soc": StubSocCfg(), "expt_path": ".", "data_path": ".",
         "metadata_path": ".", "cfg_file": str(cfg_path), "im": StubIM()}
    cls = getattr(meas, cls_name)
    expt = cls(d, qi=qi, go=False, params=params, **extra)
    return name, cls_name, recipe(expt.cfg.expt)


def draw(ax, segs, title):
    y_of = {r: len(ROWS) - 1 - i for i, r in enumerate(ROWS)}
    for n, s in enumerate(segs, 1):
        if s.row is None:
            ax.axvspan(s.t0, s.t0 + s.dur, color=WAIT, alpha=0.30, lw=0,
                       zorder=1)
            ax.text(s.t0 + s.dur / 2, -0.85, str(n), ha="center", va="center",
                    fontsize=7.5, color=INK,
                    bbox=dict(boxstyle="circle,pad=0.18", fc="white",
                              ec=WAIT, lw=0.9))
            continue
        y, c = y_of[s.row], COLOR[s.row]
        ax.add_patch(plt.Rectangle((s.t0, y - 0.30), s.dur, 0.60, lw=1.3,
                                   edgecolor=c, facecolor=c, zorder=3))
        ax.text(s.t0 + s.dur / 2, y, str(n), ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=4)
    ax.set_xlim(-max(s.t0 + s.dur for s in segs) * 0.03,
                max(s.t0 + s.dur for s in segs) * 1.03)
    ax.set_ylim(-1.5, len(ROWS) - 0.4)
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS[::-1], fontsize=8, color=INK)
    ax.set_xlabel("time ($\\mu$s)", color=INK, fontsize=9)
    ax.tick_params(colors=INK, length=3, labelsize=8)
    for sd in ("top", "right", "left"):
        ax.spines[sd].set_visible(False)
    ax.spines["bottom"].set_color("#BFBDB5")
    for i in range(len(ROWS)):
        ax.axhline(i, color="#EDEBE6", lw=0.8, zorder=0)
    ax.text(0.0, 1.03, title, transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=INK, va="bottom")


def table_lines(segs):
    out = []
    for n, s in enumerate(segs, 1):
        bits = [f"{s.dur:8.3f} us"]
        if s.gain is not None:
            bits.append(f"gain {float(s.gain):<8.5g}")
        if s.freq is not None:
            bits.append(f"{float(s.freq):10.3f} MHz")
        if s.note:
            bits.append(s.note)
        out.append(f"{n:>2}. {s.name:<24s} " + "  ".join(bits))
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="+", type=int,
                    help="script numbers, e.g. 26 35")
    ap.add_argument("--config",
                    default="2026_09_21_bfg33_smpd_v2_waymar_rfboard.yml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override a params entry (applies to every script)")
    ap.add_argument("--out", default="pulse_sequence.png")
    args = ap.parse_args()

    bad = [n for n in args.scripts if n not in RECIPES]
    if bad:
        raise SystemExit(f"no recipe for script(s) {bad}. Supported: "
                         f"{sorted(RECIPES)}")
    overrides = {}
    for item in args.set:
        k, _, v = item.partition("=")
        try:
            overrides[k] = ast.literal_eval(v)
        except Exception:
            overrides[k] = v

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = REPO / "configs" / args.config

    built = []
    for num in args.scripts:
        name, cls, segs = build(num, cfg_path, overrides)
        built.append((num, name, cls, segs))

    n = len(built)
    heights = [1.0, 0.9] * n
    fig, axes = plt.subplots(
        2 * n, 1, figsize=(13, 4.6 * n),
        gridspec_kw={"height_ratios": heights, "hspace": 0.55})
    axes = np.atleast_1d(axes)
    for i, (num, name, cls, segs) in enumerate(built):
        draw(axes[2 * i], segs, f"script {num} -- {name}  ({cls})")
        tax = axes[2 * i + 1]
        tax.axis("off")
        tax.text(0.0, 1.0, "\n".join(table_lines(segs)), va="top", ha="left",
                 family="monospace", fontsize=8.2, color=INK,
                 transform=tax.transAxes)
        print(f"script {num} ({name}, {cls}):")
        for line in table_lines(segs):
            print("   " + line)
        print()
    fig.savefig(args.out, dpi=165, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
