"""
NEP-optimal readout: single shots over frequency x gain x length.

At each grid point takes g, e and f shots under RESET_MODE and scores the g-f
(shelved) readout by the SMPD NEP at a fixed DETECTION_WINDOW,
sqrt((P(f-side|g) + P_PUMP_EXCESS) * cycle) / (1 - P(g|f)), relative to a
reference point (the config at the start) re-measured every REF_EVERY points.
Points run in random order and each is saved as it is taken (raw shots, one h5
per point). At every reference a ground-state check runs too, as in scripts
26/37/40: if either fails, the run waits for |g> to recover and retakes the
points since the last good check. Each point is also flagged if it sits off its
grid neighbours or was taken next to a drifted reference; --rescan re-acquires
flagged, missing or boxed points of an earlier run into a new one. The optimum
(of the neighbour-median map) is written to the config unless
--no-update-config. CLAUDE.md "NEP-optimal readout (script 12e)".
"""

import collections
import csv
import itertools
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator, NullLocator

import slab_qick_calib.experiments as meas
from slab_qick_calib.calib.readout_helpers import (analyze_single_shot_histograms,
                                                   rotate_coordinates)
from slab_qick_calib.helpers import config

import fwm_qc
from common import (SCRIPT_DIR, effective_final_delay, finish_experiment, parse_args,
                    prepare_experiment, suppress_duplicate_config, wait_for_plots)

CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
QI = 1
JPA_ON = True
RESET_MODE = "while"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
SHOTS = 10000
DETECTION_WINDOW = 35  # us; buffer window the duty cycle is quoted for, as in 12d
# Gain steps are the square root of the length steps, so constant gain**2 * length
# through the config point runs along grid nodes. #613's ~4 h wide grid; the
# narrow scan around its optimum was RATIO 2**(1/12), k and j -6..6, N_F 9,
# SPAN_F 0.75 (~40 min + waits).
RATIO = 2 ** (1 / 6)
LENGTH_STEPS = np.arange(-9, 10)  # length = L_cfg * RATIO**k: x2.8 either side
GAIN_STEPS = np.arange(-12, 13)  # gain = g_cfg * RATIO**(j / 2): x2 either side
N_F = 21  # odd, so the config frequency is a grid point
SPAN_F = 2.0  # total frequency span, in units of kappa
# Pump-only minus no-pump dark count at zero buffer gain, run 34 #519/#520/#522.
P_PUMP_EXCESS = 0.006
REF_EVERY = 50  # grid points between reference + ground checks
Z_FLAG = 4.0  # flag threshold, in units of the point's own NEP error
MAX_FAILURES = 3  # consecutive failed acquisitions before giving up
MAX_RETAKES = 2  # times one point is retaken after a failed check, then kept as is
MIN_GOOD_REFS = 3  # references needed before the reference can fail a check
# Per-point time = overhead + 3 * SHOTS * (length + reset tail); fit to #608.
OVERHEAD_S = 0.33

# Corner-plot order: (length, gain) is the bottom-right pair, x = length.
PARAMS = ("frequency", "readout_length", "gain")
LABELS = {"frequency": "readout frequency (MHz)",
          "readout_length": "readout length (us)", "gain": "readout gain"}
LOG_AXIS = {"frequency": False, "readout_length": True, "gain": True}
METRICS = ["p_f_given_g", "p_g_given_f", "snr_gf", "threshold_gf", "n_shots",
           "p_e_given_g", "p_g_given_e", "cycle_us", "nep", "nep_err", "time"]
POINT_FIELDS = ["i_f", "i_l", "i_g", *PARAMS, *METRICS, "order", "source", "file"]
REF_FIELDS = ["order", *METRICS, "source", "file"]
TEXT_FIELDS = ("source", "file")  # "file" is blank in #613, which saved no per-point h5
GROUND_FIELDS = ["order", "reason", "checks", "failures", "elapsed_s", "p_e_first",
                 "p_e_final", "retaken"]


def add_arguments(parser):
    parser.add_argument(
        "--rescan", metavar="RUN",
        help="re-acquire points of an earlier 12e run (its metadata folder, or a "
             "name like '#612') into a new run folder, merged with the rest of "
             "that run. Uses that run's grid and settings, not the current config.")
    parser.add_argument(
        "--select", nargs="+", default=["missing", "flagged", "unstable"],
        choices=("missing", "flagged", "unstable", "all"),
        help="which points --rescan re-acquires: never acquired, off their "
             "neighbours, taken while the reference had drifted, or every point "
             "(default: the first three).")
    for p, unit in (("frequency", "MHz"), ("length", "us"), ("gain", "DAC")):
        parser.add_argument(f"--{p}", nargs=2, type=float, metavar=("LO", "HI"),
                            help=f"restrict --rescan to this {p} range ({unit})")


def gf_metrics(data):
    """g-f readout figures on the g->f rotated axis, where f sits at +I."""
    params, _ = analyze_single_shot_histograms(data=data, plot=False)
    theta = np.radians(params["angle"])
    ig, _ = rotate_coordinates(np.ravel(data["Ig"]), np.ravel(data["Qg"]), theta)
    i_f, _ = rotate_coordinates(np.ravel(data["If"]), np.ravel(data["Qf"]), theta)
    threshold = params["thresholds"][1]
    sigma_g = 1.4826 * np.median(np.abs(ig - np.median(ig)))
    return {
        "p_f_given_g": float(np.mean(ig > threshold)),
        "p_g_given_f": float(np.mean(i_f < threshold)),
        "snr_gf": float((np.median(i_f) - np.median(ig)) / sigma_g),
        "threshold_gf": float(threshold),
        "n_shots": int(ig.size),
        "p_e_given_g": float(params["err_g"]),
        "p_g_given_e": float(params["err_e"]),
        "angle": float(params["angle"]),
    }


def nep_score(p_dark, p_g_given_f, cycle, n, p_pump_excess):
    """sqrt(dark * cycle) / eta (fixed window, CLAUDE.md 12d) and its binomial error."""
    dark = p_dark + p_pump_excess
    eta = 1 - p_g_given_f
    nep = np.sqrt(dark * cycle) / eta
    rel = np.hypot(np.sqrt(p_dark * (1 - p_dark) / n) / (2 * dark),
                   np.sqrt(p_g_given_f * (1 - p_g_given_f) / n) / eta)
    return nep, nep * rel


def measure(cfg_dict, out_dir, settings, frequency, length, gain, prefix):
    """One g/e/f histogram at (frequency, length, gain), saved with its metrics."""
    expt = meas.HistogramExperiment(
        cfg_dict, qi=QI, go=False, check_f=True, prefix=prefix,
        params={"shots": settings["shots"], "frequency": frequency, "gain": gain,
                "readout_length": length, "reset_mode": settings["reset_mode"]},
    )
    # One config yml per run folder (CLAUDE.md "One config yml per run").
    suppress_duplicate_config(expt, out_dir)
    data = expt.acquire(progress=False)
    metrics = gf_metrics(data)
    # Same period as script 40: window + readout + reset tail.
    cycle = settings["detection_window"] + length + effective_final_delay(expt.cfg.expt)
    nep, nep_err = nep_score(metrics["p_f_given_g"], metrics["p_g_given_f"], cycle,
                             metrics["n_shots"], settings["p_pump_excess"])
    row = {**metrics, "cycle_us": cycle, "nep": nep, "nep_err": nep_err, "time": time.time()}
    data.update(row)
    expt.save_data(data)
    expt.save_config()
    return {**row, "file": Path(expt.fname).name}


def neighbourhood(values, include_self):
    """Each cell's 3x3x3 neighbours stacked on axis 0, NaN past the edges."""
    padded = np.pad(values, 1, constant_values=np.nan)
    return np.array([padded[tuple(slice(1 + d, 1 + d + n) for d, n in zip(shift, values.shape))]
                     for shift in itertools.product((-1, 0, 1), repeat=values.ndim)
                     if include_self or any(shift)])


def neighbour_median(values, include_self):
    """Median over each cell's 3x3x3 neighbourhood, NaN-aware and truncated at the edges."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(neighbourhood(values, include_self), axis=0)


def read_csv(path, fields):
    """Rows of a 12e CSV with numbers parsed; extra columns are ignored."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [{k: (r.get(k) or "" if k in TEXT_FIELDS else float(r[k])) for k in fields}
                for r in csv.DictReader(handle)]


def to_grid(rows, shape):
    """Last row per grid point wins, so an appended rescan replaces what it re-took."""
    points = {}
    for r in rows:
        points[(int(r["i_f"]), int(r["i_l"]), int(r["i_g"]))] = r
    arrays = {k: np.full(shape, np.nan) for k in METRICS}
    sources = np.full(shape, "", dtype=object)
    for idx, r in points.items():
        for k in METRICS:
            arrays[k][idx] = r[k]
        sources[idx] = r["source"]
    return points, arrays, sources


def flag_points(arrays, sources, refs):
    """Neighbour outliers, and points whose closing reference had drifted."""
    nep, err = arrays["nep"], arrays["nep_err"]
    around = neighbourhood(nep, include_self=False)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(around, axis=0)
        # Local spread too, so steep or curved regions are not flagged wholesale.
        spread = 1.4826 * np.nanmedian(np.abs(around - median), axis=0)
        neighbour = np.abs(nep - median) > Z_FLAG * np.hypot(err, spread)
    # A one-sided neighbourhood reads a slope as an outlier: interior cells only.
    edge = np.ones(nep.shape, bool)
    edge[(slice(1, -1),) * nep.ndim] = False
    neighbour &= ~edge
    unstable = np.zeros(nep.shape, bool)
    ref_bad = {}
    for source in {r["source"] for r in refs}:
        rs = sorted((r for r in refs if r["source"] == source), key=lambda r: r["time"])
        t = np.array([r["time"] for r in rs])
        v = np.array([r["nep"] for r in rs])
        bad = np.abs(v - np.median(v)) > Z_FLAG * np.array([r["nep_err"] for r in rs])
        ref_bad.update({(source, r["order"], r["time"]): b for r, b in zip(rs, bad)})
        # Judged by the check that closes the point's block: a failed check is
        # followed by a wait for |g>, so what comes after it starts clean.
        sel = sources == source
        k = np.searchsorted(t, arrays["time"][sel])
        unstable[sel] = np.where(k < len(t), bad[np.clip(k, 0, len(t) - 1)], False)
    return neighbour, unstable, ref_bad


def find_run(name):
    """An earlier 12e metadata folder, by path or by a name like '#612'."""
    path = Path(name)
    if not path.is_dir():
        hits = sorted(SCRIPT_DIR.glob(f"Data/*/metadata_and_images/{name}*"))
        if len(hits) != 1:
            raise SystemExit(f"--rescan {name!r} matches {len(hits)} run folders: {hits}")
        path = hits[0]
    if not (path / "readout_nep_grid.json").exists():
        raise SystemExit(f"{path} has no readout_nep_grid.json, so it is not a 12e run "
                         "this version can rescan.")
    return path


def cell_edges(values, log):
    """pcolormesh cell edges around grid points, geometric on a log axis."""
    v = np.log(values) if log else np.asarray(values, dtype=float)
    mid = (v[1:] + v[:-1]) / 2
    e = np.concatenate([[2 * v[0] - mid[0]], mid, [2 * v[-1] - mid[-1]]])
    return np.exp(e) if log else e


def plain_log_ticks(axis):
    """Plain-number ticks on a log axis that spans about a decade."""
    axis.set_major_locator(LogLocator(subs=(1, 1.5, 2, 3, 5, 7)))
    axis.set_major_formatter(FormatStrFormatter("%g"))
    axis.set_minor_locator(NullLocator())


def set_scales(a, px, py=None):
    for p, scale, axis in ((px, a.set_xscale, a.xaxis), (py, a.set_yscale, a.yaxis)):
        if p is not None and LOG_AXIS[p]:
            scale("log")
            plain_log_ticks(axis)
        elif p is not None:
            axis.set_major_locator(MaxNLocator(5))


def constant_snr_line(ax, lengths, gain_ref, length_ref):
    """gain = g_ref * sqrt(L_ref / L): constant gain**2 * length through the reference."""
    L = np.geomspace(lengths[0], lengths[-1], 50)
    ax.plot(L, gain_ref * np.sqrt(length_ref / L), color="k", lw=2,
            label="gain² × length = reference")


def nep_norm(nep):
    """Diverging scale centred on the reference (1); capped at 2 so a bad corner keeps contrast."""
    return TwoSlopeNorm(vcenter=1, vmin=min(np.nanmin(nep), 0.999),
                        vmax=max(min(np.nanmax(nep), 2.0), 1.001))


def plot_corner(grid, smooth, raw, ref_idx, opt_idx, gain_ref, length_ref):
    """Lower-triangle corner plot; 2D panels are the best over the hidden axis."""
    n = len(PARAMS)
    norm = nep_norm(smooth)
    fig, ax = plt.subplots(n, n, figsize=(3.8 * n, 3.4 * n), constrained_layout=True)
    for i, pi in enumerate(PARAMS):
        for j, pj in enumerate(PARAMS):
            a = ax[i, j]
            if j > i:
                a.axis("off")
                continue
            x = grid[pj]
            if i == j:
                set_scales(a, pj)
                others = tuple(k for k in range(n) if k != i)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    a.plot(x, np.nanmin(smooth, axis=others), "-", label="best over the others")
                cut = tuple(slice(None) if k == i else opt_idx[k] for k in range(n))
                a.plot(x, smooth[cut], "--", label="through the optimum")
                a.plot(x, raw[cut], ".", color="0.4", ms=4, label="raw, through the optimum")
                a.axhline(1, color="0.6", lw=1)
                a.axvline(x[ref_idx[i]], color="0.4", ls=":", lw=1)
                a.set_ylabel("NEP / NEP(reference)")
                if i > 0:
                    a.yaxis.set_label_position("right")
                    a.yaxis.tick_right()
                else:
                    a.legend(fontsize="small")
            else:
                hidden = 3 - i - j
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    best = np.nanmin(smooth, axis=hidden)  # remaining axes are (j, i)
                mesh = a.pcolormesh(cell_edges(x, LOG_AXIS[pj]), cell_edges(grid[pi], LOG_AXIS[pi]),
                                    best.T, cmap="RdBu_r", norm=norm)
                set_scales(a, pj, pi)
                if (pj, pi) == ("readout_length", "gain"):
                    constant_snr_line(a, x, gain_ref, length_ref)
                a.plot(x[ref_idx[j]], grid[pi][ref_idx[i]], "kx", ms=9, mew=2, label="reference")
                a.plot(x[opt_idx[j]], grid[pi][opt_idx[i]], "*", color="gold", mec="k", ms=15,
                       label="NEP optimum")
                if (pj, pi) == ("readout_length", "gain"):
                    a.legend(fontsize="small", loc="lower left")
                if j == 0:
                    a.set_ylabel(LABELS[pi])
            if i == n - 1:
                a.set_xlabel(LABELS[pj])
    cax = ax[0, n - 1].inset_axes([0.05, 0.1, 0.08, 0.8])
    fig.colorbar(mesh, cax=cax, label="NEP / NEP(reference), neighbour median,\nbest over hidden axis")
    return fig


def plot_maps(grid, arrays, nep, ref_idx, opt_idx, gain_ref, length_ref):
    """The raw NEP ingredients over (length, gain) at the optimum frequency."""
    f = opt_idx[0]
    xe = cell_edges(grid["readout_length"], True)
    ye = cell_edges(grid["gain"], True)
    panels = [
        (arrays["snr_gf"][f], "g-f SNR", dict(cmap="viridis")),
        (arrays["p_f_given_g"][f], "P(f-side | g)", dict(cmap="viridis", norm=LogNorm())),
        (arrays["p_g_given_f"][f], "P(g | f)", dict(cmap="viridis", norm=LogNorm())),
        (nep[f], "NEP / NEP(reference), raw", dict(cmap="RdBu_r", norm=nep_norm(nep))),
    ]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for a, (values, label, style) in zip(ax, panels):
        mesh = a.pcolormesh(xe, ye, values.T, **style)
        fig.colorbar(mesh, ax=a, label=label)
        set_scales(a, "readout_length", "gain")
        constant_snr_line(a, grid["readout_length"], gain_ref, length_ref)
        a.plot(grid["readout_length"][ref_idx[1]], grid["gain"][ref_idx[2]], "kx", ms=9, mew=2)
        a.plot(grid["readout_length"][opt_idx[1]], grid["gain"][opt_idx[2]], "*",
               color="gold", mec="k", ms=15)
        a.set_xlabel(LABELS["readout_length"])
    ax[0].set_ylabel(LABELS["gain"])
    ax[0].legend(fontsize="small", loc="lower left")
    ax[0].text(0.02, 0.98, f"{grid['frequency'][f]:.3f} MHz", transform=ax[0].transAxes,
               va="top", fontsize="small")
    return fig


def plot_flags(grid, refs, ref_bad, neighbour, unstable, nep_ref):
    """Reference NEP over time, and where the flagged points sit on the grid."""
    fig, ax = plt.subplots(1, 4, figsize=(19, 4), constrained_layout=True)
    t0 = min(r["time"] for r in refs)
    for source in sorted({r["source"] for r in refs}):
        rs = [r for r in refs if r["source"] == source]
        t = np.array([(r["time"] - t0) / 3600 for r in rs])
        v = np.array([r["nep"] for r in rs]) / nep_ref
        e = np.array([r["nep_err"] for r in rs]) / nep_ref
        bad = np.array([ref_bad[(source, r["order"], r["time"])] for r in rs])
        ax[0].errorbar(t, v, e, fmt="o", ms=3, label=source)
        ax[0].plot(t[bad], v[bad], "o", mfc="none", mec="tab:red", ms=9)
    ax[0].axhline(1, color="0.6", lw=1)
    ax[0].set_xlabel("time since first reference (h)")
    ax[0].set_ylabel("reference NEP / median")
    ax[0].legend(fontsize="x-small")
    for a, (pi, pj) in zip(ax[1:], (("frequency", "readout_length"),
                                    ("frequency", "gain"), ("readout_length", "gain"))):
        hidden = 3 - PARAMS.index(pi) - PARAMS.index(pj)
        axes = (PARAMS.index(pi), PARAMS.index(pj))
        for mask, style, label in ((neighbour & ~unstable, dict(color="tab:orange", marker="o"),
                                    "off its neighbours"),
                                   (unstable, dict(color="tab:red", marker="x"), "reference drifted")):
            idx = np.argwhere(mask)
            a.scatter(grid[pi][idx[:, axes[0]]], grid[pj][idx[:, axes[1]]], s=14, alpha=0.6,
                      label=f"{label} ({len(idx)})", **style)
        set_scales(a, pi, pj)
        a.set_xlim(*cell_edges(grid[pi], LOG_AXIS[pi])[[0, -1]])
        a.set_ylim(*cell_edges(grid[pj], LOG_AXIS[pj])[[0, -1]])
        a.set_xlabel(LABELS[pi])
        a.set_ylabel(LABELS[pj])
        a.grid(alpha=0.3)
        a.text(0.02, 0.98, f"all {PARAMS[hidden].replace('readout_', '')} values overlaid",
               transform=a.transAxes, va="top", fontsize="x-small")
    ax[1].legend(fontsize="small", loc="lower right")
    return fig


if __name__ == "__main__":
    args = parse_args(__doc__, add_arguments, config=CONFIG)
    boxes = {p: getattr(args, a) for p, a in zip(PARAMS, ("frequency", "length", "gain"))
             if getattr(args, a) is not None}
    if boxes and not args.rescan:
        raise SystemExit("--frequency/--length/--gain only restrict a --rescan")
    # Before prepare_experiment, so a bad --rescan creates no run folder.
    previous = find_run(args.rescan) if args.rescan else None
    if args.no_show:
        plt.switch_backend("Agg")
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, QI, JPA_ON
    )
    out_dir = Path(cfg_dict["metadata_path"])
    source = out_dir.name
    readout = auto_cfg.device.readout

    if previous is None:
        f_ref, gain_ref, length_ref = (readout.frequency[QI], readout.gain[QI],
                                       readout.readout_length[QI])
        grid = {
            "frequency": f_ref + SPAN_F * readout.kappa[QI] * np.linspace(-0.5, 0.5, N_F),
            "readout_length": length_ref * RATIO**LENGTH_STEPS,
            "gain": gain_ref * RATIO**(GAIN_STEPS / 2),
        }
        ref_idx = (N_F // 2, int(np.flatnonzero(LENGTH_STEPS == 0)[0]),
                   int(np.flatnonzero(GAIN_STEPS == 0)[0]))
        settings = {"shots": SHOTS, "reset_mode": RESET_MODE,
                    "detection_window": DETECTION_WINDOW, "p_pump_excess": P_PUMP_EXCESS}
        old_points, old_refs = [], []
    else:
        saved = json.loads((previous / "readout_nep_grid.json").read_text())
        grid = {p: np.array(saved["grid"][p]) for p in PARAMS}
        ref_idx = tuple(saved["ref_idx"])
        settings = saved["settings"]
        old_points = read_csv(previous / "readout_nep.csv", POINT_FIELDS)
        old_refs = read_csv(previous / "readout_nep_reference.csv", REF_FIELDS)
        print(f"Rescanning {previous.name}: its grid and settings {settings}")
    shape = tuple(len(grid[p]) for p in PARAMS)
    ref_point = tuple(float(grid[p][k]) for p, k in zip(PARAMS, ref_idx))
    f_ref, length_ref, gain_ref = ref_point
    (out_dir / "readout_nep_grid.json").write_text(json.dumps(
        {"grid": {p: grid[p].tolist() for p in PARAMS}, "ref_idx": list(ref_idx),
         "settings": settings}, indent=1))

    if previous is None:
        todo = list(np.ndindex(shape))
    else:
        points, arrays, sources = to_grid(old_points, shape)
        neighbour, unstable, _ = flag_points(arrays, sources, old_refs)
        todo = [idx for idx in np.ndindex(shape)
                if all(lo <= grid[p][idx[k]] <= hi for k, p in enumerate(PARAMS)
                       if p in boxes for lo, hi in [boxes[p]])
                and ("all" in args.select
                     or ("missing" in args.select and idx not in points)
                     or ("flagged" in args.select and neighbour[idx])
                     or ("unstable" in args.select and unstable[idx]))]
        print(f"  {len(todo)} of {int(np.prod(shape))} points selected by "
              f"{args.select}{' within ' + str(boxes) if boxes else ''}")
        if not todo:
            raise SystemExit("Nothing to rescan.")
    # Random order: drift lands on scattered points, not a block that looks like structure.
    rng = np.random.default_rng()
    todo = [todo[i] for i in rng.permutation(len(todo))]

    probe = meas.HistogramExperiment(cfg_dict, qi=QI, go=False, check_f=True,
                                     params={"reset_mode": settings["reset_mode"]})
    # The "active"/"while" tail carries readout_length once per round, and this
    # sweeps length over 8x, so it has to be recomputed per length instead of
    # taken once at the config value -- otherwise the estimate is ~3x low at the
    # long end. For "f0g1"/"passive" the tail does not depend on length at all.
    def tail_for(length):
        probe.cfg.expt.readout_length = length
        return effective_final_delay(probe.cfg.expt)

    point_s = lambda L: OVERHEAD_S + 3 * settings["shots"] * (L + tail_for(L)) * 1e-6
    # Each check is a reference point plus one ground-only look at the config readout.
    check_s = (point_s(length_ref) + OVERHEAD_S
               + fwm_qc.GROUND_SHOTS
               * (readout.readout_length[QI] + tail_for(readout.readout_length[QI])) * 1e-6)
    n_checks = len(todo) // REF_EVERY + 2
    est_s = sum(point_s(grid["readout_length"][idx[1]]) for idx in todo) + n_checks * check_s
    print(f"{len(todo)} points over {shape[0]} frequencies "
          f"({grid['frequency'][0]:.3f}-{grid['frequency'][-1]:.3f} MHz) x {shape[1]} lengths "
          f"({grid['readout_length'][0]:.2f}-{grid['readout_length'][-1]:.2f} us) x {shape[2]} gains "
          f"({grid['gain'][0]:.4f}-{grid['gain'][-1]:.4f}); reference {ref_point}")
    # ~0.95 MB of raw shots per point at 20k shots (#608: 300 MB for 316 files).
    print(f"  ~{est_s / 3600:.2f} h before ground waits and retakes; "
          f"~{(len(todo) + n_checks) * 0.95e-3 * settings['shots'] / 20000:.1f} GB of raw shots")

    # Earlier points first, then this session's appended as they come: a crash leaves a
    # complete, rescannable folder, and the last row per point wins on reading.
    points_path = out_dir / "readout_nep.csv"
    refs_path = out_dir / "readout_nep_reference.csv"
    files = []
    for path, fields, old in ((points_path, POINT_FIELDS, old_points),
                              (refs_path, REF_FIELDS, old_refs),
                              (out_dir / "ground_waits.csv", GROUND_FIELDS, [])):
        handle = path.open("w", newline="")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(old)
        handle.flush()
        files.append((handle, writer))
    (points_file, points_writer), (refs_file, refs_writer), (ground_file, ground_writer) = files

    # Ground-state check as in scripts 26/37/40 (CLAUDE.md "Ground-state recovery").
    ground_check = fwm_qc.make_ground_check(cfg_dict, QI, {"reset_mode": settings["reset_mode"]})
    ground_ref = fwm_qc.reference_from_config(auto_cfg, QI, fwm_qc.GROUND_SHOTS)
    good_refs = []

    def checkpoint(order):
        """Reference point + one ground look; returns the reasons it failed, if any."""
        row = measure(cfg_dict, out_dir, settings, *ref_point,
                      prefix=f"readout_nep_reference_qubit{QI}")
        refs_writer.writerow({**row, "order": order, "source": source})
        refs_file.flush()
        p_e = fwm_qc.ground_excited_fraction(ground_check, ground_ref)
        reasons = []
        if p_e >= fwm_qc.GROUND_MAX_PE:
            reasons.append(f"ground {p_e:.3f} >= {fwm_qc.GROUND_MAX_PE}")
        if len(good_refs) >= MIN_GOOD_REFS:
            z = (row["nep"] - np.median(good_refs)) / row["nep_err"]
            if abs(z) > Z_FLAG:
                reasons.append(f"reference {z:+.1f} sigma")
        print(f"  check: reference NEP {row['nep']:.4f} +/- {row['nep_err']:.4f}, "
              f"ground {p_e:.4f}" + (f"  !! {'; '.join(reasons)}" if reasons else ""))
        if not reasons:
            good_refs.append(row["nep"])
        return "; ".join(reasons)

    def log_wait(wait, order, reason, retaken):
        ground_writer.writerow({**wait, "order": order, "reason": reason, "retaken": retaken})
        ground_file.flush()

    queue = collections.deque(todo)
    retakes = collections.Counter()
    block = []  # points since the last good check, retaken if the next one fails
    order = 0
    start = time.time()
    failures = 0
    try:
        # One pass to start, as in script 40; a mid-run failure needs the full streak.
        wait = fwm_qc.wait_for_ground(ground_check, ground_ref, consecutive=1)
        if wait["failures"]:
            log_wait(wait, 0, "before start", 0)
        reason = checkpoint(0)
        if reason:
            log_wait(fwm_qc.wait_for_ground(ground_check, ground_ref), 0, reason, 0)
        while queue:
            idx = queue.popleft()
            frequency, length, gain = (float(grid[p][k]) for p, k in zip(PARAMS, idx))
            try:
                row = measure(cfg_dict, out_dir, settings, frequency, length, gain,
                              prefix=f"readout_nep_qubit{QI}")
            except Exception as error:  # board/Pyro hiccup: the point stays missing for --select missing
                failures += 1
                print(f"  !! {idx} failed ({failures} in a row): {error!r}")
                if failures >= MAX_FAILURES:
                    raise
                continue
            failures = 0
            order += 1
            block.append(idx)
            points_writer.writerow({**row, "i_f": idx[0], "i_l": idx[1], "i_g": idx[2],
                                    "frequency": frequency, "readout_length": length,
                                    "gain": gain, "order": order, "source": source})
            points_file.flush()
            elapsed = time.time() - start
            print(f"[{order} done, {len(queue)} left, {elapsed / 3600:.2f} h, ETA "
                  f"{elapsed / order * len(queue) / 3600:.2f} h] {frequency:.4f} MHz "
                  f"{length:5.2f} us {gain:.4f}  SNR {row['snr_gf']:5.2f}  "
                  f"P(f|g) {row['p_f_given_g']:.4f}  P(g|f) {row['p_g_given_f']:.4f}")
            if len(block) >= REF_EVERY or not queue:
                reason = checkpoint(order)
                if reason:
                    wait = fwm_qc.wait_for_ground(ground_check, ground_ref)
                    redo = [i for i in block if retakes[i] < MAX_RETAKES]
                    retakes.update(redo)
                    queue.extendleft(reversed(redo))
                    log_wait(wait, order, reason, len(redo))
                    print(f"  retaking the {len(redo)} points since the last good check"
                          + (f" ({len(block) - len(redo)} kept, already retaken "
                             f"{MAX_RETAKES}x)" if len(redo) < len(block) else ""))
                block = []
    finally:
        points_file.close()
        refs_file.close()
        ground_file.close()

    # Reduce the merged run: last row per point, flags, neighbour-median map.
    all_rows = read_csv(points_path, POINT_FIELDS)
    refs = read_csv(refs_path, REF_FIELDS)
    points, arrays, sources = to_grid(all_rows, shape)
    neighbour, unstable, ref_bad = flag_points(arrays, sources, refs)
    # Normalize by the references that passed, not all of them (#613 read ~3% low).
    good = [r["nep"] for r in refs if not ref_bad[(r["source"], r["order"], r["time"])]]
    nep_ref = float(np.median(good or [r["nep"] for r in refs]))
    nep = arrays["nep"] / nep_ref
    smooth = neighbour_median(np.where(unstable, np.nan, nep), include_self=True)
    opt_idx = np.unravel_index(np.nanargmin(smooth), shape)

    replaced = [r for r in all_rows
                if points[(int(r["i_f"]), int(r["i_l"]), int(r["i_g"]))] is not r]
    columns = POINT_FIELDS + ["nep_rel", "nep_smooth_rel", "flag"]
    with points_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for idx, r in sorted(points.items()):
            flag = "+".join(name for name, m in (("neighbour", neighbour), ("unstable", unstable))
                            if m[idx])
            writer.writerow({**r, "nep_rel": nep[idx], "nep_smooth_rel": smooth[idx], "flag": flag})
    if replaced:
        with (out_dir / "readout_nep_replaced.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=POINT_FIELDS)
            writer.writeheader()
            writer.writerows(replaced)

    n_missing = int(np.prod(shape)) - len(points)
    print(f"\n{len(points)} points ({n_missing} missing); flagged: {int(neighbour.sum())} off "
          f"their neighbours, {int(unstable.sum())} next to a drifted reference "
          f"({sum(ref_bad.values())} of {len(refs)} references)")
    print(f"  {sum(retakes.values())} retakes after failed checks this session; waits in "
          f"{out_dir / 'ground_waits.csv'}")
    for src in sorted({r["source"] for r in refs}):
        v = [r["nep"] for r in refs if r["source"] == src]
        print(f"  reference NEP median {np.median(v):.4f} in {src} ({len(v)} measurements)")
    print(f"NEP relative to the reference {ref_point} ({settings['detection_window']} us window); "
          "best five of the neighbour-median map:")
    for k in np.argsort(np.nan_to_num(smooth, nan=np.inf), axis=None)[:5]:
        idx = np.unravel_index(k, shape)
        print(f"  {grid['frequency'][idx[0]]:10.4f} MHz  {grid['readout_length'][idx[1]]:6.2f} us  "
              f"gain {grid['gain'][idx[2]]:.4f}  median {smooth[idx]:.3f}  raw {nep[idx]:.3f} "
              f"+/- {arrays['nep_err'][idx] / nep_ref:.3f}")

    fig = plot_corner(grid, smooth, nep, ref_idx, opt_idx, gain_ref, length_ref)
    fig.savefig(out_dir / "readout_nep_corner.png", dpi=120)
    fig = plot_maps(grid, arrays, nep, ref_idx, opt_idx, gain_ref, length_ref)
    fig.savefig(out_dir / "readout_nep_maps.png", dpi=120)
    fig = plot_flags(grid, refs, ref_bad, neighbour, unstable, nep_ref)
    fig.savefig(out_dir / "readout_nep_flags.png", dpi=120)
    print(f"  {points_path}")
    if neighbour.any() or unstable.any() or n_missing:
        print(f"  to retake flagged/missing points: python 12e_readout_nep_optimization.py "
              f"--rescan \"{out_dir.name.split('_')[0]}\"")

    if not args.no_update_config:
        cfg_file = cfg_dict["cfg_file"]
        for p, k in zip(PARAMS, opt_idx):
            config.update_readout(cfg_file, p, float(grid[p][k]), QI)
        # Re-run at the optimum so phase, threshold and sigma match it, as script 22 does.
        shot = meas.HistogramExperiment(cfg_dict, qi=QI, go=False, check_f=True,
                                        params={"shots": settings["shots"],
                                                "reset_mode": settings["reset_mode"]})
        finish_experiment(shot, args, wait=False)
        shot.update()
    if not args.no_show:
        wait_for_plots()
