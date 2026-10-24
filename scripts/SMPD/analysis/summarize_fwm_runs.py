"""
Collect every FWM efficiency run of a fridge run into one summary section.

Answers two questions across runs rather than within one:

  * how NEP depends on the pump time -- the detection window, length_p =
    length_b -- and
  * how the active (conditional pi) and f0-g1 (unconditional) resets compare.

Each complete grid (>= MIN_PUMP_GAINS pump gains x >= MIN_BUFFER_GAINS buffer
gains, from scripts 23 / 26) is reduced by the current fit_shared_means.py --
shared-level fit, Ramsey photon calibration rescaled to each run's own buffer
attenuation and pulse shape -- so every run's NEP is on the same footing,
whatever version of the analysis was in its folder when it was taken. All
output goes under the summary folder; no run folder is written to.

    python summarize_fwm_runs.py                 # run33_Waymar
    python summarize_fwm_runs.py --refit         # redo every per-run fit
    python summarize_fwm_runs.py --run-dir run33_Yuvi --config <yml>

Layout written to Data/<run-dir>/metadata_and_images/FWM_summary/:

    nep_vs_pump_time.png      best NEP, its efficiency and dark count vs T_d
    nep_active_vs_f0g1.png    NEP vs pump gain, one panel per T_d, by reset
    fwm_runs_summary.csv      one row per run
    runs/<run>/               that run's fit_shared_means output, summary/
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import io
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
SMPD_DIR = HERE.parent
REPO_DIR = SMPD_DIR.parent.parent
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from slab_qick_calib.exp_handling.datamanagement import SlabFile  # noqa: E402

RUN_DIR = "run33_Waymar"
CONFIG = "2026_09_21_bfg33_smpd_v2_waymar_rfboard.yml"
FIT_SCRIPT = HERE / "fit_shared_means.py"

# A run is a grid worth reducing if it has enough of both axes for a slope at
# each pump gain and a curve across them; the 1-12 file runs are spot checks.
MIN_PUMP_GAINS = 3
MIN_BUFFER_GAINS = 3

# Reset modes in a fixed categorical order -- colour follows the mode in every
# figure. Slots 1-3 of the dataviz reference palette, each with its own marker
# so identity never rests on colour alone.
RESET_STYLE = {
    "passive": {"color": "#2a78d6", "marker": "o"},
    "active": {"color": "#eb6834", "marker": "s"},
    "f0g1": {"color": "#1baf7a", "marker": "^"},
}

# The readout NEP is quoted for. `e` is the readout the efficiency is defined
# on; f is in the CSV.
SCAN = "e"


def describe_run(data_dir):
    """Settings that are fixed within a run, read from its first file."""
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        return None
    gains_p, gains_b = set(), set()
    for path in files:
        with SlabFile(str(path), "r") as h5:
            expt = h5.load_config()["expt"]
        gains_p.add(round(float(expt["gain_p"]), 6))
        gains_b.add(round(float(expt["gain_b"]), 6))
    with SlabFile(str(files[0]), "r") as h5:
        cfg = h5.load_config()
    expt = cfg["expt"]
    mode = str(expt.get("reset_mode",
                        "active" if expt.get("active_reset") else "passive"))
    return {
        "run": data_dir.name.split("_", 1)[0],
        "folder": data_dir.name,
        "files": len(files),
        "pump_gains": len(gains_p),
        "buffer_gains": len(gains_b),
        "t_d_us": float(expt["length_b"]),
        "length_p_us": float(expt.get("length_p", float("nan"))),
        "reset_mode": mode,
        "pulse_type_p": expt.get("pulse_type_p", "const"),
        "pulse_type_b": expt.get("pulse_type_b"),
        "shots": int(expt.get("shots", 0)),
    }


def load_fit_module():
    spec = importlib.util.spec_from_file_location("fit_shared_means", FIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # It is written to run from inside a run's metadata folder and derives the
    # repo from there; here it runs from analysis/, so pin both locations.
    module.REPO_DIR = REPO_DIR
    return module


def fit_run(module, data_dir, out_dir, config_name):
    """
    Run fit_shared_means on one run, writing only into `out_dir`.

    Its stdout goes to out_dir/fit_log.txt. A SystemExit after summary/ is
    written (figure 3 wants a pump gain the run may not have) is not a
    failure; anything that stops it earlier is, and is logged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    module.HERE = out_dir
    argv = sys.argv
    sys.argv = ["fit_shared_means.py", "--data-dir", str(data_dir),
                "--config", config_name, "--no-show"]
    buffer = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buffer):
            module.main()
    except SystemExit as exit_:
        if exit_.code not in (None, 0):
            error = str(exit_.code)
    except Exception:  # one bad run must not stop the survey
        error = traceback.format_exc()
    finally:
        sys.argv = argv
        plt.close("all")
    (out_dir / "fit_log.txt").write_text(buffer.getvalue()
                                         + (f"\n{error}\n" if error else ""))
    if not (out_dir / "summary" / "fwm_summary.csv").exists():
        return error or "no summary written"
    return None


def read_summary(out_dir):
    """The per-pump-gain rows and the fit flags fit_shared_means wrote."""
    with (out_dir / "summary" / "fwm_summary.csv").open() as handle:
        rows = [{k: float(v) for k, v in row.items()}
                for row in csv.DictReader(handle)]
    text = (out_dir / "summary" / "summary.txt").read_text()
    flags = {
        "e": any("t_ratio " in line and "AT ITS BOUND" in line
                 for line in text.splitlines()),
        "f": any("t_ratio_f" in line and "AT ITS BOUND" in line
                 for line in text.splitlines()),
    }
    duty = ceiling = float("nan")
    for line in text.splitlines():
        if line.startswith("duty cycle"):
            duty = float(line.split("=")[1].split()[0])
        if line.startswith("ceiling"):
            ceiling = float(line.split("=")[2].split()[0])
    return rows, flags, duty, ceiling


def physical(row, ceiling):
    """
    Whether one pump gain's reduction can be believed.

    Three ways a point fails, all seen on run 33:

      * a slope through only two points fits them exactly, so its error bar
        is zero and means nothing;
      * an efficiency error above a quarter of the ceiling is a slope through
        noise -- #115 at pump 0.05 reads 50.6 +/- 27.4, "consistent" with
        anything;
      * an efficiency more than 2 sigma above the run's own ceiling --
        kappa_c/kappa * eta_qubit, which nothing can beat -- means the photon
        calibration does not apply to that run (the 40 dB grids of
        #112-#157 read ~200 +/- 20).

    Choosing the minimum NEP without this cut selects exactly those points,
    since an inflated efficiency deflates the NEP.
    """
    nep = row[f"nep_{SCAN}_zW_rtHz"]
    eta, err = row[f"eta_{SCAN}"], row[f"eta_err_{SCAN}"]
    return bool(np.isfinite(nep) and nep > 0
                and row["slope_points"] >= 3
                and err <= 0.25 * ceiling
                and eta - 2 * err <= ceiling)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=RUN_DIR)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--refit", action="store_true",
                        help="redo per-run fits even where a summary exists")
    args = parser.parse_args()

    run_root = SMPD_DIR / "Data" / args.run_dir
    out_root = run_root / "metadata_and_images" / "FWM_summary"
    runs_dir = out_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        (run_root / "data").glob("#*FWM_efficiency_vs_power*"),
        key=lambda p: int(p.name.split("_", 1)[0][1:]))
    module = load_fit_module()

    results = []
    for data_dir in candidates:
        info = describe_run(data_dir)
        if info is None:
            continue
        if (info["pump_gains"] < MIN_PUMP_GAINS
                or info["buffer_gains"] < MIN_BUFFER_GAINS):
            continue
        out_dir = runs_dir / info["folder"]
        summary_csv = out_dir / "summary" / "fwm_summary.csv"
        # A fit is reused unless it predates the analysis that produced it.
        stale = (not summary_csv.exists()
                 or summary_csv.stat().st_mtime < FIT_SCRIPT.stat().st_mtime)
        if args.refit or stale:
            print(f"fitting {info['folder']} ({info['files']} files)", flush=True)
            error = fit_run(module, data_dir, out_dir, args.config)
            if error:
                print(f"  FAILED: {error.strip().splitlines()[-1]}")
                info["error"] = error.strip().splitlines()[-1]
                results.append(info)
                continue
        rows, flags, duty, ceiling = read_summary(out_dir)
        info.update(rows=rows, flags=flags, duty=duty, ceiling=ceiling)
        results.append(info)

    good = [r for r in results if "rows" in r]
    print(f"\n{len(good)} runs reduced, "
          f"{len(results) - len(good)} failed (see runs/*/fit_log.txt)")

    # --- one row per run ------------------------------------------------
    table = []
    for r in results:
        row = {k: r[k] for k in ("run", "t_d_us", "reset_mode", "pulse_type_p",
                                 "pulse_type_b", "files", "pump_gains",
                                 "buffer_gains", "shots")}
        row["folder"] = r["folder"]
        if "rows" not in r:
            row["error"] = r.get("error", "")
            table.append(row)
            continue
        usable = [x for x in r["rows"] if physical(x, r["ceiling"])]
        row["ceiling"] = r["ceiling"]
        row["pump_gains_used"] = len(usable)
        row["pump_gains_cut"] = len(r["rows"]) - len(usable)
        if usable:
            best = min(usable, key=lambda x: x[f"nep_{SCAN}_zW_rtHz"])
            row.update({
                "best_nep_e_zW_rtHz": best["nep_e_zW_rtHz"],
                "best_nep_f_zW_rtHz": best.get("nep_f_zW_rtHz", float("nan")),
                "at_gain_p": best["gain_p"],
                "eta_e": best["eta_e"], "eta_err_e": best["eta_err_e"],
                "p_dark": best["p_dark"], "n_equiv_e": best["n_equiv_e"],
            })
        row["duty_cycle"] = r["duty"]
        row["max_eta_e"] = float(np.nanmax([x["eta_e"] for x in r["rows"]]))
        row["t_ratio_at_bound"] = r["flags"]["e"]
        row["t_ratio_f_at_bound"] = r["flags"]["f"]
        row["error"] = ""
        table.append(row)
    fields = list(dict.fromkeys(k for row in table for k in row))
    with (out_root / "fwm_runs_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    # --- figure 1: best NEP vs pump time ----------------------------------
    # Hollow = that run's shared fit sat on its decay-ratio bound, so its
    # fractions (and the NEP) are not trustworthy -- shown, not hidden.
    reduced = [row for row in table if np.isfinite(row.get("best_nep_e_zW_rtHz",
                                                           np.nan))]
    modes = [m for m in RESET_STYLE if any(r["reset_mode"] == m for r in reduced)]
    offsets = {m: (i - (len(modes) - 1) / 2) * 0.8 for i, m in enumerate(modes)}
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for mode in modes:
        style = RESET_STYLE[mode]
        group = [r for r in reduced if r["reset_mode"] == mode]
        for flagged in (False, True):
            pts = [r for r in group if bool(r["t_ratio_at_bound"]) == flagged]
            if not pts:
                continue
            x = np.array([r["t_d_us"] for r in pts]) + offsets[mode]
            kwargs = dict(marker=style["marker"], ms=8, ls="none",
                          mec=style["color"], mew=1.5,
                          mfc="none" if flagged else style["color"],
                          label=None if flagged else mode)
            axes1[0].plot(x, [r["best_nep_e_zW_rtHz"] for r in pts], **kwargs)
            axes1[1].errorbar(x, [r["eta_e"] for r in pts],
                              yerr=[r["eta_err_e"] for r in pts],
                              capsize=2, ecolor=style["color"], elinewidth=1,
                              **kwargs)
            axes1[2].plot(x, [100 * r["p_dark"] for r in pts], **kwargs)
        # Best-of-mode envelope: the lowest NEP at each pump time, through
        # trustworthy fits only -- a hollow point is drawn but not joined.
        trusted = [r for r in group if not r["t_ratio_at_bound"]]
        t_values = sorted({r["t_d_us"] for r in trusted})
        best = [min(r["best_nep_e_zW_rtHz"] for r in trusted
                    if r["t_d_us"] == t)
                for t in t_values]
        axes1[0].plot(np.array(t_values) + offsets[mode], best, lw=1.5,
                      color=style["color"], alpha=0.6)
    # Each pump time's ceiling, kappa_c/kappa * eta_qubit(T1, T_d): the most
    # any of its efficiencies could be. It falls with T_d as conversions decay.
    ceilings = sorted({(r["t_d_us"], r["ceiling"]) for r in reduced})
    axes1[1].plot([c[0] for c in ceilings], [c[1] for c in ceilings], "--",
                  color="0.5", lw=1, label="ceiling")
    axes1[0].set_yscale("log")
    # The data span less than a decade, so label plain numbers on the minor
    # ticks too; otherwise only 0.6 and 1 get a label.
    axes1[0].yaxis.set_major_formatter(ScalarFormatter())
    axes1[0].yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))
    axes1[0].set_ylabel(r"best NEP over pump gain (zW/$\sqrt{\mathrm{Hz}}$)")
    axes1[1].set_ylabel("efficiency at that pump gain")
    axes1[1].set_ylim(bottom=0)
    axes1[2].set_ylabel("dark count P(e) at that pump gain (%)")
    axes1[2].set_ylim(bottom=0)
    for axis in axes1:
        axis.set_xlabel(r"pump time $T_d$ = length$_p$ = length$_b$ ($\mu$s)")
        axis.set_xlim(left=0)
        axis.grid(alpha=0.3)
    handles, labels = axes1[0].get_legend_handles_labels()
    if any(r["t_ratio_at_bound"] for r in reduced):
        handles.append(plt.Line2D([], [], marker="o", ls="none", ms=8,
                                  mfc="none", mec="0.4", mew=1.5))
        labels.append("fit at bound")
    axes1[0].legend(handles, labels, fontsize="small", title="reset")
    axes1[1].legend(fontsize="small")
    fig1.savefig(out_root / "nep_vs_pump_time.png", dpi=150)

    # --- figure 2: active vs f0-g1, NEP vs pump gain per pump time --------
    # Only pump times that have both resets, so every panel is a comparison.
    # Every run drawn: replicates of a mode overlap, which is the point. A
    # flat_top pump plays half a const pump's amplitude at the same gain, so
    # those runs are dashed -- their gain axis is not the const runs'.
    shared = sorted(t for t in {r["t_d_us"] for r in good}
                    if {"active", "f0g1"} <= {r["reset_mode"] for r in good
                                              if r["t_d_us"] == t})
    if shared:
        fig2, axes2 = plt.subplots(1, len(shared),
                                   figsize=(4.6 * len(shared), 4.2),
                                   sharey=True, squeeze=False,
                                   constrained_layout=True)
        for axis, t_d in zip(axes2[0], shared):
            for r in good:
                if r["t_d_us"] != t_d or r["reset_mode"] not in RESET_STYLE:
                    continue
                style = RESET_STYLE[r["reset_mode"]]
                # Cut points become gaps, not interpolated across.
                gains = [x["gain_p"] for x in r["rows"]]
                nep = np.array([x[f"nep_{SCAN}_zW_rtHz"]
                                if physical(x, r["ceiling"]) else np.nan
                                for x in r["rows"]])
                if not np.isfinite(nep).any():
                    continue
                axis.plot(gains, nep, marker=style["marker"], ms=6, lw=1.5,
                          color=style["color"], alpha=0.75,
                          mfc="none" if r["flags"]["e"] else style["color"],
                          ls="--" if r["pulse_type_p"] == "flat_top" else "-")
            axis.set_yscale("log")
            axis.set_xlabel("pump gain")
            axis.set_xlim(left=0)
            axis.grid(alpha=0.3)
            axis.text(0.03, 0.97, f"$T_d$ = {t_d:g} $\\mu$s",
                      transform=axis.transAxes, va="top", fontsize=10)
        axes2[0][0].set_ylabel(r"NEP (zW/$\sqrt{\mathrm{Hz}}$)")
        present = [m for m in RESET_STYLE
                   if any(r["reset_mode"] == m and r["t_d_us"] in shared
                          for r in good)]
        handles = [plt.Line2D([], [], color=RESET_STYLE[m]["color"],
                              marker=RESET_STYLE[m]["marker"], lw=1.5)
                   for m in present]
        labels = list(present)
        if any(r["pulse_type_p"] == "flat_top" and r["t_d_us"] in shared
               for r in good):
            handles.append(plt.Line2D([], [], color="0.4", ls="--", lw=1.5))
            labels.append("flat_top pump")
        axes2[0][-1].legend(handles, labels, fontsize="small", title="reset")
        fig2.savefig(out_root / "nep_active_vs_f0g1.png", dpi=150)

    # --- printed digest -------------------------------------------------
    print(f"\nbest NEP ({SCAN} readout) per run, zW/rtHz:")
    print("   run   T_d  reset    pump  NEP     eta    dark %  duty")
    for row in sorted(reduced, key=lambda r: (r["t_d_us"], r["reset_mode"])):
        print(f"  {row['run']:>5} {row['t_d_us']:5g}  {row['reset_mode']:7s}"
              f"  {row['at_gain_p']:4.2f}  {row['best_nep_e_zW_rtHz']:6.2f}"
              f"  {row['eta_e']:5.3f}  {100 * row['p_dark']:6.2f}"
              f"  {row['duty_cycle']:.2f}"
              + ("   fit at bound" if row["t_ratio_at_bound"] else "")
              + (f"   ({row['pump_gains_cut']} pump gains cut)"
                 if row["pump_gains_cut"] else ""))
    unusable = [row for row in table if row.get("error") == ""
                and row not in reduced]
    if unusable:
        print("\nno physical point (every pump gain cut -- see physical()):")
        for row in unusable:
            print(f"  {row['run']:>5} {row['t_d_us']:5g}  {row['reset_mode']}")
    failed = [row for row in table if row.get("error")]
    if failed:
        print("\nfit failed:")
        for row in failed:
            print(f"  {row['run']:>5}  {row['error']}")
    print(f"\nwrote {out_root}")


if __name__ == "__main__":
    main()
