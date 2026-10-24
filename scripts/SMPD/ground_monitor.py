"""
Take ground-state single-shot readout over and over and watch it drift.

Diagnostic, not a calibration -- it writes nothing to the config. Answers "is the |g> blob
moving, broadening, or changing shape over time", which is what the width and
drift gates in fwm_qc key on.

It acquires with `check_e=False`, so only the ground scan runs and each
iteration costs half what 12_single_shot.py costs. That flag is fine for
*acquisition* but NOT for analysis: HistogramExperiment.analyze() calls
readout_helpers.analyze_single_shot_histograms, which reads data["Ie"]
unconditionally (readout_helpers.py:908) and raises KeyError -- the rotation
angle, threshold and fidelity are all two-state quantities, so there is nothing
for it to compute from |g> alone. This script therefore calls acquire() only
and does its own statistics.

Unnumbered like the other diagnostics here (check_*, plot_*, verify_*), since
it is not a step in the calibration sequence:

    python ground_monitor.py --iterations 60
    python ground_monitor.py --iterations 0        # until Ctrl-C
    python ground_monitor.py --iterations 0 --settle 5

Reset mode and shots are copied from 12_single_shot.py on purpose: the blob
this measures should be the same blob the calibration and the FWM sweeps
discriminate against.
"""

import csv
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SMPD_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SMPD_DIR))

import slab_qick_calib.experiments as meas       # noqa: E402
import fwm_qc                                    # noqa: E402
from common import (                             # noqa: E402
    parse_args,
    prepare_experiment,
    show_all_plots,
    wait_for_plots,
)
# Pumps the GUI once without re-showing the figures. Private, but it is what
# this folder's own shared module uses for exactly this -- plt.pause() calls
# manager.show() on every tick and makes the windows flash, which the plotting
# rules in CLAUDE.md forbid.
from common import _flush_events                  # noqa: E402

CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
QI = 1
JPA_ON = True
SHOTS = 20000
# Match 12_single_shot.py, so this measures the readout the calibration uses.
RESET_MODE = "passive"
RESET_ROUNDS = 3


def add_arguments(parser):
    parser.add_argument("--iterations", type=int, default=60,
                        help="0 = loop until Ctrl-C")
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--reset-mode", default=RESET_MODE,
                        choices=("passive", "active", "f0g1"))
    parser.add_argument("--settle", type=float, default=0.0,
                        help="seconds to wait between iterations")
    parser.add_argument("--no-plots", action="store_true",
                        help="skip the figure entirely; still writes the CSV")


def draw(axes, rows, hists, centers, reference, xlim):
    """
    Redraw every panel from the rows collected so far.

    Cleared and replotted rather than updated through set_data: five panels of
    a few dozen points each is nothing next to an acquisition, and it keeps
    one code path for the live figure and the saved one.
    """
    ax_hist, ax_peak, ax_width, ax_frac, ax_tv = axes
    elapsed = [r["elapsed_s"] for r in rows]
    for axis in axes:
        axis.clear()

    # Every iteration's |g> distribution, coloured oldest to newest. The x
    # limits are pinned to the first iteration on purpose -- an autoscaling
    # axis would follow the blob and hide the very drift this is looking for.
    shades = plt.cm.viridis(np.linspace(0, 0.9, max(len(hists), 1)))
    for hist, shade in zip(hists, shades):
        ax_hist.plot(centers, hist, lw=1.0, color=shade)
    ax_hist.axvline(reference.vg, color="0.4", ls="--", lw=1,
                    label="config g_mean")
    ax_hist.axvline(reference.threshold, color="tab:red", ls=":", lw=1,
                    label="threshold")
    if xlim is not None:
        # None only if Ctrl-C landed between recording a row and the first
        # draw, which is a normal way to stop this script.
        ax_hist.set_xlim(*xlim)
    ax_hist.set_xlabel("I")
    ax_hist.set_ylabel("Probability")
    ax_hist.set_title(f"|g> shots, {len(hists)} iterations", fontsize="small")
    ax_hist.legend(fontsize="x-small", loc="upper right")

    ax_peak.plot(elapsed, [r["peak"] for r in rows], "o-", ms=3)
    ax_peak.axhline(reference.vg, color="0.5", ls="--",
                    label=f"config {reference.vg:.2f}")
    ax_peak.set_ylabel("ground peak (I)")

    ax_width.plot(elapsed, [r["hwhm_sigma"] for r in rows], "o-", ms=3)
    ax_width.axhline(reference.sigma, color="0.5", ls="--",
                     label=f"config {reference.sigma:.2f}")
    ax_width.set_ylabel("ground width (sigma)")

    ax_frac.plot(elapsed, [r["frac_above_threshold"] for r in rows], "o-", ms=3)
    ax_frac.axhline(fwm_qc.GROUND_MAX_PE, color="0.5", ls="--",
                    label=f"GROUND_MAX_PE {fwm_qc.GROUND_MAX_PE}")
    ax_frac.set_ylabel("fraction past threshold")

    ax_tv.plot(elapsed, [r["tv_vs_first"] for r in rows], "o-", ms=3)
    ax_tv.set_ylabel("TV distance vs first")

    for axis in (ax_peak, ax_width, ax_frac, ax_tv):
        axis.set_xlabel("elapsed (s)")
    for axis in (ax_peak, ax_width, ax_frac):
        axis.legend(fontsize="x-small")
    for axis in axes:
        axis.grid(alpha=0.3)


if __name__ == "__main__":
    args = parse_args(__doc__, add_arguments, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, QI, JPA_ON
    )
    out_dir = Path(cfg_dict["metadata_path"])

    # The config's frame, used for the histogram grid and to report how far the
    # measured blob sits from what the YAML currently claims. Deliberately not
    # used as the drift reference -- see below.
    reference = fwm_qc.reference_from_config(auto_cfg, QI, args.shots)
    edges, centers = fwm_qc.bin_grid(reference)
    print(f"Config frame: {reference}")
    theta = np.deg2rad(auto_cfg.device.readout.phase[QI])

    # Built once and acquired from repeatedly. Nothing is ever saved, so the
    # filename it picked at construction is never created.
    single_shot = meas.HistogramExperiment(
        cfg_dict, qi=QI, go=False, check_f=False,
        params={
            "shots": args.shots,
            "reset_mode": args.reset_mode,
            "reset": RESET_ROUNDS,
            "check_e": False,
        },
    )

    columns = ["iteration", "elapsed_s", "median", "peak", "peak_offset_sigma",
               "std", "hwhm_sigma", "width_ratio", "frac_above_threshold",
               "tv_vs_first"]
    rows = []
    hists = []
    first_hist = None
    xlim = None
    fig = axes = None
    plots = not args.no_plots
    if plots:
        if args.no_show:
            # Nobody is watching, but the PNG is still wanted: Agg draws it
            # without ever opening a window.
            plt.switch_backend("Agg")
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.4),
                                 constrained_layout=True)
    start = time.time()
    print(f"\n{'it':>3} {'t/s':>7} {'median':>8} {'peak':>8} "
          f"{'d/sig':>7} {'std':>7} {'width/sig':>10} {'>thr':>8} {'TV':>7}")
    try:
        iteration = 0
        while args.iterations == 0 or iteration < args.iterations:
            iteration += 1
            data = single_shot.acquire(progress=False)
            # Passive-reset HistogramProgram reads out at phase 0, so rotate by
            # the config phase in software, as the 12_single_shot analysis does.
            ig = np.asarray(data["Ig"], dtype=float).flatten()
            qg = np.asarray(data["Qg"], dtype=float).flatten()
            shots = ig * np.cos(theta) - qg * np.sin(theta)
            hist = np.histogram(shots, bins=edges, density=True)[0]
            peak, hwhm_sigma = fwm_qc.ground_peak_and_width(
                centers, hist, reference)
            if first_hist is None:
                first_hist = hist
            row = {
                "iteration": iteration,
                "elapsed_s": round(time.time() - start, 2),
                "median": float(np.median(shots)),
                "peak": peak,
                "peak_offset_sigma": (peak - reference.vg) / reference.sigma,
                "std": float(np.std(shots)),
                "hwhm_sigma": hwhm_sigma,
                "width_ratio": hwhm_sigma / reference.sigma,
                # The "not g" fraction the reset floor and dark count are made
                # of, against the threshold every other experiment uses.
                "frac_above_threshold": float(
                    np.mean(shots > reference.threshold)),
                # Shape change relative to the FIRST iteration, not to the
                # config: the question is whether *this readout* is stable, and
                # a stale g_mean/sigma in the YAML would otherwise show up as a
                # constant offset on every row and hide the drift.
                "tv_vs_first": fwm_qc.total_variation(hist, first_hist, edges),
            }
            rows.append(row)
            hists.append(hist)
            print(f"{row['iteration']:3d} {row['elapsed_s']:7.1f} "
                  f"{row['median']:8.3f} {row['peak']:8.3f} "
                  f"{row['peak_offset_sigma']:+7.2f} {row['std']:7.3f} "
                  f"{row['width_ratio']:10.3f} "
                  f"{row['frac_above_threshold']:8.4f} "
                  f"{row['tv_vs_first']:7.4f}")
            if plots:
                if xlim is None:
                    # Pinned from the first iteration and never changed, so a
                    # drifting blob visibly moves instead of the axis following
                    # it.
                    lo, hi = np.percentile(shots, [0.02, 99.98])
                    pad = 0.25 * (hi - lo)
                    xlim = (lo - pad, hi + pad)
                draw(axes, rows, hists, centers, reference, xlim)
                if not args.no_show:
                    if iteration == 1:
                        # Raises the window once. Later iterations only pump
                        # events, so it does not keep stealing the foreground.
                        show_all_plots(block=False)
                    else:
                        fig.canvas.draw_idle()
                        _flush_events()
            if args.settle:
                time.sleep(args.settle)
    except KeyboardInterrupt:
        print("\ninterrupted")

    if not rows:
        raise SystemExit("no iterations completed")

    csv_path = out_dir / "ground_monitor.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    # draw() already ran inside the loop for every iteration; the figure
    # is only written to disk once, here.
    if plots:
        draw(axes, rows, hists, centers, reference, xlim)
        fig.savefig(out_dir / "ground_monitor.png")

    def spread(key):
        values = np.array([r[key] for r in rows], dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            return "n/a"
        return (f"median {np.median(values):.4f}  "
                f"peak-to-peak {values.ptp():.4f}")

    print(f"\n{len(rows)} iterations over {rows[-1]['elapsed_s']:.0f} s")
    for key in ("peak", "hwhm_sigma", "frac_above_threshold", "tv_vs_first"):
        print(f"  {key:<22} {spread(key)}")
    print(f"\n  config g_mean {reference.vg:.4f}, sigma {reference.sigma:.4f}, "
          f"threshold {reference.threshold:.4f}")
    print(f"  measured width / config sigma: "
          f"{np.median([r['width_ratio'] for r in rows]):.3f}x "
          f"(fwm_qc width gate accepts "
          f"{fwm_qc.WIDTH_LO}-{fwm_qc.WIDTH_HI}x)")
    print(f"\n  {csv_path}")
    if plots:
        print(f"  {out_dir / 'ground_monitor.png'}")
        if not args.no_show:
            wait_for_plots()
