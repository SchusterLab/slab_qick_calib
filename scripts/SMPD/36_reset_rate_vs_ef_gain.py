"""Reset rate versus e-f drive gain: find the Fig. 3a optimum by measuring it.

See scripts/SMPD/CLAUDE.md, "Runner notes, scripts 27–37".

    python 36_reset_rate_vs_ef_gain.py --no-show
    python 36_reset_rate_vs_ef_gain.py --reanalyze Data/run34_Waymar/data/#293_...
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config as cfglib
from slab_qick_calib.helpers.f0g1 import fitted_reset_rate, g_tilde_for_gain

from common import (
    REPO_DIR,
    finish_experiment,
    parse_args,
    prepare_experiment,
    wait_for_plots,
)


def load_run(data_dir):
    """(gains, gammas, t, gain_f0g1) from a saved run's h5 files."""
    files = sorted(Path(data_dir).glob("*.h5"))
    if not files:
        raise SystemExit(f"no h5 files under {data_dir}")
    gains, gammas, t, gain_f0g1 = [], [], None, None
    for f in files:
        with h5py.File(f, "r") as h:
            gammas.append(float(np.array(h["reset_gamma"]))
                          if "reset_gamma" in h else np.nan)
            expt = yaml.safe_load(h.attrs["config"])["expt"]
            gains.append(float(expt["gain_ef"]))
            gain_f0g1 = float(expt["gain_f0g1"])
            if t is None:
                t = np.asarray(h["xpts"], dtype=float).ravel()
    order = np.argsort(gains)
    return np.array(gains)[order], np.array(gammas)[order], t, gain_f0g1


def _measured_peak(gains, gammas):
    """Peak gain from a parabola through the top three points, in log gain."""
    i = int(np.argmax(gammas))
    if len(gains) < 3:
        return float(gains[i])
    j = int(np.clip(i, 1, len(gains) - 2))
    a, b, _ = np.polyfit(np.log(gains[j - 1:j + 2]), gammas[j - 1:j + 2], 2)
    return float(np.exp(-b / (2 * a))) if a < 0 else float(gains[i])


def analyze(gains, gammas, t, g_tilde, kappa, om_m, om_b):
    """Fit one Omega_ef scale, locate the peak, print the table."""
    ok = np.isfinite(gammas) & (gammas > 0)

    # Predict what an exponential fit of this run's trace reports, not the
    # asymptotic eigenvalue -- they differ by up to 3x. CLAUDE.md "33/34".
    def predict(gain_ef, scale):
        return np.array([fitted_reset_rate(g_tilde, scale * (om_m * g + om_b),
                                           kappa, t)
                         for g in np.atleast_1d(gain_ef)])

    # One Omega_ef scale over the whole curve: collapses onto Eq. 1 if only
    # omega_ef_fit is off. Residual in log, since Gamma spans ~7x here and a
    # linear one would be fit almost entirely by the peak.
    scales = np.linspace(0.1, 2.0, 400)
    resid = [np.nansum((np.log(predict(gains[ok], s))
                        - np.log(gammas[ok])) ** 2) for s in scales]
    best = float(scales[int(np.argmin(resid))])

    print("")
    print("=" * 66)
    print(f"  {'gain_ef':>9}{'nominal Om':>12}{'Gamma meas':>12}"
          f"{'Eq.1 x1':>10}{'Eq.1 xfit':>11}")
    for g, gm in zip(gains, gammas):
        print(f"  {g:>9.4f}{om_m * g + om_b:>12.4f}{gm:>12.4f}"
              f"{predict(g, 1.0)[0]:>10.4f}{predict(g, best)[0]:>11.4f}")
    if ok.sum() < 3:
        return {"scale": best, "peak_measured": float("nan")}

    scan = np.linspace(min(gains) * 0.4, max(gains) * 1.3, 500)
    peak_cal = float(scan[int(np.nanargmax(predict(scan, 1.0)))])
    peak_fit = float(scan[int(np.nanargmax(predict(scan, best)))])
    peak_meas = _measured_peak(gains[ok], gammas[ok])
    i_best = int(np.nanargmax(np.where(ok, gammas, np.nan)))
    rms = float(np.sqrt(np.nanmean((np.log(predict(gains[ok], best))
                                    - np.log(gammas[ok])) ** 2)))

    print("")
    print(f"  best single scale on Omega_ef: {best:.3f} "
          f"(rms {rms * 100:.0f}% over {int(ok.sum())} points)")
    print(f"    i.e. the drive acts at {best:.2f}x its calibrated rate,")
    print(f"    so omega_ef_fit should read "
          f"[{om_m * best:.4f}, {om_b * best:.4f}]")
    print("")
    # The peak position alone calibrates Omega_ef: at the optimum the drive is
    # at Omega_ef_opt(g~, kappa) by definition, so it needs neither the
    # absolute Gamma nor kappa to be right. CLAUDE.md "Runner notes".
    print(f"  peak gain_ef:  Eq. 1 as calibrated {peak_cal:.4f}"
          f"   Eq. 1 at x{best:.2f} {peak_fit:.4f}")
    print(f"                 measured {peak_meas:.4f} "
          f"(best sampled point {gains[i_best]:.4f}, "
          f"Gamma {gammas[i_best]:.4f})")
    print(f"    scale implied by the peak alone: "
          f"{(om_m * peak_cal + om_b) / (om_m * peak_meas + om_b):.3f}")
    print("")
    print(f"  >>> OPTIMAL e-f GAIN = {peak_meas:.4f}  "
          f"(Omega_ef {best * (om_m * peak_meas + om_b):.4f} MHz "
          f"at g~ = {g_tilde:.4f})")
    if i_best in (0, len(gains) - 1):
        print("  NOTE: the best point is at the edge of the scan, so the "
              "optimum is outside it. Widen GAIN_EF_LIST and rerun.")
    return {"scale": best, "peak_measured": peak_meas, "peak_cal": peak_cal,
            "peak_fit": peak_fit, "rms": rms}


def draw(gains, gammas, t, g_tilde, kappa, om_m, om_b, res):
    """Gamma vs e-f gain, against Eq. 1 as calibrated and rescaled."""
    ok = np.isfinite(gammas) & (gammas > 0)
    best = res["scale"]

    def predict(gain_ef, scale):
        return np.array([fitted_reset_rate(g_tilde, scale * (om_m * g + om_b),
                                           kappa, t)
                         for g in np.atleast_1d(gain_ef)])

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    fine = np.linspace(min(gains) * 0.6, max(gains) * 1.15, 300)
    ax.plot(fine, predict(fine, 1.0), "--", color="0.55", lw=1.6,
            label="Eq. 1 with omega_ef_fit as calibrated")
    ax.plot(fine, predict(fine, best), "-", color="#C1440E", lw=2,
            label=f"Eq. 1 with $\\Omega_{{ef}}$ x {best:.2f}")
    ax.plot(gains[ok], gammas[ok], "o", ms=8, color="#4C6EF5", mec="white",
            mew=1.4, zorder=5, label="measured")
    pk = res.get("peak_measured", float("nan"))
    if np.isfinite(pk):
        ax.axvline(pk, ls="-.", color="#2F9E44", lw=1.4,
                   label=f"measured peak, gain_ef = {pk:.4f}")
    ax.axhline(kappa / 3, ls=":", color="#33322E", lw=1.4,
               label=f"$\\kappa/3$ = {kappa / 3:.3f} MHz (asymptotic only)")
    ax.set_xlabel("e-f drive gain (DAC level)")
    ax.set_ylabel("reset rate $\\Gamma/2\\pi$ (MHz)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    for sd in ("top", "right"):
        ax.spines[sd].set_visible(False)
    ax.grid(axis="y", color="#E8E6E1", lw=0.8)
    ax.set_axisbelow(True)
    return fig


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    # "passive" so the reset is not measured against itself between shots.
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"

    GAIN_F0G1 = 0.74
    # Bracket the predicted optimum widely enough to see Gamma turn over.
    GAIN_EF_LIST = [0.025, 0.035, 0.046, 0.060, 0.075, 0.095, 0.120, 0.150]
    # Fine enough to resolve the peak: at the predicted optimum tau ~ 0.18 us,
    # and 50 points over 5 us gave under 2 samples per tau. See CLAUDE.md.
    MAX_LENGTH = 6.0
    EXPTS = 150
    REPS = 3000
    READOUT_GAP = 2.0

    def add_arguments(parser):
        parser.add_argument(
            "--reanalyze", metavar="DATA_DIR",
            help="re-fit a saved run's h5 files and redraw; no hardware",
        )

    args = parse_args(__doc__, add_arguments, config=CONFIG)

    if args.reanalyze:
        # No hardware, no new run folder: read the saved h5 files and redo the
        # fit. gain_f0g1 comes from the data, not GAIN_F0G1, so re-analysing an
        # old run cannot silently use today's constant.
        auto_cfg = cfglib.load(str(REPO_DIR / "configs" / args.config))
        gains, gammas, tgrid, gain_f0g1 = load_run(args.reanalyze)
        data_dir = Path(args.reanalyze)
        # Write the figure beside the run's other images, not into data/.
        parts = list(data_dir.parts)
        out_dir = data_dir
        if "data" in parts:
            parts[len(parts) - 1 - parts[::-1].index("data")] = "metadata_and_images"
            out_dir = Path(*parts)
        if not out_dir.is_dir():
            out_dir = data_dir
    else:
        gain_f0g1 = GAIN_F0G1
        run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
            args, __file__, qi, JPA_ON
        )
        out_dir = Path(cfg_dict["metadata_path"])

    kappa = float(auto_cfg.device.f0g1.kappa)
    g_tilde = g_tilde_for_gain(auto_cfg, gain_f0g1)
    om_m, om_b = (float(v) for v in auto_cfg.device.f0g1.omega_ef_fit)
    print("")
    print(f"g~ = {g_tilde:.4f} MHz at f0-g1 gain {gain_f0g1:g}, "
          f"kappa = {kappa:.4f} MHz, ceiling = {kappa / 3:.4f}")
    print(f"omega_ef_fit = {om_m:.4f}*gain {om_b:+.4f} MHz")
    print("")

    if not args.reanalyze:
        gains, gammas, tgrid = [], [], None
        for gain_ef in GAIN_EF_LIST:
            print(f"===== gain_ef = {gain_ef:.4f} "
                  f"(nominal Omega_ef = {om_m * gain_ef + om_b:.4f} MHz)")
            expt = meas.ResetTimeScan(
                cfg_dict, qi=qi, go=False,
                params={
                    "gain_f0g1": GAIN_F0G1,
                    "gain_ef": gain_ef,
                    "max_length": MAX_LENGTH,
                    "expts": EXPTS,
                    "reps": REPS,
                    "readout_gap": READOUT_GAP,
                    # g and e only: |f> costs a third of the run time and the
                    # rate comes from the |e> trace.
                    "preps": ["g", "e"],
                    "baseline_preps": ["g"],
                    "reset_mode": RESET_MODE,
                },
            )
            finish_experiment(expt, args, wait=False)
            gains.append(gain_ef)
            gammas.append(float(expt.data.get("reset_gamma", np.nan)))
            if tgrid is None:
                tgrid = np.asarray(expt.data["xpts"], dtype=float)
            for fignum in plt.get_fignums():
                plt.close(fignum)
        gains, gammas = np.array(gains), np.array(gammas)

    if tgrid is None:
        raise SystemExit("no acquisition returned a time axis -- nothing to fit")

    res = analyze(gains, gammas, tgrid, g_tilde, kappa, om_m, om_b)
    fig = draw(gains, gammas, tgrid, g_tilde, kappa, om_m, om_b, res)
    # metadata_path, not the data folder, like every other image in this tree.
    out = out_dir / "reset_rate_vs_ef_gain.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")
    if not args.no_show:
        wait_for_plots(raise_windows=False)
