"""Where the |g> floor comes from: a 2x2 drive-on/off factorial.

Diagnostic for script 35; writes nothing.
See scripts/SMPD/CLAUDE.md, "Runner notes, scripts 27–36".

    python 35b_reset_floor.py --no-show
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers.f0g1 import (
    ef_gain_for_omega,
    f0g1_frequency_for_gain,
    g_tilde_for_gain,
    optimal_omega_ef,
)

from common import finish_experiment, parse_args, prepare_experiment, wait_for_plots


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    # Passive for the same reason as script 35.
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    MAX_LENGTH = 2.0
    EXPTS = 10
    REPS = 3000
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON
    )

    # Operating point computed once so every case gets the same e-f drive.
    gain_f0g1 = float(auto_cfg.device.qubit.max_gain)
    kappa = float(auto_cfg.device.f0g1.kappa)
    g_tilde = g_tilde_for_gain(auto_cfg, gain_f0g1)
    omega_opt, gamma_pred = optimal_omega_ef(g_tilde, kappa)
    gain_ef = ef_gain_for_omega(auto_cfg, omega_opt, verbose=False)
    freq_f0g1 = f0g1_frequency_for_gain(auto_cfg, gain_f0g1)
    a_ef, c_ef = (float(v) for v in auto_cfg.device.f0g1.ef_stark_fit)
    freq_ef = a_ef * gain_f0g1**2 + c_ef

    print("")
    print(f"operating point held fixed across all four cases:")
    print(f"  f0-g1 : gain {gain_f0g1:g} at {freq_f0g1:.4f} MHz "
          f"-> g~/2pi = {g_tilde:.4f} MHz")
    print(f"  e-f   : gain {gain_ef:.5f} at {freq_ef:.4f} MHz "
          f"-> Omega_ef/2pi = {omega_opt:.4f} MHz")
    print(f"  predicted Gamma/2pi = {gamma_pred:.4f} MHz")
    print("")

    CASES = (
        ("both", gain_f0g1, gain_ef),
        ("f0g1 only", gain_f0g1, 0.0),
        ("e-f only", 0.0, gain_ef),
        ("neither", 0.0, 0.0),
    )

    results = {}
    for label, g_f, g_e in CASES:
        print(f"===== {label}: gain_f0g1 = {g_f:g}, gain_ef = {g_e:g}")
        params = {
            "gain_f0g1": g_f,
            "gain_ef": g_e,
            # Frequencies pinned, not derived, so a zero gain cannot also move
            # the tone it belongs to.
            "frequency_f0g1": freq_f0g1,
            "frequency_ef": freq_ef,
            "max_length": MAX_LENGTH,
            "expts": EXPTS,
            "reps": REPS,
            "preps": ["g", "e"],
            "reset_mode": RESET_MODE,
        }
        expt = meas.ResetTimeScan(cfg_dict, qi=qi, go=False, params=params)
        # Each experiment is built inside the loop and saved before the next is
        # constructed, or they all pick the same filename index.
        finish_experiment(expt, args, wait=False)
        results[label] = {
            "xpts": np.asarray(expt.data["xpts"], dtype=float),
            "p_e_g": np.asarray(expt.data["p_e_g"], dtype=float),
            "p_sat_g": float(expt.data["p_sat_g"]),
            "p_sat_e": float(expt.data["p_sat_e"]),
            "gamma": float(expt.data.get("reset_gamma", np.nan)),
        }
        for fignum in plt.get_fignums():
            plt.close(fignum)

    base = results["neither"]["p_sat_g"]
    print("")
    print("=" * 66)
    print("|g> floor attribution   (p_sat_g = thresholded 'not g', prep |g>)")
    print("=" * 66)
    print(f"  {'case':11s}{'p_sat_g':>10s}{'vs neither':>13s}"
          f"{'p_sat_e':>10s}{'Gamma':>9s}")
    for label, _, _ in CASES:
        r = results[label]
        print(f"  {label:11s}{r['p_sat_g'] * 100:9.2f}%"
              f"{(r['p_sat_g'] - base) * 100:+12.2f}%"
              f"{r['p_sat_e'] * 100:9.2f}%{r['gamma']:9.4f}")
    print("")
    print(f"  baseline (no drives)        {base * 100:6.2f}%"
          "   <- misassignment + thermal")
    d_f0g1 = results["f0g1 only"]["p_sat_g"] - base
    d_ef = results["e-f only"]["p_sat_g"] - base
    d_both = results["both"]["p_sat_g"] - base
    print(f"  added by the f0-g1 drive    {d_f0g1 * 100:+6.2f}%")
    print(f"  added by the e-f drive      {d_ef * 100:+6.2f}%")
    print(f"  added by both together      {d_both * 100:+6.2f}%"
          f"   (sum of the two alone: {(d_f0g1 + d_ef) * 100:+.2f}%)")
    if abs(d_both) < 0.02:
        print("  => the drives are NOT the source. Look at readout_gap and the")
        print("     passive delay next, and compare against the single-shot")
        print("     |g> blob taken with no sequence at all.")
    else:
        worst = "f0-g1" if abs(d_f0g1) > abs(d_ef) else "e-f"
        print(f"  => dominated by the {worst} drive. Scan its gain to see how the")
        print("     floor scales, which separates a coherent drive effect from")
        print("     broadband DAC noise landing on f_ge.")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for (label, _, _), color in zip(CASES, ("#C1440E", "#4C6EF5", "#2F9E44", "#868E96")):
        r = results[label]
        ax.plot(r["xpts"], r["p_e_g"] * 100, "o-", ms=5, lw=1.6, color=color,
                label=f"{label} ({r['p_sat_g'] * 100:.2f}%)")
    ax.axhline(base * 100, ls=":", color="#33322E", lw=1.2)
    ax.set_xlabel("Drive time ($\\mu$s)")
    ax.set_ylabel("P(not g) with |g> prepared (%)")
    ax.legend(fontsize=8, title="drives on", title_fontsize=8)
    ax.set_ylim(bottom=0)
    # metadata_path is where save_fig writes; run_dir is the data folder.
    out = Path(cfg_dict["metadata_path"]) / "reset_floor_attribution.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")
    if not args.no_show:
        wait_for_plots(raise_windows=False)
