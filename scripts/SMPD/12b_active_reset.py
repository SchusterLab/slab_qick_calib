"""Calibrate and qualify active reset for one qubit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import slab_qick_calib.experiments as meas

from slab_qick_calib.calib.readout_helpers import plot_reset
from slab_qick_calib.helpers import config

from common import parse_args, prepare_experiment, wait_for_plots


def quiet_figures(fn, *fn_args, **fn_kwargs):
    """
    Call `fn` with `plt.show()` stubbed out, then close whatever it drew.

    This script runs one histogram per threshold and `check_reset()` opens two
    figures each time, so without this the run ends with thirty-odd unlabelled
    windows on screen. Everything is still written to disk by the experiment's
    own `save_fig`; only the one summary figure at the end stays open.
    """
    before = set(plt.get_fignums())
    original_show = plt.show
    plt.show = lambda *a, **k: None
    try:
        return fn(*fn_args, **fn_kwargs)
    finally:
        plt.show = original_show
        # Close by number, never via plt.figure(num), which re-raises the
        # window and makes it flash (see common._raise_windows).
        for num in set(plt.get_fignums()) - before:
            plt.close(num)


def run_quietly(experiment, args):
    """Acquire, analyze and save one histogram, drawing nothing on screen."""
    quiet_figures(experiment.run, analyze=True, display=False, save=True,
                  progress=not args.no_progress)
    return experiment


def reset_metrics(experiment):
    """Run `check_reset()` for its numbers and return (reset_e, reset_g)."""
    quiet_figures(experiment.check_reset)
    return experiment.data["reset_e"], experiment.data["reset_g"]


def post_reset_population(experiment, score_threshold):
    """
    Excited population left after the reset, from an |e>-prepared shot.

    Thresholds the *last* recorded reset round, which is the true post-reset
    state only because the calibration runs with `reset_verify=True`; without
    it the final conditional pi pulse is unmeasured and this is one round
    stale.

    Returns (from_e, floor, from_f). `from_f` is the same measurement from
    an |f>-prepared shot, or None when the run had no f scan.

    Pick a threshold by minimising `from_e`, not `from_e - floor`: when the
    reset threshold sits near or below g_mean it pi-pulses nearly every shot,
    driving the preparations to the same high population, and the difference
    collapses to ~0 while both are terrible. `floor` -- the same measurement
    from a |g>-prepared shot -- is reported alongside because it is the
    scoring threshold misassigning genuine ground shots, so `from_e - floor`
    estimates the population really left behind. Useful to read, unsafe to
    optimise.

    |e> rather than |f> is the one to minimise on: with reset_f a not-ground
    shot is assumed to be |f> and walked f -> e -> g, so |f> clears in one
    round while |e> is driven up to |f> first and needs two. |e> is the
    harder case, and `from_f` is the confirmation that the pi_ef leg works.
    """
    above = lambda x: float(np.mean(np.asarray(x).ravel() > score_threshold))
    data = experiment.data
    from_f = above(data["Ifr"][-1]) if "Ifr" in data else None
    return above(data["Ier"][-1]), above(data["Igr"][-1]), from_f


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_ROUNDS = 6        # up to this many conditional pi pulses
    # Also prepare |f> and reset out of it (reset_f); costs a third acquisition per point.
    CHECK_F = True

    # Editable sweep settings.
    SHOTS = 10000
    THRESHOLD_PTS = 20      # points across +/- THRESHOLD_SIGMA of the stored value
    THRESHOLD_SIGMA = 2.0
    FINAL_DELAYS = [1, 3, 5, 7, 10]   # us, swept with reset=1

    if CHECK_F and RESET_ROUNDS < 2:
        raise SystemExit(
            "CHECK_F needs RESET_ROUNDS >= 2: reset_f assumes a not-ground "
            "shot is in |f>, so an |e> shot is driven to |f> by the pi_ef and "
            "only comes back on the following round. With one round it would "
            "end worse than not resetting."
        )

    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON
    )

    g_mean = auto_cfg.device.readout.g_mean[qi]
    e_mean = auto_cfg.device.readout.e_mean[qi]
    sigma = auto_cfg.device.readout.sigma[qi]
    threshold = auto_cfg.device.readout.threshold[qi]

    # reset() pulses on I >= threshold, so g_mean < threshold < e_mean is required;
    # CLAUDE.md "How the branch works, and the one way it can go wrong".
    ordered = g_mean < threshold < e_mean
    print(
        f"g_mean={g_mean:.4f}  threshold={threshold:.4f}  e_mean={e_mean:.4f}  "
        f"sigma={sigma:.4f}"
    )
    if ordered:
        print(
            f"  ordering OK: ground is below threshold "
            f"({(threshold - g_mean) / sigma:.2f} sigma above g, "
            f"{(e_mean - threshold) / sigma:.2f} sigma below e)"
        )
    else:
        print(
            "  ORDERING WRONG: the threshold does not separate g from e in the "
            "expected direction, so reset would excite the qubit rather than "
            "reset it. Re-run 12_single_shot.py before going further."
        )

    # --- 1. Baseline (setup_reset keeps the passive delay) --------------------
    print(f"\nBaseline with reset={RESET_ROUNDS} (setup_reset, passive delay kept)")
    baseline = meas.HistogramExperiment(
        cfg_dict, qi=qi, go=False, check_f=CHECK_F,
        params={"shots": SHOTS, "reset_mode": "active", "setup_reset": True,
                "reset": RESET_ROUNDS, "reset_verify": True},
    )
    run_quietly(baseline, args)
    baseline_e, baseline_g = reset_metrics(baseline)
    print(f"  reset_e={baseline_e:.4f}  reset_g={baseline_g:.4f}")

    # --- 2. Threshold sweep; CLAUDE.md "device.readout.reset_threshold" ------
    thresholds = np.linspace(
        threshold - THRESHOLD_SIGMA * sigma,
        threshold + THRESHOLD_SIGMA * sigma,
        THRESHOLD_PTS,
    )
    print(f"\nThreshold sweep: {THRESHOLD_PTS} points over "
          f"{thresholds[0]:.4f} .. {thresholds[-1]:.4f}")

    shots_list, fidelities, reset_es, reset_gs = [], [], [], []
    from_es, floors, from_fs = [], [], []
    for i, t in enumerate(thresholds):
        # Built inside the loop: the filename is the next unused index in the run folder.
        shot = meas.HistogramExperiment(
            cfg_dict, qi=qi, go=False, check_f=CHECK_F,
            params={"shots": SHOTS, "threshold_v": float(t),
                    "reset_mode": "active", "setup_reset": True,
                    "reset": RESET_ROUNDS, "reset_verify": True},
        )
        run_quietly(shot, args)
        r_e, r_g = reset_metrics(shot)
        from_e, floor, from_f = post_reset_population(shot, threshold)
        shots_list.append(shot)
        fidelities.append(float(shot.data["fids"][0]))
        reset_es.append(r_e)
        reset_gs.append(r_g)
        from_es.append(from_e)
        floors.append(floor)
        from_fs.append(from_f)
        print(f"  [{i + 1}/{THRESHOLD_PTS}] threshold={t:8.4f}  "
              f"|e> after reset {from_e*100:6.2f}%   |g> floor "
              f"{floor*100:5.2f}%"
              + (f"   |f> {from_f*100:6.2f}%" if from_f is not None else "")
              + f"   (left behind ~{(from_e - floor)*100:5.2f}%)")

    # Selected on the absolute post-reset |e> population, which cannot be gamed
    # (CLAUDE.md "device.readout.reset_threshold").
    best = int(np.argmin(from_es))
    best_threshold = float(thresholds[best])
    at_config = int(np.argmin(np.abs(thresholds - threshold)))
    print()
    print(f"Lowest |e> population after reset {from_es[best]*100:.2f}% at "
          f"threshold={best_threshold:.4f}; at the stored {threshold:.4f} "
          f"it is {from_es[at_config]*100:.2f}%")
    if CHECK_F and from_fs[best] is not None:
        print(f"  |f> prep at the same threshold: "
              f"{from_fs[best]*100:.2f}% (clears in one round, so it "
              f"should sit at or below the |e> number)")

    # Per-round histograms need reset >= 2; output_dir because plot_reset assumes the notebook layout.
    if RESET_ROUNDS >= 2:
        quiet_figures(plot_reset, shots_list, shots_list[-1].fname,
                      output_dir=cfg_dict["metadata_path"])

    # --- 3. Final-delay sweep: fidelity vs inter-shot delay, reset=1 ----------
    print(f"\nFinal-delay sweep with reset=1: {FINAL_DELAYS} us")
    delay_fidelities = []
    for f in FINAL_DELAYS:
        shot = meas.HistogramExperiment(
            cfg_dict, qi=qi, go=False, check_f=False,
            # reset=1 here, so reset_f must stay off: with a single round an
            # |e> shot would be driven to |f> and left there.
            params={"shots": SHOTS, "reset_mode": "active", "setup_reset": False,
                    "reset": 1, "reset_f": False, "final_delay": f},
        )
        run_quietly(shot, args)
        delay_fidelities.append(float(shot.data["fids"][0]))
        print(f"  final_delay={f} us -> fidelity={delay_fidelities[-1]:.4f}")

    # --- Summary figure (fig.savefig: save_fig blocks on plt.show()) ----------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(thresholds, [(a - b) * 100 for a, b in zip(from_es, floors)],
               "o-", label="left behind (|e> minus floor)")
    if CHECK_F and all(v is not None for v in from_fs):
        ax[0].plot(thresholds, [v * 100 for v in from_fs], "d--", lw=1,
                   label="|f> prep, last round")
    ax[0].plot(thresholds, [r * 100 for r in from_es], "^--", lw=1,
               label="|e> prep, last round")
    ax[0].plot(thresholds, [r * 100 for r in floors], "v--", lw=1,
               label="|g> prep, last round (scoring floor)")
    ax[0].axvline(threshold, color="r", ls="--", lw=1,
                  label=f"config threshold {threshold:.3f}")
    ax[0].axvline(best_threshold, color="g", ls=":", lw=1,
                  label=f"lowest |e> after reset {best_threshold:.3f}")
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_xlabel("Reset threshold (V)")
    ax[0].set_ylabel("Excited population after reset (%)")
    ax[0].legend(fontsize="x-small")

    ax[1].plot(thresholds, fidelities, "o-", label="fidelity vs reset threshold")
    ax[1].axvline(threshold, color="r", ls="--", lw=1, label="config threshold")
    ax[1].set_xlabel("Reset threshold (V)")
    ax[1].set_ylabel("Readout fidelity")
    ax[1].legend(fontsize="x-small")

    ax[2].plot(FINAL_DELAYS, delay_fidelities, "o-",
               label="fidelity vs inter-shot delay (reset=1)")
    ax[2].set_xlabel("Final delay (us)")
    ax[2].set_ylabel("Readout fidelity")
    ax[2].legend(fontsize="x-small")
    fig.tight_layout()

    summary_path = Path(cfg_dict["metadata_path"]) / "active_reset_summary.png"
    fig.savefig(summary_path)

    metadata_dir = Path(cfg_dict["metadata_path"])
    print()
    print(f"Figures written to {metadata_dir}")
    print("  active_reset_summary.png     the three panels shown on screen")
    print("  *_reset_hist_ground.png      per-threshold |g> shots, one panel per round")
    print("  *_reset_hist_excited.png     the same from a prepared |e>")
    print("  *_reset_hist.png             IQ blobs and histograms vs round count")
    print("  *_reset_performance.png      one per point, check_reset's own view")

    # --- Verdict: reset_e should be << 1, reset_g ~ 1 --------------------------
    print("\nVerdict")
    print(f"  reset_e={baseline_e:.4f} (want << 1: the excited population is gone)")
    print(f"  reset_g={baseline_g:.4f} (want ~1: reset is not exciting the ground state)")
    if not ordered:
        print("  NOT USABLE -- the threshold ordering above is wrong.")
    elif baseline_e < 0.15 and baseline_g / max(baseline_e, 1e-9) > 2:
        print("  Reset is working. RESET_MODE = 'active' is reasonable on this device.")
    else:
        print("  Reset is weak. Check the threshold sweep above before relying on it;"
              " a lower reset_e may be reachable at a different threshold.")

    if not args.no_update_config:
        config.update_readout(str(cfg_path), "reset_e", baseline_e, qi)
        config.update_readout(str(cfg_path), "reset_g", baseline_g, qi)
        # threshold stays 12_single_shot.py's; reset gets its own field.
        config.update_readout(str(cfg_path), "reset_threshold",
                              best_threshold, qi)
        print(f"  reset_threshold -> {best_threshold:.4f} "
              f"(discrimination threshold left at {threshold:.4f})")

    if not args.no_show:
        wait_for_plots()
