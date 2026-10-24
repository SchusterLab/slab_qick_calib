"""Run the FWM histogram suite over buffer length x pump gain x buffer gain, retaking any point whose scans look wrong."""

# Script 26 plus a buffer-length axis; CLAUDE.md "Buffer-length sweep (script 40)".

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import slab_qick_calib.experiments as meas

from slab_qick_calib.helpers import config

import fwm_qc
from common import (
    REPO_DIR,
    effective_final_delay,
    fwm_band_point,
    nonblocking_plots,
    parse_args,
    prepare_experiment,
    suppress_duplicate_config,
    wait_for_plots,
)


# Attempts per grid point; a point that fails them all is skipped (CLAUDE.md "QC'd efficiency sweep").
MAX_ATTEMPTS = 10


def add_arguments(parser):
    parser.add_argument(
        "--plots", choices=("none", "rejects", "all"), default="none",
        help="none (default): only the end-of-run NEP summary. rejects: also a "
             "histogram figure per rejected attempt. all: also the full "
             "per-point figure for every kept point, as script 23 draws.",
    )
    parser.add_argument(
        "--plot-every", type=int, default=10, metavar="N",
        help="draw the five scan histograms every Nth kept point (0 = never). "
             "A periodic look at the distributions without a figure per point; "
             "redundant under --plots all, which already draws every one.",
    )


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    buffer_line_qi = 0
    JPA_ON = True
    # "ramsey_spec" | "ramsey" | "stark"; CLAUDE.md "Photon number from Ramsey dephasing (scripts 38, 41)"
    PHOTON_CAL = "ramsey_spec"
    RESET_MODE = "f0g1"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"

    # Pump gains sit on the band via device.fwm.pump_power_fit. One for now; a pump scan is a longer list.
    BUFFER_GAINS = np.linspace(0, 0.03, 9)
    PUMP_GAIN = 0.6
    PUMP_GAINS = [PUMP_GAIN]
    # Log-spaced start -> centre -> end, centre always a point; length_p tracks length_b.
    N_LENGTHS = 15
    LENGTH_START, LENGTH_END = 2, 80  # us
    LENGTH_CENTER = 27  # us; None -> sqrt(2 * dead * T1_pump)
    args = parse_args(__doc__, add_arguments, config=CONFIG)

    # Before prepare_experiment, so a missing value creates no run folder.
    t1_pump = config.load(str(REPO_DIR / "configs" / args.config)).device.fwm.get("T1_pump")
    if t1_pump is None or not np.isfinite(t1_pump) or t1_pump <= 0:
        raise SystemExit(
            f"device.fwm.T1_pump is {t1_pump} in {args.config}; the buffer "
            "lengths are centred on it, so measure it first (39_t1_pump.py "
            "writes it) or enter it by hand."
        )

    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(0,)
    )

    buffer_frequency = auto_cfg.device.readout.frequency[buffer_line_qi]

    # Shared by every point; each acquires the five SMPDHistogramExperiment.SCANS.
    params = {
        "shots": 30000,
        "frequency_b": buffer_frequency,
        "length_b": 20,
        "delay_b": 0,
        "length_p": 20,
        "check_f": True,
        # No d: only the pump duration changes here (CLAUDE.md
        # "The detuned-pump scan").
        "scans": ("g", "e", "p"),
        "sep_readout": True,
        "reset_mode": RESET_MODE,
        # Shorter than the config's safe default; only safe in 26/37,
        # CLAUDE.md "Gap and ring-down in scripts 26/37".
        "f0g1_reset_gap": 0.0,
        "extra_delay": 1.0 - 0.1,
    }

    # Dead time = period - length_b, from a never-run probe so it follows RESET_MODE.
    probe = meas.SMPDHistogramExperiment(cfg_dict, qi=qi, go=False, params=params).cfg.expt
    dead_us = (max(probe.length_p, probe.delay_b + probe.length_b) - probe.length_b
               + probe.readout_length + effective_final_delay(probe))
    length_optimum = np.sqrt(2 * dead_us * t1_pump)
    if LENGTH_CENTER is None:
        LENGTH_CENTER = length_optimum
    if not LENGTH_START < LENGTH_CENTER < LENGTH_END:
        raise SystemExit(f"LENGTH_CENTER {LENGTH_CENTER:.1f} us is not inside "
                         f"({LENGTH_START}, {LENGTH_END}) us")
    # Points below the centre in proportion to its share of the log range.
    n_below = int(round((N_LENGTHS - 1) * np.log(LENGTH_CENTER / LENGTH_START)
                        / np.log(LENGTH_END / LENGTH_START)))
    n_below = min(max(n_below, 1), N_LENGTHS - 2)
    BUFFER_LENGTHS = np.unique(np.round(np.concatenate([
        np.geomspace(LENGTH_START, LENGTH_CENTER, n_below + 1),
        np.geomspace(LENGTH_CENTER, LENGTH_END, N_LENGTHS - n_below),
    ])).astype(int))
    print(f"T1_pump = {t1_pump:.1f} us, dead time = {dead_us:.1f} us, "
          f"sqrt(2 * dead * T1_pump) = {length_optimum:.1f} us; "
          f"centre {LENGTH_CENTER:.1f} us")
    print(f"Buffer lengths: {BUFFER_LENGTHS}")
    print(f"Buffer gains: {BUFFER_GAINS}")
    print(PUMP_GAINS)

    # QC reference frame and bins, fixed for the whole sweep; see fwm_qc.
    reference = fwm_qc.reference_from_config(auto_cfg, qi, params["shots"])
    edges, centers = fwm_qc.bin_grid(reference)
    monitor = fwm_qc.QCMonitor(reference, edges)
    print(f"QC reference: {reference}")

    metadata_path = Path(cfg_dict["metadata_path"])
    log_path = metadata_path / "qc_log.csv"
    rejects_path = metadata_path / "qc_rejects.h5"

    total = len(BUFFER_LENGTHS) * len(PUMP_GAINS) * len(BUFFER_GAINS)
    est_s = sum(len(PUMP_GAINS) * len(BUFFER_GAINS) * 5 * params["shots"]
                * (float(L) + dead_us) * 1e-6 for L in BUFFER_LENGTHS)
    print(
        f"{total} histogram suites: {len(BUFFER_LENGTHS)} buffer lengths x "
        f"{len(PUMP_GAINS)} pump powers x {len(BUFFER_GAINS)} buffer gains, "
        f"5 sequences of {params['shots']} shots each, up to {MAX_ATTEMPTS} "
        f"attempts per point"
    )
    print(f"  ~{est_s / 3600:.2f} h of acquisition if nothing is retaken")
    print(f"Plots: {args.plots}"
          + ("" if args.plots != "none" else
             " (the histograms of any rejected attempt are still saved; "
             "replot them later with fwm_qc.py --replot-rejects)"))

    # Flushed per attempt, so a killed run still leaves a complete log.
    log_file = log_path.open("w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=fwm_qc.LOG_COLUMNS)
    log_writer.writeheader()

    # Ground-only health check, built once and re-acquired; never saved
    # (CLAUDE.md "Ground-state recovery").
    ground_check = fwm_qc.make_ground_check(cfg_dict, qi, params)
    ground_file = (metadata_path / "ground_waits.csv").open("w", newline="")
    ground_writer = csv.DictWriter(ground_file,
                                   fieldnames=fwm_qc.GROUND_WAIT_COLUMNS)
    ground_writer.writeheader()

    # Do not start on an unhealthy qubit -- it would only burn retakes on every
    # point (CLAUDE.md "Ground-state recovery"). One pass is enough to start; a
    # mid-sweep failure needs GROUND_CONSECUTIVE in a row.
    start_wait = fwm_qc.wait_for_ground(ground_check, reference, consecutive=1)
    print(f"Ground state before the sweep: {start_wait['p_e_final']:.4f} "
          f"(bar {fwm_qc.GROUND_MAX_PE}, {start_wait['checks']} check(s))")
    if start_wait["failures"]:
        ground_writer.writerow({"index": -1, "attempt": 0, **start_wait})
        ground_file.flush()

    accepted = []
    tally = {"first_try": 0, "retaken": 0, "failed_skipped": 0}
    step = 0
    open_figures = []
    try:
      for buffer_length in BUFFER_LENGTHS:
        print(f"\n===== buffer length {buffer_length} us "
              f"(pump length tracks it) =====")

        for pump_gain in PUMP_GAINS:
            pump_frequency, pump_gain = fwm_band_point(auto_cfg, None, pump_gain)

            for buffer_gain in BUFFER_GAINS:
                step += 1
                # Close the previous point's windows by number (plt.figure(num) re-raises them).
                for fignum in open_figures:
                    plt.close(fignum)
                open_figures = []
                print(
                    f"--- {step}/{total}: length {buffer_length} us, pump "
                    f"{pump_frequency:.3f} MHz gain {pump_gain:.4f}, "
                    f"buffer gain {buffer_gain:.4f}"
                )

                ok = False
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    # Built inside the loop: the filename is the next unused index in the run folder.
                    histogram = meas.SMPDHistogramExperiment(
                        cfg_dict,
                        qi=qi,
                        go=False,
                        params={
                            **params,
                            "frequency_p": pump_frequency,
                            "gain_p": pump_gain,
                            "gain_b": buffer_gain,
                            "length_b": int(buffer_length),
                            "length_p": int(buffer_length),
                        },
                    )
                    # Not finish_experiment, which saves before QC can reject (CLAUDE.md "QC'd efficiency sweep").
                    # One config yml per run folder, not per point (CLAUDE.md "One config yml per run").
                    suppress_duplicate_config(histogram, metadata_path)
                    histogram.run(
                        progress=not args.no_progress,
                        analyze=True,
                        display=False,
                        save=False,
                    )
                    result = fwm_qc.analyze_point(
                        histogram.data, edges, centers, reference
                    )
                    ok, failed = monitor.check(
                        result["metrics"], result["hist"], pump_gain, buffer_gain
                    )

                    last = attempt == MAX_ATTEMPTS
                    status = ("ok" if ok else
                              "failed_skipped" if last else "rejected")
                    log_writer.writerow(fwm_qc.log_row(
                        step - 1, attempt, pump_gain, buffer_gain, status,
                        failed, result, histogram.data,
                    ))
                    log_file.flush()

                    if ok:
                        tally["first_try" if attempt == 1 else "retaken"] += 1
                        break

                    # The five histograms and the metrics, not the shots.
                    fwm_qc.record_reject(
                        rejects_path, step - 1, attempt, pump_gain, buffer_gain,
                        failed, result,
                    )
                    if args.plots in ("rejects", "all"):
                        # Same figure as fwm_qc --replot-rejects, so --plots none loses nothing.
                        fig = fwm_qc.plot_point(
                            result["centers"], result["hist"],
                            title=(f"#{step - 1:05d} attempt {attempt} -- pump "
                                   f"gain {pump_gain:.4f}, buffer gain "
                                   f"{buffer_gain:.4f}"),
                            failed=failed,
                        )
                        fig.savefig(metadata_path /
                                    f"qc_reject_{step - 1:05d}_attempt{attempt}.png")
                        if args.no_show:
                            plt.close(fig)
                        else:
                            open_figures.append(fig.number)
                    # A failure means the qubit may be unhealthy; hold the
                    # sweep until |g> recovers (CLAUDE.md "Ground-state
                    # recovery"). Before `if last` so a skipped point waits too.
                    wait = fwm_qc.wait_for_ground(ground_check, reference)
                    if wait["failures"]:
                        ground_writer.writerow(
                            {"index": step - 1, "attempt": attempt, **wait})
                        ground_file.flush()
                    if last:
                        tally["failed_skipped"] += 1
                        print(f"  !! skipped after {MAX_ATTEMPTS} failed "
                              f"attempts: {'; '.join(failed)}")
                    else:
                        print(f"  retake {attempt}/{MAX_ATTEMPTS}: "
                              f"{'; '.join(failed)}")

                # Never passed QC: skipped, not saved or added to the history (CLAUDE.md "QC'd efficiency sweep").
                if not ok:
                    continue

                histogram.save_data(histogram.data)
                if args.plots == "all":
                    # display() needs the shots (IQ panels) and blocks on plt.show(), hence the wrapper.
                    with nonblocking_plots():
                        histogram.display(histogram.data)
                    if args.no_show:
                        plt.close("all")
                    else:
                        open_figures = plt.get_fignums()
                else:
                    # Explicit: the per-point yml is normally written inside display().
                    histogram.save_config()


                # Periodic look at the distributions; same figure as a reject.
                if (args.plot_every and args.plots != "all"
                        and step % args.plot_every == 0):
                    fig = fwm_qc.plot_point(
                        result["centers"], result["hist"],
                        title=(f"#{step - 1:05d}  "
                               + f"{buffer_length} us, pump {pump_gain:.4f}, "
                                   f"buffer {buffer_gain:.4f}"))
                    fig.savefig(metadata_path / f"qc_point_{step - 1:05d}.png")
                    if args.no_show:
                        plt.close(fig)
                    else:
                        open_figures = list(open_figures) + [fig.number]

                expt = histogram.cfg.expt
                accepted.append({
                    "gain_p": pump_gain,
                    "gain_b": buffer_gain,
                    "length_b": float(expt.length_b),
                    "frequency_b": float(expt.frequency_b),
                    # Per point: the period grows with the buffer length.
                    "cycle_us": float(
                        max(expt.length_p, expt.delay_b + expt.length_b)
                        + expt.readout_length + effective_final_delay(expt)),
                    "p": {scan: m["p"]
                          for scan, m in result["metrics"].items()},
                })
    finally:
        log_file.close()
        ground_file.close()

    print(f"\nSaved {step - tally['failed_skipped']}/{step} histogram suites "
          f"to {run_dir}")
    print(f"  {tally['first_try']} passed first try, "
          f"{tally['retaken']} needed a retake, "
          f"{tally['failed_skipped']} skipped after {MAX_ATTEMPTS} failures "
          "(not saved; see qc_rejects.h5)")
    print(f"  attempt-by-attempt log: {log_path}")
    if rejects_path.exists():
        print(f"  rejected attempts' histograms: {rejects_path}")
        if args.plots == "none":
            print("  to plot those failures: "
                  f"python fwm_qc.py --replot-rejects {rejects_path}")

    # Indicative only (thresholded, not fitted); quote fit_shared_means.py.
    try:
        # From the experiment's own cfg, i.e. what was played -- the fields fit_shared_means.py reads.
        buffer_attn_db = fwm_qc.buffer_attenuation_db(histogram.cfg)
        buffer_pulse = histogram.cfg.expt.pulse_type_b
        calib = fwm_qc.load_calibration(
            auto_cfg, source=PHOTON_CAL,
            attn_db=buffer_attn_db, pulse=buffer_pulse,
        )
    except ValueError as error:
        print(f"\nNo NEP summary: {error}")
    else:
        if not accepted:
            raise SystemExit(
                "No point passed QC, so there is nothing to reduce. The "
                f"per-attempt reasons are in {log_path}."
            )
        expt = histogram.cfg.expt
        # Inter-shot tail via effective_final_delay, not final_delay (CLAUDE.md "`common.py` contract").
        cycle_us = (max(expt.length_p, expt.delay_b + expt.length_b)
                    + expt.readout_length + effective_final_delay(expt))
        # One reduction per pump gain, a row per length (CLAUDE.md script 40).
        rows = []
        dynamics = {}
        for pump_gain in sorted({pt["gain_p"] for pt in accepted}):
            at_pump = [pt for pt in accepted if pt["gain_p"] == pump_gain]
            sub, info = fwm_qc.reduce_to_nep(
                at_pump, calib, t1=t1_pump,
                cycle_us=cycle_us, by="length_b", dynamics=True,
            )
            dynamics[pump_gain] = info["dynamics"]
            for row in sub:
                row["gain_p"] = pump_gain
            rows += sub
        summary_path = metadata_path / "qc_summary.csv"
        columns = ["gain_p", "length_b", "cycle_us", "n_points",
                   "eta_e", "eta_err_e", "eta_f", "eta_err_f",
                   "p_dark", "n_equiv", "nep", "eta_qubit", "eta_ceiling"]
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nEfficiency summary (indicative; thresholded fractions) -> {summary_path}")
        print(f"  photon calibration: {PHOTON_CAL}, "
              f"n_bar = {calib['coeffs']} on gain powers "
              f"{fwm_qc.GAIN_FIT_POWERS[calib['fit_type']]}")
        # Keyed on the reference tags, not the scale: a tagged fit can also have scale 1.0.
        if calib["reference"] is not None:
            ref_attn, ref_pulse = calib["reference"]
            print(f"    stated at {ref_attn:.0f} dB / {ref_pulse}; this run is "
                  f"{buffer_attn_db:.0f} dB / {buffer_pulse} "
                  f"-> x{calib['scale']:.4g}")
        else:
            print(f"    used as stored, un-rescaled; this run's buffer is "
                  f"{buffer_attn_db:.0f} dB / {buffer_pulse} and the "
                  "calibration records no tag to check that against")
        print(f"  T1_pump = {t1_pump:.1f} us, kappa_c/kappa = "
              f"{calib['kappa_c'] / calib['kappa']:.3f}, dead = {dead_us:.1f} us, "
              f"sqrt(2 dead T1_pump) = {length_optimum:.1f} us")
        # Pump-off amplitude ring-up 2/kappa; the fit's tau should be at or below it.
        tau_ring = 1 / (np.pi * calib["kappa"])
        print(f"  pump-off buffer ring-up 1/(pi kappa) = {tau_ring:.2f} us; "
              f"the ceiling uses the matched (Gamma = kappa) {tau_ring / 2:.2f} us")
        for pump_gain in sorted({r["gain_p"] for r in rows}):
            at_pump = sorted((r for r in rows if r["gain_p"] == pump_gain),
                             key=lambda r: r["length_b"])
            print(f"  pump {pump_gain:.4f}\n  {'len/us':>7} {'duty':>6} "
                  f"{'eta_e':>16} {'eta_f':>16} {'eta_e*duty':>10} {'ceiling':>8}")
            for r in at_pump:
                duty = r["length_b"] / r["cycle_us"]
                print(f"  {r['length_b']:7.1f} {duty:6.3f} "
                      f"{r['eta_e']:7.4f} ± {r['eta_err_e']:6.4f} "
                      f"{r['eta_f']:7.4f} ± {r['eta_err_f']:6.4f} "
                      f"{r['eta_e'] * duty:10.4f} {r['eta_ceiling']:8.3f}")
            # fig.savefig, not save_fig: that calls tight_layout() and a blocking plt.show().
            fig, fits = fwm_qc.plot_efficiency_vs_length(
                at_pump, t1_pump, dynamics[pump_gain], tau_ring / 2)
            for scan, fit in fits.items():
                print(f"    {scan}: A = {fit['A']:.4f} ± {fit['A_err']:.4f}, "
                      f"T = {fit['T']:.1f} ± {fit['T_err']:.1f} us, "
                      f"tau = {fit['tau']:.2f} ± {fit['tau_err']:.2f} us, "
                      f"rms {fit['rms']:.4f}, eta x duty peaks at {fit['L_peak']:.1f} us")
            figure_path = metadata_path / f"qc_efficiency_vs_length_p{pump_gain:.3f}.png"
            fig.savefig(figure_path)
            print(f"  figure -> {figure_path}")

    if not args.no_show:
        wait_for_plots()
