"""Run the FWM histogram suite over a buffer-gain x pump-gain grid, retaking any point whose scans look wrong."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import slab_qick_calib.experiments as meas

import fwm_qc
from common import (
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
    RESET_MODE = "while"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"

    # The grid. Pump powers are gains; the frequency at each comes from
    # device.fwm.pump_power_fit, so every point sits on the band.
    BUFFER_GAINS = np.linspace(0.00125, 0.01, 8)
    # PUMP_GAINS = [0.05, 0.1, 0.15, 0.2,0.25,0.3,0.35, 0.4, 0.45, 0.5,0.55]
    # start at 0.95 to just see if it depends on the direction of the sweep
    PUMP_GAINS = np.arange(0.95, 0.05, -0.05)
    # PUMP_GAINS = np.arange(0.05, 0.95, 0.05)
    # PUMP_GAINS = np.arange(0.01, 0.6, 0.01)
    # PUMP_GAINS = [0.5]
    print(f"Buffer gains: {BUFFER_GAINS}")
    print(f"Pump gains: {PUMP_GAINS}")
    args = parse_args(__doc__, add_arguments, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(0,)
    )

    buffer_frequency = auto_cfg.device.readout.frequency[buffer_line_qi]

    # Shared by every point; each acquires the five SMPDHistogramExperiment.SCANS.
    params = {
        "shots": 30000,
        "frequency_b": buffer_frequency,
        "length_b": 41,
        "delay_b": 0,
        "length_p": 41,
        "check_f": True,
        # g e p d f; d is the off-band pump control (CLAUDE.md
        # "The detuned-pump scan"), w the no-reset control (CLAUDE.md
        # "The passive-wait scan") -- drop "w" to save ~13 s per point.
        "scans": ("g", "e", "p", "d", "w"),
        "detune_p": 30.0,
        "sep_readout": True,
        "reset_mode": RESET_MODE,
        # Shorter than the config's safe default; only safe in 26/37,
        # CLAUDE.md "Gap and ring-down in scripts 26/37".
        "f0g1_reset_gap": 0.0,
        "extra_delay": 1.0 - 0.1,
    }

    # QC reference frame and bins, fixed for the whole sweep; see fwm_qc.
    reference = fwm_qc.reference_from_config(auto_cfg, qi, params["shots"])
    edges, centers = fwm_qc.bin_grid(reference)
    monitor = fwm_qc.QCMonitor(reference, edges)
    print(f"QC reference: {reference}")

    metadata_path = Path(cfg_dict["metadata_path"])
    log_path = metadata_path / "qc_log.csv"
    rejects_path = metadata_path / "qc_rejects.h5"

    total = len(PUMP_GAINS) * len(BUFFER_GAINS)
    print(
        f"{total} histogram suites: {len(PUMP_GAINS)} pump powers x "
        f"{len(BUFFER_GAINS)} buffer gains, 5 sequences of "
        f"{params['shots']} shots each, up to {MAX_ATTEMPTS} attempts per point"
    )
    print(f"Plots: {args.plots}"
          + ("" if args.plots != "none" else
             " (the histograms of any rejected attempt are still saved; "
             "replot them later with fwm_qc.py --replot-rejects)"))

    # Opened once and flushed per attempt, so killing the run mid-sweep still
    # leaves a complete log of everything already taken.
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
        for pump_gain in PUMP_GAINS:
            # The band frequency for this pump power.
            pump_frequency, pump_gain = fwm_band_point(auto_cfg, None, pump_gain)

            for buffer_gain in BUFFER_GAINS:
                step += 1
                # Close the previous point's windows by number (plt.figure(num) re-raises them).
                for fignum in open_figures:
                    plt.close(fignum)
                open_figures = []
                print(
                    f"--- {step}/{total}: pump {pump_frequency:.3f} MHz "
                    f"gain {pump_gain:.4f}, buffer gain {buffer_gain:.4f}"
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


                # Periodic look at the distributions. plot_point draws from the
                # stored histograms, so this is the same figure a rejected
                # attempt gets and the same one fwm_qc --replot-rejects makes.
                if (args.plot_every and args.plots != "all"
                        and step % args.plot_every == 0):
                    fig = fwm_qc.plot_point(
                        result["centers"], result["hist"],
                        title=(f"#{step - 1:05d}  "
                               + f"pump {pump_gain:.4f}, buffer {buffer_gain:.4f}"))
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
        rows, info = fwm_qc.reduce_to_nep(
            accepted, calib, t1=float(auto_cfg.device.qubit.T1[qi]),
            cycle_us=cycle_us,
        )
        summary_path = metadata_path / "qc_summary.csv"
        columns = ["gain_p", "n_points", "eta_e", "eta_err_e", "eta_f",
                   "eta_err_f", "p_dark", "n_equiv", "nep"]
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nNEP summary (indicative; thresholded fractions) -> {summary_path}")
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
        print(f"  duty cycle {expt.length_b / cycle_us:.3f} "
              f"({expt.length_b:.1f} us armed of {cycle_us:.1f} us)")
        print(f"  qubit efficiency {info['eta_qubit']:.3f}, "
              f"ceiling kappa_c/kappa * eta_qubit = {info['eta_ceiling']:.3f}")
        for row in rows:
            print(f"  pump {row['gain_p']:.4f}: eta {row['eta_e']:.4f}, "
                  f"dark {row['p_dark']:.4f}, n_equiv {row['n_equiv']:.3e}, "
                  f"NEP {row['nep']:.3e} W/sqrt(Hz)")

        # fig.savefig, not save_fig: that calls tight_layout() and a blocking plt.show().
        fig = fwm_qc.plot_nep_summary(rows, info)
        figure_path = metadata_path / "qc_nep_summary.png"
        fig.savefig(figure_path)
        print(f"  figure -> {figure_path}")

    if not args.no_show:
        wait_for_plots()
