"""Run the FWM histogram suite over a buffer-frequency x buffer-gain grid at one pump power, retaking any point whose scans look wrong."""

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


# Attempts per grid point; see script 26.
MAX_ATTEMPTS = 10


def add_arguments(parser):
    parser.add_argument(
        "--plots", choices=("none", "rejects", "all"), default="none",
        help="none (default): only the end-of-run summary. rejects: also a "
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
    RESET_MODE = "f0g1"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"

    # One pump power for the whole sweep (frequency from the band fit); the buffer moves.
    PUMP_GAIN = 0.5

    # Offsets from device.readout.frequency[buffer_line_qi] (+-1 MHz ~ +-4.5 kappa);
    # buffer gains match script 26 so the efficiencies are comparable.
    FREQUENCY_OFFSETS = np.linspace(-0.25, 0.25, 11)
    BUFFER_GAINS = np.linspace(0, 0.01, 9)

    args = parse_args(__doc__, add_arguments, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(0,)
    )

    buffer_frequency = auto_cfg.device.readout.frequency[buffer_line_qi]
    buffer_frequencies = buffer_frequency + FREQUENCY_OFFSETS
    pump_frequency, pump_gain = fwm_band_point(auto_cfg, None, PUMP_GAIN)
    print(f"Buffer frequencies: {buffer_frequencies}")
    print(f"Buffer gains: {BUFFER_GAINS}")

    params = {
        "shots": 30000,
        "length_b": 50,
        "delay_b": 0,
        "length_p": 50,
        "frequency_p": pump_frequency,
        "gain_p": pump_gain,
        "check_f": True,
        # No d: the pump is fixed here, so the off-band control would
        # be one repeated number (CLAUDE.md "The detuned-pump scan").
        "scans": ("g", "e", "p"),
        "sep_readout": True,
        "reset_mode": RESET_MODE,
        # Shorter than the config's safe default; only safe in 26/37,
        # CLAUDE.md "Gap and ring-down in scripts 26/37".
        "f0g1_reset_gap": 0.0,
        "extra_delay": 1.0 - 0.1,
    }

    # Same reference frame and gates as script 26.
    reference = fwm_qc.reference_from_config(auto_cfg, qi, params["shots"])
    edges, centers = fwm_qc.bin_grid(reference)
    monitor = fwm_qc.QCMonitor(reference, edges)
    print(f"QC reference: {reference}")

    metadata_path = Path(cfg_dict["metadata_path"])
    log_path = metadata_path / "qc_log.csv"
    rejects_path = metadata_path / "qc_rejects.h5"

    total = len(buffer_frequencies) * len(BUFFER_GAINS)
    print(
        f"{total} histogram suites: {len(buffer_frequencies)} buffer "
        f"frequencies x {len(BUFFER_GAINS)} buffer gains at pump gain "
        f"{pump_gain:.4f}, 5 sequences of {params['shots']} shots each, up to "
        f"{MAX_ATTEMPTS} attempts per point"
    )
    print(f"Plots: {args.plots}"
          + ("" if args.plots != "none" else
             " (the histograms of any rejected attempt are still saved; "
             "replot them later with fwm_qc.py --replot-rejects)"))

    # fwm_qc's columns plus the buffer frequency, the one thing it does not
    # log because script 26 never varies it.
    log_columns = list(fwm_qc.LOG_COLUMNS)
    log_columns.insert(log_columns.index("gain_b") + 1, "frequency_b")
    log_file = log_path.open("w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=log_columns)
    log_writer.writeheader()

    # Ground-only health check, built once and re-acquired; never saved
    # (CLAUDE.md "Ground-state recovery").
    ground_check = fwm_qc.make_ground_check(cfg_dict, qi, params)
    ground_columns = list(fwm_qc.GROUND_WAIT_COLUMNS)
    ground_columns.insert(2, "frequency_b")
    ground_file = (metadata_path / "ground_waits.csv").open("w", newline="")
    ground_writer = csv.DictWriter(ground_file, fieldnames=ground_columns)
    ground_writer.writeheader()

    # Do not start on an unhealthy qubit -- it would only burn retakes on every
    # point (CLAUDE.md "Ground-state recovery"). One pass is enough to start; a
    # mid-sweep failure needs GROUND_CONSECUTIVE in a row.
    start_wait = fwm_qc.wait_for_ground(ground_check, reference, consecutive=1)
    print(f"Ground state before the sweep: {start_wait['p_e_final']:.4f} "
          f"(bar {fwm_qc.GROUND_MAX_PE}, {start_wait['checks']} check(s))")
    if start_wait["failures"]:
        ground_writer.writerow({"index": -1, "attempt": 0,
                             "frequency_b": "", **start_wait})
        ground_file.flush()

    accepted = []
    tally = {"first_try": 0, "retaken": 0, "failed_skipped": 0}
    step = 0
    open_figures = []
    try:
        for frequency_b in buffer_frequencies:
            for buffer_gain in BUFFER_GAINS:
                step += 1
                for fignum in open_figures:
                    plt.close(fignum)
                open_figures = []
                print(
                    f"--- {step}/{total}: buffer {frequency_b:.4f} MHz "
                    f"({frequency_b - buffer_frequency:+.4f}), "
                    f"gain {buffer_gain:.4f}"
                )

                ok = False
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    # Built inside the loop so each attempt claims the next
                    # unused file index; see script 26.
                    histogram = meas.SMPDHistogramExperiment(
                        cfg_dict,
                        qi=qi,
                        go=False,
                        params={
                            **params,
                            "frequency_b": frequency_b,
                            "gain_b": buffer_gain,
                        },
                    )
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
                    log_writer.writerow({
                        **fwm_qc.log_row(
                            step - 1, attempt, pump_gain, buffer_gain, status,
                            failed, result, histogram.data,
                        ),
                        "frequency_b": f"{frequency_b:.6f}",
                    })
                    log_file.flush()

                    if ok:
                        tally["first_try" if attempt == 1 else "retaken"] += 1
                        break

                    fwm_qc.record_reject(
                        rejects_path, step - 1, attempt, pump_gain, buffer_gain,
                        failed, result, frequency_b=frequency_b,
                    )
                    if args.plots in ("rejects", "all"):
                        fig = fwm_qc.plot_point(
                            result["centers"], result["hist"],
                            title=(f"#{step - 1:05d} attempt {attempt} -- "
                                   f"buffer {frequency_b:.4f} MHz, gain "
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
                        ground_writer.writerow({
                            "index": step - 1, "attempt": attempt,
                            "frequency_b": f"{frequency_b:.6f}", **wait})
                        ground_file.flush()
                    if last:
                        tally["failed_skipped"] += 1
                        print(f"  !! skipped after {MAX_ATTEMPTS} failed "
                              f"attempts: {'; '.join(failed)}")
                    else:
                        print(f"  retake {attempt}/{MAX_ATTEMPTS}: "
                              f"{'; '.join(failed)}")

                # Skipped, not kept, for the reasons given in script 26.
                if not ok:
                    continue

                histogram.save_data(histogram.data)
                if args.plots == "all":
                    with nonblocking_plots():
                        histogram.display(histogram.data)
                    if args.no_show:
                        plt.close("all")
                    else:
                        open_figures = plt.get_fignums()
                else:
                    histogram.save_config()


                # Periodic look at the distributions. plot_point draws from the
                # stored histograms, so this is the same figure a rejected
                # attempt gets and the same one fwm_qc --replot-rejects makes.
                if (args.plot_every and args.plots != "all"
                        and step % args.plot_every == 0):
                    fig = fwm_qc.plot_point(
                        result["centers"], result["hist"],
                        title=(f"#{step - 1:05d}  "
                               + f"{frequency_b:.3f} MHz, gain {buffer_gain:.4f}"))
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

    # Ramsey photon calibration rescaled as fit_shared_means.py does; indicative only,
    # quote fit_shared_means.py (CLAUDE.md "Photon number from Ramsey dephasing (scripts 38, 41)").
    soc_cfg = histogram.cfg.hw.soc
    board = soc_cfg.rfboard_active.dac_readout[
        str(soc_cfg.dacs.readout.ch[buffer_line_qi])]
    buffer_attn_db = float(board.attn1) + float(board.attn2)
    try:
        calib = fwm_qc.load_calibration(
            auto_cfg, source="ramsey_spec", attn_db=buffer_attn_db,
            pulse=histogram.cfg.expt.pulse_type_b,
        )
    except ValueError as error:
        print(f"\nNo efficiency summary: {error}")
    else:
        if not accepted:
            raise SystemExit(
                "No point passed QC, so there is nothing to reduce. The "
                f"per-attempt reasons are in {log_path}."
            )
        print(f"\nPhoton calibration: Ramsey, x{calib['scale']:.4g} for "
              f"{buffer_attn_db:g} dB / {histogram.cfg.expt.pulse_type_b} -> "
              f"{calib['coeffs']} ({calib['fit_type']})")
        expt = histogram.cfg.expt
        cycle_us = (max(expt.length_p, expt.delay_b + expt.length_b)
                    + expt.readout_length + effective_final_delay(expt))
        # The on-resonance gain -> n_bar calibration is a flux into the line, so it holds at
        # every frequency: the efficiency is per photon arriving.
        rows, info = fwm_qc.reduce_to_nep(
            accepted, calib, t1=float(auto_cfg.device.qubit.T1[qi]),
            cycle_us=cycle_us, by="frequency_b",
        )
        summary_path = metadata_path / "qc_summary.csv"
        columns = ["frequency_b", "n_points", "eta_e", "eta_err_e", "eta_f",
                   "eta_err_f", "p_dark", "n_equiv", "nep"]
        with summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Efficiency summary (indicative; thresholded fractions) -> "
              f"{summary_path}")
        print(f"  pump {pump_frequency:.3f} MHz gain {pump_gain:.4f}")
        print(f"  qubit efficiency {info['eta_qubit']:.3f}, "
              f"ceiling kappa_c/kappa * eta_qubit = {info['eta_ceiling']:.3f}")
        for row in rows:
            print(f"  buffer {row['frequency_b']:.4f} MHz "
                  f"({row['frequency_b'] - buffer_frequency:+.4f}): "
                  f"eta {row['eta_e']:.4f} +- {row['eta_err_e']:.4f}, "
                  f"dark {row['p_dark']:.4f}, NEP {row['nep']:.3e} W/sqrt(Hz)")
        etas = np.array([row["eta_e"] for row in rows])
        if np.isfinite(etas).any():
            best = rows[int(np.nanargmax(etas))]
            print(f"  peak efficiency {best['eta_e']:.4f} at "
                  f"{best['frequency_b']:.4f} MHz "
                  f"({best['frequency_b'] - buffer_frequency:+.4f} from config)")

        fig = fwm_qc.plot_nep_summary(rows, info, by="frequency_b",
                                      xlabel="buffer frequency (MHz)")
        figure_path = metadata_path / "qc_efficiency_vs_frequency.png"
        fig.savefig(figure_path)
        print(f"  figure -> {figure_path}")
        print(f"  fitted (authoritative) efficiency: python "
              f"{metadata_path / 'fit_shared_means.py'}")

    if not args.no_show:
        wait_for_plots()
