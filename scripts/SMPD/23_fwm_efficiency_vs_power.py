"""Run the full FWM histogram suite over a grid of buffer gains and pump powers."""

from pathlib import Path

import matplotlib.pyplot as plt
import slab_qick_calib.experiments as meas
import numpy as np

from common import (
    finish_experiment,
    fwm_band_point,
    parse_args,
    prepare_experiment,
    suppress_duplicate_config,
    wait_for_plots,
)


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    buffer_line_qi = 0
    JPA_ON = True
    RESET_MODE = "active"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"

    # The grid. Pump powers are gains; the frequency at each comes from
    # device.fwm.pump_power_fit, so every point sits on the band.
    BUFFER_GAINS = np.linspace(0,.03,13)
    PUMP_GAINS = np.linspace(0.1,1.0,10)
    print(f"Buffer gains: {BUFFER_GAINS}")
    print(PUMP_GAINS)
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(0,)
    )

    buffer_frequency = auto_cfg.device.readout.frequency[buffer_line_qi]

    # Shared by every point; each acquires the five SMPDHistogramExperiment.SCANS.
    params = {
        "shots": 20000,
        "frequency_b": buffer_frequency,
        "length_b": 40,
        "delay_b": 0,
        "length_p": 40,
        "check_f": True,
        "sep_readout": True,
        "reset_mode": RESET_MODE,
    }

    total = len(PUMP_GAINS) * len(BUFFER_GAINS)
    print(
        f"{total} histogram suites: {len(PUMP_GAINS)} pump powers x "
        f"{len(BUFFER_GAINS)} buffer gains, 5 sequences of "
        f"{params['shots']} shots each"
    )

    histograms = []
    open_figures = []
    step = 0
    for pump_gain in PUMP_GAINS:
        # The band frequency for this pump power.
        pump_frequency, pump_gain = fwm_band_point(auto_cfg, None, pump_gain)

        for buffer_gain in BUFFER_GAINS:
            step += 1
            # Close the previous point's window now that the next plot is about
            # to replace it. Its PNG is already written, so nothing is lost.
            for fignum in open_figures:
                plt.close(fignum)

            print(
                f"--- {step}/{total}: pump {pump_frequency:.3f} MHz "
                f"gain {pump_gain:.4f}, buffer gain {buffer_gain:.4f}"
            )

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
                    "shots": 30000
                },
            )
            # One config yml per run folder, not per point (CLAUDE.md "One config yml per run").
            suppress_duplicate_config(histogram, Path(cfg_dict["metadata_path"]))
            finish_experiment(histogram, args, wait=False)
            histograms.append(histogram)
            open_figures = plt.get_fignums()

    print(f"\nSaved {len(histograms)} histogram suites to {run_dir}")
    for index, histogram in enumerate(histograms, start=1):
        expt = histogram.cfg.expt
        fids = histogram.data.get("fids", [])
        fid = f"g-e fidelity {fids[0]:.4f}" if len(fids) else "no fidelity"
        print(
            f"  {index:3d}: pump gain {expt.gain_p:.4f}, "
            f"buffer gain {expt.gain_b:.4f} -> {fid}"
        )

    if not args.no_show:
        wait_for_plots()
