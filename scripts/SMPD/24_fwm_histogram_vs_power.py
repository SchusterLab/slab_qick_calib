"""Run FWM histograms at several pump powers along the band fit."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import slab_qick_calib.experiments as meas

from common import (
    finish_experiment,
    fwm_frequency_for_gain,
    fwm_gain_for_frequency,
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
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(0,)
    )

    buffer_frequency = auto_cfg.device.readout.frequency[buffer_line_qi]
    fit_type = auto_cfg.device.fwm.pump_power_fit_type

    # Equal steps in frequency along the fitted band (CLAUDE.md "Histogram vs pump power (script 24)").
    GAIN_MIN, GAIN_MAX = 0.05, 1.0
    N_POINTS = 10

    # Ordered from GAIN_MIN to GAIN_MAX, i.e. increasing pump power.
    pump_frequencies = np.linspace(
        fwm_frequency_for_gain(auto_cfg, GAIN_MIN, verbose=False),
        fwm_frequency_for_gain(auto_cfg, GAIN_MAX, verbose=False),
        N_POINTS,
    )

    print(
        f"Walking the {fit_type} band in {N_POINTS} equal frequency steps "
        f"from {pump_frequencies[0]:.3f} to {pump_frequencies[-1]:.3f} MHz"
    )

    # Editable FWM histogram settings, shared by every pump power. Each run
    # acquires all four sequences (see SMPDHistogramExperiment.SCANS).
    params = {
        "shots": 20000,
        "frequency_b": buffer_frequency,
        "gain_b": 0.1,
        "length_b": 40,
        "delay_b": 0,
        "length_p": 40,
        "sep_readout": True,
        "reset_mode": RESET_MODE,
    }

    histograms = []
    open_figures = []
    for step, pump_frequency in enumerate(pump_frequencies, start=1):
        # Close the previous step's window by number (plt.figure() re-raises it).
        for fignum in open_figures:
            plt.close(fignum)

        pump_gain = fwm_gain_for_frequency(auto_cfg, pump_frequency)
        print(
            f"--- step {step}/{N_POINTS}: pump {pump_frequency:.3f} MHz, "
            f"gain {pump_gain:.4f}"
        )

        # Built inside the loop: the filename is the next unused index in the run folder.
        histogram = meas.SMPDHistogramExperiment(
            cfg_dict,
            qi=qi,
            go=False,
            params={**params, "frequency_p": pump_frequency, "gain_p": pump_gain},
        )
        # wait=False so the loop keeps moving; the last figure is blocked on
        # once at the end instead of after every step.
        # One config yml per run folder, not per point (CLAUDE.md "One config yml per run").
        suppress_duplicate_config(histogram, Path(cfg_dict["metadata_path"]))
        finish_experiment(histogram, args, wait=False)
        histograms.append(histogram)
        open_figures = plt.get_fignums()

    print(f"\nSaved {len(histograms)} histograms to {run_dir}")
    for step, histogram in enumerate(histograms, start=1):
        fids = histogram.data.get("fids", [])
        fid = f"g-e fidelity {fids[0]:.4f}" if len(fids) else "no fidelity"
        print(
            f"  step {step}: pump {histogram.cfg.expt.frequency_p:.3f} MHz, "
            f"gain {histogram.cfg.expt.gain_p:.4f} -> {fid}"
        )

    # Pump+buffer histogram at every pump power, coloured by gain, plus buffer-only.
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    cmap = plt.get_cmap("viridis")
    measured = [h for h in histograms if "histe" in h.data]
    if measured:
        gains = [h.cfg.expt.gain_p for h in measured]
        norm = plt.Normalize(vmin=min(gains), vmax=max(gains))
        for histogram in measured:
            ax.plot(
                histogram.data["vhe"],
                histogram.data["histe"],
                color=cmap(norm(histogram.cfg.expt.gain_p)),
                lw=1,
            )
        # One buffer-only trace, to show where the undriven distribution sits.
        reference = measured[0]
        if "histg" in reference.data:
            ax.plot(
                reference.data["vhg"],
                reference.data["histg"],
                color="0.4",
                ls="--",
                lw=1.2,
                label="buffer only (no pump)",
            )
            ax.legend(fontsize="small")
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            label="pump gain (DAC level)",
        )
    else:
        print("No histograms available, so no comparison figure")
    ax.set_xlabel("I (ADC levels)")
    ax.set_ylabel("Probability density")
    fig.tight_layout()

    summary_path = Path(cfg_dict["metadata_path"]) / "histograms_vs_pump_power.png"
    fig.savefig(summary_path)
    print(f"Saved the pump-power comparison to {summary_path}")

    if not args.no_show:
        wait_for_plots()
