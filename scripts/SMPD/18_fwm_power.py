"""Run FWM pump-frequency and pump-gain spectroscopy."""

from pathlib import Path

import numpy as np
import slab_qick_calib.experiments as meas
from slab_qick_calib.experiments.single_qubit.qubit_spec_shot import (
    BAND_FIT_TYPES,
    expected_fwm_frequency as expected_fwm_frequency_for,
)

from common import (
    REPO_DIR,
    SCRIPT_DIR,
    finish_experiment,
    nonblocking_plots,
    parse_args,
    prepare_experiment,
    wait_for_plots,
)


def add_arguments(parser):
    parser.add_argument(
        "--reanalyze",
        metavar="N_OR_PATH",
        help="redo only the analysis of a saved run, no hardware: the #N in "
        "the data folder name, or a path to its .h5",
    )
    parser.add_argument(
        "--exclude-freq",
        metavar="MHZ",
        type=float,
        nargs="+",
        default=None,
        help="--reanalyze only: drop the pump frequency point(s) nearest "
        "these values (MHz) before re-running the analysis, e.g. a stray "
        "artifact that is pulling the band fit off",
    )
    parser.add_argument(
        "--fit-type",
        choices=("auto",) + BAND_FIT_TYPES,
        default=None,
        help="--reanalyze only: force the band model instead of the "
        "script's PUMP_POWER_FIT_TYPE constant",
    )


def find_saved_run(args):
    """The h5 and metadata folder of a previous run of this script."""
    run_dir = SCRIPT_DIR / "Data" / f"run{args.run_number}_{args.device_name}"
    target = Path(args.reanalyze)
    if target.suffix == ".h5":
        h5_path = target
    else:
        folders = sorted((run_dir / "data").glob(f"#{args.reanalyze}_*"))
        if len(folders) != 1:
            raise SystemExit(
                f"Expected one folder #{args.reanalyze}_* under "
                f"{run_dir / 'data'}, found {[f.name for f in folders]}"
            )
        files = sorted(folders[0].glob("*.h5"))
        files = [f for f in files if not f.stem.endswith("_reanalyzed")]
        if len(files) != 1:
            raise SystemExit(
                f"Expected one .h5 in {folders[0]}, found {[f.name for f in files]}"
            )
        h5_path = files[0]
    metadata_path = h5_path.parent.parent.parent / "metadata_and_images" / h5_path.parent.name
    return h5_path, metadata_path


# 2D (gain, frequency) arrays in the saved data; explicit because hist/bin_centers
# can share a length with xpts.
FREQ_AXIS_KEYS = (
    "amps", "avgi", "avgq", "phases", "p_e", "scale_data",
    "amps_with_buffer", "avgi_with_buffer", "avgq_with_buffer",
    "phases_with_buffer", "p_e_with_buffer",
    "amps_no_buffer", "avgi_no_buffer", "avgq_no_buffer",
    "phases_no_buffer", "p_e_no_buffer",
)


def drop_frequency_points(data, freqs):
    """
    Remove the frequency point(s) nearest `freqs` from a saved 2D scan.

    Slices every (gain, frequency) array in lockstep with `xpts`, so the
    excluded point is genuinely gone from the analysis rather than just
    masked out of the band fit -- the per-row Lorentzian re-fit in
    `super().analyze()` won't see it either.
    """
    xpts = data["xpts"]
    drop = sorted({int(np.argmin(np.abs(xpts - f))) for f in freqs})
    keep = np.ones(len(xpts), dtype=bool)
    keep[drop] = False
    for f, idx in zip(freqs, drop):
        print(f"  dropping xpts[{idx}] = {xpts[idx]:.3f} MHz "
              f"(nearest to requested {f:.3f} MHz)")
    for key in FREQ_AXIS_KEYS:
        if key in data:
            data[key] = data[key][:, keep]
    data["xpts"] = xpts[keep]
    return data


def reanalyze(args, pump_power_fit_type, fwm_guess_window, exclude_freqs=None):
    h5_path, metadata_path = find_saved_run(args)
    print(f"Reanalyzing {h5_path}")
    four_wm_power = meas.FourWMPowerExperiment.from_file(
        h5_path, REPO_DIR / "configs" / args.config, metadata_path
    )
    four_wm_power.cfg.expt.pump_power_fit_type = pump_power_fit_type
    four_wm_power.cfg.expt.fwm_guess_window = fwm_guess_window
    if exclude_freqs:
        drop_frequency_points(four_wm_power.data, exclude_freqs)
    four_wm_power.analyze()
    with nonblocking_plots():
        four_wm_power.display()
    if not args.no_show:
        wait_for_plots()
    return four_wm_power


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    buffer_line_qi = 0
    waste_line_qi = qi
    JPA_ON = True
    RESET_MODE = "active"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    # "auto" | "quadratic" | "linear" | "linear_quadratic" | "cubic" | "quartic"; CLAUDE.md "FWM analysis → config"
    PUMP_POWER_FIT_TYPE = "auto"
    # The fit is seeded by the best curve leaving the expected FWM frequency at
    # zero gain; its start may sit up to this far (MHz) from that expectation.
    FWM_GUESS_WINDOW = 10.0
    args = parse_args(__doc__, add_arguments, config=CONFIG)

    if args.reanalyze:
        four_wm_power = reanalyze(
            args, args.fit_type or PUMP_POWER_FIT_TYPE, FWM_GUESS_WINDOW,
            args.exclude_freq,
        )
        if not args.no_update_config:
            four_wm_power.update()
        raise SystemExit

    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(0,)
    )

    buffer_frequency = auto_cfg.device.readout.frequency[buffer_line_qi]
    # f_ge + f_waste - chi_waste - f_buffer; also the band fit's anchor.
    expected_fwm_frequency = expected_fwm_frequency_for(
        auto_cfg, waste_line_qi, buffer_line_qi
    )

    scan_start = expected_fwm_frequency - 100
    scan_start = 5200
    scan_stop = expected_fwm_frequency + 100
    scan_stop = 5230

    print(
        f"Expected FWM frequency: {expected_fwm_frequency:.3f} MHz; "
        f"scanning {scan_start:.3f} to {scan_stop:.3f} MHz"
    )

    # Buffer frequency from the config's buffer line; gain_b/length_b set directly.
    params = {
        "span": scan_stop - scan_start,
        "expts": 100,
        "reps": 500,
        "length_p": 30,
        "start": scan_start,
        "frequency_b": buffer_frequency,
        "gain_b": 0.02,
        "length_b": 30,
        "sep_readout": True,
        "start_gain": 0.000,
        "step_gain": 0.02,
        "expts_gain": 50,
        "log": False,
        "diff_buffer": True,
        "pump_power_fit_type": PUMP_POWER_FIT_TYPE,
        "fwm_guess": expected_fwm_frequency,
        "fwm_guess_window": FWM_GUESS_WINDOW,
        "reset_mode": RESET_MODE,
    }
    four_wm_power = meas.FourWMPowerExperiment(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(four_wm_power, args)
    if not args.no_update_config:
        four_wm_power.update()
