"""Run FWM buffer-frequency and buffer-gain spectroscopy."""

import slab_qick_calib.experiments as meas

from common import (
    finish_experiment,
    fwm_gain_for_frequency,
    parse_args,
    prepare_experiment,
)


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    buffer_line_qi = 0
    waste_line_qi = qi
    JPA_ON = True
    RESET_MODE = "active"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(0,)
    )

    buffer_frequency = auto_cfg.device.readout.frequency[buffer_line_qi]
    center_pump_freq = 5214
    pump_start = center_pump_freq - 3
    pump_stop = center_pump_freq + 6

    # Pump gain comes from the FWM band parabola fit by 18_fwm_power.py,
    # evaluated at the center pump frequency of this scan.
    pump_gain = fwm_gain_for_frequency(auto_cfg, center_pump_freq)

    print(
        f"scanning {pump_start:.3f} to {pump_stop:.3f} MHz"
    )

    # Editable buffer-sweep settings. The buffer frequency is centered on
    # the configured buffer readout line; the pump scans the FWM window.
    params = {
        "reps": 1200,
        "gain_p": pump_gain,
        "length_p": 30,
        "start": pump_start,
        "span": pump_stop - pump_start,
        "expts": 30,
        "center_freq_b": buffer_frequency,
        "expts_b": 30,
        "gain_b": 0.1,
        "span_b": 2,
        "length_b": 30,
        "delay_b": 0,
        "sep_readout": True,
        "reset_mode": RESET_MODE,
    }
    four_wm = meas.QubitSpecShotBuffer(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(four_wm, args)
