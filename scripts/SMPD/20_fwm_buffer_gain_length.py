"""Run FWM buffer-gain versus buffer-length spectroscopy."""

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
    JPA_ON = True
    RESET_MODE = "active"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(0,)
    )

    buffer_frequency = auto_cfg.device.readout.frequency[buffer_line_qi]
    pump_frequency = 5210

    # Pump gain comes from the FWM band parabola fit by 18_fwm_power.py.
    pump_gain = fwm_gain_for_frequency(auto_cfg, pump_frequency)

    # Editable buffer gain/length settings. The pump frequency matches
    # FWM buffer spectroscopy; the buffer frequency follows the YAML line.
    params = {
        "gain_p": pump_gain,
        "frequency_p": pump_frequency,
        "length_p": 50,
        "start": 0.001,
        "span": 0.2,
        "expts": 50,
        "frequency_b": buffer_frequency,
        "reps": 1000,
        "length_b_start": 0.1,
        "length_b_end": 50,
        "length_b_expts": 50,
        "save_shots": False,
        "reset_mode": RESET_MODE,
    }
    four_wm = meas.FWM_gainb_tb_sweep(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(four_wm, args)
