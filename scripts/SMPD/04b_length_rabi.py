"""Run length Rabi calibration for one qubit."""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
    gain = auto_cfg["device"]["qubit"]["pulses"]["pi_ge"]["gain"][qi]
    params = {"sweep": "length", "pulse_type": "const", "length": 0.05,
              "gain": gain, "reps": 1000, "expts": 50,
              "reset_mode": RESET_MODE}
    length_rabi = meas.RabiExperiment(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(length_rabi, args)