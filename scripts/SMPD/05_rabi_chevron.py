"""Run length Rabi chevron calibration for one qubit."""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
    params = {"sweep": "length", "pulse_type": "gauss", "sigma": 0.05,
              "gain": 0.25, "span_f": 50, "loop": True,
              "reset_mode": RESET_MODE}
    rabi_chevron = meas.RabiChevronExperiment(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(rabi_chevron, args)