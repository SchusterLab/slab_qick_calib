"""Run echo T2 calibration for one qubit."""

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
    params = {"experiment_type": "echo", "reset_mode": RESET_MODE}
    t2_echo = meas.T2Experiment(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(t2_echo, args)
    if not args.no_update_config and t2_echo.status:
        config.update_qubit(str(cfg_path), "T2e", t2_echo.data["best_fit"][3], qi)