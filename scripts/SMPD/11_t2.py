"""Run Ramsey T2 calibration for one qubit."""

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
    params = {"reps": 500, "reset_mode": RESET_MODE}
    t2 = meas.T2Experiment(cfg_dict, qi=qi, go=False, max_err=10, params=params)
    finish_experiment(t2, args)
    if not args.no_update_config and t2.status:
        config.update_qubit(str(cfg_path), "f_ge", t2.data["new_freq"], qi)
        config.update_qubit(str(cfg_path), "T2r", t2.data["best_fit"][3], qi)