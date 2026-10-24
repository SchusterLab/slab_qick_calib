"""Run single-shot histogram calibration for one qubit."""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    # Calibrate in the mode the measurements will use: CLAUDE.md "Reset modes".
    RESET_MODE = "f0g1"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
    params = {
        "shots": 30000,
        "reset_mode": RESET_MODE,
        # 'check_e': False
    }
    single_shot = meas.HistogramExperiment(
        cfg_dict, qi=qi, go=False, check_f=True, params=params
    )
    finish_experiment(single_shot, args)
    if not args.no_update_config:
        single_shot.update()
