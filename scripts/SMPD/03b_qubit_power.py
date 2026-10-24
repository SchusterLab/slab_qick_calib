"""Run qubit power spectroscopy for one qubit."""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    # Wide span to catch the Stark pull, the two-photon g-f line and f_ef; log gain axis
    # max_gain/rng .. max_gain. Keep log=True: log=False needs start_gain/step_gain passed.
    params = {"span": 120, "expts": 200, "reps": 500,
              "rng": 50, "max_gain": 1, "expts_gain": 10, "log": True,
              "length": 100, "readout_length": 4, "sep_readout": True,
              "reset_mode": RESET_MODE}
    qspec_power = meas.QubitSpecPower(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(qspec_power, args)
