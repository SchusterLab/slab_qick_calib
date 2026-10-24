"""Run e-f Ramsey for one qubit and set f_ef to the center of its two parity lines."""

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    # The e-f line is split in two by charge parity (~280 kHz on run 34), so
    # fit_two_lines sets new_freq to the center of the two lines. ramsey_freq
    # must exceed half the splitting for the upper line's shift to be right.
    # span/ramsey_freq are set explicitly: the defaults come from the g-e T2r.
    params = {"checkEF": True, "fit_two_lines": True, "span": 20, "ramsey_freq": 0.3,
              "reps": 3000, "reset_mode": RESET_MODE}
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
    t2_ef = meas.T2Experiment(cfg_dict, qi=qi, go=False, max_err=10, params=params)
    finish_experiment(t2_ef, args)
    data = t2_ef.data
    print(f"lines: {data['f_lines'][0]:.4f}, {data['f_lines'][1]:.4f} MHz, "
          f"splitting {data['f_splitting'] * 1e3:.1f} kHz")
    print(f"center: {data['new_freq']:.4f} MHz +- {data['f_center_err'] * 1e3:.1f} kHz")
    if not args.no_update_config and t2_ef.status and data["f_center_err"] < 0.005:
        config.update_qubit(str(cfg_path), "f_ef", data["new_freq"], qi)
