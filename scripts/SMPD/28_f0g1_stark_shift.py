"""Measure the ac Stark shift of the f0-g1 transition versus drive power.

Step 1b of the f0-g1 reset calibration; writes device.f0g1.stark_fit.
See scripts/SMPD/CLAUDE.md, "Runner notes, scripts 27–36".
"""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    params = {
        "span": 10,
        "expts": 100,
        "start_gain": 0.02,
        "max_gain": 0.5,
        "expts_gain": 20,
        "fixed_area": True,
        "gain_f0g1": 0.1,
        "length_f0g1": 2.0,
        "prep": "f",
        "reps": 500,
        "reset_mode": RESET_MODE,
    }
    stark = meas.F0G1SpecPower(
        cfg_dict, qi=qi, go=False, style="medium", params=params
    )
    finish_experiment(stark, args)
    if not args.no_update_config:
        stark.update()
