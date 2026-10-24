"""f0-g1 coupling versus drive power: g~(V_f0g1) and kappa.

Step 4b of the f0-g1 reset calibration; writes device.f0g1.g_tilde_fit and kappa.
See scripts/SMPD/CLAUDE.md, "Runner notes, scripts 27–36".
"""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "active"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    params = {
        "start_gain": 0.1,
        "max_gain": 0.5,
        "expts_gain": 12,
        "max_length": 3.0,
        "expts": 100,
        "reps": 3000,
        "reset_mode": RESET_MODE,
    }
    f0g1_rabi_power = meas.F0G1RabiPower(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(f0g1_rabi_power, args)
    if not args.no_update_config:
        f0g1_rabi_power.update()
