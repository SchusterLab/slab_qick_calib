"""Calibrate buffer photon number from Ramsey dephasing vs buffer drive power."""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    buffer_line_qi = 0
    JPA_ON = True
    RESET_MODE = "f0g1"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(buffer_line_qi,)
    )
    # frequency_b defaults to f_g = readout.frequency[0] (where Stark/FWM drive); kappa_b, chi_b from the config;
    # CLAUDE.md "Photon number from Ramsey dephasing (scripts 38, 41)"
    params = {
        "max_gain_b": 0.03,  # dephasing at f_g is ~8x the midpoint's per gain^2
        "gain_b_expts": 10,
        "reps": 20000,
        "reset_mode": RESET_MODE,
    }
    cal = meas.T2PhotonCal(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(cal, args)
    if not args.no_update_config:
        cal.update()
