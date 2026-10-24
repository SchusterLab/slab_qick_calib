"""The reset measurement: residual excitation versus reset time.

Step 5 of the f0-g1 reset protocol; needs 27-34 run with update().
See scripts/SMPD/CLAUDE.md, "Runner notes, scripts 27–36".
"""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    # Passive on purpose: the reset under test must not run between shots, and "active" starts
    # 5x dirtier (CLAUDE.md "The reset measurement (script 35)").
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON, additional_readout_qis=(0,))

    params = {
        # Leave gain_ef out to take the calibrated optimum for this gain_f0g1.
        "gain_f0g1": 0.74,
        'gain_ef': 0.075,
        "max_length": 4.0,
        "expts": 50,
        "reps": 3000,
        "preps": ["g","e",'f'],
        "readout_gap": 2,
        # "ramp_sigma_f0g1": 0.2,
        # "ramp_sigma_inc_f0g1": 3,
        # "ramp_sigma_ef": 0.2,
        # "ramp_sigma_inc_ef": 3,
        # "frequency_f0g1":131.0,
        # Same-timing baseline: rerun these preps with both drives at zero; "g" separates
        # drive-induced excitation from misassignment. One acquisition each.
        "baseline_preps": ["g"],
        "reset_mode": RESET_MODE,
    }
    reset_scan = meas.ResetTimeScan(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(reset_scan, args)
