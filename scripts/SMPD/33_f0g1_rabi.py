"""f0-g1 Rabi: sweep the drive duration with the qubit in |f>, at one amplitude.

Step 4 of the f0-g1 reset calibration; sets 34's window, writes nothing.
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
        # frequency_f0g1 is left out so it comes from device.f0g1.stark_fit.
        "gain_f0g1": 0.05,
        "max_length": 5,
        "expts": 100,
        "reps": 3000,
        # "f" = the Rabi (step 4); "g" turns it into a heating scan.
        "prep": "f",
        # Drive end to readout; default 3/kappa. Long gaps expose the deposit (run 33: |g> floor 4.9% -> 9.0%).
        "readout_gap": 5,
        "reset_mode": RESET_MODE,
    }
    f0g1_rabi = meas.F0G1Rabi(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(f0g1_rabi, args)
