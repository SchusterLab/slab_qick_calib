"""Find the f0-g1 transition: sweep the reset-drive frequency with the qubit in |f>.

Step 1 of the f0-g1 reset calibration (Magnard et al., arXiv:1801.07689).
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
        "span":30,
        "gain_f0g1": 0.1,
        "length_f0g1": 2.0,
        "prep": "f",
        "reps": 500,
        "reset_mode": RESET_MODE,
    }
    f0g1_spec = meas.F0G1Spec(
        cfg_dict, qi=qi, go=False, style="coarse", params=params
    )
    finish_experiment(f0g1_spec, args)
    if not args.no_update_config and f0g1_spec.status:
        f0g1_spec.update()
