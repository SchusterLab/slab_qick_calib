"""e-f ac Stark shift versus f0-g1 drive power.

Step 3b of the f0-g1 reset calibration; writes device.f0g1.ef_stark_fit.
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
        "start_gain": 0.02,
        "max_gain": 0.5,
        "expts_gain": 20,
        # The operating Omega_ef = 0.539 MHz; see the note above before
        # changing this, it sets both pulse lengths.
        "gain_ef": 0.046,
        "start": 3884.5,
        "span": 8,
        "expts": 70,
        "reps": 1000,
        "reset_mode": RESET_MODE,
    }
    ef_stark = meas.EFSpecF0G1Power(
        cfg_dict, qi=qi, go=False, style="medium", params=params
    )
    finish_experiment(ef_stark, args)
    if not args.no_update_config:
        ef_stark.update()
