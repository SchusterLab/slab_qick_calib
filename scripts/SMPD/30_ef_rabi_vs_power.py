"""e-f Rabi rate versus e-f drive power: Omega_ef(V_ef).

Step 2b of the f0-g1 reset calibration; writes device.f0g1.omega_ef_fit.
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
        # Gain axis (Rabi2D's own names; y sweeps gain, x sweeps pulse length).
        "start_y": 0.0075,
        "span_y": 0.043,         # 0.05 -> 0.20, 4x
        "expts_y": 10,
        # Time axis, from script 29.
        "length": 3.6,          # flat duration; max_length = 2 * num_osc * length
        "num_osc": 2.5,
        "expts": 60,
        "reps": 2000,
        "ramp_sigma": 0.02,
        "ramp_sigma_inc": 3,
        "reset_f": True,
        "reset_mode": RESET_MODE,
    }
    ef_rabi_power = meas.EFRabiPower(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(ef_rabi_power, args)
    if not args.no_update_config:
        ef_rabi_power.update()
