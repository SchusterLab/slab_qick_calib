"""e-f time Rabi with a flat-top pulse, at one drive amplitude.

Step 2 of the f0-g1 reset calibration. Exploratory, and deliberately no update().
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
        "checkEF": True,
        "sweep": "length",
        # The reset's e-f shape; "type" is the alias and follows automatically.
        "pulse_type": auto_cfg.device.f0g1.pulse_type_ef,
        "gain": 0.01,
        "length": 1.0,          # flat duration; sets the swept window
        "ramp_sigma": 0.02,
        "ramp_sigma_inc": 3,
        "num_osc": 2.5,
        "expts": 60,
        "reps": 3000,
        # reset_f clears the |f> this leaves behind; needs reset >= 2 (CLAUDE.md "Resetting out of |f>").
        "reset_f": True,
        "reset_mode": RESET_MODE,
    }
    ef_rabi = meas.RabiExperiment(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(ef_rabi, args)
    if ef_rabi.status:
        # pi_length is the fitted pi duration; the Rabi rate is its inverse
        # half-period. Reported, never written -- see the docstring.
        t_pi = ef_rabi.data["pi_length"]
        print(f"\nflat-top e-f pi length at gain {params['gain']}: {t_pi:.4f} us")
        print(f"  -> Omega_ef/2pi = {1 / (2 * t_pi):.4f} MHz")
        print("  Nothing written to the config; run 30_ef_rabi_vs_power for the "
              "rate-vs-gain calibration.")
