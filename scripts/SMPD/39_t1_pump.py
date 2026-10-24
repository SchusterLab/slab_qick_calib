"""Run T1 with the FWM pump held on during the wait time."""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"

    # Give one and leave the other None to read it off the band fit; both = off the band.
    PUMP_FREQUENCY = None
    PUMP_GAIN = 0.6

    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    params = {"reps": 1000, "pump": True,"pulse_ef": False,
              "frequency_p": PUMP_FREQUENCY, "gain_p": PUMP_GAIN,
              "reset_mode": RESET_MODE}
    t1 = meas.T1Experiment(cfg_dict, qi=qi, go=False, params=params)
    print(f"Pump: {t1.cfg.expt.frequency_p:.3f} MHz, gain {t1.cfg.expt.gain_p:.4f}")
    finish_experiment(t1, args)
    if not args.no_update_config:
        t1.update()   # -> device.fwm.T1_pump (+ the pump setting); device T1 untouched
