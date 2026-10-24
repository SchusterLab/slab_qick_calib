"""e-f spectroscopy with the f0-g1 drive on, at one f0-g1 amplitude.

Step 3 of the f0-g1 reset calibration; single-power check, writes nothing.
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
        # The drive whose shift is being measured. frequency_f0g1 is left out
        # so it comes from device.f0g1.stark_fit at this gain.
        "gain_f0g1": 0.1,
        # length_ef is derived as a pi at this gain from omega_ef_fit; see script 32 before changing it.
        "start": 3882.5,
        "gain_ef": 0.046,
        "span": 15,
        "expts": 70,
        "reps": 1000,
        "reset_mode": RESET_MODE,
    }
    ef_spec = meas.EFSpecF0G1(
        cfg_dict, qi=qi, go=False, style="medium", params=params
    )
    finish_experiment(ef_spec, args)
