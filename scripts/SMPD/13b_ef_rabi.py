"""Run e-f amplitude Rabi calibration for one qubit."""

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    first_time = False
    if first_time:
        config.update_qubit(
            str(cfg_path),
            ("pulses", "pi_ef", "sigma"),
            auto_cfg["device"]["qubit"]["pulses"]["pi_ge"]["sigma"][qi],
            qi,
        )
        config.update_qubit(
            str(cfg_path),
            ("pulses", "pi_ef", "gain"),
            auto_cfg["device"]["qubit"]["pulses"]["pi_ge"]["gain"][qi],
            qi,
        )

    params = {"checkEF": True, "reps": 2000, "reset_mode": RESET_MODE}
    ef_rabi = meas.RabiExperiment(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(ef_rabi, args)
    if ef_rabi.status:
        config.update_qubit(
            str(cfg_path), ("pulses", "pi_ef", "gain"), ef_rabi.data["pi_length"], qi
        )