"""Run T1, optionally including resonator spectroscopy, qubit spectroscopy, and Rabi."""

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

from common import finish_experiment, parse_args, prepare_experiment


def add_arguments(parser):
    parser.add_argument(
        "--include-prerequisites", action="store_true",
        help="Run resonator spectroscopy, qubit spectroscopy, and Rabi before T1",
    )


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, add_arguments, config=CONFIG)
    if args.include_prerequisites:
        # Resonator spectroscopy prerequisite.
        run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
        resonator_params = {"center": 7781, "span": 5, "gain": 0.05, "reps": 200, "expts": 100,
                            "pulse_e": False, "pulse_f": True}
        resonator = meas.ResSpec(cfg_dict, qi=qi, go=False, style="fine", params=resonator_params)
        finish_experiment(resonator, args)
        if not args.no_update_config:
            resonator.update()

        # Qubit spectroscopy prerequisite.
        run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
        qubit_params = {"span": 2, "expts": 100, "gain": 0.01, "reps": 2000,
                        "length": 30, "readout_length": 2, "sep_readout": True}
        qubit = meas.QubitSpec(cfg_dict, qi=qi, go=False, style="fine", params=qubit_params)
        finish_experiment(qubit, args)
        if not args.no_update_config and qubit.status:
                config.update_qubit(str(cfg_path), "f_ge", qubit.data["best_fit"][2], qi)
                config.update_qubit(str(cfg_path), "kappa", qubit.data["best_fit"][3], qi)

        # Amplitude Rabi prerequisite.
        run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
        rabi_params = {"max_gain": 1, "reps": 1000, "plot_all": True}
        rabi = meas.RabiExperiment(cfg_dict, qi=qi, go=False, params=rabi_params)
        finish_experiment(rabi, args)
        if not args.no_update_config and rabi.status:
            config.update_qubit(
                str(cfg_path), ("pulses", "pi_ge", "gain"), rabi.data["pi_length"], qi
            )

    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
    params = {"reps": 1000, "pulse_ef": False,
              "reset_mode": RESET_MODE}
    t1 = meas.T1Experiment(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(t1, args)
    if not args.no_update_config:
        t1.update()