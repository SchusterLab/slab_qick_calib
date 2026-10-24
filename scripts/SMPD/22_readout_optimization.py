"""Optimize readout frequency, gain, and length using single-shot fidelity."""

import matplotlib.pyplot as plt
import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment, wait_for_plots


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    # Notebook-equivalent editable readout optimization settings.
    low_gain = False
    check_f = False
    params = {
        "expts_f": 1,
        "expts_gain": 1,
        "expts_len": 10,
        "shots": 20000,
        "check_f": check_f,
        "reset_mode": RESET_MODE,
    }
    # For a full 3D optimization, use for example:
    # params = {"expts_f": 5, "expts_gain": 5, "expts_len": 5, "shots": 10000}

    shotopt = meas.SingleShotOptExperiment(
        cfg_dict,
        qi=qi,
        go=False,
        params=params,
        check_f=check_f,
        display=True,
        live_plot=True,
        style="fine",
    )

    # SingleShotOpt.analyze() returns an index tuple, so run its lifecycle
    # manually instead of sending that tuple through the generic data saver.
    if args.no_show:
        shotopt.acquire(progress=not args.no_progress)
        shotopt.save_data(shotopt.data)
        shotopt.analyze(low_gain=low_gain)
    else:
        original_show = plt.show
        plt.show = lambda *show_args, **show_kwargs: None
        try:
            shotopt.acquire(progress=not args.no_progress)
            shotopt.save_data(shotopt.data)
            shotopt.analyze(low_gain=low_gain)
            shotopt.display(shotopt.data)
        finally:
            plt.show = original_show

    shotopt.update(cfg_dict["cfg_file"])

    # Re-run the standard histogram using the optimized YAML settings.
    shot = meas.HistogramExperiment(cfg_dict, qi=qi, go=False)
    finish_experiment(shot, args, wait=False)
    shot.update()
    if not args.no_show:
        wait_for_plots()