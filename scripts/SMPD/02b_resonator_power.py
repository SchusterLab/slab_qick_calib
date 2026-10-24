"""Run resonator power spectroscopy for one qubit."""

import matplotlib.pyplot as plt
import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

from common import parse_args, prepare_experiment, show_all_plots


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
    params = {"rng": 2, "max_gain": 0.08, "expts_gain": 200, "span": 5,
              "expts": 200, "reps": 1, "log": True, "freq_offset": -1}
    rspec_power = meas.ResSpecPower(cfg_dict, qi=qi, go=False, live_plot=True, params=params)
    if args.no_show:
        rspec_power.run(display=False, save=True, progress=not args.no_progress)
    else:
        original_show = plt.show
        plt.show = lambda *show_args, **show_kwargs: None
        try:
            rspec_power.run(display=True, save=True, progress=not args.no_progress)
        finally:
            plt.show = original_show
    if not args.no_update_config:
        config.update_readout(
            str(cfg_path), "lamb", rspec_power.data["lamb_shift"], qi
        )
    if not args.no_show:
        plt.ioff()
        show_all_plots(block=False)
        plt.show(block=True)