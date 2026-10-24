"""Run fine resonator spectroscopy for one qubit."""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = False
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)
    params = { "span": 20, "gain": 0.03 , "reps": 5000, "expts": 1000,
              "pulse_e": False, "pulse_f": False}
    rspec = meas.ResSpec(cfg_dict, qi=qi, go=False, style="fine", params=params)
    finish_experiment(rspec, args)
    # rspec.update()


        
        