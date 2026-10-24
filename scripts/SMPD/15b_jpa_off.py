"""Turn off the JPA: zero the pump tone and set the DC bias to 0 V."""

from slab_qick_calib.experiments.jpa.jpa_tone import turn_off_jpa

from common import parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = False
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    turn_off_jpa(soc, cfg_dict["soc"], auto_cfg, idx=qi, off_bias_v=0.0, verbose=True)
