"""Turn on the JPA using the operating point stored in the YAML file."""

from slab_qick_calib.experiments.jpa.jpa_tone import turn_on_jpa

from common import parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = False
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    # Leave these as None to use hw.soc.dacs.jpa[qi] from the YAML file.
    pump_gain = None
    dc_bias_v = None
    pump_freq_mhz = None
    mixer_freq_mhz = None

    turn_on_jpa(
        soc,
        cfg_dict["soc"],
        auto_cfg,
        idx=qi,
        pump_gain=pump_gain,
        dc_bias_v=dc_bias_v,
        pump_freq_mhz=pump_freq_mhz,
        mixer_freq_mhz=mixer_freq_mhz,
        verbose=True,
    )