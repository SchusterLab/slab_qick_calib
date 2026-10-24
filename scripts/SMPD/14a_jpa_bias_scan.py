"""Scan the JPA DC bias and watch the waste resonance move."""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = False          # the scan drives the pump and bias itself
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    jpa = auto_cfg.hw.soc.dacs.jpa

    # The bias range to sweep, in volts. Bracket the stored operating point
    # until you know where the JPA turns on.
    DC_START = -0.1
    DC_STOP = 0.2
    DC_EXPTS = 101

    params = {
        # Waste resonance sweep taken at each bias.
        "center": auto_cfg.device.readout.frequency[qi],
        "span": 2000.0,
        "expts": 100,
        "gain": 1,
        "reps": 5000,
        "rounds": 1,
        # DC bias sweep. The JPA has its own bias channel, not the qubit flux
        # line -- hw.soc.dacs.flux does not exist on this fridge.
        "dc_start": DC_START,
        "dc_stop": DC_STOP,
        "expts_dc": DC_EXPTS,
        "bias_chan": jpa.jpa_dc_ch[qi],
        "jpa_dc_val": jpa.jpa_dc_val[qi],
        "dc_settle": 0.25,
        "reset_bias_after": True,
        "reset_bias": 0.0,
        # Pump held fixed while the bias moves.
        "pump_ch": jpa.ch[qi],
        "pump_nqz": jpa.nyquist[qi],
        "pump_freq_mhz": jpa.pump_freq[qi],
        "pump_mixer_freq_mhz": jpa.mixing_freq[qi],
        "pump_attn1_db": jpa.attn1[qi],
        "pump_attn2_db": jpa.attn2[qi],
        "pump_filter_bw_mhz": jpa.filter_bw[qi],
        "pump_filter_type": jpa.filter_type[qi],
        "pump_gain": jpa.pump_gain[qi],
        # A survey: no update; 14_jpa_fine_tuning.py picks and writes the bias.
        "rspec_update": False,
    }
    jpa_bias = meas.JPABiasScan(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(jpa_bias, args)
