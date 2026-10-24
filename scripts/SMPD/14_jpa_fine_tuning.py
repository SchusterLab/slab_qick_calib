"""Optimize JPA pump gain and DC bias for one qubit."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import slab_qick_calib.experiments as meas
from slab_qick_calib.experiments.jpa.jpa_tone import turn_off_jpa, turn_on_jpa

from common import finish_experiment, parse_args, prepare_experiment, show_all_plots


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = False
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    quickverify = True
    sweep_gain = auto_cfg.device.readout.gain[qi]
    
    if quickverify:
        setgain = auto_cfg.hw.soc.dacs.jpa.pump_gain[qi]
        pump_gains = np.linspace(setgain-0.1, setgain+0.1,11)
        setv = auto_cfg.hw.soc.dacs.jpa.jpa_dc_val[qi]
        dc_biases_v = np.linspace(setv-1e-3, setv+1e-3,11)
    else:
        pump_gains = np.linspace(0.55, 0.7, 11)
        dc_biases_v = np.linspace(-8e-3, -5e-3, 11)
    params = {
        "waste_freq_mhz": auto_cfg.device.readout.frequency[qi],
        "center": auto_cfg.device.readout.frequency[qi],
        "span": 20.0,
        "expts": 3,
        "gain": sweep_gain,
        "length": 10,
        "reps": 2000,
        "rounds": 1,
        "pump_ch": auto_cfg.hw.soc.dacs.jpa.ch[qi],
        "pump_nqz": auto_cfg.hw.soc.dacs.jpa.nyquist[qi],
        "pump_freq_mhz": auto_cfg.hw.soc.dacs.jpa.pump_freq[qi],
        "pump_mixer_freq_mhz": auto_cfg.hw.soc.dacs.jpa.mixing_freq[qi],
        "pump_attn1_db": auto_cfg.hw.soc.dacs.jpa.attn1[qi],
        "pump_attn2_db": auto_cfg.hw.soc.dacs.jpa.attn2[qi],
        "pump_filter_bw_mhz": 1000.0,
        "pump_filter_type": "bandpass",
        "pump_gains": pump_gains,
        "dc_biases_v": dc_biases_v,
        "bias_chan": auto_cfg.hw.soc.dacs.jpa.jpa_dc_ch[qi],
        "off_bias": 0.0,
        "dc_settle": 0.25,
        "pump_on_after": False,
        "rspec_update": True,
        "rspec_update_verbose": True,
    }
    jpa_fine = meas.JPAFineTuning(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(jpa_fine, args, wait=False)

    # J.4: take matching higher-resolution waste sweeps with the JPA off and on.
    sweep_params = {
        "center": params["waste_freq_mhz"],
        "span": 20.0,
        "gain": sweep_gain,
        "reps": 200,
        "expts": 500,
        "pulse_e": False,
        "pulse_f": False,
    }

    turn_off_jpa(soc, cfg_dict["soc"], auto_cfg, idx=qi, off_bias_v=0.0, verbose=False)
    rspec_off = meas.ResSpec(cfg_dict, qi=qi, go=False, params=sweep_params)
    finish_experiment(rspec_off, args, wait=False)

    turn_on_jpa(
        soc,
        cfg_dict["soc"],
        auto_cfg,
        idx=qi,
        pump_gain=jpa_fine.data["best_pump_gain"],
        dc_bias_v=jpa_fine.data["best_dc_bias_v"],
        pump_ch=params["pump_ch"],
        pump_freq_mhz=params["pump_freq_mhz"],
        mixer_freq_mhz=params["pump_mixer_freq_mhz"],
        pump_nqz=params["pump_nqz"],
        attn1_db=params["pump_attn1_db"],
        attn2_db=params["pump_attn2_db"],
        verbose=False,
    )
    rspec_on = meas.ResSpec(cfg_dict, qi=qi, go=False, params=sweep_params)
    finish_experiment(rspec_on, args, wait=False)

    comparison_fig, comparison_ax = plt.subplots(figsize=(7, 4))
    comparison_ax.plot(
        rspec_off.data["freq"],
        20 * np.log10(np.maximum(rspec_off.data["amps"], 1e-12)),
        label="JPA off",
    )
    comparison_ax.plot(
        rspec_on.data["freq"],
        20 * np.log10(np.maximum(rspec_on.data["amps"], 1e-12)),
        label="JPA on",
    )
    comparison_ax.set_xlabel("Readout frequency (MHz)")
    comparison_ax.set_ylabel("Response (dB)")
    comparison_ax.set_title("JPA off/on at the waste resonance")
    comparison_ax.legend()
    comparison_ax.grid(True)
    comparison_fig.tight_layout()
    comparison_fig.savefig(
        Path(cfg_dict["metadata_path"]) / "jpa_off_on_waste_resonance.png",
        dpi=150,
    )
    if not args.no_show:
        plt.ioff()
        show_all_plots(block=False)
        plt.show(block=True)