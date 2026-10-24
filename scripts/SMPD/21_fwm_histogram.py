"""Run an FWM single-shot histogram measurement."""

import slab_qick_calib.experiments as meas

from common import (
    finish_experiment,
    fwm_band_point,
    parse_args,
    prepare_experiment,
)


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    buffer_line_qi = 0
    waste_line_qi = qi
    JPA_ON = True
    RESET_MODE = "active"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"


    # Give one and leave the other None to read it off the band fit; both = off the band.
    PUMP_FREQUENCY =  None
    PUMP_GAIN = 0.3

    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(0,)
    )

    buffer_frequency = auto_cfg.device.readout.frequency[buffer_line_qi]
    waste_frequency = auto_cfg.device.readout.frequency[waste_line_qi]
    buffer_dac_channel = auto_cfg.hw.soc.dacs.readout.ch[buffer_line_qi]
    buffer_adc_channel = auto_cfg.hw.soc.adcs.readout.ch[buffer_line_qi]
    # The pump has its own hardware line when hw.soc.dacs.pump exists,
    # otherwise it shares the qubit DAC.
    pump_dacs = auto_cfg.hw.soc.dacs.get("pump", auto_cfg.hw.soc.dacs.qubit)
    pump_dac_channel = pump_dacs.ch[qi]
    if pump_dac_channel == auto_cfg.hw.soc.dacs.qubit.ch[qi]:
        print(
            "Note: the pump and qubit are on the same DAC channel "
            f"({pump_dac_channel}); set hw.soc.dacs.pump.ch to separate them."
        )

    pump_frequency, pump_gain = fwm_band_point(
        auto_cfg, PUMP_FREQUENCY, PUMP_GAIN
    )

    print(
        f"FWM histogram: buffer frequency={buffer_frequency:.3f} MHz "
        f"(DAC {buffer_dac_channel}, ADC {buffer_adc_channel}), "
        f"waste frequency={waste_frequency:.3f} MHz, "
        f"pump frequency={pump_frequency:.3f} MHz (qubit DAC {pump_dac_channel})"
    )

    # Editable FWM histogram settings. One fixed parameter set, no sweeps:
    # the experiment acquires pump-off (dark count) and pump-on (P(e)).
    params = {
        # Start conservatively for the RFSoC readout loop; increase after a
        # successful run if more histogram statistics are needed.
        "shots": 50000,
        "frequency_b": buffer_frequency,
        "gain_b": 0.2,
        "length_b": 40,
        "delay_b": 0,
        "frequency_p": pump_frequency,
        "gain_p": pump_gain,
        "length_p": 40,
        'check_f':True,
        "sep_readout": True,
        "reset_mode": RESET_MODE,
    }
    print("FWM histogram parameters initialized.")
    histogram = meas.SMPDHistogramExperiment(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(histogram, args)
