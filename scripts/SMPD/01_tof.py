"""Run time-of-flight calibration for one qubit."""

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import rfboard

from common import finish_experiment, parse_args, prepare_experiment

if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = False
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    params = {"frequency": 7790, "gain": 1, "readout_length": 2, "rounds": 1000}
    hw = auto_cfg.hw.soc
    adc_ch = hw.adcs.readout.ch[qi]
    dac_readout_ch = hw.dacs.readout.ch[qi]
    dac_qubit_ch = hw.dacs.qubit.ch[qi]

    rfboard.setup_readout_chain(
        qi,
        soc,
        auto_cfg,
        bw_adc=1000,
        bw_dac=1000,
    )
    print(f"ADC channel: {adc_ch}, DAC readout channel: {dac_readout_ch}, DAC qubit channel: {dac_qubit_ch}")
    soc.rfb_set_ro_rf(adc_ch, 20)
    # soc.rfb_set_gen_rf(dac_readout_ch, 0, 0)

    try:
        tof = meas.ToFCalibrationExperiment(cfg_dict, qi=qi, go=False, params=params)
        finish_experiment(tof, args)

        tof.update()

    finally:
        rfboard.activate_qubit_rf(
            qi, soc, auto_cfg, cfg_file=str(cfg_path), verbose=False
        )
