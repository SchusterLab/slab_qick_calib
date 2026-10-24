"""Stark photon calibration: qubit line shift vs buffer gain with the buffer driven off resonance; writes nstark."""

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import rfboard, config

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qubit_line_qi = 1
    stark_line_qi = 0
    readout_line_qi = 1
    JPA_ON = True
    RESET_MODE = "f0g1"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qubit_line_qi, JPA_ON, additional_readout_qis=(stark_line_qi,)
    )

    QUBIT_ATTN = (10, 30)  # same probe as script 17
    qb_ch = auto_cfg.hw.soc.dacs.qubit.ch[qubit_line_qi]
    soc.rfb_set_gen_rf(qb_ch, *QUBIT_ATTN)
    active = auto_cfg.hw.soc.rfboard_active.dac_qubit[str(qb_ch)]
    config.update_rfboard_active(str(cfg_path), "dac_qubit", qb_ch,
                                {**active, "attn1": QUBIT_ATTN[0], "attn2": QUBIT_ATTN[1]})

    # Ramsey #333/#371/#500 drove here and measured a -7.80 MHz/gain^2 qubit shift (1%);
    # off resonance the line just shifts, so its slope compares the two programs' drive directly.
    # With #554's kappa/chi/f_g: ~81 kHz at gain 0.1 if StarkSpec drives like Ramsey (~250
    # photons/gain^2 at f_g), ~103 kHz if its tone really gives the PNS value (~326)
    STARK_FREQ = 6681.5503
    try:
        params = {
            "df_stark": 0,
            "max_stark_gain": 0.1,  # ~ -78 kHz shift at the top
            "min_stark_gain": 0.0,
            "stark_expts": 6,
            "df": -0.05,  # f_ge-0.35 .. f_ge+0.25: the line moves down
            "span": 0.6,
            "expts": 120,  # 5 kHz steps
            "gain": 0.03,
            "length": 20,
            "stark_length": 30,  # 10 us lead: the buffer field is at steady state when the probe starts
            "reps": 5000,
            "stark_qubit": qubit_line_qi,
            "stark_chan": auto_cfg.hw.soc.dacs.readout.ch[stark_line_qi],
            "stark_freq": STARK_FREQ,
            "qubit": [qubit_line_qi],
            "qubit_chan": auto_cfg.hw.soc.adcs.readout.ch[readout_line_qi],
            "fit_type": "stark_shift",  # line shift -> on-resonance photons per gain^2 (nstark)
            "chi_buffer": auto_cfg.device.readout.chi[stark_line_qi],
            "reset_mode": RESET_MODE,
        }
        stark = meas.StarkSpec(cfg_dict, qi=qubit_line_qi, go=False, style="medium", params=params)
        finish_experiment(stark, args)
        if not args.no_update_config:
            stark.update()
    finally:
        rfboard.activate_qubit_rf(
            qubit_line_qi, soc, auto_cfg, cfg_file=str(cfg_path), verbose=False
        )
