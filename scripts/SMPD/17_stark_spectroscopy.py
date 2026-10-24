"""Run Stark spectroscopy for one qubit."""

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import rfboard, config

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qubit_line_qi = 1
    stark_line_qi = 0
    readout_line_qi = 1
    JPA_ON = True
    RESET_MODE = "f0g1"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args,
        __file__,
        qubit_line_qi,
        JPA_ON,
        additional_readout_qis=(stark_line_qi,),
    )

    QUBIT_ATTN = (10, 30)  # (attn1, attn2) dB, this run only; next activate() restores config
    # Qubit probe gain; saturation biases the PNS photon number, pick it from 17b's s (CLAUDE.md "Probe power and the PNS comb")
    # 17b #548 (20 us probe): pi rotation at gain 0.110; the n=0 peak is compressed
    # 10.5% / 6% / 2.7% at 0.04 / 0.03 / 0.02 (#549 = 0.04, #550 = 0.03 at 32 kHz steps)
    PROBE_LENGTH = 20  # us; weak-probe line areas stay prop. to P(n) for any length
    PROBE_GAIN = 0.03
    REPS = 10000  # ~97 us per shot with f0g1 reset
    qb_ch = auto_cfg.hw.soc.dacs.qubit.ch[qubit_line_qi]
    soc.rfb_set_gen_rf(qb_ch, *QUBIT_ATTN)
    # Record it in rfboard_active so the saved yml/h5 shows what was actually applied
    active = auto_cfg.hw.soc.rfboard_active.dac_qubit[str(qb_ch)]
    config.update_rfboard_active(str(cfg_path), "dac_qubit", qb_ch,
                                {**active, "attn1": QUBIT_ATTN[0], "attn2": QUBIT_ATTN[1]})

    # )

    try:
        # Notebook-equivalent editable Stark settings.
        params = {
            "df_stark": 0,
            "max_stark_gain": 0.06,  # ~1 photon: n=0 depletion and the n=1 line carry n_bar
            "min_stark_gain": 0.000,
            "df": -1.2,  # f_ge-2.7 .. f_ge+0.3: n = 0, 1, 2
            'expts': 300,  # 10 kHz steps: the n=0 line is only ~30 kHz HWHM at low buffer gain
            'span': 3.0,
            'gain': PROBE_GAIN,
            "stark_expts": 6,
            "length": PROBE_LENGTH,
            "stark_length": PROBE_LENGTH + 10,  # buffer photons rise as (1-exp(-kappa t/2))^2: ~99% after 10 us
            "reps": REPS,
            "stark_qubit": qubit_line_qi,
            "stark_chan": auto_cfg.hw.soc.dacs.readout.ch[stark_line_qi],
            "stark_freq": auto_cfg.device.readout.frequency[stark_line_qi],
            "qubit": [qubit_line_qi],
            "qubit_chan": auto_cfg.hw.soc.adcs.readout.ch[readout_line_qi],
            # 'two_peak' | 'stark_shift' (update() writes nstark); CLAUDE.md "Stark spectroscopy fit modes (script 17)"
            "fit_type": "two_peak",
            "chi_buffer": auto_cfg.device.readout.chi[stark_line_qi],
            "reset_mode": RESET_MODE,
        }
        stark = meas.StarkSpec(
            cfg_dict, qi=qubit_line_qi, go=False, style="medium", params=params
        )
        finish_experiment(stark, args)

    finally:
        rfboard.activate_qubit_rf(
            qubit_line_qi, soc, auto_cfg, cfg_file=str(cfg_path), verbose=False
        )