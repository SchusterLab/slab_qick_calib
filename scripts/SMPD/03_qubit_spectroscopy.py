"""Run fine qubit spectroscopy for one qubit."""

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    QUBIT_ATTN = (25, 30)  # (attn1, attn2) dB, this run only; next activate() restores config
    qb_ch = auto_cfg.hw.soc.dacs.qubit.ch[qi]
    soc.rfb_set_gen_rf(qb_ch, *QUBIT_ATTN)
    # Record it in rfboard_active so the saved yml/h5 shows what was actually applied
    active = auto_cfg.hw.soc.rfboard_active.dac_qubit[str(qb_ch)]
    config.update_rfboard_active(str(cfg_path), "dac_qubit", qb_ch,
                                {**active, "attn1": QUBIT_ATTN[0], "attn2": QUBIT_ATTN[1]})

    params = {"span": 0.2, "expts": 500, "gain": 0.1, "reps": 500,
              "length": 100, "readout_length": 4, "sep_readout": True,
              "reset_mode": RESET_MODE}
    qspec = meas.QubitSpec(cfg_dict, qi=qi, go=False, style="fine", params=params)
    finish_experiment(qspec, args)
    if not args.no_update_config and qspec.status:
        config.update_qubit(str(cfg_path), "f_ge", qspec.data["best_fit"][2], qi)
        config.update_qubit(str(cfg_path), "kappa", qspec.data["best_fit"][3], qi)