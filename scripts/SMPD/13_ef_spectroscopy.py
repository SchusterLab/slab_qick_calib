"""Run e-f qubit spectroscopy for one qubit."""

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

    QUBIT_ATTN = (0, 0)  # (attn1, attn2) dB, this run only; next activate() restores config
    qb_ch = auto_cfg.hw.soc.dacs.qubit.ch[qi]
    soc.rfb_set_gen_rf(qb_ch, *QUBIT_ATTN)
    # Record it in rfboard_active so the saved yml/h5 shows what was actually applied
    active = auto_cfg.hw.soc.rfboard_active.dac_qubit[str(qb_ch)]
    config.update_rfboard_active(str(cfg_path), "dac_qubit", qb_ch,
                                {**active, "attn1": QUBIT_ATTN[0], "attn2": QUBIT_ATTN[1]})


    params = {"checkEF": True, "gain": 0.0004, "span": 2, "length":5,"reset_mode": RESET_MODE}
    ef_spec = meas.QubitSpec(cfg_dict, qi=qi, go=False, style="medium", params=params)
    finish_experiment(ef_spec, args)
    if not args.no_update_config and ef_spec.status:
        config.update_qubit(str(cfg_path), "f_ef", ef_spec.data["best_fit"][2], qi)