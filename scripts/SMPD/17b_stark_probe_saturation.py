"""Qubit line width vs probe gain with the Stark probe's settings: picks an unsaturated probe for script 17."""

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import rfboard, config

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qubit_line_qi = 1
    JPA_ON = True
    RESET_MODE = "passive"  # QubitSpec does not configure_reset(); a long final_delay stands in for 17's reset
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qubit_line_qi, JPA_ON
    )

    QUBIT_ATTN = (10, 30)  # must equal script 17's, or the gains do not transfer
    qb_ch = auto_cfg.hw.soc.dacs.qubit.ch[qubit_line_qi]
    soc.rfb_set_gen_rf(qb_ch, *QUBIT_ATTN)
    active = auto_cfg.hw.soc.rfboard_active.dac_qubit[str(qb_ch)]
    config.update_rfboard_active(str(cfg_path), "dac_qubit", qb_ch,
                                {**active, "attn1": QUBIT_ATTN[0], "attn2": QUBIT_ATTN[1]})

    try:
        # Width^2 = w0^2 (1 + s) with s ~ gain^2: compare_photon_cal.py --saturation
        # reads the per-row Lorentzian fits; CLAUDE.md "Probe power and the PNS comb".
        params = {
            "span": 1.5,  # n=0 line only (no buffer drive)
            "expts": 100,
            "length": 20,  # must equal script 17's PROBE_LENGTH
            "pulse_type": "const",
            "max_gain": 0.3,
            "rng": 60,  # gains 0.005 .. 0.3, log spaced; a shorter probe needs more gain
            "expts_gain": 9,
            "log": True,
            "reps": 2000,
            "final_delay": 200,  # ~3 T1, so each shot starts in |g> like script 17's reset
            "reset_mode": RESET_MODE,
        }
        sat = meas.QubitSpecPower(cfg_dict, qi=qubit_line_qi, go=False, params=params)
        finish_experiment(sat, args)
    finally:
        rfboard.activate_qubit_rf(
            qubit_line_qi, soc, auto_cfg, cfg_file=str(cfg_path), verbose=False
        )
