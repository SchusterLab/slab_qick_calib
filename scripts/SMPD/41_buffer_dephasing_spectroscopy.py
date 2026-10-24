"""Buffer spectroscopy through the qubit: Ramsey dephasing and shift vs buffer frequency."""

import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    buffer_line_qi = 0
    JPA_ON = True
    RESET_MODE = "f0g1"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(buffer_line_qi,)
    )
    # Fits f_g, chi, kappa and photons per gain^2 at f_g with no config input;
    # CLAUDE.md "Photon number from Ramsey dephasing (scripts 38, 41)".
    readout = auto_cfg.device.readout
    params = {
        "gain_b": 0.03,  # ~0.2 photons at f_g; T2 ~17 -> ~5 us on resonance
        "frequency_b_pts": meas.BufferDephasingSpec.default_frequencies(
            readout.frequency[buffer_line_qi], readout.chi[buffer_line_qi], step=0.07),
        "ref_every": 8,
        "reps": 10000,  # with 70 kHz steps: ~3% on photons/gain^2, ~2% on kappa (synthetic test)
        "reset_mode": RESET_MODE,
    }
    spec = meas.BufferDephasingSpec(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(spec, args)
    if not args.no_update_config:
        spec.update()
