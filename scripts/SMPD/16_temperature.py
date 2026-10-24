"""Measure the qubit temperature from EF Rabi contrast."""

from slab_qick_calib.calib import measure_func

from common import parse_args, prepare_experiment


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    # No RESET_MODE on purpose: residual excited population is what this measures.

    # Edit the temperature measurement settings here.
    temp_guess_mk = 35
    expts = 32
    reps = 200
    
    measure_f = False

    temperature_mk, population, _, _ = measure_func.measure_temp(
        cfg_dict,
        qi=qi,
        temp=temp_guess_mk,
        expts=expts,
        reps=reps,
        measure_f=measure_f,
    )
    print(f"Qubit {qi} temperature: {temperature_mk:.2f} mK")
    print(f"Excited-state population: {population:.4g}")