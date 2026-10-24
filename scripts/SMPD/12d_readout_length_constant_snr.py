"""
Single shots vs readout length at constant SNR, projected onto the SMPD NEP.

Diagnostic, writes nothing to the config. Holds gain**2 * length at the config
value (constant readout photon number, so constant SNR for a linear readout),
and at each length takes g and e shots under RESET_MODE. Reports the measured
SNR, P(e|g) (the readout's share of the dark count), P(g|e), and the NEP
relative to the config length for a fixed DETECTION_WINDOW. CLAUDE.md
"Readout length at constant SNR (script 12d)".
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import slab_qick_calib.experiments as meas

from common import (effective_final_delay, parse_args, prepare_experiment,
                    suppress_duplicate_config, wait_for_plots)

CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
QI = 1
JPA_ON = True
RESET_MODE = "f0g1"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
SHOTS = 20000
LENGTHS = np.geomspace(2, 8, 30)  # us; the config length is added
# Buffer window the duty cycle is quoted for; script 40's LENGTH_CENTER.
DETECTION_WINDOW = 35  # us


if __name__ == "__main__":
    args = parse_args(__doc__, config=CONFIG)
    if args.no_show:
        plt.switch_backend("Agg")
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, QI, JPA_ON
    )
    out_dir = Path(cfg_dict["metadata_path"])
    gain_ref = auto_cfg.device.readout.gain[QI]
    length_ref = auto_cfg.device.readout.readout_length[QI]
    max_gain = auto_cfg.device.readout.max_gain[QI]
    lengths = np.unique(np.round(np.append(LENGTHS, length_ref), 3))

    rows = []
    print(f"reference: gain {gain_ref} at {length_ref} us; window {DETECTION_WINDOW} us")
    print(f"{'len/us':>7} {'gain':>7} {'SNR':>6} {'P(e|g)':>8} {'P(g|e)':>8} "
          f"{'cycle':>7} {'duty':>6}")
    for length in lengths:
        gain = gain_ref * np.sqrt(length_ref / length)
        if gain > max_gain:
            print(f"{length:7.2f} skipped: gain {gain:.3f} > max_gain {max_gain}")
            continue
        expt = meas.HistogramExperiment(
            cfg_dict, qi=QI, go=False, check_f=False,
            prefix=f"readout_length_qubit{QI}",
            params={"shots": SHOTS, "gain": float(gain),
                    "readout_length": float(length), "reset_mode": RESET_MODE},
        )
        # One config yml per run folder, not per point (CLAUDE.md "One config yml per run").
        suppress_duplicate_config(expt, out_dir)
        data = expt.acquire(progress=not args.no_progress)
        data = expt.analyze(data=data)
        expt.save_data(data)
        expt.save_config()

        # Same period as script 40: window + readout + reset tail.
        cycle = DETECTION_WINDOW + length + effective_final_delay(expt.cfg.expt)
        sigma = data.get("sigma", np.nan)
        rows.append({
            "readout_length": float(length), "gain": float(gain),
            "snr": abs(data["ie"] - data["ig"]) / sigma,
            "p_e_given_g": float(data["err_g"]), "p_g_given_e": float(data["err_e"]),
            "threshold": float(data["thresholds"][0]),
            "cycle_us": float(cycle), "duty": DETECTION_WINDOW / cycle,
            "file": Path(expt.fname).name,
        })
        r = rows[-1]
        print(f"{length:7.2f} {gain:7.4f} {r['snr']:6.2f} {r['p_e_given_g']:8.4f} "
              f"{r['p_g_given_e']:8.4f} {cycle:7.2f} {r['duty']:6.3f}")

    # NEP ~ sqrt(p_dark / cycle) / (eta * window / cycle) = sqrt(p_dark * cycle) / (eta * window).
    ref = next(r for r in rows if np.isclose(r["readout_length"], length_ref))
    for r in rows:
        eta_ratio = (1 - ref["p_g_given_e"]) / (1 - r["p_g_given_e"])
        cycle_ratio = r["cycle_us"] / ref["cycle_us"]
        # Dark count = readout false positives only (upper bound on the readout's effect).
        r["nep_rel_readout_dark"] = (np.sqrt(r["p_e_given_g"] / ref["p_e_given_g"]
                                             * cycle_ratio) * eta_ratio)
        # Dark count fixed by the pump, not the readout: only duty and eta move.
        r["nep_rel_fixed_dark"] = np.sqrt(cycle_ratio) * eta_ratio

    with (out_dir / "readout_length_constant_snr.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nNEP relative to {length_ref} us readout ({DETECTION_WINDOW} us window):")
    print(f"{'len/us':>7} {'readout dark':>13} {'fixed dark':>11}")
    for r in rows:
        print(f"{r['readout_length']:7.2f} {r['nep_rel_readout_dark']:13.3f} "
              f"{r['nep_rel_fixed_dark']:11.3f}")

    L = np.array([r["readout_length"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    ax[0].plot(L, [r["snr"] for r in rows], "o-")
    ax[0].set_ylabel("SNR  |ie - ig| / sigma")
    ax[1].plot(L, [r["p_e_given_g"] for r in rows], "o-", label="P(e|g)")
    ax[1].plot(L, [r["p_g_given_e"] for r in rows], "s-", label="P(g|e)")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("misassignment")
    ax[1].legend(fontsize="small")
    ax[2].plot(L, [r["nep_rel_readout_dark"] for r in rows], "o-",
               label="dark = readout P(e|g)")
    ax[2].plot(L, [r["nep_rel_fixed_dark"] for r in rows], "s-",
               label="dark fixed (pump-limited)")
    ax[2].axhline(1, color="0.5", lw=1)
    ax[2].set_ylabel(f"NEP / NEP({length_ref:g} us)")
    ax[2].legend(fontsize="small")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("readout length (us)")
        a.axvline(length_ref, color="tab:red", ls=":", lw=1)
        a.grid(alpha=0.3)
    fig.savefig(out_dir / "readout_length_constant_snr.png", dpi=120)

    print(f"  {out_dir / 'readout_length_constant_snr.csv'}")
    if not args.no_show:
        wait_for_plots()
