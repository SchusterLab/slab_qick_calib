"""
Ground-state single shots as a function of readout power.

Diagnostic, writes nothing to the config. At each readout gain it takes |g>
shots only (no pi pulse) and asks whether the |g> blob stays one clean gaussian:
where it sits, how wide it is, and what fraction of shots land far outside it.
Shots leaving the blob as the gain rises are the signature of
measurement-induced state transitions; a width that grows is JPA saturation or
measurement-induced dephasing.

Everything is done in the IQ plane. With the passive reset HistogramProgram
plays the readout at phase 0, so the config's rotation (and the threshold built
on it) does not apply here, and the blob can rotate with power anyway. Shots are
divided by the gain so a linear readout leaves the blob in place.

Each gain is its own HistogramExperiment, built inside the loop so each gets
the next filename (see "Histogram vs pump power (script 24)" in CLAUDE.md).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import slab_qick_calib.experiments as meas

from common import (parse_args, prepare_experiment, suppress_duplicate_config,
                    wait_for_plots)

CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
QI = 1
JPA_ON = True
RESET_MODE = "passive"
SHOTS = 10000
GAINS = np.geomspace(0.01, 1.0, 20)
# A 2D gaussian puts exp(-N**2/2) of its shots beyond N sigma (1.1% at 3).
OUTLIER_SIGMA = 3.0


def blob_stats(z):
    """Robust centre and width of the dominant blob, and the fraction outside it."""
    center = np.median(z.real) + 1j * np.median(z.imag)
    mad = lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))
    sigma = 0.5 * (mad(z.real) + mad(z.imag))
    outside = float(np.mean(np.abs(z - center) > OUTLIER_SIGMA * sigma))
    return center, sigma, outside


if __name__ == "__main__":
    args = parse_args(__doc__, config=CONFIG)
    if args.no_show:
        plt.switch_backend("Agg")
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, QI, JPA_ON
    )
    out_dir = Path(cfg_dict["metadata_path"])
    gain_cfg = auto_cfg.device.readout.gain[QI]

    rows, shots = [], []
    print(f"{'gain':>7} {'|c|/gain':>9} {'arg c':>7} {'sigma':>7} {'outside':>8}")
    for gain in GAINS:
        expt = meas.HistogramExperiment(
            cfg_dict, qi=QI, go=False, check_f=False,
            prefix=f"ground_vs_readout_gain_qubit{QI}",
            params={"shots": SHOTS, "gain": float(gain),
                    "reset_mode": RESET_MODE, "check_e": False},
        )
        # One config yml per run folder, not per point (CLAUDE.md
        # "One config yml per run").
        suppress_duplicate_config(expt, out_dir)
        data = expt.acquire(progress=not args.no_progress)
        data["readout_gain"] = float(gain)
        expt.save_data(data)
        expt.save_config()

        z = np.asarray(data["Ig"]).flatten() + 1j * np.asarray(data["Qg"]).flatten()
        center, sigma, outside = blob_stats(z)
        rows.append({"gain": float(gain), "center_I": center.real,
                     "center_Q": center.imag, "center_mag_per_gain": abs(center) / gain,
                     "center_phase_deg": np.degrees(np.angle(center)),
                     "sigma": sigma, "outside_frac": outside, "file": Path(expt.fname).name})
        shots.append(z / gain)
        print(f"{gain:7.4f} {abs(center) / gain:9.2f} {rows[-1]['center_phase_deg']:7.1f} "
              f"{sigma:7.3f} {outside:8.4f}")

    with (out_dir / "ground_vs_readout_gain.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # IQ histograms, one per gain, in units of signal per unit gain.
    ncol = 5
    nrow = int(np.ceil(len(GAINS) / ncol))
    fig_iq, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow),
                                constrained_layout=True)
    for ax, z, row in zip(axes.flat, shots, rows):
        # Centred on the blob, wide enough for 99.9% of shots so outliers show.
        c = (row["center_I"] + 1j * row["center_Q"]) / row["gain"]
        half = 1.3 * np.percentile(np.abs(z - c), 99.9)
        ax.hist2d(z.real, z.imag, bins=80,
                      range=[[c.real - half, c.real + half], [c.imag - half, c.imag + half]],
                      cmap="viridis", norm=plt.matplotlib.colors.LogNorm())
        ax.set_aspect("equal")
        ax.text(0.03, 0.95, f"gain {row['gain']:.3g}\noutside {row['outside_frac']:.3f}",
                transform=ax.transAxes, va="top", fontsize="small")
    for ax in axes.flat[len(GAINS):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("I / gain")
    for ax in axes[:, 0]:
        ax.set_ylabel("Q / gain")
    fig_iq.savefig(out_dir / "ground_vs_readout_gain_iq.png", dpi=120)

    # Summary vs gain.
    g = np.array([r["gain"] for r in rows])
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.4), constrained_layout=True)
    ax[0].plot(g, [r["center_mag_per_gain"] for r in rows], "o-")
    ax[0].set_ylabel("|blob centre| / gain")
    ax[1].plot(g, [r["center_phase_deg"] for r in rows], "o-")
    ax[1].set_ylabel("blob phase (deg)")
    ax[2].plot(g, [r["sigma"] for r in rows], "o-")
    ax[2].set_ylabel("blob sigma (raw I/Q)")
    ax[3].plot(g, [r["outside_frac"] for r in rows], "o-")
    ax[3].axhline(np.exp(-OUTLIER_SIGMA**2 / 2), color="0.5", ls="--",
                  label=f"gaussian, >{OUTLIER_SIGMA:g} sigma")
    ax[3].set_yscale("log")
    ax[3].set_ylabel(f"fraction > {OUTLIER_SIGMA:g} sigma from blob")
    ax[3].legend(fontsize="small")
    for a in ax:
        a.set_xscale("log")
        a.set_xlabel("readout gain")
        a.axvline(gain_cfg, color="tab:red", ls=":", lw=1)
        a.grid(alpha=0.3)
    fig.savefig(out_dir / "ground_vs_readout_gain.png", dpi=120)

    print(f"\nconfig readout gain {gain_cfg} (dotted line)")
    print(f"  {out_dir / 'ground_vs_readout_gain.csv'}")
    if not args.no_show:
        wait_for_plots()
