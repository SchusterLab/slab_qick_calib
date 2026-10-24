"""
Joint fit of several T2PhotonCal (script 38) runs into one photon calibration.

Each run measures 1/T2,Ramsey against buffer gain^2 with the buffer driven at
the dressed-resonance midpoint. Runs differ in two ways that rescale the
photon number at a given gain, and both are known exactly:

- buffer attenuation (the RF-board attn1 + attn2 on the buffer DAC), a factor
  10**(-dB/10) in photons;
- buffer pulse shape: QICK plays a flat_top plateau at half a const pulse's
  amplitude for the same gain, so const gives 4x the photons.

Every gain is therefore mapped onto one reference -- a flat_top pulse at
REF_ATTN_DB -- as gain_ref^2 = r * gain^2, and all runs are fit together as

    1/T2_ij = 1/T2(0)_i + slope * r_i * gain_ij^2

with one intercept per run (the gain-0 T2 drifts between runs) and one shared
slope. The slope converts to photons with the same function T2PhotonCal uses,
`ramsey_photons_per_rate` at the runs' common drive detuning from f_g, giving
photons per unit reference gain^2 at the drive frequency and on resonance --
the |g>-dressed buffer frequency the FWM scripts drive at.

Each run is also fit on its own, so the printout shows whether the runs agree
before they are pooled.

    python combine_t2_photon_cal.py
    python combine_t2_photon_cal.py --runs 727 728 731 --ref-attn 20
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "Data"
REPO_DIR = HERE.parents[2]
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from slab_qick_calib.experiments.single_qubit.t2 import ramsey_photons_per_rate  # noqa: E402

RUN_DIR = "run33_Waymar"
RUNS = (676, 677, 727, 728, 731)
REF_ATTN_DB = 30
REF_PULSE = "flat_top"

# Photons at a const pulse relative to flat_top at the same gain.
PULSE_PHOTON_FACTOR = {"flat_top": 1.0, "const": 4.0}

# Reference categorical palette, fixed order; markers carry identity too.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")


def load_run(run_dir, number, max_gain=None):
    """
    Read one T2PhotonCal run: its data, attenuation and buffer pulse shape.

    `max_gain` drops buffer gains above it before fitting; see the note below
    on why the full range is not always linear.
    """
    meta = glob.glob(str(run_dir / "metadata_and_images" / f"#{number}_*"))
    data = glob.glob(str(run_dir / "data" / f"#{number}_*" / "*.h5"))
    if not meta or not data:
        raise SystemExit(f"Run #{number} not found under {run_dir}")
    with open(glob.glob(meta[0] + "/*.yml")[0]) as f:
        cfg = yaml.safe_load(f)
    with h5py.File(data[0], "r") as f:
        d = {k: f[k][()] for k in ("gain_b_pts", "t2", "t2_err", "xpts")}
        if "reference" in f:
            # The closing gain-0 drift check is not part of the fit
            keep = ~f["reference"][()].astype(bool)
            d = {k: v[keep] for k, v in d.items()}

    expt = cfg["expt"]
    soc = cfg["hw"]["soc"]
    readout = soc["dacs"]["readout"]
    # What the RF board was actually set to, checked against the config value
    applied = soc["rfboard_active"]["dac_readout"][str(readout["ch"][0])]
    attn = float(applied["attn1"]) + float(applied["attn2"])
    configured = float(readout["attn1"][0]) + float(readout["attn2"][0])
    if attn != configured:
        raise SystemExit(f"Run #{number}: buffer attenuation applied {attn} dB "
                         f"but configured {configured} dB")

    # pulse_type_b is only recorded from #731 on. Before that the buffer
    # shape shows in the wait axis: flat_top ramps add 0.06 us to every wait.
    pulse = expt.get("pulse_type_b")
    if pulse is None:
        offset = d["xpts"][0, 0] - expt["start"]
        pulse = "flat_top" if offset > 0.04 else "const"

    gain2 = d["gain_b_pts"] ** 2
    gamma = 1 / d["t2"]
    gamma_err = d["t2_err"] / d["t2"] ** 2
    if max_gain is not None:
        # 1/T2 is linear in gain^2 only while the dephasing stays in the
        # perturbative regime. Run 34 #216, taken at 0 dB, leaves it around
        # gain 0.75: beyond that 1/T2 takes off and then saturates, and a fit
        # over the whole range is a compromise through a curve rather than a
        # slope. Cut to the linear part and the slope means what the formula
        # says it does.
        keep = d["gain_b_pts"] <= max_gain
        gain2, gamma, gamma_err = gain2[keep], gamma[keep], gamma_err[keep]
        if keep.sum() < 3:
            raise SystemExit(f"Run #{number}: only {keep.sum()} points at or "
                             f"below gain {max_gain}; nothing to fit.")
    return {
        "number": number,
        "gain2": gain2,
        "gamma": gamma,
        "gamma_err": gamma_err,
        "attn": attn,
        "pulse": pulse,
        "kappa_b": expt["kappa_b"],
        "chi_b": expt["chi_b"],
        "frequency_b": expt["frequency_b"],
        "detuning": expt["frequency_b"] - cfg["device"]["readout"]["frequency"][0],
    }


def weighted_lstsq(design, y, sigma):
    """Weighted linear least squares; covariance scaled up by chi2_red if > 1."""
    w = 1 / sigma
    popt, *_ = np.linalg.lstsq(design * w[:, None], y * w, rcond=None)
    resid = (y - design @ popt) / sigma
    dof = max(len(y) - len(popt), 1)
    chi2_red = float(resid @ resid) / dof
    cov = np.linalg.inv((design * w[:, None]).T @ (design * w[:, None]))
    return popt, cov * max(chi2_red, 1.0), chi2_red, dof


def fit_joint(runs):
    """Shared slope in reference gain^2, one intercept per run."""
    x = np.concatenate([r["scale"] * r["gain2"] for r in runs])
    y = np.concatenate([r["gamma"] for r in runs])
    s = np.concatenate([r["gamma_err"] for r in runs])
    design = np.zeros((len(x), 1 + len(runs)))
    design[:, 0] = x
    row = 0
    for i, r in enumerate(runs):
        design[row:row + len(r["gain2"]), 1 + i] = 1
        row += len(r["gain2"])
    popt, cov, chi2_red, dof = weighted_lstsq(design, y, s)
    return popt, np.sqrt(np.diag(cov)), chi2_red, dof


def fit_single(run):
    """One run alone, slope already in reference gain^2."""
    x = run["scale"] * run["gain2"]
    design = np.column_stack([x, np.ones_like(x)])
    popt, cov, chi2_red, _ = weighted_lstsq(design, run["gamma"], run["gamma_err"])
    return popt[0], np.sqrt(cov[0, 0]), chi2_red


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", default=RUN_DIR)
    parser.add_argument("--runs", type=int, nargs="+", default=RUNS)
    parser.add_argument("--ref-attn", type=float, default=REF_ATTN_DB)
    parser.add_argument("--max-gain", type=float, default=None,
                        help="drop buffer gains above this before "
                             "fitting, to stay in the linear regime")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    run_dir = DATA_ROOT / args.run_dir
    runs = [load_run(run_dir, n, args.max_gain) for n in args.runs]

    # Within 1% on kappa/chi and 20 kHz (a tenth of a linewidth) on the drive
    for key, tol in (("kappa_b", 0.01), ("chi_b", 0.01), ("detuning", 0.02)):
        values = np.array([r[key] for r in runs])
        spread = np.ptp(values) / (abs(values.mean()) if key != "detuning" else 1)
        if spread > tol:
            raise SystemExit(f"Runs disagree on {key}: {sorted(set(values))}; "
                             "the joint fit assumes one buffer and one drive point.")
    for key in ("kappa_b", "chi_b", "frequency_b", "detuning"):
        runs[0][key] = float(np.mean([r[key] for r in runs]))
    photons_per_rate, on_resonance = ramsey_photons_per_rate(
        runs[0]["kappa_b"], runs[0]["chi_b"], runs[0]["detuning"])
    lever = on_resonance / photons_per_rate

    for r in runs:
        r["scale"] = (10 ** ((args.ref_attn - r["attn"]) / 10)
                      * PULSE_PHOTON_FACTOR[r["pulse"]])

    popt, perr, chi2_red, dof = fit_joint(runs)
    slope, slope_err = popt[0], perr[0]
    n_mid, n_mid_err = slope * photons_per_rate, slope_err * photons_per_rate

    ref = f"{REF_PULSE} gain at {args.ref_attn:g} dB"
    print(f"Reference: {ref}.  kappa/2pi = {runs[0]['kappa_b']} MHz, "
          f"full chi = {runs[0]['chi_b']} MHz, drive {runs[0]['frequency_b']:.3f} MHz "
          f"(f_g{runs[0]['detuning']:+.4f})")
    print(f"\n  run   attn  pulse     photons x   n at drive / ref gain^2   chi2_red")
    rows = []
    for r, intercept in zip(runs, popt[1:]):
        s1, e1, c1 = fit_single(r)
        pull = (s1 - slope) / np.hypot(e1, slope_err)
        print(f"  #{r['number']}  {r['attn']:4.0f}  {r['pulse']:8s}  {r['scale']:9.4g}   "
              f"{s1 * photons_per_rate:8.4f} +/- {e1 * photons_per_rate:.4f}"
              f"   ({pull:+.1f} sigma)   {c1:5.2f}")
        rows.append({"run": r["number"], "attn_db": r["attn"], "pulse": r["pulse"],
                     "photon_scale": r["scale"], "t2_0_us": 1 / intercept,
                     "n_mid_per_ref_gain2": s1 * photons_per_rate,
                     "n_mid_err": e1 * photons_per_rate, "chi2_red": c1})

    print(f"\nJoint fit, {len(runs)} runs ({dof} dof): chi2_red = {chi2_red:.2f}")
    print(f"  slope d(1/T2)/d(gain^2) = {slope:.5g} +/- {slope_err:.2g} /us per ({ref})^2"
          "  -> device.fwm.buffer_dephasing_ramsey"
          f" (with _detuning = {runs[0]['detuning']:.4f})")
    print(f"  at the drive  n_bar = {n_mid:.4f} +/- {n_mid_err:.4f} per ({ref})^2")
    print(f"  on resonance  n_bar = {n_mid * lever:.3f} +/- {n_mid_err * lever:.3f} "
          f"per ({ref})^2   (x{lever:.1f}, from kappa and chi)")
    for attn in sorted({r["attn"] for r in runs}):
        factor = 10 ** ((args.ref_attn - attn) / 10)
        print(f"    at {attn:g} dB: on resonance {n_mid * lever * factor:.3f}, "
              f"at the drive {n_mid * factor:.4f} per ({REF_PULSE} gain)^2")

    out = run_dir / "t2_photon_cal_combined"
    rows.append({"run": "joint", "attn_db": args.ref_attn, "pulse": REF_PULSE,
                 "photon_scale": 1.0, "t2_0_us": float("nan"),
                 "n_mid_per_ref_gain2": n_mid, "n_mid_err": n_mid_err,
                 "chi2_red": chi2_red})
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Excess dephasing (each run's own fitted intercept removed) against
    # reference gain^2. Left: everything; right: the 30 dB-scale region.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x_max = max(float(np.max(r["scale"] * r["gain2"])) for r in runs)
    for ax, xlim in zip(axes, (x_max * 1.05, min(1.05, x_max * 1.05))):
        for i, (r, intercept) in enumerate(zip(runs, popt[1:])):
            ax.errorbar(r["scale"] * r["gain2"], r["gamma"] - intercept,
                        yerr=r["gamma_err"], fmt=MARKERS[i], ms=6, lw=1, capsize=0,
                        color=COLORS[i], mec="white", mew=0.8,
                        label=f"#{r['number']}  {r['attn']:g} dB, {r['pulse']}")
        dense = np.linspace(0, xlim, 100)
        ax.plot(dense, slope * dense, color="#1f1f1f", lw=1.5,
                label=f"joint: {n_mid:.4f} $\\pm$ {n_mid_err:.4f} midpoint photons")
        ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
        ax.set_xlim(-0.02 * xlim, xlim)
        ax.set_xlabel(f"({ref})$^2$")
        ax.set_ylabel("$1/T_2 - 1/T_2(0)$ ($\\mu$s$^{-1}$)")
        ax.spines[["top", "right"]].set_visible(False)
    lo, hi = axes[1].get_xlim()
    in_zoom = np.concatenate([r["scale"] * r["gain2"] <= hi for r in runs])
    y_all = np.concatenate([r["gamma"] - b for r, b in zip(runs, popt[1:])])
    axes[1].set_ylim(np.min(y_all[in_zoom]) - 0.03, np.max(y_all[in_zoom]) + 0.03)
    axes[0].legend(fontsize="small", frameon=False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"\nSaved {out.with_suffix('.png').name} and {out.with_suffix('.csv').name} "
          f"in {run_dir}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
