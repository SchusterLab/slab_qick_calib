"""
Global PNS fit of a weak-probe StarkSpec run: buffer photons per gain^2 without kappa.

n_bar = c0 + c2 gain^2 is imposed across every buffer-gain row, with one line-area
scale shared by all rows (so the n=0 line's depletion counts, not only the faint
n=1 line), the finite-pulse weak-drive line shape and chi' = 0. Per row only the
n=0 HWHM and the baseline are free, so no row can run away. On master-equation
simulated spectra (sim_photon_cal.py) it recovers c2 to ~5%; on run 34 #549-#551 it
gives ~326 per gain^2 at the drive, matching the Ramsey and line-shift routes at
kappa = 0.182. --voigt adds a shared Gaussian (quasi-static qubit frequency noise);
on #549-#551 it does not move c2. CLAUDE.md "Probe power and the PNS comb".

    python pns_global.py 549 550 551
    python pns_global.py 551 --max-gain 0.05 --voigt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from pns_lineshape import DEFAULT_RUN, SMPD_DIR, comb, stark_h5


def fit_arrays(x, g, pe, length_us, max_gain=None, voigt=False, label=""):
    """Global fit; returns dict of c2, c0, 2chi, dgamma, sigma_g with errors."""
    keep = np.ones(len(g), bool) if max_gain is None else g <= max_gain + 1e-9
    g, pe = np.asarray(g)[keep], np.asarray(pe)[keep]
    rows = len(g)
    dev0 = pe[0] - np.median(pe[0])
    f0 = x[np.argmax(np.abs(dev0))]
    amp0 = dev0[np.argmax(np.abs(dev0))] * 0.03  # signed: dips (inverted readout) or peaks
    # shared: f0, chi (half spacing), dgamma, A, c0, c2, sigma_g
    x0 = [f0, 0.62, 0.1, amp0, 0.02, 300.0, 0.008 if voigt else 0.0]
    lo = [-np.inf, 0.4, 0.0, -np.inf, 0.0, 0.0, 0.0]
    hi = [np.inf, 0.9, 1.0, np.inf, 1.0, 5000.0, 0.1 if voigt else 1e-9]
    for r in range(rows):
        x0 += [0.02 + 30 * g[r] ** 2, float(np.median(pe[r]))]
        lo += [0.001, -np.inf]
        hi += [2.0, np.inf]

    def resid(p):
        f0, chi, dg, amp, c0, c2, sg = p[:7]
        return np.concatenate([
            p[8 + 2 * r] + amp * comb(x, f0, c0 + c2 * g[r] ** 2, chi, 0.0, p[7 + 2 * r], dg,
                                      length_us, sigma_g=sg, n_max=8) - pe[r]
            for r in range(rows)])

    res = least_squares(resid, np.array(x0), bounds=(lo, hi), x_scale="jac")
    cov = np.linalg.pinv(res.jac.T @ res.jac) * (res.fun @ res.fun) / (res.fun.size - res.x.size)
    err = np.sqrt(np.abs(np.diag(cov)))
    f0, chi, dg, amp, c0, c2, sg = res.x[:7]
    out = {"c2": c2, "c2_err": err[5], "c0": c0, "c0_err": err[4], "two_chi": 2 * chi,
           "two_chi_err": 2 * err[1], "dgamma": dg, "sigma_g": sg,
           "n0_hwhm": res.x[7::2], "rms": float(np.sqrt(np.mean(res.fun ** 2)))}
    print(f"{label} rows={rows} (max gain {g.max():.3f}): c2 = {c2:.1f} +/- {err[5]:.1f}, "
          f"c0 = {c0:.3f} +/- {err[4]:.3f}, 2chi = {2 * chi:.4f} +/- {2 * err[1]:.4f}, "
          f"dgamma = {dg:.4f}" + (f", sigma_g = {1e3 * sg:.1f} kHz" if voigt else "")
          + f", rms = {out['rms']:.4f}")
    print("      n=0 HWHM (kHz): " + ", ".join(f"{1e3 * v:.0f}" for v in out["n0_hwhm"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=int, nargs="+", help="StarkSpec run numbers")
    parser.add_argument("--run", default=DEFAULT_RUN, help="Data/<run> folder name")
    parser.add_argument("--max-gain", type=float, help="fit buffer gains up to this only")
    parser.add_argument("--voigt", action="store_true", help="add a shared Gaussian line width")
    args = parser.parse_args()
    run_dir = SMPD_DIR / "Data" / args.run
    for number in args.runs:
        x, g, pe, length = stark_h5(run_dir, number)
        fit_arrays(x, g, pe, length, args.max_gain, args.voigt, label=f"#{number}")


if __name__ == "__main__":
    main()
