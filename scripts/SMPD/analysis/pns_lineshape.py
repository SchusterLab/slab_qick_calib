"""
Photon-number-splitting line shapes and the per-row comb fit, for weak finite qubit probes.

A weak drive of length L on a line dephasing at rate G (HWHM, rad/us), detuning D,
excites to second order in the drive

    P(D) ∝ Re[ L/(G - iD) - (1 - exp(-(G - iD) L)) / (G - iD)^2 ]

whose area over D does not depend on G: sinc^2 when G L << 1, L x Lorentzian when
G L >> 1. `line_shape` is that, normalised to unit area in MHz, so a comb line n
contributes A * Poisson(n) * line_shape -- line *areas* carry the photon weights, as
they do for a weak probe. `voigt_line` adds a shared Gaussian (quasi-static qubit
frequency noise). CLAUDE.md "Probe power and the PNS comb".

    python pns_lineshape.py 549          # per-row fit, shared vs free amplitude
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
SMPD_DIR = HERE.parent
DEFAULT_RUN = "run34_Waymar"

VOIGT_GRID = np.linspace(-1.5, 1.5, 6001)  # MHz, for the Gaussian convolution


def line_shape(df_mhz, hwhm_mhz, length_us):
    """Unit-area (per MHz) weak-drive line for a probe of length_us."""
    d = 2 * np.pi * np.asarray(df_mhz, dtype=float)
    z = 2 * np.pi * hwhm_mhz - 1j * d
    p = np.real(length_us / z - (1 - np.exp(-z * length_us)) / z**2)
    return 2 * p / length_us


def voigt_line(df_mhz, hwhm_mhz, sigma_g_mhz, length_us):
    """line_shape convolved with a Gaussian of standard deviation sigma_g_mhz."""
    base = line_shape(VOIGT_GRID, hwhm_mhz, length_us)
    if sigma_g_mhz > 1e-5:
        kernel = np.exp(-0.5 * (VOIGT_GRID / sigma_g_mhz) ** 2)
        base = np.convolve(base, kernel / kernel.sum(), mode="same")
    return np.interp(df_mhz, VOIGT_GRID, base)


def comb(f, f0, n_bar, chi, chi_prime, g0, dg, length_us, sigma_g=0.0, n_max=12):
    """Poisson comb: line n at f0 - (2 n chi + chi' n^2), HWHM g0 + n dg (chi = half spacing)."""
    w = poisson.pmf(np.arange(n_max), n_bar)
    out = np.zeros_like(np.asarray(f, dtype=float))
    for k in range(n_max):
        if w[k] < 1e-6:
            continue
        df = f - (f0 - (2 * k * chi + chi_prime * k**2))
        line = (voigt_line(df, g0 + k * dg, sigma_g, length_us) if sigma_g > 0
                else line_shape(df, g0 + k * dg, length_us))
        out += w[k] * line
    return out


def stark_h5(run_dir, number):
    """(xpts, stark_gain, p_e, probe length) from a StarkSpec run's h5."""
    folders = sorted((run_dir / "data").glob(f"#{number}_*"))
    files = sorted(folders[0].glob("*.h5")) if len(folders) == 1 else []
    if len(files) != 1:
        raise SystemExit(f"Expected one h5 for #{number} under {run_dir / 'data'}")
    with h5py.File(files[0], "r") as h:
        cfg = json.loads(h.attrs["config"])
        return h["xpts"][()], h["stark_gain"][()], h["p_e"][()], float(cfg["expt"]["length"])


def fit_rows(x, g, pe, length_us, share_amp=True, n_max=12):
    """
    Comb fit with a free n_bar per row: f0, chi, chi', dgamma shared; n=0 HWHM and
    baseline per row; one amplitude for all rows (share_amp) or one per row.
    Returns (n_bar, n_bar_err, c2, c2_err, c0, shared params).
    """
    rows = len(g)
    dev0 = pe[0] - np.median(pe[0])
    f0 = x[np.argmax(np.abs(dev0))]
    amp0 = dev0[np.argmax(np.abs(dev0))] * 0.03
    npr = 3 if share_amp else 4
    x0, lo, hi = [f0, 0.6, 0.0, 0.06, amp0], [-np.inf, 0.3, -0.5, 0.0, -np.inf], [np.inf, 1.0, 0.5, 1.0, np.inf]
    for r in range(rows):
        x0 += [max(250 * g[r] ** 2, 1e-3), 0.02 + 25 * g[r] ** 2, float(np.median(pe[r]))]
        lo += [0, 0.003, -np.inf]
        hi += [8, 2, np.inf]
        if not share_amp:
            x0, lo, hi = x0 + [amp0], lo + [-np.inf], hi + [np.inf]

    def resid(p):
        f0, chi, chp, dg, amp = p[:5]
        out = []
        for r in range(rows):
            q = p[5 + npr * r: 5 + npr * (r + 1)]
            a = amp if share_amp else q[3]
            out.append(q[2] + a * comb(x, f0, q[0], chi, chp, q[1], dg, length_us, n_max=n_max) - pe[r])
        return np.concatenate(out)

    res = least_squares(resid, np.array(x0), bounds=(lo, hi), x_scale="jac")
    cov = np.linalg.pinv(res.jac.T @ res.jac) * (res.fun @ res.fun) / (res.fun.size - res.x.size)
    err = np.sqrt(np.abs(np.diag(cov)))
    nb, nbe = res.x[5::npr], err[5::npr]
    ok = nbe > 0
    w = 1 / nbe[ok] ** 2
    design = np.column_stack([np.ones(ok.sum()), g[ok] ** 2])
    normal = design.T @ (w[:, None] * design)
    c = np.linalg.solve(normal, design.T @ (w * nb[ok]))
    red = max(1.0, float(((nb[ok] - design @ c) ** 2 * w).sum()) / max(ok.sum() - 2, 1))
    c2_err = float(np.sqrt(np.linalg.inv(normal)[1, 1] * red))
    return nb, nbe, float(c[1]), c2_err, float(c[0]), res.x[:5]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=int, nargs="+", help="StarkSpec run numbers")
    parser.add_argument("--run", default=DEFAULT_RUN, help="Data/<run> folder name")
    args = parser.parse_args()
    run_dir = SMPD_DIR / "Data" / args.run
    for number in args.runs:
        x, g, pe, length = stark_h5(run_dir, number)
        for share in (True, False):
            nb, nbe, c2, c2e, c0, shared = fit_rows(x, g, pe, length, share_amp=share)
            print(f"#{number} ({'shared' if share else 'per-row'} amplitude, probe {length:g} us): "
                  f"c2 = {c2:.1f} +/- {c2e:.1f}, c0 = {c0:.3f}, 2chi = {2 * shared[1]:.4f}, "
                  f"dgamma = {shared[3]:.4f} MHz")


if __name__ == "__main__":
    sys.exit(main())
