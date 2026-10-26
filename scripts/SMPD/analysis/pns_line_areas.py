"""
Per-row area of the main qubit line vs buffer gain, Lorentzian vs Voigt line.

Each row gets its own centre, Lorentz HWHM, area and baseline (finite-pulse weak-drive
line, optionally convolved with a shared Gaussian). Areas are relative to the zero-gain
row. Off resonance the main line keeps ~exp(-|alpha_g - alpha_e|^2) of its weight (the
rest goes to sidebands), not exp(-n_g): on run 34 #555 at gain 0.1 that is ~0.77
against 0.86-0.89 measured, so area loss there is not a PNS anomaly. CLAUDE.md
"Probe power and the PNS comb".

    python pns_line_areas.py 555
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import least_squares

from pns_lineshape import DEFAULT_RUN, SMPD_DIR, stark_h5, voigt_line


def fit_rows(x, pe, length_us, sigma_g_free):
    rows = len(pe)
    x0 = [0.005 if sigma_g_free else 0.0]
    for r in range(rows):
        dev = pe[r] - np.median(pe[r])
        i = np.argmax(np.abs(dev))
        x0 += [x[i], 0.02, dev[i] * 0.03, float(np.median(pe[r]))]

    def resid(p):
        sg = abs(p[0]) if sigma_g_free else 0.0
        return np.concatenate([
            p[4 + 4 * r] + p[3 + 4 * r] * voigt_line(x - p[1 + 4 * r], abs(p[2 + 4 * r]), sg, length_us)
            - pe[r] for r in range(rows)])

    res = least_squares(resid, np.array(x0), x_scale="jac")
    p = res.x
    return (abs(p[0]) if sigma_g_free else 0.0, p[3::4], np.abs(p[2::4]), p[1::4],
            float(np.sqrt(np.mean(res.fun ** 2))))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_number", type=int, help="StarkSpec run number")
    parser.add_argument("--run", default=DEFAULT_RUN, help="Data/<run> folder name")
    args = parser.parse_args()
    x, g, pe, length = stark_h5(SMPD_DIR / "Data" / args.run, args.run_number)
    for label, free in (("finite-pulse Lorentzian", False), ("Voigt, shared sigma_g free", True)):
        sg, areas, hwhm, centre, rms = fit_rows(x, pe, length, free)
        print(f"{label}: sigma_g = {1e3 * sg:.1f} kHz, rms {rms:.4f}")
        print("   gain    area/area0   Lorentz HWHM (kHz)   centre (MHz)")
        for gi, a, w, c in zip(g, areas / areas[0], hwhm, centre):
            print(f"   {gi:.3f}   {a:.3f}        {1e3 * w:6.1f}            {c:.4f}")


if __name__ == "__main__":
    main()
