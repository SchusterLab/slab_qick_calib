"""
Refit a BufferDephasingSpec (script 41) run with kappa fixed or free, from both observables or one.

41's own fit frees kappa, which its data constrain only weakly (run 34 #554: 0.207
free, rms 1.01, vs 0.241 fixed, rms 1.11) and which the T2 window bias pulls.
With kappa from the ring-down (script 42) fixed, the photons per gain^2 are better
determined. Prints both-observable, shift-only and dephasing-only results.
CLAUDE.md "Photon number from Ramsey dephasing (scripts 38, 41)".

    python refit_dephasing_spec.py 554 --kappa 0.182
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
SMPD_DIR = HERE.parent
REPO_DIR = HERE.parents[2]
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from slab_qick_calib.experiments.single_qubit.t2 import (  # noqa: E402
    BufferDephasingSpec,
    buffer_dephasing_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_number", type=int, help="BufferDephasingSpec run number")
    parser.add_argument("--run", default="run34_Waymar", help="Data/<run> folder name")
    parser.add_argument("--kappa", type=float, help="fix kappa (MHz FWHM); omit to fit it")
    args = parser.parse_args()

    path = glob.glob(str(SMPD_DIR / "Data" / args.run / "data" / f"#{args.run_number}_*" / "*.h5"))[0]
    spec = BufferDephasingSpec.from_h5file(path)
    d = spec.analyze(fit=False)
    o = ~np.asarray(d["reference"]).astype(bool)
    fr, dg, ge, ds = d["frequency_b_pts"][o], d["d_gamma"][o], d["gamma_err"][o], d["d_shift"][o]
    gain2 = spec.cfg.expt.gain_b ** 2
    readout = spec.cfg.device.readout
    seed_kappa = args.kappa if args.kappa is not None else readout.kappa[0]

    # One weighting pass from a free fit, as fit_buffer_dephasing_spectrum does
    def run(use_gamma, use_shift, sign, gamma_scale, sigma_shift):
        def resid(p):
            n_on, chi, f_g = p[:3]
            kappa = args.kappa if args.kappa is not None else p[3]
            g, s = buffer_dephasing_model(fr, n_on, kappa, chi, f_g, sign=sign)
            parts = []
            if use_gamma:
                parts.append((g - dg) / (ge * gamma_scale))
            if use_shift:
                parts.append((s - ds) / sigma_shift)
            return np.concatenate(parts)
        x0 = [0.25, readout.chi[0], readout.frequency[0]] + ([] if args.kappa is not None else [seed_kappa])
        res = least_squares(resid, x0, x_scale="jac")
        cov = np.linalg.pinv(res.jac.T @ res.jac) * max(1.0, 2 * res.cost / (res.fun.size - len(x0)))
        return res, np.sqrt(np.diag(cov))

    sign = min((1.0, -1.0), key=lambda s: run(True, True, s, 1.0, 1e-3)[0].cost)
    res, _ = run(True, True, sign, 1.0, 1e-3)
    n = len(fr)
    gamma_scale = max(1.0, float(np.sqrt(np.mean(res.fun[:n] ** 2))))
    sigma_shift = 1e-3 * float(np.sqrt(np.mean(res.fun[n:] ** 2)))
    print(f"#{args.run_number}: kappa {'fixed at ' + str(args.kappa) if args.kappa is not None else 'free'}, "
          f"dephasing errors x{gamma_scale:.2f}, shift sigma {1e3 * sigma_shift:.2f} kHz")
    for label, ug, us in (("both", True, True), ("shift only", False, True), ("dephasing only", True, False)):
        res, err = run(ug, us, sign, gamma_scale, sigma_shift)
        kappa = args.kappa if args.kappa is not None else res.x[3]
        print(f"   {label:15s} {res.x[0] / gain2:6.0f} +/- {err[0] / gain2:4.0f} photons/gain^2 at f_g   "
              f"kappa {kappa:.4f}  chi {res.x[1]:.4f}  f_g {res.x[2]:.4f}  "
              f"rms {np.sqrt(np.mean(res.fun ** 2)):.2f}")


if __name__ == "__main__":
    main()
