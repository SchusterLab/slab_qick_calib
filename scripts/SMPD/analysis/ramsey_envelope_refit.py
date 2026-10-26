"""
Refit a T2PhotonCal (script 38) run's Ramsey traces with an exponential x Gaussian envelope.

T2PhotonCal fits exp(-t/T2). The intrinsic decay also has a Gaussian part (slow
frequency noise; run 34 T2E 26 us vs T2R 15 us), and buffer-on points use windows shrunk
to 3 T2, so they capture less of it than the gain-0 reference: d(1/T2)/d(gain^2) reads
low (20-35% on run 34 #553). Here every trace is fitted with
exp(-Gamma t - (t/T_g)^2), T_g shared and chosen from the gain-0 traces, and Gamma
vs gain^2 converts to on-resonance photons with buffer_dephasing_per_photon.
CLAUDE.md "Photon number from Ramsey dephasing (scripts 38, 41)".

    python ramsey_envelope_refit.py 553 --kappa 0.182 --chi -1.293 --f-g 6682.295
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
SMPD_DIR = HERE.parent
REPO_DIR = HERE.parents[2]
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from slab_qick_calib.experiments.single_qubit.t2 import buffer_dephasing_per_photon  # noqa: E402


def envelope_model(t_g):
    def model(t, a, f, ph, c, gamma, s):
        return a * (np.sin(2 * np.pi * f * t + ph) + s) * np.exp(-gamma * t - (t / t_g) ** 2) + c
    return model


def fit_gamma(d, i, t_g):
    fp = d["fit_pars"][i]
    p0 = [fp[0], fp[1], np.deg2rad(fp[2]), fp[4], 1 / fp[3] if fp[3] > 0 else 0.05, fp[5]]
    lo = [-np.inf, 0, -np.inf, -np.inf, 0, -np.inf]
    p, _ = curve_fit(envelope_model(t_g), d["xpts"][i], d["ydata"][i], p0=p0,
                     bounds=(lo, np.inf), maxfev=20000)
    rss = float(np.sum((d["ydata"][i] - envelope_model(t_g)(d["xpts"][i], *p)) ** 2))
    return p[4], rss


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_number", type=int, help="T2PhotonCal run number")
    parser.add_argument("--run", default="run34_Waymar", help="Data/<run> folder name")
    parser.add_argument("--kappa", type=float, help="buffer FWHM (MHz), default the run's config")
    parser.add_argument("--chi", type=float, help="full g->e shift (MHz), default the run's config")
    parser.add_argument("--f-g", type=float, help="buffer f_g (MHz), default the run's config")
    parser.add_argument("--t-g", type=float, nargs="*", default=[1e6, 40, 30, 25, 20, 17, 15],
                        help="Gaussian times to try (us); 1e6 = pure exponential")
    args = parser.parse_args()

    path = glob.glob(str(SMPD_DIR / "Data" / args.run / "data" / f"#{args.run_number}_*" / "*.h5"))[0]
    with h5py.File(path, "r") as h:
        d = {k: h[k][()] for k in h.keys()}
        cfg = json.loads(h.attrs["config"])
    readout = cfg["device"]["readout"]
    kappa = args.kappa if args.kappa is not None else readout["kappa"][0]
    chi = args.chi if args.chi is not None else readout["chi"][0]
    f_g = args.f_g if args.f_g is not None else readout["frequency"][0]
    detuning = cfg["expt"]["frequency_b"] - f_g
    per_photon, _ = buffer_dephasing_per_photon(kappa, chi, detuning)
    gains = d["gain_b_pts"]
    ref = d["reference"].astype(bool) if "reference" in d else np.zeros(len(gains), bool)
    zero = np.flatnonzero(gains == 0)

    print(f"#{args.run_number}: drive at f_g{detuning:+.4f} MHz, kappa {kappa:.4f}, chi {chi:.4f}")
    print("   T_g (us)   gain-0 rss   dGamma/dg^2 (1/us)   photons/gain^2 at f_g")
    for t_g in args.t_g:
        rss0 = sum(fit_gamma(d, i, t_g)[1] for i in zero)
        gamma = np.array([fit_gamma(d, i, t_g)[0] for i in range(len(gains))])
        popt, pcov = np.polyfit(gains[~ref] ** 2, gamma[~ref], 1, cov=True)
        err = np.sqrt(pcov[0, 0])
        label = "exp only" if t_g > 1e5 else f"{t_g:6.0f}"
        print(f"   {label:>8}   {rss0:9.4f}    {popt[0]:7.1f} +/- {err:5.1f}       "
              f"{popt[0] / per_photon:6.0f} +/- {err / per_photon:4.0f}")
    print("   (pick T_g at the gain-0 rss minimum)")


if __name__ == "__main__":
    main()
