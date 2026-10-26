"""
Master-equation check of the photon-number analyses on a buffer occupation known by construction.

Dispersive qubit + buffer (qutip), in the frame of a buffer drive at f_g:
    H = chi a^dag a |e><e| + eps (a + a^dag)  [+ probe]
with buffer decay kappa, qubit T1 and pure dephasing. n_g = eps^2/(kappa/2)^2.

1. Ramsey: |<sigma_->| decay with the drive on, minus the eps = 0 rate, against
   buffer_dephasing_per_photon * n_g (run 34: ratio 1.000 at chi/kappa ~ 5).
2. PNS: weak finite probe spectra at several n_g, fitted with pns_global.fit_arrays
   (run 34 parameters: c2 302 for a true 320, ~5% low from probe compression).

    python sim_photon_cal.py --ramsey-only
    python sim_photon_cal.py --kappa 0.182 --chi -1.29
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import qutip as qt

HERE = Path(__file__).resolve().parent
REPO_DIR = HERE.parents[2]
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from slab_qick_calib.experiments.single_qubit.t2 import buffer_dephasing_per_photon  # noqa: E402
from pns_global import fit_arrays  # noqa: E402


def build(kappa_mhz, chi_mhz, t1, t2, n_max):
    a = qt.tensor(qt.qeye(2), qt.destroy(n_max))
    pe = qt.tensor(qt.basis(2, 1) * qt.basis(2, 1).dag(), qt.qeye(n_max))
    sm = qt.tensor(qt.basis(2, 0) * qt.basis(2, 1).dag(), qt.qeye(n_max))
    kap, chi = 2 * np.pi * kappa_mhz, 2 * np.pi * chi_mhz
    gamma_phi = 1 / t2 - 1 / (2 * t1)
    c_ops = [np.sqrt(kap) * a, np.sqrt(1 / t1) * sm, np.sqrt(gamma_phi / 2) * 2 * pe]
    return a, pe, sm, kap, chi, c_ops


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kappa", type=float, default=0.2457, help="buffer FWHM (MHz)")
    parser.add_argument("--chi", type=float, default=-1.2933, help="full g->e shift (MHz)")
    parser.add_argument("--t1", type=float, default=61.5, help="qubit T1 (us)")
    parser.add_argument("--t2", type=float, default=15.0, help="qubit Ramsey T2, no buffer (us)")
    parser.add_argument("--c2", type=float, default=320.0, help="true photons per gain^2 for the PNS rows")
    parser.add_argument("--probe-gain", type=float, default=0.03)
    parser.add_argument("--pi-gain", type=float, default=0.110, help="probe gain for a pi in --length")
    parser.add_argument("--length", type=float, default=20.0, help="probe length (us)")
    parser.add_argument("--ramsey-only", action="store_true")
    args = parser.parse_args()

    a, pe, sm, kap, chi, c_ops = build(args.kappa, args.chi, args.t1, args.t2, 10)
    opts = qt.Options(nsteps=100000, atol=1e-9, rtol=1e-7)
    eps = lambda n_g: np.sqrt(n_g) * kap / 2

    def ramsey_rate(n_g):
        H = chi * a.dag() * a * pe + eps(n_g) * (a + a.dag())
        psi0 = qt.tensor((qt.basis(2, 0) + qt.basis(2, 1)).unit(), qt.basis(10, 0))
        t = np.linspace(0, 25, 501)
        coh = np.abs(qt.mesolve(H, psi0, t, c_ops, [sm.dag()], options=opts).expect[0])
        return -np.polyfit(t[t > 5], np.log(coh[t > 5]), 1)[0]

    print("== Ramsey: buffer dephasing vs the formula (drive at f_g)")
    base = ramsey_rate(0.0)
    per_photon, _ = buffer_dephasing_per_photon(args.kappa, args.chi, 0.0)
    for n_g in (0.1, 0.29, 0.5):
        d = ramsey_rate(n_g) - base
        print(f"   n_g {n_g:.2f}: simulated {d:.4f} /us, formula {per_photon * n_g:.4f} /us, "
              f"ratio {d / (per_photon * n_g):.3f}")
    if args.ramsey_only:
        return

    print("== PNS: simulated spectra -> pns_global.fit_arrays")
    omega = np.pi * args.probe_gain / args.pi_gain / args.length
    n_list = np.array([0.0, 0.05, 0.15, 0.3, 0.5, 0.8])
    deltas = np.arange(-2.7, 0.3, 0.01)
    spectra = []
    for n_g in n_list:
        alpha = -1j * eps(n_g) / (kap / 2)
        rho0 = qt.ket2dm(qt.tensor(qt.basis(2, 0), qt.coherent(10, alpha)))
        row = []
        for d in deltas:
            H = (chi * a.dag() * a * pe + eps(n_g) * (a + a.dag())
                 - 2 * np.pi * d * pe + omega / 2 * (sm + sm.dag()))
            row.append(qt.mesolve(H, rho0, [0, args.length], c_ops, [pe], options=opts).expect[0][-1])
        spectra.append(row)
        print(f"   n_g {n_g:.2f}: peak P_e {max(row):.4f}")
    fit_arrays(4132.5 + deltas, np.sqrt(n_list / args.c2), np.array(spectra), args.length,
               label=f"simulated (true c2 {args.c2:g})")


if __name__ == "__main__":
    main()
