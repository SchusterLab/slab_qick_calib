"""Buffer energy decay rate kappa from its driven response and free ring-down, recorded on the buffer ADC."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
import slab_qick_calib.experiments as meas

from common import finish_experiment, parse_args, prepare_experiment, wait_for_plots


def reflection_trace(p, t):
    """
    Single-port resonator driven by a square pulse from t_on to t_off, seen in reflection.
    p = (kappa MHz FWHM, kappa_c/kappa, detuning MHz, t_on, t_off, gain re/im, offset re/im).
    Field: da/dt = -(kappa/2 - i delta) a + sqrt(kappa_c) a_in; output a_in - sqrt(kappa_c) a.
    """
    k, eta, det, t_on, t_off, gr, gi, br, bi = p
    s = np.pi * k - 2j * np.pi * det
    kc = 2 * np.pi * k * eta
    a = np.zeros_like(t, dtype=complex)
    on = (t >= t_on) & (t < t_off)
    a[on] = np.sqrt(kc) / s * (1 - np.exp(-s * (t[on] - t_on)))
    off = t >= t_off
    a[off] = np.sqrt(kc) / s * (1 - np.exp(-s * (t_off - t_on))) * np.exp(-s * (t[off] - t_off))
    return (gr + 1j * gi) * (on - np.sqrt(kc) * a) + (br + 1j * bi)


def fit_trace(t, z, p0, mask):
    lo = [0.03, 0.2, -0.5, p0[3] - 0.2, p0[4] - 0.2] + [-np.inf] * 4
    hi = [0.8, 1.0, 0.5, p0[3] + 0.2, p0[4] + 0.2] + [np.inf] * 4

    def resid(p):
        r = reflection_trace(p, t[mask]) - z[mask]
        return np.concatenate([r.real, r.imag])

    best = None
    for k0 in (0.17, 0.21, 0.25):
        for d0 in (-0.05, 0.0, 0.05):
            r = least_squares(resid, [k0, 0.95, d0, *p0[3:]], bounds=(lo, hi))
            if best is None or r.cost < best.cost:
                best = r
    cov = np.linalg.pinv(best.jac.T @ best.jac) * (best.fun @ best.fun) / (best.fun.size - 9)
    return best.x, np.sqrt(np.diag(cov))


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 0  # buffer line
    JPA_ON = False
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(args, __file__, qi, JPA_ON)

    # Ramsey's dephasing/shift ratio implies kappa ~0.21 but the ResSpec linewidth is 0.24;
    # this measures energy decay directly. Keep the field linear: a const pulse gives ~1000
    # photons per gain^2 at f_g here, and near n_crit ~100 the qubit-induced Kerr shifts the
    # buffer by ~kappa (#642 at gain 0.3: drive-on and decay segments disagreed, 0.27 vs 0.16).
    # CLAUDE.md "Photon number from Ramsey dephasing (scripts 38, 41)"
    DRIVE_LENGTH = 2.0  # us; decay is exponential from any amplitude, so no need for steady state
    params = {
        "frequency": auto_cfg.device.readout.frequency[qi],
        "gain": 0.05,  # ~2.5 photons at f_g
        "drive_length": DRIVE_LENGTH,
        "readout_length": DRIVE_LENGTH + 7.0,  # capture: drive plus ~5 field 1/e times
        "rounds": 100000,  # ~2 s; the signal is ~6x smaller than at gain 0.3
    }
    ringdown = meas.ToFCalibrationExperiment(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(ringdown, args, wait=False)  # no update(): trig_offset stays as is

    t = ringdown.data["xpts"]
    z = ringdown.data["i"] + 1j * ringdown.data["q"]
    # Seed the edges from the trace: the drive arrives where |z| first jumps
    amp = np.abs(z - z[-50:].mean())
    t_on = t[np.argmax(amp > 0.5 * amp.max())]
    p0 = [0.2, 0.95, 0.0, t_on, t_on + DRIVE_LENGTH, 0.0, 0.0, z[-50:].mean().real, z[-50:].mean().imag]
    p0[5] = amp.max() / 2
    p_all, e_all = fit_trace(t, z, p0, np.ones_like(t, bool))
    t_on, t_off = p_all[3], p_all[4]
    pre = t < t_on
    p_on, e_on = fit_trace(t, z, p_all, pre | ((t > t_on + 0.05) & (t < t_off - 0.02)))
    p_dec, e_dec = fit_trace(t, z, p_all, pre | (t > t_off + 0.05))
    print(f"Full trace:   kappa = {p_all[0]:.4f} +/- {e_all[0]:.4f} MHz, kappa_c/kappa = {p_all[1]:.3f}, "
          f"detuning = {p_all[2]:+.4f} MHz, drive {t_off - t_on:.3f} us")
    print(f"Drive-on:     kappa = {p_on[0]:.4f} +/- {e_on[0]:.4f} MHz, detuning = {p_on[2]:+.4f}")
    print(f"Decay only:   kappa = {p_dec[0]:.4f} +/- {e_dec[0]:.4f} MHz, detuning = {p_dec[2]:+.4f}")
    print("  (a linear resonator gives the same kappa in all three; config kappa[0] = "
          f"{auto_cfg.device.readout.kappa[qi]:.4f})")

    fig, (ax_amp, ax_iq) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    model = reflection_trace(p_all, t)
    ax_amp.plot(t, np.abs(z - p_all[7] - 1j * p_all[8]), ".", ms=2, label="data")
    ax_amp.plot(t, np.abs(model - p_all[7] - 1j * p_all[8]), "-", lw=1.2,
                label=f"$\\kappa$ = {p_all[0]:.4f} MHz")
    ax_amp.set_ylabel("|signal - offset| (ADC units)")
    ax_amp.legend(fontsize="small")
    ax_iq.plot(t, z.real, ".", ms=2, label="I")
    ax_iq.plot(t, z.imag, ".", ms=2, label="Q")
    ax_iq.plot(t, model.real, "k-", lw=1)
    ax_iq.plot(t, model.imag, "k-", lw=1)
    ax_iq.set_xlabel("Time (us)")
    ax_iq.set_ylabel("ADC units")
    ax_iq.legend(fontsize="small")
    fig.tight_layout()
    out = Path(cfg_dict["metadata_path"]) / "ringdown_fit.png"
    fig.savefig(out)
    print(f"Saved {out}")
    if not args.no_show:
        wait_for_plots()
