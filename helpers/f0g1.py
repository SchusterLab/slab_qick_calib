"""
f0-g1 reset: config math.

Pure config-in / numbers-out helpers for the all-microwave unconditional reset
of Magnard et al., arXiv:1801.07689. They live in `helpers/` rather than beside
the experiment classes so the base classes can use them: `qick_experiment.py`
needs `f0g1_reset_params` to fill in the drive settings when
`cfg.expt.reset_mode == "f0g1"`, and `experiments/smpd/f0g1_reset.py` imports
from the base classes, so it cannot be imported by them.

`experiments/smpd/f0g1_reset.py` re-exports everything here, so existing
imports from that module keep working.

Every rate is nu = Omega / 2*pi in MHz, matching how the configs store
frequencies. Converting one to a time constant therefore needs the 2*pi:
`t = 1 / (2*pi*nu)`.

The fits these read are written by the calibration scripts:

    device.f0g1.stark_fit     [a, c]  f0g1 resonance = a*gain^2 + c     (28)
    device.f0g1.ef_stark_fit  [a, c]  e-f resonance  = a*gain^2 + c     (32)
    device.f0g1.omega_ef_fit  [m, b]  Omega_ef/2pi   = m*gain + b       (30)
    device.f0g1.g_tilde_fit   [m, b]  g~/2pi         = m*gain + b       (34)
    device.f0g1.kappa                 waste linewidth                  (34)
"""

import numpy as np

from ..analysis import fitting as fitter
from .pulse_shapes import pulse_type_from_config, ramp_length

def f0g1_design_frequency(cfg, qi, readout_line_qi=None):
    """
    f_ge + f_ef - f_readout, the f0-g1 frequency from the bare transitions.

    Used as the scan centre before ``device.f0g1.frequency`` has been measured.
    Both f_ge and f_ef are Lamb- and Stark-shifted in ways this ignores, so
    treat it as a starting guess good to a few MHz, not a prediction.
    """
    if readout_line_qi is None:
        readout_line_qi = qi
    return (
        cfg.device.qubit.f_ge[qi]
        + cfg.device.qubit.f_ef[qi]
        - cfg.device.readout.frequency[readout_line_qi]
    )

def f0g1_frequency_for_gain(cfg, gain, verbose=False):
    """
    The Stark-shifted f0-g1 resonance at a given drive gain.

    Inverts ``device.f0g1.stark_fit`` = [a, c] as ``freq = a*gain**2 + c``.
    Raises if the fit is absent, the way scripts 19-26 raise on a missing
    ``device.fwm`` fit: an uncorrected frequency just detunes the drive, which
    looks like a weak coupling rather than a configuration error.
    """
    fit = cfg.device.get("f0g1", {}).get("stark_fit", None)
    if fit is None or len(fit) != 2 or not np.all(np.isfinite(np.array(fit, dtype=float))):
        raise ValueError(
            "device.f0g1.stark_fit is not set. Run 28_f0g1_stark_shift.py (with "
            "update) first, so the f0-g1 drive can be kept on resonance at each "
            "power."
        )
    a, c = (float(v) for v in fit)
    freq = a * gain**2 + c
    if verbose:
        print(f"f0-g1 frequency at gain {gain:.4f}: {freq:.4f} MHz "
              f"(= {a:.4f}*gain^2 + {c:.4f})")
    return freq


def omega_ef_for_gain(cfg, gain):
    """
    Omega_ef/2pi at a given e-f drive gain, from ``device.f0g1.omega_ef_fit``.

    Raises if the fit is absent -- an e-f pi pulse of unknown duration makes
    step 3's contrast unpredictable.
    """
    fit = cfg.device.get("f0g1", {}).get("omega_ef_fit", None)
    if fit is None or len(fit) != 2 or not np.all(np.isfinite(np.array(fit, dtype=float))):
        raise ValueError(
            "device.f0g1.omega_ef_fit is not set. Run 30_ef_rabi_vs_power.py "
            "(with update) first, so the e-f pi pulse length can be derived."
        )
    m, b = (float(v) for v in fit)
    rate = m * gain + b
    if rate <= 0:
        raise ValueError(
            f"omega_ef_fit gives a non-positive rate {rate:.4f} MHz at gain "
            f"{gain}; the fit or the gain is wrong."
        )
    return rate


def g_tilde_for_gain(cfg, gain):
    """
    g~/2pi at a given f0-g1 drive gain, from ``device.f0g1.g_tilde_fit``.

    Raises if the fit is absent -- without it there is no way to know where on
    the paper's Fig. 3a the drive sits, so no way to pick Omega_ef.
    """
    fit = cfg.device.get("f0g1", {}).get("g_tilde_fit", None)
    if fit is None or len(fit) != 2 or not np.all(np.isfinite(np.array(fit, dtype=float))):
        raise ValueError(
            "device.f0g1.g_tilde_fit is not set. Run 34_f0g1_rabi_vs_power.py "
            "with update (i.e. without --no-update-config) first."
        )
    m, b = (float(v) for v in fit)
    return m * gain + b


def ef_gain_for_omega(cfg, omega, verbose=True):
    """
    The e-f drive gain giving a target Omega_ef/2pi, inverting omega_ef_fit.

    Omega_ef must vanish at zero drive, so the fitted intercept ``b`` is an
    artifact -- it comes from the lowest-gain rows, where the sweep window held
    the fewest periods. Inside the measured range that hardly matters, but the
    reset's optimum Omega_ef can be far *below* the measured range, and there a
    positive ``b`` larger than the target drives the inversion negative.

    So: use the full inversion when it is positive, and fall back to the
    through-origin form ``gain = omega / m`` when it is not, saying which was
    used. A silent negative gain would be clipped to zero by the DAC and the
    e-f drive would simply be absent.
    """
    fit = cfg.device.get("f0g1", {}).get("omega_ef_fit", None)
    if fit is None or len(fit) != 2 or not np.all(np.isfinite(np.array(fit, dtype=float))):
        raise ValueError(
            "device.f0g1.omega_ef_fit is not set. Run 30_ef_rabi_vs_power.py "
            "(with update) first."
        )
    m, b = (float(v) for v in fit)
    if m <= 0:
        raise ValueError(
            f"omega_ef_fit has a non-positive slope {m}; Omega_ef cannot fall "
            "as the e-f drive gets stronger. Retake 30_ef_rabi_vs_power.py."
        )

    full = (omega - b) / m
    if full > 0:
        if verbose:
            print(
                f"  e-f gain {full:.5f} for Omega_ef/2pi = {omega:.4f} MHz "
                f"(inverting {m:.4f}*gain + {b:.4f})"
            )
        return full

    origin = omega / m
    if verbose:
        print(
            f"  e-f gain {origin:.5f} for Omega_ef/2pi = {omega:.4f} MHz, from "
            f"the through-origin form gain = omega/{m:.4f}. The full inversion "
            f"gives {full:+.5f}, which is negative because the target is below "
            f"the fitted intercept {b:+.4f} MHz -- Omega_ef must vanish at zero "
            "drive, so that intercept is a fit artifact and is dropped here. "
            "This is an extrapolation well below the range 30 measured; check "
            "the reset actually improves on doing nothing."
        )
    return origin

def reset_rate(g_tilde, omega_ef, kappa):
    """
    Reset rate Gamma from the paper's Eq. 1.

    The non-Hermitian Hamiltonian on {|e,0>, |f,0>, |g,1>} is

        H = [[0, Omega_ef, 0], [Omega_ef, 0, g~], [0, g~, i*kappa/2]]

    and Gamma = 2 * min|Im(lambda_i)| is the slowest decay rate into the dark
    state |g,0>. All three arguments and the result are in MHz (rate / 2*pi).
    """
    H = np.array(
        [
            [0.0, omega_ef, 0.0],
            [omega_ef, 0.0, g_tilde],
            [0.0, g_tilde, 1j * kappa / 2],
        ],
        dtype=complex,
    )
    return float(2 * np.min(np.abs(np.linalg.eigvals(H).imag)))


def reset_trajectory(t, g_tilde, omega_ef, kappa):
    """
    P(not g) = |c_e|^2 + |c_f|^2 against time, from Eq. 1, starting in |e,0>.

    The non-Hermitian evolution is exact rather than an approximation to a
    Lindblad solve: an emitted photon leaves the qubit in |g,0>, which is dark
    and never returns, so the no-jump trajectory is the whole story. Checked
    against a 9x9 Liouvillian: identical to 4e-5.

    Note the damping sign is -i*kappa/2, opposite `reset_rate`, which only ever
    takes |Im(lambda)| and so does not care. Eigendecomposition rather than one
    expm per time point, because the scale fits in scripts 36/37 call this
    thousands of times.

    Returns an array the shape of `t`, all-NaN if the matrix is defective.
    """
    t = np.asarray(t, dtype=float)
    H = np.array(
        [
            [0.0, omega_ef, 0.0],
            [omega_ef, 0.0, g_tilde],
            [0.0, g_tilde, -1j * kappa / 2],
        ],
        dtype=complex,
    )
    try:
        vals, vecs = np.linalg.eig(H)
        coef = np.linalg.solve(vecs, np.array([1.0, 0.0, 0.0], dtype=complex))
    except np.linalg.LinAlgError:
        return np.full(t.shape, np.nan)
    amps = (np.exp(-2j * np.pi * np.outer(t, vals)) * coef) @ vecs.T
    return np.abs(amps[:, 0]) ** 2 + np.abs(amps[:, 1]) ** 2


def fitted_reset_rate(g_tilde, omega_ef, kappa, t):
    """
    Gamma as an exponential fit of a measured trace reports it.

    `reset_rate` returns the asymptotic eigenvalue, but every reset measurement
    here fits a single exponential to a finite trace, and the two differ by up
    to 3x -- the onset delay drags the fit below the eigenvalue at weak drive,
    the fast initial transient pushes it above at strong drive. Comparing a
    measurement against `reset_rate` compares two different quantities; compare
    against this, passing the run's own time axis.

    Evolves Eq. 1 from |e,0> via `reset_trajectory` and fits P(not g) with the
    same `fitexp` the experiments use.

    Returns MHz (rate / 2*pi), or NaN if the fit fails.
    """
    t = np.asarray(t, dtype=float)
    y = reset_trajectory(t, g_tilde, omega_ef, kappa)
    if not np.all(np.isfinite(y)):
        return float("nan")
    pars, _, _ = fitter.fitexp(t, y)
    tau = float(pars[2])
    if not np.isfinite(tau) or tau <= 0:
        return float("nan")
    # A tau below one sample is unresolvable on this grid, and one past the
    # span is unconstrained by it. fitexp lands on both as degenerate minima --
    # the trace is sigmoid-shaped (flat, then a fall), which an offset
    # exponential fits badly, and at g~ = 0.702, Omega_ef = 0.566 it returned
    # tau = 0.0018 us on a trace that decays over 0.7 us, i.e. 89 MHz. Callers
    # scan this over a gain axis, so one such point would swamp a scale fit.
    if t.size > 1:
        if tau <= float(np.min(np.diff(t))) or tau > float(t[-1] - t[0]):
            return float("nan")
    return float(1 / (2 * np.pi * tau))


def fit_reset_trajectory(t, y, g_tilde, omega_ef, kappa, reps=None):
    """
    Fit a measured P(not g) trace to Eq. 1, freeing g~, Omega_ef and kappa.

    The single-exponential summary throws away the coherent structure: under
    strong drive the trace rings, and those ripples pin all three parameters at
    once. On run 34 this fits ten traces to within shot noise (rms 0.006-0.015
    against 0.005-0.009 expected), recovering g~ to 2.5% of script 34's
    `g_tilde_fit` and Omega_ef at 0.47-0.55 of `omega_ef_fit` -- the same ~2x
    overestimate scripts 36/37 find by two other routes.

    `g_tilde`, `omega_ef` and `kappa` are the config's values, used as the seed.
    `reps` sets the shot-noise reference for the quality check; the fitted
    floor and amplitude absorb readout misassignment and preparation error.

    **Only trust the result when `ripples` is True.** Without an oscillation the
    three parameters trade off freely: on `#446`'s weak-drive points the fit
    returned kappa = 2.3-2.4 and g~ 1.4-1.6x the config from monotonic decays it
    otherwise described well. The gate is a measured rebound after the first
    minimum, which is what "are there ripples" actually means -- a fitted-kappa
    or parameter-error cut would also reject a genuinely good overdamped fit.

    Returns a dict: g_tilde/omega_ef/kappa/floor/amp (+ `_err`), rms, rebound,
    ripples, ok. `ok` is False and the values NaN if the fit did not converge.
    """
    from scipy.optimize import curve_fit

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    out = {k: float("nan") for k in
           ("g_tilde", "omega_ef", "kappa", "floor", "amp", "rms", "rebound",
            "g_tilde_err", "omega_ef_err", "kappa_err")}
    out["ok"] = False
    out["ripples"] = False
    if t.size < 10 or not np.all(np.isfinite(y)):
        return out

    # Rise from the FIRST local minimum to the next local maximum, in
    # shot-noise units. Measured on the data, so it does not need the fit to
    # converge. It must be the first minimum, not the global one: the global
    # minimum normally sits in the noisy tail, where the "rebound" is just
    # scatter -- that read 0.009 on traces that visibly ring and 0.000 on one
    # whose last point was lowest. On `#446` this separates cleanly, 0.0-0.3
    # sigma for the monotonic traces against 3.9-29 sigma for the ringing ones,
    # with nothing in between, so the 3-sigma cut is not finely balanced.
    ys = np.convolve(y, np.ones(3) / 3, mode="valid")
    d = np.diff(ys)
    falling = np.where(d < 0)[0]
    rebound, sigma = 0.0, 0.0
    if falling.size:
        rising = np.where(d[falling[0]:] > 0)[0]
        if rising.size:
            i_min = falling[0] + rising[0]
            after = d[i_min:]
            falls_again = np.where(after < 0)[0]
            i_max = i_min + (falls_again[0] if falls_again.size else after.size)
            lo = float(ys[i_min])
            rebound = float(np.max(ys[i_min:i_max + 1]) - lo)
            if reps:
                sigma = float(np.sqrt(max(lo * (1 - lo), 1e-6) / reps))
    out["rebound"] = rebound
    out["ripples"] = bool(sigma > 0 and rebound > 3 * sigma)

    def model(tt, g, om, kap, floor, amp):
        return floor + amp * reset_trajectory(tt, g, om, kap)

    lo = [0.05, 0.02, 0.05, 0.0, 0.2]
    hi = [8.0, 8.0, 8.0, 0.5, 1.2]
    # Multi-start. `omega_ef` is normally the config's nominal rate, which on
    # run 34 is ~2x what the drive delivers, and seeding there drove `#446`'s
    # third point to kappa pinned at its bound with rms 0.043 where a 0.5x seed
    # reached 0.009. Cheap to just try both and keep the better one.
    best, best_rms = None, np.inf
    for om_seed in (omega_ef, 0.5 * omega_ef):
        for kap_seed in (kappa, 1.0):
            try:
                p, cov = curve_fit(
                    model, t, y,
                    p0=[g_tilde, om_seed, kap_seed,
                        float(np.min(y)), float(np.max(y))],
                    bounds=(lo, hi), maxfev=60000,
                )
            except Exception:
                continue
            rms = float(np.sqrt(np.mean((y - model(t, *p)) ** 2)))
            if rms < best_rms:
                best, best_rms = (p, cov), rms
    if best is None:
        return out
    p, cov = best
    # A parameter sitting on a bound is not a measurement, it is the optimiser
    # running out of room -- the pinned-kappa case above fit visibly badly yet
    # converged, so it would otherwise have been reported as a result.
    if any(abs(v - b) < 1e-6 * max(abs(b), 1.0)
           for v, b in zip(p[:3], lo[:3])) or any(
           abs(v - b) < 1e-6 * max(abs(b), 1.0) for v, b in zip(p[:3], hi[:3])):
        out["ripples"] = False
        return out
    err = np.sqrt(np.clip(np.diag(cov), 0, None))
    for k, v, e in zip(("g_tilde", "omega_ef", "kappa"), p[:3], err[:3]):
        out[k], out[k + "_err"] = float(v), float(e)
    out["floor"], out["amp"] = float(p[3]), float(p[4])
    out["rms"] = float(np.sqrt(np.mean((y - model(t, *p)) ** 2)))
    out["ok"] = True
    return out


def optimal_omega_ef(g_tilde, kappa, n_pts=2001):
    """
    The e-f Rabi rate that maximises Gamma at a given g~.

    This is the red dashed line of the paper's Fig. 3a. Found by scanning
    Omega_ef rather than solving analytically -- the eigenvalues of a 3x3
    non-Hermitian matrix have no useful closed form here, and a 2001-point scan
    over a range we already know brackets the optimum is both cheap and robust.

    Returns (omega_ef_opt, gamma_max), both in MHz.
    """
    omegas = np.linspace(1e-4, max(4 * g_tilde, 2 * kappa), n_pts)
    gammas = np.array([reset_rate(g_tilde, o, kappa) for o in omegas])
    i = int(np.argmax(gammas))
    return float(omegas[i]), float(gammas[i])



def f0g1_reset_params(
    cfg,
    qi,
    gain_f0g1=None,
    gain_ef=None,
    residual_target=0.01,
    length=None,
    gap=None,
    verbose=True,
):
    """
    Derive the unconditional-reset drive settings from ``device.f0g1``.

    Returns the ``cfg.expt`` keys ``QickProgram.reset_f0g1`` needs. Called by
    ``QickExperiment.configure_reset`` when ``reset_mode == "f0g1"``, and merged
    as defaults, so any of them can be overridden from a script's ``params``.

    The operating point follows ``ResetTimeScan``: drive the f0-g1 tone at full
    scale (g~ is below the kappa/3 plateau on this device, so more coupling is
    strictly better), read the optimal Omega_ef off the paper's Fig. 3a at the
    resulting g~, and invert ``omega_ef_fit`` for the e-f gain. Both tones sit
    on their Stark-shifted resonances.

    Args:
        cfg: the full config AttrDict
        qi: qubit index
        gain_f0g1: f0-g1 drive gain (default ``device.qubit.max_gain``)
        gain_ef: e-f drive gain, bypassing the Fig. 3a optimum. Worth setting
            by hand: the optimum can land far below the range
            30_ef_rabi_vs_power could measure, and Gamma falls off steeply
            *below* the optimal Omega_ef while barely moving above it -- so
            erring high is much safer than trusting the extrapolation.
        residual_target: the excited fraction the pulse length aims for; the
            default 0.01 gives ~6.1 us at the run-33 calibration, against a
            19 us tail for the conditional reset and 313 us passive
        length: explicit flat duration in us, bypassing residual_target
        gap: explicit photon-escape window in us, bypassing 3/kappa
        verbose: print the derived operating point

    Raises:
        ValueError: naming the calibration script to run, if a fit is missing.
    """
    if gain_f0g1 is None:
        gain_f0g1 = float(cfg.device.qubit.max_gain)

    kappa = cfg.device.get("f0g1", {}).get("kappa", None)
    if kappa is None or not np.isfinite(kappa):
        kappa = float(cfg.device.readout.kappa[qi])
        kappa_src = f"device.readout.kappa[{qi}]"
    else:
        kappa = float(kappa)
        kappa_src = "device.f0g1.kappa"

    g_tilde = g_tilde_for_gain(cfg, gain_f0g1)
    omega_opt, gamma_opt = optimal_omega_ef(g_tilde, kappa)

    if gain_ef is None:
        gain_ef = ef_gain_for_omega(cfg, omega_opt, verbose=verbose)
        omega_ef, gamma = omega_opt, gamma_opt
    else:
        # An explicit gain means an explicit Omega_ef, which is generally not
        # the optimum -- so recompute Gamma there rather than quoting the
        # optimum's. Through the origin, because the fitted intercept is an
        # artifact at the small gains this reset uses (see ef_gain_for_omega).
        gain_ef = float(gain_ef)
        omega_ef = float(cfg.device.f0g1.omega_ef_fit[0]) * gain_ef
        gamma = reset_rate(g_tilde, omega_ef, kappa)

    # exp(-2*pi*Gamma*t) = residual_target. Gamma is a rate/2pi, hence the 2*pi.
    # None means "use the default" here too, so a script that sets the constant
    # to None to mean "don't care" gets the default rather than a TypeError.
    if residual_target is None:
        residual_target = 0.01
    length_pinned = length is not None
    if not length_pinned:
        if gamma <= 0:
            raise ValueError(
                f"the reset rate derived from device.f0g1 is {gamma}, so no "
                "pulse length can be derived. Check g_tilde_fit and kappa "
                "(34_f0g1_rabi_vs_power.py)."
            )
        length = float(np.log(1 / residual_target) / (2 * np.pi * gamma))
    length = float(length)

    ramp_sigma, ramp_sigma_inc = 0.02, 3
    # Both tones start together and must also stop together, so one shape.
    pulse_type = pulse_type_from_config(cfg.device.get("f0g1"), "pulse_type")
    pulse_type_ef = pulse_type_from_config(cfg.device.get("f0g1"), "pulse_type_ef")
    if pulse_type != pulse_type_ef:
        raise ValueError(
            "device.f0g1.pulse_type and pulse_type_ef must match for the reset, "
            f"or one drive outlasts the other; got {pulse_type!r} and "
            f"{pulse_type_ef!r}."
        )
    gap_pinned = gap is not None
    gap = float(gap) if gap_pinned else 3 / kappa
    freq_f0g1 = f0g1_frequency_for_gain(cfg, gain_f0g1)

    ef_fit = cfg.device.get("f0g1", {}).get("ef_stark_fit", None)
    if ef_fit is not None and np.all(np.isfinite(np.array(ef_fit, dtype=float))):
        a, c = (float(v) for v in ef_fit)
        freq_ef = a * gain_f0g1**2 + c
        ef_src = "ef_stark_fit"
    else:
        freq_ef = float(cfg.device.qubit.f_ef[qi])
        ef_src = "device.qubit.f_ef (unshifted -- run 32)"

    if verbose:
        print(
            f"f0-g1 reset: f0g1 gain {gain_f0g1:.4f} at {freq_f0g1:.4f} MHz "
            f"gives g~/2pi = {g_tilde:.4f} MHz; kappa/2pi = {kappa:.4f} MHz "
            f"from {kappa_src}"
        )
        note = ("" if omega_ef == omega_opt else
                f", vs {omega_opt:.4f} at the optimum which would give "
                f"{gamma_opt:.4f}")
        print(
            f"  Omega_ef/2pi = {omega_ef:.4f} MHz (e-f gain {gain_ef:.5f} at "
            f"{freq_ef:.4f} MHz from {ef_src}) -> Gamma/2pi = {gamma:.4f} MHz, "
            f"{gamma / (kappa / 3) * 100:.0f}% of the kappa/3 ceiling{note}"
        )
        # Say which numbers were chosen and which were handed in, so a run's
        # log records what it actually did rather than what the default is.
        if length_pinned:
            reached = np.exp(-2 * np.pi * gamma * length)
            why = (f"as given (which reaches {reached:.2g} residual at "
                   f"Gamma = {gamma:.4f} MHz)")
        else:
            why = f"for a {residual_target * 100:g}% residual"
        print(
            f"  drive {length:.3f} us {why}; photon gap {gap:.3f} us"
            + (" as given" if gap_pinned else f" (3/kappa)")
        )
        print(
            f"  inter-shot tail about "
            f"{length + ramp_length(pulse_type, ramp_sigma, ramp_sigma_inc) + gap + 1:.1f}"
            " us, against "
            "~19 us for the conditional reset"
        )

    return {
        "f0g1_reset_gain": gain_f0g1,
        "f0g1_reset_freq": freq_f0g1,
        "f0g1_reset_length": length,
        "f0g1_reset_gain_ef": gain_ef,
        "f0g1_reset_freq_ef": freq_ef,
        # Equal ramps on both tones, so they start and end together. The
        # program asserts this: unequal ramps leave one drive on after the
        # other has stopped, which is not the protocol.
        "f0g1_reset_ramp_sigma": ramp_sigma,
        "f0g1_reset_ramp_sigma_inc": ramp_sigma_inc,
        "f0g1_reset_pulse_type": pulse_type,
        "f0g1_reset_pulse_type_ef": pulse_type_ef,
        # The drive deliberately puts a photon in the readout resonator; wait
        # a few 1/kappa for it to leave before the next shot's readout.
        "f0g1_reset_gap": gap,
        # Kept so a run records what it was aiming at.
        "f0g1_reset_gamma": gamma,
        "f0g1_reset_omega_ef": omega_ef,
    }
