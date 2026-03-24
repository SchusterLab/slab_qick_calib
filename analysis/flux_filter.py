"""
IIR filter computation for flux pulse distortion compensation.

Implements the inverse filter approach from arXiv:2503.04610 (Sections 1-3):
given a step response characterized as a sum of exponentials, compute a digital
IIR filter that compensates the distortions when applied as predistortion.

Usage:
    from slab_qick_calib.analysis.flux_filter import compute_inverse_iir, apply_predistortion

    # From fitted step response parameters (alphas_i, tau_i in µs)
    result = compute_inverse_iir(alphas, taus, fs=1000)  # fs in MHz (= 1/µs)
    corrected = apply_predistortion(waveform, result["sos"])

QICK envelope generation:
    from slab_qick_calib.analysis.flux_filter import make_slow_ramp, make_fast_correction

    # Slow ramp (100 µs) at QICK int-gen sample rate
    ramp = make_slow_ramp(alphas, taus, alpha0, duration_us=100, fs_mhz=430)
    # idata ready for prog.add_envelope(ch, "flux_ramp", idata=ramp["idata"])

    # Fast correction (first ~200 ns)
    fast = make_fast_correction(alphas, taus, alpha0, fs_mhz=430)
    # idata for prog.add_envelope(ch, "flux_fast", idata=fast["idata"])
"""

import numpy as np
import scipy.signal


def step_response_to_tf(alphas, taus, alpha0=1.0):
    """
    Build the continuous-time transfer function H(s) from step response parameters.

    Given step response s(t) = (α₀ + Σ αᵢ·exp(-t/τᵢ))·u(t), the impulse response
    h(t) = ds/dt has Laplace transform:

        H(s) = α₀ + Σ αᵢ·s/(s + 1/τᵢ) = N(s) / D(s)

    where D(s) = Π(s + 1/τᵢ) and N(s) = α₀·D(s) + Σ αᵢ·s·Π_{j≠i}(s + 1/τⱼ).

    Args:
        alphas: List of exponential amplitudes.
        taus: List of time constants (µs), same length as alphas.
        alpha0: Steady-state value s(t→∞). Default 1.0 (normalized step
            response). Use 0.0 for a pure high-pass bias tee.

    Returns:
        Tuple (num, den) of polynomial coefficient arrays (highest power first),
        suitable for scipy.signal functions.
    """
    alphas = np.asarray(alphas, dtype=float)
    taus = np.asarray(taus, dtype=float)
    n = len(alphas)

    # D(s) = Π(s + 1/τᵢ)  — product of first-order terms
    den = np.array([1.0])
    for tau in taus:
        den = np.polymul(den, [1.0, 1.0 / tau])

    # N(s) = α₀·D(s) + Σ αᵢ·s·Π_{j≠i}(s + 1/τⱼ)
    num = alpha0 * den.copy()
    for i in range(n):
        # Build Π_{j≠i}(s + 1/τⱼ)
        partial = np.array([1.0])
        for j in range(n):
            if j != i:
                partial = np.polymul(partial, [1.0, 1.0 / taus[j]])
        # αᵢ·s·partial = αᵢ·[partial, 0] (multiply by s = shift + append 0)
        s_partial = np.append(partial, 0.0)  # multiply polynomial by s
        num = np.polyadd(num, alphas[i] * s_partial)

    return num, den


def compute_inverse_iir(alphas, taus, fs, alpha0=1.0):
    """
    Compute the digital inverse IIR filter for flux distortion compensation.

    The inverse filter H_inv(s) = D(s)/N(s) is discretized via the bilinear
    transform and factored into second-order sections (SOS) for numerical
    stability.

    Args:
        alphas: List of exponential amplitudes.
        taus: List of time constants (µs).
        fs: Sample rate in MHz (i.e., samples per µs). For example, fs=1000
            corresponds to 1 GHz (1 sample per ns).
        alpha0: Steady-state value of the step response. Default 1.0.

    Returns:
        Dict with keys:
            'sos': Second-order sections array for scipy.signal.sosfilt
            'b', 'a': Transfer function coefficients (digital)
            'alpha0', 'alphas', 'taus': Input parameters (for reference)
            'fs': Sample rate used
            'num_s', 'den_s': Continuous-time H(s) coefficients
    """
    num_h, den_h = step_response_to_tf(alphas, taus, alpha0=alpha0)

    # Inverse filter: H_inv(s) = D(s) / N(s), i.e., swap num and den
    num_inv = den_h
    den_inv = num_h

    # Discretize via bilinear transform
    b, a = scipy.signal.bilinear(num_inv, den_inv, fs=fs)

    # Convert to second-order sections for numerical stability
    sos = scipy.signal.tf2sos(b, a)

    return {
        "sos": sos,
        "b": b,
        "a": a,
        "alpha0": alpha0,
        "alphas": list(alphas),
        "taus": list(taus),
        "fs": fs,
        "num_s": num_h,
        "den_s": den_h,
    }


def apply_predistortion(waveform, sos):
    """
    Apply the inverse IIR filter to a flux waveform as predistortion.

    Args:
        waveform: 1D array of flux waveform samples.
        sos: Second-order sections array from compute_inverse_iir.

    Returns:
        Predistorted waveform (same shape as input).
    """
    return scipy.signal.sosfilt(sos, waveform)


def validate_filter(alphas, taus, fs, alpha0=1.0, n_samples=None):
    """
    Validate the inverse filter by simulating the full round-trip:
    input → predistortion → physical distortion → output ≈ input.

    Generates a step input, applies predistortion (H_inv), then applies the
    forward distortion (H), and checks that the result is approximately a
    clean step.

    Args:
        alphas: List of exponential amplitudes.
        taus: List of time constants (µs).
        fs: Sample rate in MHz.
        alpha0: Steady-state value of the step response. Default 1.0.
        n_samples: Number of samples to simulate. Default: 10× the longest
            time constant.

    Returns:
        Dict with keys:
            'time': Time array (µs)
            'step_input': Original step input
            'predistorted': After applying H_inv
            'round_trip': After applying H then H_inv (should ≈ step)
            'residual': round_trip - step_input
            'max_error': Maximum absolute residual (excluding first few samples)
    """
    max_tau = max(taus)
    if n_samples is None:
        n_samples = int(10 * max_tau * fs)
    n_samples = np.clip(n_samples, 1000, 500_000)

    dt = 1.0 / fs
    t = np.arange(n_samples) * dt

    # Step input
    step = np.ones(n_samples)

    # Compute filters
    result = compute_inverse_iir(alphas, taus, fs, alpha0=alpha0)
    sos_inv = result["sos"]

    # Forward distortion filter: H(s) = N(s)/D(s)
    num_h, den_h = result["num_s"], result["den_s"]
    b_fwd, a_fwd = scipy.signal.bilinear(num_h, den_h, fs=fs)
    sos_fwd = scipy.signal.tf2sos(b_fwd, a_fwd)

    # Round-trip: step → predistort (H_inv) → distort (H)
    predistorted = scipy.signal.sosfilt(sos_inv, step)
    round_trip = scipy.signal.sosfilt(sos_fwd, predistorted)

    # Also show what the distorted step looks like (without compensation)
    distorted = scipy.signal.sosfilt(sos_fwd, step)

    residual = round_trip - step
    # Skip first few samples (transient from filter initialization)
    skip = min(50, n_samples // 10)
    max_error = float(np.max(np.abs(residual[skip:])))

    return {
        "time": t,
        "step_input": step,
        "distorted": distorted,
        "predistorted": predistorted,
        "round_trip": round_trip,
        "residual": residual,
        "max_error": max_error,
    }


# ---------------------------------------------------------------------------
# QICK envelope generation
# ---------------------------------------------------------------------------

def _predistort_step(alphas, taus, alpha0, n_samples, fs):
    """Compute predistorted unit step at a given sample rate."""
    step = np.ones(n_samples)
    result = compute_inverse_iir(alphas, taus, fs, alpha0)
    return scipy.signal.sosfilt(result["sos"], step)


def _to_idata(waveform, maxv):
    """Scale a float waveform to int16 idata for QICK add_envelope.

    Returns (idata, gain_scale) where gain_scale is the peak of the
    waveform.  Multiply your nominal flux gain by gain_scale to get the
    QICK pulse gain: gain = nominal_gain * gain_scale.
    """
    peak = np.max(np.abs(waveform))
    idata = np.round(waveform / peak * maxv).astype(np.int16)
    return idata, float(peak)


def make_slow_ramp(alphas, taus, alpha0, duration_us, fs_mhz, maxv=None,
                   tau_threshold_us=0.5):
    """Generate a predistorted step envelope using only the slow exponentials.

    Keeps only terms with τ > tau_threshold_us (default 0.5 µs), which
    captures the multi-µs droop and bias-tee decay.  The result is a smooth
    ramp suitable for a long arb pulse on the QICK int generator.

    Args:
        alphas: List of exponential amplitudes from step-response fit.
        taus: List of time constants (µs).
        alpha0: Steady-state coefficient.
        duration_us: Pulse duration in µs.
        fs_mhz: DAC sample rate in MHz (= soccfg['gens'][ch]['f_fabric']
                 for int generators with samps_per_clk=1).
        maxv: Maximum int16 envelope value.  If None, uses 0.9 * 32767
              (the QICK int-gen default with MAXV_SCALE=0.9).
        tau_threshold_us: Only include exponentials with τ above this (µs).

    Returns:
        Dict with:
            'idata': int16 array for prog.add_envelope(ch, name, idata=...)
            'waveform': float predistorted waveform (normalized to 1 at t=0)
            'gain_scale': peak / 1.0 — divide your nominal gain by this
            'time_us': time axis in µs
            'n_samples': number of samples
            'slow_alphas', 'slow_taus': the subset of terms used
    """
    if maxv is None:
        maxv = int(0.9 * 32767)

    alphas = np.asarray(alphas, dtype=float)
    taus = np.asarray(taus, dtype=float)
    mask = taus > tau_threshold_us
    slow_alphas = alphas[mask]
    slow_taus = taus[mask]

    if len(slow_alphas) == 0:
        raise ValueError("No exponentials with τ > %.1f µs" % tau_threshold_us)

    # Recompute alpha0 for the slow-only model: absorb the fast terms
    # into alpha0 so the slow model's t→∞ value remains α₀.
    # The fast terms contribute α_fast_i at t=0 and 0 at t→∞.
    # The slow-only step response should be: α₀ + Σ_slow αᵢ exp(-t/τᵢ)
    # which already has the correct steady-state.

    n_samples = int(np.round(duration_us * fs_mhz))
    waveform = _predistort_step(slow_alphas, slow_taus, alpha0, n_samples, fs_mhz)
    idata, gain_scale = _to_idata(waveform, maxv)

    return {
        "idata": idata,
        "waveform": waveform,
        "gain_scale": gain_scale,
        "time_us": np.arange(n_samples) / fs_mhz,
        "n_samples": n_samples,
        "slow_alphas": slow_alphas.tolist(),
        "slow_taus": slow_taus.tolist(),
    }


def make_fast_correction(alphas, taus, alpha0, fs_mhz, maxv=None,
                         tau_threshold_us=0.5, duration_ns=200):
    """Generate a short envelope capturing only the fast transient correction.

    Keeps only terms with τ ≤ tau_threshold_us.  The output is a short
    arb pulse (default 200 ns) that can be played at the start of the flux
    step, before or overlapping the slow ramp.

    The waveform represents the *difference* from a flat step: it starts
    nonzero and decays to ~0.  To use it, you play it as a separate short
    arb pulse with additive gain, or prepend it to the slow ramp.

    Args:
        alphas: List of exponential amplitudes from step-response fit.
        taus: List of time constants (µs).
        alpha0: Steady-state coefficient.
        fs_mhz: DAC sample rate in MHz.
        maxv: Maximum int16 envelope value (default: 0.9 * 32767).
        tau_threshold_us: Include only exponentials with τ ≤ this (µs).
        duration_ns: Length of the correction pulse in ns.

    Returns:
        Dict with:
            'idata': int16 array for prog.add_envelope
            'waveform': float correction waveform (decays to ~0)
            'gain_scale': peak absolute value of the correction
            'time_us': time axis in µs
            'n_samples': number of samples
            'fast_alphas', 'fast_taus': the subset of terms used
    """
    if maxv is None:
        maxv = int(0.9 * 32767)

    alphas = np.asarray(alphas, dtype=float)
    taus = np.asarray(taus, dtype=float)
    mask = taus <= tau_threshold_us
    fast_alphas = alphas[mask]
    fast_taus = taus[mask]

    if len(fast_alphas) == 0:
        raise ValueError("No exponentials with τ ≤ %.1f µs" % tau_threshold_us)

    n_samples = int(np.round(duration_ns * 1e-3 * fs_mhz))

    # Compute full predistortion and slow-only predistortion, take difference
    full = _predistort_step(alphas, taus, alpha0, n_samples, fs_mhz)
    slow_alphas_arr = alphas[~mask]
    slow_taus_arr = taus[~mask]
    if len(slow_alphas_arr) > 0:
        slow = _predistort_step(slow_alphas_arr, slow_taus_arr, alpha0, n_samples, fs_mhz)
    else:
        slow = np.ones(n_samples) / alpha0

    correction = full - slow
    idata, gain_scale = _to_idata(correction, maxv)

    return {
        "idata": idata,
        "waveform": correction,
        "gain_scale": gain_scale,
        "time_us": np.arange(n_samples) / fs_mhz,
        "n_samples": n_samples,
        "fast_alphas": fast_alphas.tolist(),
        "fast_taus": fast_taus.tolist(),
    }


def make_full_predistorted_step(alphas, taus, alpha0, duration_us, fs_mhz,
                                maxv=None):
    """Generate a complete predistorted step envelope combining all terms.

    This is the simplest approach: one arb envelope with both fast and slow
    corrections baked in.

    Args:
        alphas: List of exponential amplitudes from step-response fit.
        taus: List of time constants (µs).
        alpha0: Steady-state coefficient.
        duration_us: Pulse duration in µs.
        fs_mhz: DAC sample rate in MHz.
        maxv: Maximum int16 envelope value (default: 0.9 * 32767).

    Returns:
        Dict with:
            'idata': int16 array for prog.add_envelope
            'waveform': float predistorted waveform
            'gain_scale': peak / 1.0 — divide your nominal gain by this
            'time_us': time axis in µs
            'n_samples': number of samples
    """
    if maxv is None:
        maxv = int(0.9 * 32767)

    n_samples = int(np.round(duration_us * fs_mhz))
    waveform = _predistort_step(alphas, taus, alpha0, n_samples, fs_mhz)
    idata, gain_scale = _to_idata(waveform, maxv)

    return {
        "idata": idata,
        "waveform": waveform,
        "gain_scale": gain_scale,
        "time_us": np.arange(n_samples) / fs_mhz,
        "n_samples": n_samples,
    }
