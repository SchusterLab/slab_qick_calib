"""
Stability analysis for irregularly sampled tracking data.

Tracking rounds run back to back, so the interval between successive points on
a qubit is set by however long that round's sweeps took. Spans are re-derived
from the config each round, so the interval drifts as T1/T2 are re-measured,
and a relocation sweep or a failed fit shifts it further. The record is
therefore *not* uniformly sampled.

The standard tools assume it is. ``allantools`` takes a scalar rate, and
``analyze_tracking.ipynb`` estimates it as ``1/mean(diff(times))`` -- which
silently mis-assigns every averaging time by roughly the sampling jitter, and
smears any real feature that happens to sit near a gap.

This module provides estimators that take the timestamps seriously:

- ``sampling_report``   -- quantify the jitter and find gaps before trusting anything
- ``allan_dev``         -- Allan deviation binned by elapsed time, not by index
- ``psd_lomb``          -- Lomb-Scargle periodogram, the standard PSD for uneven sampling
- ``resample_uniform``  -- interpolate onto a grid, for feeding the existing
                           allantools path; documented with its caveats

All functions take (t, y) with t in seconds and tolerate NaNs in y.
"""

import numpy as np
from scipy.signal import lombscargle


def _clean(t, y):
    """Drop NaN samples and sort by time."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    o = np.argsort(t)
    return t[o], y[o]


def sampling_report(t, verbose=True):
    """
    Describe how uneven the sampling actually is.

    Worth running before any Allan or PSD analysis: if the jitter is a percent
    or two the uniform-rate approximation is harmless, and if it is tens of
    percent the results from the standard tools are not trustworthy.

    Returns a dict with the median interval, its spread, and any gaps longer
    than 3x median (a stalled round, a relocation, a crash and restart).
    """
    t = np.asarray(t, dtype=float)
    t = np.sort(t[np.isfinite(t)])
    if t.size < 3:
        return {"n": int(t.size)}
    dt = np.diff(t)
    med = float(np.median(dt))
    gaps = np.flatnonzero(dt > 3 * med)
    out = {
        "n": int(t.size),
        "dt_median": med,
        "dt_mean": float(np.mean(dt)),
        "dt_std": float(np.std(dt)),
        "jitter_frac": float(np.std(dt) / med) if med else np.nan,
        "dt_min": float(dt.min()),
        "dt_max": float(dt.max()),
        "duration": float(t[-1] - t[0]),
        "n_gaps": int(gaps.size),
        "gap_times": (t[gaps] - t[0]).tolist(),
    }
    if verbose:
        print(f"  n={out['n']}  span={out['duration']/3600:.2f} hr")
        print(f"  interval: median {med:.1f} s, mean {out['dt_mean']:.1f} s, "
              f"std {out['dt_std']:.1f} s ({100*out['jitter_frac']:.1f}%)")
        print(f"  range {out['dt_min']:.1f}-{out['dt_max']:.1f} s, "
              f"{out['n_gaps']} gap(s) > 3x median")
        if out["jitter_frac"] > 0.05:
            print("  -> jitter > 5%: a uniform-rate Allan analysis will be "
                  "biased; use allan_dev() here instead")
    return out


def allan_dev(t, y, taus=None, overlapping=True, min_per_bin=1):
    """
    Allan deviation of an irregularly sampled series, binned by elapsed time.

    For each averaging time tau the record is cut into consecutive bins of
    *duration* tau using the real timestamps, each bin is averaged, and the
    deviation is formed from differences of adjacent bin pairs:

        sigma_y(tau)^2 = <(ybar_{k+1} - ybar_k)^2> / 2

    counting only pairs whose bins are genuinely adjacent in time, so a gap
    contributes nothing rather than being silently bridged. With uniform
    sampling and tau = m*tau0 this reduces exactly to the textbook overlapping
    Allan deviation -- see the self-consistency check in the module tests.

    Args:
        t: timestamps in seconds
        y: values (NaNs dropped)
        taus: averaging times in seconds; default is an octave ladder from
            2x the median interval to a quarter of the record
        overlapping: slide the bin origin to use all available pairs, which
            lowers the variance of the estimate
        min_per_bin: bins holding fewer samples than this are treated as empty

    Returns:
        (taus, adev, n_pairs) -- taus with no usable pairs are dropped
    """
    t, y = _clean(t, y)
    if t.size < 4:
        return np.array([]), np.array([]), np.array([])

    dt0 = float(np.median(np.diff(t)))
    total = t[-1] - t[0]
    if taus is None:
        n_oct = max(int(np.log2(total / (4 * dt0))), 1)
        taus = 2.0 * dt0 * 2.0 ** np.arange(n_oct + 1)
    taus = np.atleast_1d(np.asarray(taus, dtype=float))

    out_tau, out_dev, out_n = [], [], []
    for tau in taus:
        if tau <= 0 or tau > total / 2:
            continue
        # Offsets shift the bin grid; averaging over them is the overlapping
        # estimator. One offset only would be the non-overlapping one.
        n_off = max(int(tau / dt0), 1) if overlapping else 1
        n_off = min(n_off, 32)                      # diminishing returns past this
        sq, npair = 0.0, 0
        for off in np.linspace(0.0, tau, n_off, endpoint=False):
            edges = np.arange(t[0] + off, t[-1] + tau, tau)
            idx = np.digitize(t, edges) - 1
            means = np.full(len(edges), np.nan)
            counts = np.bincount(idx[idx >= 0], minlength=len(edges))
            sums = np.bincount(idx[idx >= 0], weights=y[idx >= 0],
                               minlength=len(edges))
            good = counts >= min_per_bin
            means[good] = sums[good] / counts[good]
            # adjacent pairs only: both bins populated and consecutive
            a, b = means[:-1], means[1:]
            ok = np.isfinite(a) & np.isfinite(b)
            if np.any(ok):
                sq += float(np.sum((b[ok] - a[ok]) ** 2))
                npair += int(np.sum(ok))
        if npair >= 2:
            out_tau.append(tau)
            out_dev.append(np.sqrt(sq / (2.0 * npair)))
            out_n.append(npair)
    return np.array(out_tau), np.array(out_dev), np.array(out_n)


def psd_lomb(t, y, f_min=None, f_max=None, n_freq=500, detrend=True):
    """
    One-sided PSD via the Lomb-Scargle periodogram.

    Lomb-Scargle is built for unevenly sampled series: it fits sinusoids by
    least squares at each trial frequency rather than assuming a grid, so no
    interpolation is involved and no spurious power is introduced at the
    sampling jitter frequency.

    Normalised to a one-sided PSD in (signal units)^2/Hz: a white series of
    variance s^2 gives a flat PSD at 2*s^2*dt, so integrating over
    [0, 1/(2*dt)] returns the variance. scipy's unnormalized periodogram
    already averages to s^2, so the scale factor is just 2*dt.

    Returns:
        (freqs_Hz, psd)
    """
    t, y = _clean(t, y)
    if t.size < 8:
        return np.array([]), np.array([])
    t = t - t[0]
    if detrend:
        y = y - np.polyval(np.polyfit(t, y, 1), t)   # remove mean and slope
    else:
        y = y - np.mean(y)

    dt0 = float(np.median(np.diff(t)))
    total = t[-1]
    f_min = f_min or 1.0 / total
    f_max = f_max or 0.5 / dt0                       # pseudo-Nyquist
    freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_freq)
    ang = 2 * np.pi * freqs
    power = lombscargle(t, y, ang, precenter=False, normalize=False)
    return freqs, 2.0 * dt0 * power


def resample_uniform(t, y, dt=None, max_gap_factor=3.0):
    """
    Interpolate onto a uniform grid so the existing allantools path can run.

    Provided for compatibility, but prefer ``allan_dev`` and ``psd_lomb``.
    Interpolation invents samples that are correlated with their neighbours,
    which *suppresses* the Allan deviation at the shortest averaging times --
    exactly where the interesting fast noise lives. Points falling inside a gap
    longer than ``max_gap_factor`` x dt are returned as NaN rather than being
    bridged.

    Returns:
        (t_uniform, y_uniform, rate_Hz)
    """
    t, y = _clean(t, y)
    if t.size < 3:
        return np.array([]), np.array([]), np.nan
    dt = dt or float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1] + dt / 2, dt)
    out = np.interp(grid, t, y)

    # blank out anything interpolated across a real gap
    gaps = np.diff(t)
    for i in np.flatnonzero(gaps > max_gap_factor * dt):
        out[(grid > t[i]) & (grid < t[i + 1])] = np.nan
    return grid, out, 1.0 / dt
