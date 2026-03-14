"""
Allan deviation analysis using the allantools package.

Provides allantools-backed versions of Allan deviation calculations with
additional capabilities: modified Allan deviation, Hadamard deviation,
confidence intervals, and noise type classification.

The original custom implementation remains in allan.py for backward compatibility.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import allantools
except ImportError:
    raise ImportError(
        "allantools is required for this module. Install with: pip install allantools"
    )

from .collections import get_parameter_config
from .allan import (
    lorentzian,
    power_law,
    combined_model,
    combined_model_two,
    lorentz_model_two,
    fit_power_law,
    fit_lorentzian,
    fit_combined_model,
    fit_combined_model_two,
    fit_lorentz_model_two,
)

# Mapping from deviation type string to allantools function
_DEV_FUNCS = {
    'adev': allantools.adev,
    'oadev': allantools.oadev,
    'mdev': allantools.mdev,
    'tdev': allantools.tdev,
    'hdev': allantools.hdev,
    'ohdev': allantools.ohdev,
    'totdev': allantools.totdev,
    'theo1': allantools.theo1,
}

# Labels for plot legends
_DEV_LABELS = {
    'adev': 'ADEV',
    'oadev': 'OADEV',
    'mdev': 'MDEV',
    'tdev': 'TDEV',
    'hdev': 'HDEV',
    'ohdev': 'OHDEV',
    'totdev': 'TOTDEV',
    'theo1': 'Théo1',
}

# Noise type alpha values and labels
_NOISE_TYPES = {
    2: 'WPM',    # White phase modulation
    1: 'FPM',    # Flicker phase modulation
    0: 'WFM',    # White frequency modulation
    -1: 'FFM',   # Flicker frequency modulation
    -2: 'RWFM',  # Random walk frequency modulation
}

_NOISE_SLOPES = {
    # ADEV log-log slope -> alpha (noise type)
    -1.0: 2,    # WPM
    -0.5: 0,    # WFM  (also FPM, but ADEV can't distinguish)
    0.0: -1,    # FFM
    0.5: -2,    # RWFM
}


def compute_deviation(data, rate=1.0, dev_type='oadev', taus='octave',
                      normalize=True, data_type='freq'):
    """
    Compute a single Allan-type deviation using allantools.

    Parameters
    ----------
    data : array_like
        Time series of the parameter (e.g., T1 values).
    rate : float
        Sampling rate in Hz (1 / mean_time_between_samples).
    dev_type : str
        Deviation type: 'adev', 'oadev', 'mdev', 'tdev', 'hdev',
        'ohdev', 'totdev', 'theo1'.
    taus : array_like or str
        Tau values in seconds, or an allantools keyword:
        'all', 'octave', 'decade'.
    normalize : bool
        If True, divide data by its median before computing
        (fractional/relative stability).
    data_type : str
        'freq' for frequency/parameter data, 'phase' for phase data.

    Returns
    -------
    dict
        'taus'     : np.ndarray - tau values in seconds
        'devs'     : np.ndarray - deviation values
        'errs'     : np.ndarray - error estimates
        'ns'       : np.ndarray - number of analysis pairs at each tau
        'dev_type' : str
    """
    if dev_type not in _DEV_FUNCS:
        raise ValueError(
            f"Unknown dev_type '{dev_type}'. Choose from: {list(_DEV_FUNCS.keys())}"
        )

    data = np.asarray(data, dtype=float)
    if normalize:
        data = data / np.median(data)

    func = _DEV_FUNCS[dev_type]
    taus_out, devs, errs, ns = func(data, rate=rate, data_type=data_type, taus=taus)

    return {
        'taus': taus_out,
        'devs': devs,
        'errs': errs,
        'ns': ns,
        'dev_type': dev_type,
    }


def compute_multi_deviation(data, rate=1.0, dev_types=('oadev', 'mdev', 'hdev'),
                            taus='octave', normalize=True, data_type='freq'):
    """
    Compute multiple deviation types for the same dataset.

    Parameters
    ----------
    data : array_like
        Time series data.
    rate : float
        Sampling rate in Hz.
    dev_types : tuple of str
        Deviation types to compute.
    taus, normalize, data_type
        Passed to compute_deviation.

    Returns
    -------
    dict
        Mapping dev_type string -> result dict from compute_deviation.
    """
    results = {}
    for dt in dev_types:
        results[dt] = compute_deviation(
            data, rate=rate, dev_type=dt, taus=taus,
            normalize=normalize, data_type=data_type,
        )
    return results


def identify_noise_type(taus, devs):
    """
    Identify the dominant noise type at each tau from the log-log slope
    of the Allan deviation.

    Uses piecewise linear regression on adjacent pairs of (log tau, log dev)
    and maps the slope to the nearest standard noise process.

    Parameters
    ----------
    taus : array_like
        Tau values.
    devs : array_like
        Deviation values.

    Returns
    -------
    list of dict
        For each tau (except the last), a dict with:
        'alpha' : int - noise type exponent
        'label' : str - noise type abbreviation
        'slope' : float - measured log-log slope
    """
    log_tau = np.log10(taus)
    log_dev = np.log10(devs)

    standard_slopes = np.array(list(_NOISE_SLOPES.keys()))
    noise_ids = []

    for i in range(len(log_tau) - 1):
        slope = (log_dev[i + 1] - log_dev[i]) / (log_tau[i + 1] - log_tau[i])
        # Find nearest standard slope
        idx = np.argmin(np.abs(standard_slopes - slope))
        nearest_slope = standard_slopes[idx]
        alpha = _NOISE_SLOPES[nearest_slope]
        noise_ids.append({
            'alpha': alpha,
            'label': _NOISE_TYPES[alpha],
            'slope': slope,
        })

    return noise_ids


def compute_confidence_interval(dev_result, ci=0.6827, n_data=None):
    """
    Compute confidence intervals for a deviation result using Greenhall EDF.

    Parameters
    ----------
    dev_result : dict
        Output from compute_deviation().
    ci : float
        Confidence level (default 0.6827 for 1-sigma).
    n_data : int or None
        Number of data points in the original time series. Required for
        EDF calculation. If None, uses max(ns) as an estimate.

    Returns
    -------
    dict
        'ci_lower'  : np.ndarray - lower confidence bound
        'ci_upper'  : np.ndarray - upper confidence bound
        'noise_ids' : list of dict - identified noise type at each tau
        'edf'       : np.ndarray - equivalent degrees of freedom
    """
    taus = dev_result['taus']
    devs = dev_result['devs']
    ns = dev_result['ns']
    dev_type = dev_result['dev_type']

    if n_data is None:
        n_data = int(np.max(ns)) + 1

    # Determine if overlapping/modified for EDF calculation
    overlapping = dev_type in ('oadev', 'ohdev')
    modified = dev_type in ('mdev',)
    d = 2  # d=2 for Allan-type variances

    # Identify noise type at each tau for EDF
    noise_ids = identify_noise_type(taus, devs)
    # Extend to cover last tau point (use same as second-to-last)
    if len(noise_ids) > 0:
        noise_ids.append(noise_ids[-1])
    else:
        noise_ids = [{'alpha': 0, 'label': 'WFM', 'slope': -0.5}] * len(taus)

    ci_lower = np.zeros(len(taus))
    ci_upper = np.zeros(len(taus))
    edf_arr = np.zeros(len(taus))

    for i in range(len(taus)):
        alpha = noise_ids[i]['alpha']
        # m = averaging factor = tau_i / tau_0
        if i == 0 or taus[0] <= 0:
            m = 1
        else:
            m = int(round(taus[i] / taus[0]))

        try:
            edf = allantools.edf_greenhall(
                alpha=alpha, d=d, m=max(m, 1), N=n_data,
                overlapping=overlapping, modified=modified,
            )
        except Exception:
            # Fallback to simple EDF
            edf = max(ns[i] - 1, 1)

        edf_arr[i] = edf

        try:
            lo, hi = allantools.confidence_interval(dev=devs[i], edf=edf)
            ci_lower[i] = lo
            ci_upper[i] = hi
        except Exception:
            ci_lower[i] = devs[i]
            ci_upper[i] = devs[i]

    return {
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'noise_ids': noise_ids,
        'edf': edf_arr,
    }


def perform_analysis_at(tt, fname='def', param='t1', qubit_list=None,
                        dev_types=('oadev',), taus='octave', n_taus=20,
                        normalize=True, fit_models=('lorentz_two',),
                        show_ci=False, ci_level=0.6827,
                        show_noise_id=False, show_reference_slopes=False,
                        return_results=False):
    """
    Perform Allan deviation analysis using allantools.

    Drop-in companion to allan.perform_analysis() with enhanced capabilities.
    Same input format (tt structure) and similar plot layout.

    Parameters
    ----------
    tt : list of dict
        Tracking data. Each element has 'time' and parameter keys.
    fname : str
        Filename tag.
    param : str
        Parameter key to analyze (e.g., 't1', 'Gamma1', 'f_ge').
    qubit_list : list of int or None
        Qubits to analyze. None means all.
    dev_types : tuple of str
        Deviation types to compute and plot.
    taus : str or array_like or None
        Tau specification: 'octave', 'decade', 'all', or explicit array.
        If None, uses the legacy 20-point log-spaced scheme.
    n_taus : int
        Number of tau points for the legacy log-spaced scheme (when taus=None).
    normalize : bool
        Normalize data by median (for relative/fractional stability).
    fit_models : tuple of str or None
        Noise models to fit: 'power_law', 'lorentzian', 'combined',
        'combined_two', 'lorentz_two'. None disables fitting.
    show_ci : bool
        Show confidence interval shading.
    ci_level : float
        Confidence level for intervals (default 1-sigma).
    show_noise_id : bool
        Annotate the plot with identified noise types.
    show_reference_slopes : bool
        Show reference slope lines for standard noise processes.
    return_results : bool
        If True, return a dict of all computed results.

    Returns
    -------
    dict or None
        If return_results, dict mapping qubit_index -> results.
    """
    if qubit_list is None:
        qubit_list = range(len(tt))

    parameter_config = get_parameter_config()
    param_label = parameter_config.get(param, {}).get('label', param)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = axes.flatten()
    par_mean = []
    par_std = []
    all_results = {}

    _fit_funcs = {
        'power_law': fit_power_law,
        'lorentzian': fit_lorentzian,
        'combined': fit_combined_model,
        'combined_two': fit_combined_model_two,
        'lorentz_two': fit_lorentz_model_two,
    }

    _model_funcs = {
        'power_law': power_law,
        'lorentzian': lorentzian,
        'combined': combined_model,
        'combined_two': combined_model_two,
        'lorentz_two': lorentz_model_two,
    }

    for i in qubit_list:
        times_filtered = np.array(tt[i]['time']) - np.min(np.array(tt[i]['time']))
        par_filtered = np.array(tt[i][param])

        # Plot parameter vs time
        axes[0].plot(times_filtered, par_filtered, "o", markersize=1,
                     linewidth=1, label=f"{i}")

        # Compute sampling rate
        mean_time_diff = np.mean(np.diff(times_filtered))
        rate = 1.0 / mean_time_diff

        # Handle tau specification
        if taus is None:
            # Legacy log-spaced scheme (matching allan.py)
            min_tau = 1
            max_tau = len(par_filtered) // 4
            tau_samples = np.logspace(
                np.log10(min_tau), np.log10(max_tau), n_taus
            ).astype(int)
            tau_samples = np.unique(tau_samples)
            tau_seconds = tau_samples * mean_time_diff
            taus_arg = tau_seconds
        else:
            taus_arg = taus

        # Compute deviations
        qubit_results = {'dev_results': {}, 'fit_results': {}, 'ci': {}}

        for dt in dev_types:
            result = compute_deviation(
                par_filtered, rate=rate, dev_type=dt, taus=taus_arg,
                normalize=normalize, data_type='freq',
            )
            qubit_results['dev_results'][dt] = result

            label_prefix = _DEV_LABELS.get(dt, dt.upper())
            axes[1].loglog(
                result['taus'], result['devs'], "o-", markersize=4,
                linewidth=1, label=f"{i} {label_prefix}",
            )

            # Confidence intervals
            if show_ci:
                ci_result = compute_confidence_interval(
                    result, ci=ci_level, n_data=len(par_filtered),
                )
                qubit_results['ci'][dt] = ci_result
                axes[1].fill_between(
                    result['taus'], ci_result['ci_lower'], ci_result['ci_upper'],
                    alpha=0.2,
                )

            # Noise identification annotations
            if show_noise_id:
                if dt in qubit_results['ci']:
                    noise_ids = qubit_results['ci'][dt]['noise_ids']
                else:
                    noise_ids = identify_noise_type(result['taus'], result['devs'])
                    # Extend for last point
                    if len(noise_ids) > 0 and len(noise_ids) < len(result['taus']):
                        noise_ids.append(noise_ids[-1])
                for k, nid in enumerate(noise_ids):
                    if k < len(result['taus']):
                        axes[1].annotate(
                            nid['label'], (result['taus'][k], result['devs'][k]),
                            fontsize=6, ha='center', va='bottom',
                            textcoords='offset points', xytext=(0, 5),
                        )

        # Fit noise models to the first (primary) deviation type
        primary_dt = dev_types[0]
        primary_result = qubit_results['dev_results'][primary_dt]
        tau_fit = primary_result['taus']
        dev_fit = primary_result['devs']

        if fit_models is not None:
            for model_name in fit_models:
                if model_name not in _fit_funcs:
                    continue
                try:
                    popt, pcov = _fit_funcs[model_name](tau_fit, dev_fit)
                    model_func = _model_funcs[model_name]
                    tau_smooth = np.logspace(
                        np.log10(tau_fit[0]), np.log10(tau_fit[-1]), 100
                    )
                    axes[1].loglog(
                        tau_smooth, model_func(tau_smooth, *popt), "k-",
                        linewidth=1, label=f"{i} {model_name} fit",
                    )
                    qubit_results['fit_results'][model_name] = {
                        'popt': popt, 'pcov': pcov,
                    }
                except Exception as e:
                    print(f"Qubit {i}: {model_name} fit failed: {e}")

        # Reference slopes
        if show_reference_slopes:
            xlim = axes[1].get_xlim()
            x_ref = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 100)
            y_mid = np.median(dev_fit)

            # White FM noise: slope -1/2
            y_wfm = y_mid * (x_ref / np.median(tau_fit)) ** (-0.5)
            axes[1].loglog(x_ref, y_wfm, "--", color='gray', linewidth=0.8,
                           alpha=0.5, label="$\\tau^{-1/2}$ (WFM)")

            # Flicker FM noise: slope 0
            axes[1].axhline(y_mid, linestyle="--", color='green', linewidth=0.8,
                            alpha=0.5, label="$\\tau^{0}$ (FFM)")

            # Random walk FM noise: slope +1/2
            y_rwfm = y_mid * (x_ref / np.median(tau_fit)) ** (0.5)
            axes[1].loglog(x_ref, y_rwfm, "--", color='red', linewidth=0.8,
                           alpha=0.5, label="$\\tau^{+1/2}$ (RWFM)")

        # Statistics
        par_mean.append(np.mean(par_filtered))
        par_std.append(np.std(par_filtered))

        qubit_results['stats'] = {
            'mean': np.mean(par_filtered),
            'std': np.std(par_filtered),
        }
        all_results[i] = qubit_results

    # Format axes
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(param_label)
    axes[0].legend()

    dev_label = "Normalized " if normalize else ""
    dev_label += " / ".join(_DEV_LABELS.get(dt, dt) for dt in dev_types)
    axes[1].set_xlabel("$\\tau$ (s)")
    axes[1].set_ylabel(dev_label)
    axes[1].legend(fontsize=8)
    plt.tight_layout()

    # Second figure: mean/std scatter
    par_mean = np.array(par_mean)
    par_std = np.array(par_std)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    axes2[0].errorbar(range(len(par_mean)), par_mean, yerr=par_std, fmt='o')
    axes2[1].plot(par_mean, par_std / par_mean, 'o')
    plt.show()

    if return_results:
        return all_results


def plot_deviation_comparison(data, rate=1.0,
                              dev_types=('adev', 'oadev', 'mdev', 'hdev'),
                              taus='octave', normalize=True, title=None,
                              show_ci=False, ax=None):
    """
    Plot multiple deviation types on a single log-log axis for comparison.

    Useful for understanding which noise processes dominate and for choosing
    the most appropriate deviation statistic.

    Parameters
    ----------
    data : array_like
        Time series data for a single qubit/parameter.
    rate : float
        Sampling rate in Hz.
    dev_types : tuple of str
        Deviation types to compare.
    taus : str or array_like
        Tau specification.
    normalize : bool
        Normalize by median.
    title : str or None
        Plot title.
    show_ci : bool
        Show confidence interval shading.
    ax : matplotlib Axes or None
        If None, creates a new figure.

    Returns
    -------
    fig : matplotlib Figure (None if ax was provided)
    ax : matplotlib Axes
    results : dict mapping dev_type -> compute_deviation result
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    results = compute_multi_deviation(
        data, rate=rate, dev_types=dev_types, taus=taus,
        normalize=normalize, data_type='freq',
    )

    n_data = len(np.asarray(data))

    for dt, result in results.items():
        label = _DEV_LABELS.get(dt, dt.upper())
        ax.loglog(result['taus'], result['devs'], "o-", markersize=4,
                  linewidth=1.5, label=label)

        if show_ci:
            ci = compute_confidence_interval(result, n_data=n_data)
            ax.fill_between(
                result['taus'], ci['ci_lower'], ci['ci_upper'], alpha=0.15,
            )

    ax.set_xlabel("$\\tau$ (s)")
    ylabel = "Normalized deviation" if normalize else "Deviation"
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    if fig is not None:
        plt.tight_layout()

    return fig, ax, results
