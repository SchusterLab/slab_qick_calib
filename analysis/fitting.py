"""
Fitting module for quantum experiments.

This module provides functions for fitting various types of data from quantum experiments,
including exponential decays, sinusoids, Lorentzians, and more specialized functions like
Hanger resonator fits and randomized benchmarking.
"""

import numpy as np
import scipy as sp
import traceback
from typing import Tuple, List, Optional, Callable, Dict, Any, Union

# ====================================================== #
# Utility Functions
# ====================================================== #


def get_r2(
    xdata: np.ndarray, ydata: np.ndarray, fitfunc: Callable, fit_params: List[float]
) -> float:
    """
    Calculate the R-squared value for a fit.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitfunc: The fitting function
        fit_params: Parameters for the fitting function

    Returns:
        R-squared value of the fit
    """
    # Residual sum of squares
    ss_res = np.sum((fitfunc(xdata, *fit_params) - ydata) ** 2)
    # Total sum of squares
    ss_tot = np.sum((np.mean(ydata) - ydata) ** 2)
    # R^2 value
    r2 = 1 - ss_res / ss_tot
    return r2


def fix_phase(p: List[float]) -> float:
    """
    Normalize phase and calculate pi gain.

    Args:
        p: Parameters list containing phase information

    Returns:
        Pi gain value
    """
    if p[2] > 180:
        p[2] = p[2] - 360
    elif p[2] < -180:
        p[2] = p[2] + 360

    if p[2] < 0:
        pi_gain = (1 / 2 - p[2] / 180) / 2 / p[1]
    else:
        pi_gain = (3 / 2 - p[2] / 180) / 2 / p[1]
    return pi_gain


def fourier_init(
    xdata: np.ndarray, ydata: np.ndarray, debug: bool = False
) -> Tuple[float, float]:
    """
    Initialize frequency and phase using Fourier transform.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        debug: If True, plots the Fourier transform for debugging

    Returns:
        Tuple of (max_frequency, max_phase)
    """
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)

    half_N = len(ydata) // 2
    mag = np.abs(fourier[1:half_N])
    phase = fft_phases[1:half_N]
    freqs = fft_freqs[1:half_N]

    max_ind = np.argmax(mag)
    max_freq = freqs[max_ind]
    max_phase = phase[max_ind]

    if debug:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(4, 6))

        ax[0].plot(freqs, mag, ".")
        ax[1].set_xlabel("Frequency (MHz)")
        ax[0].set_ylabel("Amplitude")

        ax[1].plot(freqs, phase * 180 / np.pi, ".")
        ax[1].plot(max_freq, max_phase * 180 / np.pi, "ro")
        ax[1].set_ylabel("Phase (deg)")

        print(f"Max phase is {max_phase}")
        print(f"Max freq is {max_freq}")
        plt.show()

    return max_freq, max_phase


def validate_bounds(
    fitparams: List[float], bounds: Tuple[List[float], List[float]]
) -> List[float]:
    """
    Validate that parameters are within bounds and adjust if necessary.

    Args:
        fitparams: List of fit parameters
        bounds: Tuple of (lower_bounds, upper_bounds)

    Returns:
        Validated fit parameters
    """
    for i, param in enumerate(fitparams):
        if not (bounds[0][i] < param < bounds[1][i]):
            fitparams[i] = np.clip(param, bounds[0][i], bounds[1][i])
            print(
                f"Attempted to init fitparam {i} to {param}, which is out of bounds {bounds[0][i]} to {bounds[1][i]}. Instead init to {fitparams[i]}"
            )
    return fitparams


def generic_fit(
    fitfunc: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: List[float],
    bounds: Optional[Tuple[List[float], List[float]]] = None,
    error_message: str = "Warning: fit failed!",
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Generic fitting function that handles common fitting patterns.

    Args:
        fitfunc: The function to fit
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Initial parameters for fitting
        bounds: Optional bounds for parameters
        error_message: Message to display if fitting fails

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
    """
    if bounds:
        fitparams = validate_bounds(fitparams, bounds)

    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)

    try:
        if bounds:
            pOpt, pCov = sp.optimize.curve_fit(
                fitfunc, xdata, ydata, p0=fitparams, bounds=bounds
            )
        else:
            pOpt, pCov = sp.optimize.curve_fit(fitfunc, xdata, ydata, p0=fitparams)
    except RuntimeError:
        #print(error_message)
        pOpt = [np.nan] * len(pOpt)

    return pOpt, pCov, fitparams


# ====================================================== #
# Data Selection Functions
# ====================================================== #


def get_best_fit(
    data: Dict[str, Any],
    fitfunc: Optional[Callable] = None,
    prefixes: List[str] = ["fit"],
    check_measures: Tuple[str,] = ("amps", "avgi", "avgq"),
    get_best_data_params: Tuple[str,] = (),
    override: Optional[str] = None,
) -> List[Any]:
    """
    Compare fits between different measurements and select the best one.

    Args:
        data: Dictionary containing fit data
        fitfunc: Optional function to use for R² calculation
        prefixes: List of prefixes to check in data keys
        check_measures: Tuple of measurement types to check
        get_best_data_params: Additional parameters to retrieve for best measurement
        override: Optional measurement to use regardless of fit quality

    Returns:
        List containing [best_fit, best_fit_err, *additional_params, best_measurement_type]
    """
    # Collect fit parameters and error matrices
    fits = []
    fit_errors = []

    for measure in check_measures:
        for prefix in prefixes:
            fits.append(data[f"{prefix}_{measure}"])
            fit_errors.append(data[f"{prefix}_err_{measure}"])

    # Fix error matrices: replace zeros with infinity (indicating bad fit)
    for error_matrix in fit_errors:
        diagonal = np.diag(error_matrix)
        zero_indices = np.where(diagonal == 0)[0]
        for idx in zero_indices:
            error_matrix[idx, idx] = np.inf

    # Use override if specified
    if override is not None and override in check_measures:
        best_index = np.argwhere(np.array(check_measures) == override)[0][0]
    else:
        # Calculate fit quality metrics
        if fitfunc is not None:
            # Method 1: Use both R² and normalized parameter errors
            best_index = _find_best_fit_with_r2(
                data, fits, fit_errors, check_measures, fitfunc
            )
        else:
            # Method 2: Use only normalized parameter errors
            best_index = _find_best_fit_simple(fits, fit_errors)

    # Get the best measurement type (accounting for prefixes)
    best_measure = check_measures[best_index % len(check_measures)]

    # Collect results
    result = [fits[best_index], fit_errors[best_index]]

    # Add any additional requested parameters
    for param in get_best_data_params:
        result.append(data[f"{param}_{best_measure}"])

    # Add the measurement type
    result.append(best_measure)

    return result


def _find_best_fit_with_r2(
    data: Dict[str, Any],
    fits: List[Any],
    fit_errors: List[np.ndarray],
    check_measures: Tuple[str,],
    fitfunc: Callable,
) -> int:
    """
    Find the best fit using R² values and normalized parameter errors.

    Args:
        data: Dictionary containing fit data
        fits: List of fit parameters
        fit_errors: List of error matrices
        check_measures: Tuple of measurement types
        fitfunc: Function to use for R² calculation

    Returns:
        Index of the best fit
    """
    # Get x and y data for R² calculation
    xdata = data["xpts"]
    ydata = [data[measure] for measure in check_measures]

    # Calculate R² values
    r2_values = []
    for i, (fit, y) in enumerate(zip(fits[: len(check_measures)], ydata)):
        residual_sum_sq = np.sum((fitfunc(xdata, *fit) - y) ** 2)
        total_sum_sq = np.sum((np.mean(y) - y) ** 2)
        r2 = 1 - residual_sum_sq / total_sum_sq

        # Set R² to infinity if any parameter has infinite error
        if np.any(np.diag(fit_errors[i]) == np.inf):
            r2 = np.inf
        if np.any(np.isnan(fits)):
            r2 = np.inf

        r2_values.append(r2)

    # Calculate normalized parameter errors
    norm_errors = _calculate_normalized_errors(fits, fit_errors)

    # Return index of fit with lowest normalized error
    return np.argmin(norm_errors)


def _find_best_fit_simple(fits: List[Any], fit_errors: List[np.ndarray]) -> int:
    """
    Find the best fit using only normalized parameter errors.

    Args:
        fits: List of fit parameters
        fit_errors: List of error matrices

    Returns:
        Index of the best fit
    """
    # Calculate normalized parameter errors
    norm_errors = _calculate_normalized_errors(fits, fit_errors)

    # Return index of fit with lowest normalized error
    return np.argmin(norm_errors)


def _calculate_normalized_errors(
    fits: List[Any], fit_errors: List[np.ndarray]
) -> np.ndarray:
    """
    Calculate normalized parameter errors for each fit.

    Args:
        fits: List of fit parameters
        fit_errors: List of error matrices

    Returns:
        Array of normalized errors
    """
    norm_errors = []

    for fit, error_matrix in zip(fits, fit_errors):
        # Calculate average of sqrt(|error|/|parameter|)
        param_errors = np.sqrt(np.abs(np.diag(error_matrix)))
        param_values = np.abs(fit)
        norm_error = np.mean(param_errors / param_values)

        # Handle NaN errors
        if np.isnan(norm_error):
            norm_error = np.inf

        norm_errors.append(norm_error)

    return np.array(norm_errors)


# ====================================================== #
# Exponential Fit Functions
# ====================================================== #


def expfunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Exponential decay function.

    Args:
        x: X-axis data points
        p: Parameters [y0, yscale, decay]

    Returns:
        y = y0 + yscale*exp(-x/decay)
    """
    y0, yscale, decay = p
    return y0 + yscale * np.exp(-x / decay)


def expfunc2(x: np.ndarray, *p) -> np.ndarray:
    """
    Exponential decay function with x offset.

    Args:
        x: X-axis data points
        p: Parameters [y0, yscale, x0, decay]

    Returns:
        y = y0 + yscale*exp(-(x-x0)/decay)
    """
    y0, yscale, x0, decay = p
    return y0 + yscale * np.exp(-(x - x0) / decay)


def fitexp(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to an exponential decay.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters [y0, yscale, decay]

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
    """
    if fitparams is None:
        fitparams = [None] * 3

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = ydata[-1]  # y0
    if fitparams[1] is None:
        fitparams[1] = ydata[0] - ydata[-1]  # yscale
    if fitparams[2] is None:
        fitparams[2] = (xdata[-1] - xdata[0]) / 4  # decay

    return generic_fit(
        expfunc,
        xdata,
        ydata,
        fitparams,
        error_message="Warning: Fit exponential failed!",
    )


# ====================================================== #
# Sum-of-Exponentials (Step Response) Fit Functions
# ====================================================== #


def sum_exp_func(t: np.ndarray, *p) -> np.ndarray:
    """
    Sum-of-exponentials model for step response fitting.

    f(t) = f_inf + a1*exp(-t/tau1) + a2*exp(-t/tau2) + ...

    Args:
        t: Time data points
        p: Parameters [f_inf, a1, tau1, a2, tau2, ..., aN, tauN]

    Returns:
        Model values at each time point
    """
    f_inf = p[0]
    result = np.full_like(t, f_inf, dtype=float)
    for i in range(1, len(p), 2):
        a_i = p[i]
        tau_i = p[i + 1]
        result = result + a_i * np.exp(-t / tau_i)
    return result


def fit_step_response(
    tdata: np.ndarray,
    fdata: np.ndarray,
    n_exp: int = 3,
    fitparams: Optional[List[float]] = None,
    f_inf: Optional[float] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit step response data (frequency vs delay time) to a sum of exponentials.

    Model: f(t) = f_inf + sum_i a_i * exp(-t / tau_i)

    This corresponds to Figure 4 of arXiv:2503.04610 — characterizing the
    exponential droops in a flux control line step response.

    Args:
        tdata: Delay times (µs)
        fdata: Measured qubit frequencies (MHz)
        n_exp: Number of exponential terms (default: 3)
        fitparams: Optional initial parameters [f_inf, a1, tau1, a2, tau2, ...].
            Use None entries for auto-guess.
        f_inf: If provided, fix f_inf to this value (MHz) instead of fitting it.
            Useful when the sweet-spot frequency is known but the data window
            doesn't extend long enough to determine f_inf from data alone.
        t_range: Optional (t_min, t_max) to restrict the fit range (µs).
            Data outside this range is excluded from the fit.

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters).
        When f_inf is fixed, the returned pOpt still has f_inf as the first
        element (for compatibility with sum_exp_func).
    """
    tdata = np.asarray(tdata, dtype=float)
    fdata = np.asarray(fdata, dtype=float)

    # Apply time range filter
    if t_range is not None:
        t_lo = t_range[0] if t_range[0] is not None else -np.inf
        t_hi = t_range[1] if t_range[1] is not None else np.inf
        mask = (tdata >= t_lo) & (tdata <= t_hi)
        tdata = tdata[mask]
        fdata = fdata[mask]

    t_min_data = max(tdata[tdata > 0].min(), 1e-6) if np.any(tdata > 0) else 1e-6
    t_max_data = float(tdata.max())

    if f_inf is not None:
        # Fixed f_inf: fit only the exponential terms
        f_inf_val = float(f_inf)
        fdata_shifted = fdata - f_inf_val  # now fitting: g(t) = sum a_i exp(-t/tau_i)

        if fitparams is None:
            fitparams_inner = [None] * (2 * n_exp)
        else:
            # Strip f_inf from fitparams if it was included
            fitparams_inner = list(fitparams[1:]) if len(fitparams) == 1 + 2 * n_exp else list(fitparams)

        tau_guesses = np.logspace(np.log10(t_min_data), np.log10(t_max_data), n_exp)
        total_dev = float(fdata_shifted[0])

        for i in range(n_exp):
            idx_a = 2 * i
            idx_tau = 2 * i + 1
            if idx_a >= len(fitparams_inner) or fitparams_inner[idx_a] is None:
                if idx_a >= len(fitparams_inner):
                    fitparams_inner.extend([None, None])
                fitparams_inner[idx_a] = total_dev / n_exp
            if idx_tau >= len(fitparams_inner) or fitparams_inner[idx_tau] is None:
                if idx_tau >= len(fitparams_inner):
                    fitparams_inner.append(None)
                fitparams_inner[idx_tau] = float(tau_guesses[i])

        def _model(t, *p):
            return sum_exp_func(t, f_inf_val, *p)

        lower = [-np.inf, 1e-6] * n_exp
        upper = [np.inf, t_max_data * 100] * n_exp

        pOpt_inner, pCov_inner, pInit_inner = generic_fit(
            _model, tdata, fdata, fitparams_inner,
            bounds=(lower, upper),
            error_message="Warning: step response fit failed!",
        )

        # Reconstruct full parameter vector with f_inf
        pOpt = [f_inf_val] + list(pOpt_inner)
        # Expand covariance: add zero row/col for fixed f_inf
        n_inner = len(pOpt_inner)
        pCov = np.zeros((n_inner + 1, n_inner + 1))
        pCov[1:, 1:] = pCov_inner
        pInit = [f_inf_val] + list(pInit_inner)
        return pOpt, pCov, pInit

    # Free f_inf: fit all parameters
    if fitparams is None:
        fitparams = [None] * (1 + 2 * n_exp)

    if fitparams[0] is None:
        fitparams[0] = float(fdata[-1])

    total_deviation = float(fdata[0] - fdata[-1])
    tau_guesses = np.logspace(np.log10(t_min_data), np.log10(t_max_data), n_exp)

    for i in range(n_exp):
        idx_a = 1 + 2 * i
        idx_tau = 2 + 2 * i
        if fitparams[idx_a] is None:
            fitparams[idx_a] = total_deviation / n_exp
        if fitparams[idx_tau] is None:
            fitparams[idx_tau] = float(tau_guesses[i])

    lower = [-np.inf] + [-np.inf, 1e-6] * n_exp
    upper = [np.inf] + [np.inf, t_max_data * 100] * n_exp

    return generic_fit(
        sum_exp_func,
        tdata,
        fdata,
        fitparams,
        bounds=(lower, upper),
        error_message="Warning: step response fit failed!",
    )


def extract_step_response_params(pOpt: List[float]) -> dict:
    """
    Unpack fitted sum-of-exponentials parameters into a structured dict.

    Args:
        pOpt: Fitted parameters [f_inf, a1, tau1, a2, tau2, ...]

    Returns:
        Dict with keys 'f_inf', 'alphas' (list), 'taus' (list), sorted by tau
    """
    f_inf = pOpt[0]
    alphas = []
    taus = []
    for i in range(1, len(pOpt), 2):
        alphas.append(pOpt[i])
        taus.append(pOpt[i + 1])

    # Sort by time constant (ascending)
    order = np.argsort(taus)
    alphas = [alphas[i] for i in order]
    taus = [taus[i] for i in order]

    return {"f_inf": f_inf, "alphas": alphas, "taus": taus}


def fit_step_response_flux(
    tdata: np.ndarray,
    fdata: np.ndarray,
    flux_converter,
    flux_gain_target: float,
    n_exp: int = 3,
    alpha0: Optional[float] = None,
    fitparams: Optional[List[float]] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Fit step response in normalized flux space (arXiv:2503.04610, Section III).

    Converts measured frequencies to normalized flux using the transmon model,
    then fits s(t) = α₀ + Σ αᵢ·exp(-t/τᵢ) where s=1 at the target flux and
    s=0 at the sweet spot. This linearizes the nonlinear frequency–flux mapping.

    Args:
        tdata: Delay times (µs)
        fdata: Measured qubit frequencies (MHz)
        flux_converter: TransmonFluxConverter or QuadraticFluxConverter instance
        flux_gain_target: Target flux gain value used in the experiment
        n_exp: Number of exponential terms
        alpha0: Steady-state value α₀ = s(t→∞). If None (default), fit as a
            free parameter. If a float, fix it. Use alpha0=0 for a pure
            high-pass bias tee where flux fully decays.
        fitparams: Optional initial parameters. Layout depends on alpha0:
            - alpha0=None (free): [alpha0, a1, tau1, a2, tau2, ...]
            - alpha0=fixed: [a1, tau1, a2, tau2, ...]
        t_range: Optional (t_min, t_max) in µs to restrict the fit range

    Returns:
        Tuple of (params_dict, s_data, s_fit) where:
            params_dict: {'alpha0': float, 'alphas': [...], 'taus': [...],
                          'gain_sweet': ..., 'gain_target': ..., 'delta_gain': ...}
            s_data: Normalized step response data
            s_fit: Fitted normalized step response
    """
    tdata = np.asarray(tdata, dtype=float)
    fdata = np.asarray(fdata, dtype=float)

    # Convert to normalized flux
    gain_t = flux_converter.freq_to_gain(fdata)
    gain_sweet = flux_converter.sweet_spot
    delta_gain = flux_gain_target - gain_sweet
    s_data = (gain_t - gain_sweet) / delta_gain

    # Apply time range filter for fitting
    if t_range is not None:
        t_lo = t_range[0] if t_range[0] is not None else -np.inf
        t_hi = t_range[1] if t_range[1] is not None else np.inf
        mask = (tdata >= t_lo) & (tdata <= t_hi)
    else:
        mask = np.ones(len(tdata), dtype=bool)
    t_fit = tdata[mask]
    s_fit_data = s_data[mask]

    t_min_data = max(t_fit[t_fit > 0].min(), 1e-6) if np.any(t_fit > 0) else 1e-6
    t_max_data = float(t_fit.max())

    if alpha0 is not None:
        # Fixed α₀: fit only the exponential terms
        alpha0_val = float(alpha0)
        s_shifted = s_fit_data - alpha0_val

        if fitparams is None:
            tau_guesses = np.logspace(np.log10(t_min_data), np.log10(t_max_data), n_exp)
            total_dev = float(s_shifted[0]) if len(s_shifted) > 0 else 1.0
            fitparams_inner = []
            for i in range(n_exp):
                fitparams_inner.extend([total_dev / n_exp, float(tau_guesses[i])])
        else:
            fitparams_inner = list(fitparams)

        def s_model_fixed(t, *p):
            result = np.full_like(t, alpha0_val, dtype=float)
            for i in range(0, len(p), 2):
                result += p[i] * np.exp(-t / p[i + 1])
            return result

        lower = [-2, 1e-4] * n_exp
        upper = [2, t_max_data * 1000] * n_exp

        pOpt_inner, pCov, pInit = generic_fit(
            s_model_fixed, t_fit, s_fit_data, fitparams_inner,
            bounds=(lower, upper),
            error_message="Warning: step response flux fit failed!",
        )

        s_fit = s_model_fixed(tdata, *pOpt_inner)

        alphas = [pOpt_inner[i] for i in range(0, len(pOpt_inner), 2)]
        taus = [pOpt_inner[i] for i in range(1, len(pOpt_inner), 2)]

    else:
        # Free α₀: fit alpha0 + exponentials
        # Parameter layout: [alpha0, a1, tau1, a2, tau2, ...]
        if fitparams is None:
            tau_guesses = np.logspace(np.log10(t_min_data), np.log10(t_max_data), n_exp)
            alpha0_guess = float(s_fit_data[-1])
            total_dev = float(s_fit_data[0] - s_fit_data[-1])
            fitparams = [alpha0_guess]
            for i in range(n_exp):
                fitparams.extend([total_dev / n_exp, float(tau_guesses[i])])

        def s_model_free(t, *p):
            result = np.full_like(t, p[0], dtype=float)
            for i in range(1, len(p), 2):
                result += p[i] * np.exp(-t / p[i + 1])
            return result

        lower = [-2] + [-2, 1e-4] * n_exp
        upper = [2] + [2, t_max_data * 1000] * n_exp

        pOpt, pCov, pInit = generic_fit(
            s_model_free, t_fit, s_fit_data, fitparams,
            bounds=(lower, upper),
            error_message="Warning: step response flux fit failed!",
        )

        s_fit = s_model_free(tdata, *pOpt)

        alpha0_val = pOpt[0]
        alphas = [pOpt[i] for i in range(1, len(pOpt), 2)]
        taus = [pOpt[i] for i in range(2, len(pOpt), 2)]

    # Sort by time constant (ascending)
    order = np.argsort(taus)
    alphas = [alphas[i] for i in order]
    taus = [taus[i] for i in order]

    params = {
        "alpha0": float(alpha0_val),
        "alphas": alphas,
        "taus": taus,
        "gain_sweet": float(gain_sweet),
        "gain_target": float(flux_gain_target),
        "delta_gain": float(delta_gain),
    }

    return params, s_data, s_fit


# ====================================================== #
# Lorentzian Fit Functions
# ====================================================== #


def lorfunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Lorentzian function.

    Args:
        x: X-axis data points
        p: Parameters [y0, yscale, x0, xscale]

    Returns:
        y = y0 + yscale/(1+(x-x0)²/xscale²)
    """
    y0, yscale, x0, xscale = p
    return y0 + yscale / (1 + (x - x0) ** 2 / xscale**2)


def fitlor(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a Lorentzian function.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters [y0, yscale, x0, xscale]

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
    """
    if fitparams is None:
        fitparams = [None] * 4

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = (ydata[0] + ydata[-1]) / 2  # y0
    if fitparams[1] is None:
        fitparams[1] = max(ydata) - min(ydata)  # yscale
    if fitparams[2] is None:
        fitparams[2] = xdata[np.argmax(abs(ydata - fitparams[0]))]  # x0
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 10  # xscale

    return generic_fit(
        lorfunc,
        xdata,
        ydata,
        fitparams,
        error_message="Warning: Fit Lorentzian failed!",
    )


# ====================================================== #
# Sinusoidal Fit Functions
# ====================================================== #


def sinfunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Sinusoidal function.

    Args:
        x: X-axis data points
        p: Parameters [yscale, freq, phase_deg, y0]

    Returns:
        y = yscale*sin(2π*freq*x + phase_deg*π/180) + y0
    """
    yscale, freq, phase_deg, y0 = p
    return yscale * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) + y0


def fitsin(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: Optional[List[float]] = None,
    debug: bool = False,
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a sinusoidal function.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters [yscale, freq, phase_deg, y0]
        debug: If True, shows debug information

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
    """
    if fitparams is None:
        fitparams = [None] * 4

    # Initialize using Fourier transform
    max_freq, max_phase = fourier_init(xdata, ydata, debug)

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = 1 / 2 * (max(ydata) - min(ydata))  # yscale
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi  # phase_deg
    if fitparams[3] is None:
        fitparams[3] = np.mean(ydata)  # y0

    bounds = (
        [0.5 * fitparams[0], 1e-3, -360, np.min(ydata)],
        [2 * fitparams[0], 1000, 360, np.max(ydata)],
    )

    return generic_fit(
        sinfunc,
        xdata,
        ydata,
        fitparams,
        bounds=bounds,
        error_message="Warning: Fit sinusoidal failed!",
    )


def decaysin(x: np.ndarray, *p) -> np.ndarray:
    """
    Decaying sinusoidal function.

    Args:
        x: X-axis data points
        p: Parameters [yscale, freq, phase_deg, decay, y0]

    Returns:
        y = yscale*sin(2π*freq*x + phase_deg*π/180)*exp(-x/decay) + y0
    """
    yscale, freq, phase_deg, decay, y0 = p
    return (
        yscale
        * np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180)
        * np.exp(-x / decay)
        + y0
    )


def fitdecaysin(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: Optional[List[float]] = None,
    debug: bool = False,
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a decaying sinusoidal function.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters [yscale, freq, phase_deg, decay, y0]
        debug: If True, shows debug information

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
    """
    if fitparams is None:
        fitparams = [None] * 5

    # Initialize using Fourier transform
    max_freq, max_phase = fourier_init(xdata, ydata, debug)

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)  # yscale
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi + 90  # phase_deg
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 4  # decay
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)  # y0

    bounds = (
        [0.6 * fitparams[0], 1e-3, -360, 0.05, np.min(ydata)],
        [1.5 * fitparams[0], 1e3, 360, np.inf, np.max(ydata)],
    )

    fitparams = validate_bounds(fitparams, bounds)
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            decaysin, xdata, ydata, p0=fitparams, bounds=bounds
        )
    except RuntimeError:
        try:
            # Try with inverted phase
            fitparams[2] = -fitparams[2]
            pOpt, pCov = sp.optimize.curve_fit(
                decaysin, xdata, ydata, p0=fitparams, bounds=bounds
            )
        except:
            print("Warning: Fit decaying sine failed!")
            pOpt = [np.nan] * len(pOpt)

    return pOpt, pCov, fitparams


def decayslopesin(x: np.ndarray, *p) -> np.ndarray:
    """
    Decaying sinusoidal function with slope.

    Args:
        x: X-axis data points
        p: Parameters [yscale, freq, phase_deg, decay, y0, slope]

    Returns:
        y = yscale*(sin(2π*freq*x + phase_deg*π/180) + slope)*exp(-x/decay) + y0
    """
    yscale, freq, phase_deg, decay, y0, slope = p
    return (
        yscale
        * (np.sin(2 * np.pi * freq * x + phase_deg * np.pi / 180) + slope)
        * np.exp(-x / decay)
        + y0
    )


def fitdecayslopesin(
    xdata: np.ndarray,
    ydata: np.ndarray,
    fitparams: Optional[List[float]] = None,
    debug: bool = False,
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a decaying sinusoidal function with slope.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters [yscale, freq, phase_deg, decay, y0, slope]
        debug: If True, shows debug information

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
    """
    if fitparams is None:
        fitparams = [None] * 6

    # Initialize using Fourier transform
    max_freq, max_phase = fourier_init(xdata, ydata, debug)

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)  # yscale
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi + 90  # phase_deg
    if fitparams[3] is None:
        fitparams[3] = (max(xdata) - min(xdata)) / 4  # decay
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)  # y0
    if fitparams[5] is None:
        fitparams[5] = 0  # slope

    bounds = (
        [0.6 * fitparams[0], 1e-3, -360, 0.03, np.min(ydata), -np.inf],
        [1.5 * fitparams[0], 1e3, 360, np.inf, np.max(ydata), np.inf],
    )

    fitparams = validate_bounds(fitparams, bounds)
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)

    try:
        pOpt, pCov = sp.optimize.curve_fit(decayslopesin, xdata, ydata, p0=fitparams)
    except RuntimeError:
        try:
            # Try with phase shifted by -90 degrees
            fitparams[2] = fitparams[2] - 90
            pOpt, pCov = sp.optimize.curve_fit(
                decayslopesin, xdata, ydata, p0=fitparams
            )
        except:
            try:
                # Try with phase shifted by +180 degrees
                fitparams[2] = fitparams[2] + 180
                pOpt, pCov = sp.optimize.curve_fit(
                    decayslopesin, xdata, ydata, p0=fitparams
                )
            except:
                print("Warning: Fit decaying slope sine failed!")
                pOpt = [np.nan] * len(pOpt)

    return pOpt, pCov, fitparams


def twofreq_decaysin(x: np.ndarray, *p) -> np.ndarray:
    """
    Two-frequency decaying sinusoidal function.

    Args:
        x: X-axis data points
        p: Parameters [yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01]

    Returns:
        y = y00 + decaysin(x, *p0) * sinfunc(x, *p1)
    """
    yscale0, freq0, phase_deg0, decay0, y00, x00, yscale1, freq1, phase_deg1, y01 = p
    p0 = [yscale0, freq0, phase_deg0, decay0, 0]
    p1 = [yscale1, freq1, phase_deg1, y01]
    return y00 + decaysin(x, *p0) * sinfunc(x, *p1)


def fittwofreq_decaysin(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray]:
    """
    Fit data to a two-frequency decaying sinusoidal function.

    Args:
        xdata: X-axis data points
        ydata: Y-axis data points
        fitparams: Optional initial parameters

    Returns:
        Tuple of (optimized_parameters, covariance_matrix)
    """
    if fitparams is None:
        fitparams = [None] * 10

    # Initialize using Fourier transform
    fourier = np.fft.fft(ydata)
    fft_freqs = np.fft.fftfreq(len(ydata), d=xdata[1] - xdata[0])
    fft_phases = np.angle(fourier)
    sorted_fourier = np.sort(fourier)
    max_ind = np.argwhere(fourier == sorted_fourier[-1])[0][0]

    if max_ind == 0:
        max_ind = np.argwhere(fourier == sorted_fourier[-2])[0][0]

    max_freq = np.abs(fft_freqs[max_ind])
    max_phase = fft_phases[max_ind]

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = max(ydata) - min(ydata)  # yscale0
    if fitparams[1] is None:
        fitparams[1] = max_freq  # freq0
    if fitparams[2] is None:
        fitparams[2] = max_phase * 180 / np.pi  # phase_deg0
    if fitparams[3] is None:
        fitparams[3] = max(xdata) - min(xdata)  # decay0
    if fitparams[4] is None:
        fitparams[4] = np.mean(ydata)  # y00
    if fitparams[5] is None:
        fitparams[5] = xdata[0]  # x00
    if fitparams[6] is None:
        fitparams[6] = 1  # yscale1
    if fitparams[7] is None:
        fitparams[7] = 1 / 10  # freq1
    if fitparams[8] is None:
        fitparams[8] = 0  # phase_deg1
    if fitparams[9] is None:
        fitparams[9] = 0  # y01

    bounds = (
        [
            0.75 * fitparams[0],
            0.1 / (max(xdata) - min(xdata)),
            -360,
            0.3 * (max(xdata) - min(xdata)),
            np.min(ydata),
            xdata[0] - (xdata[-1] - xdata[0]),
            0.9,
            0.01,
            -360,
            -0.1,
        ],
        [
            1.25 * fitparams[0],
            15 / (max(xdata) - min(xdata)),
            360,
            np.inf,
            np.max(ydata),
            xdata[-1] + (xdata[-1] - xdata[0]),
            1.1,
            10,
            360,
            0.1,
        ],
    )

    return generic_fit(
        twofreq_decaysin,
        xdata,
        ydata,
        fitparams,
        bounds=bounds,
        error_message="Warning: Fit two-frequency decaying sine failed!",
    )[
        :2
    ]  # Return only pOpt and pCov


# ====================================================== #
# Hanger Resonator Fit Functions
# ====================================================== #


def hangerfunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Complex Hanger function for resonator fitting.

    Args:
        x: X-axis data points (frequency)
        p: Parameters [f0, Qi, Qe, phi, scale]

    Returns:
        Complex S21 response
    """
    f0, Qi, Qe, phi, scale = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return scale * (1 - Q0 / Qe * np.exp(1j * phi) / (1 + 2j * Q0 * (x - f0) / f0))


def hangerS21func(x: np.ndarray, *p) -> np.ndarray:
    """
    Magnitude of Hanger function for resonator fitting.

    Args:
        x: X-axis data points (frequency)
        p: Parameters [f0, Qi, Qe, phi, scale]

    Returns:
        Magnitude of S21 response
    """
    f0, Qi, Qe, phi, scale = p
    Q0 = 1 / (1 / Qi + np.real(1 / Qe))
    return np.abs(hangerfunc(x, *p))


def hangerS21func_sloped(x: np.ndarray, *p) -> np.ndarray:
    """
    Magnitude of Hanger function with slope for resonator fitting.

    Args:
        x: X-axis data points (frequency)
        p: Parameters [f0, Qi, Qe, phi, scale, slope]

    Returns:
        Magnitude of S21 response with slope
    """
    f0, Qi, Qe, phi, scale, slope = p
    return hangerS21func(x, f0, 1e4 * Qi, 1e4 * Qe, phi, scale) + slope * (x - f0)


def hangerphasefunc(x: np.ndarray, *p) -> np.ndarray:
    """
    Phase of Hanger function for resonator fitting.

    Args:
        x: X-axis data points (frequency)
        p: Parameters [f0, Qi, Qe, phi, scale]

    Returns:
        Phase of S21 response
    """
    return np.angle(hangerfunc(x, *p))


def fithanger(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit data to a Hanger function.

    Args:
        xdata: X-axis data points (frequency)
        ydata: Y-axis data points (magnitude)
        fitparams: Optional initial parameters [f0, Qi, Qe, phi, scale, slope]

    Returns:
        Tuple of (optimized_parameters, covariance_matrix, initial_parameters)
    """
    if fitparams is None:
        fitparams = [None] * 6

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = xdata[np.argmin(np.abs(ydata))]  # f0
    if fitparams[1] is None:
        fitparams[1] = 8  # Qi
    if fitparams[2] is None:
        fitparams[2] = 3  # Qe
    if fitparams[3] is None:
        fitparams[3] = 0  # phi
    if fitparams[4] is None:
        fitparams[4] = max(ydata)  # scale
    if fitparams[5] is None:
        fitparams[5] = 0  # slope

    bounds = (
        [np.min(xdata), 0, 0, -np.inf, 0, -np.inf],
        [np.max(xdata), np.inf, np.inf, np.inf, np.inf, np.inf],
    )

    fitparams = validate_bounds(fitparams, bounds)
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            hangerS21func_sloped, xdata, ydata, p0=fitparams, bounds=bounds
        )
        pOpt, pCov = sp.optimize.curve_fit(
            hangerS21func_sloped, xdata, ydata, p0=pOpt, bounds=bounds
        )
    except RuntimeError:
        print("Warning: Fit hanger failed!")
        traceback.print_exc()

    return pOpt, pCov, fitparams


# ====================================================== #
# Randomized Benchmarking Fit Functions
# ====================================================== #


def rb_func(depth: np.ndarray, p: float, a: float, b: float) -> np.ndarray:
    """
    Randomized benchmarking function.

    Args:
        depth: Sequence depth
        p: Depolarizing parameter
        a: Amplitude
        b: Offset

    Returns:
        Fidelity as a function of sequence depth
    """
    return a * p**depth + b


def rb_error(p: float, d: int) -> float:
    """
    Calculate average error rate over all gates in sequence.

    Args:
        p: Depolarizing parameter
        d: Dimension of system (2^number of qubits)

    Returns:
        Average error rate
    """
    return 1 - (p + (1 - p) / d)


def error_fit_err(cov_p: float, d: int) -> float:
    """
    Return covariance of randomized benchmarking error.

    Args:
        cov_p: Covariance of depolarizing parameter
        d: Dimension of system (2^number of qubits)

    Returns:
        Covariance of error
    """
    return cov_p * (1 / d - 1) ** 2


def rb_gate_fidelity(p_rb: float, p_irb: float, d: int) -> float:
    """
    Calculate gate fidelity from regular and interleaved RB.

    Args:
        p_rb: Depolarizing parameter from regular RB
        p_irb: Depolarizing parameter from interleaved RB
        d: Dimension of system (2^number of qubits)

    Returns:
        Gate fidelity
    """
    return 1 - (d - 1) * (1 - p_irb / p_rb) / d


def fitrb(
    xdata: np.ndarray, ydata: np.ndarray, fitparams: Optional[List[float]] = None
) -> Tuple[List[float], np.ndarray]:
    """
    Fit data to a randomized benchmarking function.

    Args:
        xdata: X-axis data points (sequence depth)
        ydata: Y-axis data points (fidelity)
        fitparams: Optional initial parameters [p, a, b]

    Returns:
        Tuple of (optimized_parameters, covariance_matrix)
    """
    if fitparams is None:
        fitparams = [None] * 3

    # Initialize parameters if not provided
    if fitparams[0] is None:
        fitparams[0] = 0.9  # p
    if fitparams[1] is None:
        fitparams[1] = np.max(ydata) - np.min(ydata)  # a
    if fitparams[2] is None:
        fitparams[2] = np.min(ydata)  # b

    bounds = ([0, 0, 0], [1, 10 * np.max(ydata) - np.min(ydata), np.max(ydata)])

    fitparams = validate_bounds(fitparams, bounds)
    pOpt = fitparams
    pCov = np.full(shape=(len(fitparams), len(fitparams)), fill_value=np.inf)

    try:
        pOpt, pCov = sp.optimize.curve_fit(
            rb_func, xdata, ydata, p0=fitparams, bounds=bounds
        )
        print(pOpt)
        print(pCov[0][0], pCov[1][1], pCov[2][2])
    except RuntimeError:
        print("Warning: Fit randomized benchmarking failed!")
        traceback.print_exc()
        pOpt = [np.nan] * len(pOpt)

    return pOpt, pCov


# ====================================================== #
# Adiabatic Pi Pulse Functions
# ====================================================== #


def adiabatic_amp(
    t: np.ndarray, amp_max: float, beta: float, period: float
) -> np.ndarray:
    """
    Amplitude function for adiabatic pi pulse.

    Args:
        t: Time points
        amp_max: Maximum amplitude
        beta: Slope of frequency sweep
        period: Period of pulse

    Returns:
        Amplitude as a function of time
    """
    return amp_max / np.cosh(beta * (2 * t / period - 1))


def adiabatic_phase(t: np.ndarray, mu: float, beta: float, period: float) -> np.ndarray:
    """
    Phase function for adiabatic pi pulse.

    Args:
        t: Time points
        mu: Width of frequency sweep
        beta: Slope of frequency sweep
        period: Period of pulse

    Returns:
        Phase as a function of time
    """
    return mu * np.log(adiabatic_amp(t, amp_max=1, beta=beta, period=period))


def adiabatic_iqamp(
    t: np.ndarray, amp_max: float, mu: float, beta: float, period: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate I and Q amplitudes for adiabatic pi pulse.

    Args:
        t: Time points
        amp_max: Maximum amplitude
        mu: Width of frequency sweep
        beta: Slope of frequency sweep
        period: Period of pulse

    Returns:
        Tuple of (I amplitude, Q amplitude)
    """
    amp = np.abs(adiabatic_amp(t, amp_max=amp_max, beta=beta, period=period))
    phase = adiabatic_phase(t, mu=mu, beta=beta, period=period)
    iamp = amp * (np.cos(phase) + 1j * np.sin(phase))
    qamp = amp * (-np.sin(phase) + 1j * np.cos(phase))
    return np.real(iamp), np.real(qamp)


# ====================================================== #
# Transmon Flux Model
# Koch et al., Phys. Rev. A 76, 042319 (2007)
# ====================================================== #


def transmon_flux(
    g: np.ndarray,
    f_max: float,
    E_C: float,
    g_period: float,
    g_offset: float,
    d: float,
) -> np.ndarray:
    """
    Transmon qubit frequency as a function of external flux (gain).

    f(g) = (f_max + E_C) * (cos^2(phi) + d^2 * sin^2(phi))^0.25 - E_C

    where phi = pi * (g - g_offset) / g_period.

    Args:
        g: Flux gain values
        f_max: Maximum qubit frequency at the sweet spot (MHz)
        E_C: Charging energy (MHz), typically 150-300 for transmons
        g_period: Gain period corresponding to one flux quantum
        g_offset: Gain value at the sweet spot
        d: Junction asymmetry, (EJ1 - EJ2) / (EJ1 + EJ2), in [0, 1]

    Returns:
        Qubit frequency in MHz
    """
    phi = np.pi * (g - g_offset) / g_period
    return (f_max + E_C) * (np.cos(phi) ** 2 + d**2 * np.sin(phi) ** 2) ** 0.25 - E_C


def fit_transmon_flux(
    gain: np.ndarray,
    freq: np.ndarray,
    E_C: float,
    fitparams: Optional[List[Optional[float]]] = None,
) -> Tuple[List[float], np.ndarray, List[float]]:
    """
    Fit transmon frequency vs flux gain with E_C fixed.

    Fits 4 free parameters (f_max, g_period, g_offset, d) while holding E_C
    constant (from device config).

    Args:
        gain: Flux gain values
        freq: Measured qubit frequencies (MHz)
        E_C: Charging energy (MHz), held fixed
        fitparams: Optional initial guesses [f_max, g_period, g_offset, d].
            Use None for individual entries to auto-guess.

    Returns:
        Tuple of (pOpt, pCov, fitparams) where pOpt is the full 5-parameter
        tuple (f_max, E_C, g_period, g_offset, d).
    """
    if fitparams is None:
        fitparams = [None, None, None, None]

    # Auto-guess initial parameters
    f_max_guess = fitparams[0] if fitparams[0] is not None else np.nanmax(freq)
    g_period_guess = fitparams[1] if fitparams[1] is not None else 2 * abs(gain[-1] - gain[0])
    g_offset_guess = fitparams[2] if fitparams[2] is not None else gain[np.nanargmax(freq)]
    d_guess = fitparams[3] if fitparams[3] is not None else 0.01

    p0 = [f_max_guess, g_period_guess, g_offset_guess, d_guess]

    # Filter out NaN/inf entries before fitting
    mask = np.isfinite(freq) & np.isfinite(gain)
    gain_fit = gain[mask]
    freq_fit = freq[mask]

    bounds = (
        [0, 0.001, gain_fit.min() - abs(np.ptp(gain_fit)), 0],
        [np.inf, np.inf, gain_fit.max() + abs(np.ptp(gain_fit)), 1],
    )

    # Inner function with E_C closed over
    def _model(g, f_max, g_period, g_offset, d):
        return transmon_flux(g, f_max, E_C, g_period, g_offset, d)

    pOpt, pCov, p0_used = generic_fit(
        _model, gain_fit, freq_fit, p0, bounds=bounds,
        error_message="Warning: transmon flux fit failed!",
    )

    # Return full 5-parameter tuple, inserting E_C
    pOpt_full = [pOpt[0], E_C, pOpt[1], pOpt[2], pOpt[3]]
    # Expand covariance to 5x5 with zeros for the fixed E_C row/col
    pCov_full = np.zeros((5, 5))
    idx = [0, 2, 3, 4]  # indices in full param list that were fitted
    for i, ii in enumerate(idx):
        for j, jj in enumerate(idx):
            pCov_full[ii, jj] = pCov[i, j]

    return pOpt_full, pCov_full, p0_used


class QuadraticFluxConverter:
    """Convert between flux gain and qubit frequency using a quadratic model."""

    def __init__(self, a, b, c, direction="pos"):
        self.a = a
        self.b = b
        self.c = c
        self.direction = direction

    @property
    def sweet_spot(self):
        return -self.b / (2 * self.a)

    @property
    def f_max(self):
        return self.c - self.b**2 / (4 * self.a)

    def gain_to_freq(self, g):
        g = np.asarray(g)
        return self.a * g**2 + self.b * g + self.c

    def freq_to_gain(self, f):
        f = np.asarray(f)
        discriminant = self.b**2 - 4 * self.a * (self.c - f)
        discriminant = np.maximum(discriminant, 0)
        root_pos = (-self.b + np.sqrt(discriminant)) / (2 * self.a)
        root_neg = (-self.b - np.sqrt(discriminant)) / (2 * self.a)
        if self.direction == "pos":
            return np.maximum(root_pos, root_neg)
        else:
            return np.minimum(root_pos, root_neg)


class TransmonFluxConverter:
    """Convert between flux gain and qubit frequency using the transmon SQUID model."""

    def __init__(self, f_max, E_C, g_period, g_offset, d, direction="pos"):
        self.params = (f_max, E_C, g_period, g_offset, d)
        self.direction = direction

    @property
    def sweet_spot(self):
        return self.params[3]  # g_offset

    @property
    def f_max(self):
        return self.params[0]

    def gain_to_freq(self, g):
        return transmon_flux(np.asarray(g, dtype=float), *self.params)

    def freq_to_gain(self, f):
        """Invert the transmon model numerically using brentq on the monotonic branch."""
        f = np.asarray(f, dtype=float)
        scalar = f.ndim == 0
        f = np.atleast_1d(f)

        g_offset = self.params[3]
        g_period = self.params[2]

        # Search on the monotonic half-period branch.
        # g_period is a full period of the cosine, so f(g_offset) = f(g_offset ± g_period) = f_max.
        # Using the full period makes brentq fail (same value at both endpoints).
        half = g_period / 2
        if self.direction == "pos":
            g_lo, g_hi = g_offset, g_offset + half
        else:
            g_lo, g_hi = g_offset - half, g_offset

        results = np.empty_like(f)
        for i, fi in enumerate(f):
            try:
                results[i] = sp.optimize.brentq(
                    lambda g: transmon_flux(g, *self.params) - fi,
                    g_lo, g_hi,
                )
            except ValueError:
                # Target frequency outside model range — clamp to boundary
                f_lo = transmon_flux(g_lo, *self.params)
                f_hi = transmon_flux(g_hi, *self.params)
                results[i] = g_lo if abs(f_lo - fi) < abs(f_hi - fi) else g_hi

        return float(results[0]) if scalar else results


def flux_converter_from_config(cfg_flux, cfg_qubit, qi, direction="pos"):
    """
    Factory: load a FluxConverter from config, preferring transmon params over quadratic.

    Args:
        cfg_flux: Config section for hw.soc.dacs.flux
        cfg_qubit: Config section for device.qubit
        qi: Qubit index
        direction: 'pos' or 'neg' sweep direction from sweet spot

    Returns:
        TransmonFluxConverter, QuadraticFluxConverter, or None
    """
    if cfg_flux is not None and hasattr(cfg_flux, "transmon_f_max"):
        return TransmonFluxConverter(
            cfg_flux.transmon_f_max[qi],
            cfg_flux.transmon_E_C[qi],
            cfg_flux.transmon_g_period[qi],
            cfg_flux.transmon_g_offset[qi],
            cfg_flux.transmon_d[qi],
            direction,
        )
    if cfg_flux is not None:
        a = cfg_flux.quad_a[qi] if hasattr(cfg_flux, "quad_a") else 0
        b = cfg_flux.quad_b[qi] if hasattr(cfg_flux, "quad_b") else 0
        c = cfg_flux.quad_c[qi] if hasattr(cfg_flux, "quad_c") else 0
        if a != 0 or b != 0:
            return QuadraticFluxConverter(a, b, c, direction)
    return None
