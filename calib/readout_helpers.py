"""
Readout Helpers Module
======================

This module provides utilities for analyzing and visualizing quantum readout data.
It includes tools for:
- Histogram generation and analysis
- IQ data rotation and processing
- Gaussian fitting of state distributions
- Fidelity calculation between quantum states
- Modeling T1 decay effects on readout signals

Author: Schuster Lab
"""

from __future__ import annotations

import datetime
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.special import erf

# Type aliases for better code readability
ArrayLike = Union[np.ndarray, List[float]]
IQData = Dict[str, np.ndarray]
FitParams = Tuple[np.ndarray, np.ndarray]

# Standard colors for plotting quantum states
COLORS = {
    'ground': "#4053d3",    # Blue for ground state |g⟩
    'excited': "#b51d14",   # Red for excited state |e⟩
    'f_state': "#2ca02c",   # Green for f state |f⟩
    'threshold': "#333333", # Dark gray for threshold lines
    'ground_comp': "#9aa5e8",   # Light blue: not-excited (ground-level) component of the e-shot fit
    'excited_comp': "#e39b95",  # Light red: excited component (peak + T1 tail) of the e-shot fit
    'f_comp': "#9ed49e",      # Light green: f component (peak + T1 tail) of a three-state fit
}

# Default plot settings
DEFAULT_PLOT_CONFIG = {
    'alpha': 0.25,
    'marker_size': 0.7,
    'bins': 100,
    'histogram_bins': 200,
    'line_width': 1.0,
}

# Physical constants and defaults
DEFAULT_SPAN_FACTOR = 0.5  # Factor for automatic span calculation
GAUSSIAN_NORMALIZATION = 1 / np.sqrt(2 * np.pi)


#------------------------------------------------------------------------------
# Basic Utility Functions
#------------------------------------------------------------------------------

def rotate_coordinates(x: ArrayLike, y: ArrayLike, theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate points in the x-y plane by angle theta.
    
    This function applies a 2D rotation matrix to transform coordinates:
    [x'] = [cos(θ) -sin(θ)] [x]
    [y']   [sin(θ)  cos(θ)] [y]
    
    Parameters
    ----------
    x : array_like
        x-coordinates to rotate
    y : array_like  
        y-coordinates to rotate
    theta : float
        Rotation angle in radians (positive = counterclockwise)
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Rotated (x, y) coordinates as numpy arrays
        
    Examples
    --------
    >>> x, y = [1, 0], [0, 1]
    >>> x_rot, y_rot = rotate_coordinates(x, y, np.pi/2)
    >>> # Rotates 90° counterclockwise: (1,0) -> (0,1), (0,1) -> (-1,0)
    """
    x, y = np.asarray(x), np.asarray(y)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
    x_rotated = x * cos_theta - y * sin_theta
    y_rotated = x * sin_theta + y * cos_theta
    
    return x_rotated, y_rotated


def rotate_iq_data(data: IQData, theta: float) -> IQData:
    """
    Rotate all IQ data in a dictionary by angle theta.
    
    This function rotates the I and Q quadratures for all quantum states
    present in the data dictionary. It preserves the original data by
    creating a deep copy before rotation.
    
    Parameters
    ----------
    data : dict
        Dictionary containing IQ data with keys like 'Ig', 'Qg', 'Ie', 'Qe'
        and optionally 'If', 'Qf' for ground, excited, and f states
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    dict
        New dictionary with rotated IQ data, preserving all original keys
        
    Notes
    -----
    The function automatically detects and rotates f-state data if present.
    All other keys in the dictionary are preserved unchanged.
    """
    rotated_data = deepcopy(data)
    
    # Rotate ground and excited state data (always present)
    rotated_data["Ig"], rotated_data["Qg"] = rotate_coordinates(
        data["Ig"], data["Qg"], theta
    )
    rotated_data["Ie"], rotated_data["Qe"] = rotate_coordinates(
        data["Ie"], data["Qe"], theta
    )
    
    # Rotate f state data if available
    if "If" in data and "Qf" in data:
        rotated_data["If"], rotated_data["Qf"] = rotate_coordinates(
            data["If"], data["Qf"], theta
        )
        
    return rotated_data


def create_histogram(data: ArrayLike, n_bins: int = DEFAULT_PLOT_CONFIG['histogram_bins']) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a normalized histogram from data.
    
    Parameters
    ----------
    data : array_like
        Input data to histogram
    n_bins : int, optional
        Number of histogram bins, default from DEFAULT_PLOT_CONFIG
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Bin centers and normalized histogram values (probability density)
        
    Notes
    -----
    The histogram is normalized so that the integral over all bins equals 1,
    making it a proper probability density function.
    """
    data = np.asarray(data)
    hist_values, bin_edges = np.histogram(data, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist_values


#------------------------------------------------------------------------------
# Statistical Distribution Functions
#------------------------------------------------------------------------------

def gaussian(x: ArrayLike, magnitude: float, center: float, width: float) -> np.ndarray:
    """
    Normalized Gaussian function for fitting quantum state distributions.
    
    The function is normalized so that the integral equals the magnitude parameter.
    This is useful for fitting probability distributions where the magnitude
    represents the relative population of a quantum state.
    
    Parameters
    ----------
    x : array_like
        Input x values
    magnitude : float
        Magnitude (area under curve, proportional to state population)
    center : float
        Center position (mean of the distribution)
    width : float
        Width (standard deviation) of the distribution
        
    Returns
    -------
    np.ndarray
        Gaussian values at input x points
        
    Notes
    -----
    The normalization ensures ∫ gaussian(x) dx = magnitude
    """
    x = np.asarray(x)
    normalization = magnitude * GAUSSIAN_NORMALIZATION / width
    exponent = -((x - center) ** 2) / (2 * width**2)
    return normalization * np.exp(exponent)


def double_gaussian(x: ArrayLike, mag1: float, center1: float, width: float, 
                   mag2: float, center2: float) -> np.ndarray:
    """
    Sum of two Gaussian functions with shared width parameter.
    
    This function is useful for modeling situations where two quantum states
    have the same measurement noise (width) but different populations and
    center positions, such as when there's leakage between states.
    
    Parameters
    ----------
    x : array_like
        Input x values
    mag1 : float
        Magnitude of first Gaussian (state population)
    center1 : float
        Center of first Gaussian
    width : float
        Shared width (standard deviation) for both Gaussians
    mag2 : float
        Magnitude of second Gaussian (state population)
    center2 : float
        Center of second Gaussian
        
    Returns
    -------
    np.ndarray
        Sum of two Gaussians at input x points
    """
    x = np.asarray(x)
    normalization = GAUSSIAN_NORMALIZATION / width
    
    gaussian1 = mag1 * np.exp(-((x - center1) ** 2) / (2 * width**2))
    gaussian2 = mag2 * np.exp(-((x - center2) ** 2) / (2 * width**2))
    
    return normalization * (gaussian1 + gaussian2)

def double_gaussian_xnorm(x: ArrayLike, mag1: float, center1: float, width1: float, 
                   mag2: float, center2: float, width2: float) -> np.ndarray:
    """
    Sum of two Gaussian functions with shared width parameter.
    
    This function is useful for modeling situations where two quantum states
    have the same measurement noise (width) but different populations and
    center positions, such as when there's leakage between states.
    
    Parameters
    ----------
    x : array_like
        Input x values
    mag1 : float
        Magnitude of first Gaussian (state population)
    center1 : float
        Center of first Gaussian
    width : float
        Shared width (standard deviation) for both Gaussians
    mag2 : float
        Magnitude of second Gaussian (state population)
    center2 : float
        Center of second Gaussian
        
    Returns
    -------
    np.ndarray
        Sum of two Gaussians at input x points
    """
    x = np.asarray(x)
    
    gaussian1 = mag1 * np.exp(-0.5 * ((x - center1) ** 2) / (width1**2))
    gaussian2 = mag2 * np.exp(-0.5 * ((x - center2) ** 2) / (width2**2))
    
    return gaussian1 + gaussian2


def t1_decay_distribution(voltage: ArrayLike, v_ground: float, v_excited: float, 
                         sigma: float, t_ratio: float) -> np.ndarray:
    """
    Distribution function modeling T1 decay during measurement.
    
    This function models the probability distribution of measurement outcomes
    when T1 relaxation occurs during the measurement process. The excited state
    population decays exponentially during measurement, creating a characteristic
    asymmetric distribution.
    
    Parameters
    ----------
    voltage : array_like
        Voltage values where to evaluate the distribution
    v_ground : float
        Ground state voltage level
    v_excited : float
        Excited state voltage level  
    sigma : float
        Standard deviation of measurement noise
    t_ratio : float
        Ratio of measurement time to T1 time (t_meas/T1)
        
    Returns
    -------
    np.ndarray
        Probability distribution values
        
    Notes
    -----
    The distribution accounts for:
    - Exponential decay during measurement: exp(-t_meas/T1)
    - Gaussian measurement noise with width sigma
    - Voltage separation between ground and excited states
    
    For t_ratio → 0 (fast measurement), this approaches a Gaussian.
    For large t_ratio (slow measurement), significant decay occurs.
    """
    voltage = np.asarray(voltage)
    dv = v_excited - v_ground
    
    # Avoid division by zero
    if np.abs(dv) < 1e-10:
        return np.zeros_like(voltage)
    
    # Calculate the exponential and error function terms
    exp_term = np.exp(t_ratio * (t_ratio * sigma**2 / (2 * dv**2) - (voltage - v_ground) / dv))
    
    erf_term1 = erf((t_ratio * sigma**2 / dv + v_excited - voltage) / (np.sqrt(2) * sigma))
    erf_term2 = erf((-t_ratio * sigma**2 / dv + voltage - v_ground) / (np.sqrt(2) * sigma))
    
    distribution = (t_ratio / (2 * dv)) * exp_term * (erf_term1 + erf_term2)
    
    return np.abs(distribution)


def excited_state_model(x: ArrayLike, v_ground: float, v_excited: float, 
                       sigma: float, t_ratio: float) -> np.ndarray:
    """
    Complete model for excited state distribution including T1 decay effects.
    
    This function combines:
    1. A Gaussian for the population that remains excited (exp(-t_meas/T1))
    2. The T1 decay distribution for the population that decays during measurement
    
    Parameters
    ----------
    x : array_like
        Input x values (voltage levels)
    v_ground : float
        Ground state voltage level
    v_excited : float
        Excited state voltage level
    sigma : float
        Standard deviation of measurement noise
    t_ratio : float
        Ratio of measurement time to T1 time (t_meas/T1)
        
    Returns
    -------
    np.ndarray
        Complete excited state distribution
        
    Notes
    -----
    The total distribution is:
    P(x) = exp(-t_meas/T1) * Gaussian(x; v_excited, σ) + T1_decay_dist(x)
    
    This accounts for both the undecayed excited population and the
    population that decays during the measurement process.
    """
    x = np.asarray(x)
    
    # Remaining excited population (undecayed)
    undecayed_component = gaussian(x, np.exp(-t_ratio), v_excited, sigma)
    
    # Population that decayed during measurement
    decay_component = t1_decay_distribution(x, v_ground, v_excited, sigma, t_ratio)
    
    return undecayed_component + decay_component


def combined_state_model(x: ArrayLike, mag_ground: float, v_ground: float, 
                        v_excited: float, sigma: float, t_ratio: float) -> np.ndarray:
    """
    Combined model for both ground and excited state distributions.
    
    This function models the complete readout distribution when both ground
    and excited states are present, accounting for T1 decay in the excited state.
    
    Parameters
    ----------
    x : array_like
        Input x values (voltage levels)
    mag_ground : float
        Relative magnitude (population) of ground state
    v_ground : float
        Ground state voltage level
    v_excited : float
        Excited state voltage level
    sigma : float
        Standard deviation of measurement noise
    t_ratio : float
        Ratio of measurement time to T1 time (t_meas/T1)
        
    Returns
    -------
    np.ndarray
        Combined distribution for both states
        
    Notes
    -----
    The excited state magnitude is automatically calculated as (1 - mag_ground)
    to ensure proper normalization of the total probability.
    """
    x = np.asarray(x)
    mag_excited = 1 - mag_ground
    
    # Ground state component (simple Gaussian)
    ground_component = gaussian(x, mag_ground, v_ground, sigma)
    
    # Excited state component (with T1 decay effects)
    excited_undecayed = gaussian(x, mag_excited * np.exp(-t_ratio), v_excited, sigma)
    excited_decay = mag_excited * t1_decay_distribution(x, v_ground, v_excited, sigma, t_ratio)
    
    return ground_component + excited_undecayed + excited_decay


def combined_fit_components(x: ArrayLike, mag_ground: float, data: IQData,
                            tm: Optional[float] = None) -> List[Tuple[np.ndarray, str]]:
    """
    Split a combined_state_model fit into its two components.

    Both the g-shot and e-shot histograms are fit with combined_state_model: a
    ground-level gaussian (population read as |g⟩ -- prepared there or decayed
    there during readout) plus an excited gaussian carrying the T1-decay tail.
    The two components sum to the total fit curve, so overlaying them shows how
    much of each blob is |g⟩ vs |e⟩.

    Parameters
    ----------
    x : array_like
        Voltage levels to evaluate the components at
    mag_ground : float
        Ground-state weight of this fit (paramsg2[0] for the g-shot,
        mag_ground_e for the e-shot)
    data : dict
        Fit results containing 'vg', 've', 'sigma', 'tm'
    tm : float, optional
        Tm/T1 to use for this shot. Defaults to data['tm']; pass data['tm_g']
        for the g-shot when float_tm gives the two shots different Tm/T1 values,
        so the components match the total fit drawn with that shot's tm.

    Returns
    -------
    list of (np.ndarray, str)
        [(ground_level_component, color), (excited_component, color)]
    """
    x = np.asarray(x)
    if tm is None:
        tm = data['tm']
    total = combined_state_model(x, mag_ground, data['vg'], data['ve'], data['sigma'], tm)
    ground_level = gaussian(x, mag_ground, data['vg'], data['sigma'])
    excited = total - ground_level
    return [(ground_level, COLORS['ground_comp']), (excited, COLORS['excited_comp'])]


def three_state_model(x: ArrayLike, p_g: float, p_e: float, v_g: float, v_e: float,
                      v_f: float, sigma: float, tm_g: float, tm_e: float,
                      tm_f: float) -> np.ndarray:
    """
    Model for one single-shot histogram when |g>, |e> and |f> are all populated.

    The shot is described by the populations actually prepared, (p_g, p_e, p_f),
    with p_f = 1 - p_g - p_e. Each population contributes a gaussian for the
    fraction that did not change state during the window, plus a tail for the
    fraction that did:
    - |g>: exp(-tm_g) at v_g plus a tail running UP to v_e (readout-induced
      heating out of |g>, the one transition available to it)
    - |e>: exp(-tm_e) at v_e plus its T1-decay tail running down to v_g
    - |f>: exp(-tm_f) at v_f plus its T1-decay tail running down to v_g

    Unlike combined_state_model (two levels, one Tm/T1) the rate is a property
    of the *state*, not of the shot: tm_g governs every |g> component, tm_e
    every |e> component and tm_f every |f> component, in all three shots.
    tm_g = 0 recovers a plain gaussian at v_g.

    Each tail is the distribution of the time-averaged voltage for a shot that
    switched level once during the window, so the upward |g> tail is
    t1_decay_distribution with its two levels swapped.

    Parameters
    ----------
    x : array_like
        Input x values (voltage levels)
    p_g, p_e : float
        Prepared |g> and |e> populations of this shot; |f> takes the remainder
    v_g, v_e, v_f : float
        Voltage levels of the three states
    sigma : float
        Standard deviation of measurement noise (shared by all blobs)
    tm_g : float
        Ratio of measurement time to the |g> -> |e> excitation time
    tm_e, tm_f : float
        Ratio of measurement time to T1 for |e> and |f>

    Returns
    -------
    np.ndarray
        Distribution for this shot
    """
    x = np.asarray(x)
    p_f = 1 - p_g - p_e

    out = p_g * (gaussian(x, np.exp(-tm_g), v_g, sigma)
                 + t1_decay_distribution(x, v_e, v_g, sigma, tm_g))
    out = out + p_e * (gaussian(x, np.exp(-tm_e), v_e, sigma)
                       + t1_decay_distribution(x, v_g, v_e, sigma, tm_e))
    out = out + p_f * (gaussian(x, np.exp(-tm_f), v_f, sigma)
                       + t1_decay_distribution(x, v_g, v_f, sigma, tm_f))
    return out


def three_state_components(x: ArrayLike, p_g: float, p_e: float,
                           data: IQData) -> List[Tuple[np.ndarray, str]]:
    """
    Split a three_state_model fit into its |g>, |e> and |f> components.

    The three components sum to the total fit curve, so overlaying them shows
    how much of each blob came from each prepared state.

    Parameters
    ----------
    x : array_like
        Voltage levels to evaluate the components at
    p_g, p_e : float
        Prepared |g> and |e> populations of the shot being drawn
    data : dict
        Fit results containing 'vg', 've', 'vf', 'sigma', 'tm_g', 'tm_e', 'tm_f'

    Returns
    -------
    list of (np.ndarray, str)
        [(g_component, color), (e_component, color), (f_component, color)]
    """
    x = np.asarray(x)
    vg, ve, vf = data['vg'], data['ve'], data['vf']
    sigma, tm_e, tm_f = data['sigma'], data['tm_e'], data['tm_f']
    tm_g = data['tm_g']
    p_f = 1 - p_g - p_e

    g = p_g * (gaussian(x, np.exp(-tm_g), vg, sigma)
               + t1_decay_distribution(x, ve, vg, sigma, tm_g))
    e = p_e * (gaussian(x, np.exp(-tm_e), ve, sigma)
               + t1_decay_distribution(x, vg, ve, sigma, tm_e))
    f = p_f * (gaussian(x, np.exp(-tm_f), vf, sigma)
               + t1_decay_distribution(x, vg, vf, sigma, tm_f))
    return [(g, COLORS['ground_comp']), (e, COLORS['excited_comp']), (f, COLORS['f_comp'])]

def plot_three_state_fit(plot_fn, data: IQData, x_g: ArrayLike, x_e: ArrayLike,
                         x_f: ArrayLike) -> None:
    """
    Overlay a three-state joint fit on one axis: each shot's total plus components.

    Parameters
    ----------
    plot_fn : callable
        The axis plotting method to draw with, e.g. ``ax.plot`` or ``ax.semilogy``
    data : dict
        Three-state fit results from fit_single_shot_gef
    x_g, x_e, x_f : array_like
        Voltage grids to draw the g-, e- and f-shot curves on
    """
    levels = (data['vg'], data['ve'], data['vf'], data['sigma'],
              data['tm_g'], data['tm_e'], data['tm_f'])
    for x, p_g, p_e in ((x_g, data['pg_g'], data['pe_g']),
                        (x_e, data['pg_e'], data['pe_e']),
                        (x_f, data['pg_f'], data['pe_f'])):
        plot_fn(x, three_state_model(x, p_g, p_e, *levels), 'k', linewidth=1)
        for comp, ccol in three_state_components(x, p_g, p_e, data):
            # A population pinned to zero by the fit's constraint toggles would
            # otherwise draw a dashed line along y = 0.
            if np.max(comp) > 0:
                plot_fn(x, comp, color=ccol, linewidth=1, linestyle='--')


def fit_fidelity(data: IQData, n: int = 2001) -> Tuple[Optional[float], Optional[float]]:
    """
    Readout fidelity between the correctly-prepared components of each fit.

    Rather than comparing the full fitted distributions, this isolates the two
    components that represent successful preparation-and-readout:
    - the ground component of the no-pi (g-shot) fit -- observing |g> with no pi
    - the excited component of the with-pi (e-shot) fit -- observing |e> with a pi
      (the excited peak plus its T1-decay tail)
    Each component is normalized to a probability distribution and the fidelity is
    the max cumulative-distribution contrast, F = max_threshold |CDF_g(v) - CDF_e(v)|.
    This measures the intrinsic separation of the two blobs, excluding the residual
    (mis-prepared) populations that the histogram-based fidelity folds in.

    Parameters
    ----------
    data : dict
        Fit results containing 'vg', 've', 'sigma', 'tm', 'mag_ground_e' and
        'paramsg2' (the [mag_gg, vg, ve, sigma, tm] g-shot fit parameters)
    n : int, optional
        Number of grid points, default 2001

    Returns
    -------
    tuple of (float, float) or (None, None)
        (fidelity, threshold voltage) at the optimal discrimination point,
        or (None, None) if the required fit parameters are missing
    """
    required = ('vg', 've', 'sigma', 'tm', 'mag_ground_e')
    if not all(k in data for k in required) or 'paramsg2' not in data or np.ndim(data['paramsg2']) == 0:
        return None, None

    vg, ve, sigma = data['vg'], data['ve'], data['sigma']
    lo = min(vg, ve) - 6 * sigma
    hi = max(vg, ve) + 6 * sigma
    x = np.linspace(lo, hi, n)

    # Ground component of the no-pi fit; excited component of the with-pi fit
    g = combined_fit_components(x, data['paramsg2'][0], data)[0][0]
    e = combined_fit_components(x, data['mag_ground_e'], data)[1][0]

    # Normalize each component to unit area, then compare cumulative distributions
    cg = np.cumsum(g) / np.sum(g)
    ce = np.cumsum(e) / np.sum(e)
    contrast = np.abs(cg - ce)
    tind = int(contrast.argmax())
    return float(contrast[tind]), float(x[tind])


def fit_fidelity_gef(data: IQData, n: int = 2001) -> Tuple[Optional[float], Optional[float]]:
    """
    g-f readout fidelity between the correctly-prepared components of a three-state fit.

    The |f>-state counterpart of fit_fidelity: it compares the |g> component of
    the no-pi shot with the |f> component of the pi_ge+pi_ef shot (the |f> peak
    plus its T1-decay tail), each normalized to a probability distribution, and
    returns F = max_threshold |CDF_g(v) - CDF_f(v)|. g-f is the discrimination
    the f-state measurement exists to characterize, so this is the fidelity
    reported for a three-state fit.

    Parameters
    ----------
    data : dict
        Three-state fit results containing 'vg', 've', 'vf', 'sigma', 'tm_e',
        'tm_f' and the per-shot populations 'pg_g', 'pe_g', 'pg_f', 'pe_f'
    n : int, optional
        Number of grid points, default 2001

    Returns
    -------
    tuple of (float, float) or (None, None)
        (fidelity, threshold voltage) at the optimal discrimination point,
        or (None, None) if the required fit parameters are missing
    """
    required = ('vg', 've', 'vf', 'sigma', 'tm_e', 'tm_f',
                'pg_g', 'pe_g', 'pg_f', 'pe_f')
    if not all(k in data for k in required):
        return None, None

    vg, ve, vf, sigma = data['vg'], data['ve'], data['vf'], data['sigma']
    lo = min(vg, ve, vf) - 6 * sigma
    hi = max(vg, ve, vf) + 6 * sigma
    x = np.linspace(lo, hi, n)

    # |g> component of the no-pi shot; |f> component of the pi_ge+pi_ef shot
    g = three_state_components(x, data['pg_g'], data['pe_g'], data)[0][0]
    f = three_state_components(x, data['pg_f'], data['pe_f'], data)[2][0]

    if np.sum(g) <= 0 or np.sum(f) <= 0:
        return None, None

    cg = np.cumsum(g) / np.sum(g)
    cf = np.cumsum(f) / np.sum(f)
    contrast = np.abs(cg - cf)
    tind = int(contrast.argmax())
    return float(contrast[tind]), float(x[tind])


#------------------------------------------------------------------------------
# Fitting Functions
#------------------------------------------------------------------------------

def fit_gaussian_distribution(data: ArrayLike, n_bins: int = DEFAULT_PLOT_CONFIG['histogram_bins'], 
                            initial_guess: Optional[List[float]] = None, 
                            plot_result: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a Gaussian distribution to data using histogram-based fitting.
    
    This function creates a histogram of the input data and fits a normalized
    Gaussian function to it. This approach is more robust than fitting to
    raw data points when dealing with noisy quantum measurements.
    
    Parameters
    ----------
    data : array_like
        Input data to fit
    n_bins : int, optional
        Number of histogram bins for fitting
    initial_guess : list of float, optional
        Initial parameter guess [center, width]. If None, estimated from data
    plot_result : bool, optional
        Whether to create a plot showing the fit result
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Fitted parameters [center, width], bin centers, and histogram values
        
    Notes
    -----
    The magnitude parameter is fixed to 1 during fitting, so the returned
    Gaussian represents a probability density function.
    """
    data = np.asarray(data)
    
    # Create histogram
    bin_centers, hist_values = create_histogram(data, n_bins)
    
    # Estimate initial parameters if not provided
    if initial_guess is None:
        # Weighted mean and standard deviation
        center_estimate = np.average(bin_centers, weights=hist_values)
        width_estimate = np.std(data)
        initial_guess = [center_estimate, width_estimate]
    
    # Fit Gaussian with fixed magnitude of 1
    try:
        fitted_params, _ = curve_fit(
            lambda x, center, width: gaussian(x, 1, center, width), 
            bin_centers, hist_values, 
            p0=initial_guess,
            maxfev=5000  # Increase max iterations for better convergence
        )
    except RuntimeError as e:
        print(f"Warning: Gaussian fit failed ({e}). Using initial guess.")
        fitted_params = np.array(initial_guess)
    
    # Create plot if requested
    if plot_result:
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, hist_values, 'ko', markersize=3, label="Data", alpha=0.7)
        plt.plot(bin_centers, gaussian(bin_centers, 1, *fitted_params), 
                'r-', linewidth=2, label=f"Fit: μ={fitted_params[0]:.3f}, σ={fitted_params[1]:.3f}")
        plt.xlabel("Value")
        plt.ylabel("Probability Density")
        plt.title("Gaussian Distribution Fit")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    return fitted_params, bin_centers, hist_values


#------------------------------------------------------------------------------
# Single Shot Analysis Functions  
#------------------------------------------------------------------------------

def calculate_optimal_rotation_angle(data: IQData, use_f_state: bool = True) -> float:
    """
    Calculate the optimal rotation angle to maximize state separation.
    
    The optimal angle rotates the IQ plane so that the quantum states are
    maximally separated along the I-axis, improving discrimination fidelity.
    
    Parameters
    ----------
    data : dict
        Dictionary containing IQ data with keys 'Ig', 'Qg', 'Ie', 'Qe'
        and optionally 'If', 'Qf'
    use_f_state : bool, optional
        If True and f-state data is available, use g-f separation for rotation.
        Otherwise, use g-e separation.
        
    Returns
    -------
    float
        Optimal rotation angle in radians
        
    Notes
    -----
    The rotation angle is calculated to align the line connecting state centers
    with the I-axis. This maximizes the projection of the state separation
    onto the measurement axis.
    """
    # Calculate median positions (more robust than mean for noisy data)
    i_ground, q_ground = np.median(data["Ig"]), np.median(data["Qg"])
    i_excited, q_excited = np.median(data["Ie"]), np.median(data["Qe"])
    
    # Use f-state for rotation if available and requested
    if use_f_state and "If" in data and "Qf" in data:
        i_f, q_f = np.median(data["If"]), np.median(data["Qf"])
        # Calculate angle to align g-f separation with I-axis
        angle = -np.arctan2(q_f - q_ground, i_f - i_ground)
    else:
        # Calculate angle to align g-e separation with I-axis  
        angle = -np.arctan2(q_excited - q_ground, i_excited - i_ground)
    
    return angle


def calculate_discrimination_metrics(ground_data: ArrayLike, excited_data: ArrayLike, 
                                   n_bins: int = DEFAULT_PLOT_CONFIG['bins']) -> Dict[str, float]:
    """
    Calculate optimal threshold and discrimination fidelity between two states.
    
    This function finds the threshold that maximizes the discrimination contrast
    between two quantum states and calculates the associated error rates.
    
    Parameters
    ----------
    ground_data : array_like
        Measurement data for ground state
    excited_data : array_like
        Measurement data for excited state
    n_bins : int, optional
        Number of bins for histogram analysis
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'threshold': Optimal discrimination threshold
        - 'fidelity': Maximum discrimination fidelity (0-1)
        - 'error_ground': Probability of misidentifying ground as excited
        - 'error_excited': Probability of misidentifying excited as ground
        
    Notes
    -----
    The fidelity is calculated as the maximum contrast:
    F = max |∫_{-∞}^{threshold} (P_g - P_e) dV| / (0.5 * (∫P_g + ∫P_e))
    
    where P_g and P_e are the probability distributions of ground and excited states.
    """
    ground_data, excited_data = np.asarray(ground_data), np.asarray(excited_data)
    
    # Determine histogram range covering both datasets
    data_min = min(np.min(ground_data), np.min(excited_data))
    data_max = max(np.max(ground_data), np.max(excited_data))
    hist_range = (data_min, data_max)
    
    # Create normalized histograms
    hist_ground, bin_edges = np.histogram(ground_data, bins=n_bins, range=hist_range, density=True)
    hist_excited, _ = np.histogram(excited_data, bins=n_bins, range=hist_range, density=True)
    
    # Calculate cumulative distributions
    cumsum_ground = np.cumsum(hist_ground)
    cumsum_excited = np.cumsum(hist_excited)
    
    # Find optimal threshold by maximizing contrast
    total_norm = 0.5 * (hist_ground.sum() + hist_excited.sum())
    contrast = np.abs((cumsum_ground - cumsum_excited) / total_norm)
    
    optimal_index = np.argmax(contrast)
    optimal_threshold = bin_edges[optimal_index]
    max_fidelity = contrast[optimal_index]
    
    # Calculate error rates at optimal threshold
    error_excited = cumsum_excited[optimal_index] / hist_excited.sum()  # e→g error
    error_ground = 1 - (cumsum_ground[optimal_index] / hist_ground.sum())  # g→e error
    
    return {
        'threshold': optimal_threshold,
        'fidelity': max_fidelity,
        'error_ground': error_ground,
        'error_excited': error_excited
    }


def analyze_single_shot_histograms(data: IQData, plot: bool = True, span: Optional[float] = None,
                                 ax: Optional[List[plt.Axes]] = None, verbose: bool = False,
                                 qubit: int = 0, seq_mode: str = "pi",
                                 rotate: bool = True) -> Tuple[Dict[str, Any], Optional[plt.Figure]]:
    """
    Analyze and visualize IQ data histograms for quantum state discrimination.

    This function:
    1. Optionally rotates IQ data to maximize separation along I-axis
    2. Computes optimal threshold for state discrimination
    3. Calculates readout fidelity
    4. Optionally plots the results

    Parameters
    ----------
    data : dict
        Dictionary containing 'Ig', 'Qg', 'Ie', 'Qe' arrays and optionally 'If', 'Qf'
    plot : bool, optional
        Whether to plot the results, default is True
    span : float, optional
        Histogram limit is the mean +/- span, default is auto-determined
    ax : list of matplotlib.axes, optional
        Axes for plotting, default is to create new axes
    verbose : bool, optional
        Whether to print detailed information, default is False
    qubit : int, optional
        Qubit index for plot title, default is 0
    seq_mode : str, optional
        Sequence mode for plot title, default is "pi"
    rotate : bool, optional
        Whether to derive a rotation angle from the g/e separation and apply
        it, default True. Pass False when the readout phase is already
        calibrated (single-shot optimisation has been run and
        `device.readout.phase` is applied to the readout pulse), so I is
        already the discriminating axis -- rotating again would
        double-correct, and the derived angle is unreliable when the two
        states are not well separated.

    Returns
    -------
    tuple
        Parameters dictionary and figure handle (if plot=True)
    """
    # Extract IQ data for ground and excited states
    Ig = data["Ig"]
    Qg = data["Qg"]
    Ie = data["Ie"]
    Qe = data["Qe"]
    
    # Check if f state data is available
    if "If" in data.keys():
        plot_f = True
        If = data["If"]
        Qf = data["Qf"]
    else:
        plot_f = False

    # Number of bins for histograms
    numbins = 100

    # Calculate median positions of each state
    xg, yg = np.median(Ig), np.median(Qg)
    xe, ye = np.median(Ie), np.median(Qe)
    if plot_f:
        xf, yf = np.median(If), np.median(Qf)

    # Print unrotated data statistics if verbose
    if verbose:
        print("Unrotated:")
        print(
            f"Ig {xg:0.3f} +/- {np.std(Ig):0.3f} \t Qg {yg:0.3f} +/- {np.std(Qg):0.3f} \t Amp g {np.abs(xg+1j*yg):0.3f}"
        )
        print(
            f"Ie {xe:0.3f} +/- {np.std(Ie):0.3f} \t Qe {ye:0.3f} +/- {np.std(Qe):0.3f} \t Amp e {np.abs(xe+1j*ye):0.3f}"
        )
        if plot_f:
            print(
                f"If {xf:0.3f} +/- {np.std(If)} \t Qf {yf:0.3f} +/- {np.std(Qf):0.3f} \t Amp f {np.abs(xf+1j*yf):0.3f}"
            )

    # Compute the rotation angle to maximize separation along I-axis
    if not rotate:
        # The readout phase is already calibrated, so take I as acquired.
        theta = 0.0
    else:
        theta = -np.arctan2((ye - yg), (xe - xg))
        if plot_f:
            # Use g-f separation for rotation if f state is available
            theta = -np.arctan2((yf - yg), (xf - xg))

    # Rotate the IQ data
    Ig_new = Ig * np.cos(theta) - Qg * np.sin(theta)
    Qg_new = Ig * np.sin(theta) + Qg * np.cos(theta)

    Ie_new = Ie * np.cos(theta) - Qe * np.sin(theta)
    Qe_new = Ie * np.sin(theta) + Qe * np.cos(theta)

    if plot_f:
        If_new = If * np.cos(theta) - Qf * np.sin(theta)
        Qf_new = If * np.sin(theta) + Qf * np.cos(theta)

    # Calculate post-rotation median positions
    xg_new, yg_new = np.median(Ig_new), np.median(Qg_new)
    xe_new, ye_new = np.median(Ie_new), np.median(Qe_new)
    if plot_f:
        xf, yf = np.median(If_new), np.median(Qf_new)
    
    # Print rotated data statistics if verbose
    if verbose:
        print("Rotated:")
        print(
            f"Ig {xg_new:.3f} +/- {np.std(Ig):.3f} \t Qg {yg_new:.3f} +/- {np.std(Qg):.3f} \t Amp g {np.abs(xg_new+1j*yg_new):.3f}"
        )
        print(
            f"Ie {xe_new:.3f} +/- {np.std(Ie):.3f} \t Qe {ye_new:.3f} +/- {np.std(Qe):.3f} \t Amp e {np.abs(xe_new+1j*ye_new):.3f}"
        )
        if plot_f:
            print(
                f"If {xf:.3f} +/- {np.std(If)} \t Qf {yf:.3f} +/- {np.std(Qf):.3f} \t Amp f {np.abs(xf+1j*yf):.3f}"
            )

    # Determine histogram span if not provided
    all_i = [Ig_new, Ie_new]
    if plot_f:
        all_i.append(If_new)
    combined_i = np.concatenate(all_i)

    if span is None:
        # Cover the full range of the data and centre the window on that range.
        # Centring on the state medians instead clips whichever tail sits
        # further from them, and since np.histogram(range=xlims) discards
        # out-of-range shots those were dropped from the fidelity too.
        lo, hi = np.min(combined_i), np.max(combined_i)
        span = (hi - lo) / 2
        x_center = (hi + lo) / 2
    else:
        # An explicit span stays centred between the two states, as before.
        x_center = (xg_new + xe_new) / 2

    # Set histogram limits
    xlims = [x_center - span, x_center + span]
    ylims = [yg_new - span, yg_new + span]

    # Lists to store fidelities and thresholds
    fids = []
    thresholds = []

    # Create histograms for ground and excited states
    ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
    ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
    if plot_f:
        nf, binsf = np.histogram(If_new, bins=numbins, range=xlims, density=True)

    # Compute optimal threshold and fidelity for g-e discrimination
    contrast = np.abs(
        ((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum()))
    )
    tind = contrast.argmax()
    thresholds.append(binsg[tind])
    fids.append(contrast[tind])
    
    # Calculate error rates for each state
    err_e = np.cumsum(ne)[tind]/ne.sum()  # Excited state misidentified as ground
    err_g = 1-np.cumsum(ng)[tind]/ng.sum()  # Ground state misidentified as excited

    # Compute thresholds and fidelities for f state if available
    if plot_f:
        # g-f discrimination
        contrast = np.abs(
            ((np.cumsum(ng) - np.cumsum(nf)) / (0.5 * ng.sum() + 0.5 * nf.sum()))
        )
        tind = contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])

        # e-f discrimination
        contrast = np.abs(
            ((np.cumsum(ne) - np.cumsum(nf)) / (0.5 * ne.sum() + 0.5 * nf.sum()))
        )
        tind = contrast.argmax()
        thresholds.append(binsg[tind])
        fids.append(contrast[tind])
    
    # Plot settings
    m = DEFAULT_PLOT_CONFIG['marker_size']
    a = DEFAULT_PLOT_CONFIG['alpha']
    
    # Create plots if requested
    if plot:
        if ax is None:
            # Create new figure with 2x2 subplots
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
            ax = [axs[0,1], axs[1,0]]
            
            # Plot unrotated IQ data
            axs[0, 0].plot(Ig, Qg, ".", label="g", color=COLORS['ground'], alpha=a, markersize=m, rasterized=True)
            axs[0, 0].plot(Ie, Qe, ".", label="e", color=COLORS['excited'], alpha=a, markersize=m, rasterized=True)
            if plot_f:
                axs[0, 0].plot(If, Qf, ".", label="f", color=COLORS['f_state'], alpha=a, markersize=m, rasterized=True)
            
            # Mark median positions
            axs[0, 0].plot(xg, yg, color="k", marker="o")
            axs[0, 0].plot(xe, ye, color="k", marker="o")
            if plot_f:
                axs[0, 0].plot(xf, yf, color="k", marker="o")

            axs[0,0].set_xlabel('I (ADC levels)')
            axs[0, 0].set_ylabel("Q (ADC levels)")
            axs[0, 0].legend(loc="upper right")
            axs[0, 0].set_title("Unrotated")
            axs[0, 0].axis("equal")
            set_fig=True

            # Plot log histogram
            bin_cent_g = (binsg[1:] + binsg[:-1]) / 2
            axs[1,1].semilogy(bin_cent_g, ng, color=COLORS['ground'])
            bin_cent_e = (binse[1:] + binse[:-1]) / 2
            axs[1,1].semilogy(bin_cent_e, ne, color=COLORS['excited'])
            if plot_f:
                bin_cent_f = (binsf[1:] + binsf[:-1]) / 2
                axs[1,1].semilogy(bin_cent_f, nf, color=COLORS['f_state'])

            # Overlay fitted curves, matching the linear-scale panel
            if seq_mode == 'fwm':
                if all(k in data for k in ("mag1_g", "mu1_g", "sig1_g", "mag2_g", "mu2_g", "sig2_g")):
                    axs[1,1].semilogy(bin_cent_g, double_gaussian_xnorm(bin_cent_g, data["mag1_g"], data["mu1_g"], data["sig1_g"], data["mag2_g"], data["mu2_g"], data["sig2_g"]), 'k', linewidth=1)
                if all(k in data for k in ("mag1_e", "mu1_e", "sig1_e", "mag2_e", "mu2_e", "sig2_e")):
                    axs[1,1].semilogy(bin_cent_e, double_gaussian_xnorm(bin_cent_e, data["mag1_e"], data["mu1_e"], data["sig1_e"], data["mag2_e"], data["mu2_e"], data["sig2_e"]), 'k', linewidth=1)
            elif plot_f and data.get('gef_fit'):
                # Three-state joint fit: one total plus g/e/f components per shot
                plot_three_state_fit(axs[1,1].semilogy, data, bin_cent_g, bin_cent_e, bin_cent_f)
            elif 'paramsg' in data and 'vg' in data and 've' in data and 'sigma' in data and 'tm' in data:
                if 'paramsg2' in data and np.ndim(data['paramsg2']) > 0:
                    axs[1,1].semilogy(bin_cent_g, combined_state_model(bin_cent_g, *data['paramsg2']), 'k', linewidth=1)
                    # Overlay the two components of the g-shot fit in lighter colors
                    # (use the g-shot's own tm so they match the total)
                    for comp, ccol in combined_fit_components(bin_cent_g, data['paramsg2'][0], data, tm=data['paramsg2'][4]):
                        axs[1,1].semilogy(bin_cent_g, comp, color=ccol, linewidth=1, linestyle='--')
                else:
                    axs[1,1].semilogy(bin_cent_g, gaussian(bin_cent_g, 1, *data['paramsg']), 'k', linewidth=1)
                if 'mag_ground_e' in data:
                    axs[1,1].semilogy(bin_cent_e, combined_state_model(bin_cent_e, data['mag_ground_e'], data['vg'], data['ve'], data['sigma'], data['tm']), 'k', linewidth=1)
                    # Overlay the two components of the e-shot fit in lighter colors
                    for comp, ccol in combined_fit_components(bin_cent_e, data['mag_ground_e'], data):
                        axs[1,1].semilogy(bin_cent_e, comp, color=ccol, linewidth=1, linestyle='--')
                else:
                    axs[1,1].semilogy(bin_cent_e, excited_state_model(bin_cent_e, data['vg'], data['ve'], data['sigma'], data['tm']), 'k', linewidth=1)

            # Clamp the log y-axis to just below the measured data so the fit's tiny
            # tails don't stretch the axis far below the histogram values.
            pos_arrays = [ng[ng > 0], ne[ne > 0]]
            if plot_f:
                pos_arrays.append(nf[nf > 0])
            pos_vals = np.concatenate(pos_arrays)
            if pos_vals.size:
                axs[1,1].set_ylim(bottom=0.5 * pos_vals.min())

            axs[1,1].set_title('Log Histogram')
            axs[1, 1].set_xlabel("I (ADC levels)")
            axs[0, 0].set_xlabel("I (ADC levels)")

            plt.subplots_adjust(hspace=0.25, wspace=0.15)
            if qubit is not None: 
                fig.suptitle(f"Single Shot Histogram Analysis Q{qubit}")
        else:
            set_fig=False
            fig = None
        
        # Plot rotated IQ data
        ax[0].plot(Ig_new, Qg_new, ".", label="g", color=COLORS['ground'], alpha=a, markersize=m, rasterized=True)
        ax[0].plot(Ie_new, Qe_new, ".", label="e", color=COLORS['excited'], alpha=a, markersize=m, rasterized=True)
        if plot_f:
            ax[0].plot(If_new, Qf_new, ".", label="f", color=COLORS['f_state'], alpha=a, markersize=m, rasterized=True)
        
        # Add text annotation with state positions
        ax[0].text(0.95, 0.95, f'g: {xg_new:.2f}\ne: {xe_new:.2f}', 
                   transform=ax[0].transAxes, fontsize=10, 
                   verticalalignment='top', horizontalalignment='right', 
                   bbox=dict(facecolor='white', alpha=0.5))

        ax[0].set_xlabel('I (ADC levels)')
        lgnd=ax[0].legend(loc='lower right')
        ax[0].set_title("Angle: {:.2f}$^\circ$".format(theta * 180 / np.pi))
        ax[0].axis("equal")        

        # Plot histogram with fits
        ax[1].set_ylabel("Probability")
        ax[1].set_xlabel("I (ADC levels)")
        if seq_mode == 'fwm':
            pe_str = f'{data["Pe"]:.3f}' if "Pe" in data else "N/A"
            ax[1].set_title(f'P_e : {pe_str}')
            #ax[1].set_title(f"Histogram (Fidelity g-e: {100*fids[0]:.3}%)")
        elif plot_f:
            ax[1].set_title(f"Histogram (Fidelity g-f: {100*fids[1]:.3}%)")
        else:
            ax[1].set_title(f"Histogram (Fidelity g-e: {100*fids[0]:.3}%)")

        # Plot the relevant discrimination threshold (g-f when the f state is
        # measured, otherwise g-e)
        ax[1].axvline(thresholds[1] if plot_f else thresholds[0], color="0.2", linestyle="--")
        
        # Plot histograms and fits
        if data.get('gef_fit'):
            # Three-state joint fit: the g, e and f histograms are all fit, so each
            # gets its own color, fit curve and g/e/f components.
            for vh, hist, color, label in (
                (data["vhg"], data["histg"], COLORS['ground'], "g"),
                (data["vhe"], data["histe"], COLORS['excited'], "e"),
                (data["vhf"], data["histf"], COLORS['f_state'], "f"),
            ):
                ax[1].plot(vh, hist, '.-', color=color, markersize=0.5, linewidth=0.3)
                ax[1].fill_between(vh, hist, color=color, alpha=0.3, label=label)
            plot_three_state_fit(ax[1].plot, data, data["vhg"], data["vhe"], data["vhf"])

            txt = f"Threshold: {thresholds[1]:.2f}"
            txt += f" \n Width: {data['sigma']:.2f}"
            txt += f" \n $T_m/T_1$: e {data['tm_e']:.2f} / f {data['tm_f']:.2f}"
            txt += f" \n $T_m/T_{{up}}$ (g): {data['tm_g']:.4f}"
            # g vs not-g, NOT the e/f split. With chi >> kappa the |e> and
            # |f> blobs land 0.13 sigma apart on run 33 (95% overlap,
            # against 4.38 sigma for g-e), so how the not-g population
            # divides between them is not identifiable from these
            # histograms -- the fit can move weight from e to f at no cost
            # to the likelihood, which is what put negative numbers in this
            # box. The SUM is well determined, so report that.
            txt += f" \n P(not g) no-$\\pi$: {1 - data['pg_g']:.3g}"
            txt += (f" \n P(g) $\\pi_{{ge}}$: {data['pg_e']:.3g}"
                    f"   $\\pi_{{ef}}$: {data['pg_f']:.3g}")
            if data.get('fit_fidelity') is not None:
                txt += f" \n Fit fid g-f: {100*data['fit_fidelity']:.1f}%"
            ax[1].text(0.025, 0.965, txt,
                transform=ax[1].transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='none', edgecolor='black', alpha=0.5))
            ax[1].legend(loc="upper right", fontsize=8)
        elif 'vhg' in data and 'histg' in data:
            # When the f state is measured the fitted 'excited' distribution is f,
            # so draw it green (and add the raw e histogram separately below).
            excited_color = COLORS['f_state'] if plot_f else COLORS['excited']
            excited_label = "f" if plot_f else "e"
            ax[1].plot(data["vhg"], data["histg"], '.-', color=COLORS['ground'], markersize=0.5, linewidth=0.3)
            ax[1].fill_between(data["vhg"], data["histg"], color=COLORS['ground'], alpha=0.3, label="g")
            ax[1].fill_between(data["vhe"], data["histe"], color=excited_color, alpha=0.3, label=excited_label)
            ax[1].plot(data["vhe"], data["histe"], '.-', color=excited_color, markersize=0.5, linewidth=0.3)
            
            # Plot fitted curves
            
            if seq_mode == 'fwm':
                
                ax[1].plot(data["vhg"], double_gaussian_xnorm(data["vhg"], data["mag1_g"], data["mu1_g"], data["sig1_g"], data["mag2_g"], data["mu2_g"],data["sig2_g"]), 'k', linewidth=1)
                ax[1].plot(data["vhe"], double_gaussian_xnorm(data["vhe"], data["mag1_e"], data["mu1_e"], data["sig1_e"], data["mag2_e"], data["mu2_e"],data["sig2_e"]), 'k', linewidth=1)
            else:
                if 'paramsg' in data and 'vg' in data and 've' in data and 'sigma' in data and 'tm' in data:
                    if 'paramsg2' in data and np.ndim(data['paramsg2']) > 0:
                        ax[1].plot(data["vhg"], combined_state_model(data["vhg"], *data['paramsg2']), 'k', linewidth=1)
                        # Overlay the two components of the g-shot fit in lighter colors
                        # (use the g-shot's own tm so they match the total)
                        for comp, ccol in combined_fit_components(data["vhg"], data['paramsg2'][0], data, tm=data['paramsg2'][4]):
                            ax[1].plot(data["vhg"], comp, color=ccol, linewidth=1, linestyle='--')
                    else:
                        ax[1].plot(data["vhg"], gaussian(data["vhg"], 1, *data['paramsg']), 'k', linewidth=1)
                    if 'mag_ground_e' in data:
                        ax[1].plot(data["vhe"], combined_state_model(data["vhe"], data['mag_ground_e'], data['vg'], data['ve'], data['sigma'], data['tm']), 'k', linewidth=1)
                        # Overlay the two components of the e-shot fit in lighter colors
                        for comp, ccol in combined_fit_components(data["vhe"], data['mag_ground_e'], data):
                            ax[1].plot(data["vhe"], comp, color=ccol, linewidth=1, linestyle='--')
                    else:
                        ax[1].plot(data["vhe"], excited_state_model(data["vhe"], data['vg'], data['ve'], data['sigma'], data['tm']), 'k', linewidth=1)

                    # Add text annotation with fit parameters
                    sigma = data['sigma']
                    tm = data['tm']
                    txt = f"Threshold: {thresholds[0]:.2f}"
                    txt += f" \n Width: {sigma:.2f}"
                    if data.get('tm_g') is not None and abs(data['tm_g'] - tm) > 1e-6:
                        txt += f" \n $T_m/T_1$: g {data['tm_g']:.2f} / e {tm:.2f}"
                    else:
                        txt += f" \n $T_m/T_1$: {tm:.2f}"
                    # Residual population ratios from the fitted weights:
                    #   no-pi (g-shot): excited/ground;  with-pi (e-shot): ground/excited
                    if 'paramsg2' in data and np.ndim(data['paramsg2']) > 0:
                        mag_gg = data['paramsg2'][0]
                        if 0 < mag_gg < 1:
                            txt += f" \n no-$\\pi$ e/g: {(1 - mag_gg) / mag_gg:.3g}"
                    if 'mag_ground_e' in data and 0 <= data['mag_ground_e'] < 1:
                        txt += f" \n $\\pi$ g/e: {data['mag_ground_e'] / (1 - data['mag_ground_e']):.3g}"
                    if data.get('fit_fidelity') is not None:
                        txt += f" \n Fit fid: {100*data['fit_fidelity']:.1f}%"
                    ax[1].text(0.025, 0.965, txt, 
                        transform=ax[1].transAxes, fontsize=10, 
                        verticalalignment='top', horizontalalignment='left', 
                        bbox=dict(facecolor='none', edgecolor='black', alpha=0.5))
        
        # When the f state is measured, the fitted 'excited' curve above is f (green);
        # also show the raw e-state histogram (red, density-normalized) so both show.
        # The three-state fit already drew the e histogram with its own fit.
        if plot_f and not data.get('gef_fit'):
            ax[1].hist(
                Ie_new, bins=numbins, range=xlims, color=COLORS['excited'], label="e", alpha=0.5,
                density=True,
            )
            ax[1].legend(loc="upper right", fontsize=8)

        if set_fig:
            fig.tight_layout()
        
        plt.show()
    else:
        fig = None
        # Create histograms without plotting
        ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims)
        if plot_f:
            nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

    # Return parameters and figure handle
    params = {
        'fids': fids, 
        'thresholds': thresholds, 
        'angle': theta * 180 / np.pi, 
        'ig': xg_new, 
        'ie': xe_new, 
        'err_e': err_e, 
        'err_g': err_g
    }
    return params, fig


def fit_single_shot(data: IQData, plot: bool = True, rot: bool = True,
                    float_heights: bool = False, float_tm: bool = False) -> Tuple[IQData, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Comprehensive analysis of single-shot readout data.

    This function:
    1. Optionally rotates the IQ data
    2. Jointly fits the ground and excited state I histograms -- each modeled as a
       residual-opposite-state gaussian plus a same-state gaussian with a T1 decay
       tail (combined_state_model), sharing width/vg/ve across both
    3. Fits Gaussians to the Q data (for a rotation-angle cross-check)
    4. Returns processed data and fit parameters

    Constraint toggles:
    - float_tm (default False): if False, one Tm/T1 (tm) is shared by both shots;
      if True, tm_g and tm_e are fit independently.
    - float_heights (default False): if False, symmetric state prep is enforced,
      tying the e-shot residual ground weight to the g-shot residual excited weight
      (mag_ground_e = 1 - mag_gg); if True, mag_ground_e floats independently.
    
    Parameters
    ----------
    data : dict
        Dictionary containing 'Ig', 'Qg', 'Ie', 'Qe' arrays
    plot : bool, optional
        Whether to plot the results, default is True
    rot : bool, optional
        Whether to rotate the data, default is True
        
    Returns
    -------
    tuple
        Processed data dictionary, parameters dictionary, ground state parameters, excited state parameters
    """
    # Make a deep copy to avoid modifying the original data
    d = deepcopy(data)
    
    # Rotate the data if requested
    if rot:
        # Get rotation angle from histogram analysis
        params, _ = analyze_single_shot_histograms(d, plot=False, verbose=False)
        theta = np.pi * params['angle'] / 180
        data_rotated = rotate_iq_data(d, theta)
    else:
        data_rotated = d
        theta = 0
        
    # Quick independent Gaussian fits, used only to seed the joint fit below
    paramsg0, vhg, histg = fit_gaussian_distribution(data_rotated["Ig"], n_bins=100, plot_result=False)
    paramsqg, vhqg, histqg = fit_gaussian_distribution(data_rotated["Qg"], n_bins=100, plot_result=False)

    # Fit Gaussian to excited state Q data
    paramsqe, vhqe, histqe = fit_gaussian_distribution(data_rotated["Qe"], n_bins=100, plot_result=False)

    # Extract parameters from fits
    vqg = paramsqg[0]  # Q center for ground state
    vqe = paramsqe[0]  # Q center for excited state
    vg0, sigma0 = paramsg0  # Initial guesses for ground state center and shared width

    vhe, histe = create_histogram(data_rotated["Ie"], n_bins=100)
    ve0 = np.mean(vhe * histe) / np.mean(histe)  # Initial guess for excited state center

    # Joint fit of the ground and excited state I histograms. Both are modeled with
    # combined_state_model: a gaussian for the population that stayed in the state
    # it was prepared in, plus a residual-opposite-state gaussian with an erf-based
    # T1 decay tail. Both histograms always share vg/ve/sigma.
    #
    # Two constraints are toggleable:
    #  - float_tm=False (default) shares one tm (Tm/T1) across both shots; True fits
    #    a separate tm_g and tm_e.
    #  - float_heights=False (default) enforces symmetric state prep, tying the e-shot
    #    residual ground weight to the g-shot residual excited weight
    #    (mag_ground_e = 1 - mag_gg); True lets mag_ground_e float independently.
    n_g = len(vhg)
    x_joint = np.concatenate([vhg, vhe])
    y_joint = np.concatenate([histg, histe])

    def joint_state_model(x, *p):
        mag_gg, vg, ve, sigma = p[0], p[1], p[2], p[3]
        i = 4
        if float_tm:
            tm_g, tm_e = p[i], p[i + 1]; i += 2
        else:
            tm_g = tm_e = p[i]; i += 1
        mag_ground_e = p[i] if float_heights else 1 - mag_gg
        fg = combined_state_model(x[:n_g], mag_gg, vg, ve, sigma, tm_g)
        fe = combined_state_model(x[n_g:], mag_ground_e, vg, ve, sigma, tm_e)
        return np.concatenate([fg, fe])

    # Build p0 / bounds to match the toggled parameters
    p0 = [0.99, vg0, ve0, sigma0]
    lo = [0, -np.inf, -np.inf, 1e-9]
    hi = [1, np.inf, np.inf, np.inf]
    if float_tm:
        p0 += [0.2, 0.2]; lo += [0, 0]; hi += [np.inf, np.inf]
    else:
        p0 += [0.2]; lo += [0]; hi += [np.inf]
    if float_heights:
        p0 += [0.02]; lo += [0]; hi += [1]

    try:
        popt, params_err = curve_fit(joint_state_model, x_joint, y_joint, p0=p0, bounds=(lo, hi), max_nfev=10000)
    except RuntimeError:
        # If fitting fails, use initial guess
        popt = p0

    mag_gg, vg, ve, sigma = popt[0], popt[1], popt[2], popt[3]
    i = 4
    if float_tm:
        tm_g, tm_e = popt[i], popt[i + 1]; i += 2
    else:
        tm_g = tm_e = popt[i]; i += 1
    mag_ground_e = popt[i] if float_heights else 1 - mag_gg
    tm = tm_e  # primary Tm/T1 (excited-state decay) used for display/storage

    # Reassemble in the shapes expected by the plotting code below
    paramsg = [vg, sigma]                          # gaussian(x, 1, *paramsg)
    paramsg2 = [mag_gg, vg, ve, sigma, tm_g]       # combined_state_model(x, *paramsg2)
    paramse2 = [mag_ground_e, vg, ve, sigma, tm_e] # combined_state_model(x, *paramse2)

    # Calculate correction angle from Gaussian fits
    theta_corr = -np.arctan2((vqe - vqg), (ve - vg))

    # Rotate data for analysis (using zero angle here, actual rotation done earlier)
    g_rot = rotate_coordinates(data_rotated['Ig'], data_rotated['Qg'], 0)
    e_rot = rotate_coordinates(data_rotated['Ie'], data_rotated['Qe'], 0)

    # Store rotated data and histograms in the data dictionary
    data_rotated["Ie_rot"] = e_rot[0]
    data_rotated["Qe_rot"] = e_rot[1]
    data_rotated["Ig_rot"] = g_rot[0]
    data_rotated["Qg_rot"] = g_rot[1]

    data_rotated["vhg"] = vhg
    data_rotated["histg"] = histg
    data_rotated["vhe"] = vhe
    data_rotated["histe"] = histe

    data_rotated["vqg"] = vhqg
    data_rotated["histqg"] = histqg
    data_rotated["vqe"] = vhqe
    data_rotated["histqe"] = histqe
    
    # Store fit parameters in the data dictionary
    data_rotated["paramsg"] = paramsg
    data_rotated["paramsg2"] = paramsg2
    data_rotated["vg"] = vg
    data_rotated["ve"] = ve
    data_rotated["sigma"] = sigma
    data_rotated["tm"] = tm
    data_rotated["tm_g"] = tm_g
    data_rotated["tm_e"] = tm_e
    data_rotated["mag_ground_e"] = mag_ground_e

    # Plot results if requested
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(vhg, histg, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhe, histe, "k.-", markersize=0.5, linewidth=0.3)
        if np.ndim(paramsg2) > 0:
            plt.plot(vhg, combined_state_model(vhg, *paramsg2), label="g", linewidth=1)
        else:
            plt.plot(vhg, gaussian(vhg, 1, *paramsg), label="g", linewidth=1)
        plt.plot(vhe, combined_state_model(vhe, mag_ground_e, vg, ve, sigma, tm), label="e", linewidth=1)
        plt.ylabel("Probability")
        plt.title("Single Shot Histograms")
        plt.xlabel("Voltage")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Collect all parameters in a dictionary
    p = {
        'theta': theta,           # Initial rotation angle
        'vg': vg,                 # Ground state I center
        've': ve,                 # Excited state I center
        'sigma': sigma,           # Width of distributions
        'tm': tm,                 # T1 decay parameter (excited-state / tm_e)
        'tm_g': tm_g,             # Tm/T1 used for the g-shot (== tm unless float_tm)
        'tm_e': tm_e,             # Tm/T1 used for the e-shot (== tm)
        'mag_ground_e': mag_ground_e,  # Residual ground-state population in excited-state shots
        'mag_excited_g': 1 - mag_gg,  # Population that leaked to excited in ground-state shots
        'vqg': vqg,               # Ground state Q center
        'vqe': vqe,               # Excited state Q center
        'theta_corr': theta_corr,  # Correction angle from Gaussian fits,
        'g_mean': np.mean(data_rotated['Ig']),  # Mean of ground state I data
        'e_mean': np.mean(data_rotated['Ie']),  # Mean of excited state I data
        'g_peak': vhg[np.argmax(histg)],  # I value at ground state histogram peak
        'e_peak': vhe[np.argmax(histe)],  # I value at excited state histogram peak
    }
    
    return data_rotated, p, paramsg, paramse2


def fit_single_shot_gef(data: IQData, plot: bool = False, rot: bool = True,
                        float_pf: bool = False,
                        float_pe_in_f: bool = False) -> Tuple[IQData, Dict[str, Any]]:
    """
    Joint three-state analysis of single-shot readout data with |g>, |e> and |f> shots.

    The |f>-state counterpart of fit_single_shot. Where that function fits two
    histograms with a two-level model, this fits all three histograms
    simultaneously with three_state_model:

    - The rate belongs to the state, not the shot: tm_e governs the |e>
      component of all three shots (so the no-pi shot's thermal |e> residual
      decays at the same rate as the prepared |e>), tm_f governs every |f>
      component, and tm_g governs every |g> component -- |g> cannot decay, so
      its rate is readout-induced heating up to |e>.
    - vg/ve/vf and sigma are shared across all three histograms.
    - Nothing ties the residual populations of one shot to those of another.

    By default each shot is a two-component mixture of |g> and the state it was
    prepared in, which is 10 free parameters: p_g for each of the three shots,
    plus vg, ve, vf, sigma, tm_g, tm_e, tm_f. The two toggles below each add
    back one cross-state population per shot they apply to.

    Constraint toggles:
    - float_pf (default False): only the pi_ge+pi_ef shot can populate |f>, so
      by default p_f is pinned to 0 in the no-pi and pi_ge shots -- neither has
      a pi_ef in front of it, and thermal |f> population is negligible. Set
      True to let them carry an |f> population too (+2 parameters), which is
      worth doing when an imperfect reset can leave |f> populated at the start
      of the next shot; otherwise the extra freedom mostly absorbs tail-shape
      error and can bias tm_f.
    - float_pe_in_f (default False): p_e is pinned to 0 in the pi_ge+pi_ef shot.
      An imperfect pi_ef does leave |e> population there, but with chi >> kappa
      the e and f blobs sit on top of each other, so that population is not
      separable from p_f and fitting it just splits one blob two ways
      arbitrarily. Pinned, the shot's p_f reads as "P(not g)". Set True when the
      e and f blobs are actually resolved (chi < kappa), where the |e> residual
      becomes identifiable and measures the pi_ef infidelity (+1 parameter).

    Parameters
    ----------
    data : dict
        Dictionary containing 'Ig', 'Qg', 'Ie', 'Qe', 'If', 'Qf' arrays
    plot : bool, optional
        Whether to plot the three histograms and their fits, default False
    rot : bool, optional
        Whether to rotate the data onto the discriminating axis, default True
    float_pf : bool, optional
        Whether the no-pi and pi_ge shots carry a free |f> population,
        default False
    float_pe_in_f : bool, optional
        Whether the pi_ge+pi_ef shot carries a free |e> population,
        default False

    Returns
    -------
    tuple
        (processed data dictionary, parameters dictionary)
    """
    d = deepcopy(data)

    if rot:
        # analyze_single_shot_histograms derives the angle from the g-f
        # separation when f shots are present, which is the axis we want.
        params, _ = analyze_single_shot_histograms(d, plot=False, verbose=False)
        theta = np.pi * params['angle'] / 180
        data_rotated = rotate_iq_data(d, theta)
    else:
        data_rotated = d
        theta = 0

    # Quick independent Gaussian fits, used only to seed the joint fit below
    paramsg0, vhg, histg = fit_gaussian_distribution(data_rotated["Ig"], n_bins=100, plot_result=False)
    paramsqg, vhqg, histqg = fit_gaussian_distribution(data_rotated["Qg"], n_bins=100, plot_result=False)
    paramsqe, vhqe, histqe = fit_gaussian_distribution(data_rotated["Qe"], n_bins=100, plot_result=False)

    vqg = paramsqg[0]  # Q center for ground state
    vqe = paramsqe[0]  # Q center for excited state
    vg0, sigma0 = paramsg0  # Initial guesses for ground state center and shared width

    vhe, histe = create_histogram(data_rotated["Ie"], n_bins=100)
    vhf, histf = create_histogram(data_rotated["If"], n_bins=100)
    ve0 = np.mean(vhe * histe) / np.mean(histe)  # Initial guess for excited state center
    vf0 = np.mean(vhf * histf) / np.mean(histf)  # Initial guess for f state center

    n_g, n_e = len(vhg), len(vhe)
    x_joint = np.concatenate([vhg, vhe, vhf])
    y_joint = np.concatenate([histg, histe, histf])

    # Stick-breaking, so all three populations are non-negative by
    # construction. The fitted pair per shot is (p_g, r_e) with
    #
    #     p_e = (1 - p_g) * r_e        p_f = (1 - p_g) * (1 - r_e)
    #
    # Bounding p_g and p_e to [0, 1] independently -- which is what this did
    # before -- leaves their SUM free, and p_f = 1 - p_g - p_e then goes
    # negative whenever the fit pushes p_g + p_e past 1. That is what put
    # negative "populations" in the |f> row of the readout box: not a
    # measurement, just an unconstrained derived parameter.
    #
    # r_e is therefore the split of the non-ground weight between |e> and |f>,
    # and pinning it is how the toggles drop a cross-state population: r_e = 1
    # puts all of it in |e> (p_f = 0, for the two shots with no pi_ef in front
    # of them) and r_e = 0 puts all of it in |f> (p_e = 0, for the pi_ge+pi_ef
    # shot). A shot whose r_e is pinned drops it and keeps only p_g.
    #                   fit r_e?         value when pinned
    shot_r_e = ((float_pf,      1.0),    # no-pi
                (float_pf,      1.0),    # pi_ge
                (float_pe_in_f, 0.0))    # pi_ge + pi_ef

    # vg, ve, vf, sigma, tm_g, tm_e, tm_f -- shared by all three shots and
    # always the tail of the fit vector, after the per-shot populations.
    N_LEVELS = 7

    def unpack(p):
        """(p_g, p_e) per shot and the shared levels, from the fit vector."""
        i = 0
        pops = []
        for fit_r_e, r_e_pinned in shot_r_e:
            p_g = p[i]; i += 1
            if fit_r_e:
                r_e = p[i]; i += 1
            else:
                r_e = r_e_pinned
            pops.append((p_g, (1 - p_g) * r_e))
        return pops, p[i:i + N_LEVELS]

    def joint_gef_model(x, *p):
        pops, levels = unpack(p)
        fg = three_state_model(x[:n_g], *pops[0], *levels)
        fe = three_state_model(x[n_g:n_g + n_e], *pops[1], *levels)
        ff = three_state_model(x[n_g + n_e:], *pops[2], *levels)
        return np.concatenate([fg, fe, ff])

    # Seeds converted from direct populations: r_e = p_e / (1 - p_g).
    #                                        vg   ve   vf  sigma   tm_g tm_e tm_f
    levels0 = [vg0, ve0, vf0, sigma0, 0.02, 0.2, 0.2]
    p0 = [0.97] + ([0.02 / 0.03] if float_pf else []) \
        + [0.03] + ([0.95 / 0.97] if float_pf else []) \
        + [0.03] + ([0.05 / 0.97] if float_pe_in_f else []) \
        + levels0
    n_pops = len(p0) - N_LEVELS
    lo = [0] * n_pops + [-np.inf, -np.inf, -np.inf, 1e-9, 0, 0, 0]
    hi = [1] * n_pops + [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    try:
        popt, _ = curve_fit(joint_gef_model, x_joint, y_joint, p0=p0, bounds=(lo, hi), max_nfev=20000)
    except RuntimeError:
        # If fitting fails, use initial guess
        popt = p0

    pops, levels = unpack(popt)
    (pg_g, pe_g), (pg_e, pe_e), (pg_f, pe_f) = pops
    vg, ve, vf, sigma, tm_g, tm_e, tm_f = levels

    g_rot = rotate_coordinates(data_rotated["Ig"], data_rotated["Qg"], 0)
    e_rot = rotate_coordinates(data_rotated["Ie"], data_rotated["Qe"], 0)
    f_rot = rotate_coordinates(data_rotated["If"], data_rotated["Qf"], 0)

    data_rotated["Ig_rot"], data_rotated["Qg_rot"] = g_rot[0], g_rot[1]
    data_rotated["Ie_rot"], data_rotated["Qe_rot"] = e_rot[0], e_rot[1]
    data_rotated["If_rot"], data_rotated["Qf_rot"] = f_rot[0], f_rot[1]

    data_rotated["vhg"], data_rotated["histg"] = vhg, histg
    data_rotated["vhe"], data_rotated["histe"] = vhe, histe
    data_rotated["vhf"], data_rotated["histf"] = vhf, histf

    data_rotated["vqg"], data_rotated["histqg"] = vhqg, histqg
    data_rotated["vqe"], data_rotated["histqe"] = vhqe, histqe

    p = {
        "gef_fit": True,          # Marks the three-state fit for the plotting code
        "theta": theta,           # Initial rotation angle
        "vg": vg,                 # Ground state I center
        "ve": ve,                 # Excited state I center
        "vf": vf,                 # f state I center
        "sigma": sigma,           # Width of all three blobs
        # Rates, each shared by all three shots. Note tm_g is NOT the same
        # quantity as fit_single_shot's tm_g: that one is the g-SHOT's decay
        # rate, this is |g>'s own upward g -> e rate during the readout.
        "tm_g": tm_g,             # Tm/T_up of |g> (readout-induced heating)
        "tm_e": tm_e,             # Tm/T1 of |e>
        "tm_f": tm_f,             # Tm/T1 of |f>
        "tm": tm_e,               # Primary Tm/T1 (the |e> decay) for display/storage
        "pg_g": pg_g, "pe_g": pe_g, "pf_g": 1 - pg_g - pe_g,  # no-pi shot populations
        "pg_e": pg_e, "pe_e": pe_e, "pf_e": 1 - pg_e - pe_e,  # pi_ge shot populations
        "pg_f": pg_f, "pe_f": pe_f, "pf_f": 1 - pg_f - pe_f,  # pi_ge+pi_ef shot populations
        "vqg": vqg,               # Ground state Q center
        "vqe": vqe,               # Excited state Q center
        "theta_corr": -np.arctan2((vqe - vqg), (ve - vg)),
        "g_mean": np.mean(data_rotated["Ig"]),  # Mean of ground state I data
        "e_mean": np.mean(data_rotated["Ie"]),  # Mean of excited state I data
        "f_mean": np.mean(data_rotated["If"]),  # Mean of f state I data
        "g_peak": vhg[np.argmax(histg)],  # I value at ground state histogram peak
        "e_peak": vhe[np.argmax(histe)],  # I value at excited state histogram peak
        "f_peak": vhf[np.argmax(histf)],  # I value at f state histogram peak
    }

    if plot:
        plt.figure(figsize=(8, 5))
        for vh, hist, pgs, pes, label, color in (
            (vhg, histg, pg_g, pe_g, "g", COLORS["ground"]),
            (vhe, histe, pg_e, pe_e, "e", COLORS["excited"]),
            (vhf, histf, pg_f, pe_f, "f", COLORS["f_state"]),
        ):
            plt.plot(vh, hist, ".-", color=color, markersize=0.5, linewidth=0.3)
            plt.plot(vh, three_state_model(vh, pgs, pes, vg, ve, vf, sigma,
                                           tm_g, tm_e, tm_f),
                     color=color, label=label, linewidth=1)
        plt.ylabel("Probability")
        plt.xlabel("Voltage")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return data_rotated, p


def plot_reset(experiment_data: List[Any], filename: str,
               output_dir: Optional[str] = None) -> None:
    """
    Create comprehensive plots for active reset analysis.
    
    This function generates detailed visualizations of reset performance
    across different threshold voltages and reset iterations.
    
    Parameters
    ----------
    experiment_data : list
        List of experiment objects containing reset data
    filename : str
        Base filename for saving plots
    output_dir : str, optional
        Where to write the PNGs. Defaults to `<filename>/../../images`, the
        flat notebook layout. The SMPD scripts use per-experiment folders
        (`metadata_and_images/#N_.../`) instead, so they pass
        `cfg_dict["metadata_path"]`.
    """
    if not experiment_data:
        print("Warning: No experiment data provided for reset analysis.")
        return
    
    n_experiments = len(experiment_data)
    
    # Color palette for different reset iterations
    n_rounds = len(experiment_data[0].data["Igr"])
    reset_colors = sns.color_palette("ch:s=-.2,r=.6", n_colors=n_rounds)
    # Once the reset converges the later rounds sit on top of each other --
    # on run 33 #200 rounds 2 and 3 differ by a total-variation distance of
    # 0.04 -- and a sequential palette alone makes adjacent hues too close to
    # tell apart, so the last round reads as missing. Vary the dash pattern
    # too, so overlapping rounds stay countable.
    reset_styles = ["-", "--", ":", "-."]
    style = lambda j: reset_styles[j % len(reset_styles)]
    
    # Plot 1: Ground state reset histograms.
    # Not sharey: these are log histograms whose scale varies by orders of
    # magnitude across the threshold sweep, and a shared range squashes the
    # later reset rounds -- the ones with the least population, which are
    # exactly the ones worth seeing -- flat against the axis.
    fig_ground, ax = plt.subplots(
        int(np.ceil(n_experiments / 4)), 4, figsize=(14, 1 * n_experiments)
    )
    ax = ax.flatten()

    for i, shot in enumerate(experiment_data):
        v, hist = create_histogram(shot.data["Ig"], n_bins=50)
        ax[i].semilogy(v, hist, color=COLORS['ground'], label='Ground')
        ax[i].set_title(f"{shot.cfg.expt.threshold_v:0.2f}")
        ax[i].axvline(x=shot.cfg.expt.threshold_v, color="k", linestyle="--")
        for j in range(len(shot.data["Igr"])):
            v, hist = create_histogram(shot.data["Igr"][j], n_bins=50)
            ax[i].semilogy(v, hist, color=reset_colors[j], ls=style(j),
                           label=f"Reset {j+1}")
        if i==0:
            ax[i].legend(fontsize=8)

    fig_ground.suptitle("Ground state reset Histograms")
    fig_ground.tight_layout()
    
    
    # Plot 2: Excited state reset histograms
    fig_excited, ax = plt.subplots(
        int(np.ceil(n_experiments / 4)), 4, figsize=(14, 1 * n_experiments)
    )
    ax = ax.flatten()
    for i, shot in enumerate(experiment_data):
        v, hist = create_histogram(shot.data["Ig"], n_bins=50)
        ax[i].semilogy(v, hist, color=COLORS['ground'])
        v, hist = create_histogram(shot.data["Ie"], n_bins=50)
        ax[i].semilogy(v, hist, color=COLORS['excited'])
        ax[i].set_title(f"{shot.cfg.expt.threshold_v:0.2f}")
        ax[i].axvline(x=shot.cfg.expt.threshold_v, color="k", linestyle="--")
        for j in range(len(shot.data["Ier"])):
            v, hist = create_histogram(shot.data["Ier"][j], n_bins=50)
            ax[i].semilogy(v, hist, color=reset_colors[j], ls=style(j))

    fig_excited.suptitle("Excited state reset Histograms")
    fig_excited.tight_layout()

    # Plot 3: Reset efficiency summary
    nplots = experiment_data[0].data["Igr"].shape[0]+1
    fig, ax = plt.subplots(2, nplots, figsize=(nplots * 4, 8))
    
    b = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(experiment_data))

    for i, shot in enumerate(experiment_data):
        vg, histg = create_histogram(shot.data["Ig"], n_bins=50)
        ve, histe = create_histogram(shot.data["Ie"], n_bins=50)
        for j in range(nplots-1):

            ax[0, j].semilogy(vg, histg, color=COLORS['ground'], linewidth=1)
            ax[1, j].semilogy(vg, histg, color=COLORS['ground'], linewidth=1)

            ax[1, j].semilogy(ve, histe, color=COLORS['excited'], linewidth=1)

            v, hist = create_histogram(shot.data["Igr"][j, :], n_bins=50)
            ax[0, j].semilogy(
                v, hist, label=f"{shot.cfg.expt.threshold_v:0.1f}", color=b[i]
            )

            v, hist = create_histogram(shot.data["Ier"][j, :], n_bins=50)
            ax[1, j].semilogy(v, hist, label=shot.cfg.expt.threshold_v, color=b[i])
            
    ax[0, 0].legend(ncol=int(np.ceil(len(experiment_data) / 6)), fontsize=8)

    ax[0, 0].set_title("Ground state")
    ax[1, 0].set_title("Excited state")
    fig.suptitle("Vary number of resets")
    fig.tight_layout()

    # Save plots. All three, not just the last: under a non-interactive
    # backend the first two are otherwise drawn and thrown away.
    file_path = Path(filename)
    stem = file_path.name.rsplit(".", 1)[0]
    if output_dir is None:
        out_dir = file_path.parent.parent / "images"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_ground.savefig(out_dir / f"{stem}_reset_hist_ground.png")
    fig_excited.savefig(out_dir / f"{stem}_reset_hist_excited.png")
    fig.savefig(out_dir / f"{stem}_reset_hist.png")
    

def fit_hist(bin_centers: np.ndarray, histogram: np.ndarray, 
            data: ArrayLike) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fit a double Gaussian model to histogram data.
    
    This function attempts to fit two Gaussian peaks to histogram data,
    which is useful for detecting state leakage or mixed populations.
    
    Parameters
    ----------
    bin_centers : np.ndarray
        Histogram bin center positions
    histogram : np.ndarray
        Histogram values (probability density)
    data : array_like
        Original data used to create histogram
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray or None]
        Scaled data and fit parameters (or None if fit failed)
    """
    data = np.asarray(data)
    v_rng = np.max(bin_centers) - np.min(bin_centers)

    p0 = [
        0.5,
        np.min(bin_centers) + v_rng / 3,
        0.5,
        v_rng / 10,
        np.max(bin_centers) - v_rng / 3,
    ]
    try:
        popt, pcov = curve_fit(double_gaussian, bin_centers, histogram, p0=p0)
        vg = popt[1]
        ve = popt[4]
        dv = ve - vg
        scale_data = (data - popt[1]) / dv
        hist_fit = popt
    except:
        scale_data = data 
        hist_fit = None
    return scale_data, hist_fit
