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
                                 qubit: int = 0, seq_mode: str = "pi") -> Tuple[Dict[str, Any], Optional[plt.Figure]]:
    """
    Analyze and visualize IQ data histograms for quantum state discrimination.
    
    This function:
    1. Rotates IQ data to maximize separation along I-axis
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
    if span is None:
        span = (
            np.max(np.concatenate((Ie_new, Ig_new)))
            - np.min(np.concatenate((Ie_new, Ig_new)))
        ) / 2
    
    # Set histogram limits
    xlims = [(xg_new + xe_new) / 2 - span, (xg_new + xe_new) / 2 + span]
    ylims = [yg_new - span, yg_new + span]

    # Lists to store fidelities and thresholds
    fids = []
    thresholds = []

    # Create histograms for ground and excited states
    ng, binsg = np.histogram(Ig_new, bins=numbins, range=xlims, density=True)
    ne, binse = np.histogram(Ie_new, bins=numbins, range=xlims, density=True)
    if plot_f:
        nf, binsf = np.histogram(If_new, bins=numbins, range=xlims)

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
            bin_cent = (binsg[1:] + binsg[:-1]) / 2
            axs[1,1].semilogy(bin_cent, ng, color=COLORS['ground'])
            bin_cent = (binse[1:] + binse[:-1]) / 2
            axs[1,1].semilogy(bin_cent, ne, color=COLORS['excited'])
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
        if seq_mode == '4wm':
            pe_str = f'{data["Pe"]:.3f}' if "Pe" in data else "N/A"
            ax[1].set_title(f'P_e : {pe_str}')
            #ax[1].set_title(f"Histogram (Fidelity g-e: {100*fids[0]:.3}%)")
        else:
            ax[1].set_title(f"Histogram (Fidelity g-e: {100*fids[0]:.3}%)")
        
        # Plot threshold line
        ax[1].axvline(thresholds[0], color="0.2", linestyle="--")
        
        # Plot histograms and fits
        if 'vhg' in data and 'histg' in data:
            ax[1].plot(data["vhg"], data["histg"], '.-', color=COLORS['ground'], markersize=0.5, linewidth=0.3)
            ax[1].fill_between(data["vhg"], data["histg"], color=COLORS['ground'], alpha=0.3)
            ax[1].fill_between(data["vhe"], data["histe"], color=COLORS['excited'], alpha=0.3)
            ax[1].plot(data["vhe"], data["histe"], '.-', color=COLORS['excited'], markersize=0.5, linewidth=0.3)
            
            # Plot fitted curves
            
            if seq_mode == '4wm':
                
                ax[1].plot(data["vhg"], double_gaussian_xnorm(data["vhg"], data["mag1_g"], data["mu1_g"], data["sig1_g"], data["mag2_g"], data["mu2_g"],data["sig2_g"]), 'k', linewidth=1)
                ax[1].plot(data["vhe"], double_gaussian_xnorm(data["vhe"], data["mag1_e"], data["mu1_e"], data["sig1_e"], data["mag2_e"], data["mu2_e"],data["sig2_e"]), 'k', linewidth=1)
            else:
                if 'paramsg' in data and 'vg' in data and 've' in data and 'sigma' in data and 'tm' in data:
                    ax[1].plot(data["vhg"], gaussian(data["vhg"], 1, *data['paramsg']), 'k', linewidth=1)
                    ax[1].plot(data["vhe"], excited_state_model(data["vhe"], data['vg'], data['ve'], data['sigma'], data['tm']), 'k', linewidth=1)
                    
                    # Add text annotation with fit parameters
                    sigma = data['sigma']
                    tm = data['tm']
                    txt = f"Threshold: {thresholds[0]:.2f}"
                    txt += f" \n Width: {sigma:.2f}"
                    txt += f" \n $T_m/T_1$: {tm:.2f}"
                    ax[1].text(0.025, 0.965, txt, 
                        transform=ax[1].transAxes, fontsize=10, 
                        verticalalignment='top', horizontalalignment='left', 
                        bbox=dict(facecolor='none', edgecolor='black', alpha=0.5))
        
        # Plot f state if available
        if plot_f:
            nf, binsf, pf = ax[1].hist(
                If_new, bins=numbins, range=xlims, color=COLORS['f_state'], label="f", alpha=0.5
            )
            ax[1].axvline(thresholds[1], color="0.2", linestyle="--")
            ax[1].axvline(thresholds[2], color="0.2", linestyle="--")

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


def fit_single_shot(data: IQData, plot: bool = True, rot: bool = True) -> Tuple[IQData, Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Comprehensive analysis of single-shot readout data.
    
    This function:
    1. Optionally rotates the IQ data
    2. Fits Gaussians to ground state I and Q data
    3. Fits excited state data including T1 decay effects
    4. Returns processed data and fit parameters
    
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
        
    # Fit Gaussians to the ground state I and Q data
    paramsg, vhg, histg = fit_gaussian_distribution(data_rotated["Ig"], n_bins=100, plot_result=False)
    paramsqg, vhqg, histqg = fit_gaussian_distribution(data_rotated["Qg"], n_bins=100, plot_result=False)
    
    # Fit Gaussian to excited state Q data
    paramsqe, vhqe, histqe = fit_gaussian_distribution(data_rotated["Qe"], n_bins=100, plot_result=False)

    # Extract parameters from fits
    vqg = paramsqg[0]  # Q center for ground state
    vqe = paramsqe[0]  # Q center for excited state
    vg = paramsg[0]    # I center for ground state
    sigma = paramsg[1] # Width of ground state distribution

    # Fit the excited state I data, including T1 decay effects
    vhe, histe = create_histogram(data_rotated["Ie"], n_bins=100)
    
    # Initial guess for excited state center and T1 decay parameter
    p0 = [np.mean(vhe * histe) / np.mean(histe), 0.2]
    
    # Fit using the excited_state_model with fixed ground state parameters
    try:
        paramse, params_err = curve_fit(
            lambda x, ve, tm: excited_state_model(x, vg, ve, sigma, tm), vhe, histe, p0=p0
        )
    except RuntimeError:
        # If fitting fails, use initial guess
        paramse = p0
    
    # Extract excited state parameters
    ve = paramse[0]  # I center for excited state
    tm = paramse[1]  # T1 decay parameter (t_meas/T1)
    paramse2 = [vg, ve, sigma, tm]
    
    # Calculate correction angle from Gaussian fits
    theta_corr = -np.arctan2((vqe - vqg), (ve - vg))

    # Rotate data for analysis (using zero angle here, actual rotation done earlier)
    g_rot = rotate_coordinates(data_rotated['Ig'], data_rotated['Qg'], 0)
    e_rot = rotate_coordinates(data_rotated['Ie'], data_rotated['Qe'], 0)

    # Try to fit two Gaussians to ground state data (for potential leakage detection)
    pg = [0.99, vg, sigma, 0.01, ve]  # Initial parameters
    try: 
        paramsg2, params_err = curve_fit(double_gaussian, vhg, histg, p0=pg)
    except:
        paramsg2 = np.nan

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
    data_rotated["vg"] = vg
    data_rotated["ve"] = ve
    data_rotated["sigma"] = sigma
    data_rotated["tm"] = tm

    # Plot results if requested
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(vhg, histg, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhe, histe, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhg, gaussian(vhg, 1, *paramsg), label="g", linewidth=1)
        plt.plot(vhe, excited_state_model(vhe, vg, ve, sigma, tm), label="e", linewidth=1)
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
        'tm': tm,                 # T1 decay parameter
        'vqg': vqg,               # Ground state Q center
        'vqe': vqe,               # Excited state Q center
        'theta_corr': theta_corr,  # Correction angle from Gaussian fits,
        'g_mean': np.mean(data_rotated['Ig']),  # Mean of ground state I data
        'e_mean': np.mean(data_rotated['Ie']),  # Mean of excited state I data
    }
    
    return data_rotated, p, paramsg, paramse2


def plot_reset(experiment_data: List[Any], filename: str) -> None:
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
    """
    if not experiment_data:
        print("Warning: No experiment data provided for reset analysis.")
        return
    
    n_experiments = len(experiment_data)
    
    # Color palette for different reset iterations
    reset_colors = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(experiment_data[0].data["Igr"]))
    
    # Plot 1: Ground state reset histograms
    fig, ax = plt.subplots(
        int(np.ceil(n_experiments / 4)), 4, figsize=(14, 1 * n_experiments), sharey=True
    )
    ax = ax.flatten()

    for i, shot in enumerate(experiment_data):
        v, hist = create_histogram(shot.data["Ig"], n_bins=50)
        ax[i].semilogy(v, hist, color=COLORS['ground'], label='Ground')
        ax[i].set_title(f"{shot.cfg.expt.threshold_v:0.2f}")
        ax[i].axvline(x=shot.cfg.expt.threshold_v, color="k", linestyle="--")
        for j in range(len(shot.data["Igr"])):
            v, hist = create_histogram(shot.data["Igr"][j], n_bins=50)
            ax[i].semilogy(v, hist, color=reset_colors[j], label=f"Reset {j+1}")
        if i==0:
            ax[i].legend(fontsize=8)

    fig.suptitle("Ground state reset Histograms")
    fig.tight_layout()
    
    
    # Plot 2: Excited state reset histograms
    fig, ax = plt.subplots(
        int(np.ceil(n_experiments / 4)), 4, figsize=(14, 1 * n_experiments), sharey=True
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
            ax[i].semilogy(v, hist, color=reset_colors[j])

    fig.suptitle("Excited state reset Histograms")
    fig.tight_layout()

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

    # Save plots
    file_path = Path(filename)
    new_filename = file_path.name.rsplit(".", 1)[0] + "_reset_hist.png"
    fig.savefig(file_path.parent / "images" / new_filename)
    

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
