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
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from copy import deepcopy
import seaborn as sns
import datetime

# Standard colors for plotting
BLUE = "#4053d3"  # Color for ground state
RED = "#b51d14"   # Color for excited state
GREEN = "#2ca02c"  # Color for f state

# Default plot settings
DEFAULT_ALPHA = 0.25
DEFAULT_MARKER_SIZE = 0.7
DEFAULT_BINS = 100


#------------------------------------------------------------------------------
# Basic Utility Functions
#------------------------------------------------------------------------------

def rotate(x, y, theta):
    """
    Rotate points in the x-y plane by angle theta.
    
    Parameters
    ----------
    x : array_like
        x-coordinates
    y : array_like
        y-coordinates
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    tuple
        Rotated (x, y) coordinates
    """
    return (
        x * np.cos(theta) - y * np.sin(theta),
        x * np.sin(theta) + y * np.cos(theta)
    )


def full_rotate(d, theta):
    """
    Rotate all IQ data in a dictionary by angle theta.
    
    Parameters
    ----------
    d : dict
        Dictionary containing 'Ig', 'Qg', 'Ie', 'Qe' arrays
    theta : float
        Rotation angle in radians
        
    Returns
    -------
    dict
        Dictionary with rotated IQ data
    """
    d = deepcopy(d)
    d["Ig"], d["Qg"] = rotate(d["Ig"], d["Qg"], theta)
    d["Ie"], d["Qe"] = rotate(d["Ie"], d["Qe"], theta)
    
    # Rotate f state if available
    if "If" in d and "Qf" in d:
        d["If"], d["Qf"] = rotate(d["If"], d["Qf"], theta)
        
    return d


def make_hist(d, nbins=200):
    """
    Create a normalized histogram from data.
    
    Parameters
    ----------
    d : array_like
        Input data
    nbins : int, optional
        Number of histogram bins, default is 200
        
    Returns
    -------
    tuple
        Bin centers and histogram values
    """
    hist, bin_edges = np.histogram(d, bins=nbins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist


#------------------------------------------------------------------------------
# Fitting Functions
#------------------------------------------------------------------------------

def gaussian(x, mag, cen, wid):
    """
    Gaussian function for fitting.
    
    Parameters
    ----------
    x : array_like
        Input x values
    mag : float
        Magnitude (area under curve)
    cen : float
        Center position
    wid : float
        Width (standard deviation)
        
    Returns
    -------
    array_like
        Gaussian values at x
    """
    return mag / np.sqrt(2 * np.pi) / wid * np.exp(-((x - cen) ** 2) / 2 / wid**2)


def two_gaussians(x, mag1, cen1, wid, mag2, cen2):
    """
    Sum of two Gaussian functions with shared width.
    
    Parameters
    ----------
    x : array_like
        Input x values
    mag1 : float
        Magnitude of first Gaussian
    cen1 : float
        Center of first Gaussian
    wid : float
        Width (standard deviation) for both Gaussians
    mag2 : float
        Magnitude of second Gaussian
    cen2 : float
        Center of second Gaussian
        
    Returns
    -------
    array_like
        Sum of two Gaussians at x
    """
    return 1 / np.sqrt(2 * np.pi) / wid * (
        mag1 * np.exp(-((x - cen1) ** 2) / 2 / wid**2) + 
        mag2 * np.exp(-((x - cen2) ** 2) / 2 / wid**2)
    )


def distfn(v, vg, ve, sigma, tm):
    """
    Distribution function modeling T1 decay during measurement.
    
    This function models the distribution of measurement outcomes when
    T1 decay occurs during the measurement process.
    
    Parameters
    ----------
    v : array_like
        Voltage values
    vg : float
        Ground state voltage
    ve : float
        Excited state voltage
    sigma : float
        Standard deviation of measurement noise
    tm : float
        Ratio of measurement time to T1 time (tm = t_meas/T1)
        
    Returns
    -------
    array_like
        Probability distribution values
    """
    dv = ve - vg
    return np.abs(
        tm
        / 2
        / dv
        * np.exp(tm * (tm * sigma**2 / 2 / dv**2 - (v - vg) / dv))
        * (
            erf((tm * sigma**2 / dv + ve - v) / np.sqrt(2) / sigma)
            + erf((-tm * sigma**2 / dv + v - vg) / np.sqrt(2) / sigma)
        )
    )


def excited_func(x, vg, ve, sigma, tm):
    """
    Fit function for excited state including T1 decay effects.
    
    This combines a Gaussian for the excited state with the T1 decay distribution.
    
    Parameters
    ----------
    x : array_like
        Input x values
    vg : float
        Ground state voltage
    ve : float
        Excited state voltage
    sigma : float
        Standard deviation of measurement noise
    tm : float
        Ratio of measurement time to T1 time (tm = t_meas/T1)
        
    Returns
    -------
    array_like
        Excited state distribution values
    """
    # Gaussian component for remaining excited population + T1 decay distribution
    y = gaussian(x, 1, ve, sigma) * np.exp(-tm) + distfn(x, vg, ve, sigma, tm)
    return y


def fit_all(x, mag_g, vg, ve, sigma, tm):
    """
    Fit function for combined ground and excited state distributions.
    
    Parameters
    ----------
    x : array_like
        Input x values
    mag_g : float
        Relative magnitude of ground state
    vg : float
        Ground state voltage
    ve : float
        Excited state voltage
    sigma : float
        Standard deviation of measurement noise
    tm : float
        Ratio of measurement time to T1 time (tm = t_meas/T1)
        
    Returns
    -------
    array_like
        Combined distribution values
    """
    # Ground state component
    yg = gaussian(x, mag_g, vg, sigma)
    
    # Excited state component with T1 decay
    ye = gaussian(x, 1 - mag_g, ve, sigma) * np.exp(-tm) + (1 - mag_g) * distfn(
        x, vg, ve, sigma, tm
    )
    
    return ye + yg


def fit_gaussian(d, nbins=200, p0=None, plot=False):
    """
    Fit a Gaussian distribution to data.
    
    Parameters
    ----------
    d : array_like
        Input data
    nbins : int, optional
        Number of histogram bins, default is 200
    p0 : list, optional
        Initial parameter guess [center, width]
    plot : bool, optional
        Whether to plot the fit, default is False
        
    Returns
    -------
    tuple
        Fitted parameters [center, width], bin centers, and histogram values
    """
    # Create histogram
    v, hist = make_hist(d, nbins)
    
    # Set initial parameters if not provided
    if p0 is None:
        p0 = [np.mean(v * hist) / np.mean(hist), np.std(d)]
    
    # Fit Gaussian with fixed magnitude of 1
    try:
        params, _ = curve_fit(lambda x, a, b: gaussian(x, 1, a, b), v, hist, p0=p0)
    except RuntimeError:
        # If fitting fails, use initial guess
        params = p0
    
    # Plot if requested
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(v, hist, "k.", label="Data")
        plt.plot(v, gaussian(v, 1, *params), label="Fit")
        plt.xlabel("Value")
        plt.ylabel("Probability")
        plt.title("Gaussian Fit")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return params, v, hist


#------------------------------------------------------------------------------
# Single Shot Analysis Functions
#------------------------------------------------------------------------------

def hist(data, plot=True, span=None, ax=None, verbose=False, qubit=0):
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
        
    Returns
    -------
    tuple
        Parameters dictionary and figure handle (if plot=True)
    
    Notes
    -----
    The f state analysis is currently not fully implemented.
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
    m = DEFAULT_MARKER_SIZE
    a = DEFAULT_ALPHA
    
    # Create plots if requested
    if plot:
        if ax is None:
            # Create new figure with 2x2 subplots
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
            ax = [axs[0,1], axs[1,0]]
            
            # Plot unrotated IQ data
            axs[0, 0].plot(Ig, Qg, ".", label="g", color=BLUE, alpha=a, markersize=m)
            axs[0, 0].plot(Ie, Qe, ".", label="e", color=RED, alpha=a, markersize=m)
            if plot_f:
                axs[0, 0].plot(If, Qf, ".", label="f", color=GREEN, alpha=a, markersize=m)
            
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
            axs[1,1].semilogy(bin_cent, ng, color=BLUE)
            bin_cent = (binse[1:] + binse[:-1]) / 2
            axs[1,1].semilogy(bin_cent, ne, color=RED)
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
        ax[0].plot(Ig_new, Qg_new, ".", label="g", color=BLUE, alpha=a, markersize=m)
        ax[0].plot(Ie_new, Qe_new, ".", label="e", color=RED, alpha=a, markersize=m)
        if plot_f:
            ax[0].plot(If_new, Qf_new, ".", label="f", color=GREEN, alpha=a, markersize=m)
        
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
        ax[1].set_title(f"Histogram (Fidelity g-e: {100*fids[0]:.3}%)")
        
        # Plot threshold line
        ax[1].axvline(thresholds[0], color="0.2", linestyle="--")
        
        # Plot histograms and fits
        if 'vhg' in data and 'histg' in data:
            ax[1].plot(data["vhg"], data["histg"], '.-', color=BLUE, markersize=0.5, linewidth=0.3)
            ax[1].fill_between(data["vhg"], data["histg"], color=BLUE, alpha=0.3)
            ax[1].fill_between(data["vhe"], data["histe"], color=RED, alpha=0.3)
            ax[1].plot(data["vhe"], data["histe"], '.-', color=RED, markersize=0.5, linewidth=0.3)
            
            # Plot fitted curves
            if 'paramsg' in data and 'vg' in data and 've' in data and 'sigma' in data and 'tm' in data:
                ax[1].plot(data["vhg"], gaussian(data["vhg"], 1, *data['paramsg']), 'k', linewidth=1)
                ax[1].plot(data["vhe"], excited_func(data["vhe"], data['vg'], data['ve'], data['sigma'], data['tm']), 'k', linewidth=1)
                
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
                If_new, bins=numbins, range=xlims, color=GREEN, label="f", alpha=0.5
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


def fit_single_shot(d, plot=True, rot=True):
    """
    Comprehensive analysis of single-shot readout data.
    
    This function:
    1. Optionally rotates the IQ data
    2. Fits Gaussians to ground state I and Q data
    3. Fits excited state data including T1 decay effects
    4. Returns processed data and fit parameters
    
    Parameters
    ----------
    d : dict
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
    d = deepcopy(d)
    
    # Rotate the data if requested
    if rot:
        # Get rotation angle from histogram analysis
        params, _ = hist(d, plot=False, verbose=False)
        theta = np.pi * params['angle'] / 180
        data = full_rotate(d, theta)
    else:
        data = d
        theta = 0
    # Fit Gaussians to the ground state I and Q data
    paramsg, vhg, histg = fit_gaussian(data["Ig"], nbins=100, p0=None, plot=False)
    paramsqg, vhqg, histqg = fit_gaussian(data["Qg"], nbins=100, p0=None, plot=False)
    
    # Fit Gaussian to excited state Q data
    paramsqe, vhqe, histqe = fit_gaussian(data["Qe"], nbins=100, p0=None, plot=False)

    # Extract parameters from fits
    vqg = paramsqg[0]  # Q center for ground state
    vqe = paramsqe[0]  # Q center for excited state
    vg = paramsg[0]    # I center for ground state
    sigma = paramsg[1] # Width of ground state distribution

    # Fit the excited state I data, including T1 decay effects
    vhe, histe = make_hist(data["Ie"], nbins=100)
    
    # Initial guess for excited state center and T1 decay parameter
    p0 = [np.mean(vhe * histe) / np.mean(histe), 0.2]
    
    # Fit using the excited_func with fixed ground state parameters
    try:
        paramse, params_err = curve_fit(
            lambda x, ve, tm: excited_func(x, vg, ve, sigma, tm), vhe, histe, p0=p0
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
    g_rot = rotate(data['Ig'], data['Qg'], 0)
    e_rot = rotate(data['Ie'], data['Qe'], 0)

    # Try to fit two Gaussians to ground state data (for potential leakage detection)
    pg = [0.99, vg, sigma, 0.01, ve]  # Initial parameters
    try: 
        paramsg2, params_err = curve_fit(two_gaussians, vhg, histg, p0=pg)
    except:
        paramsg2 = np.nan

    # Store rotated data and histograms in the data dictionary

    data["Ie_rot"] = e_rot[0]
    data["Qe_rot"] = e_rot[1]
    data["Ig_rot"] = g_rot[0]
    data["Qg_rot"] = g_rot[1]

    data["vhg"] = vhg
    data["histg"] = histg
    data["vhe"] = vhe
    data["histe"] = histe

    data["vqg"] = vhqg
    data["histqg"] = histqg
    data["vqe"] = vhqe
    data["histqe"] = histqe
    
    # Store fit parameters in the data dictionary
    data["paramsg"] = paramsg
    data["vg"] = vg
    data["ve"] = ve
    data["sigma"] = sigma
    data["tm"] = tm

    # Plot results if requested
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(vhg, histg, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhe, histe, "k.-", markersize=0.5, linewidth=0.3)
        plt.plot(vhg, gaussian(vhg, 1, *paramsg), label="g", linewidth=1)
        plt.plot(vhe, excited_func(vhe, vg, ve, sigma, tm), label="e", linewidth=1)
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
        'g_mean': np.mean(data['Ig']),  # Mean of ground state I data
        'e_mean': np.mean(data['Ie']),  # Mean of excited state I data
    }
    
    return data, p, paramsg, paramse2


def plot_reset(d):
    blue = "#4053d3"
    red = "#b51d14"

    num_plots = len(d)
    fig, ax = plt.subplots(
        int(np.ceil(num_plots / 4)), 4, figsize=(14, 1 * num_plots), sharey=True
    )
    ax = ax.flatten()

    b = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(d[0].data["Igr"]))
    for i, shot in enumerate(d):
        v, hist = make_hist(shot.data["Ig"], nbins=50)
        ax[i].semilogy(v, hist, color=blue)
        ax[i].set_title(f"{shot.cfg.expt.threshold_v:0.2f}")
        ax[i].axvline(x=shot.cfg.expt.threshold_v, color="k", linestyle="--")
        for j in range(len(shot.data["Igr"])):
            v, hist = make_hist(shot.data["Igr"][j], nbins=50)
            ax[i].semilogy(v, hist, color=b[j])

    fig.tight_layout()
    fig, ax = plt.subplots(
        int(np.ceil(num_plots / 4)), 4, figsize=(14, 1 * num_plots), sharey=True
    )
    ax = ax.flatten()
    for i, shot in enumerate(d):
        v, hist = make_hist(shot.data["Ig"], nbins=50)
        ax[i].semilogy(v, hist, color=blue)
        v, hist = make_hist(shot.data["Ie"], nbins=50)
        ax[i].semilogy(v, hist, color=red)
        ax[i].set_title(f"{shot.cfg.expt.threshold_v:0.2f}")
        ax[i].axvline(x=shot.cfg.expt.threshold_v, color="k", linestyle="--")
        for j in range(len(shot.data["Ier"])):
            v, hist = make_hist(shot.data["Ier"][j], nbins=50)
            ax[i].semilogy(v, hist, color=b[j])

    fig.tight_layout()

    nplots = 6
    fig, ax = plt.subplots(2, nplots, figsize=(nplots * 4, 8))
    b = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(d))

    for i, shot in enumerate(d):
        vg, histg = make_hist(shot.data["Ig"], nbins=50)
        ve, histe = make_hist(shot.data["Ie"], nbins=50)
        for j in range(nplots):

            ax[0, j].semilogy(vg, histg, color=blue, linewidth=1)
            ax[1, j].semilogy(vg, histg, color=blue, linewidth=1)

            ax[1, j].semilogy(ve, histe, color=red, linewidth=1)

            v, hist = make_hist(shot.data["Igr"][j, :], nbins=50)
            ax[0, j].semilogy(
                v, hist, label=f"{shot.cfg.expt.threshold_v:0.1f}", color=b[i]
            )

            v, hist = make_hist(shot.data["Ier"][j, :], nbins=50)
            ax[1, j].semilogy(v, hist, label=shot.cfg.expt.threshold_v, color=b[i])
            
    ax[0, 0].legend(ncol=int(np.ceil(len(d) / 6)), fontsize=8)

    ax[0, 0].set_title("Ground state")
    ax[1, 0].set_title("Excited state")
    fig.tight_layout()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # fig.savefig(
    #             shot.fname[0 : -len(imname)] + "images\\" +  + ".png"
    #         )
    fig.savefig(f"reset_hist_{current_time}.png")

def fit_hist(bin_centers, hist, data):
        v_rng = np.max(bin_centers) - np.min(bin_centers)

        p0 = [
            0.5,
            np.min(bin_centers) + v_rng / 3,
            0.5,
            v_rng / 10,
            np.max(bin_centers) - v_rng / 3,
        ]
        try:
            popt, pcov = curve_fit(two_gaussians, bin_centers, hist, p0=p0)
            vg = popt[1]
            ve = popt[4]
            dv = ve - vg
            scale_data = (data - popt[1]) / dv
            hist_fit = popt
        except:
            scale_data = data 
            hist_fit = None
        return scale_data, hist_fit