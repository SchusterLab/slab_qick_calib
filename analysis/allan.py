import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from datetime import datetime
from scipy.optimize import curve_fit
from pathlib import Path
from .collections import get_parameter_config, _get_tracking_base



def lorentzian(tau, A, tau0):
        return (
            A * tau0  / tau
            * (4 * np.exp(-tau / tau0) - np.exp(-2 * tau / tau0) - 3 + 2 * tau / tau0) ** 0.5
        )
def power_law(t, A, alpha):
        return A * np.power(t, -(alpha / 2 + 0.5))


def combined_model(tau, A, alpha, B, tau0):
        return power_law(tau, A, alpha) + lorentzian(tau, B, tau0)

def combined_model_two(tau, A, alpha, B, tau0, C, tau1):
        return power_law(tau, A, alpha) + lorentzian(tau, B, tau0) + lorentzian(tau, C, tau1)

def lorentz_model_two(tau, B, tau0, C, tau1):
        return lorentzian(tau, B, tau0) + lorentzian(tau, C, tau1)

def fit_power_law(tau, allan_dev):
    """Fit Allan deviation to a power law: A * tau^alpha."""

    popt, pcov = curve_fit(power_law, tau, allan_dev, p0=[1, -0.5])
    return popt, pcov


def fit_lorentzian(tau, allan_dev):
    """Fit Allan deviation to a Lorentzian"""
    
    popt, pcov = curve_fit(
        lorentzian, tau, allan_dev, p0=[np.max(allan_dev), 2000]
    )
    return popt, pcov

def fit_combined_model(tau, allan_dev):
    """Fit Allan deviation to a combined model: power law + Lorentzian"""

    popt, pcov = curve_fit(
        combined_model,
        tau,
        allan_dev,
        p0=[1, -0.5, np.max(allan_dev), 2000],
    )
    return popt, pcov

def fit_combined_model_two(tau, allan_dev):
    """Fit Allan deviation to a combined model: power law + Lorentzian"""

    popt, pcov = curve_fit(
        combined_model_two,
        tau,
        allan_dev,
        p0=[1, -0.5, np.max(allan_dev), 1500, np.max(allan_dev), 3000],
    )
    return popt, pcov

def fit_lorentz_model_two(tau, allan_dev):
    """Fit Allan deviation to a combined model: power law + Lorentzian"""

    popt, pcov = curve_fit(
        lorentz_model_two,
        tau,
        allan_dev,
        p0=[np.max(allan_dev)/2, 1500, np.max(allan_dev)/2, 3000],
    )
    return popt, pcov

def calculate_allan_variance(data, tau_values):
    """
    Calculate the Allan variance for a time series.

    Parameters:
    data (array): Time series data
    tau_values (array): Array of tau values to calculate Allan variance for

    Returns:
    array: Allan variance for each tau value
    """
    allan_var = np.zeros(len(tau_values))

    for i, tau in enumerate(tau_values):
        # Skip if tau is too large for the dataset
        if tau >= len(data) // 2:
            allan_var[i] = np.nan
            continue

        # Calculate differences between consecutive bins of size tau
        bins = np.floor(len(data) / tau).astype(int)
        bin_data = np.zeros(bins)

        for j in range(bins):
            bin_data[j] = np.mean(data[j * tau : (j + 1) * tau])

        # Calculate Allan variance
        diff = np.diff(bin_data)
        allan_var[i] = np.sum(diff**2) / (2 * (bins - 1))

    return allan_var


def calculate_overlapping_allan_variance(data, tau_values):
    """
    Calculate the overlapping Allan variance for a time series.

    Parameters:
    data (array): Time series data
    tau_values (array): Array of tau values to calculate Allan variance for

    Returns:
    array: Overlapping Allan variance for each tau value
    """
    data = data/np.median(data)
    allan_var = np.zeros(len(tau_values))

    for i, tau in enumerate(tau_values):
        # Skip if tau is too large for the dataset
        if tau >= len(data) // 2:
            allan_var[i] = np.nan
            continue

        # Calculate overlapping differences
        n = len(data)
        m = int(n - 2 * tau + 1)

        if m <= 0:
            allan_var[i] = np.nan
            continue

        # Calculate overlapping Allan variance
        sum_var = 0
        for j in range(m):
            y1 = np.mean(data[j : j + tau])
            y2 = np.mean(data[j + tau : j + 2 * tau])
            sum_var += (y2 - y1) ** 2

        allan_var[i] = sum_var / (2 * m)

    return allan_var


def perform_analysis(tt,fname='def', param='t1', qubit_list=None, save_path=None, tracking_id=None):

    if qubit_list is None:
        qubit_list = range(len(tt))

    parameter_config = get_parameter_config()
    param_label = parameter_config.get(param, {}).get('label', param)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes = axes.flatten()
    par_mean =[]
    par_std =[]
    for i in qubit_list:
        times_filtered = np.array(tt[i]['time'])-np.min(np.array(tt[i]['time']))
        par_filtered = np.array(tt[i][param])

        # Plot T1 vs time
        axes[0].plot(times_filtered , par_filtered, "o", markersize=1, linewidth=1, label=f"{i}")

        # Calculate tau values for Allan variance (logarithmically spaced)
        min_tau = 1
        max_tau = len(par_filtered) // 4
        n_taus = 20
        tau_values = np.logspace(np.log10(min_tau), np.log10(max_tau), n_taus).astype(
            int
        )
        tau_values = np.unique(tau_values)  # Remove duplicates

        # Calculate Allan variance
        allan_var = calculate_allan_variance(par_filtered, tau_values)

        # Calculate overlapping Allan variance for better statistics
        overlap_allan_var = calculate_overlapping_allan_variance(
            par_filtered, tau_values
        )

        # Convert tau values to hours using the time differences
        mean_time_diff = np.mean(np.diff(times_filtered))
        tau_seconds = (
            tau_values * mean_time_diff 
        )  # mean_time_diff is in hours, convert to seconds

        # Calculate and plot Allan deviation (square root of Allan variance)
        axes[1].loglog(tau_seconds, np.sqrt(overlap_allan_var), "o-", markersize=4, linewidth=1, label=f"{i}")

        # Fit the Allan deviation to the models
        try:
            popt_powerlaw, pcov_powerlaw = fit_power_law(
                tau_seconds, np.sqrt(overlap_allan_var)
            )
            A_powerlaw, alpha_powerlaw = popt_powerlaw
            # axes[1].loglog(
            #     tau_seconds,
            #     A_powerlaw * tau_seconds**alpha_powerlaw,
            #     "--",
            #     label=f"Power Law: A={A_powerlaw:.2e}, α={alpha_powerlaw:.2f}",
            # )
        except Exception as e:
            print(f"Power law fit failed: {e}")

        try:
            popt_lorentzian, pcov_lorentzian = fit_lorentzian(
                tau_seconds, np.sqrt(overlap_allan_var)
            )
            B_lorentzian, tau0_lorentzian = popt_lorentzian
            # axes[1].loglog(
            #     tau_seconds,
            #     B_lorentzian / (1 + (tau_seconds / tau0_lorentzian) ** 2),
            #     "--",
            #     label=f"Lorentzian: B={B_lorentzian:.2e}, τ0={tau0_lorentzian:.2f}",
            # )
        except Exception as e:
            print(f"Lorentzian fit failed: {e}")

        try:
            popt_combined, pcov_combined = fit_combined_model(tau_seconds, np.sqrt(overlap_allan_var))
            A_combined, alpha_combined, B_combined, tau0_combined  = popt_combined
            # axes[1].loglog(tau_seconds, combined_model(tau_seconds, *popt_combined), "k-",
            #     label=f"α={alpha_combined:.2f}, $τ_0$={tau0_combined:.2f}",
            # )
            # label=f"Combined: A={A_combined:.2e}, α={alpha_combined:.2f}, B={B_combined:.2e}, τ0={tau0_combined:.2f}",
        except Exception as e:
             print(f"Combined fit failed: {e}")

        try:
            popt_combined, pcov_combined = fit_combined_model_two(tau_seconds, np.sqrt(overlap_allan_var))
            A_combined, alpha_combined, B_combined, tau0_combined, C_combined, tau1_combined = popt_combined
            # axes[1].loglog(tau_seconds, combined_model_two(tau_seconds, *popt_combined), "k-",
            #     label=f"α={alpha_combined:.2f}, $τ_0$={tau0_combined:.2f}, C={C_combined:.2f}, $τ_1$={tau1_combined:.2f}",
            # )
            # label=f"Combined: A={A_combined:.2e}, α={alpha_combined:.2f}, B={B_combined:.2e}, τ0={tau0_combined:.2f}",
        except Exception as e:
             print(f"Combined fit failed: {e}")

        try:
            popt_combined, pcov_combined = fit_lorentz_model_two(tau_seconds, np.sqrt(overlap_allan_var))
            B_combined, tau0_combined, C_combined, tau1_combined = popt_combined
            axes[1].loglog(tau_seconds, lorentz_model_two(tau_seconds, *popt_combined), "k-",
                label=f"$τ_0$={tau0_combined:.2f}, C={C_combined:.2f}, $τ_1$={tau1_combined:.2f}",
            )
                # label=f"Combined: A={A_combined:.2e}, α={alpha_combined:.2f}, B={B_combined:.2e}, τ0={tau0_combined:.2f}",
        except Exception as e:
              print(f"Combined fit failed: {e}")

        # Print some statistics
        par_mean.append(np.mean(par_filtered))
        par_std.append(np.std(par_filtered))
        # # print(f"  Mean T1: {np.mean(par_filtered):.2f} µs")
        # # print(f"  Std T1: {np.std(par_filtered):.2f} µs")
        # # print(
        # #     f"  Relative stability (std/mean): {np.std(par_filtered)/np.mean(par_filtered):.4f}"
        # # )

        # # Find the minimum Allan variance and corresponding tau
        # min_idx = np.nanargmin(overlap_allan_var)
        # if min_idx < len(tau_seconds):
        #     print(
        #         f"  Minimum Allan variance at tau = {tau_seconds[min_idx]:.2f} seconds"
        #     )
        #     print(f"  Minimum Allan variance: {overlap_allan_var[min_idx]:.2e}")
        #     print(
        #         f"  Minimum Allan deviation: {np.sqrt(overlap_allan_var[min_idx]):.2e}"
        #     )

        # Add a slope reference line for white frequency noise (slope = -1)

        x_ref = np.logspace(np.log10(axes[1].get_xlim()[0]), np.log10(axes[1].get_xlim()[1]), 100)

        # # For deviation, slope is -1/2
        # y_ref = x_ref ** (-0.5) * 10 ** (-2) * np.mean(np.sqrt(overlap_allan_var)) * 1e3
        # axes[1].loglog(x_ref, y_ref, "k--", linewidth=1, label="τ⁻¹/² (white FM noise)")

        # # For deviation, random walk has slope +1/2
        # y_ref = x_ref ** (0.5) * 10 ** (-4) * np.mean(np.sqrt(overlap_allan_var)) * 1e5
        # # axes[1].loglog(x_ref, y_ref, "r--", linewidth=1, label="τ¹/² (random walk)")
        # # For deviation, flicker has slope 0
        # y_ref = np.ones_like(x_ref) * 10 ** (-3) * np.mean(np.sqrt(overlap_allan_var)) * 1e3
        # axes[1].loglog(x_ref, y_ref, "g--", linewidth=1, label="τ⁰ (flicker)")

        # axes[1].legend(fontsize=8)
        # Set plot labels and titles
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(param_label)
    axes[0].legend()

    axes[1].set_xlabel("$\\tau$ (s)")
    axes[1].set_ylabel("Normalized Overlapping Allan Deviation")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    if save_path is not None:
        # Save to Tracking/images/ with tracking_id in filename
        tracking_base = _get_tracking_base(save_path)
        images_dir = tracking_base / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{tracking_id}_" if tracking_id else ""
        save_file = images_dir / f"{prefix}allan_variance_{param}_{fname}.png"
        plt.savefig(save_file, dpi=300)

    par_mean = np.array(par_mean)
    par_std = np.array(par_std)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].errorbar(range(len(par_mean)), par_mean, yerr=par_std, fmt='o')
    axes[0].set_xlabel("Qubit")
    axes[0].set_ylabel(f"Mean {param_label}")
    axes[1].plot(par_mean, par_std/par_mean, 'o')
    axes[1].set_xlabel(f"Mean {param_label}")
    axes[1].set_ylabel("Coefficient of Variation")
    plt.tight_layout()
    if save_path is not None:
        # Save to Tracking/images/ with tracking_id in filename
        tracking_base = _get_tracking_base(save_path)
        images_dir = tracking_base / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{tracking_id}_" if tracking_id else ""
        save_file = images_dir / f"{prefix}allan_stats_{param}_{fname}.png"
        fig.savefig(save_file, dpi=300)
    plt.show()

