from scipy import signal
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from .collections import _get_tracking_base

def welch_psd(
    data,
    fs=1.0,
    window="hann",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    average="mean",
):
    """
    Calculate the power spectral density (PSD) using Welch's method.

    This function estimates the power spectral density of a signal by dividing
    the data into overlapping segments, windowing each segment, computing the
    FFT, and averaging the periodograms.

    Parameters:
    -----------
    data : array_like
        Input signal data
    fs : float, optional
        Sampling frequency of the input signal. Default is 1.0.
    window : str or tuple or array_like, optional
        Desired window to use. Default is 'hann'. See scipy.signal.get_window
        for a list of windows and required parameters.
    nperseg : int, optional
        Length of each segment. If None, uses scipy default (256 samples).
    noverlap : int, optional
        Number of points to overlap between segments. If None, uses 50% overlap.
    nfft : int, optional
        Length of the FFT used. If None, uses nperseg.
    detrend : str or function or False or None, optional
        Specifies how to detrend each segment. Default is 'constant'.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real inputs. Default is True.
    scaling : {'density', 'spectrum'}, optional
        Selects between computing the power spectral density ('density')
        or power spectrum ('spectrum'). Default is 'density'.
    average : {'mean', 'median'}, optional
        Method to use when averaging periodograms. Default is 'mean'.

    Returns:
    --------
    freqs : ndarray
        Array of sample frequencies
    psd : ndarray
        Power spectral density or power spectrum of the input signal

    Notes:
    ------
    The Welch method is an improvement over the periodogram method of spectral
    density estimation. The periodogram method applies a window function to
    the entire signal and then computes the FFT. Welch's method divides the
    signal into overlapping segments, applies a window to each segment,
    computes the FFT of each windowed segment, and then averages the
    periodograms to reduce noise.

    Examples:
    ---------
    >>> # Generate a test signal with two frequency components
    >>> fs = 1000  # Sampling frequency
    >>> t = np.linspace(0, 1, fs, endpoint=False)
    >>> signal = np.sin(2*np.pi*50*t) + 0.5*np.sin(2*np.pi*120*t) + 0.1*np.random.randn(len(t))
    >>>
    >>> # Calculate PSD using Welch method
    >>> freqs, psd = welch_psd(signal, fs=fs)
    >>>
    >>> # Plot the result
    >>> plt.semilogy(freqs, psd)
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('PSD [V²/Hz]')
    >>> plt.show()
    """

    # Convert input to numpy array
    data = np.asarray(data)

    # Check if data is 1D
    if data.ndim != 1:
        raise ValueError("Input data must be 1-dimensional")

    # Check for sufficient data length
    if len(data) < 2:
        raise ValueError("Input data must have at least 2 samples")

    # Use scipy's welch function for the core computation
    freqs, psd = signal.welch(
        data,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        return_onesided=return_onesided,
        scaling=scaling,
        average=average,
    )

    return freqs, psd

def analyze_qubit_psd(
    data,
    fs=1.0,
    parameter_name="T1",
    qubit_id=None,
    window="hann",
    nperseg=None,
    plot=True,
    save_path=None,
    tracking_id=None,
    fname="",
):
    """
    Analyze the power spectral density of qubit measurement data.

    This function is specifically designed for analyzing the frequency content
    of qubit parameters like T1, T2, frequency drift, etc.

    Parameters:
    -----------
    data : array_like
        Time series data of the qubit parameter
    fs : float, optional
        Sampling frequency. Default is 1.0 Hz.
    parameter_name : str, optional
        Name of the parameter being analyzed. Default is "T1".
    qubit_id : int or str, optional
        Qubit identifier for labeling plots
    window : str, optional
        Window function to use. Default is 'hann'.
    nperseg : int, optional
        Length of each segment for Welch method
    plot : bool, optional
        Whether to create plots. Default is True.
    save_path : str, optional
        Base path for saving plots (will save to Tracking/images/)
    tracking_id : str, optional
        Tracking ID to include in filename
    fname : str, optional
        Additional string to append to filename

    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'freqs': frequency array
        - 'psd': power spectral density
        - 'peak_freq': frequency of maximum power
        - 'total_power': total power in the signal
        - 'noise_floor': estimated noise floor
        - 'snr_db': signal-to-noise ratio in dB
    """

    # Remove NaN values and detrend
    data_clean = data[~np.isnan(data)]

    if len(data_clean) < 10:
        warnings.warn("Insufficient data points for reliable PSD analysis")
        return None

    # Calculate PSD using Welch method
    freqs, psd = welch_psd(data_clean, fs=fs, window=window, nperseg=nperseg)

    # Find peak frequency (excluding DC component)
    if len(freqs) > 1:
        peak_idx = np.argmax(psd[1:]) + 1  # Skip DC component
        peak_freq = freqs[peak_idx]
    else:
        peak_freq = 0

    # Calculate total power
    total_power = np.trapz(psd, freqs)

    # Estimate noise floor (median of upper half of frequencies)
    if len(psd) > 4:
        noise_floor = np.median(psd[len(psd) // 2 :])
    else:
        noise_floor = np.median(psd)

    # Calculate SNR
    signal_power = np.max(psd[1:]) if len(psd) > 1 else np.max(psd)
    snr_db = 10 * np.log10(signal_power / noise_floor) if noise_floor > 0 else np.inf

    # Create results dictionary
    results = {
        "freqs": freqs,
        "psd": psd,
        "peak_freq": peak_freq,
        "total_power": total_power,
        "noise_floor": noise_floor,
        "snr_db": snr_db,
        "n_samples": len(data_clean),
        "sampling_freq": fs,
    }

    # Create plots if requested
    if plot:
        # Create figure with subplots
        fig, ax2 = plt.subplots(1, 1, figsize=(8, 5))

        # Plot time series

        def power_law(f, A, alpha):
            return A * np.power(f, alpha)

        # Use only nonzero frequencies for fitting (avoid DC and Nyquist)
        fit_mask = (freqs[1:-1] > 0) & np.isfinite(psd[1:-1]) & (psd[1:-1] > 0)
        fit_freqs = freqs[1:-1][fit_mask]
        fit_psd = psd[1:-1][fit_mask]

        if len(fit_freqs) > 2:
            # Fit in log-log space for stability
            log_f = np.log10(fit_freqs)
            log_psd = np.log10(fit_psd)
            popt, pcov = curve_fit(lambda f, A, alpha: A + alpha * f, log_f, log_psd)
            A_fit = 10 ** popt[0]
            alpha_fit = popt[1]
            # Plot the fitted power law
            ax2.plot(
                fit_freqs,
                A_fit * fit_freqs**alpha_fit,
                "k--",
                linewidth=1.5,
                label=f"Power law: $f^{{{alpha_fit:.2f}}}$",
            )
                # Plot PSD
        ax2.loglog(freqs[1:-1], psd[1:-1], "o", linewidth=1.5, markersize=2)
        A_fit, alpha_fit = np.nan, np.nan

        # --- Fit PSD to a sum of power law and Lorentzian: PSD(f) = A * f^alpha + B / (1 + ((f - f0)/gamma)^2) ---
        def power_law_lorentzian(f, A, alpha, B, gamma):
            return A * np.power(f, alpha) + B / (1 + (f / gamma) ** 2)
        
        def white_lorentzian(f, A, B, gamma):
            return A + B / (1 + (f / gamma) ** 2)
            # --- Fit PSD to white noise + Lorentzian: PSD(f) = A + B / (1 + (f/gamma)^2) ---

        p0_white = [
        np.median(fit_psd),
        np.max(fit_psd),
        (fit_freqs.max() - fit_freqs.min()) / 10,
        ]
        bounds_white = (
        [0, 0, 1e-8],
        [np.inf, np.inf, (fit_freqs.max() - fit_freqs.min())],
        )
        popt_white, pcov_white = curve_fit(
        white_lorentzian,
        fit_freqs,
        fit_psd,
        p0=p0_white,
        bounds=bounds_white,
        maxfev=10000,
        )
        A_white, B_white, gamma_white = popt_white
        ax2.plot(
        fit_freqs,
        white_lorentzian(fit_freqs, *popt_white),
        linewidth=1.5,
        linestyle=":",
        label=f"White+Lorentz: $A$={A_white:.2e}, $\\gamma$={1/gamma_white:.0f} s",
            )

        # Initial guesses: A_fit, alpha_fit, B=fit_psd.max(), f0=fit_freqs[np.argmax(fit_psd)], gamma=fit_freqs.ptp()/10
        p0 = [
            np.median(fit_psd),
            -1.0,
            np.max(fit_psd),
            (fit_freqs.max() - fit_freqs.min()) / 10,
        ]
        bounds = (
            [0, -10, 0, 1e-8],
            [
                np.inf,
                10,
                np.inf,
                (fit_freqs.max() - fit_freqs.min()),
            ],
        )
        popt2, pcov2 = curve_fit(
            power_law_lorentzian,
            fit_freqs,
            fit_psd,
            p0=p0,
            bounds=bounds,
            maxfev=10000,
        )
        A2, alpha2, B2, gamma2 = popt2
        ax2.plot(
            fit_freqs,
            power_law_lorentzian(fit_freqs, *popt2),
            linewidth=1.5,
            label=f"Power+Lorentz: $f^{{{alpha2:.2f}}}$ + $\gamma$({1/gamma2:.0f} s)",
        )

        # Add fit results to plot
        ax2.legend()



        # ax2.axhline(
        #     noise_floor,
        #     linestyle="--",
        #     alpha=0.7,
        #     label=f"Noise floor: {noise_floor:.2e}",
        # )

        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("PSD")
        ax2.set_title(f"{parameter_name} Power Spectral Density (SNR: {snr_db:.1f} dB)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save plot if path provided
        if save_path:
            # Save to Tracking/images/ with tracking_id in filename
            tracking_base = _get_tracking_base(save_path)
            images_dir = tracking_base / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            prefix = f"{tracking_id}_" if tracking_id else ""
            filename = f"{prefix}{parameter_name}_psd"
            if qubit_id is not None:
                filename += f"_q{qubit_id}"
            if fname:
                filename += f"_{fname}"
            filename += ".png"

            full_path = images_dir / filename
            plt.savefig(full_path, dpi=300, bbox_inches="tight")

        plt.show()

    return results