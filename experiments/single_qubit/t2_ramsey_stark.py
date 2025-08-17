"""
T2 Ramsey experiments with AC Stark effect measurements.

This module implements Ramsey interferometry experiments with AC Stark effects,
allowing for detailed characterization of qubit coherence properties under
external drive conditions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from typing import Dict, Any, Optional, List, Union

from qick import *
from qick.asm_v2 import QickSweep1D

from ..general.qick_experiment import QickExperiment2DSimple, QickExperiment
from .t2 import T2Program
from ...analysis import fitting as fitter


class RamseyStarkExperiment(QickExperiment):
    """
    Ramsey interferometry experiment with AC Stark effect characterization.
    
    This experiment measures T2* (dephasing time) while applying an off-resonant
    drive pulse to induce AC Stark shifts. The Ramsey sequence creates superposition
    states that are sensitive to frequency shifts caused by the Stark effect.
    
    The pulse sequence is:
    1. π/2 pulse to create superposition
    2. Wait time τ (swept parameter) with optional Stark drive
    3. π/2 pulse with phase advance
    4. Readout
    
    Parameters:
        start (float): Initial wait time τ in microseconds
        step (float): Wait time increment. Ensure Nyquist frequency 
                     0.5 * (1/step) > ramsey_freq for proper sampling
        expts (int): Number of time points in the sweep
        ramsey_freq (float): Frequency for phase advance in MHz
        reps (int): Number of averages per time point
        rounds (int): Number of complete experiment rounds
        stark_gain (float): Amplitude of the Stark drive pulse
        stark_freq (float): Frequency of the Stark drive in MHz
        df (float): Frequency offset from qubit frequency for Stark drive
        checkEF (bool): Perform experiment on EF transition instead of GE
        acStark (bool): Enable AC Stark drive during wait time
        
    Returns:
        Oscillatory decay data that can be fit to extract T2* and frequency shifts
    """

    def __init__(
        self,
        cfg_dict: Dict[str, Any],
        qi: int = 0,
        go: bool = True,
        params: Dict[str, Any] = None,
        prefix: Optional[str] = None,
        progress: Optional[Any] = None,
        style: str = "",
        min_r2: Optional[float] = None,
        max_err: Optional[float] = None,
        check_params: bool = True
    ) -> None:
        """
        Initialize Ramsey Stark experiment.
        
        Args:
            cfg_dict: Configuration dictionary containing hardware and device settings
            qi: Qubit index for the experiment
            go: Whether to immediately run the experiment
            params: Override parameters for experiment configuration
            prefix: Custom filename prefix (auto-generated if None)
            progress: Progress callback function
            style: Experiment style modifier ("fine", "fast", or "")
            min_r2: Minimum R-squared value for fit acceptance
            max_err: Maximum fit error for acceptance
            check_params: Whether to validate experiment parameters
        """
        if params is None:
            params = {}
            
        if prefix is None:
            prefix = f"ramsey_stark_qubit{qi}"

        super().__init__(
            cfg_dict=cfg_dict, 
            prefix=prefix, 
            progress=progress, 
            qi=qi, 
            check_params=check_params
        )

        # Define default experimental parameters
        params_def = {
            "expts": 200,
            "reps": 2 * self.reps,
            "rounds": 2 * self.rounds,
            "start": 0.1,  # Initial wait time in μs
            "ramsey_freq": "smart",  # Will be auto-calculated if "smart"
            "stark_gain": 0.5,  # Stark drive amplitude
            "step": cfg_dict['soc'].cycles2us(1) + 0.001,  # Minimum time step
            "df": 70,  # Frequency offset from qubit frequency in MHz
            "acStark": True,  # Enable AC Stark drive
            "checkEF": False,  # Use GE transition by default
            'active_reset': self.cfg.device.readout.active_reset[qi],
            'experiment_type': 'ramsey',
            "num_pi": 0,  # Number of π pulses (0 for standard Ramsey)
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        
        # Merge user parameters with defaults
        params = {**params_def, **params}
        
        # Configure frequency and pulse parameters based on transition type
        if params["checkEF"]:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]
            
        # Copy qubit-specific pulse parameters
        for key in cfg_qub:
            params_def[key] = cfg_qub[key][qi]
            
        # Auto-calculate optimal Ramsey frequency based on T2*
        if params["ramsey_freq"] == "smart":
            # Set to 1/(4*T2*) for good oscillation visibility
            params["ramsey_freq"] = np.pi / (2 * self.cfg.device.qubit.T2r[qi])

        # Apply style-specific parameter modifications
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 30
            
        # Final parameter merge and Stark frequency calculation
        params = {**params_def, **params}
        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]
        self.cfg.expt = params

        # Validate parameters and configure active reset if needed
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()

        # Run experiment immediately if requested
        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress: bool = False) -> Dict[str, Any]:
        """
        Acquire Ramsey data by sweeping wait time.
        
        Sets up the time sweep and executes the T2Program for each time point.
        
        Args:
            progress: Whether to display progress during acquisition
            
        Returns:
            Dictionary containing experimental data including time points and I/Q values
        """
        # Configure sweep parameter metadata
        self.param = {
            "label": "waiting", 
            "param": "t", 
            "param_type": "time"
        }
        
        # Set up time sweep using QickSweep1D
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop",
            self.cfg.expt.start,
            self.cfg.expt.start + self.cfg.expt.step * self.cfg.expt.expts,
        )

        # Execute the experiment using inherited T2Program
        data = super().acquire(T2Program, progress=progress)
        return data

    def analyze(
        self, 
        data: Optional[Dict[str, Any]] = None, 
        fit: bool = True, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze Ramsey data with exponentially decaying sinusoid fit.
        
        Fits the data to: A * exp(-t/T2*) * sin(2πft + φ) + offset
        where T2* is the dephasing time and f is the Ramsey frequency.
        
        Args:
            data: Experimental data dictionary (uses self.data if None)
            fit: Whether to perform fitting
            **kwargs: Additional arguments passed to parent analyze method
            
        Returns:
            Analysis results including fit parameters and uncertainties
        """
        if data is None:
            data = self.data
            
        # Define fitting functions for exponentially decaying oscillation
        fitterfunc = fitter.fitdecaysin  # Fitting routine
        fitfunc = fitter.decaysin       # Model function
        
        if fit:
            super().analyze(fitfunc=fitfunc, fitterfunc=fitterfunc, data=data)

        return self.data

    def display(
        self,
        data: Optional[Dict[str, Any]] = None,
        fit: bool = True,
        debug: bool = False,
        plot_all: bool = False,
        ax: Optional[plt.Axes] = None,
        show_hist: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Display Ramsey Stark experimental results with fit.
        
        Creates a plot showing the exponentially decaying oscillation with
        extracted T2* and frequency parameters.
        
        Args:
            data: Data to plot (uses self.data if None)
            fit: Whether to overlay fit curve
            debug: Enable debug plotting features
            plot_all: Plot all rounds individually
            ax: Matplotlib axes to use for plotting
            show_hist: Whether to show histogram of individual shots
            **kwargs: Additional plotting arguments
            
        Returns:
            Plotted data dictionary
        """
        qubit = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[qubit]
        
        # Create descriptive title with key experimental parameters
        title = (
            f"$T_2^*$ Ramsey Stark Q{qubit} "
            f"Freq: {df:.1f} MHz, Amp: {self.cfg.expt.stark_gain:.3f}"
        )
        xlabel = "Wait Time ($\\mu$s)"

        # Define fit function for display
        fitfunc = fitter.decaysin

        # Configure parameter display in plot caption
        caption_params = [
            {
                "index": 3, 
                "format": "$T_2^*$ Ramsey : {val:.4} $\\pm$ {err:.2g} $\\mu$s"
            },
            {
                "index": 1, 
                "format": "Freq. : {val:.3} $\\pm$ {err:.1g} MHz"
            },
        ]

        # Use parent class display method with custom formatting
        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=show_hist,
            fitfunc=fitfunc,
            caption_params=caption_params,
        )

        return data


class RamseyStarkPowerExperiment(QickExperiment2DSimple):
    """
    2D Ramsey experiment sweeping both wait time and Stark drive amplitude.
    
    This experiment characterizes how the AC Stark shift depends on the drive
    power by performing Ramsey measurements at different Stark drive amplitudes.
    The frequency shift typically shows quadratic dependence on drive amplitude.
    
    The experiment sweeps:
    - X-axis: Wait time τ (like standard Ramsey)
    - Y-axis: Stark drive gain/amplitude
    
    Each (τ, gain) point yields an oscillatory trace that can be fit to extract
    the instantaneous Ramsey frequency, revealing the power-dependent Stark shift.
    """

    def __init__(
        self,
        cfg_dict: Dict[str, Any],
        qi: int = 0,
        go: bool = True,
        params: Dict[str, Any] = None,
        prefix: str = "",
        progress: bool = False,
        style: str = "",
        min_r2: Optional[float] = None,
        max_err: Optional[float] = None,
    ) -> None:
        """
        Initialize 2D Ramsey Stark power experiment.
        
        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index
            go: Whether to run immediately
            params: Parameter overrides
            prefix: Filename prefix (auto-generated if empty)
            progress: Progress display flag
            style: Experiment style modifier
            min_r2: Minimum R-squared for fit acceptance
            max_err: Maximum fit error for acceptance
        """
        if params is None:
            params = {}
            
        if prefix == "":
            prefix = f"ramsey_stark_amp_qubit{qi}"

        super().__init__(
            cfg_dict=cfg_dict, 
            qi=qi, 
            prefix=prefix, 
            progress=progress
        )

        # Define amplitude sweep parameters
        params_def = {
            "end_gain": self.cfg.device.qubit.max_gain,  # Maximum safe gain
            "expts_gain": 20,  # Number of gain points
            "start_gain": 0.15,  # Minimum gain for detectable shift
            "qubit": [qi],
        }
        
        # Create base experiment instance without running
        exp_name = RamseyStarkExperiment
        self.expt = exp_name(
            cfg_dict, 
            qi=qi, 
            go=False, 
            params=params, 
            check_params=False
        )
        
        # Merge parameters and ensure gain limits are respected
        params = {**params_def, **params}
        params.end_gain = np.min([self.cfg.device.qubit.max_gain, params.end_gain])
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        if go:
            super().run(min_r2=min_r2, max_err=max_err, progress=progress)

    def acquire(self, progress: bool = False) -> Dict[str, Any]:
        """
        Acquire 2D data sweeping Stark drive amplitude.
        
        For each gain value, performs a full Ramsey time sweep to extract
        the frequency shift at that drive power.
        
        Args:
            progress: Whether to show acquisition progress
            
        Returns:
            2D dataset with gain and time axes
        """
        # Create linear gain sweep
        gainpts = np.linspace(
            self.cfg.expt["start_gain"],
            self.cfg.expt["end_gain"],
            self.cfg.expt["expts_gain"],
        )

        # Configure 2D sweep with gain as the Y-axis parameter
        y_sweep = [{"var": "stark_gain", "pts": gainpts}]
        super().acquire(y_sweep=y_sweep, progress=progress)

        return self.data

    def analyze(
        self, 
        data: Optional[Dict[str, Any]] = None, 
        fit: bool = True, 
        **kwargs
    ) -> None:
        """
        Analyze 2D Ramsey data and fit quadratic gain dependence.
        
        Each gain slice is fit to extract the Ramsey frequency, then a quadratic
        fit is applied to the frequency vs. gain data to characterize the
        power dependence of the AC Stark shift.
        
        Args:
            data: Experimental data (uses self.data if None)
            fit: Whether to perform fitting
            **kwargs: Additional arguments
        """
        if data is None:
            data = self.data

        # Fit each time trace to extract oscillation parameters
        fitterfunc = fitter.fitsin
        super().analyze(
            fitfunc=fitter.decaysin, 
            fitterfunc=fitterfunc, 
            data=data
        )

        # Extract frequency from each gain slice
        freq = [data["fit_avgi"][i][1] for i in range(len(data["stark_gain_pts"]))]
        
        # Fit quadratic dependence: f(gain) = a*gain² + b*gain + c
        popt, pcov = curve_fit(_quad_fit, data["stark_gain_pts"], freq)
        data["quad_fit"] = popt

    def display(
        self, 
        data: Optional[Dict[str, Any]] = None, 
        fit: bool = True, 
        plot_both: bool = False, 
        **kwargs
    ) -> None:
        """
        Display 2D Ramsey results with quadratic fit analysis.
        
        Shows both the 2D time-gain data and the extracted frequency vs. gain
        relationship with quadratic fit overlay.
        
        Args:
            data: Data to display
            fit: Whether to show fit results
            plot_both: Whether to plot both 2D and derived data
            **kwargs: Additional plotting arguments
        """
        if data is None:
            data = self.data
            
        qubit = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[qubit]

        # Display main 2D plot
        title = f"Stark Power Ramsey Q{qubit} Freq: {df:.1f} MHz"
        ylabel = "Gain [DAC units]"
        xlabel = "Wait Time ($\\mu$s)"
        super().display(
            plot_both=False, 
            title=title, 
            xlabel=xlabel, 
            ylabel=ylabel
        )

        # Create derived analysis plot showing frequency vs. gain
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax = [ax]
        
        if fit:
            freq = [data["fit_avgi"][i][1] for i in range(len(data["stark_gain_pts"]))]
            ax[0].plot(data["stark_gain_pts"], freq, "o", label="Data")

            # Overlay quadratic fit
            ax[0].plot(
                data["stark_gain_pts"],
                _quad_fit(data["stark_gain_pts"], *data["quad_fit"]),
                label=f"Fit: {data['quad_fit'][0]:.3g}$x^2$+{data['quad_fit'][1]:.3g}$x$+{data['quad_fit'][2]:.3g}",
            )
            ax[0].set_xlabel("Gain [DAC units]")
            ax[0].set_ylabel("Frequency [MHz]")
            ax[0].legend()
            ax[0].set_title(f"Stark Power Ramsey Q{qubit} Freq: {df:.1f} MHz")

        # Create waterfall plot of raw data traces
        fig3, ax3 = plt.subplots(1, 1, figsize=(6, 8))
        offset = 0
        for i, gain in enumerate(data["stark_gain_pts"]):
            ax3.plot(
                data["xpts"], 
                data["avgi"][i] + offset,
                label=f'Gain {gain:.3f}' if i % 3 == 0 else ""
            )
            # Offset each trace for clarity
            offset += 2 * data["fit_avgi"][i][0] if fit else 0.1

        ax3.set_xlabel("Wait Time ($\\mu$s)")
        ax3.set_ylabel("Signal + Offset")
        ax3.set_title("Raw Ramsey Traces vs. Stark Gain")
        ax3.legend()

        # Save quadratic fit plot
        try:
            imname = self.fname.split("\\")[-1]
            fig.savefig(
                self.fname[0:-len(imname)] + "images\\" + imname[0:-3] + "quad_fit.png"
            )
        except (AttributeError, IndexError):
            pass  # Graceful handling if filename path parsing fails

        plt.show()


class RamseyStarkFreqExperiment(QickExperiment2DSimple):
    """
    2D Ramsey experiment sweeping wait time and Stark drive frequency.
    
    This experiment maps out the frequency dependence of the AC Stark shift
    by varying the Stark drive frequency while measuring Ramsey oscillations.
    The shift magnitude typically follows dispersive patterns around resonant
    features, particularly showing strong effects near the EF transition.
    
    Key physics: AC Stark shift ∝ Ω²/(Δ) where Ω is drive strength and 
    Δ is detuning from nearby transitions.
    """

    def __init__(
        self,
        cfg_dict: Dict[str, Any],
        qi: int = 0,
        go: bool = True,
        params: Dict[str, Any] = None,
        prefix: str = "",
        progress: bool = False,
        style: str = "",
        min_r2: Optional[float] = None,
        max_err: Optional[float] = None,
    ) -> None:
        """
        Initialize frequency-sweep Ramsey Stark experiment.
        
        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index
            go: Whether to run immediately
            params: Parameter overrides
            prefix: Filename prefix
            progress: Progress display
            style: Experiment style
            min_r2: Minimum fit R-squared
            max_err: Maximum fit error
        """
        if params is None:
            params = {}
            
        if prefix == "":
            prefix = f"ramsey_stark_freq_qubit{qi}"

        super().__init__(
            cfg_dict=cfg_dict, 
            qi=qi, 
            prefix=prefix, 
            progress=progress
        )

        # Define frequency sweep range relative to qubit frequency
        params_def = {
            "end_df": 200,    # End frequency offset in MHz
            "expts_df": 20,   # Number of frequency points
            "start_df": 5,    # Start frequency offset in MHz
            "qubit": [qi],
        }
        
        # Create base experiment without execution
        exp_name = RamseyStarkExperiment
        self.expt = exp_name(
            cfg_dict, 
            qi=qi, 
            go=False, 
            params=params, 
            check_params=False
        )
        
        # Calculate absolute frequencies from relative offsets
        params = {**params_def, **params}
        params["start_freq"] = self.cfg.device.qubit.f_ge[qi] + params["start_df"]
        params["end_freq"] = self.cfg.device.qubit.f_ge[qi] + params["end_df"]
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress: bool = False) -> Dict[str, Any]:
        """
        Acquire data sweeping Stark drive frequency.
        
        Args:
            progress: Progress display flag
            
        Returns:
            2D dataset with frequency and time dimensions
        """
        # Create linear frequency sweep
        freq_pts = np.linspace(
            self.cfg.expt["start_freq"],
            self.cfg.expt["end_freq"],
            self.cfg.expt["expts_df"],
        )

        # Configure 2D sweep with stark_freq as Y parameter
        y_sweep = [{"var": "stark_freq", "pts": freq_pts}]
        super().acquire(y_sweep=y_sweep, progress=progress)

        return self.data

    def analyze(
        self, 
        data: Optional[Dict[str, Any]] = None, 
        fit: bool = True, 
        **kwargs
    ) -> None:
        """
        Analyze frequency-dependent Ramsey data.
        
        Fits each frequency slice to extract oscillation parameters,
        particularly the Ramsey frequency which reveals AC Stark shifts.
        
        Args:
            data: Data to analyze
            fit: Whether to perform fitting
            **kwargs: Additional arguments
        """
        if data is None:
            data = self.data

        # Fit sinusoidal oscillations for each frequency point
        fitterfunc = fitter.fitsin
        super().analyze(
            fitfunc=fitter.sinfunc, 
            fitterfunc=fitterfunc, 
            data=data
        )
        
        # Extract frequency shifts from fit results
        self.data["freq"] = [
            data["fit_avgi"][i][1] for i in range(len(data["ypts"]))
        ]

    def display(
        self, 
        data: Optional[Dict[str, Any]] = None, 
        fit: bool = True, 
        plot_both: bool = False, 
        **kwargs
    ) -> None:
        """
        Display frequency-dependent Ramsey results.
        
        Shows 2D time-frequency data and derived frequency shift analysis
        with comparison to theoretical dispersive model.
        
        Args:
            data: Data to display
            fit: Show fit overlays
            plot_both: Show both 2D and derived plots
            **kwargs: Additional arguments
        """
        if data is None:
            data = self.data
            
        qubit = self.cfg.expt.qubit[0]
        gain = self.cfg.expt.stark_gain

        # Main 2D display
        title = f"Stark Freq Ramsey Q{qubit} Gain: {gain:.3f}"
        ylabel = "Frequency (MHz)"
        xlabel = "Wait Time ($\\mu$s)"
        super().display(
            plot_both=False, 
            title=title, 
            xlabel=xlabel, 
            ylabel=ylabel
        )

        # Analysis plot showing frequency dependence
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax = [ax]
        
        if fit:
            self.data["freq"] = [
                data["fit_avgi"][i][1] for i in range(len(data["ypts"]))
            ]
            ax[0].plot(data["ypts"], self.data["freq"], "o", label="Measured")
            
            # Calculate theoretical AC Stark shift using dispersive formula
            # Shift ∝ α * Ω² / (Δ_ge * Δ_ef) where α is anharmonicity
            alpha = (
                self.cfg.device.qubit.f_ge[qubit] 
                - self.cfg.device.qubit.f_ef[qubit]
            )
            df_ge = data["ypts"] - self.cfg.device.qubit.f_ge[qubit]
            df_ef = data["ypts"] - self.cfg.device.qubit.f_ef[qubit]
            
            # Theoretical shift (arbitrary scaling factor)
            exp_val = np.abs(alpha * gain**2 / df_ge / df_ef) * 500
            
            # Only plot reasonable values to avoid divergences
            valid_inds = exp_val < 30
            if np.any(valid_inds):
                ax[0].plot(
                    data["ypts"][valid_inds], 
                    exp_val[valid_inds], 
                    ".", 
                    label="Theory (scaled)"
                )
            
            ax[0].set_xlabel("Stark Drive Frequency (MHz)")
            ax[0].set_ylabel("Ramsey Frequency (MHz)")
            ax[0].legend()
            ax[0].set_title(title)

        # Waterfall plot of raw traces
        fig3, ax3 = plt.subplots(1, 1, figsize=(6, 8))
        offset = 0
        for i, freq in enumerate(data["ypts"]):
            ax3.plot(
                data["xpts"], 
                data["avgi"][i] + offset,
                label=f'f = {freq:.1f} MHz' if i % 3 == 0 else ""
            )
            offset += 2 * data["fit_avgi"][i][0] if fit else 0.1

        ax3.set_xlabel("Wait Time ($\\mu$s)")
        ax3.set_ylabel("Signal + Offset")
        ax3.set_title("Raw Ramsey Traces vs. Stark Frequency")
        ax3.legend()

        plt.show()


def _quad_fit(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    Quadratic function for fitting power-dependent frequency shifts.
    
    Args:
        x: Input variable (typically drive amplitude)
        a: Quadratic coefficient
        b: Linear coefficient  
        c: Constant offset
        
    Returns:
        Quadratic function values: a*x² + b*x + c
    """
    return a * x**2 + b * x + c
