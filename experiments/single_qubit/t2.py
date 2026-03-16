"""
T2 Measurement Module

This module implements T2 (dephasing time) measurements for superconducting qubits.
T2 is a measure of how long a qubit maintains phase coherence in the x-y plane of the Bloch sphere.

The module supports three main measurement protocols:
1. Ramsey: Uses two π/2 pulses separated by a variable delay time
2. Echo: Uses two π/2 pulses with one or more π pulses in between to refocus dephasing
3. CPMG: Uses a sequence of π pulses to dynamically decouple the qubit from the environment

Additional features include:
- AC Stark shift measurements during Ramsey experiments
- EF transition measurements (first excited to second excited state)
- Automatic frequency error detection and correction
"""

import numpy as np
import time
from qick import *
from qick.asm_v2 import QickSweep1D
from tqdm import tqdm

from ...analysis import fitting as fitter
from ..general.qick_experiment import QickExperiment, QickExperiment2DSimple
from ..general.qick_program import QickProgram

from ...exp_handling.datamanagement import AttrDict


class T2Program(QickProgram):
    """
    Quantum program for T2 measurements (Ramsey, Echo, and CPMG protocols).

    This class defines the pulse sequences for T2 measurements:
    - Ramsey: π/2 - wait - π/2 sequence to measure phase coherence
    - Echo: π/2 - wait - π - wait - π/2 sequence to refocus dephasing
    - CPMG: π/2 - (wait - π)^n - wait - π/2 sequence for dynamical decoupling

    Additional options include AC Stark shift during wait time and EF transition measurements.
    """

    def __init__(self, soccfg, final_delay, cfg, final_wait=0):
        """
        Initialize the T2 program.

        Args:
            soccfg: SOC configuration
            final_delay: Delay after measurement before next experiment
            cfg: Configuration dictionary containing experiment parameters
        """
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg, final_wait=0)

    def _initialize(self, cfg):
        """
        Initialize the program by setting up the pulse sequence.

        Creates the necessary pulses for T2 measurements:
        - Two π/2 pulses (prep and read)
        - Optional π pulse(s) for Echo and CPMG
        - Optional AC Stark pulse for Ramsey with AC Stark shift

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)

        # Create loop for sweeping wait time
        self.add_loop("wait_loop", cfg.expt.expts)

        # Initialize standard readout
        super()._initialize(cfg, readout="standard")

        # Create π/2 pulses for Ramsey/Echo sequence
        # First π/2 pulse has phase=0, second has phase based on wait time and Ramsey frequency
        pulse = {
            "sigma": cfg.expt.sigma,  # Half sigma for π/2 pulse
            "sigma_inc": cfg.expt.sigma_inc,
            "freq": cfg.expt.freq,
            "gain": cfg.expt.gain / 2,
            "phase": 0,  # First pulse has zero phase
            "type": cfg.expt.type,
        }

        # Create first π/2 pulse (preparation)
        super().make_pulse(pulse, "pi2_prep")

        # Create second π/2 pulse (readout) with phase that depends on wait time
        # Phase advances at rate of ramsey_freq (MHz) * wait_time (μs) * 360 (deg/cycle)
        pulse["phase"] = cfg.expt.wait_time * 360 * cfg.expt.ramsey_freq
        super().make_pulse(pulse, "pi2_read")

        

        # For AC Stark shift in Ramsey experiments
        if hasattr(cfg.expt, "acStark") and cfg.expt.acStark:
            # Create pulse to apply during wait time
            pulse = {
                "sigma": cfg.expt.wait_time,  # Duration matches wait time
                "sigma_inc": 0,
                "freq": cfg.expt.stark_freq,
                "gain": cfg.expt.stark_gain,
                "phase": 0,
                "type": "flat_top",  # Constant amplitude pulse
            }
            super().make_pulse(pulse, "stark_pulse")
        elif cfg.expt.flux:
            self.declare_gen(cfg.expt.flux_chan, nqz=1, mixer_freq=0)
            if cfg.expt.num_pi > 0:
                # Echo: separate flux pulses for half-segments and full segments
                flux_half = {
                    "chan": cfg.expt.flux_chan,
                    "freq": 0,
                    "phase": 0,
                    "gain": cfg.expt.flux_gain,
                    "length": cfg.expt.wait_time / cfg.expt.num_pi / 2,
                    "type": "const",
                }
                super().make_pulse(flux_half, "flux_half")
                flux_full = {
                    "chan": cfg.expt.flux_chan,
                    "freq": 0,
                    "phase": 0,
                    "gain": cfg.expt.flux_gain,
                    "length": cfg.expt.wait_time / cfg.expt.num_pi,
                    "type": "const",
                }
                super().make_pulse(flux_full, "flux_full")
            else:
                # Ramsey: single flux pulse for full wait time
                flux_pulse = {
                    "chan": cfg.expt.flux_chan,
                    "freq": 0,
                    "phase": 0,
                    "gain": cfg.expt.flux_gain,
                    "length": cfg.expt.wait_time,
                    "type": "const",
                }
                super().make_pulse(flux_pulse, "flux_pulse")

        # Create π pulse for Echo or EF check
        if cfg.expt.experiment_type == "cpmg":
            cfg.device.qubit.pulses.pi_ge.phase = 90 * np.ones(
                len(cfg.device.qubit.f_ge)
            )
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
        elif (
            cfg.expt.checkEF
            or cfg.expt.experiment_type == "echo"
            or cfg.expt.active_reset
        ):
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        if cfg.expt.checkEF and cfg.expt.experiment_type == "echo":
            # Create π pulse for EF transition check
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

        # for i in range(1000):
        #     self.nop()

    def _body(self, cfg):
        """
        Define the main body of the pulse sequence.

        Implements the actual T2 measurement sequence:
        - Ramsey: π/2 - wait - π/2
        - Echo: π/2 - wait/2 - π - wait/2 - π/2
        - With options for AC Stark and EF measurements

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)
        # Configure readout
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # For EF transition check in Ramsey: Apply π pulse to excite |g⟩ to |e⟩ first
        if hasattr(cfg.expt, "checkEF") and cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef")  # Small buffer delay

        # First π/2 pulse (preparation)
        self.pulse(ch=self.qubit_ch, name="pi2_prep", t=0.0)

        # Handle different experiment types
        if (
            hasattr(cfg.expt, "acStark") and cfg.expt.acStark
        ):  # Ramsey with AC Stark shift
            # Note: AC Stark shift is not compatible with Echo protocol
            self.delay_auto(t=0.01, tag="wait st")  # Small buffer delay
            self.pulse(
                ch=self.qubit_ch, name="stark_pulse", t=0
            )  # Apply AC Stark pulse
            self.delay_auto(t=0.025, tag="waiting")  # Additional wait time
        elif cfg.expt.flux:
            # Flux pulse applied only during wait segments (not during pi pulses)
            if cfg.expt.num_pi > 0:
                # Echo with flux: flux on during each wait segment
                self.pulse(ch=cfg.expt.flux_chan, name="flux_half", t=0)
                self.delay_auto(t=0.01, tag="wait")

                for i in range(cfg.expt.num_pi):
                    self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
                    self.delay_auto(t=0.01, tag=f"wait_pi{i}")
                    if i < cfg.expt.num_pi - 1:
                        self.pulse(ch=cfg.expt.flux_chan, name="flux_full", t=0)
                        self.delay_auto(t=0.01, tag=f"wait{i}")

                self.pulse(ch=cfg.expt.flux_chan, name="flux_half", t=0)
                self.delay_auto(t=0.01, tag=f"wait{cfg.expt.num_pi}")
            else:
                # Ramsey with flux
                self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse", t=0)
                self.delay_auto(t=0.05, tag="wait")
        else:
            # Standard Ramsey or Echo sequence
            # For Echo, divide wait time by (num_pi + 1) to get segments between pulses
            if cfg.expt.num_pi > 0:
                self.delay_auto(t=cfg.expt.wait_time / cfg.expt.num_pi / 2, tag="wait")

                # Apply π pulses for Echo protocol (or multiple-pulse Echo)
                for i in range(cfg.expt.num_pi):
                    self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)  # π pulse
                    if i < cfg.expt.num_pi - 1:
                        self.delay_auto(
                            t=cfg.expt.wait_time / cfg.expt.num_pi + 0.01,
                            tag=f"wait{i}",
                        )  # Wait time
                self.delay_auto(
                    t=cfg.expt.wait_time / cfg.expt.num_pi / 2 + 0.01, tag=f"wait{i+1}"
                )
            else:
                self.delay_auto(t=cfg.expt.wait_time, tag="wait")

        # Second π/2 pulse (readout)
        self.pulse(ch=self.qubit_ch, name="pi2_read", t=0)
        self.delay_auto(t=0.01, tag="wait rd")  # Small buffer delay

        # For EF transition check in Ramsey: Apply π pulse to return to |g⟩ for readout
        if hasattr(cfg.expt, "checkEF") and cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef 2")  # Small buffer delay

        # Measure the qubit state
        super().measure(cfg)


class T2Experiment(QickExperiment):
    """
    T2 Experiment - Supports Ramsey, Echo, and CPMG protocols

    Experimental Config for Ramsey:
    expt = dict(
        experiment_type: "ramsey", "echo", or "cpmg"
        start: total wait time b/w the two pi/2 pulses start sweep [us]
        span: total increment of wait time across experiments [us]
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        acStark: True/False (Ramsey only)
        checkEF: True/False (Ramsey only)
    )

    Additional config for Echo:
    expt = dict(
        num_pi: number of pi pulses
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        fname=None,
        progress=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        display=True,
        print=False,
        check_params=True,
    ):
        """
        Initialize the T2 experiment.

        This experiment measures T2 using Ramsey, Echo, or CPMG protocols.
        Default `params` values:
        - 'reps': Number of repetitions, doubled from default (default: `2 * self.reps`)
        - 'rounds': Number of software averages, from `self.rounds`
        - 'expts': Number of wait time points (default: 100)
        - 'span': Total span of wait times in µs, set to ~3xT2 (default: `3 * self.cfg.device.qubit[par][qi]`)
        - 'start': Start time for wait sweep in µs (default: 0.01)
        - 'ramsey_freq': Ramsey frequency for phase advancement, 'smart' sets it to 1.5/T2 (default: 'smart')
        - 'active_reset': If True, uses active reset (default: from `cfg.device.readout.active_reset[qi]`)
        - 'experiment_type': 'ramsey', 'echo', or 'cpmg' (default: 'ramsey')
        - 'acStark': If True, applies an AC Stark pulse during the wait time (Ramsey only) (default: False)
        - 'checkEF': If True, measures the |e>-|f> transition (default: False)
        - 'num_pi': Number of π pulses for Echo/CPMG (default: 1 for 'echo', 0 for 'ramsey')

        Args:
            cfg_dict (dict): Configuration dictionary.
            qi (int): Qubit index to measure.
            go (bool): Whether to immediately run the experiment.
            params (dict): Additional parameters to override defaults.
            prefix (str): Filename prefix for saved data.
            fname (str): Full filename for saved data.
            progress (bool): Whether to show a progress bar.
            style (str): Measurement style ('fine' for more averages, 'fast' for fewer points).
            disp_kwargs (dict): Display options.
            min_r2 (float): Minimum R² value for acceptable fit.
            max_err (float): Maximum error for acceptable fit.
            display (bool): Whether to display results.
            print (bool): If True, prints the experiment config and exits.
        """
        # Determine experiment type and parameter name based on protocol
        if "experiment_type" in params and params["experiment_type"] == "echo":
            par = "T2e"  # Echo uses T2e parameter
            name = "echo"
        else:
            par = "T2r"  # Ramsey uses T2r parameter
            name = "ramsey"

        # Set appropriate filename prefix
        if prefix is None:
            ef = "ef_" if "checkEF" in params and params["checkEF"] else ""
            prefix = f"{name}_{ef}qubit{qi}"

        # Initialize parent class
        super().__init__(
            cfg_dict=cfg_dict, prefix=prefix, fname=fname, progress=progress, qi=qi, check_params=check_params
        )

        # Define default parameters
        params_def = {
            "reps": 2 * self.reps,  # Number of repetitions (inner loop)
            "rounds": self.rounds,  # Number of averages (outer loop)
            "expts": 100,  # Number of wait time points
            "span": 3
            * self.cfg.device.qubit[par][
                qi
            ],  # Total span of wait times (μs), set to ~3*T2
            "start": 0.01,  # Start time for wait sweep (μs)
            "ramsey_freq": "smart",  # Ramsey frequency for phase advancement
            "active_reset": self.cfg.device.readout.active_reset[
                qi
            ],  # Use active qubit reset
            "qubit": [qi],  # Qubit index as a list
            "experiment_type": "ramsey",  # Default to Ramsey protocol
            "acStark": False,  # No AC Stark shift by default
            "checkEF": False,  # No EF transition check by default
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],  # Readout channel
            "flux": False,  # Whether to apply flux pulse during wait time
            "flux_chan": self.cfg.hw.soc.dacs.flux.ch[qi],  # DAC channel for flux pulse
            "flux_gain": 0.1,  # Gain for flux pulse
        }

        # Adjust parameters based on measurement style
        if style == "fine":
            params_def["rounds"] = (
                params_def["rounds"] * 2
            )  # Double averages for fine measurements
        elif style == "fast":
            params_def["expts"] = 50  # Fewer points for fast measurements

        # Merge user parameters with defaults
        params = {**params_def, **params}

        # Set Ramsey frequency intelligently if "smart" is specified
        if params["ramsey_freq"] == "smart":
            # Set Ramsey frequency to 1.5/T2 for optimal oscillation visibility
            params["ramsey_freq"] = 1.5 / self.cfg.device.qubit[par][qi]

        # Set number of π pulses based on experiment type
        if params["experiment_type"] == "echo":
            params_def["num_pi"] = 1  # Standard echo has 1 π pulse
        else:
            params_def["num_pi"] = 0  # Ramsey has 0 π pulses

        # Set pulse parameters based on transition type (g-e or e-f)
        if "checkEF" in params and params["checkEF"]:
            # For e-f transition measurements
            cfg_qub = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            # For g-e transition measurements (standard)
            cfg_qub = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]

        # Copy pulse parameters from configuration
        for key in cfg_qub:
            params_def[key] = cfg_qub[key][qi]

        # Final parameter merge and assignment
        self.cfg.expt = {**params_def, **params}

        # Check for unexpected parameters
        super().check_params(params_def)

        super().qubit_run(
            qi=qi,
            go=go,
            display=display,
            progress=progress,
            min_r2=min_r2,
            max_err=max_err,
            print=print,
            disp_kwargs=disp_kwargs,
        )

    def acquire(self, progress=False):
        """
        Acquire T2 measurement data.

        This method:
        1. Sets up the wait time sweep parameters
        2. Runs the T2Program to collect data for each wait time
        3. Adjusts x-axis values to account for echo protocol

        Args:
            progress: Whether to show progress bar

        Returns:
            Measurement data dictionary
        """
        # Define parameter metadata for plotting
        self.param = {"label": "wait", "param": "t", "param_type": "time"}

        # Ensure sweep step size is above DAC minimum resolution
        self.check_dac_timing(num_segments=self.cfg.expt.num_pi + 1)

        # Create a 1D sweep for the wait time from start to start+span
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        # Run the T2Program to acquire data

        super().acquire(T2Program, progress=progress)

        # Adjust x-axis values to account for echo protocol
        # For echo, the effective wait time is longer due to the π pulses
        if self.cfg.expt.num_pi == 0:
            coef = 1
        else:
            coef = 2*self.cfg.expt.num_pi  # For echo, we have num_pi + 1 segments
        self.data["xpts"] = coef * self.data["xpts"]

        return self.data

    def analyze(
        self,
        data=None,
        fit=True,
        fit_twofreq=False,
        refit=False,
        verbose=False,
        **kwargs,
    ):
        """
        Analyze T2 measurement data by fitting to a decaying sinusoid.

        This method:
        1. Fits the data to a decaying sinusoid (with or without slope)
        2. Calculates frequency errors from the fit
        3. Determines the corrected qubit frequency

        Args:
            data: Data dictionary to analyze (uses self.data if None)
            fit: Whether to perform fitting
            fit_twofreq: Whether to fit to a two-frequency model
            refit: Whether to refit without slope
            verbose: Whether to print detailed information
            **kwargs: Additional arguments passed to the analyzer

        Returns:
            Data dictionary with added fit results
        """
        if data is None:
            data = self.data

        # Define indices for fit parameters
        inds = [0, 1, 2, 3, 4]  # yscale, freq, phase_deg, decay, y0

        if fit:
            # Select appropriate fitting function based on parameters
            if fit_twofreq:
                # Two-frequency model for more complex oscillations
                self.fitterfunc = fitter.fittwofreq_decaysin
                self.fitfunc = fitter.twofreq_decaysin
            elif refit:
                # Simple decaying sine without slope
                self.fitfunc = fitter.decaysin
                self.fitterfunc = fitter.fitdecaysin
            else:
                # Decaying sine with slope (default)
                self.fitfunc = fitter.decayslopesin
                self.fitterfunc = fitter.fitdecayslopesin
                # Parameters: yscale, freq, phase_deg, decay, y0, slope

            # Perform the fit
            super().analyze(
                data=data,
                inds=inds,
                **kwargs,
            )

            # If the fit fails, try again without slope
            if not self.status and not refit:
                self.fitfunc = fitter.decaysin
                self.fitterfunc = fitter.fitdecaysin
                super().analyze(
                    data=data,
                    inds=inds,
                    **kwargs,
                )

            # Calculate average fit error
            inds = np.arange(5)
            data["fit_err"] = np.mean(np.abs(data["fit_err_par"][inds]))

            # Calculate frequency adjustments for each data type (amps, avgi, avgq)
            ydata_lab = ["amps", "avgi", "avgq"]
            for i, ydata in enumerate(ydata_lab):
                if isinstance(data["fit_" + ydata], (list, np.ndarray)):
                    # Calculate possible frequency errors
                    # The fitted frequency can be either ramsey_freq + fit_freq or ramsey_freq - fit_freq
                    data["f_adjust_ramsey_" + ydata] = sorted(
                        (
                            self.cfg.expt.ramsey_freq - data["fit_" + ydata][1],
                            self.cfg.expt.ramsey_freq + data["fit_" + ydata][1],
                        ),
                        key=abs,  # Sort by absolute value to get the smallest error first
                    )

                    # For two-frequency model, calculate additional adjustments
                    if fit_twofreq and self.cfg.expt.experiment_type == "ramsey":
                        data["f_adjust_ramsey_" + ydata + "2"] = sorted(
                            (
                                self.cfg.expt.ramsey_freq - data["fit_" + ydata][7],
                                self.cfg.expt.ramsey_freq - data["fit_" + ydata][6],
                            ),
                            key=abs,
                        )

            # Get the best frequency adjustment
            if not self.cfg.device.qubit.tuned_up[self.cfg.expt.qubit[0]]:
                # For untuned qubits, use a more sophisticated method to find the best fit
                fit_pars, fit_err, t2r_adjust, i_best = fitter.get_best_fit(
                    self.data, get_best_data_params=["f_adjust_ramsey"]
                )
            else:
                # For tuned qubits, use the I quadrature adjustment
                t2r_adjust = data["f_adjust_ramsey_avgi"]

            # Store the frequency adjustment
            data["t2r_adjust"] = t2r_adjust

            # Get the reference frequency based on transition type
            if self.cfg.expt.checkEF:
                f_pi_test = self.cfg.device.qubit.f_ef[self.cfg.expt.qubit[0]]
            else:
                f_pi_test = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]

            # Print possible frequency errors if verbose
            if self.cfg.expt.experiment_type == "ramsey" and verbose:
                print(
                    f"Possible errors are {t2r_adjust[0]:.3f} and {t2r_adjust[1]:.3f} MHz "
                    f"for Ramsey frequency {self.cfg.expt.ramsey_freq:.3f} MHz"
                )

            # Store the frequency error and corrected frequency
            data["f_err"] = t2r_adjust[0]  # Use the smallest error
            data["new_freq"] = (
                f_pi_test + t2r_adjust[0]
            )  # Calculate corrected frequency

        return data

    def display(
        self,
        data=None,
        fit=True,
        fit_twofreq=False,
        debug=False,
        plot_all=False,
        ax=None,
        savefig=True,
        refit=False,
        show_hist=False,
        rescale=False,
        **kwargs,
    ):
        """
        Display T2 measurement results.

        Creates a plot showing the qubit state vs wait time and the exponentially decaying sinusoidal fit.

        Args:
            data: Data dictionary to display (uses self.data if None)
            fit: Whether to show the fit curve
            fit_twofreq: Whether to use two-frequency model for display
            debug: Whether to show debug information
            plot_all: Whether to make plots for I/Q/Amps or just I
            ax: Matplotlib axis to plot on (creates one if None)
            savefig: Whether to save the figure to disk
            refit: Whether to use refit data for display
            show_hist: Whether to show histogram of the data
            **kwargs: Additional arguments passed to the display function
        """
        if data is None:
            data = self.data

        # Get qubit index for plot title
        q = self.cfg.expt.qubit[0]

        # Set experiment name based on type
        name = "Echo " if self.cfg.expt.experiment_type == "echo" else ""
        if self.cfg.expt.num_pi > 1:
            name += f"{self.cfg.expt.num_pi} π pulses "

        # Set x-axis label
        xlabel = "Wait Time ($\mu$s)"

        # Add EF prefix if checking EF transition
        ef = "EF " if self.cfg.expt.checkEF else ""

        # Create plot title
        title = f"{ef} Ramsey {name}Q{q} (Freq: {self.cfg.expt.ramsey_freq:.4} MHz)"

        # Set up caption parameters to display T2 and frequency values
        if self.cfg.expt.experiment_type == "echo":
            caption_params = [
                {"index": 3, "format": "$T_2$ : {val:.4} $\pm$ {err:.2g} $\mu$s"},
                {"index": 1, "format": "Freq. : {val:.3} $\pm$ {err:.1} MHz"},
            ]
        else:  # ramsey
            caption_params = [
                {
                    "index": 3,
                    "format": "$T_2$ : {val:.4} $\pm$ {err:.2g} $\mu$s",
                },
                {"index": 1, "format": "Freq. : {val:.3} $\pm$ {err:.1} MHz"},
            ]

        # Call parent class display method
        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            debug=debug,
            show_hist=show_hist,
            fitfunc=self.fitfunc,
            caption_params=caption_params,
            savefig=savefig,
            rescale=rescale,
            **kwargs,
        )


class T2FastFlux(QickExperiment2DSimple):
    """
    T2FastFlux: Sweep T2 across flux gain values.

    Runs a T2Experiment at each flux gain point, collecting T2 vs flux gain (and
    optionally vs frequency). Saves a single HDF5 file with save_interim after
    each gain point. Performs adaptive span adjustment between scans.

    Automatically reads quadratic fit coefficients (quad_a, quad_b, quad_c) and
    sweet_spot_ac from the config (saved by QubitSpecFastFlux) when available.
    The gain sweep starts at sweet_spot_ac by default.

    Sweep parameters (passed via params dict):
        gain_start: Starting flux gain value (default: sweet_spot_ac)
        gain_stop: Ending flux gain value (default: sweet_spot_ac + 0.4)
        direction: 'pos' or 'neg' — sweep direction from sweet spot (default: 'pos').
            Sets gain_stop relative to sweet spot if gain_stop not explicitly given.
        freq_span: Frequency span in MHz. When provided with freq_coeffs, converts
            to a gain range using the quadratic. Overrides gain_stop.
        expts_gain: Number of gain points in the sweep
        freq_coeffs: Optional (a, b, c) tuple for quadratic freq = a*g^2 + b*g + c.
            If not provided, reads quad_a/b/c from config.
        lin_freq: If True and freq_coeffs provided, space points linearly in
            frequency (inverting the quadratic) instead of linearly in gain.
        experiment_type: 'ramsey' or 'echo' (default: 'ramsey')
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=False,
        display=True,
    ):
        if prefix is None:
            prefix = f"t2_fastflux_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, qi=qi)

        # Read quadratic coefficients and sweet spot from config if available
        cfg_flux = getattr(getattr(getattr(getattr(self.cfg, "hw", None), "soc", None), "dacs", None), "flux", None)
        cfg_qubit = getattr(getattr(self.cfg, "device", None), "qubit", None)

        config_freq_coeffs = None
        sweet_spot = 0.0
        if cfg_flux is not None:
            a = cfg_flux.quad_a[qi] if hasattr(cfg_flux, "quad_a") else 0
            b = cfg_flux.quad_b[qi] if hasattr(cfg_flux, "quad_b") else 0
            c = cfg_flux.quad_c[qi] if hasattr(cfg_flux, "quad_c") else 0
            if a != 0 or b != 0:
                config_freq_coeffs = (a, b, c)
        if cfg_qubit is not None and hasattr(cfg_qubit, "sweet_spot_ac"):
            sweet_spot = cfg_qubit.sweet_spot_ac[qi]

        # Determine sweep direction and gain range
        direction = params.get("direction", "pos")
        sign = 1 if direction == "pos" else -1
        freq_coeffs = params.get("freq_coeffs", config_freq_coeffs)
        freq_span = params.get("freq_span", None)

        # Convert freq_span to gain_stop using the quadratic
        if freq_span is not None and freq_coeffs is not None and "gain_stop" not in params:
            a, b, c = freq_coeffs
            f_sweet = a * sweet_spot**2 + b * sweet_spot + c
            f_target = f_sweet - freq_span  # frequency decreases away from sweet spot
            discriminant = b**2 - 4 * a * (c - f_target)
            if discriminant >= 0:
                root_pos = (-b + np.sqrt(discriminant)) / (2 * a)
                root_neg = (-b - np.sqrt(discriminant)) / (2 * a)
                if sign > 0:
                    gain_stop = max(root_pos, root_neg)
                else:
                    gain_stop = min(root_pos, root_neg)
            else:
                gain_stop = sweet_spot + sign * 0.4
        elif "gain_stop" not in params:
            gain_stop = sweet_spot + sign * 0.4
        else:
            gain_stop = params["gain_stop"]

        # start_t2: assumed T2 for the first gain point, sets initial span
        start_t2 = params.pop("start_t2", None)

        # Define default parameters for the 2D sweep
        params_def = {
            "gain_start": sweet_spot,
            "gain_stop": gain_stop,
            "start": 0.05,
            "expts": 25,
            "expts_gain": 50,
            "freq_coeffs": freq_coeffs,
            "freq_span": freq_span,
            "direction": direction,
            "lin_freq": True,
            "t2_max": float("inf"),  # Upper bound on T2 when setting next span
        }

        if start_t2 is not None:
            params_def["span"] = 4.1 * start_t2
        params = {**params_def, **params}
        # Create inner T2 experiment (go=False) to inherit its config
        self.expt = T2Experiment(cfg_dict, qi, go=False, params=params, check_params=False)
        params = {**self.expt.cfg.expt, **params, "flux": True}
        self.cfg.expt = {**params_def, **params}

        # Store references
        self._qi = qi
        self._cfg_dict = cfg_dict

        # Build gain_pts and freq_pts from config
        cfg_e = self.cfg.expt
        freq_coeffs = cfg_e["freq_coeffs"]

        if cfg_e["lin_freq"] and freq_coeffs is not None:
            # Linearly spaced in frequency, invert quadratic to get gains
            a, b, c = freq_coeffs
            freq_start = a * cfg_e["gain_start"]**2 + b * cfg_e["gain_start"] + c
            freq_stop = a * cfg_e["gain_stop"]**2 + b * cfg_e["gain_stop"] + c
            self.freq_pts = np.linspace(freq_start, freq_stop, cfg_e["expts_gain"])
            discriminant = b**2 - 4 * a * (c - self.freq_pts)
            root_pos = (-b + np.sqrt(discriminant)) / (2 * a)
            root_neg = (-b - np.sqrt(discriminant)) / (2 * a)
            if cfg_e["direction"] == "pos":
                self.gain_pts = np.maximum(root_pos, root_neg)
            else:
                self.gain_pts = np.minimum(root_pos, root_neg)
        else:
            # Linearly spaced in gain
            self.gain_pts = np.linspace(cfg_e["gain_start"], cfg_e["gain_stop"], cfg_e["expts_gain"])
            if freq_coeffs is not None:
                a, b, c = freq_coeffs
                self.freq_pts = a * self.gain_pts**2 + b * self.gain_pts + c
            else:
                self.freq_pts = None

        if go:
            self.acquire(progress=progress, display=display)
            self.display()

    def acquire(self, progress=False, display=True):
        data = {"time": []}
        t2_list = []
        offset_list = []
        amp_list = []
        q_offset_list = []
        q_amp_list = []

        for i, g in enumerate(tqdm(self.gain_pts, disable=not progress)):
            # Update inner experiment config for this gain point
            self.expt.cfg.expt["flux_gain"] = float(g)

            # Run inner T2 scan and analyze
            data_new = self.expt.acquire(progress=False)
            self.expt.analyze(data=data_new)
            if display:
                self.expt.display(data=data_new)

            # Accumulate raw 2D data
            for key in ("avgi", "avgq", "amps", "phases", "xpts"):
                if key in data_new:
                    if i == 0:
                        data[key] = []
                    data[key].append(data_new[key])
            data["time"].append(time.time())

            # T2 fit: [yscale, freq, phase_deg, decay, y0] — T2 is at index 3
            best_fit = data_new.get("best_fit", [np.nan, np.nan, np.nan, np.nan, np.nan])
            t2_val = best_fit[3]
            t2_list.append(t2_val)
            offset_list.append(best_fit[4])
            amp_list.append(best_fit[0])
            fit_q = data_new.get("fit_avgq", [np.nan, np.nan, np.nan, np.nan, np.nan])
            q_offset_list.append(fit_q[4] if len(fit_q) > 4 else np.nan)
            q_amp_list.append(fit_q[0] if len(fit_q) > 0 else np.nan)

            # Adjust span for next scan based on extracted T2
            if np.isfinite(t2_val) and t2_val > 0:
                self.expt.cfg.expt["span"] = 4.1 * min(
                    t2_val, self.cfg.expt.get("t2_max", float("inf"))
                )

            # Store analysis summaries in data for interim save
            data["gain_pts"] = self.gain_pts
            if self.freq_pts is not None:
                data["freq_pts"] = self.freq_pts
            data["t2_list"] = np.array(t2_list)
            data["offset_list"] = np.array(offset_list)
            data["amp_list"] = np.array(amp_list)
            data["q_offset_list"] = np.array(q_offset_list)
            data["q_amp_list"] = np.array(q_amp_list)

            if self.save_interim:
                self.save_data(data=data)

        # Convert lists to arrays
        for k, a in data.items():
            data[k] = np.array(a)
        self.data = data
        return data

    def display(self, data=None, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        gain_pts = data["gain_pts"]
        t2_list = data["t2_list"]

        fig, ax = plt.subplots()
        ax.plot(gain_pts, t2_list, ".-")
        ax.set_xlabel("Flux Gain")
        ax.set_ylabel("$T_2$ ($\mu$s)")
        ax.set_title(f"$T_2$ vs Flux Gain Q{self._qi}")
        self.save_fig(fig, suffix="_t2_vs_gain")

        if "freq_pts" in data:
            freq_pts = data["freq_pts"]

            fig, ax = plt.subplots()
            ax.plot(freq_pts, t2_list, ".-")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("$T_2$ ($\mu$s)")
            ax.set_title(f"$T_2$ vs Frequency Q{self._qi}")
            self.save_fig(fig, suffix="_t2_vs_freq")

            fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
            panels = [
                (axes[0, 0], data["amp_list"], "Amplitude"),
                (axes[0, 1], data["offset_list"], "Offset"),
                (axes[1, 0], data["q_amp_list"], "Q Amplitude"),
                (axes[1, 1], data["q_offset_list"], "Q Offset"),
            ]
            for ax, vals, label in panels:
                vals = np.array(vals, dtype=float)
                mask = np.isfinite(vals)
                if mask.sum() > 2:
                    q1, q3 = np.percentile(vals[mask], [25, 75])
                    iqr = q3 - q1
                    inlier = mask & (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
                    ax.plot(freq_pts[inlier], vals[inlier], ".-")
                    ax.plot(freq_pts[~inlier & mask], vals[~inlier & mask], "rx", ms=6)
                else:
                    ax.plot(freq_pts, vals, ".-")
                ax.set_ylabel(label)
            axes[1, 0].set_xlabel("Frequency (MHz)")
            axes[1, 1].set_xlabel("Frequency (MHz)")
            self.save_fig(fig, suffix="_fit_params")
