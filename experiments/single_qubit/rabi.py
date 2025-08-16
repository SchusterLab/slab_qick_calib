"""
Rabi Oscillation Experiment

This module implements Rabi oscillation experiments for qubit characterization.
Rabi oscillations are observed by varying either the amplitude or length of a driving pulse
and measuring the resulting qubit state. This allows determination of the π-pulse parameters
(amplitude and duration) needed for qubit control.

The module includes:
- RabiProgram: Defines the pulse sequence for the Rabi experiment
- RabiExperiment: Main experiment class for amplitude or length Rabi oscillations
- ReadoutCheck: Class for checking readout parameters
- RabiChevronExperiment: 2D version that sweeps both frequency and amplitude/length
- Rabi2D: 2D version that sweeps both length and gain
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from qick import *
from qick.asm_v2 import QickSweep1D

from ...analysis import fitting as fitter
from ..general.qick_experiment import (
    QickExperiment,
    QickExperiment2DSimple,
    QickExperimentLoop,
)
from ..general.qick_program import QickProgram

from ...exp_handling.datamanagement import AttrDict

# ====================================================== #


class RabiProgram(QickProgram):
    """
    Defines the pulse sequence for a Rabi oscillation experiment.

    The sequence consists of:
    1. Optional π pulse on |g>-|e> transition (if checking EF transition)
    2. Variable amplitude/length pulse on the qubit
    3. Optional second π pulse on |g>-|e> transition
    4. Optional wait time
    5. Measurement
    """

    def __init__(self, soccfg, final_delay, cfg):
        """
        Initialize the Rabi program.

        Args:
            soccfg: SOC configuration
            final_delay: Delay time after measurement
            cfg: Configuration dictionary
        """
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """
        Initialize the program with the necessary pulses and loops.

        Args:
            cfg: Configuration dictionary containing experiment parameters
        """
        cfg = AttrDict(self.cfg)
        q = cfg.expt.qubit[0]

        # Initialize with standard readout
        super()._initialize(cfg, readout="standard")

        # Add sweep loop for the experiment
        self.add_loop("sweep_loop", cfg.expt.expts)
    
        # Define the qubit pulse with parameters from config
        if cfg.expt.type not in ["flat_top"]:
            pulse = {
                "sigma": cfg.expt.sigma,
                "length": cfg.expt.length,
                "freq": cfg.expt.freq,
                "gain": cfg.expt.gain,
                "phase": 0,
                "type": cfg.expt.type,
            }
        else:
            pulse = {
                "length": cfg.expt.length,
                "freq": cfg.expt.freq,
                "gain": cfg.expt.gain,
                "phase": 0,
                "type": cfg.expt.type,
                "ramp_sigma": cfg.expt.ramp_sigma,
                "ramp_sigma_inc": cfg.expt.ramp_sigma_inc,
            }
        super().make_pulse(pulse, "qubit_pulse")

        # If checking EF transition and using ge pulse, create a pi pulse
        if (cfg.expt.checkEF and cfg.expt.pulse_ge) or cfg.expt.active_reset:
            super().make_pi_pulse(q, cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        """
        Define the main body of the experiment sequence.

        Args:
            cfg: Configuration dictionary containing experiment parameters
        """
        cfg = AttrDict(self.cfg)

        # Configure readout
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # If checking EF transition with ge pulse, apply first pi pulse
        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef")

        # Apply the main qubit pulse (variable amplitude or length)
        for i in range(cfg.expt.n_pulses):
            self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
            self.delay_auto(t=0.01)

        # If checking EF transition with ge pulse, apply second pi pulse
        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef 2")

        # Add optional end wait time
        if "end_wait" in cfg.expt:
            self.delay_auto(t=cfg.expt.end_wait, tag="end_wait")

        # Perform measurement
        super().measure(cfg)


class RabiExperiment(QickExperiment):
    """
    Main experiment class for Rabi oscillations.

    This class implements Rabi oscillation experiments by sweeping either the amplitude
    or length of a driving pulse and measuring the resulting qubit state. The oscillation
    pattern allows determination of the π-pulse parameters needed for qubit control.

    Parameters:
    - 'expts': Number of experiments to run (default: 60)
    - 'reps': Number of repetitions for each experiment (default: self.reps)
    - 'rounds': Number of rounds for each experiment (default: self.rounds)
    - 'gain': Gain value for pi pulse (default: cfg pi pulse gain)
    - 'max_gain': Maximum gain used in amplitude sweep (if specified, used instead of gain)
    - 'sigma': Standard deviation of the Gaussian pulse (default: cfg pip pulse sigma)
    - 'max_length': Maximum length used in length sweep (if specified, used instead of sigma)
    - 'checkEF': Boolean flag to check EF interaction (default: False)
    - 'pulse_ge': Boolean flag to indicate if pulse is for ground to excited state transition (default: True)
    - 'start': Starting point for the experiment, either in time or gain (default: as close to 0 as is linear/allowed)
    - 'step': Step size for the gain (calculated as int(params['gain']/params['expts']))
    - 'qubit': List of qubits involved in the experiment (default: [qi])
    - 'pulse_type': Type of pulse used in the experiment (default: 'gauss')
    - 'num_pulses': Number of pulses used in the experiment (default: 1)
    - 'qubit_chan': Channel for the qubit readout (default: self.cfg.hw.soc.adcs.readout.ch[qi])
    - 'sweep': Type of sweep to perform ('amp' or 'length') (default: 'amp')
    - 'freq': Frequency of the qubit pulse (default: self.cfg.device.qubit.f_ge[qi])

    Additional keys may be added based on the specific requirements of the experiment.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=True,
        display=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        print=False,
        check_params=True,
    ):
        """
        Initialize the Rabi experiment.

        This experiment sweeps the amplitude or length of a pulse to observe Rabi oscillations.
        Default `params` values:
        - 'expts': Number of points in the sweep (default: 60)
        - 'reps': Number of repetitions, from `self.reps`
        - 'rounds': Number of software averages, from `self.rounds`
        - 'checkEF': If True, performs the Rabi experiment on the |e>-|f> transition (default: False)
        - 'pulse_ge': If True, applies a π-pulse on |g>-|e> before the EF Rabi pulse (default: True)
        - 'num_osc': Try to set max gain or length for this number of oscillations (default: 2.5)
        - 'n_pulses': Number of Rabi pulses to apply (default: 1)
        - 'sweep': Type of sweep, 'amp' or 'length' (default: 'amp')
        - 'active_reset': If True, uses active reset (default: from `cfg.device.readout.active_reset[qi]`)
        - 'loop': If True, uses loop-based acquisition (default: False)
        - 'temp': Temperature parameter for temperature-dependent measurements (default: 40)

        Pulse parameters (`sigma`, `gain`, `type`, etc.) are taken from the device config for either
        the |g>-|e> or |e>-|f> transition, depending on `checkEF`.

        Args:
            cfg_dict (dict): Configuration dictionary.
            qi (int): Qubit index.
            go (bool): Whether to run the experiment immediately.
            params (dict): Additional parameters to override defaults.
            prefix (str): Prefix for data files.
            progress (bool): Whether to show a progress bar.
            display (bool): Whether to display results.
            style (str): Experiment style ('fine', 'fast', 'temp').
            disp_kwargs (dict): Display keyword arguments.
            min_r2 (float): Minimum R² value for fit quality.
            max_err (float): Maximum error for fit quality.
            print (bool): If True, prints the experiment config and exits.
        """

        if "checkEF" in params and params["checkEF"]:
            if "pulse_ge" in params and not params["pulse_ge"]:
                ef = "ef_no_ge_"
            else:
                ef = "ef_"
        else:
            ef = ""
        name = "length" if "sweep" in params and params["sweep"] == "length" else "amp"

        prefix = f"{name}_rabi_{ef}qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, check_params=check_params)
        params_def = {
            "expts": 60,
            "reps": self.reps,
            "rounds": self.rounds,
            "checkEF": False,
            "pulse_ge": True,
            "num_osc": 2.5,
            "n_pulses": 1,
            "sweep": "amp",
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "loop": False,
            "temp": 40,
        }

        min_gain = 2**-15  # Minimum DAC gain value for linear operation

        # Apply style modifications for different experiment modes
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 25

        params = {**params_def, **params}

        # Configure pulse parameters based on transition type (GE or EF)
        if params["checkEF"]:
            qubit_pulse_config = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            qubit_pulse_config = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]
        # Copy pulse parameters from device config, usually contains sigma, gain, sigma_inc, type
        for key in qubit_pulse_config:
            params_def[key] = qubit_pulse_config[key][qi]

        # Override pulse type if specified
        if "pulse_type" in params:
            params_def["type"] = params["pulse_type"]
        params = {**params_def, **params}

        # Configure sweep range based on sweep type
        if params["sweep"] == "amp":
            if params["n_pulses"] == 1:
                # Amplitude sweep: set max gain to cover desired number of oscillations
                params_def['max_gain'] = params['gain'] * params['num_osc'] * 2
                params_def['start'] = 0.003  # Minimum gain value that maintains linearity
            else:
                params_def["max_gain"]=params["gain"](1+1/ params["n_pulses"])
                params_def["start"]=params["gain"](1-1/ params["n_pulses"])
            params_def["max_gain"] = np.min([params_def["max_gain"], self.cfg.device.qubit.max_gain]) # Do not exceed max_gain of RFSoC
        elif params["sweep"] == "length":
            # Length sweep: set max length to cover desired number of oscillations
            params_def["max_length"] = 2 * params["num_osc"] * params["sigma"]
            params_def["start"] = 3 * cfg_dict["soc"].cycles2us(
                1
            )  # Minimum allowed length

        # Special configuration for temperature-dependent measurements
        if style == "temp":
            params["reps"] = int(40 * params["reps"])
            params["rounds"] = int(
                np.ceil(20 * params["rounds"] * 1.5 ** (params["temp"] / 40))
            )
            params["pulse_ge"] = False

        self.cfg.expt = {**params_def, **params}

        # Ensure minimum gain spacing for amplitude sweeps to avoid DAC resolution issues
        if params["sweep"] == "amp":
            gain_spacing = self.cfg.expt["max_gain"] / self.cfg.expt["expts"]
            if gain_spacing < min_gain:
                self.cfg.expt["max_gain"] = min_gain * self.cfg.expt["expts"]
        super().check_params(params_def)
        if go:
            super().qubit_run(
                qi=qi,
                display=display,
                progress=progress,
                min_r2=min_r2,
                max_err=max_err,
                print=print,
                disp_kwargs=disp_kwargs,
            )

    def acquire(self, progress=False, debug=False):
        """
        Acquire data for the Rabi experiment.

        Args:
            progress: Whether to show progress bar
            debug: Whether to print debug information

        Returns:
            Acquired data
        """
        self.qubit = self.cfg.expt.qubit

        # Configure the sweep based on whether we're sweeping amplitude or length
        # Note: 2d scans will break if you use gain/length to store the max_vals of gain/length when making qicksweep
        # Pulse definition takes length and sigma as parameters; which works for gauss and const pulses.
        if self.cfg.expt.sweep == "amp":
            # Amplitude sweep configuration
            param_pulse = "gain"  # Parameter used to get xvals from QICK
            self.cfg.expt["gain"] = QickSweep1D(
                "sweep_loop", self.cfg.expt.start, self.cfg.expt["max_gain"]
            )
            if self.cfg.expt.type == "gauss":
                self.cfg.expt["length"] = self.cfg.expt.sigma * self.cfg.expt.sigma_inc
            elif self.cfg.expt.type == "const":
                if "length" not in self.cfg.expt:
                    self.cfg.expt["length"] = self.cfg.expt.sigma
            elif self.cfg.expt.type == "flat_top":
                pass
                #self.cfg.expt["length"] = self.cfg.expt.sigma
        elif self.cfg.expt.sweep == "length":
            # Length sweep configuration
            param_pulse = "total_length"  # Parameter used to get xvals from QICK
            if (self.cfg.expt.type == "gauss"):  
                # This does not work with QICK sweeps right now
                # For Gaussian pulses, sweep sigma
                par = "sigma"
                # self.cfg.expt['length'] = QickSweep1D(
                #     "sweep_loop", self.cfg.expt.start, 4*self.cfg.expt['max_length'])
            else:
                # For other pulse types, sweep length directly
                par = "length"

            self.cfg.expt[par] = QickSweep1D(
                "sweep_loop", self.cfg.expt.start, self.cfg.expt.max_length
            )

        # Set the parameter to sweep
        self.param = {
            "label": "qubit_pulse",
            "param": param_pulse,
            "param_type": "pulse",
        }

        # Acquire data using the RabiProgram
        if not self.cfg.expt.loop:
            super().acquire(RabiProgram, progress=progress)
        else:
            # Loop acquisition with custom frequency points
            cfg_dict = {
                "soc": self.soccfg,
                "cfg_file": self.config_file,
                "im": self.im,
                "expt_path": "dummy",
            }
            exp = QickExperimentLoop(
                cfg_dict=cfg_dict,
                prefix="dummy",
                progress=progress,
                qi=self.cfg.expt.qubit[0],
            )
            exp.cfg.expt = copy.deepcopy(self.cfg.expt)
            exp.param = self.param

            len_pts = np.linspace(
                self.cfg.expt.start, self.cfg.expt.max_length, self.cfg.expt.expts
            )
            exp.cfg.expt["sigma"] = len_pts
            # Set up experiment with single point per loop
            exp.cfg.expt.expts = 1
            x_sweep = [
                {"pts": len_pts, "var": "sigma"},
                {
                    "pts": len_pts
                    * self.cfg.device.qubit.pulses.pi_ge.sigma_inc[
                        self.cfg.expt.qubit[0]
                    ],
                    "var": "length",
                },
            ]

            # Acquire data
            data = exp.acquire(RabiProgram, x_sweep, progress=progress)
            self.data = data

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data to extract Rabi oscillation parameters.

        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data to a sinusoidal function
            **kwargs: Additional arguments for the fit

        Returns:
            Analyzed data with fit parameters and π-pulse length
        """
        if data is None:
            data = self.data

        if fit:
            # Fit the data to a sinusoidal function
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset]
            self.fitterfunc = fitter.fitsin
            self.fitfunc = fitter.sinfunc
            data = super().analyze(
                fitfunc=self.fitfunc, fitterfunc=self.fitterfunc, fit=fit, **kwargs
            )

        # Calculate π-pulse length from the fit for each data type
        ydata_lab = ["amps", "avgi", "avgq"]
        for ydata in ydata_lab:
            pi_length = fitter.fix_phase(data["fit_" + ydata])
            data["pi_length_" + ydata] = pi_length
        data["pi_length_scale_data"] = data["pi_length_avgi"]

        # Store the best π-pulse length
        data["pi_length"] = fitter.fix_phase(data["best_fit"])
        return data

    def display(
        self,
        data=None,
        fit=True,
        plot_all=False,
        ax=None,
        show_hist=False,
        rescale=False,
        **kwargs,
    ):
        """
        Display the results of the Rabi experiment.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit curve
            plot_all: Whether to plot all data types (I, Q, amplitude)
            ax: Matplotlib axis to plot on
            show_hist: Whether to show histogram
            rescale: Whether to rescale the plot
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data

        # Set up plot title and labels based on sweep type
        q = self.cfg.expt.qubit[0]
        if self.cfg.expt.sweep == "amp":
            title = "Amplitude"
            xlabel = "Gain / Max Gain"
            if self.cfg.expt.type == "gauss":
                param = "sigma"
            elif self.cfg.expt.type == "const":
                param = "length"
            elif self.cfg.expt.type == "flat_top":
                param = "length"
        else:
            title = "Length"
            param = "gain"
            xlabel = "Pulse Length ($\mu$s)"

        title += f" Rabi Q{q} (Pulse {param} {self.cfg.expt[param]}"

        # Set up fit function and caption parameters
        caption_params = [{"index": "pi_length", "format": "$\pi$ length: {val:.3f}"}]

        # Add EF indicator to title if applicable
        if self.cfg.expt.checkEF:
            title = title + ", EF)"
        else:
            title = title + ")"

        # Display the results
        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=show_hist,
            fitfunc=self.fitfunc,
            caption_params=caption_params,
            rescale=rescale,
        )


class ReadoutCheck(QickExperiment):
    """
    Class for checking readout parameters.

    This experiment is used to characterize the readout by sweeping either
    the end wait time or the gain of a qubit pulse and measuring the response.
    It uses the same RabiProgram as the RabiExperiment but with different parameters.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=True,
        display=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
    ):
        """
        Initialize the ReadoutCheck experiment.

        This experiment checks readout parameters by sweeping either the end wait time or pulse gain.
        Default `params` values:
        - 'expts': Number of points in the sweep (default: 30)
        - 'reps': Number of repetitions, multiplied by 5 (default: `5 * self.reps`)
        - 'rounds': Number of software averages, from `self.rounds`
        - 'checkEF': If True, checks the |e>-|f> transition (default: False)
        - 'pulse_ge': If True, applies a |g>-|e> pulse (default: True)
        - 'active_reset': If True, uses active reset (default: from `cfg.device.readout.active_reset[qi]`)
        - 'df': Frequency detuning in MHz (default: 50)
        - 'qubit_freq': Qubit frequency, from `cfg.device.qubit.f_ge[qi]`
        - 'length': Pulse length in µs (default: 10)
        - 'gain': Pulse gain (default: 0.5)
        - 'type': Pulse type, 'const' (default: 'const')
        - 'max_wait': Maximum wait time for 'end_wait' sweep (default: 10)
        - 'max_gain': Maximum gain for 'gain' sweep (default: 1)
        - 'end_wait': Initial end wait time (default: 0.2)
        - 'start': Start value for the sweep (default: 0)
        - 'expt_type': Type of sweep, 'end_wait' or 'gain' (default: 'end_wait')

        Args:
            cfg_dict (dict): Configuration dictionary.
            qi (int): Qubit index.
            go (bool): Whether to run the experiment immediately.
            params (dict): Additional parameters to override defaults.
            prefix (str): Prefix for data files.
            progress (bool): Whether to show a progress bar.
            display (bool): Whether to display results.
            style (str): Experiment style ('fine' or 'fast').
            disp_kwargs (dict): Display keyword arguments.
            min_r2 (float): Minimum R² value for fit quality.
            max_err (float): Maximum error for fit quality.
        """
        # Set the prefix for data files
        prefix = f"'readout_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        # Default parameters
        params_def = {
            "expts": 30,
            "reps": 5 * self.reps,
            "rounds": self.rounds,
            "checkEF": False,
            "pulse_ge": True,
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "df": 50,
            "qubit_freq": self.cfg.device.qubit.f_ge[qi],
            "phase": 0,
            "length": 10,
            "gain": 0.5,
            "type": "const",
            "max_wait": 10,
            "max_gain": 1,
            "end_wait": 0.2,
            "start": 0,
            "expt_type": "end_wait",  # Can be 'end_wait' or 'gain'
        }

        min_gain = 2**-15

        # Apply style modifications
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 25

        # Merge default and user-provided parameters
        params = {**params_def, **params}
        params["sigma"] = params["length"]  # Set sigma equal to length
        params_def["freq"] = (
            params["qubit_freq"] + params["df"]
        )  # Set frequency with offset

        # Set experiment configuration
        self.cfg.expt = {**params_def, **params}

        # Check parameters and configure reset if needed
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()

        # Set display parameters for untuned qubits
        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {"plot_all": True}

        # Run the experiment if requested
        if go:
            super().run(
                display=display,
                progress=progress,
                min_r2=min_r2,
                max_err=max_err,
                disp_kwargs=disp_kwargs,
            )

    def acquire(self, progress=False, debug=False, single=False):
        """
        Acquire data for the ReadoutCheck experiment.

        Args:
            progress: Whether to show progress bar
            debug: Whether to print debug information
            single: Whether to run a single acquisition

        Returns:
            Acquired data
        """
        self.qubit = self.cfg.expt.qubit

        # Configure the sweep based on experiment type
        if self.cfg.expt.expt_type == "end_wait":
            # Sweep end wait time
            self.cfg.expt["end_wait"] = QickSweep1D(
                "sweep_loop", self.cfg.expt.start, self.cfg.expt["max_wait"]
            )
            self.param = {"label": "end_wait", "param": "t", "param_type": "time"}
        else:
            # Sweep gain
            self.cfg.expt["gain"] = QickSweep1D(
                "sweep_loop", self.cfg.expt.start, self.cfg.expt["max_gain"]
            )
            self.param = {
                "label": "qubit_pulse",
                "param": "gain",
                "param_type": "pulse",
            }

        # Acquire data using the RabiProgram
        super().acquire(RabiProgram, progress=progress, single=single)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data.

        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data
            **kwargs: Additional arguments for the fit

        Returns:
            Analyzed data
        """
        if data is None:
            data = self.data
        # No specific analysis for ReadoutCheck, just return the data
        return data

    def display(self, data=None, fit=False, plot_all=False, **kwargs):
        """
        Display the results of the ReadoutCheck experiment.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_all: Whether to plot all data types
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data

        # Use the parent class display method
        super().display(data=data, fit=fit, plot_all=plot_all, **kwargs)


class RabiChevronExperiment(QickExperiment2DSimple):
    """
    2D Rabi experiment that sweeps both frequency and amplitude/length.

    This experiment performs a 2D sweep of both qubit frequency and pulse amplitude/length
    to map out the Rabi chevron pattern. This allows visualization of how the Rabi
    oscillation frequency changes with detuning from the qubit frequency.

    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz),
        step_f: frequency step (MHz),
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        style="",
        prefix=None,
        progress=False,
    ):
        """
        Initialize the RabiChevronExperiment.

        This experiment performs a 2D sweep of frequency and either pulse amplitude or length
        to generate a Rabi chevron plot.
        Default `params` values:
        - 'span_f': Frequency span in MHz (default: 20)
        - 'expts_f': Number of frequency points (default: 30)
        - 'sweep': Type of sweep for the inner loop, 'amp' or 'length' (default: 'amp')

        The start frequency `start_f` is calculated based on the qubit frequency (`f_ge` or `f_ef`)
        and the frequency span.

        Args:
            cfg_dict (dict): Configuration dictionary.
            qi (int): Qubit index.
            go (bool): Whether to run the experiment immediately.
            params (dict): Additional parameters to override defaults.
            style (str): Experiment style.
            prefix (str): Prefix for data files.
            progress (bool): Whether to show a progress bar.
        """
        # Determine prefix based on parameters
        if "type" in params:
            pre = params["type"]
        else:
            pre = "amp"
        if "checkEF" in params and params["checkEF"]:
            ef = "ef_"
        else:
            ef = ""
        prefix = f"{pre}_rabi_chevron_{ef}qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        # Default parameters
        params_def = {"span_f": 20, "expts_f": 30, "sweep": "amp"}
        params = {**params_def, **params}

        # Set frequency range based on whether we're checking EF transition
        if "checkEF" in params and params["checkEF"]:
            params_def["start_f"] = (
                self.cfg.device.qubit.f_ef[qi] - params["span_f"] / 2
            )
        else:
            params_def["start_f"] = (
                self.cfg.device.qubit.f_ge[qi] - params["span_f"] / 2
            )

        # Create a RabiExperiment instance but don't run it yet
        self.expt = RabiExperiment(
            cfg_dict, qi=qi, go=False, params=params, style=style, check_params=False
        )
        params = {**params_def, **params}
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        # Run the experiment if requested
        if go:
            super().run(progress=progress)

    def acquire(self, progress=False, debug=False):
        """
        Acquire data for the RabiChevronExperiment.

        Args:
            progress: Whether to show progress bar
            debug: Whether to print debug information

        Returns:
            Acquired data
        """
        # Create frequency points for the sweep
        freqpts = np.linspace(
            self.cfg.expt["start_f"],
            self.cfg.expt["start_f"] + self.cfg.expt["span_f"],
            self.cfg.expt["expts_f"],
        )

        # Set up the y-sweep (frequency sweep)
        ysweep = [{"pts": freqpts, "var": "freq"}]

        # Acquire data
        super().acquire(ysweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data.

        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data
            **kwargs: Additional arguments for the fit

        Returns:
            Analyzed data with fit parameters
        """
        if data is None:
            data = self.data

        if fit:
            # Fit the data to a sinusoidal function for each frequency
            fitterfunc = fitter.fitsin
            fitfunc = fitter.sinfunc
            data = super().analyze(
                fitfunc=fitfunc, fitterfunc=fitterfunc, fit=fit, **kwargs
            )

            # Extract qubit frequency and fit parameters
            qubit_freq = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            freq = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]
            amp = [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))]
            data["chevron_freqs"] = freq
            data["chevron_amps"] = amp

            data["best_freq"] = data["freq_pts"][np.argmax(data["chevron_amps"])]
            # Fit the chevron pattern (for length rabi)
            try:
                p, _ = curve_fit(chevron_freq, data["ypts"] - qubit_freq, freq)
                p2, _ = curve_fit(chevron_amp, data["ypts"] - qubit_freq, amp)
                data["chevron_freq"] = p
                data["chevron_amp"] = p2
            except:
                # Silently fail if the fit doesn't converge
                pass

        return data

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        """
        Display the results of the RabiChevronExperiment.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_both: Whether to plot both amplitude and phase
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data

        # Set up plot title and labels
        if self.cfg.expt.checkEF:
            title = "EF"
        else:
            title = ""

        if self.cfg.expt.sweep == "amp":
            title = "Amplitude"
            unit='µs'
            if self.cfg.expt.type == "gauss":
                param = "sigma"
            elif self.cfg.expt.type == "const":
                param = "length"
            elif self.cfg.expt.type == "flat_top":
                param = "length"
            xlabel = "Gain / Max Gain"
        else:
            title = "Length"
            unit = ''
            param = "gain"
            xlabel = "Pulse Length ($\mu$s)"

        title += (
            f" Rabi Q{self.cfg.expt.qubit[0]} (Pulse {param} {self.cfg.expt[param]} {unit})"
        )

        xlabel = xlabel
        ylabel = "Frequency (MHz)"

        # Display the 2D plot
        super().display(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            data=data,
            fit=fit,
            plot_both=plot_both,
            **kwargs,
        )

        # If fit is enabled, also display the frequency and amplitude vs. detuning
        if fit:
            fig, ax = plt.subplots(2, 1, figsize=(6, 6))
            qubit_freq = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            freq = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]
            amp = [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))]

            # Plot frequency vs. detuning
            ax[0].plot(data["ypts"] - qubit_freq, freq)
            ax[0].set_ylabel("Frequency (MHz)")

            # Plot amplitude vs. detuning
            ax[1].plot(data["ypts"] - qubit_freq, amp)
            ax[1].set_xlabel("$\Delta$ Frequency (MHz)")
            ax[1].set_ylabel("Amplitude")

            plt.show()


class Rabi2D(QickExperiment2DSimple):
    """
    2D Rabi experiment that sweeps both length and gain.
    This experiment performs a 2D sweep of both qubit pulse length and gain
    to map out the Rabi oscillations.

    Experimental Config:
    expt = dict(
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        style="",
        prefix=None,
        progress=False,
    ):
        """
        Initialize the 2D Rabi experiment.

        This experiment performs a 2D sweep of pulse length and gain.
        Default `params` values:
        - 'span_y': Span of the y-axis sweep (gain) (default: 1)
        - 'expts_y': Number of points in the y-axis sweep (default: 30)
        - 'start_y': Start value for the y-axis sweep (default: 0)
        - 'sweep': Type of sweep for the inner loop, must be 'length' (default: 'length')
        - 'loop': If True, uses loop-based acquisition (default: True)
        - 'yval': The parameter to sweep on the y-axis (default: 'gain')

        Args:
            cfg_dict (dict): Configuration dictionary.
            qi (int): Qubit index.
            go (bool): Whether to run the experiment immediately.
            params (dict): Additional parameters to override defaults.
            style (str): Experiment style.
            prefix (str): Prefix for data files.
            progress (bool): Whether to show a progress bar.
        """
        # Determine prefix based on parameters
        if "type" in params:
            pre = params["type"]
        else:
            pre = "amp"
        if "checkEF" in params and params["checkEF"]:
            ef = "ef"
        else:
            ef = ""
        prefix = f"{pre}_rabi_chevron_{ef}_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        # Default parameters
        params_def = {
            "span_y": 1,
            "expts_y": 30,
            "start_y": 0,
            "sweep": "length",
            "loop": True,
            "yval": "gain",
        }
        params = {**params_def, **params}

        # Create a RabiExperiment instance but don't run it yet
        self.expt = RabiExperiment(
            cfg_dict, qi=qi, go=False, params=params, style=style, check_params=False
        )
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        # Run the experiment if requested
        if go:
            super().run(progress=progress)

    def acquire(self, progress=False, debug=False):
        """
        Acquire data for the RabiChevronExperiment.

        Args:
            progress: Whether to show progress bar
            debug: Whether to print debug information

        Returns:
            Acquired data
        """
        # Create frequency points for the sweep
        ypts = np.linspace(
            self.cfg.expt["start_y"],
            self.cfg.expt["start_y"] + self.cfg.expt["span_y"],
            self.cfg.expt["expts_y"],
        )

        # Set up the y-sweep (frequency sweep)
        ysweep = [{"pts": ypts, "var": "yvar"}]

        # Acquire data
        super().acquire(ysweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data.

        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data
            **kwargs: Additional arguments for the fit

        Returns:
            Analyzed data with fit parameters
        """
        if data is None:
            data = self.data

        if fit:
            # Fit the data to a sinusoidal function for each frequency
            fitterfunc = fitter.fitsin
            fitfunc = fitter.sinfunc
            data = super().analyze(
                fitfunc=fitfunc, fitterfunc=fitterfunc, fit=fit, **kwargs
            )

            # Extract qubit frequency and fit parameters
            qubit_freq = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            freq = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]
            amp = [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))]

            # Fit the chevron pattern (for length rabi)
            try:
                p, _ = curve_fit(chevron_freq, data["ypts"] - qubit_freq, freq)
                p2, _ = curve_fit(chevron_amp, data["ypts"] - qubit_freq, amp)
                data["chevron_freq"] = p
                data["chevron_amp"] = p2
            except:
                # Silently fail if the fit doesn't converge
                pass

        return data

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        """
        Display the results of the RabiChevronExperiment.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_both: Whether to plot both amplitude and phase
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data

        # Set up plot title and labels
        if self.cfg.expt.checkEF:
            title = "EF"
        else:
            title = ""

        if self.cfg.expt.sweep == "amp":
            title = "Amplitude"
            param = "sigma"
            xlabel = "Gain / Max Gain"
        else:
            title = "Length"
            param = "gain"
            xlabel = "Pulse Length ($\mu$s)"

        title += (
            f" Rabi Q{self.cfg.expt.qubit[0]} (Pulse {param} {self.cfg.expt[param]})"
        )

        xlabel = xlabel
        ylabel = "Frequency (MHz)"

        # Display the 2D plot
        super().display(
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            data=data,
            fit=fit,
            plot_both=plot_both,
            **kwargs,
        )

        # If fit is enabled, also display the frequency and amplitude vs. detuning
        if fit:
            fig, ax = plt.subplots(2, 1, figsize=(6, 6))
            qubit_freq = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            freq = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]
            amp = [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))]

            # Plot frequency vs. detuning
            ax[0].plot(data["ypts"] - qubit_freq, freq)
            ax[0].set_ylabel("Frequency (MHz)")

            # Plot amplitude vs. detuning
            ax[1].plot(data["ypts"] - qubit_freq, amp)
            ax[1].set_xlabel("$\Delta$ Frequency (MHz)")
            ax[1].set_ylabel("Amplitude")


# Helper functions for fitting the chevron pattern


def chevron_freq(x, w0):
    """
    Calculate the Rabi frequency as a function of detuning.

    The Rabi frequency is given by sqrt(w0^2 + x^2), where w0 is the
    on-resonance Rabi frequency and x is the detuning.

    Args:
        x: Detuning from resonance
        w0: On-resonance Rabi frequency

    Returns:
        Rabi frequency
    """
    return np.sqrt(w0**2 + x**2)


def chevron_amp(x, w0, a):
    """
    Calculate the Rabi oscillation amplitude as a function of detuning.

    The amplitude is given by a/(1 + (x/w0)^2), where a is the
    on-resonance amplitude, w0 is related to the on-resonance Rabi
    frequency, and x is the detuning.

    Args:
        x: Detuning from resonance
        w0: Width parameter
        a: On-resonance amplitude

    Returns:
        Oscillation amplitude
    """
    return a / (1 + (x / w0) ** 2)
