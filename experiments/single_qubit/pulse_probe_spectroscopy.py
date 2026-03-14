"""
Pulse Probe Spectroscopy Experiment

This module implements pulse probe spectroscopy experiments for qubit characterization.
Pulse probe spectroscopy measures the qubit frequency by applying a probe pulse with
variable frequency and measuring the resulting qubit state. This allows determination
of the qubit transition frequencies (f_ge and f_ef).

The module includes:
- QubitSpecProgram: Defines the pulse sequence for the spectroscopy experiment
- QubitSpec: Main experiment class for frequency spectroscopy
- QubitSpecPower: 2D version that sweeps both frequency and power

This experiment is particularly useful for finding qubit frequencies and characterizing
the qubit spectrum as a function of probe power.
"""

import numpy as np
from qick import *
from qick.asm_v2 import QickSweep1D
from tqdm import tqdm

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment, QickExperiment2DSimple
from ..general.qick_program import QickProgram

from ...analysis import fitting as fitter
from .resonator_spectroscopy import ResSpec
FIT_FUNC = fitter.lorfunc
FITTER_FUNC = fitter.fitlor
XLABEL = "Qubit Frequency (MHz)"
YLABEL = "Qubit Gain (DAC level)"
DCYLABEL = "DC Bias (V)"

class QubitSpecProgram(QickProgram):
    """
    Defines the pulse sequence for a pulse probe spectroscopy experiment.

    The sequence consists of:
    1. Optional π pulse on |g>-|e> transition (if checking EF transition)
    2. Variable frequency probe pulse
    3. Optional second π pulse on |g>-|e> transition (if checking EF transition)
    4. Measurement
    """

    def __init__(self, soccfg, final_delay, cfg):
        """
        Initialize the spectroscopy program.

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

        # Get readout parameters from config
        q = cfg.expt.qubit[0]
        self.frequency = cfg.device.readout.frequency[q]
        self.gain = cfg.device.readout.gain[q]
        self.readout_length = cfg.expt.readout_length
        self.phase = cfg.device.readout.phase[q]

        # Initialize with standard readout
        super()._initialize(cfg, readout="standard")

        # Define the probe pulse with variable frequency
        if cfg.expt.flux:
            self.declare_gen(cfg.expt.flux_chan, nqz=1, mixer_freq=0)
            pulse = {
            "chan": cfg.expt.flux_chan,
            "freq": 0,  # Pulse frequency
            "phase": 0,  # Pulse phase
            "gain": cfg.expt.flux_gain,  # Pulse amplitude
            "length": cfg.expt.length,  # Pulse duration same as qubit pulse
            "type": 'const'            
            }
            super().make_pulse(pulse, "flux_pulse")
        
        pulse = {
            "freq": cfg.expt.frequency,
            "gain": cfg.expt.gain,
            "type": cfg.expt.pulse_type,
            "sigma": cfg.expt.length,
            "phase": 0,
        }
        super().make_pulse(pulse, "qubit_pulse")

        # Add frequency sweep loop
        self.add_loop("freq_loop", cfg.expt.expts)

        # If checking EF transition, create a pi pulse for |g>-|e> transition
        if cfg.expt.checkEF:
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

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

        # If checking EF transition, apply first pi pulse to excite |g> to |e>
        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait 1")
        
        if cfg.expt.flux:
            # Apply the probe pulse with variable frequency
            self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
            # Apply the flux pulse at the same time
            self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse", t=0)
            self.delay_auto(t=0.01, tag="wait flux")
        else:
            # Apply the probe pulse with variable frequency
            self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        

        # Add delay if separate readout is enabled
        if cfg.expt.sep_readout:
            self.delay_auto(t=0.01, tag="wait")

        # If checking EF transition, apply second pi pulse to return to |g> for readout
        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait 2")

        # Perform measurement
        super().measure(cfg)


class QubitSpec(QickExperiment):
    """
    Main experiment class for pulse probe spectroscopy.

    This class implements pulse probe spectroscopy by sweeping the frequency of a probe pulse
    and measuring the resulting qubit state. This allows determination of the qubit transition
    frequencies (f_ge or f_ef).

    Parameters:
    - 'start': Start frequency for the probe sweep (MHz)
    - 'span': Frequency span for the probe sweep (MHz)
    - 'expts': Number of frequency points
    - 'reps': Number of repetitions for each experiment
    - 'rounds': Number of software averages
    - 'length': Probe pulse length (μs)
    - 'gain': Probe pulse gain (DAC units)
    - 'pulse_type': Type of pulse ('const' or 'gauss')
    - 'checkEF': Whether to check the |e>-|f> transition
    - 'sep_readout': Whether to separate the probe pulse and readout
    - 'readout_length': Length of the readout pulse
    - 'final_delay': Delay time between repetitions
    - 'active_reset': Whether to use active reset

    The style parameter can be:
    - 'huge': Very wide frequency span with high power
    - 'coarse': Wide frequency span with medium power
    - 'medium': Medium frequency span with low power
    - 'fine': Narrow frequency span with very low power

    Default parameters are defined in the `__init__` method.
    The `style` parameter sets default values for 'gain', 'span', 'expts', and 'reps'.
    Other default parameters include:
    - 'rounds': self.rounds
    - 'final_delay': 10
    - 'length': 10
    - 'readout_length': from config
    - 'pulse_type': 'const'
    - 'checkEF': False
    - 'qubit': [qi]
    - 'qubit_chan': from config
    - 'sep_readout': True
    - 'active_reset': False
    The 'start' frequency is calculated from the qubit frequency in the config and the span.
    The 'length' can be set to 't1' to be calculated from the T1 time in the config.
    If 'checkEF' is True, 'gain' and 'reps' are increased.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        display=True,
        style="medium",
        min_r2=None,
        max_err=None,
        print=False,
        check_params=True,
    ):
        """
        Initialize the pulse probe spectroscopy experiment.

        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index
            go: Whether to immediately run the experiment
            params: Additional parameters to override defaults
            prefix: Prefix for data files
            progress: Whether to show progress bar
            display: Whether to display results
            style: Style of experiment ('huge', 'coarse', 'medium', or 'fine')
            min_r2: Minimum R² value for fit quality
            max_err: Maximum error for fit quality
        """
        # Currently no control of readout time; may want to change for simultaneious readout

        # Set prefix based on whether we're checking EF transition
        ef = "ef_" if "checkEF" in params and params["checkEF"] else ""
        prefix = f"qubit_spectroscopy_{ef}{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, check_params=check_params, qi=qi)

        # Define default parameters
        max_length = 100  # Based on qick error messages, but not investigated
        spec_gain = self.cfg.device.qubit.spec_gain[qi]
        low_gain = self.cfg.device.qubit.low_gain

        # Set style-specific parameters
        if style == "huge":
            # Very wide frequency span with high power
            params_def = {
                "gain": 80 * low_gain * spec_gain,
                "span": 1500,
                "expts": 1000,
                "reps": self.reps,
            }
        elif style == "coarse":
            # Wide frequency span with medium power
            params_def = {
                "gain": 20 * low_gain * spec_gain,
                "span": 500,
                "expts": 500,
                "reps": self.reps,
            }
        elif style == "medium":
            # Medium frequency span with low power
            params_def = {
                "gain": 5 * low_gain * spec_gain,
                "span": 50,
                "expts": 200,
                "reps": self.reps,
            }
        elif style == "fine":
            # Narrow frequency span with very low power
            params_def = {
                "gain": low_gain * spec_gain,
                "span": 5,
                "expts": 100,
                "reps": 2 * self.reps,
            }

        # Adjust parameters for EF transition
        if "checkEF" in params and params["checkEF"]:
            params_def["gain"] = (
                3 * params_def["gain"]
            )  # Higher power for EF transition
            params_def["reps"] = (
                5 * params_def["reps"]
            )  # More repetitions for better SNR

        # Additional default parameters
        params_def2 = {
            "rounds": self.rounds,
            "final_delay": 10,
            "length": 10,
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "pulse_type": "const",
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            "active_reset": False,
            "flux": False, # Whether to apply flux while pulse
            "flux_chan": self.cfg.hw.soc.dacs.flux.ch[qi], # DAC channel for flux pulse
            "flux_gain": 0.1, # Gain for flux pulse
        }
        params_def = {**params_def2, **params_def}

        # Merge default and user-provided parameters
        params = {**params_def, **params}

        # Set start frequency based on transition type
        if params["checkEF"]:
            params_def["start"] = self.cfg.device.qubit.f_ef[qi] - params["span"] / 2
        else:
            params_def["start"] = self.cfg.device.qubit.f_ge[qi] - params["span"] / 2
        params = {**params_def, **params}

        # Adjust pulse length based on transition type
        if params["length"] == "t1":
            if not params["checkEF"]:
                params["length"] = (
                    3 * self.cfg.device.qubit.T1[qi]
                )  # Longer pulse for GE
            else:
                params["length"] = (
                    self.cfg.device.qubit.T1[qi] / 4
                )  # Shorter pulse for EF

        # Limit pulse length to maximum allowed
        if params["length"] > max_length:
            params["length"] = max_length

        # Set readout length equal to pulse length if not separate
        if not params["sep_readout"]:
            params["readout_length"] = params["length"]

        # Set experiment configuration
        self.cfg.expt = params

        # Check for unexpected parameters
        super().check_params(params_def)

        if print:
            super().print()
            go = False
        # Run the experiment if requested
        if go:
            super().run(
                min_r2=min_r2, max_err=max_err, display=display, progress=progress
            )

    def acquire(self, progress=False):
        """
        Acquire data for the pulse probe spectroscopy experiment.

        Args:
            progress: Whether to show progress bar

        Returns:
            Acquired data
        """
        # Get qubit index and set final delay
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay

        # Set parameter to sweep
        self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}

        # Configure frequency sweep
        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        # Acquire data using the QubitSpecProgram
        super().acquire(QubitSpecProgram, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data to extract qubit parameters.

        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data to a Lorentzian model
            **kwargs: Additional arguments for the fit

        Returns:
            Analyzed data with fit parameters
        """
        if data is None:
            data = self.data

        if fit:
            # Fit the data to a Lorentzian model
            self.fitterfunc = FITTER_FUNC
            self.fitfunc = FIT_FUNC
            super().analyze(use_i=False)

            # Store the fitted qubit frequency
            data["new_freq"] = data["best_fit"][2]

        return self.data

    def display(self, fit=True, ax=None, plot_all=True, **kwargs):
        """
        Display the results of the pulse probe spectroscopy experiment.

        Args:
            fit: Whether to show the fit curve
            ax: Matplotlib axis to plot on
            plot_all: Whether to plot all data types
            **kwargs: Additional arguments for the display
        """
        # Set up fit function and labels
        fitfunc = self.fitfunc 
        xlabel = "Qubit Frequency (MHz)"

        # Set up plot title
        title = f"Spectroscopy Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain})"
        if self.cfg.expt.flux:
            title += f" (Flux Gain {self.cfg.expt.flux_gain})"
        if self.cfg.expt.checkEF:
            title = "EF " + title

        # Define which fit parameters to display in caption
        # Index 2 is frequency, index 3 is kappa
        caption_params = [
            {"index": 2, "format": "$f$: {val:.6} MHz"},
            {"index": 3, "format": "$\kappa$: {val:.3} MHz"},
        ]

        # Display the results
        super().display(
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=False,
            fitfunc=fitfunc,
            caption_params=caption_params,  # Pass the new structured parameter list
        )


class QubitSpecPower(QickExperiment2DSimple):
    """
    2D pulse probe spectroscopy experiment that sweeps both frequency and power.

    This experiment performs a 2D sweep of both probe frequency and power (gain)
    to map out how the qubit spectrum changes with power. This allows visualization
    of power-dependent effects like AC Stark shifts and multi-photon transitions.

    Parameters:
    - 'span': Frequency span for the probe sweep (MHz)
    - 'expts': Number of frequency points
    - 'reps': Number of repetitions for each experiment
    - 'rng': Range for logarithmic gain sweep
    - 'max_gain': Maximum gain for the sweep
    - 'expts_gain': Number of gain points
    - 'log': Whether to use logarithmic gain spacing
    - 'checkEF': Whether to check the |e>-|f> transition

    The style parameter can be:
    - 'coarse': Wide frequency span with many points
    - 'fine': Narrow frequency span with fewer points

    Default parameters are defined in the `__init__` method.
    The `style` parameter sets default values for 'span' and 'expts'.
    Other default parameters include:
    - 'reps': 2 * self.reps
    - 'rng': 50
    - 'max_gain': from config
    - 'expts_gain': 10
    - 'log': True
    This experiment uses the `QubitSpec` experiment, so its parameters are also relevant.
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=None,
        qi=0,
        go=True,
        params={},
        style="",
        display=True,
        min_r2=None,
        max_err=None,
        live_plot=False,
    ):
        """
        Initialize the 2D pulse probe spectroscopy experiment.

        Args:
            cfg_dict: Configuration dictionary
            prefix: Prefix for data files
            progress: Whether to show progress bar
            qi: Qubit index
            go: Whether to immediately run the experiment
            params: Additional parameters to override defaults
            style: Style of experiment ('coarse' or 'fine')
            display: Whether to display results
            min_r2: Minimum R² value for fit quality
            max_err: Maximum error for fit quality
        """
        # Set prefix based on whether we're checking EF transition
        ef = "ef_" if "checkEF" in params and params["checkEF"] else ""
        prefix = f"qubit_spectroscopy_power_{ef}{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, live_plot=live_plot)

        # Set style-specific parameters
        if style == "coarse":
            # Wide frequency span with many points
            params_def = {"span": 800, "expts": 500}
        elif style == "fine":
            # Narrow frequency span with fewer points
            params_def = {"span": 40, "expts": 100}
        else:
            # Default parameters
            params_def = {"span": 120, "expts": 200}

        # Additional default parameters
        params_def2 = {
            "reps": 2 * self.reps,
            "rng": 50,  # Range for logarithmic gain sweep
            "max_gain": self.cfg.device.qubit.max_gain,  # Maximum gain value
            "expts_gain": 10,  # Number of gain points
            "log": True,  # Use logarithmic gain spacing
        }

        # Merge default parameters
        params_def = {**params_def, **params_def2}

        # Merge with user-provided parameters
        params = {**params_def, **params}

        # Create a QubitSpec experiment but don't run it
        exp_name = QubitSpec
        self.expt = exp_name(cfg_dict, qi=qi, go=False, params=params, check_params=False)

        # Get parameters from the QubitSpec experiment
        params = {**self.expt.cfg.expt, **params}

        # Set experiment configuration
        self.cfg.expt = params

        # Run the experiment if requested
        if go:
            self.run(progress=progress, display=display)

    def acquire(self, progress=False):
        """
        Acquire data for the 2D pulse probe spectroscopy experiment.

        Args:
            progress: Whether to show progress bar

        Returns:
            Acquired data
        """
        # Generate gain points for the sweep
        if "log" in self.cfg.expt and self.cfg.expt.log == True:
            # Use logarithmic gain spacing for better dynamic range
            rng = self.cfg.expt.rng
            rat = rng ** (-1 / (self.cfg.expt["expts_gain"] - 1))

            max_gain = self.cfg.expt["max_gain"]
            gainpts = max_gain * rat ** (np.arange(self.cfg.expt["expts_gain"]))
        else:
            # Use linear gain spacing
            gainpts = self.cfg.expt["start_gain"] + self.cfg.expt[
                "step_gain"
            ] * np.arange(self.cfg.expt["expts_gain"])

        # Set up the y-sweep (gain sweep)
        ysweep = [{"pts": gainpts, "var": "gain"}]

        # Configure experiment parameters
        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}

        # Set up frequency sweep
        # self.cfg.expt.frequency = QickSweep1D(
        #     "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        # )

        # Acquire data
        self.xlabel = XLABEL
        self.ylabel = YLABEL
        super().acquire(ysweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data.

        Fits each frequency slice to a Lorentzian model to extract
        qubit parameters as a function of probe power.

        Args:
            fit: Whether to fit the data
            **kwargs: Additional arguments for the fit

        Returns:
            Analyzed data with fit parameters
        """
        if fit:
            # Fit each frequency slice to a Lorentzian model
            super().analyze(fitterfunc=FITTER_FUNC, fitfunc=FIT_FUNC)

        return self.data

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):
        """
        Display the results of the 2D pulse probe spectroscopy experiment.

        Creates a 2D color plot showing the qubit response as a function
        of both frequency and power.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_amps: Whether to plot amplitude data (vs. phase)
            ax: Matplotlib axis to plot on
            **kwargs: Additional arguments for the display
        """
        # Set up plot title
        title = f"Spectroscopy Power Sweep Q{self.cfg.expt.qubit[0]}"
        if self.cfg.expt.checkEF:
            title = f"EF " + title

        # Set axis labels
        

        # Display the 2D plot
        super().display(
            data=data,
            ax=ax,
            plot_amps=plot_amps,
            title=title,
            xlabel=XLABEL,
            ylabel=YLABEL,
            fit=fit,
            **kwargs,
        )

class QubitSpecFlux(QickExperiment2DSimple):
    """
    2D pulse probe spectroscopy: frequency (X) × DC bias (Y).

    Sweeps a fixed frequency span for each DC bias setpoint using the SoC bias port.
    Hardware DC is set per-row via soc.rfb_set_bias(bias_chan, volts).

    Parameters (in params dict):
      # Spectroscopy (inner 1D, same as QubitSpec)
      - 'span' (MHz), 'expts' (# points), 'reps', 'gain', 'pulse_type', 'checkEF', etc.

      # Flux/DC sweep (outer 2D)
      - 'dc_start' (V): start of DC sweep
      - 'dc_stop'  (V): end of DC sweep
      - 'expts_dc' (# rows)
      - 'bias_chan' (int): SoC DC channel to use
      - 'dc_settle' (s): sleep after setting bias (optional)
      - 'dc_verify_print' (bool): print readback line per row (optional)
     optional table to update readout per bias
      - 'bias_table_V': 1D list/array of DC bias values (V)
      - 'f0_table_MHz': 1D list/array of fitted resonator f0 (MHz), same length as bias_table_V
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=None,
        qi=0,
        go=True,
        params={},
        style="",
        display=True,
        min_r2=None,
        max_err=None,
        live_plot=False,
    ):
        # Prefix like other classes
        ef = "ef_" if params.get("checkEF", False) else ""
        prefix = f"qubit_spectroscopy_flux_{ef}{style}_qubit{qi}" if not prefix else prefix
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, live_plot=live_plot)
        self.cfg_dict = cfg_dict

        # ---- style presets for the X (frequency) axis ----
        if style == "coarse":
            x_defaults = {"span": 800, "expts": 500}
        elif style == "fine":
            x_defaults = {"span": 40, "expts": 100}
        else:
            x_defaults = {"span": 120, "expts": 200}

        # ---- defaults for inner 1D and outer DC sweep ----
        inner_defaults = {
            "reps": 2 * self.reps,
            "gain": self.cfg.device.qubit.low_gain * self.cfg.device.qubit.spec_gain[qi],
            "pulse_type": "const",
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            "active_reset": False,
        }

        # Outer DC sweep defaults
        dc_defaults = {
            "dc_start": 0.0,
            "dc_stop":  0.50,
            "expts_dc":  41,
            "bias_chan": 0,
            "dc_settle": 0.0,
            "dc_verify_print": False,
            "recenter_res": False,        # run ResSpec at each bias to recenter readout
            "recenter_res_span": 5,       # MHz span for the ResSpec sweep
            "recenter_res_expts": 100,    # number of points for ResSpec
        }

        # Merge defaults
        params_def = {**x_defaults, **inner_defaults, **dc_defaults}
        params = {**params_def, **params}

        # Determine start frequency around f_ge or f_ef
        if params.get("checkEF", False):
            params["start"] = self.cfg.device.qubit.f_ef[qi] - params["span"] / 2
        else:
            params["start"] = self.cfg.device.qubit.f_ge[qi] - params["span"] / 2

        # Accept an optional bias→f0 table and store sorted copies
        bias_tbl = params.get("bias_table_V", None)
        f0_tbl   = params.get("f0_table_MHz", None)
        if bias_tbl is not None and f0_tbl is not None:
            bt = np.asarray(bias_tbl, dtype=float)
            ft = np.asarray(f0_tbl, dtype=float)
            if bt.shape != ft.shape:
                raise ValueError("bias_table_V and f0_table_MHz must have the same shape.")
            order = np.argsort(bt)
            self.bias_table_V = bt[order]
            self.f0_table_MHz = ft[order]
            # store also in cfg for provenance/saving
            params["bias_table_V"]  = self.bias_table_V.tolist()
            params["f0_table_MHz"]  = self.f0_table_MHz.tolist()

        # Build inner 1D experiment but don't run yet
        self.expt = QubitSpec(cfg_dict, qi=qi, go=False, params=params, check_params=False)

        # Copy params back to outer cfg so both sides agree
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        # Axis labels for plots
        self.ylabel = XLABEL
        self.xlabel = DCYLABEL

        if go:
            self.run(progress=progress, display=display, analyze=False, min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=True):
        """
        Per-row:
          - set dc → soc.rfb_set_bias(bias_chan, dc)
          - optional settle + readback print
          - run inner 1D spectroscopy and stack results
        """
        import time

        # Build DC sweep points
        dcpts = np.linspace(self.cfg.expt.dc_start, self.cfg.expt.dc_stop, int(self.cfg.expt.expts_dc))

        # Prepare containers (same keys as QickExperiment2DSimple.acquire)
        data = {}
        yvals = np.arange(len(dcpts))
        data["time"] = []

         # Precompute if we have a bias→f0 table & container for used f0
        have_table = hasattr(self, "bias_table_V") and hasattr(self, "f0_table_MHz")
        if not have_table:
            # also accept from cfg if class attributes not present (e.g. reload)
            if "bias_table_V" in self.cfg.expt and "f0_table_MHz" in self.cfg.expt:
                bt = np.asarray(self.cfg.expt["bias_table_V"], dtype=float)
                ft = np.asarray(self.cfg.expt["f0_table_MHz"], dtype=float)
                if bt.shape == ft.shape:
                    order = np.argsort(bt)
                    self.bias_table_V  = bt[order]
                    self.f0_table_MHz  = ft[order]
                    have_table = True
        f0_used_each_row = []  # record per-row readout used

        # For each DC bias row
        for i in tqdm(yvals, disable=not progress): 
            dc_val = float(dcpts[i])

            # 1) write into the inner experiment config
            self.expt.cfg.expt["dc"] = dc_val

            # 2) set DC on hardware
            soc = self.im[self.cfg.aliases.soc]
            chan = int(self.cfg.expt["bias_chan"])
            soc.rfb_set_bias(chan, dc_val)

            # optional settle and readback
            if self.cfg.expt.get("dc_settle", 0) > 0:
                time.sleep(self.cfg.expt.dc_settle)
            readback = soc.rfb_get_bias(chan)
            if self.cfg.expt.get("dc_verify_print", False):
                print(f"[DC Bias] ch {chan}: set {dc_val:.6f} V, read {readback:.6f} V")

            # Recenter readout by running a quick ResSpec at this bias
            q = self.expt.cfg.expt["qubit"][0]
            if self.cfg.expt.get("recenter_res", False):
                res_params = {
                    "center": self.cfg.device.readout.frequency[q],
                    "span": self.cfg.expt.get("recenter_res_span", 5),
                    "expts": self.cfg.expt.get("recenter_res_expts", 100),
                    "qubit": [q],
                    "qubit_chan": self.cfg.expt["qubit_chan"],
                }
                res = ResSpec(self.cfg_dict, qi=q, go=False, params=res_params, check_params=False)
                res.acquire(progress=False)
                res.analyze(fit=True, hanger=True)
                if res.data.get("freq_fit") is not None:
                    f0_row = float(res.data["freq_fit"][0])
                    self.expt.cfg.device.readout.frequency[q] = f0_row
                    self.cfg.device.readout.frequency[q]      = f0_row
                    f0_used_each_row.append(f0_row)
            # If we have a bias→f0 table, interpolate and update readout frequency
            elif have_table:
                f0_row = float(np.interp(dc_val, self.bias_table_V, self.f0_table_MHz))
                # update both the inner expt cfg and outer cfg so QickProgram picks it up
                self.expt.cfg.device.readout.frequency[q] = f0_row
                self.cfg.device.readout.frequency[q]      = f0_row
                f0_used_each_row.append(f0_row)

            # 3) run the inner 1D experiment and append
            try:
                data_new = self.expt.acquire(progress=progress)
            except RuntimeError as e:
                # Readout loop error — retry once after tproc reset
                print(f"[QubitSpecFlux] RuntimeError at bias {dc_val:.4f} V: {e}")
                try:
                    soc.tproc.reset()
                    print("[QubitSpecFlux] tproc reset, retrying...")
                    time.sleep(0.5)
                    data_new = self.expt.acquire(progress=progress)
                except RuntimeError:
                    print(f"[QubitSpecFlux] Retry failed. Saving {len(data.get('time',[]))} rows and stopping.")
                    break
            for key in data_new:
                if i == 0:
                    data[key] = []
                data[key].append(data_new[key])
            data["time"].append(time.time())

            # Optional live update/auto-save from base class
            if getattr(self, "save_interim", False):
                super().save_data(data=data)

        freq = data["xpts"][0] if np.ndim(data["xpts"]) > 1 else data["xpts"]
        n_completed = len(data.get("time", []))
        bias = dcpts[:n_completed]

        # overwrite axes
        data["xpts"] = bias
        data["ypts"] = freq
        data["dc_pts"] = bias

        # transpose 2D data so shape matches (len(freq), len(bias))
        for key in ["avgi", "avgq", "amps", "phases"]:
            if key in data:
                data[key] = np.array(data[key]).T  # was (Nbias, Nfreq), make (Nfreq, Nbias)

        # convert lists → numpy arrays
        for k, a in data.items():
            if not isinstance(a, np.ndarray):
                data[k] = np.array(a)

        # Store the readout frequency actually used at each bias (if available)
        if len(f0_used_each_row) > 0:
            data["f0_used_MHz"] = np.asarray(f0_used_each_row, dtype=float)

        if "amps" in data:
            A = np.array(data["amps"])
            data["amps_raw"] = A.copy()
            # subtract per-column minimum (each bias)
            A_min = np.min(A, axis=0, keepdims=True)
            A_sub = A - A_min
            data["amps"] = A_sub
        else:
            A_sub = None

        self.data = data
        return data

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):
        title = f"Spectroscopy vs DC Bias Q{self.cfg.expt.qubit[0]}"
        if self.cfg.expt.checkEF:
            title = "EF " + title
        super().display(
            data=data,
            ax=ax,
            plot_both=False,
            plot_amps=plot_amps,
            title=title,
            xlabel=DCYLABEL,
            ylabel=XLABEL,
            **kwargs,
        )