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
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from qick import *
from qick.asm_v2 import QickSweep1D

from ...analysis import fitting as fitter
from ..general.qick_experiment import (
    QickExperiment,
    QickExperiment2DSimple
)
from ..general.qick_program import QickProgram

from ...exp_handling.datamanagement import AttrDict
from ...helpers import config
FITTER_FUNC = fitter.fitsin
FIT_FUNC = fitter.sinfunc
# ====================================================== #


class RabiProgram(QickProgram):
    """
    Defines the pulse sequence for a Rabi oscillation experiment.

    The sequence consists of:
    1. Optional π pulse on |g>-|e> transition (if checking EF transition)
    2. Variable amplitude/length pulse on the qubit (repeated n_pulses times)
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

        # Create the main qubit pulse
        pulse_params = self._get_pulse_params(cfg)
        super().make_pulse(pulse_params, "qubit_pulse")

        # If checking EF transition and using ge pulse, create a pi pulse
        if (cfg.expt.checkEF and cfg.expt.pulse_ge) or cfg.expt.active_reset:
            super().make_cfg_pulse(q, cfg.device.qubit.f_ge, "pi_ge")


    def _get_pulse_params(self, cfg):
        """
        Get pulse parameters based on pulse type and sweep configuration.
        
        Args:
            cfg: Configuration dictionary
            
        Returns:
            Dictionary of pulse parameters
        """
        pulse = {
            "freq": cfg.expt.freq,
            "gain": cfg.expt.gain,
            "phase": cfg.expt.phase,
            "type": cfg.expt.pulse_type,
        }
        
        # Set length/sigma based on pulse type
        if cfg.expt.pulse_type == "gauss":
            pulse["sigma"] = cfg.expt.sigma
            pulse["length"] = cfg.expt.length
        elif cfg.expt.pulse_type == "flat_top":
            pulse["length"] = cfg.expt.length
            pulse["ramp_sigma"] = cfg.expt.ramp_sigma
            pulse["ramp_sigma_inc"] = cfg.expt.ramp_sigma_inc
        else:  # const
            pulse["length"] = cfg.expt.length
            
        return pulse

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

        # Apply the main qubit pulse (repeated n_pulses times)
        for i in range(cfg.expt.n_pulses):
            self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
            if i < cfg.expt.n_pulses - 1:  # Don't add delay after last pulse
                self.delay_auto(t=0.01)

        # If checking EF transition with ge pulse, apply second pi pulse
        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.delay_auto(t=0.01, tag="wait ef 2")
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

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
    - 'reps': Number of repetitions, from `self.reps``
    - 'rounds': Number of software averages, from `self.rounds`
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
    - 'num_osc': Try to set max gain or length for this number of oscillations (default: 2.5)
    - 'n_pulses': Number of Rabi pulses to apply (default: 1)
    - 'active_reset': If True, uses active reset (default: from `cfg.device.readout.active_reset[qi]`)
    - 'loop': If True, uses loop-based acquisition (default: False)
    - 'temp': Temperature parameter for temperature-dependent measurements (default: 40)
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
        """Initialize the Rabi experiment."""
        
        # Generate experiment prefix
        if prefix is None:
            prefix = self._generate_prefix(params, qi)

        super().__init__(
            cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, check_params=check_params
        )

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

        # Apply style modifications for different experiment modes
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 25
        elif style == "temp":
            # Special configuration for temperature-dependent measurements
            params_def["reps"] = int(40 * params_def["reps"])
            params_def["rounds"] = int(
                np.ceil(20 * params_def["rounds"] * np.exp( 2*(40 / params.get("temp", 40)-1)))
            )
            params_def["pulse_ge"] = False
        
        params = {**params_def, **params}
        
        # Configure pulse parameters based on transition type
        params=self._configure_pulse_params(params, params_def, qi)
        params = {**params_def, **params}
        
        # Configure sweep parameters
        params=self._configure_sweep_params(params, params_def)
        
        # Set final experiment configuration
        self.cfg.expt = {**params_def, **params}
        
        # Check parameters and run if requested
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

    def _generate_prefix(self, params, qi):
        """Generate experiment prefix based on parameters."""
        if "checkEF" in params and params["checkEF"]:
            ef = "ef_" if params.get("pulse_ge", True) else "ef_no_ge_"
        else:
            ef = ""
        
        sweep_type = params.get("sweep", "amp")
        name = "length" if sweep_type == "length" else "amp"
        
        return f"{name}_rabi_{ef}qubit{qi}"

    def _configure_pulse_params(self, params, params_def, qi):
        """Configure pulse parameters based on transition type."""
        if params["checkEF"]:
            pulse_config = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            pulse_config = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]
        
        # Copy pulse parameters from device config
        for key in pulse_config:
            if key not in params:  # Don't override user-provided parameters
                params_def[key] = pulse_config[key][qi]

        # Handle pulse_type parameter as alias for type

        params_def["pulse_type"] = params_def["type"]
        params = {**params_def, **params}

        # Resolve length from pulse type so downstream code can use direct access
        if "length" not in params or params.get("length") is None:
            if params["pulse_type"] == "gauss":
                params["length"] = params["sigma"] * params["sigma_inc"]
            else:
                params["length"] = params["sigma"]

        # Resolve flat_top ramp defaults
        if params["pulse_type"] == "flat_top":
            if "ramp_sigma" not in params:
                params["ramp_sigma"] = 0.02
            if "ramp_sigma_inc" not in params:
                params["ramp_sigma_inc"] = 3

        return params
        

    def _configure_sweep_params(self, params, params_def):
        """Configure sweep range based on sweep type."""
        min_gain = 2**-15  # Minimum DAC gain for linear operation
        
        if params["sweep"] == "amp":
            if params["n_pulses"] == 1:
                # Single pulse: sweep from near zero to cover desired oscillations
                params_def["start"] = 0.005 # Minimum gain that works for current QICK
                params_def["max_gain"] = params["gain"] * params["num_osc"] * 2
            else:
                # Multiple pulses: sweep around nominal gain for pi pulse in small range 
                gain_range = params["gain"] / params["n_pulses"]
                params_def["start"] = params["gain"] - gain_range
                params_def["max_gain"] = params["gain"] + gain_range
            
            # Ensure we don't exceed hardware limits
            params = {**params_def, **params}
            params["max_gain"] = min(params["max_gain"], self.cfg.device.qubit.max_gain)
            
            # Ensure minimum gain spacing to avoid DAC resolution issues
            gain_spacing = (params["max_gain"]-params['start']) / params["expts"]
            if gain_spacing < min_gain:
                span = min_gain * params["expts"]
                params["max_gain"]=params['gain']+span/2
                params["start"]=params['gain']-span/2
                
        elif params["sweep"] == "length":
            # Length sweep: set max length to cover desired oscillations
            if params["pulse_type"] == "gauss":
                # For Gaussian pulses, we sweep sigma
                params_def["start"] = self.soccfg.cycles2us(1)  
                params_def["max_length"] = 2 * params["num_osc"] * params["sigma"]
            else:
                # For const and flat_top pulses, sweep length directly
                params_def["start"] = 3 * self.soccfg.cycles2us(1) # Minimum length
                params = {**params_def, **params}
                params_def["max_length"] = 2 * params["num_osc"] * params["length"]

            params = {**params_def, **params}
        return params

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
        self.param = {"label": "qubit_pulse", "param_type": "pulse"}

        if self.cfg.expt.loop:
            # Use loop-based acquisition for complex sweeps
            return self._acquire_loop(progress)
        else:
            if self.cfg.expt.pulse_type == 'gauss' and self.cfg.expt.sweep=='length':
                print('Cannot use qick sweep for a length sweep with Gaussian pulses. Using loop mode instead.')
                return self._acquire_loop(progress)
            # Use QICK sweep-based acquisition (default)
            return self._acquire_qick_sweep(progress)

    def _acquire_qick_sweep(self, progress):
        """Acquire data using QICK sweeps."""
     
        if self.cfg.expt.sweep == "amp":
            # Amplitude sweep
            self.cfg.expt["gain"] = QickSweep1D(
                "sweep_loop", self.cfg.expt.start, self.cfg.expt.max_gain
            )
            self.param['param'] = "gain"
            
        elif self.cfg.expt.sweep == "length":
            # Length sweep
            sweep = QickSweep1D("sweep_loop", self.cfg.expt.start, self.cfg.expt.max_length)
            if self.cfg.expt.pulse_type == "gauss":
                # For Gaussian pulses, sweep sigma
                self.cfg.expt["sigma"] = sweep
                self.param['param'] = "sigma"
            else:
                # For other pulse types, sweep length
                self.cfg.expt["length"] = sweep
                self.param['param'] = "total_length"

        # Acquire data using the RabiProgram
        super().acquire(RabiProgram, progress=progress)
        return self.data

    def _acquire_loop(self, progress):
        """Acquire data using loop-based acquisition."""
        if self.cfg.expt.sweep == "length":
            # Set up length points for the loop
            len_pts = np.linspace(
                self.cfg.expt.start, self.cfg.expt.max_length, self.cfg.expt.expts
            )
            
            if self.cfg.expt.pulse_type == "gauss":
                # For Gaussian pulses, sweep sigma and adjust length accordingly
                x_sweep = [
                    {"pts": len_pts, "var": "sigma"},
                    {
                        "pts": len_pts * self.cfg.expt.sigma_inc, # is this needed
                        "var": "length",
                    },
                ]
                self.param["param"]="total_length"
            else:
                # For other pulse types, sweep length directly
                x_sweep = [{"pts": len_pts, "var": "length"}]
                self.param["param"]="total_length"
            
        else:  # amplitude sweep
            gain_pts = np.linspace(
                self.cfg.expt.start, self.cfg.expt.max_gain, self.cfg.expt.expts
            )
            x_sweep = [{"pts": gain_pts, "var": "gain"}]
            self.param["param"]="gain"

        # Use loop acquisition method from base class
        self.data = super().run_loop(RabiProgram, x_sweep, progress=progress)
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
            self.fitterfunc = FITTER_FUNC
            self.fitfunc = FIT_FUNC
            data = super().analyze(fit=fit, **kwargs)

        # Calculate π-pulse length from the fit for each data type
        ydata_lab = ["amps", "avgi", "avgq"]
        for ydata in ydata_lab:
            if f"fit_{ydata}" in data:
                pi_length = fitter.fix_phase(data[f"fit_{ydata}"])
                data[f"pi_length_{ydata}"] = pi_length
        
        # Set the best π-pulse length
        if "best_fit" in data:
            data["pi_length"] = fitter.fix_phase(data["best_fit"])
            data["pi_length_scale_data"] = data.get("pi_length_avgi", data["pi_length"])

        return data

    def update(self, verbose=True):
        """Write the fitted pi pulse gain or length/sigma back to the config file."""
        qi = self.cfg.expt.qubit[0]
        if "pi_length" not in self.data:
            print("No pi_length found in data, cannot update.")
            return

        pi_val = self.data["pi_length"]
        pulse = "pi_ef" if self.cfg.expt.checkEF else "pi_ge"

        if self.cfg.expt.sweep == "amp":
            config.update_config(
                self.config_file, f"device.qubit.pulses.{pulse}", "gain",
                pi_val, index=qi, verbose=verbose,
            )
        elif self.cfg.expt.sweep == "length":
            if self.cfg.expt.pulse_type == "gauss":
                config.update_config(
                    self.config_file, f"device.qubit.pulses.{pulse}", "sigma",
                    pi_val, index=qi, verbose=verbose,
                )
            else:
                config.update_config(
                    self.config_file, f"device.qubit.pulses.{pulse}", "sigma",
                    pi_val, index=qi, verbose=verbose,
                )

    def display(
        self,
        data=None,
        fit=True,
        plot_all=True,
        ax=None,
        show_hist=False,
        rescale=False,
        **kwargs,
    ):
        """Plot the Rabi oscillation with the fitted pi length in the caption."""
        if data is None:
            data = self.data

        # Set up plot title and labels
        title, xlabel = self._get_plot_labels()
        caption_params = [{"index": "pi_length", "format": "$\pi$ length: {val:.3f}"}]

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
            **kwargs,
        )

    def _get_plot_labels(self):
        """Generate plot title and xlabel based on experiment configuration."""
        q = self.cfg.expt.qubit[0]
        
        if self.cfg.expt.sweep == "amp":
            title = "Amplitude"
            xlabel = "Gain / Max Gain"
            #param_name = "sigma" if self.cfg.expt.type == "gauss" else "length"
            param_name = "sigma"
        else:
            title = "Length"
            xlabel = "Pulse Length ($\mu$s)"
            param_name = "gain"
        
        param_value = self.cfg.expt[param_name]
        title += f" Rabi Q{q} (Pulse {param_name} {param_value}"
        
        if self.cfg.expt.checkEF:
            title += ", EF)"
        else:
            title += ")"
            
        return title, xlabel


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
        """Initialize the ReadoutCheck experiment."""
        
        prefix = f"readout_qubit{qi}" if prefix is None else prefix
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

        # Apply style modifications
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 25

        # Merge parameters and configure experiment
        params = {**params_def, **params}
        params["sigma"] = params["length"]  # Set sigma equal to length
        params["freq"] = params["qubit_freq"] + params["df"]  # Set frequency with offset

        self.cfg.expt = {**params_def, **params}

        # Check parameters and configure reset if needed
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()

        # Run the experiment if requested
        if go:
            if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
                disp_kwargs = {"plot_all": True}
                
            super().run(
                display=display,
                progress=progress,
                min_r2=min_r2,
                max_err=max_err,
                disp_kwargs=disp_kwargs,
            )

    def acquire(self, progress=False, debug=False, single=False):
        """Acquire data for the ReadoutCheck experiment."""
        self.qubit = self.cfg.expt.qubit

        # Configure the sweep based on experiment type
        if self.cfg.expt.expt_type == "end_wait":
            # Sweep end wait time
            self.cfg.expt["end_wait"] = QickSweep1D(
                "sweep_loop", self.cfg.expt.start, self.cfg.expt.max_wait
            )
            self.param = {"label": "end_wait", "param": "t", "param_type": "time"}
        else:
            # Sweep gain
            self.cfg.expt["gain"] = QickSweep1D(
                "sweep_loop", self.cfg.expt.start, self.cfg.expt.max_gain
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
        """Analyze the acquired data (no specific analysis for ReadoutCheck)."""
        if data is None:
            data = self.data
        return data

    def display(self, data=None, fit=False, plot_all=False, **kwargs):
        """Display the results of the ReadoutCheck experiment."""
        if data is None:
            data = self.data
        super().display(data=data, fit=fit, plot_all=plot_all, **kwargs)


class RabiChevronExperiment(QickExperiment2DSimple):
    """
    2D Rabi experiment that sweeps both frequency and amplitude/length.

    This experiment performs a 2D sweep of both qubit frequency and pulse amplitude/length
    to map out the Rabi chevron pattern. This allows visualization of how the Rabi
    oscillation frequency changes with detuning from the qubit frequency.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        style="",
        prefix=None,
        progress=True,
        live_plot=False,
    ):
        """Initialize the RabiChevronExperiment."""
        
        if prefix is None:
            prefix = self._get_prefix(params, qi)

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, live_plot=live_plot)

        # Default parameters
        params_def = {"span_f": 20, "expts_f": 30, "sweep": "amp"}
        params = {**params_def, **params}

        # Set frequency range based on transition type
        if params.get("checkEF", False):
            f_qubit = self.cfg.device.qubit.f_ef[qi]
        else:
            f_qubit = self.cfg.device.qubit.f_ge[qi]
        params_def["start_f"] = f_qubit - params["span_f"] / 2

        # Create a RabiExperiment instance but don't run it yet
        self.expt = RabiExperiment(
            cfg_dict, qi=qi, go=False, params=params, style=style, check_params=False
        )
        params = {**params_def, **params}
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        # Run the experiment if requested
        if go:
            super().run(progress=progress)

    def _get_prefix(self, params, qi):
        """Generate a prefix for the experiment."""
        sweep_type = params.get("sweep", "amp")
        ef = "ef_" if params.get("checkEF", False) else ""
        return f"{sweep_type}_rabi_chevron_{ef}qubit{qi}"

    def acquire(self, progress=False, debug=False):
        """Acquire data for the RabiChevronExperiment."""
        # Create frequency points for the sweep
        freqpts = np.linspace(
            self.cfg.expt["start_f"],
            self.cfg.expt["start_f"] + self.cfg.expt["span_f"],
            self.cfg.expt["expts_f"],
        )

        # Set up the y-sweep (frequency sweep)
        ysweep = [{"pts": freqpts, "var": "freq"}]

        # Acquire data
        self._get_title_and_labels()
        super().acquire(ysweep, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """Analyze the acquired data."""
        if data is None:
            data = self.data

        if fit:
            # Fit the data to a sinusoidal function for each frequency
            data = super().analyze(
                fitfunc=FIT_FUNC, fitterfunc=FITTER_FUNC, fit=fit, **kwargs
            )

            # Extract Rabi frequency and amplitude vs detuning
            if "fit_avgi" in data:
                qubit_freq = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
                data["chevron_freqs"] = np.array([data["fit_avgi"][i][1] for i in range(len(data["ypts"]))], dtype=float)
                data["chevron_amps"] = np.array([data["fit_avgi"][i][0] for i in range(len(data["ypts"]))], dtype=float)

                # Filter out NaN values from failed row fits
                valid = np.isfinite(data["chevron_freqs"]) & np.isfinite(data["chevron_amps"])
                if np.any(valid):
                    data["best_freq"] = data["ypts"][valid][np.argmax(data["chevron_amps"][valid])]
                else:
                    data["best_freq"] = data["ypts"][len(data["ypts"]) // 2]

                # Fit the chevron pattern
                try:
                    if np.sum(valid) < 3:
                        raise RuntimeError("Not enough valid points for chevron fit")
                    detuning = data["ypts"][valid] - qubit_freq
                    p_freq, _ = curve_fit(chevron_freq, detuning, data["chevron_freqs"][valid])
                    p_amp, _ = curve_fit(chevron_amp, detuning, data["chevron_amps"][valid])
                    data["chevron_freq_fit"] = np.array(p_freq)
                    data["chevron_amp_fit"] = np.array(p_amp)
                except (RuntimeError, TypeError, ValueError):
                    print("Chevron fit failed to converge.")
                    data["chevron_freq_fit"] = np.array([])
                    data["chevron_amp_fit"] = np.array([])

        return data

    def display(self, data=None, fit=True, plot_all=False, **kwargs):
        """Display the results of the RabiChevronExperiment."""
        if data is None:
            data = self.data

        

        # Display the 2D plot
        super().display(
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            data=data,
            fit=fit,
            plot_all=plot_all,
            **kwargs,
        )

        # If fit is enabled, also display the chevron fit results
        if fit and "chevron_freqs" in data:
            self._plot_chevron_fits(data)

    def _get_title_and_labels(self):
        """Get title and labels for the plot."""
        title_prefix = "EF " if self.cfg.expt.checkEF else ""
        
        if self.cfg.expt.sweep == "amp":
            sweep_type = "Amplitude"
            self.xlabel = "Gain / Max Gain"
            if self.cfg.expt.type == "gauss":
                param = "sigma"
            else:  # const or flat_top
                param = "length"
        else:
            sweep_type = "Length"
            self.xlabel = "Pulse Length ($\mu$s)"
            param = "gain"

        self.title = (
            f"{title_prefix}{sweep_type} Rabi Q{self.cfg.expt.qubit[0]} "
            f"(Pulse {param} {self.cfg.expt[param]})"
        )
        self.ylabel = "Frequency (MHz)"
        

    def _plot_chevron_fits(self, data):
        """Plot the chevron fit results."""
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        qubit_freq = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
        detuning = data["ypts"] - qubit_freq

        # Plot frequency vs. detuning
        ax[0].plot(detuning, data["chevron_freqs"], 'o')
        if data.get("chevron_freq_fit") is not None and len(data["chevron_freq_fit"]) > 0:
            ax[0].plot(detuning, chevron_freq(detuning, *data["chevron_freq_fit"]), 'r-')
        ax[0].set_ylabel("Rabi Frequency (MHz)")
        ax[0].set_title("Chevron Frequency vs Detuning")

        # Plot amplitude vs. detuning
        ax[1].plot(detuning, data["chevron_amps"], 'o')
        if data.get("chevron_amp_fit") is not None and len(data["chevron_amp_fit"]) > 0:
            ax[1].plot(detuning, chevron_amp(detuning, *data["chevron_amp_fit"]), 'r-')
        ax[1].set_xlabel("$\Delta$ Frequency (MHz)")
        ax[1].set_ylabel("Rabi Amplitude")
        ax[1].set_title("Chevron Amplitude vs Detuning")

        plt.tight_layout()
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
        
        if prefix is None:
            prefix = f"{pre}_rabi_2d_{ef}_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        # Default parameters
        params_def = {
            "span_y": 0.05,
            "expts_y": 30,
            "start_y": 0.003,
            "sweep": "amp",
            "loop": False,
            "yval": "sigma",
            "sigma_inc": 4
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
        Acquire data for the 2D Rabi experiment.

        Args:
            progress: Whether to show progress bar
            debug: Whether to print debug information

        Returns:
            Acquired data
        """
        # Create y-axis points for the sweep
        ypts = np.linspace(
            self.cfg.expt["start_y"],
            self.cfg.expt["start_y"] + self.cfg.expt["span_y"],
            self.cfg.expt["expts_y"],
        )
        

        # Set up the y-sweep
        ysweep = [{"pts": ypts, "var": self.cfg.expt["yval"]}]
        if self.cfg.expt["yval"] == "sigma":
            ysweep.append({"pts": ypts * self.cfg.expt.sigma_inc, "var": "length"})
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
            # Fit the data to a sinusoidal function for each y value
            data = super().analyze(
                fitfunc=fitter.sinfunc, fitterfunc=fitter.fitsin, fit=fit, **kwargs
            )

        return data

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        """
        Display the results of the 2D Rabi experiment.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_both: Whether to plot both amplitude and phase
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data

        # Set up plot title and labels
        title = "2D Rabi"
        if self.cfg.expt.checkEF:
            title = "EF " + title
        
        if self.cfg.expt.sweep == "length":
            xlabel = "Pulse Length ($\mu$s)"
            param = "gain"
        else:
            xlabel = "Gain / Max Gain"
            param = "sigma" if self.cfg.expt.type == "gauss" else "length"

        title += f" Q{self.cfg.expt.qubit[0]} (Pulse {param} {self.cfg.expt[param]})"
        ylabel = f"{self.cfg.expt['yval'].title()}"

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


# Helper functions for fitting the chevron pattern

def chevron_freq(x, w0, x0):
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
    return np.sqrt(w0**2 + (x-x0)**2)


def chevron_amp(x, w0, a, x0):
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
    return a / (1 + ((x-x0) / w0) ** 2)
