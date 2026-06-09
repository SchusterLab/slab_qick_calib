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
from ...calib import readout_helpers as helpers

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment, QickExperiment2DSimple
from ..general.qick_program import QickProgram

from ...analysis import fitting as fitter
FIT_FUNC = fitter.lorfunc
FITTER_FUNC = fitter.fitlor
XLABEL = "Qubit Frequency (MHz)"
YLABEL = "Qubit Gain (DAC level)"
DCYLABEL = "DC Bias (V)"

class FWM_tb_sweep_Program(QickProgram):
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
        self.declare_gen(ch = cfg.hw.soc.dacs.readout.ch[0])

        # Define the probe pulse with variable frequency
        qubit_pulse = {
            "freq": cfg.expt.frequency_p,
            "gain": cfg.expt.gain_p,
            "type": cfg.expt.pulse_type_p,
            "sigma": cfg.expt.length_p,
            "phase": 0,
        }
        super().make_pulse(qubit_pulse, "qubit_pulse")

        buffer_pulse = {
            "freq": cfg.expt.frequency_b,
            "gain": cfg.expt.gain_b,
            "type": cfg.expt.pulse_type_b,
            "sigma": cfg.expt.length_b,
            "phase": 0,
            "chan": cfg.hw.soc.dacs.readout.ch[0]
        }
        super().make_pulse(buffer_pulse, "buffer_pulse")

        # Add frequency sweep loop
        self.add_loop("length_b_loop", cfg.expt.expts)

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

        # Apply the probe pulse with variable frequency
        if cfg.expt.do_pump:
            self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        if cfg.expt.do_buffer:
            self.pulse(ch=cfg.hw.soc.dacs.readout.ch[0], name="buffer_pulse", t=cfg.expt.length_p-cfg.expt.length_b)

        # Add delay if separate readout is enabled
        if cfg.expt.sep_readout:
            self.delay_auto(t=0.01, tag="wait")

        # If checking EF transition, apply second pi pulse to return to |g> for readout
        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait 2")

        # Perform measurement
        super().measure(cfg)


class FWM_tb_sweep(QickExperiment):
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
                "gain_p": 80 * low_gain * spec_gain,
                "span": 1500,
                "expts": 1000,
                "reps": self.reps,
            }
        elif style == "coarse":
            # Wide frequency span with medium power
            params_def = {
                "gain_p": 20 * low_gain * spec_gain,
                "span": 500,
                "expts": 500,
                "reps": self.reps,
            }
        elif style == "medium":
            # Medium frequency span with low power
            params_def = {
                "gain_p": 5 * low_gain * spec_gain,
                "span": 50,
                "expts": 200,
                "reps": self.reps,
            }
        elif style == "fine":
            # Narrow frequency span with very low power
            params_def = {
                "gain_p": low_gain * spec_gain,
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
            "length_p": 10,
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "pulse_type_p": "const",
            "gain_b": params_def['gain_p'],
            "frequency_b": self.cfg.device.readout.frequency[0],
            "pulse_type_b": "const",
            'start':1,
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            "active_reset": False,
            "save_shots": False,
            "do_buffer": True,
            "do_pump": True
        }
        params_def = {**params_def2, **params_def}

        # Merge default and user-provided parameters
        params = {**params_def, **params}
        

        # Adjust pulse length based on transition type
        if params["length_p"] == "t1":
            if not params["checkEF"]:
                params["length_p"] = (
                    3 * self.cfg.device.qubit.T1[qi]
                )  # Longer pulse for GE
            else:
                params["length_p"] = (
                    self.cfg.device.qubit.T1[qi] / 4
                )  # Shorter pulse for EF
            params["length_b"] = params["length_p"]

        # Limit pulse length to maximum allowed
        if params["length_p"] > max_length:
            params["length_p"] = max_length 

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
        self.param = {"label": "buffer_pulse", "param": "length", "param_type": "pulse"}

        # Configure length_b sweep
        self.cfg.expt.length_b = QickSweep1D(
            "length_b_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        # Acquire data using the FWM_tb_sweep_Program
        super().acquire(FWM_tb_sweep_Program, progress=progress,shots=True)

        return self.data

    def analyze(self, data=None, fit=True, span=None, verbose=False, **kwargs):
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
        
        
        if 'shots' in data:
            qi = self.cfg.expt.qubit[0]
            threshold = self.cfg.device.readout.threshold[qi]
            i_shots = data['shots'][0, 0, :, :, 0]  # (n_rounds, n_sweep_pts)
            data['p_e'] = np.mean(i_shots > threshold, axis=0)
            if not self.cfg.expt.save_shots:
                data.pop("shots")  # Remove raw shots from data to save space
        

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
        xlabel = "Buffer Length (us)"

        # Set up plot title
        title = f"4WM sweep Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain_p})"
        if self.cfg.expt.checkEF:
            title = "EF " + title

        # Define which fit parameters to display in caption
        # Index 2 is frequency, index 3 is kappa
        caption_params = [
            {"index": 2, "format": "$f$: {val:.6} MHz"},
            {"index": 3, "format": "$\kappa$: {val:.3} MHz"},
        ]


        # plt.figure()
        # plt.plot(self.data['length_b'], self.data['p_e'])
        
        # # Display the results
        super().display(
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=False,
            caption_params=caption_params,  # Pass the new structured parameter list
        )



class FWM_gainb_sweep_Program(QickProgram):
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
        self.declare_gen(ch = cfg.hw.soc.dacs.readout.ch[0])

        # Define the probe pulse with variable frequency
        qubit_pulse = {
            "freq": cfg.expt.frequency_p,
            "gain": cfg.expt.gain_p,
            "type": cfg.expt.pulse_type_p,
            "sigma": cfg.expt.length_p,
            "phase": 0,
        }
        super().make_pulse(qubit_pulse, "qubit_pulse")

        buffer_pulse = {
            "freq": cfg.expt.frequency_b,
            "gain": cfg.expt.gain_b,
            "type": cfg.expt.pulse_type_b,
            "sigma": cfg.expt.length_b,
            "phase": 0,
            "chan": cfg.hw.soc.dacs.readout.ch[0]
        }
        super().make_pulse(buffer_pulse, "buffer_pulse")

        # Add frequency sweep loop
        self.add_loop("gain_b_loop", cfg.expt.expts)

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

        # Apply the probe pulse with variable frequency
        if cfg.expt.do_pump:
            self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        if cfg.expt.do_buffer:
            self.pulse(ch=cfg.hw.soc.dacs.readout.ch[0], name="buffer_pulse", t=cfg.expt.length_p-cfg.expt.length_b)

        # Add delay if separate readout is enabled
        if cfg.expt.sep_readout:
            self.delay_auto(t=0.01, tag="wait")

        # If checking EF transition, apply second pi pulse to return to |g> for readout
        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait 2")

        # Perform measurement
        super().measure(cfg)


class FWM_gainb_sweep(QickExperiment):
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


        # Additional default parameters
        params_def2 = {
            "rounds": self.rounds,
            "final_delay": 10,
            "length_p": 10,
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "pulse_type_p": "const",
            "length_b": 2,
            "frequency_b": self.cfg.device.readout.frequency[0],
            "pulse_type_b": "const",
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            "active_reset": False,
            "save_shots": False,
            "do_buffer": True,
            "do_pump": True
        }

        # Merge default and user-provided parameters
        params = {**params_def2, **params}
        

        # Adjust pulse length based on transition type
        if params["length_p"] == "t1":
            if not params["checkEF"]:
                params["length_p"] = (
                    3 * self.cfg.device.qubit.T1[qi]
                )  # Longer pulse for GE
            else:
                params["length_p"] = (
                    self.cfg.device.qubit.T1[qi] / 4
                )  # Shorter pulse for EF
            params["length_b"] = params["length_p"]

        # Limit pulse length to maximum allowed
        if params["length_p"] > max_length:
            params["length_p"] = max_length 

        # Set readout length equal to pulse length if not separate
        if not params["sep_readout"]:
            params["readout_length"] = params["length"]

        # Set experiment configuration
        self.cfg.expt = params

        # Check for unexpected parameters
        super().check_params(params_def2)

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
        self.param = {"label": "buffer_pulse", "param": "gain", "param_type": "pulse"}

        # Configure gain_b sweep
        self.cfg.expt.gain_b = QickSweep1D(
            "gain_b_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        # Acquire data using the FWM_gainb_sweep_Program
        super().acquire(FWM_gainb_sweep_Program, progress=progress,shots=True)

        return self.data

    def analyze(self, data=None, fit=True, span=None, verbose=False, **kwargs):
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
        
        
        if 'shots' in data:
            qi = self.cfg.expt.qubit[0]
            threshold = self.cfg.device.readout.threshold[qi]
            i_shots = data['shots'][0, 0, :, :, 0]  # (n_rounds, n_sweep_pts)
            data['p_e'] = np.mean(i_shots > threshold, axis=0)
            if not self.cfg.expt.save_shots:
                data.pop("shots")  # Remove raw shots from data to save space
        

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
        xlabel = "buffer gain (ADC units)"

        # Set up plot title
        title = f"4WM sweep Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain_p})"
        if self.cfg.expt.checkEF:
            title = "EF " + title

        # Define which fit parameters to display in caption
        # Index 2 is frequency, index 3 is kappa
        caption_params = [
            {"index": 2, "format": "$f$: {val:.6} MHz"},
            {"index": 3, "format": "$\kappa$: {val:.3} MHz"},
        ]


        # plt.figure()
        # plt.plot(self.data['length_b'], self.data['p_e'])
        
        # # Display the results
        super().display(
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=False,
            caption_params=caption_params,  # Pass the new structured parameter list
        )

class FWM_gainb_tb_sweep(QickExperiment2DSimple):
    """
    4WM sweep over buffer gain and length

    This experiment performs a 2D sweep of both buffer gain and length
    to map out how the efficiency changes with these parameters

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
        prefix = f"4WM_sweep_lengthb_gainb_{ef}{style}_qubit{qi}"
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
            "length_b_start": 0.1,
            "length_b_end": 10,
            "length_b_expts": 20,
            "save_shots": False
        }

        # Merge default parameters
        params_def = {**params_def, **params_def2}

        # Merge with user-provided parameters
        params = {**params_def, **params}

        # Create a FWM_gainb_sweep experiment but don't run it
        exp_name = FWM_gainb_sweep
        self.expt = exp_name(cfg_dict, qi=qi, go=False, params=params, check_params=False)

        # Get parameters from the FWM_gainb_sweep experiment
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
        # Generate length points for the sweep
        length_b_points = np.linspace(self.cfg.expt.length_b_start, self.cfg.expt.length_b_end, self.cfg.expt.length_b_expts)


        ysweep = [{"pts": length_b_points, "var": "length_b"}]

        # Configure experiment parameters
        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {"label": "buffer_pulse", "param": "length", "param_type": "pulse"}

        

        # Set up frequency sweep
        # self.cfg.expt.frequency_b = QickSweep1D(
        #     "freq_b_loop", self.cfg.expt.center_freq_b - self.cfg.expt.span_b / 2, self.cfg.expt.center_freq_b + self.cfg.expt.span_b / 2, self.cfg.expt.expts_b
        # )

        # Acquire data
        self.xlabel = 'buffer gain'
        self.ylabel = 'buffer length (us)'
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
        # if fit:
        #     # Fit each frequency slice to a Lorentzian model
        #     super().analyze(fitterfunc=FITTER_FUNC, fitfunc=FIT_FUNC)
        
        if 'shots' in data:
            qi = self.cfg.expt.qubit[0]
            threshold = self.cfg.device.readout.threshold[qi]
            data['p_e'] = np.zeros((data['shots'].shape[0], data['shots'].shape[4]))  # Initialize p_e array
            for i in range(len(data['length_b_pts'])):
                i_shots = data['shots'][i, 0, 0, :, :, 0]  # (n_rounds, n_sweep_pts)
                data['p_e'][i] = np.mean(i_shots > threshold, axis=0)


            if not self.cfg.expt.save_shots:
                data.pop("shots")  # Remove raw shots from data to save space


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
        title = f"pump gain = {self.cfg.expt.gain_p}"
        if self.cfg.expt.checkEF:
            title = f"EF " + title

        # Set axis labels
        

        # Display the 2D plot
        super().display(
            data=data,
            ax=ax,
            plot_amps=plot_amps,
            title=title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            fit=fit,
            **kwargs,
        )

class FWM_tp_sweep_Program(QickProgram):
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
        qubit_pulse = {
            "freq": cfg.expt.frequency_p,
            "gain": cfg.expt.gain_p,
            "type": cfg.expt.pulse_type_p,
            "sigma": cfg.expt.length_p,
            "phase": 0,
        }
        super().make_pulse(qubit_pulse, "qubit_pulse")


        # Add frequency sweep loop
        self.add_loop("length_p_loop", cfg.expt.expts)

        # If checking EF transition, create a pi pulse for |g>-|e> transition
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

        # Apply the probe pulse with variable frequency
        if cfg.expt.do_pump:
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

        self.delay_auto(t=200, tag="wait 2")


class FWM_tp_sweep(QickExperiment):
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
        prefix = f"4WM_sweep_tp_{ef}{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, check_params=check_params, qi=qi)

        # Define default parameters
        max_length = 100  # Based on qick error messages, but not investigated
        spec_gain = self.cfg.device.qubit.spec_gain[qi]
        low_gain = self.cfg.device.qubit.low_gain


        # Additional default parameters
        params_def2 = {
            "rounds": self.rounds,
            "final_delay": 10,
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "pulse_type_p": "const",
            "start":1,
            "stop":30,
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            "active_reset": True,
            "save_shots": False,
            "do_pump": True,
            "threshold": self.cfg.device.readout.threshold[qi]
        }

        # Merge default and user-provided parameters
        params = {**params_def2, **params}
        



        # Set readout length equal to pulse length if not separate
        if not params["sep_readout"]:
            params["readout_length"] = params["length"]

        # Set experiment configuration
        self.cfg.expt = params

        if self.cfg.expt.active_reset:
            super().configure_reset()

        # Check for unexpected parameters
        super().check_params(params_def2)

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
        self.param = {"label": "qubit_pulse", "param": "length", "param_type": "pulse"}

        # Configure length_p sweep
        self.cfg.expt.length_p = QickSweep1D(
            "length_p_loop", self.cfg.expt.start, self.cfg.expt.stop
        )

        # Acquire data using the FWM_tp_sweep_Program
        super().acquire(FWM_tp_sweep_Program, progress=progress,shots=True)

        return self.data

    def analyze(self, data=None, fit=True, span=None, verbose=False, **kwargs):
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
        
        
        if 'shots' in data:
            qi = self.cfg.expt.qubit[0]
            threshold = self.cfg.device.readout.threshold[qi]
            i_shots = data['shots'][0, 0, :, :, 0]  # (n_rounds, n_sweep_pts)
            data['p_e'] = np.mean(i_shots > threshold, axis=0)
            if not self.cfg.expt.save_shots:
                data.pop("shots")  # Remove raw shots from data to save space
        

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
        xlabel = "pump pulse length (us)"

        # Set up plot title
        title = f"4WM sweep Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain_p})"
        if self.cfg.expt.checkEF:
            title = "EF " + title

        # Define which fit parameters to display in caption
        # Index 2 is frequency, index 3 is kappa
        caption_params = [
            {"index": 2, "format": "$f$: {val:.6} MHz"},
            {"index": 3, "format": "$\kappa$: {val:.3} MHz"},
        ]


        # plt.figure()
        # plt.plot(self.data['length_b'], self.data['p_e'])
        
        # # Display the results
        super().display(
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=False,
            caption_params=caption_params,  # Pass the new structured parameter list
        )
