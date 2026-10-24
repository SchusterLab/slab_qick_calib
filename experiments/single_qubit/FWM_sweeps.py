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

import time

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.asm_v2 import QickSweep1D
from tqdm import tqdm
from ...calib import readout_helpers as helpers

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment, QickExperiment2DSimple
from ..general.qick_program import QickProgram, reset_mode_of, reset_plays_pi_pulses, pulse_type_from_config
from .qubit_spec_shot import band_frequency_from_fit

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
        # The pump goes on the pump line, which is the qubit channel unless
        # hw.soc.dacs.pump says otherwise.
        qubit_pulse = {
            "freq": cfg.expt.frequency_p,
            "gain": cfg.expt.gain_p,
            "type": cfg.expt.pulse_type_p,
            "sigma": cfg.expt.length_p,
            "phase": 0,
            "chan": self.pump_ch,
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

        # If checking EF transition, create a pi pulse for |g>-|e> transition.
        # pi_ge is also what QickProgram.reset() plays, so it has to be
        # declared whenever a measurement-based reset is on.
        if cfg.expt.checkEF or reset_plays_pi_pulses(cfg.expt):
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
        # reset_f plays pi_ef ahead of each pi_ge, so it needs the pulse too.
        if cfg.expt.reset_f:
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

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
            self.pulse(ch=self.pump_ch, name="qubit_pulse", t=0)
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
    - 'reset_mode': 'passive' | 'active' | 'f0g1'

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
    - 'reset_mode': 'passive'
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
            "pulse_type_p": pulse_type_from_config(self.cfg.device.get("fwm"), "pump_pulse_type"),
            "gain_b": params_def['gain_p'],
            "frequency_b": self.cfg.device.readout.frequency[0],
            "pulse_type_b": pulse_type_from_config(self.cfg.device.get("fwm"), "buffer_pulse_type"),
            'start':1,
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            # reset is the round count for reset_mode "active" only;
            # QickProgram.reset() plays up to that many conditional pi pulses.
            # setup_reset keeps the long passive inter-shot delay whatever the
            # mode, so a reset that is not working cannot quietly corrupt the
            # data.
            # Which reset runs between shots: "passive" (the full T1
            # wait), "active" (measurement-based conditional pi pulses)
            # or "f0g1" (the unconditional all-microwave reset, which
            # also clears |f> -- see QickProgram.reset_f0g1).
            "reset_mode": "passive",
            # "f0g1" only, and all optional -- everything else is
            # derived from device.f0g1. residual sets the drive length
            # as ln(1/residual)/(2*pi*Gamma), so it rescales on its own
            # as Gamma improves; f0g1_reset_length pins the length
            # instead. None means device.f0g1.reset_*, else derived.
            "f0g1_reset_residual": None,
            "f0g1_reset_gain": None,
            "f0g1_reset_gain_ef": None,
            "f0g1_reset_length": None,
            "f0g1_reset_gap": None,
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            # Calibration only: re-measure after the final pi pulse so the
            # last recorded round is the true post-reset state. Costs one
            # readout per shot and nothing acts on it, so leave it off in use.
            "reset_verify": False,
            # Add a pi_ef ahead of each pi_ge, for sequences that can leave
            # population in |f>. Needs reset >= 2: |f> clears in one round but
            # |e> is driven to |f> first and clears on the next.
            "reset_f": False,
            # "while excited" semantics at a fixed trip count: once a round
            # reads ground, a latch suppresses every later pi pulse. Measures
            # every round, so the trigger count stays fixed.
            "reset_latch": False,
            # Mean rounds the while loop takes, for the cycle-time estimate
            # only; `reset` is its hard cap. See CLAUDE.md.
            "reset_while_typical": 1,
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

        # configure_reset() derives threshold / read_wait / extra_delay from
        # device.readout, which QickProgram.reset() needs. It runs after
        # check_params so those derived keys are not reported as unexpected,
        # and here rather than in qubit_run() because the scripts use go=False.
        if reset_mode_of(self.cfg.expt) != "passive":
            super().configure_reset()

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
            self.add_p_e(data)
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
        title = f"FWM sweep Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain_p})"
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
        # The pump goes on the pump line, which is the qubit channel unless
        # hw.soc.dacs.pump says otherwise.
        qubit_pulse = {
            "freq": cfg.expt.frequency_p,
            "gain": cfg.expt.gain_p,
            "type": cfg.expt.pulse_type_p,
            "sigma": cfg.expt.length_p,
            "phase": 0,
            "chan": self.pump_ch,
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

        # If checking EF transition, create a pi pulse for |g>-|e> transition.
        # pi_ge is also what QickProgram.reset() plays, so it has to be
        # declared whenever a measurement-based reset is on.
        if cfg.expt.checkEF or reset_plays_pi_pulses(cfg.expt):
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
        # reset_f plays pi_ef ahead of each pi_ge, so it needs the pulse too.
        if cfg.expt.reset_f:
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

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
            self.pulse(ch=self.pump_ch, name="qubit_pulse", t=0)
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
    - 'reset_mode': 'passive' | 'active' | 'f0g1'

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
    - 'reset_mode': 'passive'
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
            "pulse_type_p": pulse_type_from_config(self.cfg.device.get("fwm"), "pump_pulse_type"),
            "length_b": 2,
            "frequency_b": self.cfg.device.readout.frequency[0],
            "pulse_type_b": pulse_type_from_config(self.cfg.device.get("fwm"), "buffer_pulse_type"),
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            # reset is the round count for reset_mode "active" only;
            # QickProgram.reset() plays up to that many conditional pi pulses.
            # setup_reset keeps the long passive inter-shot delay whatever the
            # mode, so a reset that is not working cannot quietly corrupt the
            # data.
            # Which reset runs between shots: "passive" (the full T1
            # wait), "active" (measurement-based conditional pi pulses)
            # or "f0g1" (the unconditional all-microwave reset, which
            # also clears |f> -- see QickProgram.reset_f0g1).
            "reset_mode": "passive",
            # "f0g1" only, and all optional -- everything else is
            # derived from device.f0g1. residual sets the drive length
            # as ln(1/residual)/(2*pi*Gamma), so it rescales on its own
            # as Gamma improves; f0g1_reset_length pins the length
            # instead. None means device.f0g1.reset_*, else derived.
            "f0g1_reset_residual": None,
            "f0g1_reset_gain": None,
            "f0g1_reset_gain_ef": None,
            "f0g1_reset_length": None,
            "f0g1_reset_gap": None,
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            # Calibration only: re-measure after the final pi pulse so the
            # last recorded round is the true post-reset state. Costs one
            # readout per shot and nothing acts on it, so leave it off in use.
            "reset_verify": False,
            # Add a pi_ef ahead of each pi_ge, for sequences that can leave
            # population in |f>. Needs reset >= 2: |f> clears in one round but
            # |e> is driven to |f> first and clears on the next.
            "reset_f": False,
            # "while excited" semantics at a fixed trip count: once a round
            # reads ground, a latch suppresses every later pi pulse. Measures
            # every round, so the trigger count stays fixed.
            "reset_latch": False,
            # Mean rounds the while loop takes, for the cycle-time estimate
            # only; `reset` is its hard cap. See CLAUDE.md.
            "reset_while_typical": 1,
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

        # configure_reset() derives threshold / read_wait / extra_delay from
        # device.readout, which QickProgram.reset() needs. It runs after
        # check_params so those derived keys are not reported as unexpected,
        # and here rather than in qubit_run() because the scripts use go=False.
        if reset_mode_of(self.cfg.expt) != "passive":
            super().configure_reset()

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
            self.add_p_e(data)
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
        title = f"FWM sweep Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain_p})"
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
    FWM sweep over buffer gain and length

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
        prefix = f"FWM_sweep_lengthb_gainb_{ef}{style}_qubit{qi}"
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
        nstark = self.cfg.device.readout.get("nstark", None)
        params_def2 = {
            "length_b_start": 0.1,
            "length_b_end": 10,
            "length_b_expts": 20,
            "save_shots": False,
            # Efficiency reduction: photons per unit buffer gain**2, from Stark
            # spectroscopy. Without it the buffer gain cannot be converted to a
            # photon probability, and the efficiency analysis is skipped.
            "n_photon_per_gain": nstark[qi] if nstark is not None else None,
            "slope_lo": 0.05,
            "slope_hi": 0.2,
            "length_b_min": 40.0,
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

        Thresholds the shots into P(e), then reduces each buffer length to a
        conversion efficiency: the slope of P(e) against the incident-photon
        probability 1 - P(0), measured in the linear window
        [slope_lo, slope_hi]. See `_efficiency_from_pe`.

        Args:
            fit: Whether to fit the data
            **kwargs: Additional arguments for the fit

        Returns:
            Analyzed data. When the photon calibration is available this adds
            'slopes' (efficiency at each buffer length), 'efficiency' (their
            average past length_b_min) and 'photon_prob' (the converted gain
            axis).
        """
        if data is None:
            data = self.data

        # if fit:
        #     # Fit each frequency slice to a Lorentzian model
        #     super().analyze(fitterfunc=FITTER_FUNC, fitfunc=FIT_FUNC)

        if 'shots' in data:
            self.add_p_e(data, stacked=True)


            if not self.cfg.expt.save_shots:
                data.pop("shots")  # Remove raw shots from data to save space

        if 'p_e' in data:
            n_photon = self.cfg.expt.n_photon_per_gain
            if n_photon is None:
                print(
                    "Skipping efficiency analysis: n_photon_per_gain is unset, "
                    "so buffer gain cannot be converted to photon number. Run "
                    "17_stark_spectroscopy.py with update() to populate "
                    "device.readout.nstark, or pass n_photon_per_gain."
                )
            else:
                data.update(
                    _efficiency_from_pe(
                        data['xpts'],
                        data['length_b_pts'],
                        data['p_e'],
                        n_photon,
                        slope_lo=self.cfg.expt.slope_lo,
                        slope_hi=self.cfg.expt.slope_hi,
                        length_b_min=self.cfg.expt.length_b_min,
                    )
                )
                if data["efficiency_fallback"]:
                    window = (
                        "mean slope over all buffer lengths -- none exceed "
                        f"length_b_min={self.cfg.expt.length_b_min:g} us"
                    )
                else:
                    window = (
                        f"mean slope past {self.cfg.expt.length_b_min:g} us"
                    )
                print(
                    f"Efficiency {data['efficiency']:.4f} ({window}); "
                    f"peak {np.nanmax(data['slopes']):.4f} at "
                    f"{data['length_b_pts'][np.nanargmax(data['slopes'])]:.2f} us"
                )

        return self.data

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):
        """
        Display the results of the 2D pulse probe spectroscopy experiment.

        Creates 2D color plots showing amplitude, phase, and (if available)
        the excited-state population P(e) as a function of buffer gain and
        buffer length.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_amps: Whether to plot amplitude data (vs. phase)
            ax: Matplotlib axis to plot on
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data

        # Set up plot title
        title = f"pump gain = {self.cfg.expt.gain_p}"
        if self.cfg.expt.checkEF:
            title = f"EF " + title

        self.display_pe(
            data=data,
            ax=ax,
            plot_amps=plot_amps,
            title=title,
            fit=fit,
            **kwargs,
        )

        if "slopes" in data:
            self.display_efficiency(data=data)

    def display_efficiency(self, data=None, ax=None):
        """
        Plot conversion efficiency against buffer length.

        Each point is the slope of P(e) versus 1 - P(0) at that buffer length;
        the dashed line marks the average past `length_b_min` that the single
        efficiency number is taken from.

        Args:
            data: Data to display (if None, use self.data)
            ax: Matplotlib axis to plot on, or None to create a figure
        """
        if data is None:
            data = self.data

        save_fig = ax is None
        if save_fig:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))

        length_b = data["length_b_pts"]
        length_b_min = self.cfg.expt.length_b_min
        fallback = data.get("efficiency_fallback", False)

        ax.plot(length_b, data["slopes"], marker="o", ms=3, lw=1)
        if fallback:
            mean_label = f"mean (all lengths) = {data['efficiency']:.3f}"
        else:
            mean_label = f"mean past {length_b_min:g} us = {data['efficiency']:.3f}"
        ax.axhline(data["efficiency"], color="red", ls="--", lw=1, label=mean_label)
        if not fallback:
            ax.axvline(length_b_min, color="grey", ls=":", lw=1)
        ax.set_xlabel("buffer length (us)")
        ax.set_ylabel("efficiency (slope of P(e) vs 1 - P(0))")
        ax.legend(fontsize="small")

        if save_fig:
            # save_fig applies tight_layout and shows the figure; the config
            # was already written alongside the P(e) panels.
            self.save_fig(fig, suffix="_efficiency")


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
        # The pump goes on the pump line, which is the qubit channel unless
        # hw.soc.dacs.pump says otherwise.
        qubit_pulse = {
            "freq": cfg.expt.frequency_p,
            "gain": cfg.expt.gain_p,
            "type": cfg.expt.pulse_type_p,
            "sigma": cfg.expt.length_p,
            "phase": 0,
            "chan": self.pump_ch,
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
            self.pulse(ch=self.pump_ch, name="qubit_pulse", t=0)

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
    - 'reset_mode': 'passive' | 'active' | 'f0g1'

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
    - 'reset_mode': 'passive'
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
        prefix = f"FWM_sweep_tp_{ef}{style}_qubit{qi}"
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
            "pulse_type_p": pulse_type_from_config(self.cfg.device.get("fwm"), "pump_pulse_type"),
            "start":1,
            "stop":30,
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            # reset is the round count for reset_mode "active" only;
            # QickProgram.reset() plays up to that many conditional pi pulses.
            # setup_reset keeps the long passive inter-shot delay whatever the
            # mode, so a reset that is not working cannot quietly corrupt the
            # data.
            # Which reset runs between shots: "passive" (the full T1
            # wait), "active" (measurement-based conditional pi pulses)
            # or "f0g1" (the unconditional all-microwave reset, which
            # also clears |f> -- see QickProgram.reset_f0g1).
            "reset_mode": "passive",
            # "f0g1" only, and all optional -- everything else is
            # derived from device.f0g1. residual sets the drive length
            # as ln(1/residual)/(2*pi*Gamma), so it rescales on its own
            # as Gamma improves; f0g1_reset_length pins the length
            # instead. None means device.f0g1.reset_*, else derived.
            "f0g1_reset_residual": None,
            "f0g1_reset_gain": None,
            "f0g1_reset_gain_ef": None,
            "f0g1_reset_length": None,
            "f0g1_reset_gap": None,
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            # Calibration only: re-measure after the final pi pulse so the
            # last recorded round is the true post-reset state. Costs one
            # readout per shot and nothing acts on it, so leave it off in use.
            "reset_verify": False,
            # Add a pi_ef ahead of each pi_ge, for sequences that can leave
            # population in |f>. Needs reset >= 2: |f> clears in one round but
            # |e> is driven to |f> first and clears on the next.
            "reset_f": False,
            # "while excited" semantics at a fixed trip count: once a round
            # reads ground, a latch suppresses every later pi pulse. Measures
            # every round, so the trigger count stays fixed.
            "reset_latch": False,
            # Mean rounds the while loop takes, for the cycle-time estimate
            # only; `reset` is its hard cap. See CLAUDE.md.
            "reset_while_typical": 1,
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

        # Check for unexpected parameters
        super().check_params(params_def2)

        # configure_reset() derives threshold / read_wait / extra_delay from
        # device.readout, which QickProgram.reset() needs. It runs after
        # check_params so those derived keys are not reported as unexpected,
        # and here rather than in qubit_run() because the scripts use go=False.
        if reset_mode_of(self.cfg.expt) != "passive":
            super().configure_reset()

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
            self.add_p_e(data)
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
        title = f"FWM sweep Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain_p})"
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


def _efficiency_from_pe(gain_b, length_b, p_e, n_photon_per_gain,
                        slope_lo=0.05, slope_hi=0.2, length_b_min=40.0):
    """
    Reduce one buffer gain/length P(e) map to a single conversion efficiency.

    The buffer gain is first converted to an incident-photon probability using
    Poisson statistics: a coherent buffer tone of gain g carries
    n = g**2 * n_photon_per_gain photons on average, so the probability of at
    least one photon is 1 - P(0) = 1 - exp(-n). The FWM conversion efficiency
    is the slope of P(e) against that probability, taken in the low-occupancy
    window [slope_lo, slope_hi] where the response is still linear.

    The slope is measured independently at every buffer length and then
    averaged over the lengths above `length_b_min`, where the conversion has
    saturated and the slope has plateaued.

    Args:
        gain_b: 1D array of buffer gains (the inner sweep's x axis)
        length_b: 1D array of buffer lengths (us)
        p_e: 2D array of excited-state population, shape
            (len(length_b), len(gain_b))
        n_photon_per_gain: Photons per unit buffer gain squared, i.e. the
            `nstark` calibration from Stark spectroscopy
        slope_lo, slope_hi: 1 - P(0) window used for the linear fit
        length_b_min: Only buffer lengths above this (us) enter the average

    Returns:
        dict with 'efficiency' (scalar), 'slopes' (per length_b) and
        'photon_prob' (the converted gain axis)
    """
    gain_b = np.asarray(gain_b)
    length_b = np.asarray(length_b)
    p_e = np.asarray(p_e)

    n_photons = gain_b**2 * n_photon_per_gain
    photon_prob = 1 - np.exp(-n_photons)  # == 1 - poisson.pmf(0, n_photons)

    idx_lo = np.argmin(np.abs(photon_prob - slope_lo))
    idx_hi = np.argmin(np.abs(photon_prob - slope_hi))

    slopes = np.full(len(length_b), np.nan)
    if idx_hi > idx_lo:
        for j in range(len(length_b)):
            slopes[j] = np.polyfit(
                photon_prob[idx_lo:idx_hi], p_e[j, idx_lo:idx_hi], 1
            )[0]

    late = length_b > length_b_min
    fallback = not late.any()
    if fallback:
        # Buffer never reaches the plateau window; fall back to all lengths so
        # a short scan still yields a number rather than a silent NaN.
        late = np.ones(len(length_b), dtype=bool)

    with np.errstate(invalid="ignore"):
        efficiency = float(np.nanmean(slopes[late]))

    return {
        "efficiency": efficiency,
        "slopes": slopes,
        "photon_prob": photon_prob,
        # True when no buffer length passed length_b_min, so `efficiency` is an
        # average over every length rather than just the plateau.
        "efficiency_fallback": fallback,
    }


def _efficiency_model(gain_p, scale, kappa_ratio, amplitude):
    """Four-wave-mixing efficiency versus pump gain; see fit_efficiency_vs_power."""
    cooperativity = (gain_p / scale) ** 2
    return 4 * amplitude * cooperativity / (1 + kappa_ratio + cooperativity) ** 2


def fit_efficiency_vs_power(gain_p, efficiency):
    """
    Fit conversion efficiency versus pump gain to the four-wave-mixing form.

        eta = 4*A*C / (1 + kappa_i/kappa_c + C)**2,    C = (gain_p / scale)**2

    The cooperativity C grows as the square of the pump amplitude, so
    efficiency peaks at matched coupling and falls off again once the pump
    overdrives the conversion.

    Returns (popt, perr) with popt = [scale, kappa_ratio, A], or (None, None)
    if the fit does not converge.
    """
    from scipy.optimize import curve_fit

    gain_p = np.asarray(gain_p, dtype=float)
    efficiency = np.asarray(efficiency, dtype=float)
    good = np.isfinite(gain_p) & np.isfinite(efficiency)
    if good.sum() < 4:
        return None, None

    x, y = gain_p[good], efficiency[good]
    scale_guess = x[np.argmax(y)] * 1.7
    if scale_guess <= 0:
        scale_guess = float(np.max(x))
    try:
        popt, pcov = curve_fit(
            _efficiency_model,
            x,
            y,
            p0=[scale_guess, 0.08, np.max(y)],
            bounds=(0, np.inf),
        )
    except Exception:
        return None, None
    return popt, np.sqrt(np.diag(pcov))


class FWMEfficiencyVsPower(QickExperiment):
    """
    FWM conversion efficiency versus pump power, walking the fitted band.

    At each pump power this runs a full buffer gain/length sweep
    (`FWM_gainb_tb_sweep`) and reduces the resulting P(e) map to a single
    efficiency number, giving efficiency as a function of pump gain.

    The pump frequency is not a free axis: the pump must stay on the FWM band,
    whose position moves with pump power. So the pump gain is stepped and the
    matching frequency is read off the band fit that `18_fwm_power.py` stores
    in `device.fwm`:

        quadratic:         frequency_p = a * gain_p**2 + c
        linear:            frequency_p = a * gain_p + c
        linear_quadratic:  frequency_p = a * gain_p**2 + b * gain_p + c
        cubic:             frequency_p = a * gain_p**3 + b * gain_p**2 + c * gain_p + d
        quartic:           frequency_p = a * gain_p**4 + b * gain_p**3 + c * gain_p**2 + d * gain_p + e

    Parameters:
    - 'pump_gain_start', 'pump_gain_stop', 'pump_expts': the pump gain walk
    - 'pump_power_fit', 'pump_power_fit_type': band fit, defaults from config
    - 'n_photon_per_gain': photons per unit gain**2, defaults to
      device.readout.nstark[qi] (written by Stark spectroscopy)
    - 'slope_lo', 'slope_hi': 1 - P(0) window for the efficiency slope
    - 'length_b_min': buffer lengths above this (us) enter the slope average
    - everything else is forwarded to `FWM_gainb_tb_sweep`

    Default parameters are defined in the `__init__` method.
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=True,
        qi=0,
        go=True,
        params={},
        display=True,
        min_r2=None,
        max_err=None,
    ):
        """
        Initialize the efficiency-versus-pump-power experiment.

        Args:
            cfg_dict: Configuration dictionary
            prefix: Prefix for data files
            progress: Whether to show progress bar
            qi: Qubit index
            go: Whether to immediately run the experiment
            params: Additional parameters to override defaults
            display: Whether to display results
            min_r2: Minimum R2 value for fit quality
            max_err: Maximum error for fit quality
        """
        prefix = f"FWM_efficiency_vs_power_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        fwm_cfg = self.cfg.device.get("fwm", {})
        nstark = self.cfg.device.readout.get("nstark", None)

        params_def = {
            # Pump power walk along the band.
            "pump_gain_start": 0.05,
            "pump_gain_stop": self.cfg.device.qubit.max_gain,
            "pump_expts": 10,
            "pump_power_fit": fwm_cfg.get("pump_power_fit", None),
            "pump_power_fit_type": fwm_cfg.get("pump_power_fit_type", None),
            # Efficiency reduction.
            "n_photon_per_gain": nstark[qi] if nstark is not None else None,
            "slope_lo": 0.05,
            "slope_hi": 0.2,
            "length_b_min": 40.0,
            # Forwarded to the nested buffer gain/length sweep.
            "start": 0.001,
            "span": 0.2,
            "expts": 50,
            "length_p": 50,
            "length_b_start": 0.1,
            "length_b_end": 50,
            "length_b_expts": 50,
            "frequency_b": self.cfg.device.readout.frequency[0],
            "reps": 1000,
            "save_shots": False,
        }
        params = {**params_def, **params}

        if params["pump_power_fit"] is None or params["pump_power_fit_type"] is None:
            raise ValueError(
                "No FWM band fit in the config. Run 18_fwm_power.py with "
                "update() first, or pass pump_power_fit and "
                "pump_power_fit_type explicitly."
            )

        # Build the nested buffer sweep once and retune its pump at each step,
        # so the whole walk lands in a single data file.
        self.expt = FWM_gainb_tb_sweep(cfg_dict, qi=qi, go=False, params=params)
        # The nested sweep would otherwise rewrite its own file on every pump
        # step, leaving only the last one behind; this experiment saves the
        # full walk itself.
        self.expt.save_interim = False

        self.cfg.expt = {**self.expt.cfg.expt, **params}

        if go:
            self.run(progress=progress, display=display, min_r2=min_r2, max_err=max_err)

    def _pump_points(self):
        """Pump gains to visit and the band frequency at each, from the fit."""
        gains = np.linspace(
            self.cfg.expt.pump_gain_start,
            self.cfg.expt.pump_gain_stop,
            self.cfg.expt.pump_expts,
        )
        freqs = band_frequency_from_fit(
            self.cfg.expt.pump_power_fit,
            self.cfg.expt.pump_power_fit_type,
            gains,
        )
        return gains, freqs

    def _set_pump(self, gain_p, frequency_p):
        """
        Point the pump at one spot on the band.

        The value has to reach the innermost experiment (`FWM_gainb_sweep`),
        since that is the config the Program actually reads --
        QickExperiment2DSimple.acquire only propagates its own y-sweep
        variable. The middle level is set too so its display title matches.
        """
        for cfg in (self.expt.cfg.expt, self.expt.expt.cfg.expt):
            cfg["gain_p"] = gain_p
            cfg["frequency_p"] = frequency_p

    def acquire(self, progress=False):
        """
        Walk the band, running a buffer gain/length sweep at each pump power.

        Args:
            progress: Whether to show progress bar

        Returns:
            Acquired data
        """
        gains, freqs = self._pump_points()

        self.xlabel = "buffer gain"
        self.ylabel = "buffer length (us)"

        data = {
            "gain_p_pts": gains,
            "frequency_p_pts": freqs,
            "p_e": [],
            "length_b_pts": [],
            "xpts": [],
            "time": [],
        }

        for gain_p, frequency_p in zip(tqdm(gains, disable=not progress), freqs):
            self._set_pump(gain_p, frequency_p)
            inner = self.expt.acquire(progress=False)
            inner = self.expt.analyze(inner, fit=False)

            if "p_e" not in inner:
                raise RuntimeError(
                    "Nested buffer sweep produced no P(e); it must collect "
                    "shots for the efficiency reduction to work."
                )

            data["p_e"].append(inner["p_e"])
            data["length_b_pts"].append(inner["length_b_pts"])
            data["xpts"].append(inner["xpts"])
            data["time"].append(time.time())

        # The buffer axes are identical at every pump power, so collapse them.
        data["length_b_pts"] = np.array(data["length_b_pts"][0])
        data["xpts"] = np.array(data["xpts"][0])
        for key in ("p_e", "time", "gain_p_pts", "frequency_p_pts"):
            data[key] = np.array(data[key])

        self.data = data
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Reduce each pump power's P(e) map to an efficiency, then fit the trend.

        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit efficiency versus pump gain
            **kwargs: Unused, accepted for interface compatibility

        Returns:
            Analyzed data with 'efficiency', 'slopes', 'photon_prob' and, when
            the fit converges, 'eff_fit' / 'eff_fit_err'
        """
        if data is None:
            data = self.data

        n_photon = self.cfg.expt.n_photon_per_gain
        if n_photon is None:
            raise ValueError(
                "n_photon_per_gain is unset, so buffer gain cannot be "
                "converted to photon number. Run 17_stark_spectroscopy.py with "
                "update() to populate device.readout.nstark, or pass "
                "n_photon_per_gain explicitly."
            )

        efficiency = np.zeros(len(data["gain_p_pts"]))
        slopes = np.zeros((len(data["gain_p_pts"]), len(data["length_b_pts"])))
        for i in range(len(data["gain_p_pts"])):
            reduced = _efficiency_from_pe(
                data["xpts"],
                data["length_b_pts"],
                data["p_e"][i],
                n_photon,
                slope_lo=self.cfg.expt.slope_lo,
                slope_hi=self.cfg.expt.slope_hi,
                length_b_min=self.cfg.expt.length_b_min,
            )
            efficiency[i] = reduced["efficiency"]
            slopes[i] = reduced["slopes"]
            data["photon_prob"] = reduced["photon_prob"]

        data["efficiency"] = efficiency
        data["slopes"] = slopes

        best = int(np.nanargmax(efficiency)) if np.isfinite(efficiency).any() else 0
        data["best_idx"] = best
        print(
            f"Peak efficiency {efficiency[best]:.4f} at pump gain "
            f"{data['gain_p_pts'][best]:.4f}, frequency "
            f"{data['frequency_p_pts'][best]:.3f} MHz"
        )

        if fit:
            popt, perr = fit_efficiency_vs_power(data["gain_p_pts"], efficiency)
            if popt is None:
                print("Efficiency vs power fit did not converge")
            else:
                data["eff_fit"] = popt
                data["eff_fit_err"] = perr
                print(
                    f"Efficiency fit: scale={popt[0]:.4f}, "
                    f"kappa_i/kappa_c={popt[1]:.4f}, A={popt[2]:.4f}"
                )

        return data

    def display(self, data=None, fit=True, ax=None, **kwargs):
        """
        Plot efficiency versus pump gain, with the per-length slopes alongside.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to overlay the efficiency model fit
            ax: Sequence of two axes to plot into, or None to create a figure
            **kwargs: Unused, accepted for interface compatibility
        """
        if data is None:
            data = self.data

        save_fig = ax is None
        if save_fig:
            fig, (ax_eff, ax_slope) = plt.subplots(1, 2, figsize=(11, 4.5))
        else:
            ax_eff, ax_slope = ax

        gains = data["gain_p_pts"]
        ax_eff.plot(gains, data["efficiency"], marker="o", ls="none", label="measured")
        if fit and "eff_fit" in data:
            dense = np.linspace(0, gains.max(), 200)
            popt = data["eff_fit"]
            ax_eff.plot(
                dense,
                _efficiency_model(dense, *popt),
                color="red",
                lw=1,
                label=(
                    r"$4\mathcal{C}/(1+\kappa_i/\kappa_c+\mathcal{C})^2$"
                    "\n"
                    f"scale={popt[0]:.3f}, "
                    r"$\kappa_i/\kappa_c$="
                    f"{popt[1]:.3f}, A={popt[2]:.3f}"
                ),
            )
        ax_eff.set_xlabel("pump gain (DAC level)")
        ax_eff.set_ylabel("efficiency")
        ax_eff.set_xlim(left=0)
        ax_eff.legend(fontsize="small")

        # Slope versus buffer length for every pump power, so the plateau the
        # efficiency average relies on is visible.
        cmap = plt.get_cmap("viridis")
        n_gain = len(gains)
        for i, gain_p in enumerate(gains):
            label = None
            if i in (0, n_gain - 1):
                label = f"{gain_p:.3f}"
            ax_slope.plot(
                data["length_b_pts"],
                data["slopes"][i],
                color=cmap(i / max(n_gain - 1, 1)),
                lw=1,
                label=label,
            )
        ax_slope.axvline(
            self.cfg.expt.length_b_min,
            color="grey",
            ls="--",
            lw=1,
            label=f"average above {self.cfg.expt.length_b_min:g} us",
        )
        ax_slope.set_xlabel("buffer length (us)")
        ax_slope.set_ylabel("slope of P(e) vs 1 - P(0)")
        ax_slope.legend(fontsize="small", title="pump gain")

        if save_fig:
            fig.tight_layout()
            self.save_fig(fig)
            self.save_config()
            plt.show()
