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


class QubitBufferSpecShotProgram(QickProgram):
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
        self.add_loop("freq_loop", cfg.expt.expts)
        self.add_loop("freq_b_loop", cfg.expt.expts_b)

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
        self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        self.pulse(ch=cfg.hw.soc.dacs.readout.ch[0], name="buffer_pulse", t=0)

        # Add delay if separate readout is enabled
        if cfg.expt.sep_readout:
            self.delay_auto(t=0.01, tag="wait")

        # If checking EF transition, apply second pi pulse to return to |g> for readout
        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait 2")

        # Perform measurement
        super().measure(cfg)


class QubitBufferSpecShot(QickExperiment):
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
        prefix = f"qubit_buffer_spectroscopy_{ef}{style}_qubit{qi}"
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
            "length_b": 10,
            "pulse_type_b": "const",
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            "active_reset": False,
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
        if params["length_b"] > max_length:
            params["length_b"] = max_length    

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
        self.cfg.expt.frequency_p = QickSweep1D(
            "freq_p_loop", self.cfg.expt.start_p, self.cfg.expt.start_p + self.cfg.expt.span_p
        )

        self.cfg.expt.frequency_b = QickSweep1D(
            "freq_b_loop", self.cfg.expt.start_b, self.cfg.expt.start_b + self.cfg.expt.span_b
        )

        # Acquire data using the QubitSpecProgram
        super().acquire(QubitSpecShotProgram, progress=progress,shots=True)

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

        if fit:
            # Fit the data to a Lorentzian model
            self.fitterfunc = FITTER_FUNC
            self.fitfunc = FIT_FUNC
            super().analyze(use_i=False)

            # Store the fitted qubit frequency
            data["new_freq"] = data["best_fit"][2]

        # # Perform initial histogram analysis
        # params, _ = helpers.analyze_single_shot_histograms(data=data, plot=False, span=span, verbose=verbose)
        # data.update(params)

        # # Perform detailed single-shot analysis with fitting
        # try:
        #     # Fit single-shot data
        #     data2, p, paramsg, paramse2 = helpers.fit_single_shot(data, plot=False)

        #     # Update data with fit results
        #     data.update(p)
        #     data["vhg"] = data2["vhg"]
        #     data["histg"] = data2["histg"]
        #     data["vhe"] = data2["vhe"]
        #     data["histe"] = data2["histe"]
        #     data["paramsg"] = paramsg
        #     data["shots"] = self.cfg.expt.shots
        #     data['e_mean'] = p['e_mean']
        #     data['g_mean'] = p['g_mean']
        #     dv = self.data['ve'] - self.data['vg']
        #     data['e_norm'] = (self.data['e_mean']-self.data['vg'])/dv

        #     data['g_norm'] = (self.data['g_mean']-self.data['vg'])/dv
        # except Exception as e:
        #     print(f"Fits failed: {str(e)}")

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
        title = f"Spectroscopy Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain_p})"
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

