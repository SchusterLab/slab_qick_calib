"""
Photon Number Splitting Experiment

This module implements simultaneous pulse in buffer and qubit spectroscopy, which is used to observe photon number splitting in the qubit spectrum due to the presence of photons in the readout resonator.
It modifies pulse probe spectroscopy by applying a simultaneous added pulse on the 
buffer channel while applying the qubit drive pulse.
**NOTE** 20260507:
- it's assumed that the buffer channel is the first DAC readout channel.
- the analysis function here is inherited from the qubit power spec which fits to a single Lorentzian. This is not ideal for the photon number splitting experiment since we want to use it to calibrate the average photon number. A better fit function should take the amplitudes of the 0 photon peak and do an exponential fit to extract the average photon number. This will be implemented in the future.

The module includes:
- PhotonNumberSplittingProgram: Defines the pulse sequence for the spectroscopy experiment
- PhotonNumberSplitting: Main experiment class for frequency spectroscopy
- PhotonNumberSplittingPower: 2D version that sweeps both qubit frequency and added pulse gain

Examples:
photonNsplitting = meas.PhotonNumberSplittingPower(cfg_dict, qi=1, style='', 
                                params={'start':4360,
                                        'span':15,
                                        'reps':5000,
                                        'rounds':5,
                                        'expts':600,
                                        'qubit_gain': 0.0001,
                                        'log': False, # Use a linear sweep instead of logarithmic
                                        'start_buffer_gain': 0.00,
                                        'step_buffer_gain': 0.002,
                                        'expts_buffer_gain': 30,
                                        'length':10,
                                        'sep_readout':True})

"""

import numpy as np
from qick import *
from qick.asm_v2 import QickSweep1D

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment, QickExperiment2DSimple
from ..general.qick_program import QickProgram

from ...analysis import fitting as fitter


class PhotonNumberSplittingProgram(QickProgram):
    """
    Defines the pulse sequence for a simultaneous pulse spectroscopy experiment.

    The sequence consists of:
    1. Simultaneous variable frequency qubit pulse and buffer pulse on readout channel
    2. Optional delay (sep_readout)
    3. Measurement
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

        # Get readout parameters from config input arguments
        q = cfg.expt.qubit[0]
        self.frequency = cfg.device.readout.frequency[q]
        self.gain = cfg.device.readout.gain[q]
        self.readout_length = cfg.device.readout.readout_length[q]
        self.phase = cfg.device.readout.phase[q]

        # Initialize with standard readout
        super()._initialize(cfg, readout="standard")

        # Initialize the first DAC readout channel for the added/buffer pulse
        self.buffer_ch = cfg.hw.soc.dacs.readout.ch[0] ## TODO: change back to 0 for the real buffer. For the line below as well
        if q != 0:
            nqz = cfg.hw.soc.dacs.readout.nyquist[0] if "nyquist" in cfg.hw.soc.dacs.readout else 1
            self.declare_gen(ch=self.buffer_ch, nqz=nqz)

        # Define the qubit pulse with variable frequency
        pulse = {
            "freq": cfg.expt.frequency,
            "gain": cfg.expt.qubit_gain,
            "type": cfg.expt.pulse_type,
            "sigma": cfg.expt.length,
            "phase": 0,
        }
        super().make_pulse(pulse, "qubit_pulse")

        # Define the buffer pulse applied to the resonator (same length as qubit pulse)
        buffer_pulse = {
            "chan": self.buffer_ch,
            "freq": cfg.expt.buffer_freq,
            "gain": cfg.expt.buffer_gain,
            "type": cfg.expt.pulse_type,
            "sigma": cfg.expt.length,
            "phase": 0,
        }
        super().make_pulse(buffer_pulse, "buffer_pulse")

        # Add frequency sweep loop
        self.add_loop("freq_loop", cfg.expt.expts)

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

        # Apply the qubit probe pulse AND the buffer pulse on the readout channel simultaneously
        self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        self.pulse(ch=self.buffer_ch, name="buffer_pulse", t=0)

        # Add delay if separate readout is enabled
        if cfg.expt.sep_readout:
            self.delay_auto(t=0.01, tag="wait")

        # Perform measurement
        super().measure(cfg)


class PhotonNumberSplitting(QickExperiment):
    """
    Main experiment class for simultaneous pulse spectroscopy.
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
        print_cfg=False,
        check_params=True,
    ):
        """
        Initialize the simultaneous pulse spectroscopy experiment.
        """
        prefix = f"simultaneous_spectroscopy_{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, check_params=check_params, qi=qi)

        max_length = 100  # Based on qick error messages, but not investigated
        spec_gain = self.cfg.device.qubit.spec_gain[qi]
        low_gain = self.cfg.device.qubit.low_gain

        # Set style-specific parameters
        if style == "huge":
            # Very wide frequency span with high power
            params_def = {
                "qubit_gain": 80 * low_gain * spec_gain,
                "span": 1500,
                "expts": 1000,
                "reps": self.reps,
            }
        elif style == "coarse":
            # Wide frequency span with medium power
            params_def = {
                "qubit_gain": 20 * low_gain * spec_gain,
                "span": 500,
                "expts": 500,
                "reps": self.reps,
            }
        elif style == "medium":
            # Medium frequency span with low power
            params_def = {
                "qubit_gain": 5 * low_gain * spec_gain,
                "span": 50,
                "expts": 200,
                "reps": self.reps,
            }
        elif style == "fine":
            # Narrow frequency span with very low power
            params_def = {
                "qubit_gain": low_gain * spec_gain,
                "span": 5,
                "expts": 100,
                "reps": 2 * self.reps,
            }

        # Additional default parameters for the added pulse and the explicit readout
        params_def2 = {
            "rounds": self.rounds,
            "final_delay": 10,
            "length": 10,
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "readout_freq": self.cfg.device.readout.frequency[qi],
            "readout_gain": self.cfg.device.readout.gain[qi],
            "buffer_freq": self.cfg.device.readout.frequency[0],
            "buffer_gain": 0.01,  # Default value
            "pulse_type": "const",
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            "active_reset": False,
        }
        params_def = {**params_def2, **params_def}

        # Merge default and user-provided parameters
        params = {**params_def, **params}

        # Set start frequency
        params_def["start"] = self.cfg.device.qubit.f_ge[qi] - params["span"] / 2
        params = {**params_def, **params}

        # Adjust pulse length
        if params["length"] == "t1":
            params["length"] = 3 * self.cfg.device.qubit.T1[qi]  

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

        if print_cfg:
            super().print()
            go = False
        # Run the experiment if requested
        if go:
            super().run(
                min_r2=min_r2, max_err=max_err, display=display, progress=progress
            )

    def acquire(self, progress=False):
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay

        # Set parameter to sweep
        self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}

        # Configure frequency sweep
        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        super().acquire(PhotonNumberSplittingProgram, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            # Fit the data to a Lorentzian model
            fitterfunc = fitter.fitlor
            fitfunc = fitter.lorfunc
            super().analyze(fitfunc, fitterfunc, use_i=False)

            # Store the fitted qubit frequency
            data["new_freq"] = data["best_fit"][2]

        return self.data

    def display(self, fit=True, ax=None, plot_all=True, **kwargs):
        fitfunc = fitter.lorfunc
        xlabel = "Qubit Frequency (MHz)"
        title = f"Photon Number Splitting Q{self.cfg.expt.qubit[0]} (Qubit Gain {self.cfg.expt.qubit_gain:.4f}, Buffer Gain {self.cfg.expt.buffer_gain:.4f})"

        caption_params = [
            {"index": 2, "format": "$f$: {val:.6} MHz"},
            {"index": 3, "format": "$\kappa$: {val:.3} MHz"},
        ]

        super().display(
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=False,
            fitfunc=fitfunc,
            caption_params=caption_params, 
        )


class PhotonNumberSplittingPower(QickExperiment2DSimple):
    """
    2D simultaneous pulse spectroscopy experiment that sweeps both frequency and the added pulse's gain.
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
    ):
        prefix = f"photon_number_splitting_power_{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        # Set style-specific parameters
        if style == "coarse":
            params_def = {"span": 800, "expts": 500}
        elif style == "fine":
            params_def = {"span": 40, "expts": 100}
        else:
            params_def = {"span": 120, "expts": 200}

        params_def2 = {
            "reps": 2 * self.reps,
            "rng": 50,  
            "max_buffer_gain": self.cfg.device.readout.max_gain,
            "start_buffer_gain": 0.0,
            "step_buffer_gain": 0.05,
            "expts_buffer_gain": 10, 
            "log": True, 
        }

        params_def = {**params_def, **params_def2}
        params = {**params_def, **params}

        # Create a PhotonNumberSplitting experiment but don't run it
        exp_name = PhotonNumberSplitting
        self.expt = exp_name(cfg_dict, qi=qi, go=False, params=params, check_params=False)
        params = {**self.expt.cfg.expt, **params}
        
        self.cfg.expt = params

        if go:
            self.run(progress=progress, display=display)

    def acquire(self, progress=False):
        # Generate added gain points for the sweep
        if "log" in self.cfg.expt and self.cfg.expt.log == True:
            rng = self.cfg.expt.rng
            rat = rng ** (-1 / (self.cfg.expt["expts_buffer_gain"] - 1))
            max_gain = self.cfg.expt["max_buffer_gain"]
            gainpts = max_gain * rat ** (np.arange(self.cfg.expt["expts_buffer_gain"]))
        else:
            gainpts = self.cfg.expt["start_buffer_gain"] + self.cfg.expt[
                "step_buffer_gain"
            ] * np.arange(self.cfg.expt["expts_buffer_gain"])

        ysweep = [{"pts": gainpts, "var": "buffer_gain"}]

        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}

        super().acquire(ysweep, progress=progress)
        return self.data

    def analyze(self, fit=True, **kwargs):
        if fit:
            fitterfunc = fitter.fitlor
            try:
                super().analyze(fitterfunc=fitterfunc)
            except Exception as e:
                print(f"Fit failed, skipping analysis. Error: {e}")

        return self.data

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):
        title = f"Photon Number Splitting Power Sweep Q{self.cfg.expt.qubit[0]}"
        xlabel = "Qubit Frequency (MHz)"
        ylabel = "Buffer Pulse Gain (DAC level)"

        super().display(
            data=data,
            ax=ax,
            plot_amps=plot_amps,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            fit=fit,
            **kwargs,
        )