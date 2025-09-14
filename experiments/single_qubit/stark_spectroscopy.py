"""
Stark Spectroscopy Experiment

This module implements AC Stark spectroscopy experiments for qubit characterization.
Stark spectroscopy measures the shift in qubit frequency due to the presence of a
drive field (Stark pulse) applied to the readout resonator. This allows measurement
of the dispersive shift and other qubit-resonator coupling parameters.

The module includes:
- QubitStarkSpecProgram: Defines the pulse sequence for the Stark spectroscopy experiment
- StarkSpec: Main experiment class for Stark spectroscopy
"""

import numpy as np
import matplotlib.pyplot as plt
from qick.asm_v2 import QickSweep1D
from qick import *
from scipy.optimize import curve_fit


from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment2DSweep
from ..general.qick_program import QickProgram


from ...analysis import fitting as fitter


class QubitStarkSpecProgram(QickProgram):
    """
    Defines the pulse sequence for a Stark spectroscopy experiment.

    The sequence consists of:
    1. Simultaneous qubit pulse (variable frequency) and Stark pulse (fixed frequency)
    2. Wait time
    3. Measurement
    """

    def __init__(self, soccfg, final_delay, cfg):
        """
        Initialize the Stark spectroscopy program.

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

        # Initialize with standard readout
        super()._initialize(cfg, readout="standard")

        # Define the qubit pulse with variable frequency
        pulse = {
            "freq": cfg.expt.frequency,
            "gain": cfg.expt.gain,
            "type": cfg.expt.pulse_type,
            "sigma": cfg.expt.length,
            "phase": 0,
        }
        super().make_pulse(pulse, "qubit_pulse")

        # Define the Stark pulse applied to the resonator
        stark_pulse = {
            "chan": self.res_ch,
            "freq": cfg.expt.stark_freq,
            "gain": cfg.expt.stark_gain,
            "type": cfg.expt.pulse_type,
            "sigma": cfg.expt.length,
            "phase": 0,
        }
        super().make_pulse(stark_pulse, "stark_pulse")

        # Add sweep loops for Stark gain and qubit frequency
        self.add_loop("stark_loop", cfg.expt.stark_expts)
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

        # Apply qubit and Stark pulses simultaneously
        self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        self.pulse(ch=self.res_ch, name="stark_pulse", t=0)

        # Wait before measurement
        self.delay_auto(t=0.01, tag="wait")

        # Perform measurement
        super().measure(cfg)


class StarkSpec(QickExperiment2DSweep):
    """
    Main experiment class for Stark spectroscopy.

    This class implements Stark spectroscopy by sweeping the qubit frequency while
    applying a Stark pulse to the readout resonator. The Stark pulse shifts the qubit
    frequency due to the AC Stark effect, allowing measurement of the dispersive shift.

    The default values for the parameters are determined by the selected style, which can be:
    - 'huge': Very wide frequency span with high power.
    - 'coarse': Wide frequency span with medium power.
    - 'medium': Medium frequency span with low power.
    - 'fine': Narrow frequency span with very low power.

    If no parameters are specified, the default values are:
    - 'rounds': As defined in the configuration.
    - 'final_delay': 10 μs
    - 'length': 5 μs
    - 'stark_expts': 30
    - 'df_stark': 0 MHz
    - 'max_stark_gain': 1
    - 'min_stark_gain': 0
    - 'df': 0 MHz
    - 'stark_rng': 15
    - 'pulse_type': 'const'
    - 'active_reset': False
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
    ):
        """
        Initialize the Stark spectroscopy experiment.

        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index
            go: Whether to run the experiment immediately
            params: Additional parameters to override defaults
            prefix: Prefix for data files
            progress: Whether to show progress bar
            display: Whether to display results
            style: Style of experiment ('huge', 'coarse', 'medium', or 'fine')
            min_r2: Minimum R² value for fit quality
            max_err: Maximum error for fit quality
        """
        # Currently no control of readout time; may want to change for simultaneious readout

        # Set prefix for data files
        prefix = f"stark_spectroscopy_{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

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

        # Additional default parameters
        params_def2 = {
            "rounds": self.rounds,
            "final_delay": 10,
            "length": 5,
            "stark_expts": 30,
            "df_stark": 0,
            "max_stark_gain": 1,
            "min_stark_gain": 0,
            "df": 0,
            "stark_rng": 15,
            "pulse_type": "const",
            "qubit": [qi],
            "active_reset": False,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params_def = {**params_def2, **params_def}

        # Merge default and user-provided parameters
        params = {**params_def, **params}

        # Calculate start frequency
        params_def["start"] = (
            self.cfg.device.qubit.f_ge[qi] - params["span"] / 2 + params["df"]
        )
        params = {**params_def, **params}

        # Set Stark pulse frequency
        params["stark_freq"] = (
            self.cfg.device.readout.frequency[qi] + params["df_stark"]
        )

        # Adjust pulse length if needed
        if params["length"] == "t1":
            # Set pulse length to T1/4 if specified as "t1"
            params["length"] = self.cfg.device.qubit.T1[qi] / 4
        if params["length"] > max_length:
            # Limit pulse length to maximum allowed
            params["length"] = max_length

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
        Acquire data for the Stark spectroscopy experiment.

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

        # Configure Stark gain sweep
        self.cfg.expt.stark_gain = QickSweep1D(
            "stark_loop", self.cfg.expt.min_stark_gain, self.cfg.expt.max_stark_gain
        )

        # Acquire data using the QubitStarkSpecProgram

        super().acquire(QubitStarkSpecProgram, progress=progress, get_hist=False)
        self.data["stark_gain"] = np.linspace(
            self.cfg.expt.max_stark_gain / self.cfg.expt.stark_rng,
            self.cfg.expt.max_stark_gain,
            self.cfg.expt.stark_expts,
        )
        self.data["ypts"] = self.data["stark_gain"]
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

        # Fit the data to a Lorentzian model
        self.fitterfunc = fitter.fitlor
        self.fitfunc = fitter.lorfunc
        ydata_lab = ["avgi"]  # Typically only fit I quadrature for speed

        # Fit each row (y value) separately
        for i, ydata in enumerate(ydata_lab):
            data["fit_" + ydata] = []
            data["fit_err_" + ydata] = []

            # Iterate through each y value
            for j in range(len(data["ypts"])):
                # Fit this row to the model function
                fit_pars, fit_err, init = self.fitterfunc(
                    data["xpts"], data[ydata][j], fitparams=None
                )
                # Store fit parameters and errors
                data["fit_" + ydata].append(fit_pars)
                data["fit_err_" + ydata].append(fit_err)

        # fit frequency of resonance moving
        f = [self.data["fit_avgi"][i][2] for i in range(len(self.data["fit_avgi"]))]
        self.data['f'] = f

        # Fit the data
        try:
            popt, pcov = curve_fit(quadratic, self.data["stark_gain"], f)
            self.data["popt"] = popt
            Delta = (
                self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
                - self.cfg.device.readout.frequency[self.cfg.expt.qubit[0]]
            )
            ng2 = popt[0] / 2 * Delta
            self.data["ng2"] = ng2
            print(f"ng2: {ng2}") # This is n photons * g**2
            
        except:
            print('Fit failed')

        # Store the fitted qubit frequency
        #        data["new_freq"] = data["best_fit"][2]

        return self.data

    def display(self, data=None, fit=True, ax=None, plot_all=True, **kwargs):
        """
        Display the results of the Stark spectroscopy experiment.

        This method is currently disabled (pass statement).
        When enabled, it would display the spectroscopy results with fit parameters.

        Args:
            fit: Whether to show the fit
            ax: Matplotlib axis to plot on
            plot_all: Whether to plot all data types
            **kwargs: Additional arguments for the display
        """
        # Display functionality is currently disabled

        # Example of how display could be implemented:
        xlabel = "Qubit Frequency (MHz)"

        title = (
            f"Stark Spectroscopy Q{self.cfg.expt.qubit[0]} (Gain {self.cfg.expt.gain})"
        )

        # Define which fit parameters to display in caption
        # Index 2 is frequency, index 3 is kappa

        super().display(
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=False,
            fitfunc=self.fitfunc,
            caption_params=[],  # Pass the new structured parameter list
        )

        # Plot the fitted curve
        plt.plot(self.data["stark_gain"], self.data["f"], "o")
        x_fit = np.linspace(
            min(self.data["stark_gain"]), max(self.data["stark_gain"]), 100
        )
        try:
            y_fit = quadratic(x_fit, *self.data["popt"])

            plt.plot(x_fit, y_fit, label="Quadratic Fit")
        except:
            print('No fit to plot')
        gain = self.cfg.device.readout.gain[self.cfg.expt.qubit[0]]
        plt.axvline(gain, color="k", linestyle="--", label="Readout gain")
        plt.legend()
        self.data['df_readout']=self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]] - quadratic(gain, *self.data["popt"])


# Define a quadratic function
@staticmethod
def quadratic(x, a, c):
    return a * x**2 + c
