import numpy as np
from qick import *
from qick.asm_v2 import QickSweep1D

from ...helpers import config
from ...analysis import fitting as fitter
from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment, QickExperiment2DSimple
from ..general.qick_program import QickProgram


"""
T1 Experiment Module

This module implements T1 relaxation time measurements for superconducting qubits.
T1 (energy relaxation time) is measured by:
1. Exciting the qubit to |1⟩ state with a π pulse
2. Waiting for a variable delay time
3. Measuring the qubit state
4. Fitting the decay of the |1⟩ state population to an exponential function

The module provides several experiment classes:
- T1Program: Low-level pulse sequence implementation
- T1Experiment: Standard T1 measurement
- T1_2D: 2D T1 measurement for stability analysis
"""


class T1Program(QickProgram):
    """
    T1Program: Implements the pulse sequence for T1 measurement

    This class defines the actual quantum program that runs on the QICK hardware.
    It creates the necessary pulses and timing for a T1 measurement:
    1. Apply a π pulse to excite the qubit to |1⟩ state
    2. Wait for a variable time (the wait_time parameter)
    3. Measure the qubit state

    Optional AC Stark shift can be applied during the wait time.
    """

    def __init__(self, soccfg, final_delay, cfg):
        """
        Initialize the T1 program

        Args:
            soccfg: SOC configuration
            final_delay: Delay after measurement before next experiment
            cfg: Configuration dictionary containing experiment parameters
        """
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """
        Initialize the program by setting up the pulse sequence

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)
        # Create a loop for sweeping the wait time
        self.add_loop("wait_loop", cfg.expt.expts)

        # Initialize standard readout
        super()._initialize(cfg, readout="standard")

        # Create a π pulse to excite the qubit from |0⟩ to |1⟩
        super().make_pi_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        # If AC Stark shift is enabled, create a constant pulse to apply during wait time
        if cfg.expt.acStark:
            pulse = {
                "sigma": cfg.expt.wait_time,  # Duration of the pulse
                "sigma_inc": 0,  # No increment in duration
                "freq": cfg.expt.stark_freq,  # Frequency of the AC Stark pulse
                "gain": cfg.expt.stark_gain,  # Amplitude of the AC Stark pulse
                "phase": 0,  # Phase of the pulse
                "type": "flat_top",  
            }
            super().make_pulse(pulse, "stark_pulse")

    def _body(self, cfg):
        """
        Define the main body of the pulse sequence

        This method implements the actual T1 measurement sequence:
        1. Apply π pulse to excite qubit
        2. Wait for variable time (with or without AC Stark)
        3. Measure qubit state

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)
        # Configure readout
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # Apply π pulse to excite qubit from |0⟩ to |1⟩
        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        # Handle wait time with or without AC Stark shift
        if cfg.expt.acStark:
            # Small buffer delay before AC Stark pulse
            self.delay_auto(t=0.01, tag="wait_stark")
            # Apply AC Stark pulse during wait time
            self.pulse(ch=self.qubit_ch, name="stark_pulse", t=0)
            # Small buffer delay after AC Stark pulse
            self.delay_auto(t=cfg.expt["end_wait"], tag="wait")
        else:
            # Simple delay for standard T1 measurement
            self.delay_auto(t=cfg.expt["wait_time"] + 0.01, tag="wait")

        # Measure the qubit state
        super().measure(cfg)


class T1Experiment(QickExperiment):
    """
    T1Experiment: Main class for running T1 relaxation time measurements

    This class handles the complete T1 experiment workflow:
    1. Setting up experiment parameters
    2. Running the T1Program to acquire data
    3. Analyzing the results by fitting to an exponential decay
    4. Displaying and saving the results

    Configuration parameters (self.cfg.expt: dict):
        - span (float): The total span of the wait time sweep in microseconds.
        - expts (int): The number of experiments to be performed.
        - reps (int): The number of repetitions for each experiment (inner loop)
        - rounds (int): The number of rounds for the experiment (outer loop)
        - qubit (int): The index of the qubit being used in the experiment.
        - qubit_chan (int): The channel of the qubit being read out.
        - acStark (bool): Whether to apply AC Stark shift during wait time
        - active_reset (bool): Whether to use active qubit reset

    Default values are defined in the `params_def` dictionary in the __init__ function.
    The default values are:
        - reps (int): 2 * self.reps
        - rounds (int): self.rounds
        - expts (int): 60
        - start (float): 0
        - span (float): 3.7 * self.cfg.device.qubit.T1[qi]
        - acStark (bool): False
        - active_reset (bool): self.cfg.device.readout.active_reset[qi]
        - qubit (list): [qi]
        - qubit_chan (int): self.cfg.hw.soc.adcs.readout.ch[qi]
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
        Initialize the T1 experiment

        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index to measure
            go: Whether to immediately run the experiment
            params: Additional parameters to override defaults
            prefix: Filename prefix for saved data
            progress: Whether to show progress bar
            style: Measurement style ('fine' for more averages, 'fast' for fewer points)
            disp_kwargs: Display options
            min_r2: Minimum R² value for acceptable fit
            max_err: Maximum error for acceptable fit
            display: Whether to display results
        """
        # Set default prefix based on qubit index if not provided
        if prefix is None:
            prefix = f"t1_qubit{qi}"

        super().__init__(
            cfg_dict=cfg_dict,
            prefix=prefix,
            fname=fname,
            progress=progress,
            qi=qi,
            check_params=check_params,
        )

        # Define default parameters
        params_def = {
            "reps": 2 * self.reps,  # Number of repetitions (inner loop)
            "rounds": self.rounds,  # Number of averages (outer loop)
            "expts": 60,  # Number of wait time points
            "start": 0,  # Start time for wait sweep (μs)
            "span": 3.7
            * self.cfg.device.qubit.T1[
                qi
            ],  # Total span of wait times (μs), set to ~3.7*T1
            "acStark": False,  # Whether to apply AC Stark shift
            "active_reset": self.cfg.device.readout.active_reset[
                qi
            ],  # Use active qubit reset
            "qubit": [qi],  # Qubit index as a list
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[
                qi
            ],  # Readout channel for this qubit
        }

        # Adjust parameters based on measurement style
        if style == "fine":
            params_def["rounds"] = (
                params_def["rounds"] * 2
            )  # Double averages for fine measurements
        elif style == "fast":
            params_def["expts"] = 30  # Fewer points for fast measurements

        # Merge default parameters with user-provided parameters
        self.cfg.expt = {**params_def, **params}
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
        Acquire T1 measurement data

        This method:
        1. Sets up the wait time sweep parameters
        2. Runs the T1Program to collect data for each wait time

        Args:
            progress: Whether to show progress bar
            debug: Whether to run in debug mode

        Returns:
            Measurement data dictionary
        """
        # Define parameter metadata for plotting
        self.param = {"label": "wait", "param": "t", "param_type": "time"}

        # Create a 1D sweep for the wait time from start to start+span
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        # Run the T1Program to acquire data
        super().acquire(T1Program, progress=progress)

        return self.data

    def analyze(self, data=None, **kwargs):
        """
        Analyze T1 measurement data by fitting to exponential decay

        Args:
            data: Data dictionary to analyze (uses self.data if None)
            **kwargs: Additional arguments passed to the analyzer

        Returns:
            Data dictionary with added fit results
        """
        if data is None:
            data = self.data

        # Fit to exponential decay function
        # fitparams=[y-offset, amp, x-offset, decay rate]
        self.fitfunc = fitter.expfunc  # Exponential decay function
        self.fitterfunc = fitter.fitexp  # Fitting function for exponential decay
        super().analyze(data, **kwargs)

        # Extract T1 time from fit parameters
        data["new_t1"] = data["best_fit"][2]  # T1 from combined I/Q fit
        data["new_t1_i"] = data["fit_avgi"][2]  # T1 from I quadrature fit
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
        Display T1 measurement results

        Creates a plot showing the qubit state vs wait time and the exponential fit

        Args:
            data: Data dictionary to display (uses self.data if None)
            fit: Whether to show the exponential fit curve
            plot_all: Whether to make plots for I/Q/Amps or just I
            ax: matplotlib axis to plot on, default is to create one
            show_hist: Whether to show histogram of the data
            rescale: Whether to rescale data based on histogram, from 0->1
            **kwargs: Additional arguments passed to the display function
        """
        # Get qubit index for plot title
        qubit = self.cfg.expt.qubit[0]
        title = f"$T_1$ Q{qubit}"
        xlabel = "Wait Time ($\mu$s)"

        # Define caption parameters to display T1 fit result
        caption_params = [
            {"index": 2, "format": "$T_1$ fit: {val:.3} $\pm$ {err:.2} $\mu$s"},
        ]

        # Call parent class display method
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

    def update(self, cfg_file, rng_vals=[1, 500], first_time=False, verbose=True):
        qi = self.cfg.expt.qubit[0]
        if self.status:
            config.update_qubit(
                cfg_file,
                "T1",
                self.data["new_t1_i"],
                qi,
                sig=2,
                rng_vals=rng_vals,
                verbose=verbose,
            )
            config.update_readout(
                cfg_file,
                "final_delay",
                6 * self.data["new_t1_i"],
                qi,
                sig=2,
                rng_vals=[rng_vals[0] * 10, rng_vals[1] * 3],
                verbose=verbose,
            )
            if first_time:
                config.update_qubit(
                    cfg_file,
                    "T2r",
                    self.data["new_t1_i"],
                    qi,
                    sig=2,
                    rng_vals=[rng_vals[0], rng_vals[1] * 2],
                    verbose=verbose,
                )
                config.update_qubit(
                    cfg_file,
                    "T2e",
                    2 * self.data["new_t1_i"],
                    qi,
                    sig=2,
                    rng_vals=[rng_vals[0], rng_vals[1] * 2],
                    verbose=verbose,
                )


class T1_2D(QickExperiment2DSimple):
    """
    T1_2D: Class for 2D T1 measurements to track stability over time

    This class performs repeated T1 measurements over time to create a 2D map
    of T1 values, allowing for tracking of T1 fluctuations and stability.
    The first dimension is the wait time, and the second dimension is time.

    Configuration parameters:
    - sweep_pts: Number of points in the 2D sweep (time dimension)
    - expts: Number of wait time points in each T1 measurement
    - span: Total span of wait times in microseconds
    - Other parameters similar to T1Experiment
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=None,
        style="",
        min_r2=None,
        max_err=None,
    ):
        """
        Initialize the T1_2D experiment

        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index to measure
            go: Whether to immediately run the experiment
            params: Additional parameters to override defaults
            prefix: Filename prefix for saved data
            progress: Whether to show progress bar
            style: Measurement style ('fine' for more averages, 'fast' for fewer points)
            min_r2: Minimum R² value for acceptable fit
            max_err: Maximum error for acceptable fit
        """
        # Set default prefix based on qubit index if not provided
        if prefix is None:
            prefix = f"t1_2d_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        # Define default parameters
        params_def = {
            "sweep_pts": 200,  # Number of time points (2nd dimension)
        }

        # Merge default parameters with user-provided parameters
        exp_name = T1Experiment
        self.expt = exp_name(cfg_dict, qi, go=False, params=params, check_params=False)
        params = {**params_def, **params}
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        # Run the experiment if go=True
        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        """
        Acquire 2D T1 measurement data

        This method:
        1. Sets up the 2D sweep parameters (wait time and time points)
        2. Runs the T1Program for each point in the 2D grid

        Args:
            progress: Whether to show progress bar
            debug: Whether to run in debug mode

        Returns:
            Measurement data dictionary with 2D data
        """
        # Create array for second dimension (time points)
        sweep_pts = np.arange(self.cfg.expt["sweep_pts"])
        y_sweep = [{"pts": sweep_pts, "var": "count"}]

        # Run the T1Program for each point in the 2D sweep
        super().acquire(y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze 2D T1 measurement data

        Fits each 1D slice (T1 measurement) to an exponential decay

        Args:
            data: Data dictionary to analyze (uses self.data if None)
            fit: Whether to perform fitting
            **kwargs: Additional arguments passed to the analyzer
        """
        if data is None:
            data = self.data

        # Use exponential decay function and fitter
        self.fitfunc = fitter.expfunc
        self.fitterfunc = fitter.fitexp
        super().analyze(data)

    def display(self, data=None, fit=True, ax=None, **kwargs):
        """
        Display 2D T1 measurement results

        Creates a 2D plot showing T1 values over time

        Args:
            data: Data dictionary to display (uses self.data if None)
            fit: Whether to show fit results
            ax: Matplotlib axis to plot on
            **kwargs: Additional arguments passed to the display function
        """
        if data is None:
            data = self.data

        # Set plot labels
        title = f"$T_1$ 2D Q{self.cfg.expt.qubit}"
        xlabel = f"Wait Time ($\mu$s)"
        ylabel = "Time (s)"

        # Call parent class display method
        super().display(
            data=data,
            ax=ax,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            fit=fit,
            **kwargs,
        )
