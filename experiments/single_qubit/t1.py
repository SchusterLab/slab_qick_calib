import numpy as np
import time
from datetime import datetime
from qick import *
from qick.asm_v2 import QickSweep1D
from tqdm import tqdm

from ...helpers import config
from ...analysis import fitting as fitter
from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment, QickExperiment2DSimple
from ..general.qick_program import QickProgram

FIT_FUNC = fitter.expfunc
FITTER_FUNC = fitter.fitexp
XLABEL = "Wait Time ($\mu$s)"
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
        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        # If AC Stark shift is enabled, create a constant pulse to apply during wait time
        if cfg.expt.acStark:
            pulse = {
                "length": cfg.expt.wait_time,  # Duration of the pulse
                "ramp_sigma": 0.035,  # No increment in duration
                "ramp_sigma_inc":4,
                "freq": cfg.expt.stark_freq,  # Frequency of the AC Stark pulse
                "gain": cfg.expt.stark_gain,  # Amplitude of the AC Stark pulse
                "phase": 0,  # Phase of the pulse
                "type": "flat_top",  
            }
            super().make_pulse(pulse, "stark_pulse")
        elif cfg.expt.flux:
            self.declare_gen(cfg.expt.flux_chan, nqz=1, mixer_freq=0)
            pulse = {
            "chan": cfg.expt.flux_chan,
            "freq": 0,  # Pulse frequency
            "phase": 0,  # Pulse phase
            "gain": cfg.expt.flux_gain,  # Pulse amplitude
            "length": cfg.expt.wait_time,  # Pulse duration
            "type": 'const'            
            }
            super().make_pulse(pulse, "flux_pulse")

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
        elif cfg.expt.flux:
            # Apply flux pulse during wait time
            self.delay_auto(t=0.01, tag="wait_flux")
            # Apply flux pulse during wait time
            self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse", t=0)
            # Small buffer delay after flux pulse
            self.delay_auto(t=cfg.expt.flux_readout_wait, tag="wait")
            #self.delay_auto(t=cfg.expt["end_wait"], tag="wait")
        else:
            # Simple delay for standard T1 measurement
            self.delay_auto(t=cfg.expt["wait_time"] + 0.03, tag="wait")

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
            "reps": int(1.5*self.reps),  # Number of repetitions (inner loop)
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
            "flux": False, # Whether to apply flux pulse during wait time
            "flux_chan": self.cfg.hw.soc.dacs.flux.ch[qi], # DAC channel for flux pulse
            "flux_gain": 0.1, # Gain for flux pulse,
            "flux_readout_wait": 0.1 # Wait time after flux pulse before readout
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

    def acquire(self, progress=False, debug=False, **kwargs):
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

        # Ensure sweep step size is above DAC minimum resolution
        self.check_dac_timing(num_segments=1)

        # Create a 1D sweep for the wait time from start to start+span
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        # Run the T1Program to acquire data
        super().acquire(T1Program, progress=progress, **kwargs)

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
        self.fitfunc = FIT_FUNC  # Exponential decay function
        self.fitterfunc = FITTER_FUNC  # Fitting function for exponential decay
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
        if self.cfg.expt.flux:
            flux_gain = self.cfg.expt.flux_gain
            title += f" (Flux Gain {flux_gain:0.3f}"
            converter = fitter.flux_converter_from_config(
                self.cfg.hw.soc.dacs.get("flux", None),
                self.cfg.device.qubit,
                qubit,
            )
            if converter is not None:
                freq = converter.gain_to_freq(flux_gain)
                title += f", Freq {freq:.3f} MHz"
            title += ")"

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
            xlabel=XLABEL,
            fit=fit,
            show_hist=show_hist,
            fitfunc=self.fitfunc,
            caption_params=caption_params,
            rescale=rescale,
            **kwargs,
        )

    def update(self, rng_vals=[0.5, 500], first_time=False, no_final=False, verbose=True):
        qi = self.cfg.expt.qubit[0]
        if self.status:
            config.update_qubit(
                self.config_file,
                "T1",
                self.data["new_t1_i"],
                qi,
                sig=2,
                rng_vals=rng_vals,
                verbose=verbose,
            )
            if not no_final:
                config.update_readout(
                    self.config_file,
                    "final_delay",
                    6 * self.data["new_t1_i"],
                    qi,
                    sig=2,
                    rng_vals=[rng_vals[0] * 10, rng_vals[1] * 3],
                    verbose=verbose,
                )
            if first_time:
                config.update_qubit(
                    self.config_file,
                    "T2r",
                    self.data["new_t1_i"],
                    qi,
                    sig=2,
                    rng_vals=[rng_vals[0], rng_vals[1] * 2],
                    verbose=verbose,
                )
                config.update_qubit(
                    self.config_file,
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
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = {**params_def, **params}

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


class T1FastFlux(QickExperiment2DSimple):
    """
    T1FastFlux: Sweep T1 across flux gain values.

    Runs a T1Experiment at each flux gain point, collecting T1 vs flux gain (and
    optionally vs frequency). Saves a single HDF5 file with save_interim after
    each gain point. Performs adaptive span adjustment between scans.

    Automatically reads flux model parameters from config (saved by QubitSpecFastFlux).
    Supports both transmon SQUID model (transmon_*) and quadratic (quad_a/b/c) parameters,
    preferring the transmon model when available. The gain sweep starts at sweet_spot_ac
    by default.

    Sweep parameters (passed via params dict):
        gain_start: Starting flux gain value (default: sweet_spot_ac)
        gain_stop: Ending flux gain value (default: sweet_spot_ac + 0.4)
        direction: 'pos' or 'neg' — sweep direction from sweet spot (default: 'pos').
            Sets gain_stop relative to sweet spot if gain_stop not explicitly given.
        freq_span: Frequency span in MHz. When provided with a flux model, converts
            to a gain range. Overrides gain_stop.
        expts_gain: Number of gain points in the sweep
        flux_converter: Optional FluxConverter instance. If not provided, loads from config.
        lin_freq: If True and flux model available, space points linearly in
            frequency instead of linearly in gain.
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
    ):
        if prefix is None:
            prefix = f"t1_fastflux_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, qi=qi)

        # Load flux model from config (prefers transmon, falls back to quadratic)
        cfg_flux = getattr(getattr(getattr(getattr(self.cfg, "hw", None), "soc", None), "dacs", None), "flux", None)
        cfg_qubit = getattr(getattr(self.cfg, "device", None), "qubit", None)

        sweet_spot = 0.0
        if cfg_qubit is not None and hasattr(cfg_qubit, "sweet_spot_ac"):
            sweet_spot = cfg_qubit.sweet_spot_ac[qi]

        # Determine sweep direction and gain range
        direction = params.get("direction", "pos")
        sign = 1 if direction == "pos" else -1

        converter = params.get("flux_converter", None)
        if converter is None:
            converter = fitter.flux_converter_from_config(cfg_flux, cfg_qubit, qi, direction)
        freq_span = params.get("freq_span", None)

        # Convert freq_span to gain_stop using the flux model
        if freq_span is not None and converter is not None and "gain_stop" not in params:
            f_sweet = converter.gain_to_freq(sweet_spot)
            f_target = f_sweet - freq_span  # frequency decreases away from sweet spot
            gain_stop = float(converter.freq_to_gain(f_target))
        elif "gain_stop" not in params:
            gain_stop = sweet_spot + sign * 0.4
        else:
            gain_stop = params["gain_stop"]

        # start_t1: assumed T1 for the first gain point, sets initial span
        start_t1 = params.pop("start_t1", None)

        # Define default parameters for the 2D sweep
        params_def = {
            "gain_start": sweet_spot,
            "gain_stop": gain_stop,
            "start": 0.05,
            "expts":25,
            "expts_gain": 50,
            "freq_span": freq_span,
            "direction": direction,
            "lin_freq": True,
            "t1_max": float("inf"),  # Upper bound on T1 when setting next span
        }

        if start_t1 is not None:
            params_def["span"] = 4.1 * start_t1
        params = {**params_def, **params}
        # Create inner T1 experiment (go=False) to inherit its config
        self.expt = T1Experiment(cfg_dict, qi, go=False, params=params, check_params=False)
        params = {**self.expt.cfg.expt, **params, "flux": True}
        self.cfg.expt = {**params_def, **params}
        self.expt.cfg.expt = {**self.expt.cfg.expt, **params}

        # Store references
        self._qi = qi
        self._cfg_dict = cfg_dict
        self._flux_converter = converter

        # Build gain_pts and freq_pts from config
        cfg_e = self.cfg.expt

        if cfg_e["lin_freq"] and converter is not None:
            # Linearly spaced in frequency, invert flux model to get gains
            freq_start = converter.gain_to_freq(cfg_e["gain_start"])
            freq_stop = converter.gain_to_freq(cfg_e["gain_stop"])
            self.freq_pts = np.linspace(float(freq_start), float(freq_stop), cfg_e["expts_gain"])
            self.gain_pts = converter.freq_to_gain(self.freq_pts)
        else:
            # Linearly spaced in gain
            self.gain_pts = np.linspace(cfg_e["gain_start"], cfg_e["gain_stop"], cfg_e["expts_gain"])
            if converter is not None:
                self.freq_pts = converter.gain_to_freq(self.gain_pts)
            else:
                self.freq_pts = None

        if go:
            self.acquire(progress=progress, display=display)
            self.display()

    def acquire(self, progress=True, display=True):
        from pathlib import Path

        data = {"time": []}
        t1_list = []
        offset_list = []
        amp_list = []
        q_offset_list = []
        q_amp_list = []
        s_offset_list = []
        s_amp_list = []
        timestamp_list = []
        elapsed_list = []
        t_acq_start = time.time()
        csv_path = Path(self.fname).with_suffix(".csv")

        n_total = len(self.gain_pts)
        for i, g in enumerate(tqdm(self.gain_pts, disable=not progress)):
            # Update inner experiment config for this gain point
            self.expt.cfg.expt["flux_gain"] = float(g)

            # Run inner T1 scan and analyze
            data_new = self.expt.acquire(progress=False)
            self.expt.analyze(data=data_new, verbose=False, rescale=True)
            if display:
                self.expt.display(data=data_new)

            # Accumulate raw 2D data
            for key in ("avgi", "avgq", "amps", "phases", "xpts", "scale_data"):
                if key in data_new:
                    if i == 0:
                        data[key] = []
                    data[key].append(data_new[key])
            now = time.time()
            data["time"].append(now)
            timestamp_list.append(datetime.fromtimestamp(now).isoformat())
            elapsed_list.append(now - t_acq_start)

            # Extract fit results
            best_fit = data_new.get("best_fit", [np.nan, np.nan, np.nan])
            t1_val = best_fit[2]
            t1_list.append(t1_val)
            offset_list.append(best_fit[0])
            amp_list.append(best_fit[1])
            fit_q = data_new.get("fit_avgq", [np.nan, np.nan, np.nan])
            q_offset_list.append(fit_q[0])
            q_amp_list.append(fit_q[1])
            fit_s = data_new.get("fit_scale_data", [np.nan, np.nan, np.nan])
            if fit_s is not None:
                s_offset_list.append(fit_s[0])
                s_amp_list.append(fit_s[1])
            else:
                s_offset_list.append(np.nan)
                s_amp_list.append(np.nan)

            # Adjust span for next scan based on extracted T1
            if np.isfinite(t1_val) and t1_val > 0:
                self.expt.cfg.expt["span"] = 3.7 * min(
                    t1_val, self.cfg.expt.t1_max
                )

            # Store analysis summaries in data for interim save
            data["gain_pts"] = self.gain_pts
            if self.freq_pts is not None:
                data["freq_pts"] = self.freq_pts
            data["t1_list"] = np.array(t1_list)
            data["offset_list"] = np.array(offset_list)
            data["amp_list"] = np.array(amp_list)
            data["q_offset_list"] = np.array(q_offset_list)
            data["q_amp_list"] = np.array(q_amp_list)
            data["s_offset_list"] = np.array(s_offset_list)
            data["s_amp_list"] = np.array(s_amp_list)

            if self.save_interim:
                self.save_data(data=data)

            # Save CSV incrementally (with timestamp and elapsed time)
            gains_so_far = self.gain_pts[: i + 1]
            if self.freq_pts is not None:
                num_cols = np.column_stack((gains_so_far, self.freq_pts[: i + 1],
                    data["t1_list"], data["offset_list"], data["amp_list"],
                    data["q_offset_list"], data["q_amp_list"],
                    np.array(elapsed_list)))
                header = "timestamp,gain,freq,t1,offset,amplitude,q_offset,q_amplitude,elapsed_s"
            else:
                num_cols = np.column_stack((gains_so_far,
                    data["t1_list"], data["offset_list"], data["amp_list"],
                    data["q_offset_list"], data["q_amp_list"],
                    np.array(elapsed_list)))
                header = "timestamp,gain,t1,offset,amplitude,q_offset,q_amplitude,elapsed_s"
            with open(csv_path, "w") as f_csv:
                f_csv.write(header + "\n")
                for row_idx in range(num_cols.shape[0]):
                    ts = timestamp_list[row_idx]
                    nums = ",".join(f"{v}" for v in num_cols[row_idx])
                    f_csv.write(f"{ts},{nums}\n")

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
        t1_list = data["t1_list"]

        fig, ax = plt.subplots()
        t1_arr = np.array(t1_list, dtype=float)
        t1_mask = np.isfinite(t1_arr)
        if t1_mask.sum() > 2:
            q1, q3 = np.percentile(t1_arr[t1_mask], [25, 75])
            iqr = q3 - q1
            t1_inlier = t1_mask & (t1_arr >= q1 - 1.5 * iqr) & (t1_arr <= q3 + 1.5 * iqr)
            ax.plot(gain_pts[t1_inlier], t1_arr[t1_inlier], ".-")
            ax.plot(gain_pts[~t1_inlier & t1_mask], t1_arr[~t1_inlier & t1_mask], "rx", ms=6)
            inlier_vals = t1_arr[t1_inlier]
            if len(inlier_vals) > 0:
                ymin, ymax = inlier_vals.min(), inlier_vals.max()
                margin = (ymax - ymin) * 0.1 if ymax > ymin else abs(ymax) * 0.1 or 0.1
                ax.set_ylim(ymin - margin, ymax + margin)
        else:
            ax.plot(gain_pts, t1_list, ".-")
        ax.set_xlabel("Flux Gain")
        ax.set_ylabel("$T_1$ ($\mu$s)")
        self.save_fig(fig, suffix="_t1_vs_gain")

        if "freq_pts" in data:
            freq_pts = data["freq_pts"]

            fig, ax = plt.subplots()
            t1_arr = np.array(t1_list, dtype=float)
            t1_mask = np.isfinite(t1_arr)
            if t1_mask.sum() > 2:
                q1, q3 = np.percentile(t1_arr[t1_mask], [25, 75])
                iqr = q3 - q1
                t1_inlier = t1_mask & (t1_arr >= q1 - 1.5 * iqr) & (t1_arr <= q3 + 1.5 * iqr)
                ax.plot(freq_pts[t1_inlier], t1_arr[t1_inlier], ".-")
                ax.plot(freq_pts[~t1_inlier & t1_mask], t1_arr[~t1_inlier & t1_mask], "rx", ms=6)
                inlier_vals = t1_arr[t1_inlier]
                if len(inlier_vals) > 0:
                    ymin, ymax = inlier_vals.min(), inlier_vals.max()
                    margin = (ymax - ymin) * 0.1 if ymax > ymin else abs(ymax) * 0.1 or 0.1
                    ax.set_ylim(ymin - margin, ymax + margin)
            else:
                ax.plot(freq_pts, t1_list, ".-")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("$T_1$ ($\mu$s)")
            self.save_fig(fig, suffix="_t1_vs_freq")

            fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)

            panels = [
                (axes[0, 0], data["amp_list"], "I Amplitude"),
                (axes[0, 1], data["offset_list"], "I Offset"),
                (axes[1, 0], data["q_amp_list"], "Q Amplitude"),
                (axes[1, 1], data["q_offset_list"], "Q Offset"),
                (axes[2, 0], data.get("s_amp_list", []), "Rescaled Amplitude"),
                (axes[2, 1], data.get("s_offset_list", []), "Rescaled Offset"),
            ]
            for ax, vals, label in panels:
                vals = np.array(vals, dtype=float)
                if vals.size == 0:
                    ax.set_ylabel(label)
                    continue
                mask = np.isfinite(vals)
                if mask.sum() > 2:
                    q1, q3 = np.percentile(vals[mask], [25, 75])
                    iqr = q3 - q1
                    inlier = mask & (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
                    ax.plot(freq_pts[inlier], vals[inlier], ".-")
                    ax.plot(freq_pts[~inlier & mask], vals[~inlier & mask], "rx", ms=6)
                    inlier_vals = vals[inlier]
                    if len(inlier_vals) > 0:
                        ymin, ymax = inlier_vals.min(), inlier_vals.max()
                        margin = (ymax - ymin) * 0.1 if ymax > ymin else abs(ymax) * 0.1 or 0.1
                        ax.set_ylim(ymin - margin, ymax + margin)
                else:
                    ax.plot(freq_pts, vals, ".-")
                ax.set_ylabel(label)
            axes[2, 0].set_xlabel("Frequency (MHz)")
            axes[2, 1].set_xlabel("Frequency (MHz)")
            self.save_fig(fig, suffix="_fit_params")


class T1FastFluxRepeated(T1FastFlux):
    """
    Repeatedly run T1FastFlux sweeps and build a T1(frequency, time) heatmap.

    Runs T1FastFlux N times (or for a specified duration), accumulating
    T1 vs frequency data. After each sweep, updates a live 2D heatmap
    (x=frequency, y=elapsed time, color=T1) and appends to a master CSV.

    Extra parameters (passed via params dict, in addition to T1FastFlux params):
        n_repeats: Number of sweeps to run (default: None = run forever)
        duration_hours: Max duration in hours (default: None = no limit).
            If both n_repeats and duration_hours are given, stops at whichever
            comes first. If both None, runs until KeyboardInterrupt.
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
    ):
        # Pop repeated-specific params before passing to parent
        self._n_repeats = params.pop("n_repeats", None)
        self._duration_hours = params.pop("duration_hours", None)

        if prefix is None:
            prefix = f"t1_fastflux_repeated_qubit{qi}"

        # Init parent (sets up gain_pts, freq_pts) but don't run
        super().__init__(
            cfg_dict=cfg_dict, qi=qi, go=False, params=params,
            prefix=prefix, progress=progress, display=display,
        )

        if go:
            self.acquire(progress=progress, display=display)
            self.display()

    def acquire(self, progress=False, display=True):
        from pathlib import Path

        try:
            from IPython.display import display as ipy_display, clear_output
            _has_ipy = True
        except ImportError:
            _has_ipy = False

        self.t1_list = []
        self.timestamps = []
        t_start = time.time()
        sweep_idx = 0

        # Create a dedicated folder for this run's CSVs
        csv_dir = Path(self.fname).with_suffix("") / "csvs"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Build inner T1FastFlux params from cfg.expt, excluding repeated-specific keys
        _repeated_keys = ("n_repeats", "duration_hours")
        sweep_params = {k: v for k, v in self.cfg.expt.items() if k not in _repeated_keys}

        try:
            while True:
                # Check stopping conditions
                if self._n_repeats is not None and sweep_idx >= self._n_repeats:
                    break
                elapsed_h = (time.time() - t_start) / 3600
                if self._duration_hours is not None and elapsed_h > self._duration_hours:
                    break

                # Run one T1FastFlux sweep (go=False, no save)
                # Disable tqdm in inner sweep when live plotting — text-based
                # tqdm stderr output interferes with clear_output in Jupyter
                inner_progress = progress and not _has_ipy
                sweep = T1FastFlux(
                    self._cfg_dict, qi=self._qi, go=False, params={**sweep_params},
                    progress=inner_progress, display=False,
                )
                sweep.gain_pts = self.gain_pts
                sweep.freq_pts = self.freq_pts
                sweep.save_interim = False  # outer loop handles saving
                # Redirect CSV into the dedicated folder
                ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
                sweep.fname = str(csv_dir / f"sweep_{sweep_idx:04d}_{ts_label}")
                if _has_ipy:
                    print(f"Sweep {sweep_idx + 1}" +
                          (f"/{self._n_repeats}" if self._n_repeats else "") +
                          f" — {len(self.gain_pts)} gain points ...")
                data = sweep.acquire(progress=inner_progress, display=False)

                # Accumulate results
                elapsed_h = (time.time() - t_start) / 3600
                self.t1_list.append(data["t1_list"].copy())
                self.timestamps.append(elapsed_h)

                # Carry adaptive span to next sweep
                last_t1 = data["t1_list"]
                finite_t1s = last_t1[np.isfinite(last_t1) & (last_t1 > 0)]
                if len(finite_t1s) > 0:
                    median_t1 = np.median(finite_t1s)
                    t1_max = self.cfg.expt.t1_max
                    sweep_params["span"] = 4.1 * min(median_t1, t1_max)

                # Build and save interim data
                self.data = {
                    "freq_pts": self.freq_pts,
                    "gain_pts": self.gain_pts,
                    "t1_list": np.array(self.t1_list),
                    "timestamps": np.array(self.timestamps),
                }
                self.save_data()

                # Update live heatmap (always show in Jupyter, regardless of display)
                if _has_ipy:
                    self._update_heatmap(clear_output, ipy_display)

                sweep_idx += 1

        except KeyboardInterrupt:
            print(f"\nInterrupted after {sweep_idx} sweep(s).")

        self.data = {
            "freq_pts": self.freq_pts,
            "gain_pts": self.gain_pts,
            "t1_list": np.array(self.t1_list) if self.t1_list else np.array([]).reshape(0, 0),
            "timestamps": np.array(self.timestamps),
        }
        return self.data

    def _update_heatmap(self, clear_output, ipy_display):
        import sys
        import io
        import matplotlib.pyplot as plt
        from IPython.display import Image

        try:
            t1_2d = np.array(self.t1_list)
            freq = self.freq_pts if self.freq_pts is not None else self.gain_pts
            times = np.array(self.timestamps)

            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            # Clamp color scale to inlier range
            finite_vals = t1_2d[np.isfinite(t1_2d)]
            vmin, vmax = None, None
            if len(finite_vals) > 2:
                q1, q3 = np.percentile(finite_vals, [25, 75])
                iqr = q3 - q1
                inlier_vals = finite_vals[(finite_vals >= q1 - 1.5 * iqr) & (finite_vals <= q3 + 1.5 * iqr)]
                if len(inlier_vals) > 0:
                    vmin, vmax = inlier_vals.min(), inlier_vals.max()
            mesh = ax.pcolormesh(freq, times, t1_2d, cmap="viridis", shading="auto",
                                 vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_xlabel("Frequency (MHz)" if self.freq_pts is not None else "Flux Gain")
            ax.set_ylabel("Elapsed Time (hours)")
            ax.set_title(f"$T_1$ vs Frequency and Time — Q{self._qi}")
            plt.colorbar(mesh, ax=ax, label="$T_1$ ($\\mu$s)")
            fig.tight_layout()

            # Render to PNG and close figure immediately so inline backend
            # doesn't auto-display it as SVG
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            # Flush stdio so tqdm output is fully written before clearing
            sys.stdout.flush()
            sys.stderr.flush()
            clear_output(wait=True)
            ipy_display(Image(buf.read()))
            print(f"Completed {len(self.t1_list)} sweep(s), {times[-1]:.2f} hours elapsed")
        except Exception as e:
            print(f"[LivePlot] Heatmap update failed: {e}")

    def display(self, data=None, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        t1_2d = data["t1_list"]
        if t1_2d.size == 0:
            print("No sweep data to display.")
            return

        freq = data.get("freq_pts", data["gain_pts"])
        times = data["timestamps"]

        fig, ax = plt.subplots(figsize=(10, 6))
        # Clamp color scale to inlier range
        finite_vals = t1_2d[np.isfinite(t1_2d)]
        vmin, vmax = None, None
        if len(finite_vals) > 2:
            q1, q3 = np.percentile(finite_vals, [25, 75])
            iqr = q3 - q1
            inlier_vals = finite_vals[(finite_vals >= q1 - 1.5 * iqr) & (finite_vals <= q3 + 1.5 * iqr)]
            if len(inlier_vals) > 0:
                vmin, vmax = inlier_vals.min(), inlier_vals.max()
        mesh = ax.pcolormesh(freq, times, t1_2d, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Frequency (MHz)" if "freq_pts" in data and data["freq_pts"] is not None else "Flux Gain")
        ax.set_ylabel("Elapsed Time (hours)")
        ax.set_title(f"$T_1$ vs Frequency and Time — Q{self._qi}")
        plt.colorbar(mesh, ax=ax, label="$T_1$ ($\\mu$s)")
        self.save_fig(fig, suffix="_heatmap")
