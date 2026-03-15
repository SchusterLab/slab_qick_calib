import numpy as np
from pathlib import Path
from datetime import datetime
from qick import *
from qick.asm_v2 import QickSweep1D

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
            self.delay_auto(t=0.01, tag="wait_stark")
            # Apply flux pulse during wait time
            self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse", t=0)
            # Small buffer delay after flux pulse
            self.delay_auto(t=0.1, tag="wait")
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
            "flux_gain": 0.1, # Gain for flux pulse
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

        # Ensure sweep step size is above DAC minimum resolution
        self.check_dac_timing(num_segments=1)

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
        if self.cfg.expt.get("flux", False):
            title += f" (Flux Gain {self.cfg.expt.flux_gain:0.3f})"

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


class T1FastFlux(T1Experiment):
    """
    T1FastFlux: Sweep T1 across flux gain values.

    Runs a T1Experiment at each flux gain point, collecting T1 vs flux gain (and
    optionally vs frequency). Saves CSV incrementally after each point.

    Automatically reads quadratic fit coefficients (quad_a, quad_b, quad_c) and
    sweet_spot_ac from the config (saved by QubitSpecFastFlux) when available.
    The gain sweep starts at sweet_spot_ac by default.

    Sweep parameters (passed via params dict):
        gain_start: Starting flux gain value (default: sweet_spot_ac)
        gain_stop: Ending flux gain value (default: sweet_spot_ac + 0.4)
        direction: 'pos' or 'neg' — sweep direction from sweet spot (default: 'pos').
            Sets gain_stop relative to sweet spot if gain_stop not explicitly given.
        freq_span: Frequency span in MHz. When provided with freq_coeffs, converts
            to a gain range using the quadratic. Overrides gain_stop.
        expts_gain: Number of gain points in the sweep
        freq_coeffs: Optional (a, b, c) tuple for quadratic freq = a*g^2 + b*g + c.
            If not provided, reads quad_a/b/c from config.
        lin_freq: If True and freq_coeffs provided, space points linearly in
            frequency (inverting the quadratic) instead of linearly in gain.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=False,
        display=True,
    ):
        if prefix is None:
            prefix = f"t1_fastflux_qubit{qi}"

        # Initialize parent T1Experiment without running
        super().__init__(
            cfg_dict=cfg_dict, qi=qi, go=False, params=params,
            prefix=prefix, progress=progress, check_params=False,
        )

        # Read quadratic coefficients and sweet spot from config if available
        cfg_flux = getattr(getattr(getattr(getattr(self.cfg, "hw", None), "soc", None), "dacs", None), "flux", None)
        cfg_qubit = getattr(getattr(self.cfg, "device", None), "qubit", None)

        config_freq_coeffs = None
        sweet_spot = 0.0
        if cfg_flux is not None:
            a = cfg_flux.quad_a[qi] if hasattr(cfg_flux, "quad_a") else 0
            b = cfg_flux.quad_b[qi] if hasattr(cfg_flux, "quad_b") else 0
            c = cfg_flux.quad_c[qi] if hasattr(cfg_flux, "quad_c") else 0
            if a != 0 or b != 0:
                config_freq_coeffs = (a, b, c)
        if cfg_qubit is not None and hasattr(cfg_qubit, "sweet_spot_ac"):
            sweet_spot = cfg_qubit.sweet_spot_ac[qi]

        # Determine sweep direction and gain range
        direction = params.get("direction", "pos")
        sign = 1 if direction == "pos" else -1
        freq_coeffs = params.get("freq_coeffs", config_freq_coeffs)
        freq_span = params.get("freq_span", None)

        # Convert freq_span to gain_stop using the quadratic
        if freq_span is not None and freq_coeffs is not None and "gain_stop" not in params:
            a, b, c = freq_coeffs
            f_sweet = a * sweet_spot**2 + b * sweet_spot + c
            f_target = f_sweet - freq_span  # frequency decreases away from sweet spot
            discriminant = b**2 - 4 * a * (c - f_target)
            if discriminant >= 0:
                # Pick the root in the correct direction from sweet spot
                root_pos = (-b + np.sqrt(discriminant)) / (2 * a)
                root_neg = (-b - np.sqrt(discriminant)) / (2 * a)
                if sign > 0:
                    gain_stop = max(root_pos, root_neg)
                else:
                    gain_stop = min(root_pos, root_neg)
            else:
                gain_stop = sweet_spot + sign * 0.4
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
            "freq_coeffs": freq_coeffs,
            "freq_span": freq_span,
            "direction": direction,
            "lin_freq": True,
        }

        if start_t1 is not None:
            params_def["span"] = 4.1 * start_t1
        params = {**params_def, **params}
        # Create inner T1 experiment (go=False) to inherit its config
        self.expt = T1Experiment(cfg_dict, qi, go=False, params=params, check_params=False)
        params = {**self.expt.cfg.expt, **params, "flux": True}
        self.cfg.expt = {**params_def, **params}

        # Store references
        self._qi = qi
        self._cfg_dict = cfg_dict

        # Build gain_pts and freq_pts from config
        cfg_e = self.cfg.expt
        freq_coeffs = cfg_e["freq_coeffs"]

        if cfg_e["lin_freq"] and freq_coeffs is not None:
            # Linearly spaced in frequency, invert quadratic to get gains
            a, b, c = freq_coeffs
            freq_start = a * cfg_e["gain_start"]**2 + b * cfg_e["gain_start"] + c
            freq_stop = a * cfg_e["gain_stop"]**2 + b * cfg_e["gain_stop"] + c
            self.freq_pts = np.linspace(freq_start, freq_stop, cfg_e["expts_gain"])
            discriminant = b**2 - 4 * a * (c - self.freq_pts)
            root_pos = (-b + np.sqrt(discriminant)) / (2 * a)
            root_neg = (-b - np.sqrt(discriminant)) / (2 * a)
            if cfg_e["direction"] == "pos":
                self.gain_pts = np.maximum(root_pos, root_neg)
            else:
                self.gain_pts = np.minimum(root_pos, root_neg)
        else:
            # Linearly spaced in gain
            self.gain_pts = np.linspace(cfg_e["gain_start"], cfg_e["gain_stop"], cfg_e["expts_gain"])
            if freq_coeffs is not None:
                a, b, c = freq_coeffs
                self.freq_pts = a * self.gain_pts**2 + b * self.gain_pts + c
            else:
                self.freq_pts = None

        # Set up folder structure: t1_fast_flux/{timestamp}_T1_fastflux_Q{qi}/
        base_dir = Path(cfg_dict["expt_path"]) / "t1_fast_flux"
        base_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = base_dir / f"{timestamp}_T1_fastflux_Q{qi}"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if go:
            self.acquire(progress=progress)
            if display:
                self.display()

    def acquire(self, progress=False):
        original_path = self._cfg_dict["expt_path"]
        self._cfg_dict["expt_path"] = str(self.save_dir)
        self.t1_data = []
        t1_list = []
        offset_list = []
        amp_list = []
        csv_path = self.save_dir / "t1_vs_gain.csv"

        # Build inner T1 params from cfg.expt, excluding our 2D-specific keys
        _2d_keys = ("gain_start", "gain_stop", "expts_gain", "freq_coeffs", "lin_freq", "freq_span", "direction")
        t1_params = {k: v for k, v in self.cfg.expt.items() if k not in _2d_keys}

        try:
            for i, g in enumerate(self.gain_pts):
                t1 = T1Experiment(
                    self._cfg_dict,
                    qi=self._qi,
                    params={**t1_params, "flux_gain": float(g)},
                    progress=progress,
                    display=False,
                )
                self.t1_data.append(t1.data)
                best_fit = t1.data.get("best_fit", [np.nan, np.nan, np.nan])
                t1_val = best_fit[2]
                t1_list.append(t1_val)
                offset_list.append(best_fit[0])
                amp_list.append(best_fit[1])

                # Adjust span for next scan based on extracted T1
                if np.isfinite(t1_val) and t1_val > 0:
                    t1_params["span"] = 4.1 * t1_val

                # Save CSV incrementally
                gains_so_far = self.gain_pts[: i + 1]
                t1s_so_far = np.array(t1_list)
                offsets_so_far = np.array(offset_list)
                amps_so_far = np.array(amp_list)
                if self.freq_pts is not None:
                    freqs_so_far = self.freq_pts[: i + 1]
                    table = np.column_stack((gains_so_far, freqs_so_far, t1s_so_far, offsets_so_far, amps_so_far))
                    header = "gain,freq,t1,offset,amplitude"
                    fmt = ["%.6f", "%.10f", "%.6f", "%.6f", "%.6f"]
                else:
                    table = np.column_stack((gains_so_far, t1s_so_far, offsets_so_far, amps_so_far))
                    header = "gain,t1,offset,amplitude"
                    fmt = ["%.6f", "%.6f", "%.6f", "%.6f"]
                np.savetxt(csv_path, table, delimiter=",", header=header, comments="", fmt=fmt)
        finally:
            self._cfg_dict["expt_path"] = original_path

        self.data = {
            "gain_pts": self.gain_pts,
            "t1_list": np.array(t1_list),
            "offset_list": np.array(offset_list),
            "amp_list": np.array(amp_list),
            "t1_data": self.t1_data,
        }
        if self.freq_pts is not None:
            self.data["freq_pts"] = self.freq_pts

        return self.data

    def display(self, data=None, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        gain_pts = data["gain_pts"]
        t1_list = data["t1_list"]

        plt.figure()
        plt.plot(gain_pts, t1_list, ".-")
        plt.xlabel("Flux Gain")
        plt.ylabel("$T_1$ ($\mu$s)")
        plt.title(f"$T_1$ vs Flux Gain Q{self._qi}")
        plt.savefig(self.save_dir / "t1_vs_flux_gain.png")
        plt.show()

        if "freq_pts" in data:
            freq_pts = data["freq_pts"]

            plt.figure()
            plt.plot(freq_pts, t1_list, ".-")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("$T_1$ ($\mu$s)")
            plt.title(f"$T_1$ vs Frequency Q{self._qi}")
            plt.savefig(self.save_dir / "t1_vs_freq.png")
            plt.show()

            plt.figure()
            plt.plot(freq_pts, data["amp_list"], ".-")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Amplitude")
            plt.title(f"Amplitude vs Frequency Q{self._qi}")
            plt.savefig(self.save_dir / "amp_vs_freq.png")
            plt.show()

            plt.figure()
            plt.plot(freq_pts, data["offset_list"], ".-")
            plt.xlabel("Frequency (MHz)")
            plt.ylabel("Offset")
            plt.title(f"Offset vs Frequency Q{self._qi}")
            plt.savefig(self.save_dir / "offset_vs_freq.png")
            plt.show()
