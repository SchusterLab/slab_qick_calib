"""
T2 Measurement Module

This module implements T2 (dephasing time) measurements for superconducting qubits.
T2 is a measure of how long a qubit maintains phase coherence in the x-y plane of the Bloch sphere.

The module supports three main measurement protocols:
1. Ramsey: Uses two π/2 pulses separated by a variable delay time
2. Echo: Uses two π/2 pulses with one or more π pulses in between to refocus dephasing
3. CPMG: Uses a sequence of π pulses to dynamically decouple the qubit from the environment

Additional features include:
- AC Stark shift measurements during Ramsey experiments
- EF transition measurements (first excited to second excited state)
- Automatic frequency error detection and correction
"""

import numpy as np
import time
from qick import *
from qick.asm_v2 import QickSweep1D
from tqdm import tqdm

from ...analysis import fitting as fitter
from ..general.qick_experiment import QickExperiment, QickExperiment2DSimple
from ..general.qick_flux_experiment import QickFluxSweep
from ..general.qick_program import QickProgram, mixer_freq_for, reset_mode_of, reset_plays_pi_pulses, pulse_type_from_config
from ...helpers import config

from ...exp_handling.datamanagement import AttrDict


class T2Program(QickProgram):
    """
    Quantum program for T2 measurements (Ramsey, Echo, and CPMG protocols).

    This class defines the pulse sequences for T2 measurements:
    - Ramsey: π/2 - wait - π/2 sequence to measure phase coherence
    - Echo: π/2 - wait - π - wait - π/2 sequence to refocus dephasing
    - CPMG: π/2 - (wait - π)^n - wait - π/2 sequence for dynamical decoupling

    Additional options include AC Stark shift during wait time and EF transition measurements.
    """

    def __init__(self, soccfg, final_delay, cfg, final_wait=0):
        """
        Initialize the T2 program.

        Args:
            soccfg: SOC configuration
            final_delay: Delay after measurement before next experiment
            cfg: Configuration dictionary containing experiment parameters
        """
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg, final_wait=0)

    def _initialize(self, cfg):
        """
        Initialize the program by setting up the pulse sequence.

        Creates the necessary pulses for T2 measurements:
        - Two π/2 pulses (prep and read)
        - Optional π pulse(s) for Echo and CPMG
        - Optional AC Stark pulse for Ramsey with AC Stark shift

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)

        # Create loop for sweeping wait time
        self.add_loop("wait_loop", cfg.expt.expts)

        # Initialize standard readout
        super()._initialize(cfg, readout="standard")

        # Create π/2 pulses for Ramsey/Echo sequence
        # First π/2 pulse has phase=0, second has phase based on wait time and Ramsey frequency
        pulse = {
            "sigma": cfg.expt.sigma,  # Half sigma for π/2 pulse
            "sigma_inc": cfg.expt.sigma_inc,
            "freq": cfg.expt.freq,
            "gain": cfg.expt.gain / 2,
            "phase": 0,  # First pulse has zero phase
            "type": cfg.expt.type,
        }

        # Create first π/2 pulse (preparation)
        super().make_pulse(pulse, "pi2_prep")

        # Create second π/2 pulse (readout) with phase that depends on wait time
        # Phase advances at rate of ramsey_freq (MHz) * wait_time (μs) * 360 (deg/cycle)
        pulse["phase"] = cfg.expt.wait_time * 360 * cfg.expt.ramsey_freq
        super().make_pulse(pulse, "pi2_read")

        

        # For AC Stark shift in Ramsey experiments
        if hasattr(cfg.expt, "acStark") and cfg.expt.acStark:
            # Create pulse to apply during wait time
            pulse = {
                "sigma": cfg.expt.wait_time,  # Duration matches wait time
                "sigma_inc": 0,
                "freq": cfg.expt.stark_freq,
                "gain": cfg.expt.stark_gain,
                "phase": 0,
                "type": "flat_top",  # Constant amplitude pulse
            }
            super().make_pulse(pulse, "stark_pulse")
        elif cfg.expt.flux:
            self.declare_gen(cfg.expt.flux_chan, nqz=1, mixer_freq=0)
            if cfg.expt.num_pi > 0:
                # Echo: separate flux pulses for half-segments and full segments
                flux_half = {
                    "chan": cfg.expt.flux_chan,
                    "freq": 0,
                    "phase": 0,
                    "gain": cfg.expt.flux_gain,
                    "length": cfg.expt.wait_time / cfg.expt.num_pi / 2,
                    "type": "const",
                }
                super().make_pulse(flux_half, "flux_half")
                flux_full = {
                    "chan": cfg.expt.flux_chan,
                    "freq": 0,
                    "phase": 0,
                    "gain": cfg.expt.flux_gain,
                    "length": cfg.expt.wait_time / cfg.expt.num_pi,
                    "type": "const",
                }
                super().make_pulse(flux_full, "flux_full")
            else:
                # Ramsey: single flux pulse for full wait time
                flux_pulse = {
                    "chan": cfg.expt.flux_chan,
                    "freq": 0,
                    "phase": 0,
                    "gain": cfg.expt.flux_gain,
                    "length": cfg.expt.wait_time,
                    "type": "const",
                }
                super().make_pulse(flux_pulse, "flux_pulse")
        elif cfg.expt.buffer:
            # Ramsey only: drive the buffer resonator for the whole wait, so
            # the qubit dephases at the measurement-induced rate set by the
            # buffer photon number (T2PhotonCal).
            buffer_gen_args = {"ch": cfg.expt.buffer_chan}
            if self.soccfg["gens"][cfg.expt.buffer_chan].get("has_mixer", False):
                buffer_gen_args["mixer_freq"] = mixer_freq_for(cfg.expt.frequency_b)
            self.declare_gen(**buffer_gen_args)
            buffer_pulse = {
                "chan": cfg.expt.buffer_chan,
                "freq": cfg.expt.frequency_b,
                "phase": 0,
                "gain": cfg.expt.gain_b,
                "length": cfg.expt.wait_time,
                # flat_top (the default), like the FWM buffer, so the photon
                # calibration is in the same gain units (flat_top is half
                # const's amplitude at equal gain). For flat_top, wait_time is
                # the plateau and the ramps add a fixed 0.06 us to the wait.
                "type": cfg.expt.pulse_type_b,
            }
            super().make_pulse(buffer_pulse, "buffer_pulse")

        # Create π pulse for Echo or EF check
        if cfg.expt.experiment_type == "cpmg":
            cfg.device.qubit.pulses.pi_ge.phase = 90 * np.ones(
                len(cfg.device.qubit.f_ge)
            )
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
        elif (
            cfg.expt.checkEF
            or cfg.expt.experiment_type == "echo"
            or reset_plays_pi_pulses(cfg.expt)
        ):
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        if (cfg.expt.checkEF and cfg.expt.experiment_type == "echo") or cfg.expt.reset_f:
            # Create π pulse for EF transition check
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

        # for i in range(1000):
        #     self.nop()

    def _body(self, cfg):
        """
        Define the main body of the pulse sequence.

        Implements the actual T2 measurement sequence:
        - Ramsey: π/2 - wait - π/2
        - Echo: π/2 - wait/2 - π - wait/2 - π/2
        - With options for AC Stark and EF measurements

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)
        # Configure readout
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # For EF transition check in Ramsey: Apply π pulse to excite |g⟩ to |e⟩ first
        if hasattr(cfg.expt, "checkEF") and cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef")  # Small buffer delay

        # First π/2 pulse (preparation)
        self.pulse(ch=self.qubit_ch, name="pi2_prep", t=0.0)

        # Handle different experiment types
        if (
            hasattr(cfg.expt, "acStark") and cfg.expt.acStark
        ):  # Ramsey with AC Stark shift
            # Note: AC Stark shift is not compatible with Echo protocol
            self.delay_auto(t=0.01, tag="wait st")  # Small buffer delay
            self.pulse(
                ch=self.qubit_ch, name="stark_pulse", t=0
            )  # Apply AC Stark pulse
            self.delay_auto(t=0.025, tag="waiting")  # Additional wait time
        elif cfg.expt.flux:
            # Flux pulse applied only during wait segments (not during pi pulses)
            if cfg.expt.num_pi > 0:
                # Echo with flux: flux on during each wait segment
                self.pulse(ch=cfg.expt.flux_chan, name="flux_half", t=0)
                self.delay_auto(t=0.01, tag="wait")

                for i in range(cfg.expt.num_pi):
                    self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
                    self.delay_auto(t=0.01, tag=f"wait_pi{i}")
                    if i < cfg.expt.num_pi - 1:
                        self.pulse(ch=cfg.expt.flux_chan, name="flux_full", t=0)
                        self.delay_auto(t=0.01, tag=f"wait{i}")

                self.pulse(ch=cfg.expt.flux_chan, name="flux_half", t=0)
                self.delay_auto(t=0.01, tag=f"wait{cfg.expt.num_pi}")
            else:
                # Ramsey with flux
                self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse", t=0)
                self.delay_auto(t=0.05, tag="wait")
        elif cfg.expt.buffer:
            # Ramsey with the buffer driven from the end of the prep pulse to
            # the start of the read pulse
            self.delay_auto(t=0.01, tag="wait bf")
            self.pulse(ch=cfg.expt.buffer_chan, name="buffer_pulse", t=0)
            self.delay_auto(t=0.01, tag="wait")
        else:
            # Standard Ramsey or Echo sequence
            # For Echo, divide wait time by (num_pi + 1) to get segments between pulses
            if cfg.expt.num_pi > 0:
                self.delay_auto(t=cfg.expt.wait_time / cfg.expt.num_pi / 2, tag="wait")

                # Apply π pulses for Echo protocol (or multiple-pulse Echo)
                for i in range(cfg.expt.num_pi):
                    self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)  # π pulse
                    if i < cfg.expt.num_pi - 1:
                        self.delay_auto(
                            t=cfg.expt.wait_time / cfg.expt.num_pi + 0.01,
                            tag=f"wait{i}",
                        )  # Wait time
                self.delay_auto(
                    t=cfg.expt.wait_time / cfg.expt.num_pi / 2 + 0.01, tag=f"wait{i+1}"
                )
            else:
                self.delay_auto(t=cfg.expt.wait_time, tag="wait")

        # Second π/2 pulse (readout)
        self.pulse(ch=self.qubit_ch, name="pi2_read", t=0)
        self.delay_auto(t=0.01, tag="wait rd")  # Small buffer delay

        # For EF transition check in Ramsey: Apply π pulse to return to |g⟩ for readout
        if hasattr(cfg.expt, "checkEF") and cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ef 2")  # Small buffer delay

        # Measure the qubit state
        super().measure(cfg)


class T2ProgramASM(QickProgram):
    """
    CPMG program using hardware ASM loops for large num_pi (e.g. 1000).

    Instead of unrolling pi pulses with a Python for-loop (which generates one
    tProc instruction per pulse and overflows instruction memory), this uses
    open_loop/close_loop to repeat the pi-pulse + delay block in hardware.

    Only supports the standard CPMG path (no flux, acStark, or EF).
    """

    def __init__(self, soccfg, final_delay, cfg, final_wait=0):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg, final_wait=0)

    def _initialize(self, cfg):
        """Set up readout, pi/2 prep/read pulses, and the CPMG pi pulse (phase 90)."""
        cfg = AttrDict(self.cfg)

        self.add_loop("wait_loop", cfg.expt.expts)
        super()._initialize(cfg, readout="standard")

        # pi/2 prep pulse (phase=0)
        pulse = {
            "sigma": cfg.expt.sigma,
            "sigma_inc": cfg.expt.sigma_inc,
            "freq": cfg.expt.freq,
            "gain": cfg.expt.gain / 2,
            "phase": 0,
            "type": cfg.expt.type,
        }
        super().make_pulse(pulse, "pi2_prep")

        # pi/2 read pulse (phase advances with Ramsey freq)
        pulse["phase"] = cfg.expt.wait_time * 360 * cfg.expt.ramsey_freq
        super().make_pulse(pulse, "pi2_read")

        # CPMG pi pulse (phase=90 relative to pi/2 pulses)
        cfg.device.qubit.pulses.pi_ge.phase = 90 * np.ones(
            len(cfg.device.qubit.f_ge)
        )
        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        """Play pi/2, then num_pi CPMG pi pulses via a hardware loop, then the phase-advanced pi/2 and readout."""
        cfg = AttrDict(self.cfg)
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        tau = cfg.expt.wait_time / cfg.expt.num_pi / 2

        # pi/2 preparation
        self.pulse(ch=self.qubit_ch, name="pi2_prep", t=0.0)
        self.delay_auto(t=tau, tag="wait_first_half")

        if cfg.expt.num_pi > 1:
            # N-1 pi pulses in hardware loop
            self.open_loop(cfg.expt.num_pi - 1, "cpmg_loop")
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=2 * tau + 0.01, tag="wait_cpmg")
            self.close_loop()

        # Last pi pulse + final half-wait
        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        self.delay_auto(t=tau + 0.01, tag="wait_last_half")

        # pi/2 readout
        self.pulse(ch=self.qubit_ch, name="pi2_read", t=0)
        self.delay_auto(t=0.01, tag="wait_rd")

        super().measure(cfg)


class T2Experiment(QickExperiment):
    """
    T2 Experiment - Supports Ramsey, Echo, and CPMG protocols

    Experimental Config for Ramsey:
    expt = dict(
        experiment_type: "ramsey", "echo", or "cpmg"
        start: total wait time b/w the two pi/2 pulses start sweep [us]
        span: total increment of wait time across experiments [us]
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        acStark: True/False (Ramsey only)
        checkEF: True/False (Ramsey only)
    )

    Additional config for Echo:
    expt = dict(
        num_pi: number of pi pulses
    )
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
        Initialize the T2 experiment.

        This experiment measures T2 using Ramsey, Echo, or CPMG protocols.
        Default `params` values:
        - 'reps': Number of repetitions, doubled from default (default: `2 * self.reps`)
        - 'rounds': Number of software averages, from `self.rounds`
        - 'expts': Number of wait time points (default: 100)
        - 'span': Total span of wait times in µs, set to ~3xT2 (default: `3 * self.cfg.device.qubit[par][qi]`)
        - 'start': Start time for wait sweep in µs (default: 0.01)
        - 'ramsey_freq': Ramsey frequency for phase advancement, 'smart' sets it to 1.5/T2 (default: 'smart')
        - 'reset_mode': "passive" (default), "active" or "f0g1" (see reset_mode_of)
        - 'experiment_type': 'ramsey', 'echo', or 'cpmg' (default: 'ramsey')
        - 'acStark': If True, applies an AC Stark pulse during the wait time (Ramsey only) (default: False)
        - 'checkEF': If True, measures the |e>-|f> transition (default: False)
        - 'fit_two_lines': If True, fits two decaying sines (a parity-split line) and sets
          new_freq to their center (default: False)
        - 'num_pi': Number of π pulses for Echo/CPMG (default: 1 for 'echo', 0 for 'ramsey')

        Args:
            cfg_dict (dict): Configuration dictionary.
            qi (int): Qubit index to measure.
            go (bool): Whether to immediately run the experiment.
            params (dict): Additional parameters to override defaults.
            prefix (str): Filename prefix for saved data.
            fname (str): Full filename for saved data.
            progress (bool): Whether to show a progress bar.
            style (str): Measurement style ('fine' for more averages, 'fast' for fewer points).
            disp_kwargs (dict): Display options.
            min_r2 (float): Minimum R² value for acceptable fit.
            max_err (float): Maximum error for acceptable fit.
            display (bool): Whether to display results.
            print (bool): If True, prints the experiment config and exits.
        """
        # Determine experiment type and parameter name based on protocol
        if "experiment_type" in params and params["experiment_type"] == "echo":
            par = "T2e"  # Echo uses T2e parameter
            name = "echo"
        else:
            par = "T2r"  # Ramsey uses T2r parameter
            name = "ramsey"

        # Set appropriate filename prefix
        if prefix is None:
            ef = "ef_" if "checkEF" in params and params["checkEF"] else ""
            prefix = f"{name}_{ef}qubit{qi}"

        # Initialize parent class
        super().__init__(
            cfg_dict=cfg_dict, prefix=prefix, fname=fname, progress=progress, qi=qi, check_params=check_params
        )

        # Define default parameters
        params_def = {
            "reps": 2 * self.reps,  # Number of repetitions (inner loop)
            "rounds": self.rounds,  # Number of averages (outer loop)
            "expts": 100,  # Number of wait time points
            "span": 3
            * self.cfg.device.qubit[par][
                qi
            ],  # Total span of wait times (μs), set to ~3*T2
            "start": 0.01,  # Start time for wait sweep (μs)
            "ramsey_freq": "smart",  # Ramsey frequency for phase advancement
            "reset_mode": "passive",  # "passive" | "active" | "f0g1"
            # Active reset. QickProgram.reset() plays up to `reset`
            # conditional pi pulses; setup_reset keeps the long passive
            # inter-shot delay so a reset that is not working is still safe.
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
            "qubit": [qi],  # Qubit index as a list
            "experiment_type": "ramsey",  # Default to Ramsey protocol
            "acStark": False,  # No AC Stark shift by default
            "checkEF": False,  # No EF transition check by default
            # Fit two decaying sines, one per line of a charge-parity split
            # transition; new_freq is then the center of the two lines
            "fit_two_lines": False,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],  # Readout channel
            "flux": False,  # Whether to apply flux pulse during wait time
            "buffer": False,  # Whether to drive the buffer during wait time (Ramsey only)
        }
        if params.get("flux", False):
            flux_params = {
                "flux_chan": self.cfg.hw.soc.dacs.flux.ch[qi],
                "flux_gain": 0.1,
            }
            params_def = {**params_def, **flux_params}
        if params.get("buffer", False):
            # The buffer is readout line 0, as in the FWM experiments
            buffer_params = {
                "buffer_chan": self.cfg.hw.soc.dacs.readout.ch[0],
                "frequency_b": self.cfg.device.readout.frequency[0],
                "gain_b": 0,
                # Match the FWM experiments: a flat_top plateau is half a
                # const pulse's amplitude at the same gain (4x in photons)
                "pulse_type_b": pulse_type_from_config(self.cfg.device.get("fwm"), "buffer_pulse_type"),
            }
            params_def = {**params_def, **buffer_params}

        # Adjust parameters based on measurement style
        if style == "fine":
            params_def["rounds"] = (
                params_def["rounds"] * 2
            )  # Double averages for fine measurements
        elif style == "fast":
            params_def["expts"] = 50  # Fewer points for fast measurements

        # Merge user parameters with defaults
        params = {**params_def, **params}

        # Set Ramsey frequency intelligently if "smart" is specified
        if params["ramsey_freq"] == "smart":
            # Set Ramsey frequency to 1.5/T2 for optimal oscillation visibility
            params["ramsey_freq"] = 1.5 / self.cfg.device.qubit[par][qi]

        # Set number of π pulses based on experiment type
        if params["experiment_type"] == "cpmg":
            params_def["num_pi"] = 1  # CPMG requires at least 1 π pulse
        elif params["experiment_type"] == "echo":
            params_def["num_pi"] = 1  # Standard echo has 1 π pulse
        else:
            params_def["num_pi"] = 0  # Ramsey has 0 π pulses

        # Set pulse parameters based on transition type (g-e or e-f)
        if "checkEF" in params and params["checkEF"]:
            # For e-f transition measurements
            cfg_qub = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            # For g-e transition measurements (standard)
            cfg_qub = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]

        # Copy pulse parameters from configuration
        for key in cfg_qub:
            params_def[key] = cfg_qub[key][qi]

        # Final parameter merge and assignment
        self.cfg.expt = {**params_def, **params}

        # Check for unexpected parameters
        super().check_params(params_def)

        if self.cfg.expt.buffer and self.cfg.expt.num_pi > 0:
            raise ValueError("The buffer drive is only implemented for Ramsey (num_pi = 0).")

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

    def acquire(self, progress=False):
        """
        Acquire T2 measurement data.

        This method:
        1. Sets up the wait time sweep parameters
        2. Runs the T2Program to collect data for each wait time
        3. Adjusts x-axis values to account for echo protocol

        Args:
            progress: Whether to show progress bar

        Returns:
            Measurement data dictionary
        """
        # Define parameter metadata for plotting
        self.param = {"label": "wait", "param": "t", "param_type": "time"}

        # Ensure sweep step size is above DAC minimum resolution
        self.check_dac_timing(num_segments=self.cfg.expt.num_pi + 1)

        # Create a 1D sweep for the wait time from start to start+span
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        # Use ASM hardware loop for CPMG with many pi pulses
        if self.cfg.expt.experiment_type == "cpmg" and self.cfg.expt.num_pi > 10:
            prog_cls = T2ProgramASM
        else:
            prog_cls = T2Program

        super().acquire(prog_cls, progress=progress)

        # Adjust x-axis values to account for echo protocol
        # For echo, the effective wait time is longer due to the π pulses
        if self.cfg.expt.num_pi == 0:
            coef = 1
        else:
            coef = 2*self.cfg.expt.num_pi  # For echo, we have num_pi + 1 segments
        self.data["xpts"] = coef * self.data["xpts"]

        return self.data

    def analyze(
        self,
        data=None,
        fit=True,
        fit_twofreq=False,
        refit=False,
        verbose=False,
        **kwargs,
    ):
        """
        Analyze T2 measurement data by fitting to a decaying sinusoid.

        This method:
        1. Fits the data to a decaying sinusoid (with or without slope)
        2. Calculates frequency errors from the fit
        3. Determines the corrected qubit frequency

        Args:
            data: Data dictionary to analyze (uses self.data if None)
            fit: Whether to perform fitting
            fit_twofreq: Whether to fit to a two-frequency model
            refit: Whether to refit without slope
            verbose: Whether to print detailed information
            **kwargs: Additional arguments passed to the analyzer

        Returns:
            Data dictionary with added fit results
        """
        if data is None:
            data = self.data

        # Define indices for fit parameters
        inds = [0, 1, 2, 3, 4]  # yscale, freq, phase_deg, decay, y0

        if fit:
            # Select appropriate fitting function based on parameters
            if self.cfg.expt.fit_two_lines:
                # Two lines: [yscale0, freq0, phase_deg0, decay0, y0, yscale1, freq1, phase_deg1, decay1]
                self.fitfunc = fitter.two_decaysin
                self.fitterfunc = fitter.fittwo_decaysin
            elif fit_twofreq:
                # Two-frequency model for more complex oscillations
                self.fitterfunc = fitter.fittwofreq_decaysin
                self.fitfunc = fitter.twofreq_decaysin
            elif refit:
                # Simple decaying sine without slope
                self.fitfunc = fitter.decaysin
                self.fitterfunc = fitter.fitdecaysin
            else:
                # Decaying sine with slope (default)
                self.fitfunc = fitter.decayslopesin
                self.fitterfunc = fitter.fitdecayslopesin
                # Parameters: yscale, freq, phase_deg, decay, y0, slope

            # Perform the fit
            super().analyze(
                data=data,
                inds=inds,
                **kwargs,
            )

            # If the fit fails, try again without slope
            if not self.status and not refit and not self.cfg.expt.fit_two_lines:
                self.fitfunc = fitter.decaysin
                self.fitterfunc = fitter.fitdecaysin
                super().analyze(
                    data=data,
                    inds=inds,
                    **kwargs,
                )

            # Calculate average fit error
            inds = np.arange(5)
            data["fit_err"] = np.mean(np.abs(data["fit_err_par"][inds]))

            # Calculate frequency adjustments for each data type (amps, avgi, avgq)
            ydata_lab = ["amps", "avgi", "avgq"]
            for i, ydata in enumerate(ydata_lab):
                if isinstance(data["fit_" + ydata], (list, np.ndarray)):
                    # Calculate possible frequency errors
                    # The fitted frequency can be either ramsey_freq + fit_freq or ramsey_freq - fit_freq
                    data["f_adjust_ramsey_" + ydata] = sorted(
                        (
                            self.cfg.expt.ramsey_freq - data["fit_" + ydata][1],
                            self.cfg.expt.ramsey_freq + data["fit_" + ydata][1],
                        ),
                        key=abs,  # Sort by absolute value to get the smallest error first
                    )

                    # For two-frequency model, calculate additional adjustments
                    if fit_twofreq and self.cfg.expt.experiment_type == "ramsey":
                        data["f_adjust_ramsey_" + ydata + "2"] = sorted(
                            (
                                self.cfg.expt.ramsey_freq - data["fit_" + ydata][7],
                                self.cfg.expt.ramsey_freq - data["fit_" + ydata][6],
                            ),
                            key=abs,
                        )

            # Get the best frequency adjustment
            if not self.cfg.device.qubit.tuned_up[self.cfg.expt.qubit[0]]:
                # For untuned qubits, use a more sophisticated method to find the best fit
                fit_pars, fit_err, t2r_adjust, i_best = fitter.get_best_fit(
                    self.data, get_best_data_params=["f_adjust_ramsey"]
                )
            else:
                # For tuned qubits, use the I quadrature adjustment
                t2r_adjust = data["f_adjust_ramsey_avgi"]

            # Store the frequency adjustment
            data["t2r_adjust"] = t2r_adjust

            # Get the reference frequency based on transition type
            if self.cfg.expt.checkEF:
                f_pi_test = self.cfg.device.qubit.f_ef[self.cfg.expt.qubit[0]]
            else:
                f_pi_test = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]

            # Print possible frequency errors if verbose
            if self.cfg.expt.experiment_type == "ramsey" and verbose:
                print(
                    f"Possible errors are {t2r_adjust[0]:.3f} and {t2r_adjust[1]:.3f} MHz "
                    f"for Ramsey frequency {self.cfg.expt.ramsey_freq:.3f} MHz"
                )

            # Store the frequency error and corrected frequency
            data["f_err"] = t2r_adjust[0]  # Use the smallest error
            data["new_freq"] = (
                f_pi_test + t2r_adjust[0]
            )  # Calculate corrected frequency

            if self.cfg.expt.fit_two_lines:
                # Each line sits at drive + ramsey_freq - its fringe frequency,
                # which is right for any line below drive + ramsey_freq. The
                # center of the two does not move with offset charge.
                p = data["best_fit"]
                cov = data["fit_err_" + data["i_best"].decode()]
                data["f_lines"] = np.sort(
                    f_pi_test + self.cfg.expt.ramsey_freq - np.array([p[1], p[6]])
                )
                data["f_splitting"] = data["f_lines"][1] - data["f_lines"][0]
                data["new_freq"] = np.mean(data["f_lines"])
                data["f_center_err"] = np.sqrt(cov[1, 1] + cov[6, 6] + 2 * cov[1, 6]) / 2
                data["f_err"] = data["new_freq"] - f_pi_test

        return data

    def display(
        self,
        data=None,
        fit=True,
        fit_twofreq=False,
        debug=False,
        plot_all=True,
        ax=None,
        savefig=True,
        refit=False,
        show_hist=False,
        rescale=False,
        **kwargs,
    ):
        """
        Display T2 measurement results.

        Creates a plot showing the qubit state vs wait time and the exponentially decaying sinusoidal fit.

        Args:
            data: Data dictionary to display (uses self.data if None)
            fit: Whether to show the fit curve
            fit_twofreq: Whether to use two-frequency model for display
            debug: Whether to show debug information
            plot_all: Whether to make plots for I/Q/Amps or just I
            ax: Matplotlib axis to plot on (creates one if None)
            savefig: Whether to save the figure to disk
            refit: Whether to use refit data for display
            show_hist: Whether to show histogram of the data
            **kwargs: Additional arguments passed to the display function
        """
        if data is None:
            data = self.data

        # Get qubit index for plot title
        q = self.cfg.expt.qubit[0]

        # Set experiment name based on type
        name = "Echo " if self.cfg.expt.experiment_type == "echo" else ""
        if self.cfg.expt.num_pi > 1:
            name += f"{self.cfg.expt.num_pi} π pulses "

        # Set x-axis label
        xlabel = "Wait Time ($\mu$s)"

        # Add EF prefix if checking EF transition
        ef = "EF " if self.cfg.expt.checkEF else ""

        # Create plot title
        title = f"{ef} Ramsey {name}Q{q} (Freq: {self.cfg.expt.ramsey_freq:.4} MHz)"

        # Set up caption parameters to display T2 and frequency values
        if self.cfg.expt.experiment_type == "echo":
            caption_params = [
                {"index": 3, "format": "$T_2$ : {val:.4} $\pm$ {err:.2g} $\mu$s"},
                {"index": 1, "format": "Freq. : {val:.3} $\pm$ {err:.1} MHz"},
            ]
        else:  # ramsey
            caption_params = [
                {
                    "index": 3,
                    "format": "$T_2$ : {val:.4} $\pm$ {err:.2g} $\mu$s",
                },
                {"index": 1, "format": "Freq. : {val:.3} $\pm$ {err:.1} MHz"},
            ]
        if self.cfg.expt.fit_two_lines:
            caption_params.append(
                {"index": 6, "format": "Freq. 2 : {val:.3} $\pm$ {err:.1} MHz"}
            )

        # Call parent class display method
        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            debug=debug,
            show_hist=show_hist,
            fitfunc=self.fitfunc,
            caption_params=caption_params,
            savefig=savefig,
            rescale=rescale,
            **kwargs,
        )


class T2FastFlux(QickFluxSweep):
    """
    T2FastFlux: Sweep T2 across flux gain values.

    Runs a T2Experiment at each flux gain point, collecting T2 vs flux gain (and
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
        experiment_type: 'ramsey' or 'echo' (default: 'ramsey')
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
            prefix = f"t2_fastflux_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, qi=qi)

        sweet_spot, gain_stop = self._init_flux_sweep(qi, params)
        direction = params.get("direction", "pos")
        freq_span = params.get("freq_span", None)

        # start_t2: assumed T2 for the first gain point, sets initial span
        start_t2 = params.pop("start_t2", None)

        params_def = {
            "gain_start": sweet_spot,
            "gain_stop": gain_stop,
            "start": 0.05,
            "expts": 25,
            "expts_gain": 50,
            "freq_span": freq_span,
            "direction": direction,
            "lin_freq": True,
            "t2_max": float("inf"),
        }

        if start_t2 is not None:
            params_def["span"] = 4.1 * start_t2
        params = {**params_def, **params}

        self.expt = T2Experiment(cfg_dict, qi, go=False, params=params, check_params=False)
        params = {**self.expt.cfg.expt, **params, "flux": True}
        self.cfg.expt = {**params_def, **params}

        self._qi = qi
        self._cfg_dict = cfg_dict

        self._build_gain_freq_pts()

        if go:
            self.acquire(progress=progress, display=display)
            self.display()

    def acquire(self, progress=False, display=True):
        """Run the inner T2 experiment at each flux gain point, saving interim data and a CSV."""
        data = {"time": []}
        t2_list = []
        offset_list = []
        amp_list = []
        q_offset_list = []
        q_amp_list = []

        for i, g in enumerate(tqdm(self.gain_pts, disable=not progress)):
            # Update inner experiment config for this gain point
            self.expt.cfg.expt["flux_gain"] = float(g)

            # Run inner T2 scan and analyze
            data_new = self.expt.acquire(progress=False)
            self.expt.analyze(data=data_new)
            if display:
                self.expt.display(data=data_new)

            # Accumulate raw 2D data
            for key in ("avgi", "avgq", "amps", "phases", "xpts"):
                if key in data_new:
                    if i == 0:
                        data[key] = []
                    data[key].append(data_new[key])
            data["time"].append(time.time())

            # T2 fit: [yscale, freq, phase_deg, decay, y0] — T2 is at index 3
            best_fit = data_new.get("best_fit", [np.nan, np.nan, np.nan, np.nan, np.nan])
            t2_val = best_fit[3]
            t2_list.append(t2_val)
            offset_list.append(best_fit[4])
            amp_list.append(best_fit[0])
            fit_q = data_new.get("fit_avgq", [np.nan, np.nan, np.nan, np.nan, np.nan])
            q_offset_list.append(fit_q[4] if len(fit_q) > 4 else np.nan)
            q_amp_list.append(fit_q[0] if len(fit_q) > 0 else np.nan)

            # Adjust span for next scan based on extracted T2
            if np.isfinite(t2_val) and t2_val > 0:
                self.expt.cfg.expt["span"] = 4.1 * min(
                    t2_val, self.cfg.expt.t2_max
                )

            # Store analysis summaries in data for interim save
            data["gain_pts"] = self.gain_pts
            if self.freq_pts is not None:
                data["freq_pts"] = self.freq_pts
            data["t2_list"] = np.array(t2_list)
            data["offset_list"] = np.array(offset_list)
            data["amp_list"] = np.array(amp_list)
            data["q_offset_list"] = np.array(q_offset_list)
            data["q_amp_list"] = np.array(q_amp_list)

            if self.save_interim:
                self.save_data(data=data)

        # Convert lists to arrays
        for k, a in data.items():
            data[k] = np.array(a)
        self.data = data
        return data

    def display(self, data=None, **kwargs):
        """Plot fitted T2 vs flux gain (and vs frequency when a flux model is available)."""
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        gain_pts = data["gain_pts"]
        t2_list = data["t2_list"]

        fig, ax = plt.subplots()
        ax.plot(gain_pts, t2_list, ".-")
        ax.set_xlabel("Flux Gain")
        ax.set_ylabel("$T_2$ ($\mu$s)")
        self.save_fig(fig, suffix="_t2_vs_gain")

        if "freq_pts" in data:
            freq_pts = data["freq_pts"]

            fig, ax = plt.subplots()
            ax.plot(freq_pts, t2_list, ".-")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("$T_2$ ($\mu$s)")
            self.save_fig(fig, suffix="_t2_vs_freq")

            fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
            panels = [
                (axes[0, 0], data["amp_list"], "Amplitude"),
                (axes[0, 1], data["offset_list"], "Offset"),
                (axes[1, 0], data["q_amp_list"], "Q Amplitude"),
                (axes[1, 1], data["q_offset_list"], "Q Offset"),
            ]
            for ax, vals, label in panels:
                vals = np.array(vals, dtype=float)
                mask = np.isfinite(vals)
                if mask.sum() > 2:
                    q1, q3 = np.percentile(vals[mask], [25, 75])
                    iqr = q3 - q1
                    inlier = mask & (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
                    ax.plot(freq_pts[inlier], vals[inlier], ".-")
                    ax.plot(freq_pts[~inlier & mask], vals[~inlier & mask], "rx", ms=6)
                else:
                    ax.plot(freq_pts, vals, ".-")
                ax.set_ylabel(label)
            axes[1, 0].set_xlabel("Frequency (MHz)")
            axes[1, 1].set_xlabel("Frequency (MHz)")
            self.save_fig(fig, suffix="_fit_params")


def buffer_dephasing_per_photon(kappa_mhz, chi_mhz, detuning_mhz):
    """
    Ramsey dephasing and frequency shift per on-resonance buffer photon.

    The buffer is driven at f_g + detuning_mhz, where f_g is its |g>-dressed
    resonance (device.readout.frequency[0]) and f_g + chi_mhz its |e>-dressed
    one. kappa_mhz is the FWHM and chi_mhz the full signed g->e shift, as the
    config stores them. With the steady-state fields
    alpha_x = eps / (kappa/2 - i Delta_x) in each qubit branch,

        Gamma = (kappa/2) |alpha_g - alpha_e|^2      (1/us)
        delta = chi Re(alpha_g alpha_e*)             (rad/us)

    both per n_on = eps^2 / (kappa/2)^2, the photon number a drive at f_g
    gives -- what Stark/PNS and the FWM analyses use. The sign of delta is
    convention-dependent, so fits take its magnitude or a free sign.
    CLAUDE.md "Photon number from Ramsey dephasing (scripts 38, 41)".
    """
    kappa = 2 * np.pi * np.asarray(kappa_mhz, dtype=float)
    chi = 2 * np.pi * np.asarray(chi_mhz, dtype=float)
    delta_g = 2 * np.pi * np.asarray(detuning_mhz, dtype=float)
    alpha_g = 1 / (kappa / 2 - 1j * delta_g)
    alpha_e = 1 / (kappa / 2 - 1j * (delta_g - chi))
    n_on = 1 / (kappa / 2) ** 2
    gamma = kappa / 2 * np.abs(alpha_g - alpha_e) ** 2 / n_on
    shift = chi * np.real(alpha_g * np.conj(alpha_e)) / n_on
    return gamma, shift


def ramsey_photons_per_rate(kappa_mhz, chi_mhz, detuning_mhz=None):
    """
    Buffer photons per unit Ramsey dephasing rate (us).

    kappa_mhz is the buffer FWHM and chi_mhz the full g->e shift, as the
    config stores them; detuning_mhz is the drive's offset from f_g, default
    the dressed midpoint chi_mhz/2. Returns (n_bar at the drive frequency,
    on-resonance n_bar) per 1/us of dephasing; the second is what a drive at
    the |g>-dressed resonance with the same incident power gives.
    """
    if detuning_mhz is None:
        detuning_mhz = chi_mhz / 2
    gamma, _ = buffer_dephasing_per_photon(kappa_mhz, chi_mhz, detuning_mhz)
    on_resonance = 1 / gamma
    kappa = 2 * np.pi * kappa_mhz
    lorentzian = (kappa / 2) ** 2 / ((kappa / 2) ** 2 + (2 * np.pi * detuning_mhz) ** 2)
    return on_resonance * lorentzian, on_resonance


def ramsey_kappa_from_ratio(ratio, chi_mhz, detuning_mhz, kappa_guess_mhz):
    """
    Buffer kappa (MHz FWHM) at which dephasing / |frequency shift| per photon
    equals `ratio` (both measured slopes vs gain^2, shift in rad/us).

    The ratio is independent of the photon number, so a Ramsey run fixes
    kappa given chi and the drive detuning. Returns the root nearest
    kappa_guess_mhz, or NaN if there is none in 0.01-2 MHz.
    """
    from scipy.optimize import brentq

    def residual(kappa):
        gamma, shift = buffer_dephasing_per_photon(kappa, chi_mhz, detuning_mhz)
        return gamma / abs(shift) - ratio

    grid = np.geomspace(0.01, 2.0, 400)
    values = np.array([residual(k) for k in grid])
    roots = [brentq(residual, a, b) for a, b, fa, fb in
             zip(grid[:-1], grid[1:], values[:-1], values[1:]) if fa * fb < 0]
    if not roots:
        return np.nan
    return min(roots, key=lambda k: abs(k - kappa_guess_mhz))


class T2PhotonCal(QickExperiment):
    """
    Buffer photon-number calibration from measurement-induced dephasing.

    Runs a Ramsey T2Experiment with the buffer driven for the whole wait, at a
    series of buffer gains starting from 0, then once more at gain 0 as a
    drift check. Buffer photons both dephase the qubit (1/T2 rises) and shift
    its frequency (the Ramsey fringe moves), each linearly in gain^2.
    `buffer_dephasing_per_photon` converts either slope to on-resonance
    photons per gain^2 -- photons for a drive at f_g =
    device.readout.frequency[0], the units of the Stark/PNS calibration -- for
    a drive at any frequency_b. The ratio of the two slopes depends only on
    kappa (given chi), so each run also gives a self-consistent kappa.

    frequency_b defaults to f_g, where Stark and FWM drive. There the
    dephasing per photon is ~kappa/2, so the conversion needs kappa only
    linearly; at the dressed midpoint f_g + chi/2 it needs ~1/kappa^3.

    Between gain points the Ramsey window follows the last measured T2 with
    T2Experiment's own rules (span = 3 T2, ramsey_freq = 4.5 / span), so a
    user span / ramsey_freq only applies to the gain-0 points.

    Parameters (passed via params dict):
        max_gain_b: Largest buffer gain; points are linspace(0, max_gain_b)
        gain_b_expts: Number of buffer gains, including 0 (the closing gain-0
            reference is extra)
        frequency_b: Buffer drive frequency (MHz), default f_g
        kappa_b: Buffer linewidth kappa/2pi (MHz, FWHM), default device.readout.kappa[0]
        chi_b: Full g->e buffer shift (MHz), default device.readout.chi[0]
        pulse_type_b: Buffer pulse shape, "flat_top" (default, as in the FWM
            experiments) or "const"; photons per gain^2 only transfer
            between experiments using the same shape
        Everything else is forwarded to T2Experiment.
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
            prefix = f"t2_photon_cal_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        # The buffer is readout line 0. device.readout.frequency is its
        # |g>-dressed resonance and chi the full g->e shift.
        readout = self.cfg.device.readout
        params_def = {
            "max_gain_b": 0.03,
            "gain_b_expts": 10,
            "frequency_b": readout.frequency[0],
            "kappa_b": readout.kappa[0],
            "chi_b": readout.chi[0],
        }
        params = {**params_def, **params, "buffer": True}

        self.expt = T2Experiment(cfg_dict, qi=qi, go=False, params=params, check_params=False)
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        if go:
            self.run(progress=progress, display=display)

    def acquire(self, progress=False):
        """Run the inner Ramsey at each buffer gain, then a closing gain-0 reference."""
        gains = np.linspace(0, self.cfg.expt.max_gain_b, self.cfg.expt.gain_b_expts)
        inner_cfg = self.expt.cfg.expt
        start_window = (inner_cfg["span"], inner_cfg["ramsey_freq"])
        points = [(float(g), False) for g in gains] + [(0.0, True)]

        data = {"gain_b_pts": [], "reference": [], "ramsey_freq": [], "xpts": [],
                "ydata": [], "fit_pars": [], "t2": [], "t2_err": [], "time": []}
        for gain_b, reference in tqdm(points, disable=not progress):
            if reference:
                # Same window as the first gain-0 point, so the two compare directly
                inner_cfg["span"], inner_cfg["ramsey_freq"] = start_window
            inner_cfg["gain_b"] = gain_b
            inner = self.expt.acquire(progress=False)
            inner = self.expt.analyze(data=inner)

            # Pad a no-slope fit with slope = 0 so every row is a
            # decayslopesin parameter set.
            fit_pars = np.zeros(6)
            fit_pars[: len(inner["best_fit"])] = inner["best_fit"]
            t2 = fit_pars[3]

            data["gain_b_pts"].append(gain_b)
            data["reference"].append(reference)
            data["ramsey_freq"].append(inner_cfg["ramsey_freq"])
            data["xpts"].append(inner["xpts"])
            data["ydata"].append(inner[inner["i_best"].decode()])
            data["fit_pars"].append(fit_pars)
            data["t2"].append(t2)
            data["t2_err"].append(abs(inner["fit_err_par"][3] * t2))
            data["time"].append(time.time())

            # T2 only shortens as gain grows, so never widen the window: that
            # also keeps one bad fit from blowing the span up.
            if np.isfinite(t2) and t2 > 0:
                inner_cfg["span"] = min(3 * t2, inner_cfg["span"])
                inner_cfg["ramsey_freq"] = 4.5 / inner_cfg["span"]

        for key, val in data.items():
            data[key] = np.array(val)
        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        """Fit 1/T2 and the Ramsey frequency shift vs gain^2; convert both to photons at f_g."""
        if data is None:
            data = self.data

        expt = self.cfg.expt
        gain2 = data["gain_b_pts"] ** 2
        t2 = data["t2"]
        # Runs from before the closing reference / stored ramsey_freq have neither
        if "reference" in data:
            reference = np.asarray(data["reference"]).astype(bool)
        else:
            reference = np.zeros(len(t2), bool)
        # Rates in 1/us; Gamma has the same relative error as T2
        gamma = 1 / t2
        gamma_err = data["t2_err"] / t2**2

        detuning = expt.frequency_b - self.cfg.device.readout.frequency[0]
        _, photons_per_rate = ramsey_photons_per_rate(expt.kappa_b, expt.chi_b, detuning)
        _, shift_per_photon = buffer_dephasing_per_photon(expt.kappa_b, expt.chi_b, detuning)

        data["detuning"] = detuning
        data["gamma"] = gamma
        data["gamma_err"] = gamma_err
        data["d_gamma"] = gamma - gamma[0]
        data["n_bar"] = data["d_gamma"] * photons_per_rate
        data["photons_per_rate"] = photons_per_rate
        if "ramsey_freq" in data:
            # MHz; the sign is convention-dependent, only the slope's size is used
            data["shift"] = data["fit_pars"][:, 1] - data["ramsey_freq"]

        if reference.any():
            ref = np.flatnonzero(reference)[-1]
            drift = f"T2 {t2[0]:.2f} -> {t2[ref]:.2f} us"
            if "shift" in data:
                drift += f", Ramsey frequency {1e3 * (data['shift'][ref] - data['shift'][0]):+.1f} kHz"
            print(f"Gain-0 drift over the run: {drift}")

        if fit:
            ok = np.isfinite(gamma) & (t2 > 0) & ~reference
            if ok.sum() < 4:
                print(f"Only {ok.sum()} good T2 fits; skipping the 1/T2 vs gain^2 fit")
                return data
            w = 1 / gamma_err[ok]
            if not np.all(np.isfinite(w)):
                w = None
            popt, pcov = np.polyfit(gain2[ok], gamma[ok], 1, w=w, cov=True)
            perr = np.sqrt(np.diag(pcov))
            data["gamma_fit"] = popt  # [slope (1/us per gain^2), 1/T2(0)]
            data["gamma_fit_err"] = perr
            data["n_per_gain2"] = popt[0] * photons_per_rate
            data["n_per_gain2_err"] = perr[0] * photons_per_rate
            print(
                f"Buffer at f_g{detuning:+.4f} MHz (kappa/2pi = {expt.kappa_b:.4g} MHz, "
                f"chi/2pi = {expt.chi_b:.4g} MHz full shift)"
            )
            print(
                f"From 1/T2:   {data['n_per_gain2']:.4g} +/- {data['n_per_gain2_err']:.2g} "
                f"on-resonance photons per gain^2"
            )

            if "shift" in data:
                popt_s, pcov_s = np.polyfit(gain2[ok], data["shift"][ok], 1, cov=True)
                perr_s = np.sqrt(np.diag(pcov_s))
                data["shift_fit"] = popt_s  # [MHz per gain^2, MHz]
                data["shift_fit_err"] = perr_s
                scale = 2 * np.pi / abs(shift_per_photon)
                data["n_per_gain2_shift"] = abs(popt_s[0]) * scale
                data["n_per_gain2_shift_err"] = perr_s[0] * scale
                # Near f_g the shift is ~(kappa/2chi)^2 of the dephasing and usually unresolved
                if abs(popt_s[0]) < 3 * perr_s[0]:
                    print(f"Ramsey shift not resolved ({1e3 * popt_s[0]:.0f} +/- "
                          f"{1e3 * perr_s[0]:.0f} kHz per gain^2); no shift-based photons or kappa")
                    return data
                ratio = popt[0] / (2 * np.pi * abs(popt_s[0]))
                data["kappa_self"] = ramsey_kappa_from_ratio(
                    ratio, expt.chi_b, detuning, expt.kappa_b)
                print(
                    f"From shift:  {data['n_per_gain2_shift']:.4g} +/- "
                    f"{data['n_per_gain2_shift_err']:.2g} on-resonance photons per gain^2 "
                    f"({1e3 * popt_s[0]:.0f} kHz per gain^2)"
                )
                print(
                    f"Self-consistent kappa (shift/dephasing ratio): "
                    f"{data['kappa_self']:.4g} MHz vs {expt.kappa_b:.4g} assumed"
                )

        return data

    def display(self, data=None, fit=True, **kwargs):
        """Plot 1/T2 and the Ramsey shift vs gain^2, and the Ramsey traces."""
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        gains = data["gain_b_pts"]
        gain2 = gains**2
        if "reference" in data:
            reference = np.asarray(data["reference"]).astype(bool)
        else:
            reference = np.zeros(len(gains), bool)
        fig, (ax_rate, ax_shift, ax_trace) = plt.subplots(1, 3, figsize=(17, 5))

        ax_rate.errorbar(gain2[~reference], data["gamma"][~reference],
                         yerr=data["gamma_err"][~reference], fmt="o", ms=4)
        if reference.any():
            ax_rate.plot(gain2[reference], data["gamma"][reference], "o", ms=6,
                         mfc="none", color="k", label="closing gain-0")
        if fit and "gamma_fit" in data:
            dense = np.linspace(0, gain2.max(), 100)
            ax_rate.plot(
                dense, np.polyval(data["gamma_fit"], dense), lw=1,
                label=(f"{data['n_per_gain2']:.3g} $\\pm$ {data['n_per_gain2_err']:.2g}"
                       " photons / gain$^2$ at $f_g$"),
            )
        if ax_rate.get_legend_handles_labels()[0]:
            ax_rate.legend(fontsize="small")
        ax_rate.set_xlabel("Buffer gain$^2$")
        ax_rate.set_ylabel("$1/T_{2,\\mathrm{Ramsey}}$ ($\\mu$s$^{-1}$)")

        if "shift" in data:
            shift = 1e3 * (data["shift"] - data["shift"][0])
            ax_shift.plot(gain2[~reference], shift[~reference], "o", ms=4)
            if reference.any():
                ax_shift.plot(gain2[reference], shift[reference], "o", ms=6,
                              mfc="none", color="k")
            if fit and "shift_fit" in data:
                dense = np.linspace(0, gain2.max(), 100)
                ax_shift.plot(
                    dense, 1e3 * data["shift_fit"][0] * dense, lw=1,
                    label=(f"{data['n_per_gain2_shift']:.3g} $\\pm$ "
                           f"{data['n_per_gain2_shift_err']:.2g} photons / gain$^2$ at $f_g$"),
                )
                ax_shift.legend(fontsize="small")
        ax_shift.set_xlabel("Buffer gain$^2$")
        ax_shift.set_ylabel("Ramsey frequency shift (kHz)")

        # Every trace normalized to [0, 1] and stacked, with its fit
        cmap = plt.get_cmap("viridis")
        for i, gain_b in enumerate(gains):
            x, y = data["xpts"][i], data["ydata"][i]
            lo, hi = np.nanmin(y), np.nanmax(y)
            scale = hi - lo if hi > lo else 1
            color = "k" if reference[i] else cmap(i / max(len(gains) - 1, 1))
            ax_trace.plot(x, (y - lo) / scale + 1.2 * i, "o", ms=2, color=color,
                          label=f"{gain_b:.3g}" + (" (ref)" if reference[i] else ""))
            if np.all(np.isfinite(data["fit_pars"][i])):
                dense = np.linspace(x.min(), x.max(), 400)
                yfit = fitter.decayslopesin(dense, *data["fit_pars"][i])
                ax_trace.plot(dense, (yfit - lo) / scale + 1.2 * i, lw=1, color=color)
        ax_trace.set_xlabel("Wait Time ($\\mu$s)")
        ax_trace.set_ylabel("Normalized signal + offset")
        ax_trace.legend(fontsize="small", title="buffer gain")

        fig.tight_layout()
        self.save_fig(fig)
        self.save_config()

    def update(self, verbose=True):
        """
        Write the photon calibration to the config.

        device.readout.nramsey[qi] gets the on-resonance photons per gain^2.
        device.fwm.buffer_dephasing_ramsey gets the raw slope d(1/T2)/d(gain^2)
        and buffer_shift_ramsey the Ramsey shift slope (MHz per gain^2), stated
        at this run's buffer attenuation, pulse shape and drive detuning from
        f_g; consumers turn them into photons with the config's kappa and chi
        at analysis time. CLAUDE.md "Photon number from Ramsey dephasing".
        """
        if "n_per_gain2" not in self.data:
            print("No photons-per-gain fit available; nothing to update")
            return
        config.update_readout(
            self.config_file, "nramsey", self.data["n_per_gain2"],
            self.cfg.expt.qubit[0], verbose=verbose,
        )

        soc = self.cfg.hw.soc
        applied = soc.rfboard_active.dac_readout[str(soc.dacs.readout.ch[0])]
        fields = [
            ("buffer_dephasing_ramsey", [self.data["gamma_fit"][0]]),
            ("buffer_dephasing_ramsey_attn_db", float(applied.attn1) + float(applied.attn2)),
            ("buffer_dephasing_ramsey_pulse", self.cfg.expt.pulse_type_b),
            ("buffer_dephasing_ramsey_detuning", float(self.data["detuning"])),
        ]
        if "shift_fit" in self.data:
            fields.append(("buffer_shift_ramsey", [self.data["shift_fit"][0]]))
        for field, value in fields:
            config.update_fwm(self.config_file, field, value, verbose=verbose)


def buffer_dephasing_model(freqs, n_on, kappa, chi, f_g, sign=1.0):
    """
    Ramsey dephasing (1/us) and frequency shift (MHz) for a buffer drive at
    `freqs` producing n_on photons when driven at f_g. See
    buffer_dephasing_per_photon; `sign` absorbs the shift's convention.
    """
    gamma, shift = buffer_dephasing_per_photon(kappa, chi, np.asarray(freqs) - f_g)
    return n_on * gamma, sign * n_on * shift / (2 * np.pi)


def fit_buffer_dephasing_spectrum(freqs, d_gamma, d_gamma_err, d_shift, seed):
    """
    Joint fit of Ramsey dephasing and frequency shift vs buffer drive frequency.

    Args:
        freqs: buffer drive frequencies (MHz)
        d_gamma, d_gamma_err: buffer-induced 1/T2 and its error (1/us)
        d_shift: buffer-induced Ramsey frequency shift (MHz), any sign convention
        seed: dict with n_on, kappa, chi, f_g starting values

    Both observables share n_on (photons at f_g for this drive), kappa, chi
    and f_g, so the spectrum fixes all four without any config input. The
    shift's sign convention is fit as a discrete choice. Each block is scaled
    by its own reduced chi^2 (the shift has no per-point error), then refit.

    Returns dict of best values, 1-sigma errors, sign, and reduced chi^2s.
    """
    from scipy.optimize import least_squares

    freqs = np.asarray(freqs, dtype=float)
    names = ("n_on", "kappa", "chi", "f_g")
    x0 = np.array([seed[n] for n in names], dtype=float)
    lower = [0, 0.01, -np.inf, -np.inf]
    upper = [np.inf, 2.0, np.inf, np.inf]

    def run(sign, sigma_shift, gamma_scale):
        def resid(p):
            g, s = buffer_dephasing_model(freqs, *p, sign=sign)
            return np.concatenate([(g - d_gamma) / (d_gamma_err * gamma_scale),
                                   (s - d_shift) / sigma_shift])
        return least_squares(resid, x0, bounds=(lower, upper), x_scale="jac")

    n = len(freqs)
    # Pick the sign with fixed weights: costs after per-block rescaling are not comparable
    sign = min((1.0, -1.0), key=lambda s: run(s, 1e-3, 1.0).cost)
    sigma_shift, gamma_scale = 1e-3, 1.0
    for _ in range(2):
        res = run(sign, sigma_shift, gamma_scale)
        r_gamma, r_shift = res.fun[:n], res.fun[n:]
        gamma_scale *= max(1.0, np.sqrt(np.mean(r_gamma**2)))
        sigma_shift *= np.sqrt(np.mean(r_shift**2))
    res = run(sign, sigma_shift, gamma_scale)
    dof = max(res.fun.size - len(x0), 1)
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * max(1.0, 2 * res.cost / dof)
        errs = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        errs = np.full(len(x0), np.nan)
    out = {name: float(v) for name, v in zip(names, res.x)}
    out.update({f"{name}_err": float(e) for name, e in zip(names, errs)})
    out.update(sign=sign, sigma_shift=sigma_shift, gamma_scale=gamma_scale)
    return out


class BufferDephasingSpec(QickExperiment):
    """
    Buffer spectroscopy read out through the qubit: Ramsey vs buffer drive frequency.

    At one fixed buffer gain, runs a Ramsey T2Experiment with the buffer
    driven for the whole wait at each of frequency_b_pts, with a gain-0
    Ramsey every ref_every points (and at the end) as the drift baseline. The
    buffer-induced dephasing has a peak of width ~kappa at each dressed
    resonance f_g and f_g + chi, and the Ramsey frequency shift a dispersive
    shape; `fit_buffer_dephasing_spectrum` fits both together for f_g, chi,
    kappa and n_on, the photons this gain gives at f_g. n_on / gain_b^2 is
    then the on-resonance photons per gain^2 of the Stark/PNS calibration,
    with no config kappa or chi involved. CLAUDE.md "Photon number from
    Ramsey dephasing (scripts 38, 41)".

    The Ramsey window tracks the previous point's T2 (span = 3 T2, at most
    the gain-0 span); the sweep is monotonic in frequency so T2 changes
    smoothly between neighbours. References use the gain-0 window.

    Parameters (passed via params dict):
        gain_b: Buffer gain for every non-reference point
        frequency_b_pts: Buffer drive frequencies (MHz); default
            `default_frequencies()` around f_g and f_g + chi from the config
        ref_every: A gain-0 reference before every this many points
        kappa_b, chi_b: Seeds for the fit (MHz), default device.readout
        Everything else is forwarded to T2Experiment.
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
            prefix = f"buffer_dephasing_spec_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        readout = self.cfg.device.readout
        params_def = {
            "gain_b": 0.03,
            "frequency_b_pts": self.default_frequencies(readout.frequency[0], readout.chi[0]),
            "ref_every": 8,
            "kappa_b": readout.kappa[0],
            "chi_b": readout.chi[0],
        }
        params = {**params_def, **params, "buffer": True}
        params["frequency_b_pts"] = [float(f) for f in params["frequency_b_pts"]]

        self.expt = T2Experiment(cfg_dict, qi=qi, go=False, params=params, check_params=False)
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        if go:
            self.run(progress=progress, display=display)

    @staticmethod
    def default_frequencies(f_g, chi, half_width=0.35, step=0.05):
        """
        Dense around each dressed resonance (f_g and f_g + chi), plus the
        midpoint and one point outside each, ascending.
        """
        window = np.arange(-half_width, half_width + step / 2, step)
        f_e = f_g + chi
        pts = np.concatenate([f_g + window, f_e + window,
                              [f_g + chi / 2, f_g - 2 * half_width * np.sign(chi),
                               f_e + 2 * half_width * np.sign(chi)]])
        return np.unique(np.round(pts, 4))

    def acquire(self, progress=False):
        """Ramsey at each buffer frequency, with gain-0 references interleaved."""
        expt = self.cfg.expt
        inner_cfg = self.expt.cfg.expt
        start_window = (inner_cfg["span"], inner_cfg["ramsey_freq"])

        points = []
        for i, f in enumerate(expt.frequency_b_pts):
            if i % expt.ref_every == 0:
                points.append((f, 0.0, True))
            points.append((f, expt.gain_b, False))
        points.append((expt.frequency_b_pts[-1], 0.0, True))

        data = {"frequency_b_pts": [], "gain_b_pts": [], "reference": [], "ramsey_freq": [],
                "xpts": [], "ydata": [], "fit_pars": [], "t2": [], "t2_err": [], "time": []}
        for frequency_b, gain_b, reference in tqdm(points, disable=not progress):
            if reference:
                inner_cfg["span"], inner_cfg["ramsey_freq"] = start_window
            inner_cfg["frequency_b"] = frequency_b
            inner_cfg["gain_b"] = gain_b
            inner = self.expt.acquire(progress=False)
            inner = self.expt.analyze(data=inner)

            fit_pars = np.zeros(6)
            fit_pars[: len(inner["best_fit"])] = inner["best_fit"]
            t2 = fit_pars[3]

            data["frequency_b_pts"].append(frequency_b)
            data["gain_b_pts"].append(gain_b)
            data["reference"].append(reference)
            data["ramsey_freq"].append(inner_cfg["ramsey_freq"])
            data["xpts"].append(inner["xpts"])
            data["ydata"].append(inner[inner["i_best"].decode()])
            data["fit_pars"].append(fit_pars)
            data["t2"].append(t2)
            data["t2_err"].append(abs(inner["fit_err_par"][3] * t2))
            data["time"].append(time.time())

            # Next window from this T2, never wider than the gain-0 one
            if np.isfinite(t2) and t2 > 0:
                inner_cfg["span"] = min(3 * t2, start_window[0])
                inner_cfg["ramsey_freq"] = 4.5 / inner_cfg["span"]

        for key, val in data.items():
            data[key] = np.array(val)
        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        """Subtract the drifting gain-0 baseline and fit the dephasing spectrum."""
        if data is None:
            data = self.data

        expt = self.cfg.expt
        reference = np.asarray(data["reference"]).astype(bool)
        t2 = data["t2"]
        gamma = 1 / t2
        gamma_err = data["t2_err"] / t2**2
        shift = data["fit_pars"][:, 1] - data["ramsey_freq"]
        times = data["time"]

        # Baseline at each point's time, interpolated between references
        good_ref = reference & np.isfinite(gamma) & (t2 > 0)
        gamma0 = np.interp(times, times[good_ref], gamma[good_ref])
        shift0 = np.interp(times, times[good_ref], shift[good_ref])
        data["gamma"] = gamma
        data["gamma_err"] = gamma_err
        data["shift"] = shift
        data["d_gamma"] = gamma - gamma0
        data["d_shift"] = shift - shift0
        print(
            f"Gain-0 references: T2 {np.min(t2[good_ref]):.2f}-{np.max(t2[good_ref]):.2f} us, "
            f"Ramsey frequency spread {1e3 * np.ptp(shift[good_ref]):.1f} kHz"
        )

        if fit:
            ok = ~reference & np.isfinite(gamma) & (t2 > 0)
            freqs = data["frequency_b_pts"][ok]
            readout = self.cfg.device.readout
            per_photon, _ = buffer_dephasing_per_photon(expt.kappa_b, expt.chi_b, 0.0)
            seed = {
                "n_on": max(np.max(data["d_gamma"][ok]), 1e-3) / per_photon,
                "kappa": expt.kappa_b,
                "chi": expt.chi_b,
                "f_g": readout.frequency[0],
            }
            fit_out = fit_buffer_dephasing_spectrum(
                freqs, data["d_gamma"][ok], gamma_err[ok], data["d_shift"][ok], seed)
            for key, val in fit_out.items():
                data[f"spec_{key}"] = val
            data["n_per_gain2"] = fit_out["n_on"] / expt.gain_b**2
            data["n_per_gain2_err"] = fit_out["n_on_err"] / expt.gain_b**2
            print(
                f"f_g = {fit_out['f_g']:.4f} +/- {fit_out['f_g_err']:.4f} MHz "
                f"(config {readout.frequency[0]:.4f})\n"
                f"chi = {fit_out['chi']:.4f} +/- {fit_out['chi_err']:.4f} MHz "
                f"(config {readout.chi[0]:.4f})\n"
                f"kappa = {fit_out['kappa']:.4f} +/- {fit_out['kappa_err']:.4f} MHz "
                f"(config {readout.kappa[0]:.4f})\n"
                f"{data['n_per_gain2']:.4g} +/- {data['n_per_gain2_err']:.2g} "
                f"on-resonance photons per gain^2"
            )

        return data

    def display(self, data=None, fit=True, **kwargs):
        """Dephasing and shift vs buffer frequency with the fit; references vs time."""
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        reference = np.asarray(data["reference"]).astype(bool)
        freqs = data["frequency_b_pts"]
        fig, (ax_gamma, ax_shift, ax_ref) = plt.subplots(1, 3, figsize=(17, 5))

        ax_gamma.errorbar(freqs[~reference], data["d_gamma"][~reference],
                          yerr=data["gamma_err"][~reference], fmt="o", ms=4)
        ax_shift.plot(freqs[~reference], 1e3 * data["d_shift"][~reference], "o", ms=4)
        if fit and "spec_n_on" in data:
            dense = np.linspace(freqs.min(), freqs.max(), 600)
            g, s = buffer_dephasing_model(
                dense, data["spec_n_on"], data["spec_kappa"], data["spec_chi"],
                data["spec_f_g"], sign=data["spec_sign"])
            label = (f"$\\kappa$ = {data['spec_kappa']:.3f}, $\\chi$ = {data['spec_chi']:.3f} MHz, "
                     f"{data['n_per_gain2']:.3g} photons / gain$^2$ at $f_g$")
            ax_gamma.plot(dense, g, lw=1, label=label)
            ax_shift.plot(dense, 1e3 * s, lw=1)
            for axis in (ax_gamma, ax_shift):
                for f in (data["spec_f_g"], data["spec_f_g"] + data["spec_chi"]):
                    axis.axvline(f, color="0.6", ls="--", lw=1)
            ax_gamma.legend(fontsize="small")
        ax_gamma.set_xlabel("Buffer frequency (MHz)")
        ax_gamma.set_ylabel("Buffer-induced $1/T_2$ ($\\mu$s$^{-1}$)")
        ax_shift.set_xlabel("Buffer frequency (MHz)")
        ax_shift.set_ylabel("Ramsey frequency shift (kHz)")

        minutes = (data["time"][reference] - data["time"][0]) / 60
        ax_ref.plot(minutes, data["t2"][reference], "o-", ms=4, label="T2 (us)")
        ax_ref.set_xlabel("Time (min)")
        ax_ref.set_ylabel("Gain-0 $T_2$ ($\\mu$s)")
        ax_ref2 = ax_ref.twinx()
        ax_ref2.plot(minutes, 1e3 * (data["shift"][reference] - data["shift"][reference][0]),
                     "s--", ms=4, color="C1")
        ax_ref2.set_ylabel("Gain-0 Ramsey frequency drift (kHz)", color="C1")

        fig.tight_layout()
        self.save_fig(fig)
        self.save_config()

    def update(self, verbose=True):
        """
        Record the spectrum fit in device.fwm, tagged like the other buffer
        calibrations: buffer_photon_ramsey_spec = [photons per gain^2 at f_g],
        buffer_ramsey_spec_fit = [f_g, chi, kappa] (MHz), plus _attn_db / _pulse.
        device.readout is left alone; which chi/kappa it holds is decided when
        the calibrations are reconciled.
        """
        if "n_per_gain2" not in self.data:
            print("No spectrum fit available; nothing to update")
            return
        soc = self.cfg.hw.soc
        applied = soc.rfboard_active.dac_readout[str(soc.dacs.readout.ch[0])]
        d = self.data
        for field, value in (
            ("buffer_photon_ramsey_spec", [d["n_per_gain2"]]),
            ("buffer_ramsey_spec_fit", [d["spec_f_g"], d["spec_chi"], d["spec_kappa"]]),
            ("buffer_photon_ramsey_spec_attn_db", float(applied.attn1) + float(applied.attn2)),
            ("buffer_photon_ramsey_spec_pulse", self.cfg.expt.pulse_type_b),
        ):
            config.update_fwm(self.config_file, field, value, verbose=verbose)
