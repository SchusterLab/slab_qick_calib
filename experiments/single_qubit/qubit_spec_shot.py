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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.asm_v2 import QickSweep1D
from scipy.stats import f as f_dist
from tqdm import tqdm
from ...calib import readout_helpers as helpers

from ...exp_handling.datamanagement import AttrDict, SlabFile
from ..general.qick_experiment import QickExperiment, QickExperiment2DSimple
from ..general.qick_program import QickProgram, mixer_freq_for, reset_mode_of, reset_plays_pi_pulses, pulse_type_from_config
from ...helpers import config

from ...analysis import fitting as fitter
FIT_FUNC = fitter.lorfunc
FITTER_FUNC = fitter.fitlor
XLABEL = "Qubit Frequency (MHz)"
YLABEL = "Qubit Gain (DAC level)"
DCYLABEL = "DC Bias (V)"

class QubitSpecShotProgram(QickProgram):
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
        buffer_ch = cfg.hw.soc.dacs.readout.ch[0]
        buffer_gen_args = {"ch": buffer_ch}
        if self.soccfg["gens"][buffer_ch].get("has_mixer", False):
            buffer_gen_args["mixer_freq"] = mixer_freq_for(cfg.expt.frequency_b)
        self.declare_gen(**buffer_gen_args)

        # Define the probe pulse with variable frequency. It goes on the pump
        # line, which is the qubit channel unless hw.soc.dacs.pump says
        # otherwise.
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
        self.add_loop("freq_loop", cfg.expt.expts)

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
            self.pulse(ch=cfg.hw.soc.dacs.readout.ch[0], name="buffer_pulse", t=cfg.expt.delay_b)

        # Add delay if separate readout is enabled
        if cfg.expt.sep_readout:
            self.delay_auto(t=0.01, tag="wait")

        # If checking EF transition, apply second pi pulse to return to |g> for readout
        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait 2")

        # Perform measurement
        super().measure(cfg)


class QubitSpecShot(QickExperiment):
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
    - 'reset_mode': "passive" | "active" | "f0g1"

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
    - 'reset_mode': "passive"
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
            "length_b": 10,
            "frequency_b": self.cfg.device.readout.frequency[0],
            "pulse_type_b": pulse_type_from_config(self.cfg.device.get("fwm"), "buffer_pulse_type"),
            'delay_b': 0,
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            # Which reset runs between shots: "passive" (the full T1 wait),
            # "active" (conditional pi pulses) or "f0g1" (the unconditional
            # all-microwave drive). QickProgram.measure() dispatches on it, and
            # QickProgram.reset() then plays up to `reset` conditional pi
            # pulses. setup_reset keeps the long passive inter-shot delay so
            # a reset that is not working cannot quietly corrupt the data.
            "reset_mode": "passive",
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            # Calibration only: re-measure after the final pi pulse so the
            # last recorded round is the true post-reset state. Costs one
            # readout per shot and nothing acts on it, so leave it off in use.
            "reset_verify": False,
            # Add a pi_ef ahead of each pi_ge, for sequences that can leave
            # population in |f>. Needs reset >= 2: |f> clears in one round but
            # |e> is driven to |f> first and clears on the next.
            # This class has no f-state scan, so off unless asked for.
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
        
        # print(params)
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
        self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}

        # Configure frequency sweep
        self.cfg.expt.frequency_p = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
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
        
        if 'shots' in data:
            self.add_p_e(data)
            if not self.cfg.expt.save_shots:
                data.pop("shots")  # Remove raw shots from data to save space


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


BAND_FIT_TYPES = ("quadratic", "linear", "linear_quadratic", "cubic", "quartic")


def expected_fwm_frequency(cfg, qi, buffer_qi=0):
    """Zero-pump-gain FWM frequency: f_ge + f_waste - chi_waste - f_buffer."""
    dev = cfg.device
    return (
        dev.qubit.f_ge[qi]
        + dev.readout.frequency[qi]
        - dev.readout.chi[qi]
        - dev.readout.frequency[buffer_qi]
    )


def _seed_band_from_guess(xpts, ypts, p_e, f_guess, guess_window, window_pts,
                          n_sigma=3.0, max_shift_spans=4, n_mix=6):
    """
    Seed the band by searching for a curve that starts at the FWM guess.

    Candidates are `f0 + s*(m*u**2 + (1-m)*u)` with u the gain scaled to [0, 1]:
    m = 0 is a line, m = 1 a vertex-at-zero parabola, and in between the
    concave-down linear_quadratic shapes. f0 ranges over `f_guess +/-
    guess_window`, s over total shifts up to `max_shift_spans` scan spans. The
    winner maximises summed P(e) along it, after subtracting each row's median
    so a flat background scores nothing; rows where the curve has left the scan
    window score zero rather than being extrapolated.

    Each row is then refined to its P(e) peak within `window_pts` bins of the
    winning curve, and kept only if that peak stands `n_sigma` robust std devs
    above the background -- so the no-signal rows near zero gain drop out.

    Returns (seed_gains, seed_freqs, seed_curve); seed_curve is NaN outside
    the scan window.
    """
    dx = float(np.median(np.diff(xpts)))
    z = p_e - np.median(p_e, axis=1, keepdims=True)
    z = np.apply_along_axis(lambda r: np.convolve(r, np.ones(3) / 3, "same"), 1, z)
    span = xpts[-1] - xpts[0]
    u = (ypts - ypts.min()) / (ypts.max() - ypts.min())
    rows = np.arange(len(ypts))

    f0s = np.arange(f_guess - guess_window, f_guess + guess_window + 1e-9, dx / 2)
    best_score, best = -np.inf, None
    for m in np.linspace(0, 1, n_mix):
        shape = m * u**2 + (1 - m) * u
        for s in np.arange(-max_shift_spans * span, span + 1e-9, dx):
            if m > 0 and s > 0:
                continue  # concave up: excluded, the band only bends downward
            idx = np.rint((f0s[:, None] + s * shape - xpts[0]) / dx).astype(int)
            inside = (idx >= 0) & (idx < len(xpts))
            vals = z[rows, np.clip(idx, 0, len(xpts) - 1)]
            score = np.where(inside, vals, 0.0).sum(axis=1)
            k = int(np.argmax(score))
            if score[k] > best_score:
                best_score, best = score[k], (f0s[k], s, m)

    f0, s, m = best
    curve = f0 + s * (m * u**2 + (1 - m) * u)
    curve = np.where((curve >= xpts[0]) & (curve <= xpts[-1]), curve, np.nan)

    noise = 1.4826 * np.median(np.abs(z - np.median(z)))
    seed_gains, seed_freqs = [], []
    for i, f in enumerate(curve):
        if np.isnan(f):
            continue
        center = int(np.argmin(np.abs(xpts - f)))
        lo, hi = max(center - window_pts, 0), min(center + window_pts + 1, len(xpts))
        j = lo + int(np.argmax(z[i, lo:hi]))
        if z[i, j] > n_sigma * noise:
            seed_gains.append(ypts[i])
            seed_freqs.append(xpts[j])
    return np.array(seed_gains), np.array(seed_freqs), curve


def _band_designs(gains, fit_type):
    """
    Design vectors for a band model, excluding the constant column.

    Every model is `freq = sum(coeff_i * design_i) + offset`, so the returned
    list fixes the coefficient order that `pump_power_fit` is stored in:

        'quadratic':        [gain**2]                    -> [a, c],          a*g**2 + c
        'linear':           [gain]                       -> [a, c],          a*g + c
        'linear_quadratic': [gain**2, gain]               -> [a, b, c],       a*g**2 + b*g + c
        'cubic':            [gain**3, gain**2, gain]       -> [a, b, c, d],    a*g**3 + b*g**2 + c*g + d
        'quartic':          [gain**4, gain**3, gain**2, gain] -> [a, b, c, d, e],
                            a*g**4 + b*g**3 + c*g**2 + d*g + e
    """
    if fit_type == "quadratic":
        return [gains**2]
    if fit_type == "linear":
        return [gains]
    if fit_type == "linear_quadratic":
        return [gains**2, gains]
    if fit_type == "cubic":
        return [gains**3, gains**2, gains]
    if fit_type == "quartic":
        return [gains**4, gains**3, gains**2, gains]
    raise ValueError(
        f"Unknown band fit type {fit_type!r}; expected one of {BAND_FIT_TYPES}."
    )


def _fit_band_poly(designs, freqs, n_iter, sigma, nonpositive_lead=False):
    """
    Sigma-clipped least-squares fit of freqs against several design vectors.

    Returns (coeffs, keep) with coeffs ordered `[*design coefficients, offset]`
    and `keep` marking the points that survived clipping.

    `nonpositive_lead` constrains the first coefficient to be <= 0. With a
    single inequality constraint on a linear model the solution is either the
    unconstrained one or the one on the boundary, so this just drops that
    column and refits when the unconstrained fit comes out positive -- no
    optimizer needed.
    """
    if nonpositive_lead:
        coeffs, keep = _fit_band_poly(designs, freqs, n_iter, sigma)
        if coeffs[0] <= 0:
            return coeffs, keep
        # On the boundary: leading coefficient pinned to zero.
        rest, keep = _fit_band_poly(designs[1:], freqs, n_iter, sigma)
        return np.concatenate(([0.0], rest)), keep

    matrix = np.column_stack([*designs, np.ones(len(freqs))])
    n_coeff = matrix.shape[1]
    keep = np.ones(len(freqs), dtype=bool)
    coeffs = np.zeros(n_coeff)
    coeffs[-1] = float(np.median(freqs)) if len(freqs) else 0.0
    for _ in range(n_iter):
        # Need at least one point more than parameters, or the fit is exact
        # and the residual scale below collapses to zero.
        if keep.sum() < n_coeff + 1:
            break
        coeffs = np.linalg.lstsq(matrix[keep], freqs[keep], rcond=None)[0]
        resid = freqs - matrix @ coeffs
        scale = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        keep = np.abs(resid) < sigma * max(scale, 1e-9)
    return coeffs, keep


def _fit_band_model(design, freqs, n_iter, sigma):
    """
    Sigma-clipped straight-line fit of freqs against a single design vector.

    Returns (slope, offset, keep) where `keep` marks the points that survived
    clipping. Both two-parameter band models are `freq = slope * design +
    offset`, so this handles the quadratic (design = gain**2) and linear
    (design = gain) cases identically. `f0g1_reset.py` fits every one of its
    power dependences through here.

    Thin wrapper over `_fit_band_poly` so there is one clipping loop rather
    than two; `np.polyfit(x, y, 1)` is least squares against `[x, 1]`, which
    is exactly what that builds for a single design vector.
    """
    coeffs, keep = _fit_band_poly([design], freqs, n_iter, sigma)
    return coeffs[0], coeffs[1], keep


def _cubic_turning_points(a, b, c):
    """
    Real roots of a cubic's derivative, 3*a*x**2 + 2*b*x + c.

    A turning point inside the playable gain range [0, 1] means the cubic
    folds back on itself there, so the gain <-> frequency map is ambiguous.
    Shared between `band_gain_from_fit` (inverting a stored fit) and
    `_track_fwm_band` (deciding whether 'auto' may pick 'cubic' at all).

    Closed form rather than `_real_roots`, since a quadratic's roots don't
    need a general eigenvalue solver.
    """
    if a == 0:
        return []
    disc = (2 * b) ** 2 - 4 * (3 * a) * c
    if disc < 0:
        return []
    return [(-2 * b + s * np.sqrt(disc)) / (6 * a) for s in (1, -1)]


def _real_roots(poly_coeffs, tol=1e-6):
    """
    Real roots of a polynomial, highest-degree coefficient first.

    `np.roots` always returns a complex array; a numerically real root still
    carries a tiny nonzero imaginary part from floating-point error rather
    than an exact zero, so "real" here means within `tol` of the real axis
    (relative to the root's own magnitude), not literally zero imaginary
    part. Used where a closed form would need the general quartic formula
    (quartic's own equation, and the cubic derivative that finds its turning
    points) -- `_cubic_turning_points` keeps the quadratic formula instead,
    since there `np.roots` would be overkill.
    """
    roots = np.roots(poly_coeffs)
    return [
        float(r.real) for r in roots
        if abs(r.imag) < tol * max(abs(r.real), 1.0)
    ]


def band_frequency_from_fit(coeffs, fit_type, gain):
    """
    Evaluate a stored band fit: pump gain -> pump frequency.

    `coeffs` is `device.fwm.pump_power_fit` and `fit_type` the matching
    `pump_power_fit_type`. Scalar or array `gain`.
    """
    gain = np.asarray(gain, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    designs = _band_designs(gain, fit_type)
    if len(coeffs) != len(designs) + 1:
        raise ValueError(
            f"{fit_type!r} needs {len(designs) + 1} coefficients, got "
            f"{len(coeffs)}: {list(coeffs)}. The fit and its type tag are out "
            "of step -- re-run 18_fwm_power.py to refresh both."
        )
    freq = coeffs[-1] + sum(c * d for c, d in zip(coeffs, designs))
    return float(freq) if freq.ndim == 0 else freq


def band_gain_from_fit(coeffs, fit_type, frequency):
    """
    Invert a stored band fit: pump frequency -> pump gain.

    Raises if the frequency is off the fitted band (no real non-negative
    root), since there is no gain that reaches it and silently returning a
    clamped or complex value would put the pump somewhere it was never asked
    to be.

    For 'linear_quadratic', 'cubic', and 'quartic' the polynomial can have
    more than one non-negative root, and the one *nearest zero gain* is
    taken: the offset term anchors the fit at gain 0, so that is the branch
    the measured data is continuously connected to. If the band is not
    monotonic across the physical gain range [0, 1] -- a turning point for
    'linear_quadratic', a turning point of the derivative for 'cubic' and
    'quartic' -- the inverse is genuinely ambiguous there and this raises
    rather than picking a side. A zero leading coefficient is the
    concave-down constraint's boundary (for 'linear_quadratic') where the
    model degenerates to a lower-order one and inverts as such; 'cubic' and
    'quartic' carry no such constraint (see `_track_fwm_band`) but still fall
    back the same way, one order at a time, if their own leading coefficient
    comes out exactly zero.
    """
    coeffs = np.asarray(coeffs, dtype=float)
    n_expected = len(_band_designs(np.zeros(1), fit_type)) + 1
    if len(coeffs) != n_expected:
        raise ValueError(
            f"{fit_type!r} needs {n_expected} coefficients, got {len(coeffs)}: "
            f"{list(coeffs)}. The fit and its type tag are out of step -- "
            "re-run 18_fwm_power.py to refresh both."
        )

    stale = (
        "Re-run 18_fwm_power.py to refresh device.fwm.pump_power_fit."
    )
    if fit_type == "linear":
        a, c = coeffs
        gain = (frequency - c) / a
        if gain < 0:
            raise ValueError(
                f"Pump frequency {frequency:.3f} MHz needs a negative gain on "
                f"the fitted linear FWM band (a={a:.4f}, c={c:.4f}, zero-gain "
                f"band edge {c:.3f} MHz). {stale}"
            )
        return float(gain)

    if fit_type == "quadratic":
        a, c = coeffs
        if a == 0:
            raise ValueError(
                "quadratic band fit has a zero gain**2 coefficient, so the "
                "band it describes is flat and no frequency maps to a gain. "
                f"{stale}"
            )
        arg = (frequency - c) / a
        if arg < 0:
            raise ValueError(
                f"Pump frequency {frequency:.3f} MHz is unreachable on the "
                f"fitted quadratic FWM band (a={a:.4f}, c={c:.4f}, zero-gain "
                f"band edge {c:.3f} MHz). {stale}"
            )
        return float(np.sqrt(arg))

    if fit_type == "cubic":
        a, b, c, d = coeffs
        if a == 0:
            # No constraint pins this boundary for 'cubic' (unlike
            # 'linear_quadratic'), but an exact zero still means the fit is
            # really a lower-order one.
            return band_gain_from_fit([b, c, d], "linear_quadratic", frequency)
        for turn in _cubic_turning_points(a, b, c):
            if 0 < turn < 1:
                raise ValueError(
                    f"cubic band fit has a turning point at gain {turn:.4f}, "
                    "inside the playable range, so a frequency can map to "
                    f"more than one gain (a={a:.4f}, b={b:.4f}, c={c:.4f}, "
                    f"d={d:.4f}). A band that doubles back is almost always "
                    f"a bad fit. {stale}"
                )
        roots = [r for r in _real_roots([a, b, c, d - frequency]) if r >= 0]
        if not roots:
            raise ValueError(
                f"Pump frequency {frequency:.3f} MHz is unreachable on the "
                f"fitted cubic FWM band (a={a:.4f}, b={b:.4f}, c={c:.4f}, "
                f"d={d:.4f}, zero-gain band edge {d:.3f} MHz). {stale}"
            )
        return min(roots)

    if fit_type == "quartic":
        a, b, c, d, e = coeffs
        if a == 0:
            # Same fallback as 'cubic': no constraint pins this boundary, but
            # an exact zero still means the fit is really one order lower.
            return band_gain_from_fit([b, c, d, e], "cubic", frequency)
        # Turning points are where the derivative -- itself a cubic,
        # 4*a*g**3 + 3*b*g**2 + 2*c*g + d -- is zero.
        for turn in _real_roots([4 * a, 3 * b, 2 * c, d]):
            if 0 < turn < 1:
                raise ValueError(
                    f"quartic band fit has a turning point at gain "
                    f"{turn:.4f}, inside the playable range, so a frequency "
                    f"can map to more than one gain (a={a:.4f}, b={b:.4f}, "
                    f"c={c:.4f}, d={d:.4f}, e={e:.4f}). A band that doubles "
                    f"back is almost always a bad fit. {stale}"
                )
        roots = [r for r in _real_roots([a, b, c, d, e - frequency]) if r >= 0]
        if not roots:
            raise ValueError(
                f"Pump frequency {frequency:.3f} MHz is unreachable on the "
                f"fitted quartic FWM band (a={a:.4f}, b={b:.4f}, c={c:.4f}, "
                f"d={d:.4f}, e={e:.4f}, zero-gain band edge {e:.3f} MHz). "
                f"{stale}"
            )
        return min(roots)

    a, b, c = coeffs
    if a == 0:
        # The concave-down constraint's boundary: the fit landed on a = 0, so
        # this *is* the linear model and inverts as one.
        return band_gain_from_fit([b, c], "linear", frequency)
    vertex = -b / (2 * a)
    if 0 < vertex < 1:
        raise ValueError(
            f"linear_quadratic band fit turns over at gain {vertex:.4f}, "
            "inside the playable range, so a frequency can map to two gains "
            f"(a={a:.4f}, b={b:.4f}, c={c:.4f}). A band that doubles back is "
            f"almost always a bad fit. {stale}"
        )
    disc = b**2 - 4 * a * (c - frequency)
    if disc < 0:
        raise ValueError(
            f"Pump frequency {frequency:.3f} MHz is unreachable on the fitted "
            f"linear_quadratic FWM band (a={a:.4f}, b={b:.4f}, c={c:.4f}, "
            f"turning point {a * vertex**2 + b * vertex + c:.3f} MHz). {stale}"
        )
    roots = [(-b + s * np.sqrt(disc)) / (2 * a) for s in (1, -1)]
    roots = [float(r) for r in roots if r >= 0]
    if not roots:
        raise ValueError(
            f"Pump frequency {frequency:.3f} MHz needs a negative gain on the "
            f"fitted linear_quadratic FWM band (a={a:.4f}, b={b:.4f}, "
            f"c={c:.4f}, zero-gain band edge {c:.3f} MHz). {stale}"
        )
    return min(roots)


def _track_fwm_band(
    xpts,
    ypts,
    p_e,
    fit_type="auto",
    min_gain_frac=0.3,
    n_iter=4,
    sigma=3.0,
    window_pts=5,
    f_guess=None,
    guess_window=10.0,
):
    """
    Robustly track the bright FWM band across a pump frequency/gain scan.

    A naive per-row `argmax(p_e)` picks up stray high-P(e) points away from
    the band (noise, other features), especially at low gain where the band
    has little contrast. Instead a polynomial model is fit to the band's
    trajectory using iterative sigma-clipping to reject outliers:

    - 'quadratic':        `freq(gain) = a * gain**2 + c`, a forced-vertex
      parabola (the AC Stark-like frequency pull is roughly quadratic in gain)
    - 'linear':           `freq(gain) = a * gain + c`
    - 'linear_quadratic': `freq(gain) = a * gain**2 + b * gain + c`, the
      parabola with its vertex free
    - 'cubic':            `freq(gain) = a * gain**3 + b * gain**2 + c * gain
      + d`, one further order
    - 'quartic':          `freq(gain) = a * gain**4 + b * gain**3 + c * gain**2
      + d * gain + e`, one more again

    With fit_type='auto' all five are fit and one is chosen by a forward chain
    of F-tests, because they do not all have the same number of free
    parameters and are not all nested in each other:

    1. 'quadratic' vs 'linear', both two-parameter, compared directly by the
       median absolute residual over *all* seed points -- over all seed points
       rather than each fit's own inliers, so a model cannot win by clipping
       away the points it fits badly.
    2. 'linear_quadratic' against that winner by an F-test at 95% on the one
       extra parameter. It contains both two-parameter models as special cases
       (b = 0 and a = 0), so it can *always* reach at least as small a
       residual and a direct comparison would hand it every scan. The F-test
       is the textbook comparison for nested models and is the same rule
       `fit_ef_doublet` uses to accept a doublet over a single Lorentzian.
    3. 'cubic' against 'linear_quadratic' by the same F-test, **only if stage
       2 adopted 'linear_quadratic'**. 'cubic' nests 'linear_quadratic' by
       exactly one parameter (the cubic term), so that is the only baseline
       the test is valid against -- it differs from 'quadratic' or 'linear' by
       *two* terms (also the linear or quadratic one), which would make the
       F-test's degrees of freedom wrong. This also means a scan whose vertex
       term wasn't justified never gets to try cubic in 'auto'; force
       `fit_type='cubic'` to fit it regardless.
    4. 'quartic' against 'cubic' the same way, **only if stage 3 adopted
       'cubic'**, for the same nesting reason one level up. Reaching
       'quartic' in 'auto' therefore needs *three* F-tests to all clear 95%
       on the same noise draw, which is an increasingly strict bar -- and a
       5-parameter fit against the ~30-70 seed points a typical scan has
       little contrast left to resolve with. Treat an 'auto'-chosen 'quartic'
       as worth a second look rather than a settled result; force it
       deliberately if that is really the fit wanted.

    The F-test uses R2 over all seed points for the same reason the median
    residual does.

    Every model with a gain**2 term is fit with that coefficient constrained
    to be **<= 0**: the pump Stark shift grows with power, so the band can
    only bend downward. 'cubic' and 'quartic' carry no such constraint --
    there is no equally simple physical argument for the sign of a higher-
    order term, and enforcing concavity of a cubic or quartic over an
    interval needs real constrained optimization rather than the "drop a
    column and see" trick the quadratic terms use. Forcing a fit_type skips
    the selection but not any constraint the model does have.

    The final P(e) for each gain point is then read out only from a small
    frequency window around the chosen trajectory, so points far from the band
    can never be reported as the band maximum.

    Args:
        xpts: 1D array of pump frequency points (MHz)
        ypts: 1D array of pump gain points
        p_e: 2D array of excited-state population, shape (len(ypts), len(xpts))
        fit_type: 'auto' (fit all, keep the best justified), or one of
            'quadratic', 'linear', 'linear_quadratic', 'cubic', 'quartic' to
            force it
        min_gain_frac: Fraction of the gain range below which rows are excluded
            from the trajectory fit (band has too little contrast there)
        n_iter: Number of sigma-clipping iterations for the trajectory fit
        sigma: Rejection threshold (in robust std devs) for outlier points
        window_pts: Half-width (in frequency bins) of the window around the
            fitted trajectory used to re-extract P(e)
        f_guess: Expected FWM frequency at zero pump gain (MHz). When given,
            the seed points come from `_seed_band_from_guess` -- the best curve
            leaving this frequency -- instead of per-row argmax, and
            `min_gain_frac` is not used.
        guess_window: How far (MHz) the curve's zero-gain end may sit from
            `f_guess`

    Returns:
        dict with 'band_fit_type', 'band_fit_coeffs' (the coefficient list in
        the order `pump_power_fit` stores it), 'band_fit_a' / 'band_fit_b' /
        'band_fit_c' (the first three coefficients positionally -- 'c' is
        always the model's offset; NaN for 'band_fit_b' where the model has
        no second coefficient), 'band_freq', 'band_p_e', 'band_freq_idx', the
        median absolute residual of every candidate as 'band_resid_<name>'
        (MHz), 'band_f_stat' / 'band_f_crit' from stage 2's comparison,
        'band_f_stat_cubic' / 'band_f_crit_cubic' from stage 3's, and
        'band_f_stat_quartic' / 'band_f_crit_quartic' from stage 4's (each
        pair NaN when its stage did not run)
    """
    if fit_type not in ("auto",) + BAND_FIT_TYPES:
        raise ValueError(
            f"fit_type must be 'auto' or one of {BAND_FIT_TYPES}, "
            f"got {fit_type!r}"
        )

    xpts = np.asarray(xpts)
    ypts = np.asarray(ypts)
    p_e = np.asarray(p_e)

    seed_curve = np.full(len(ypts), np.nan)
    if f_guess is not None:
        seed_gains, seed_freqs, seed_curve = _seed_band_from_guess(
            xpts, ypts, p_e, f_guess, guess_window, window_pts
        )
        if len(seed_gains) < 4:
            raise ValueError(
                f"only {len(seed_gains)} rows have a significant P(e) peak "
                f"near the curve from the FWM guess {f_guess:.3f} MHz"
            )
    else:
        # Seed guess: naive per-row argmax, restricted to gains where the band
        # has enough contrast to be reliably located.
        gain_cut = ypts.min() + min_gain_frac * (ypts.max() - ypts.min())
        seed_mask = ypts >= gain_cut
        seed_gains = ypts[seed_mask]
        seed_freqs = xpts[np.argmax(p_e[seed_mask], axis=1)]

    # Total sum of squares for the R2 the F-test below needs.
    tss = float(np.sum((seed_freqs - np.mean(seed_freqs)) ** 2)) if len(
        seed_freqs
    ) else 0.0

    # Fit every candidate model, scoring each on the full seed set.
    fits = {}
    for name in BAND_FIT_TYPES:
        designs = _band_designs(seed_gains, name)
        # The pump Stark shift grows with power, so the band can only bend
        # downward: the gain**2 coefficient is constrained to be <= 0. A scan
        # with no concave-down curvature in it therefore lands on the a = 0
        # boundary, where linear_quadratic degenerates to the linear model.
        coeffs, keep = _fit_band_poly(
            designs, seed_freqs, n_iter, sigma,
            nonpositive_lead=(name in ("quadratic", "linear_quadratic")),
        )
        resid = seed_freqs - (coeffs[-1] + sum(
            co * d for co, d in zip(coeffs, designs)
        ))
        rss = float(np.sum(resid**2))
        fits[name] = {
            "coeffs": [float(v) for v in coeffs],
            "resid": float(np.median(np.abs(resid))) if len(resid) else np.inf,
            "r2": 1.0 - rss / tss if tss > 0 else -np.inf,
            "n_kept": int(keep.sum()),
        }

    f_stat = np.nan
    f_crit = np.nan
    f_stat_cubic = np.nan
    f_crit_cubic = np.nan
    f_stat_quartic = np.nan
    f_crit_quartic = np.nan
    if fit_type == "auto":
        # Stage 1: the two-parameter models, head to head.
        chosen = min(("quadratic", "linear"), key=lambda n: fits[n]["resid"])
        # Stage 2: is the free vertex worth its extra parameter?
        full = fits["linear_quadratic"]
        dof = len(seed_freqs) - 3
        if dof > 0 and full["r2"] < 1 and fits[chosen]["r2"] > -np.inf:
            f_stat = ((full["r2"] - fits[chosen]["r2"]) / 1) / (
                (1 - full["r2"]) / dof
            )
            f_crit = float(f_dist.ppf(0.95, 1, dof))
            # A vertex inside the playable gain range means the band doubles
            # back, so `band_gain_from_fit` cannot invert it and every script
            # that reads the fit would raise. At 95% the F-test accepts a
            # spurious extra term on 1 scan in 20, and a spurious one lands
            # its vertex in range often -- measured on synthetic quadratic
            # data, F = 4.35 against F_crit = 4.24 with the vertex at gain
            # 0.18. Auto must not choose a model the rest of the tree cannot
            # use; forcing 'linear_quadratic' still gives it to you, and then
            # the inversion says why it is unusable.
            vertex_ok = full["coeffs"][0] != 0 and not (
                0 < -full["coeffs"][1] / (2 * full["coeffs"][0]) < 1
            )
            if f_stat > f_crit and vertex_ok:
                chosen = "linear_quadratic"
        # Stage 3: cubic against linear_quadratic, only reached if that
        # vertex term was itself adopted above. Cubic nests linear_quadratic
        # by exactly one parameter (the cubic term); it differs from
        # quadratic or linear by two, which this F-test cannot score.
        if chosen == "linear_quadratic":
            cubic_fit = fits["cubic"]
            dof_cubic = len(seed_freqs) - 4
            if dof_cubic > 0 and cubic_fit["r2"] < 1:
                f_stat_cubic = ((cubic_fit["r2"] - full["r2"]) / 1) / (
                    (1 - cubic_fit["r2"]) / dof_cubic
                )
                f_crit_cubic = float(f_dist.ppf(0.95, 1, dof_cubic))
                # Same reasoning as the vertex gate: 'auto' must not choose a
                # cubic whose own inverse is ambiguous.
                cubic_ok = not any(
                    0 < turn < 1
                    for turn in _cubic_turning_points(*cubic_fit["coeffs"][:3])
                )
                if f_stat_cubic > f_crit_cubic and cubic_ok:
                    chosen = "cubic"
        # Stage 4: quartic against cubic, only reached if that cubic term was
        # itself adopted above -- same nesting argument one order up.
        if chosen == "cubic":
            quartic_fit = fits["quartic"]
            dof_quartic = len(seed_freqs) - 5
            if dof_quartic > 0 and quartic_fit["r2"] < 1:
                f_stat_quartic = (
                    (quartic_fit["r2"] - fits["cubic"]["r2"]) / 1
                ) / ((1 - quartic_fit["r2"]) / dof_quartic)
                f_crit_quartic = float(f_dist.ppf(0.95, 1, dof_quartic))
                # Same reasoning as the cubic gate, one derivative order up:
                # the turning points of a quartic's derivative are the real
                # roots of the cubic 4*a*g**3 + 3*b*g**2 + 2*c*g + d.
                qa, qb, qc, qd = quartic_fit["coeffs"][:4]
                quartic_ok = not any(
                    0 < turn < 1
                    for turn in _real_roots([4 * qa, 3 * qb, 2 * qc, qd])
                )
                if f_stat_quartic > f_crit_quartic and quartic_ok:
                    chosen = "quartic"
    else:
        chosen = fit_type

    coeffs = fits[chosen]["coeffs"]
    a = coeffs[0]
    b = coeffs[1] if len(coeffs) > 2 else np.nan
    c = coeffs[-1]
    band_freq = band_frequency_from_fit(coeffs, chosen, ypts)

    # Re-extract P(e) for every gain point, restricted to a window around
    # the fitted band so stray excursions elsewhere in the row are ignored.
    band_p_e = np.zeros(len(ypts))
    band_freq_idx = np.zeros(len(ypts), dtype=int)
    for i, freq in enumerate(band_freq):
        center = np.argmin(np.abs(xpts - freq))
        lo = max(center - window_pts, 0)
        hi = min(center + window_pts + 1, len(xpts))
        local_idx = lo + np.argmax(p_e[i, lo:hi])
        band_freq_idx[i] = local_idx
        band_p_e[i] = p_e[i, local_idx]

    out = {
        "band_fit_type": chosen,
        "band_fit_coeffs": coeffs,
        "band_fit_a": a,
        "band_fit_b": b,
        "band_fit_c": c,
        "band_freq": band_freq,
        "band_p_e": band_p_e,
        "band_freq_idx": band_freq_idx,
        "band_f_stat": f_stat,
        "band_f_crit": f_crit,
        "band_f_stat_cubic": f_stat_cubic,
        "band_f_crit_cubic": f_crit_cubic,
        "band_f_stat_quartic": f_stat_quartic,
        "band_f_crit_quartic": f_crit_quartic,
        "band_seed_freq": seed_curve,
    }
    out.update({f"band_resid_{name}": fits[name]["resid"] for name in BAND_FIT_TYPES})
    return out


# Kept per pass, as `<key>_with_buffer` / `<key>_no_buffer`, when diff_buffer is on.
DIFF_SCAN_KEYS = ("avgi", "avgq", "amps", "phases", "p_e")


class FourWMPowerExperiment(QickExperiment2DSimple):
    """
    2D pulse probe spectroscopy experiment that sweeps both frequency and power.

    This experiment performs a 2D sweep of both probe frequency and power (gain)
    to map out how the qubit spectrum changes with power. This allows visualization
    of power-dependent effects like AC Stark shifts and multi-photon transitions.

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
        progress=True,
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
        prefix = f"qubit_spectroscopy_power_{ef}{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, live_plot=live_plot)
        self.nested_progress = False
        self.progress_desc = "FWM gain sweep"

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
            "reps": 2 * self.reps,
            "rng": 50,  # Range for logarithmic gain sweep
            "max_gain": self.cfg.device.qubit.max_gain,  # Maximum gain value
            "expts_gain": 10,  # Number of gain points
            "log": True,  # Use logarithmic gain spacing
            # Band trajectory model: 'auto' fits all five and keeps the best
            # justified one, or force 'quadratic' / 'linear' /
            # 'linear_quadratic' / 'cubic' / 'quartic'.
            "pump_power_fit_type": "auto",
            # If True, acquire with and without buffer and fit the difference.
            # do_buffer is managed internally when this is set.
            "diff_buffer": False,
            # Zero-pump-gain band frequency the tracker's seed curve starts
            # from, and how far (MHz) the band's start may sit from it.
            "fwm_guess": expected_fwm_frequency(self.cfg, qi),
            "fwm_guess_window": 10.0,
        }

        # Merge default parameters
        params_def = {**params_def, **params_def2}

        # Merge with user-provided parameters
        params = {**params_def, **params}

        # Create a QubitSpec experiment but don't run it
        exp_name = QubitSpecShot
        self.expt = exp_name(cfg_dict, qi=qi, go=False, params=params, check_params=False)

        # Get parameters from the QubitSpecShot experiment
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
        # Generate gain points for the sweep
        if "log" in self.cfg.expt and self.cfg.expt.log == True:
            # Use logarithmic gain spacing for better dynamic range
            rng = self.cfg.expt.rng
            rat = rng ** (-1 / (self.cfg.expt["expts_gain"] - 1))

            max_gain = self.cfg.expt["max_gain"]
            gainpts = max_gain * rat ** (np.arange(self.cfg.expt["expts_gain"]))
        else:
            # Use linear gain spacing
            gainpts = self.cfg.expt["start_gain"] + self.cfg.expt[
                "step_gain"
            ] * np.arange(self.cfg.expt["expts_gain"])

        # Set up the y-sweep (gain sweep)
        ysweep = [{"pts": gainpts, "var": "gain_p"}]

        # Configure experiment parameters
        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}

        # Set up frequency sweep
        # self.cfg.expt.frequency = QickSweep1D(
        #     "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        # )

        # Acquire data
        self.xlabel = 'pump frequency (MHz)'
        self.ylabel = 'pump gain (DAC level)'

        if self.cfg.expt.diff_buffer:
            # First scan: with buffer
            print("diff_buffer: acquiring with buffer on...")
            self.cfg.expt.do_buffer = True
            self.expt.cfg.expt.do_buffer = True
            super().acquire(ysweep, progress=progress)
            # p_e is not yet computed — shots get thresholded in analyze(). Do
            # it here so we have it before the second pass overwrites self.data.
            self.add_p_e(self.data, stacked=True)
            with_buffer = {k: np.array(self.data[k]) for k in DIFF_SCAN_KEYS}

            # Second scan: without buffer
            print("diff_buffer: acquiring with buffer off...")
            self.cfg.expt.do_buffer = False
            self.expt.cfg.expt.do_buffer = False
            super().acquire(ysweep, progress=progress)
            self.add_p_e(self.data, stacked=True)

            for k in DIFF_SCAN_KEYS:
                self.data[f'{k}_with_buffer'] = with_buffer[k]
                self.data[f'{k}_no_buffer'] = np.array(self.data[k])
            self.data['p_e'] = self.data['p_e_with_buffer'] - self.data['p_e_no_buffer']
            # Remove shots so analyze() doesn't recompute p_e from the
            # second pass alone and overwrite the difference.
            self.data.pop('shots', None)
        else:
            super().acquire(ysweep, progress=progress)

        return self.data

    @classmethod
    def from_file(cls, h5_path, config_file, metadata_path, suffix="_reanalyzed"):
        """
        Rebuild the experiment from a saved h5 for offline re-analysis.

        Skips __init__, which needs a live soccfg. The stored config supplies
        cfg.expt; `config_file` is where update() writes. Figures go to
        `metadata_path` under the original name plus `suffix`, so the
        original PNG/yml are never overwritten.
        """
        expt = cls.__new__(cls)
        with SlabFile(str(h5_path), "r") as f:
            expt.cfg = AttrDict(f.load_config())
            expt.data = expt.load_data(f)
        expt.data.pop("attrs", None)
        # Files saved before these params existed.
        expt.cfg.expt.setdefault(
            "fwm_guess", expected_fwm_frequency(expt.cfg, expt.cfg.expt.qubit[0])
        )
        expt.cfg.expt.setdefault("fwm_guess_window", 10.0)
        h5_path = Path(h5_path)
        expt.fname = str(h5_path.with_name(h5_path.stem + suffix + h5_path.suffix))
        expt.path = str(h5_path.parent)
        expt.metadata_path = str(metadata_path)
        expt.config_file = str(config_file)
        expt._display_mode = None
        expt.xlabel = 'pump frequency (MHz)'
        expt.ylabel = 'pump gain (DAC level)'
        return expt

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
        if data is None:
            data = self.data

        if fit:
            # Fit each frequency slice to a Lorentzian model
            super().analyze(fitterfunc=FITTER_FUNC, fitfunc=FIT_FUNC)

        if 'shots' in data:
            self.add_p_e(data, stacked=True)

            data.pop("shots")  # Remove raw shots from data to save space

        if 'p_e' in data:
            # Track the bright FWM band robustly so stray excursions away
            # from the band aren't reported as the per-gain maximum.
            try:
                data.update(
                    _track_fwm_band(
                        data['xpts'],
                        data['ypts'],
                        data['p_e'],
                        fit_type=self.cfg.expt.pump_power_fit_type,
                        f_guess=self.cfg.expt.fwm_guess,
                        guess_window=self.cfg.expt.fwm_guess_window,
                    )
                )
                print(f"FWM band seeded from {self.cfg.expt.fwm_guess:.3f} MHz "
                      f"at zero gain; seed curve starts at "
                      f"{data['band_seed_freq'][np.argmin(data['ypts'])]:.3f} MHz")
                resids = ", ".join(
                    f"{name} {data[f'band_resid_{name}']:.4f}"
                    for name in BAND_FIT_TYPES
                )
                print(f"FWM band fit: {data['band_fit_type']} "
                      f"(median |residual| in MHz: {resids})")
                if np.isfinite(data["band_f_stat"]):
                    # Whether the free vertex earned its extra parameter.
                    # 'cubic'/'quartic' imply it did -- each later stage only
                    # runs on top of the previous one's adoption -- so they
                    # count as kept too.
                    verdict = (
                        "kept"
                        if data["band_fit_type"]
                        in ("linear_quadratic", "cubic", "quartic")
                        else "rejected"
                    )
                    print(
                        f"  linear_quadratic {verdict}: F = "
                        f"{data['band_f_stat']:.2f} against F_crit(95%) = "
                        f"{data['band_f_crit']:.2f}"
                    )
                if np.isfinite(data["band_f_stat_cubic"]):
                    verdict = (
                        "kept" if data["band_fit_type"] in ("cubic", "quartic")
                        else "rejected"
                    )
                    print(
                        f"  cubic {verdict}: F = "
                        f"{data['band_f_stat_cubic']:.2f} against F_crit(95%) = "
                        f"{data['band_f_crit_cubic']:.2f}"
                    )
                if np.isfinite(data["band_f_stat_quartic"]):
                    verdict = (
                        "kept" if data["band_fit_type"] == "quartic" else "rejected"
                    )
                    print(
                        f"  quartic {verdict}: F = "
                        f"{data['band_f_stat_quartic']:.2f} against F_crit(95%) = "
                        f"{data['band_f_crit_quartic']:.2f}"
                    )
            except Exception as e:
                print(f"FWM band tracking failed: {e}")

        return data

    def update(self, verbose=True):
        """
        Write the fitted FWM band model to device.fwm.

        `pump_power_fit` holds the coefficients and `pump_power_fit_type`
        records which model they belong to, so consumers can invert the right
        form:

            'quadratic'        [a, c]           a*gain**2 + c
            'linear'           [a, c]           a*gain + c
            'linear_quadratic' [a, b, c]        a*gain**2 + b*gain + c
            'cubic'            [a, b, c, d]     a*gain**3 + b*gain**2 + c*gain + d
            'quartic'          [a, b, c, d, e]  a*gain**4 + b*gain**3 + c*gain**2 + d*gain + e

        The list length therefore varies with the model, which is why the
        pair must always be read together -- `band_frequency_from_fit` /
        `band_gain_from_fit` raise if they are out of step.
        """
        if "band_fit_coeffs" not in self.data:
            print("No FWM band fit available; run analyze() first")
            return
        config.update_fwm(
            self.config_file,
            "pump_power_fit",
            list(self.data["band_fit_coeffs"]),
            verbose=verbose,
        )
        config.update_fwm(
            self.config_file,
            "pump_power_fit_type",
            self.data["band_fit_type"],
            verbose=verbose,
        )

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):
        """
        Display the results of the 2D pulse probe spectroscopy experiment.

        Creates 2D color plots showing amplitude, phase, and (if available)
        the excited-state population P(e) as a function of frequency and power.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_amps: Whether to plot amplitude data (vs. phase)
            ax: Matplotlib axis to plot on
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data
        if self.cfg.expt.diff_buffer and ax is None:
            self._display_diff(data)
            return

        # Set up plot title
        title = f"buffer frequency = {self.cfg.expt.frequency_b} MHz, buffer gain = {self.cfg.expt.gain_b}"
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

    def _display_diff(self, data):
        """Buffer-on and buffer-off amp/phase/P(e) figures, then the P(e) difference alone."""
        for scan in ("with_buffer", "no_buffer"):
            if not all(f"{k}_{scan}" in data for k in ("amps", "phases", "p_e")):
                print(f"No {scan} amplitude/phase in this file (saved before "
                      "they were kept); skipping that figure")
                continue
            scan_data = {"xpts": data["xpts"], "ypts": data["ypts"]}
            scan_data.update({k: data[f"{k}_{scan}"] for k in ("amps", "phases", "p_e")})
            fig = plt.figure(figsize=(11, 6))
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 2])
            axes = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]),
                    fig.add_subplot(gs[:, 1]))
            self.display_pe(data=scan_data, ax=axes)
            self.save_fig(fig, suffix=f"_{scan}")

        xpts, ypts = data["xpts"], data["ypts"]
        fig, ax = plt.subplots(figsize=(6, 6))
        pcm = ax.pcolormesh(xpts, ypts, data["p_e"], cmap="viridis",
                            shading="auto", rasterized=True)
        plt.colorbar(pcm, ax=ax, label="P(e), buffer on - off")
        if "band_freq" in data:
            ax.plot(data["band_freq"], ypts, color="red", lw=1,
                    label=f"{data['band_fit_type']} band fit")
            ax.legend(loc="upper right", fontsize="small")
        ax.set_xlim(np.min(xpts), np.max(xpts))
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.cfg.expt.log:
            ax.set_yscale("log")
        self.save_fig(fig)
        self.save_config()



QubitSpecShotPower = FourWMPowerExperiment


class QubitSpecShotBuffer(QickExperiment2DSimple):
    """
    2D pulse probe spectroscopy experiment that sweeps both frequency and power.

    This experiment performs a 2D sweep of both probe frequency and power (gain)
    to map out how the qubit spectrum changes with power. This allows visualization
    of power-dependent effects like AC Stark shifts and multi-photon transitions.

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
        prefix = f"qubit_spectroscopy_buffer_{ef}{style}_qubit{qi}"
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
            "center_freq_b": self.cfg.device.readout.frequency[0],  # Center frequency for buffer pulse
            "span_b": self.cfg.device.readout.kappa[0]*5,  # frequency span
            "expts_b": 50,  # Number of frequency points
        }

        # Merge default parameters
        params_def = {**params_def, **params_def2}

        # Merge with user-provided parameters
        params = {**params_def, **params}

        # Create a QubitSpec experiment but don't run it
        exp_name = QubitSpecShot
        self.expt = exp_name(cfg_dict, qi=qi, go=False, params=params, check_params=False)

        # Get parameters from the QubitSpecShot experiment
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
        # Generate gain points for the sweep
        
        # Set up the y-sweep (buffer sweep)
        freq_b_start = self.cfg.expt.center_freq_b - self.cfg.expt.span_b / 2
        freq_b_end = self.cfg.expt.center_freq_b + self.cfg.expt.span_b / 2
        freq_b_pts = np.linspace(freq_b_start, freq_b_end, self.cfg.expt.expts_b)
        ysweep = [{"pts": freq_b_pts, "var": "frequency_b"}]

        # Configure experiment parameters
        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {"label": "buffer_pulse", "param": "freq", "param_type": "pulse"}

        

        # Set up frequency sweep
        # self.cfg.expt.frequency_b = QickSweep1D(
        #     "freq_b_loop", self.cfg.expt.center_freq_b - self.cfg.expt.span_b / 2, self.cfg.expt.center_freq_b + self.cfg.expt.span_b / 2, self.cfg.expt.expts_b
        # )

        # Acquire data
        self.xlabel = 'pump frequency (MHz)'
        self.ylabel = 'buffer frequency (MHz)'
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
        if data is None:
            data = self.data

        if fit:
            # Fit each frequency slice to a Lorentzian model
            super().analyze(fitterfunc=FITTER_FUNC, fitfunc=FIT_FUNC)

        if 'shots' in data:
            self.add_p_e(data, stacked=True)

            data.pop("shots")  # Remove raw shots from data to save space


        return self.data

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):
        """
        Display the results of the 2D pulse probe spectroscopy experiment.

        Creates 2D color plots showing amplitude, phase, and (if available)
        the excited-state population P(e) as a function of pump frequency and
        buffer frequency.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_amps: Whether to plot amplitude data (vs. phase)
            ax: Matplotlib axis to plot on
            **kwargs: Additional arguments for the display
        """
        # Set up plot title
        title = f"pump gain = {self.cfg.expt.gain_p}, buffer gain = {self.cfg.expt.gain_b}"
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

