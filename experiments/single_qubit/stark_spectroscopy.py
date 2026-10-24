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

import builtins

import numpy as np
import matplotlib.pyplot as plt
from qick.asm_v2 import QickSweep1D
from qick import *
from scipy.optimize import curve_fit

from ...helpers import config
from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment2DSweep
from ..general.qick_program import QickProgram, mixer_freq_for, reset_mode_of, reset_plays_pi_pulses, pulse_type_from_config
from .t2 import buffer_dephasing_per_photon


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
        stark_gen_args = {"ch": cfg.expt.stark_chan}
        if self.soccfg["gens"][cfg.expt.stark_chan].get("has_mixer", False):
            stark_gen_args["mixer_freq"] = mixer_freq_for(cfg.expt.stark_freq)
        self.declare_gen(**stark_gen_args)

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
            "chan": cfg.expt.stark_chan,
            "freq": cfg.expt.stark_freq,
            "gain": cfg.expt.stark_gain,
            "type": cfg.expt.stark_pulse_type,
            "sigma": cfg.expt.stark_length,
            "phase": 0,
        }
        super().make_pulse(stark_pulse, "stark_pulse")

        # pi_ge is what QickProgram.reset() plays, so it has to be declared
        # whenever a measurement-based reset is on.
        if reset_plays_pi_pulses(cfg.expt):
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
        # reset_f plays pi_ef ahead of each pi_ge, so it needs the pulse too.
        if cfg.expt.reset_f:
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

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
        
        self.pulse(ch=cfg.expt.stark_chan, name="stark_pulse", t=0)
        # End the probe where the Stark tone's full-amplitude part ends. A
        # flat_top's plateau starts after half its ramp (make_pulse's default
        # 0.02 us x 3, so 0.03 us).
        half_ramp = 0.03 if cfg.expt.stark_pulse_type == "flat_top" else 0
        self.delay(t=half_ramp + cfg.expt.stark_length - cfg.expt.length)
        # Apply qubit pulses 
        self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
        

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
    - 'pulse_type': 'const' (qubit probe)
    - 'stark_pulse_type': 'flat_top' (Stark tone)
    - 'reset_mode': "passive"
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
            "final_delay": 20,
            "length": 5,
            "stark_chan": self.cfg.hw.soc.dacs.readout.ch[qi],
            "stark_qubit": qi,
            "stark_length":15,
            "stark_expts": 30,
            "stark_freq": None,
            "df_stark": 0,
            "max_stark_gain": 1,
            "min_stark_gain": 0,
            "df": 0,
            "stark_rng": 15,
            "pulse_type": "const",  # qubit probe
            # Stark tone on the buffer line. flat_top like the FWM buffer, so
            # the nstark it yields is in the same gain units (a flat_top
            # plateau is half const's amplitude at equal gain).
            "stark_pulse_type": pulse_type_from_config(self.cfg.device.get("fwm"), "buffer_pulse_type"),
            "qubit": [qi],
            # reset_mode: "passive" | "active" | "f0g1". For "active",
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
            "reset_f": False,
            # "while excited" semantics at a fixed trip count: once a round
            # reads ground, a latch suppresses every later pi pulse. Measures
            # every round, so the trigger count stays fixed.
            "reset_latch": False,
            # Mean rounds the while loop takes, for the cycle-time estimate
            # only; `reset` is its hard cap. See CLAUDE.md.
            "reset_while_typical": 1,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            # 'stark_shift' fits one Lorentzian per row and extracts the
            # quadratic Stark pull (photons per unit gain). 'two_peak' instead
            # fits the first two photon-number peaks and reports their
            # amplitudes versus buffer gain.
            "fit_type": "stark_shift",
            # Expected separation of those two peaks: the buffer line's
            # dispersive shift, i.e. the first element of device.readout.chi.
            "chi_buffer": self.cfg.device.readout.chi[0],
        }
        params_def = {**params_def2, **params_def}

        # Merge default and user-provided parameters
        params = {**params_def, **params}

        # Calculate start frequency
        params_def["start"] = (
            self.cfg.device.qubit.f_ge[qi] - params["span"] / 2 + params["df"]
        )
        params = {**params_def, **params}

        # The Stark frequency normally follows the readout resonator associated
        # with the Stark DAC. An explicit stark_freq supports shared-line SMPD wiring.
        if params["stark_freq"] is None:
            params["stark_freq"] = (
                self.cfg.device.readout.frequency[params["stark_qubit"]]
                + params["df_stark"]
            )

        # StarkSpec touches three independent channels -- print them so it's
        # never ambiguous which physical line does what, especially since
        # readout channel/frequency come from `qi`, not from stark_chan/stark_qubit.
        builtins.print(
            f"StarkSpec channels: readout ch {self.cfg.hw.soc.dacs.readout.ch[qi]} "
            f"(qi={qi}, {self.cfg.device.readout.frequency[qi]:.3f} MHz), "
            f"qubit drive ch {self.cfg.hw.soc.dacs.qubit.ch[qi]}, "
            f"stark ch {params['stark_chan']} "
            f"(stark_qubit={params['stark_qubit']}, {params['stark_freq']:.3f} MHz)"
        )

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

        # shots=True so analyze() can threshold them into P(e), the quantity
        # both fit modes work on.
        super().acquire(
            QubitStarkSpecProgram, progress=progress, get_hist=False, shots=True
        )

        self.data["stark_gain"] = np.linspace(
            self.cfg.expt.min_stark_gain,
            self.cfg.expt.max_stark_gain,
            self.cfg.expt.stark_expts,
        )
        self.data["ypts"] = self.data["stark_gain"]
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data to extract qubit parameters.

        Two techniques, selected by `cfg.expt.fit_type`:

        - 'stark_shift' (default): one Lorentzian per buffer gain, then a
          quadratic fit of the pulled frequency to get photons per unit gain.
        - 'two_peak': the first two photon-number peaks, separated by roughly
          `chi_buffer`, giving their amplitudes versus buffer gain.

        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data to a Lorentzian model
            **kwargs: Additional arguments for the fit

        Returns:
            Analyzed data with fit parameters
        """
        # P(e) by thresholding single shots -- the same protocol every other
        # experiment uses (QickExperiment.add_p_e). StarkSpec runs both sweeps
        # in one program, so the shots are not stacked per y step.
        self.add_p_e(self.data)
        if "p_e" not in self.data:
            raise ValueError(
                "No shots in the data, so P(e) cannot be computed. Re-acquire "
                "with shots=True (StarkSpec.acquire does this by default)."
            )
        self.data.pop("shots", None)  # Remove raw shots to save space (absent when reloaded from h5)

        if self.cfg.expt.fit_type == "two_peak":
            return self._analyze_two_peak()
        if self.cfg.expt.fit_type != "stark_shift":
            raise ValueError(
                f"Unknown fit_type {self.cfg.expt.fit_type!r}; "
                "expected 'stark_shift' or 'two_peak'."
            )

        # Fit the data to a Lorentzian model
        self.fitterfunc = fitter.fitlor
        self.fitfunc = fitter.lorfunc
        ydata_lab = ["p_e"]  # Fit the excited-state population

        # Fit each row (y value) separately
        for i, ydata in enumerate(ydata_lab):
            self.data["fit_" + ydata] = []
            self.data["fit_err_" + ydata] = []

            # Iterate through each y value
            for j in range(len(self.data["ypts"])):
                # Fit this row to the model function
                fit_pars, fit_err, init = self.fitterfunc(
                    self.data["xpts"], self.data[ydata][j], fitparams=None
                )
                # Store fit parameters and errors
                self.data["fit_" + ydata].append(fit_pars)
                self.data["fit_err_" + ydata].append(fit_err)   
        # fit frequency of resonance moving
        f = [self.data["fit_p_e"][i][2] for i in range(len(self.data["fit_p_e"]))]
        gamma = [self.data["fit_p_e"][i][3] for i in range(len(self.data["fit_p_e"]))]
        self.data['f'] = f
        self.data['gamma'] = gamma
        # Remove entries where f is NaN from f and corresponding stark_gain (and ypts)
        f_arr = np.array(self.data["f"], dtype=float)
        stark_arr = np.array(self.data["stark_gain"], dtype=float)

        mask = ~np.isnan(f_arr)
        if not np.all(mask):
            f_arr = f_arr[mask]
            stark_arr = stark_arr[mask]

        # Fit the data
        try:
            popt, pcov = curve_fit(quadratic, self.data["stark_gain"], f)
            self.data["popt"] = popt
            # Delta = (
            #     self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            #     - self.cfg.device.readout.frequency[self.cfg.expt.qubit[0]]
            # )
            # Buffer (readout line 0) driven off resonance: the line shift per photon at f_g
            # follows the Ramsey physics; CLAUDE.md "Photon number from Ramsey dephasing (scripts 38, 41)"
            readout = self.cfg.device.readout
            detuning = self.cfg.expt.stark_freq - readout.frequency[0]
            if abs(detuning) < 2 * readout.kappa[0]:
                print(f"Warning: Stark tone {detuning:+.3f} MHz from f_g; near resonance the "
                      "line splits (PNS) instead of shifting, so n below is not meaningful")
            _, shift_per_photon = buffer_dephasing_per_photon(
                readout.kappa[0], self.cfg.expt.chi_buffer, detuning)
            n = 2 * np.pi * abs(popt[0]) / abs(shift_per_photon)
            self.data["n"] = n
            print(f"Line shift {1e3 * popt[0]:.0f} kHz per gain^2 at f_g{detuning:+.3f} MHz -> "
                  f"{n:.4g} on-resonance photons per gain^2")
            self.data['df_readout']=self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]] - quadratic(self.cfg.device.readout.gain[self.cfg.expt.qubit[0]], *self.data["popt"])
            self.data['n_readout'] = n * self.cfg.device.readout.gain[self.cfg.expt.qubit[0]]**2
        
        except:
             print('Fit failed')

        # Store the fitted qubit frequency
        #        data["new_freq"] = data["best_fit"][2]

        return self.data

    def _analyze_two_peak(self):
        """
        Fit the first two photon-number peaks at every buffer gain.

        Adds to self.data: 'peak_amp0' / 'peak_amp1' (the two peak
        amplitudes), 'peak_freq' (n=0 centre), 'peak_delta' (fitted
        separation), 'peak_gamma' (shared linewidth) and 'peak_y0'
        (baseline), each one entry per buffer gain with NaN where the fit
        failed. Together they are the full `two_lorentzians` parameter set, so
        the fit can be redrawn for any row.
        """
        # The 2D display still needs these for the per-row fit overlay.
        self.fitterfunc = fitter.fitlor
        self.fitfunc = fitter.lorfunc

        chi_buffer = self.cfg.expt.chi_buffer
        n_rows = len(self.data["ypts"])
        y0 = np.full(n_rows, np.nan)
        amp0 = np.full(n_rows, np.nan)
        amp1 = np.full(n_rows, np.nan)
        freq = np.full(n_rows, np.nan)
        delta = np.full(n_rows, np.nan)
        gamma = np.full(n_rows, np.nan)

        for j in range(n_rows):
            popt, _ = fit_two_lorentzians(
                self.data["xpts"], self.data["p_e"][j], chi_buffer
            )
            if popt is None:
                continue
            y0[j], amp0[j], amp1[j], freq[j], delta[j], gamma[j] = popt

        self.data["peak_y0"] = y0
        self.data["peak_amp0"] = amp0
        self.data["peak_amp1"] = amp1
        self.data["peak_freq"] = freq
        self.data["peak_delta"] = delta
        self.data["peak_gamma"] = gamma

        n_ok = int(np.isfinite(amp0).sum())
        print(
            f"Two-peak fit: {n_ok}/{n_rows} buffer gains fit "
            f"(expected separation chi_buffer = {chi_buffer:.4f} MHz)"
        )
        if n_ok:
            print(
                f"  fitted separation {np.nanmean(delta):.4f} +/- "
                f"{np.nanstd(delta):.4f} MHz, "
                f"linewidth {np.nanmean(gamma):.4f} MHz"
            )

        return self.data

    def update(self, verbose=True):
        """Write nstark: on-resonance (f_g) buffer photons per gain^2 from the line shift."""
        if "n" not in self.data:
            print(
                "No photons-per-gain fit available; update() needs "
                "fit_type='stark_shift'"
            )
            return
        qi = self.cfg.expt.qubit[0]
        cfg_file = self.config_file
        config.update_readout(cfg_file, "nstark", self.data["n"], qi, verbose=verbose)

    def display(self, data=None, fit=True, ax=None, plot_all=True, **kwargs):
        """
        Display the results of the Stark spectroscopy experiment.

        Everything lands on one figure: the 2D P(e) map spans the left column,
        and the analysis panels stack down the right. Which panels appear
        depends on `cfg.expt.fit_type` -- peak amplitudes / separation /
        example cuts for 'two_peak', Stark pull and linewidth for
        'stark_shift'.

        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            ax: Unused; kept for interface compatibility
            plot_all: Unused; kept for interface compatibility
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data

        two_peak = self.cfg.expt.fit_type == "two_peak"
        n_right = 3 if two_peak else 2

        fig = plt.figure(figsize=(14, 4.0 + 2.4 * n_right))
        gs = fig.add_gridspec(n_right, 2, width_ratios=[1.15, 1], hspace=0.35,
                              wspace=0.28)
        ax_2d = fig.add_subplot(gs[:, 0])
        ax_right = [fig.add_subplot(gs[i, 1]) for i in range(n_right)]

        # Left: the P(e) map that both fit modes are fit to.
        x_sweep = np.asarray(data["xpts"])
        if np.ndim(x_sweep) > 1:
            x_sweep = x_sweep[0]
        pcm = ax_2d.pcolormesh(
            x_sweep, data["ypts"], data["p_e"], cmap="viridis",
            shading="auto", rasterized=True,
        )
        plt.colorbar(pcm, ax=ax_2d, label="P(e)")
        ax_2d.set_xlabel("Qubit Frequency (MHz)")
        ax_2d.set_ylabel("Buffer Gain")

        if two_peak:
            self.display_peak_amplitudes(data=data, ax=ax_right)
            # Trace the fitted n=0 and n=1 peak positions over the map.
            if "peak_freq" in data:
                ax_2d.plot(data["peak_freq"], data["ypts"], color="red", lw=1,
                           label="n = 0")
                ax_2d.plot(data["peak_freq"] + data["peak_delta"], data["ypts"],
                           color="orange", lw=1, label="n = 1")
                ax_2d.legend(loc="upper left", fontsize="small")
        else:
            self._display_stark_shift(data=data, ax=ax_right)
            if "f" in data:
                ax_2d.plot(data["f"], data["ypts"], color="red", lw=1,
                           label="Lorentzian centre")
                ax_2d.legend(loc="upper left", fontsize="small")

        self.save_fig(fig)
        self.save_config()

    def _display_stark_shift(self, data=None, ax=None):
        """
        Stark-pull panels: fitted centre frequency and linewidth vs buffer gain.

        Args:
            data: Data to display (if None, use self.data)
            ax: Sequence of two axes to plot into
        """
        if data is None:
            data = self.data
        ax_f, ax_gamma = ax

        qi = self.cfg.expt.qubit[0]
        gains = data["stark_gain"]

        ax_f.plot(gains, data["f"], "o", ms=4)
        ax_f.axhline(
            self.cfg.device.qubit.f_ge[qi] + self.cfg.device.readout.chi[qi] / 2,
            color="k", lw=1,
        )
        ax_f.axhline(self.cfg.device.qubit.f_ge[qi], color="k", lw=1)
        try:
            x_fit = np.linspace(0, max(gains), 100)
            ax_f.plot(x_fit, quadratic(x_fit, *data["popt"]), label="Quadratic Fit")
        except Exception:
            print("No fit to plot")
        ax_f.axvline(
            self.cfg.device.readout.gain[qi], color="k", linestyle="--",
            label="Readout gain",
        )
        ax_f.set_ylabel("Frequency (MHz)")
        ax_f.legend(fontsize="small")
        ax_f.tick_params(labelbottom=False)

        ax_gamma.plot(gains, data["gamma"], "o", ms=4)
        ax_gamma.set_xlabel("Buffer Gain")
        ax_gamma.set_ylabel("Gamma (MHz)")

    def display_peak_amplitudes(self, data=None, ax=None):
        """
        Plot the two photon-number peak amplitudes against buffer gain.

        Three panels: the peak amplitudes, the fitted separation against the
        expected chi_buffer (so it is obvious whether the two-peak model
        tracked the photon-number splitting or drifted onto noise), and the
        fitted spectra at the first, middle and last buffer gain that fit.

        Args:
            data: Data to display (if None, use self.data)
            ax: Sequence of three axes to plot into, or None to create a
                standalone figure
        """
        if data is None:
            data = self.data

        save_fig = ax is None
        if save_fig:
            # The first two panels share buffer gain on x; the third is a
            # frequency cut, so it gets its own x axis.
            fig = plt.figure(figsize=(8, 11))
            gs = fig.add_gridspec(3, 1, hspace=0.35)
            ax_amp = fig.add_subplot(gs[0])
            ax_sep = fig.add_subplot(gs[1], sharex=ax_amp)
            ax_cut = fig.add_subplot(gs[2])
            ax = [ax_amp, ax_sep, ax_cut]
        else:
            ax_amp, ax_sep, ax_cut = ax
        ax_amp.tick_params(labelbottom=False)

        # Panels below read from `data` so this works on a reloaded dict too.
        gains = data["stark_gain"]

        ax_amp.plot(gains, data["peak_amp0"], "o-", ms=4, lw=1, label="n = 0 peak")
        ax_amp.plot(gains, data["peak_amp1"], "s-", ms=4, lw=1, label="n = 1 peak")
        ax_amp.set_ylabel("Peak amplitude in P(e)")

        chi_buffer = self.cfg.expt.chi_buffer
        ax_sep.plot(gains, data["peak_delta"], "o", ms=4)
        ax_sep.axhline(
            chi_buffer,
            color="red",
            ls="--",
            lw=1,
            label=f"chi_buffer = {chi_buffer:.4f} MHz",
        )
        ax_sep.set_xlabel("Buffer Gain")
        ax_sep.set_ylabel("Fitted separation (MHz)")
        ax_sep.legend(fontsize="small")

        # Example spectra: first, middle and last buffer gain whose fit
        # converged, so a failed row never gets picked as an example.
        ok = np.flatnonzero(np.isfinite(data["peak_amp0"]))
        picks = sorted({ok[0], ok[len(ok) // 2], ok[-1]}) if len(ok) else []
        xpts = np.asarray(data["xpts"])
        x_dense = np.linspace(xpts.min(), xpts.max(), 400)

        for j in picks:
            (line,) = ax_cut.plot(
                xpts,
                data["p_e"][j],
                "o",
                ms=3,
                label=f"gain = {gains[j]:.4g}",
            )
            ax_cut.plot(
                x_dense,
                two_lorentzians(
                    x_dense,
                    data["peak_y0"][j],
                    data["peak_amp0"][j],
                    data["peak_amp1"][j],
                    data["peak_freq"][j],
                    data["peak_delta"][j],
                    data["peak_gamma"][j],
                ),
                "-",
                lw=1,
                color=line.get_color(),
            )
            # Mark which gains the example cuts came from.
            ax_amp.axvline(gains[j], color=line.get_color(), ls=":", lw=1, alpha=0.6)

        ax_cut.set_xlabel("Qubit Frequency (MHz)")
        ax_cut.set_ylabel("P(e)")
        ax_cut.legend(fontsize="small")

        ax_amp.legend(fontsize="small")

        if save_fig:
            self.save_fig(fig, suffix="_peak_amplitudes")


def quadratic(x, a, c):
    """Quadratic with no linear term: a * x^2 + c."""
    return a * x**2 + c


def two_lorentzians(x, y0, a0, a1, x0, delta, gamma):
    """
    Two Lorentzians of shared width, separated by `delta`.

    Models the first two photon-number peaks of a dispersively coupled qubit:
    the n=0 peak sits at `x0` and the n=1 peak at `x0 + delta`, where delta is
    the buffer's dispersive shift. The two share a linewidth `gamma`, which is
    the more robust assumption at low SNR -- one qubit linewidth sets both.

    Args:
        x: Qubit probe frequency (MHz)
        y0: Baseline offset
        a0: Amplitude of the n=0 peak
        a1: Amplitude of the n=1 peak
        x0: Centre of the n=0 peak (MHz)
        delta: Peak separation (MHz), nominally chi_buffer
        gamma: Shared half-width at half-maximum (MHz)
    """
    return (
        y0
        + a0 / (1 + (x - x0) ** 2 / gamma**2)
        + a1 / (1 + (x - x0 - delta) ** 2 / gamma**2)
    )


def fit_two_lorentzians(xdata, ydata, delta_guess, gamma_guess=None):
    """
    Fit two photon-number peaks and return their amplitudes.

    `delta` is seeded at `delta_guess` (chi_buffer) and bounded to a factor of
    a few around it, so the fit can track the real separation without the two
    peaks collapsing onto each other or wandering off onto noise.

    Args:
        xdata: Qubit probe frequencies (MHz)
        ydata: Measured signal for one buffer gain
        delta_guess: Expected peak separation (MHz), signed
        gamma_guess: Initial linewidth; defaults to 1/20 of the sweep span

    Returns:
        (popt, perr) matching `two_lorentzians`, or (None, None) if the fit
        fails. popt = [y0, a0, a1, x0, delta, gamma].
    """
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)
    good = np.isfinite(xdata) & np.isfinite(ydata)
    if good.sum() < 7:  # six free parameters
        return None, None
    xdata, ydata = xdata[good], ydata[good]

    span = xdata.max() - xdata.min()
    if gamma_guess is None:
        gamma_guess = span / 20

    # Seed the n=0 peak at the largest excursion from baseline, keeping its
    # sign so the fit works whether the feature is a peak or a dip.
    y0_guess = float(np.median(ydata))
    deviation = ydata - y0_guess
    peak_idx = int(np.argmax(np.abs(deviation)))
    a0_guess = float(deviation[peak_idx])
    x0_guess = float(xdata[peak_idx])

    if delta_guess == 0:
        return None, None
    lo_delta, hi_delta = sorted((0.3 * delta_guess, 3.0 * delta_guess))

    bounds = (
        [-np.inf, -np.inf, -np.inf, xdata.min() - span, lo_delta, 1e-6],
        [np.inf, np.inf, np.inf, xdata.max() + span, hi_delta, span],
    )

    # The brightest excursion is the n=0 peak at low photon number but the n=1
    # peak once the buffer fills up. `delta` is sign-bounded, so seeding x0 on
    # the wrong peak leaves the fit unable to reach the other one. Try both
    # assignments and keep whichever lands the lower residual.
    seeds = (
        (x0_guess, a0_guess, a0_guess / 3),
        (x0_guess - delta_guess, a0_guess / 3, a0_guess),
    )
    best = None
    for x0_seed, a0_seed, a1_seed in seeds:
        p0 = [y0_guess, a0_seed, a1_seed, x0_seed, delta_guess, gamma_guess]
        p0[3] = min(max(p0[3], bounds[0][3]), bounds[1][3])
        try:
            popt, pcov = curve_fit(
                two_lorentzians, xdata, ydata, p0=p0, bounds=bounds, maxfev=20000
            )
        except Exception:
            continue
        resid = float(np.sum((ydata - two_lorentzians(xdata, *popt)) ** 2))
        if best is None or resid < best[0]:
            best = (resid, popt, pcov)

    if best is None:
        return None, None
    return best[1], np.sqrt(np.diag(best[2]))
