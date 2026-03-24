"""
Qubit spectroscopy with predistorted flux pulse.

Uses the IIR inverse filter from flux step-response characterization to
generate a predistorted arb envelope on the int generator, compensating
for exponential distortions (reflections, bias-tee droop, etc.).

The predistorted envelope is loaded via add_envelope and played as an
arb-style pulse with freq=0 (DC flux).

Classes:
- QubitSpecPredistProgram: QickProgram with predistorted flux envelope
- QubitSpecPredist: Experiment wrapper (same interface as QubitSpec)
"""

import numpy as np
from qick.asm_v2 import QickSweep1D

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment
from ..general.qick_program import QickProgram
from ...analysis.flux_filter import make_full_predistorted_step, make_slow_ramp

from ...analysis import fitting as fitter
FIT_FUNC = fitter.lorfunc
FITTER_FUNC = fitter.fitlor


class QubitSpecPredistProgram(QickProgram):
    """
    Pulse sequence for qubit spectroscopy with a predistorted flux pulse.

    The flux pulse is an arb envelope generated from the IIR inverse filter,
    played on the int generator at freq=0.  The sequence is:

    1. Predistorted flux pulse starts
    2. Wait flux_lead_time for the flux to settle
    3. Qubit probe pulse (frequency swept)
    4. Wait flux_trail_time
    5. Measurement
    """

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        q = cfg.expt.qubit[0]
        self.frequency = cfg.device.readout.frequency[q]
        self.gain = cfg.device.readout.gain[q]
        self.readout_length = cfg.expt.readout_length
        self.phase = cfg.device.readout.phase[q]

        super()._initialize(cfg, readout="standard")

        # --- Predistorted flux envelope ---
        if cfg.expt.flux:
            flux_ch = cfg.expt.flux_chan
            self.declare_gen(flux_ch, nqz=1, mixer_freq=0)

            gen_cfg = self.soccfg['gens'][flux_ch]
            fs_mhz = gen_cfg['f_fabric']
            maxv = self.soccfg.get_maxv(flux_ch)

            total_length_us = (cfg.expt.length
                               + cfg.expt.flux_lead_time
                               + cfg.expt.flux_trail_time)

            # Select which terms to include
            flux_alphas = cfg.expt.flux_alphas
            flux_taus = cfg.expt.flux_taus
            flux_alpha0 = cfg.expt.flux_alpha0

            if cfg.expt.flux_slow_only:
                env = make_slow_ramp(
                    flux_alphas, flux_taus, flux_alpha0,
                    duration_us=total_length_us, fs_mhz=fs_mhz,
                    maxv=maxv,
                    tau_threshold_us=cfg.expt.flux_tau_threshold)
            else:
                env = make_full_predistorted_step(
                    flux_alphas, flux_taus, flux_alpha0,
                    duration_us=total_length_us, fs_mhz=fs_mhz,
                    maxv=maxv)

            self.add_envelope(flux_ch, name="flux_predist", idata=env["idata"])

            # Scale gain: envelope is normalized so peak maps to maxv.
            # Multiply by gain_scale to preserve amplitude, and by 2 because
            # QICK arb pulses have half the amplitude of const pulses.
            scaled_gain = 2 * cfg.expt.flux_gain * env["gain_scale"]
            self.flux_gain_scale = env["gain_scale"]
            print(f"[predist] flux_gain={cfg.expt.flux_gain}, gain_scale={env['gain_scale']:.4f}, "
                  f"scaled_gain={scaled_gain:.4f} (2*flux_gain*gain_scale), "
                  f"n_samples={env['n_samples']}, "
                  f"waveform=[{env['waveform'][0]:.3f}..{env['waveform'][-1]:.3f}], "
                  f"maxv={maxv}, idata=[{env['idata'][0]}..{env['idata'][-1]}]")

            self.add_pulse(
                ch=flux_ch,
                name="flux_pulse",
                style="arb",
                envelope="flux_predist",
                freq=0,
                phase=0,
                gain=scaled_gain,
                outsel="product",
            )

        # --- Qubit probe pulse ---
        pulse = {
            "freq": cfg.expt.frequency,
            "gain": cfg.expt.gain,
            "type": cfg.expt.pulse_type,
            "sigma": cfg.expt.length,
            "sigma_inc": 4,
            "phase": 0,
        }
        super().make_pulse(pulse, "qubit_pulse")

        self.add_loop("freq_loop", cfg.expt.expts)

        if cfg.expt.checkEF:
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)

        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait 1")

        if cfg.expt.flux:
            self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse", t=0)
            self.delay(t=cfg.expt.flux_lead_time, tag="wait flux")
            self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
            self.delay_auto(t=cfg.expt.flux_readout_wait, tag="wait flux_2")
        else:
            self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
            if cfg.expt.sep_readout:
                self.delay_auto(t=0.01, tag="wait")

        if cfg.expt.checkEF:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait 2")

        super().measure(cfg)


class QubitSpecPredist(QickExperiment):
    """
    Qubit spectroscopy with predistorted flux pulse.

    Same interface as QubitSpec but uses a predistorted arb envelope
    for the flux pulse instead of a const pulse.

    Additional parameters (in params dict):
        flux_alphas: List of exponential amplitudes from step-response fit
        flux_taus: List of time constants (µs)
        flux_alpha0: Steady-state coefficient (default: 1.0)
        flux_slow_only: If True, only use slow exponentials (default: False)
        flux_tau_threshold: Threshold for slow/fast split in µs (default: 0.5)
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
        check_params=True,
    ):
        user_params = params
        prefix = prefix or f"qspec_predist_{style}_qubit{qi}"
        super().__init__(
            cfg_dict=cfg_dict, prefix=prefix, progress=progress,
            check_params=check_params, qi=qi)

        spec_gain = self.cfg.device.qubit.spec_gain[qi]
        low_gain = self.cfg.device.qubit.low_gain

        if style == "huge":
            style_defaults = {"gain": 80 * low_gain * spec_gain,
                              "span": 1500, "expts": 1000, "reps": self.reps}
        elif style == "coarse":
            style_defaults = {"gain": 20 * low_gain * spec_gain,
                              "span": 500, "expts": 500, "reps": self.reps}
        elif style == "medium":
            style_defaults = {"gain": 5 * low_gain * spec_gain,
                              "span": 50, "expts": 200, "reps": self.reps}
        elif style == "fine":
            style_defaults = {"gain": low_gain * spec_gain,
                              "span": 5, "expts": 100, "reps": 2 * self.reps}
        else:
            style_defaults = {"gain": 5 * low_gain * spec_gain,
                              "span": 50, "expts": 200, "reps": self.reps}

        params_def = {
            "rounds": self.rounds,
            "final_delay": 10,
            "length": 10,
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "pulse_type": "const",
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            "active_reset": False,
            # Flux defaults
            "flux": True,
            "flux2": False,
            "flux_chan": self.cfg.hw.soc.dacs.flux.ch[qi],
            "flux_gain": self.cfg.device.qubit.sweet_spot_ac[qi],
            "flux_lead_time": 0.035,
            "flux_trail_time": 0.025,
            "flux_readout_wait": 0.1,
            # Predistortion parameters (loaded from config if not provided)
            "flux_alphas": [],
            "flux_taus": [],
            "flux_alpha0": 1.0,
            "flux_slow_only": False,
            "flux_tau_threshold": 0.5,
        }
        params_def = {**params_def, **style_defaults}
        params = {**params_def, **params}

        # Load predistortion params from config if not provided explicitly
        cfg_flux = self.cfg.hw.soc.dacs.flux
        if not params["flux_alphas"]:
            params["flux_alphas"] = list(cfg_flux.step_alphas[qi])
        if not params["flux_taus"]:
            params["flux_taus"] = list(cfg_flux.step_taus[qi])
        if "flux_alpha0" not in user_params:
            step_alpha0 = getattr(cfg_flux, "step_alpha0", None)
            if step_alpha0 is not None:
                val = step_alpha0[qi]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    params["flux_alpha0"] = val

        if not params["flux_alphas"] or not params["flux_taus"]:
            raise ValueError(
                "flux_alphas and flux_taus must be provided or present in config "
                "(hw.soc.dacs.flux.step_alphas/step_taus). "
                "Run a flux step-response fit first.")

        if "start" not in user_params:
            if params["checkEF"]:
                params["start"] = self.cfg.device.qubit.f_ef[qi] - params["span"] / 2
            else:
                params["start"] = self.cfg.device.qubit.f_ge[qi] - params["span"] / 2

        max_length = 100
        if params["length"] > max_length:
            params["length"] = max_length

        self.cfg.expt = params
        super().check_params(params_def)

        if go:
            super().run(min_r2=min_r2, max_err=max_err,
                        display=display, progress=progress)

    def acquire(self, progress=False):
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay

        self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}
        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span)

        super().acquire(QubitSpecPredistProgram, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        if fit:
            self.fitterfunc = FITTER_FUNC
            self.fitfunc = FIT_FUNC
            super().analyze(use_i=False, **kwargs)
            data["new_freq"] = data["best_fit"][2]
        return self.data

    def plot_envelope(self, ax=None):
        """Plot the predistorted flux envelope that will be used by the program."""
        import matplotlib.pyplot as plt

        e = self.cfg.expt
        total_length_us = e.length + e.flux_lead_time + e.flux_trail_time

        # Use the same config the hardware would use; fall back to default maxv
        soccfg = getattr(self, "soccfg", None)
        if soccfg is not None:
            flux_ch = e.flux_chan
            fs_mhz = soccfg['gens'][flux_ch]['f_fabric']
            maxv = soccfg.get_maxv(flux_ch)
        else:
            fs_mhz = 430.08
            maxv = int(0.9 * 32767)

        if e.flux_slow_only:
            env = make_slow_ramp(
                e.flux_alphas, e.flux_taus, e.flux_alpha0,
                duration_us=total_length_us, fs_mhz=fs_mhz, maxv=maxv,
                tau_threshold_us=e.flux_tau_threshold)
        else:
            env = make_full_predistorted_step(
                e.flux_alphas, e.flux_taus, e.flux_alpha0,
                duration_us=total_length_us, fs_mhz=fs_mhz, maxv=maxv)

        # Simulate the expected flux at the qubit (after analog distortion)
        import scipy.signal
        from ...analysis.flux_filter import step_response_to_tf

        alphas = np.asarray(e.flux_alphas)
        taus = np.asarray(e.flux_taus)
        num, den = step_response_to_tf(alphas, taus, e.flux_alpha0)
        b_fwd, a_fwd = scipy.signal.bilinear(num, den, fs=fs_mhz)
        sos_fwd = scipy.signal.tf2sos(b_fwd, a_fwd)

        # Flux at qubit = flux_gain * (distortion applied to predist waveform)
        flux_at_qubit = e.flux_gain * scipy.signal.sosfilt(sos_fwd, env["waveform"])

        # Also show what an uncorrected const step would produce
        const_step = np.ones(env["n_samples"])
        flux_uncorrected = e.flux_gain * scipy.signal.sosfilt(sos_fwd, const_step)

        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        else:
            axes = [ax, ax.twinx()]
            fig = ax.get_figure()

        # Top: flux at qubit (after distortion)
        axes[0].plot(env["time_us"], flux_at_qubit, label="predistorted")
        axes[0].plot(env["time_us"], flux_uncorrected, ls="--", alpha=0.5,
                     label="uncorrected")
        axes[0].axhline(e.flux_gain, color="gray", ls=":", lw=0.8)
        axes[0].axvline(e.flux_lead_time, color="C2", ls=":", lw=0.8)
        axes[0].set_ylabel("Flux at qubit")
        axes[0].legend(fontsize=8)

        # Bottom: DAC output (what actually gets sent)
        dac_output = e.flux_gain * env["waveform"]
        axes[1].plot(env["time_us"], dac_output)
        axes[1].axhline(e.flux_gain, color="gray", ls=":", lw=0.8)
        axes[1].axvline(e.flux_lead_time, color="C2", ls=":", lw=0.8)
        axes[1].set_xlabel("Time (µs)")
        axes[1].set_ylabel("DAC output")
        info = (f"gain_scale={env['gain_scale']:.3f}, "
                f"QICK gain={e.flux_gain * env['gain_scale']:.4f}")
        axes[1].text(0.98, 0.02, info, transform=axes[1].transAxes,
                     ha="right", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.show()
        return env

    def display(self, fit=True, ax=None, plot_all=True, **kwargs):
        fitfunc = self.fitfunc
        xlabel = "Qubit Frequency (MHz)"
        title = (f"Spectroscopy Q{self.cfg.expt.qubit[0]} "
                 f"(Flux Gain {self.cfg.expt.flux_gain}, predistorted)")
        if self.cfg.expt.checkEF:
            title = "EF " + title

        caption_params = [
            {"index": 2, "format": "$f$: {val:.6} MHz"},
            {"index": 3, "format": r"$\kappa$: {val:.3} MHz"},
        ]

        super().display(
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fitfunc=fitfunc,
            caption_params=caption_params,
            **kwargs,
        )
