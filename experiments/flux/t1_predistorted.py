"""
T1 measurement with predistorted flux pulse.

Uses the IIR inverse filter from flux step-response characterization to
generate a predistorted arb envelope on the int generator, compensating
for exponential distortions (reflections, bias-tee droop, etc.).

Because the arb envelope length is fixed at program load time, the wait-time
sweep is done in a Python loop: each wait time builds a fresh program with
the correctly-sized predistorted envelope.

Classes:
- T1PredistProgram: QickProgram with predistorted flux envelope (single point)
- T1Predist: Experiment wrapper (same interface as T1Experiment)
"""

import numpy as np
from tqdm import tqdm

from ...exp_handling.datamanagement import AttrDict
from ...helpers import config
from ..general.qick_experiment import QickExperiment, DataProcessor, get_current_time_string
from ..general.qick_program import QickProgram
from ...analysis.flux_filter import make_full_predistorted_step, make_slow_ramp

from ...analysis import fitting as fitter
FIT_FUNC = fitter.expfunc
FITTER_FUNC = fitter.fitexp
XLABEL = "Wait Time ($\\mu$s)"


class T1PredistProgram(QickProgram):
    """
    Pulse sequence for a single-point T1 measurement with a predistorted flux pulse.

    cfg.expt.wait_time must be a scalar (not a sweep).  The predistorted
    envelope is sized to exactly wait_time.

    Sequence:
    1. Pi pulse to excite qubit to |1>
    2. Predistorted flux pulse for wait_time
    3. Measurement
    4. (Optional) Negative reset pulse
    """

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """Set up readout, pi pulse, and the predistorted flux envelope sized to wait_time."""
        cfg = AttrDict(self.cfg)

        q = cfg.expt.qubit[0]

        super()._initialize(cfg, readout="standard")

        # --- Pi pulse ---
        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        # --- Predistorted flux envelope ---
        if cfg.expt.flux:
            flux_ch = cfg.expt.flux_chan
            self.declare_gen(flux_ch, nqz=1, mixer_freq=0)

            gen_cfg = self.soccfg['gens'][flux_ch]
            fs_mhz = gen_cfg['f_fabric']
            maxv = self.soccfg.get_maxv(flux_ch)

            wait_time = cfg.expt.wait_time
            total_length_us = wait_time

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

            scaled_gain = 2 * cfg.expt.flux_gain * env["gain_scale"]
            self.flux_gain_scale = env["gain_scale"]

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

            if cfg.expt.flux_negative_reset:
                self.add_pulse(
                    ch=flux_ch,
                    name="flux_pulse_neg",
                    style="arb",
                    envelope="flux_predist",
                    freq=0,
                    phase=0,
                    gain=-scaled_gain,
                    outsel="product",
                )

    def _body(self, cfg):
        """Play pi pulse, predistorted flux pulse, readout, and the optional negative reset pulse."""
        cfg = AttrDict(self.cfg)

        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # Pi pulse to excite qubit
        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.flux:
            self.delay_auto(t=0.01, tag="wait_pre_flux")
            self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse", t=0)
            self.delay_auto(t=cfg.expt.flux_readout_wait, tag="wait_readout")
        else:
            self.delay_auto(t=cfg.expt.wait_time + 0.03, tag="wait")

        super().measure(cfg)

        if cfg.expt.flux and cfg.expt.flux_negative_reset:
            self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse_neg", t=0)


class T1Predist(QickExperiment):
    """
    T1 measurement with predistorted flux pulse.

    Same interface as T1Experiment but uses a predistorted arb envelope
    for the flux pulse instead of a const pulse.  Because each wait time
    requires a different envelope length, the sweep is done in a Python
    loop rather than a hardware sweep.

    Additional parameters (in params dict):
        flux_alphas: List of exponential amplitudes from step-response fit
        flux_taus: List of time constants (us)
        flux_alpha0: Steady-state coefficient (default: 1.0)
        flux_slow_only: If True, only use slow exponentials (default: False)
        flux_tau_threshold: Threshold for slow/fast split in us (default: 0.5)
        flux_negative_reset: If True, play negative flux pulse after readout (default: False)
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
        if prefix is None:
            prefix = f"t1_predist_qubit{qi}"
        super().__init__(
            cfg_dict=cfg_dict, prefix=prefix, fname=fname,
            progress=progress, qi=qi, check_params=check_params)

        user_params = params

        params_def = {
            "reps": int(1.5 * self.reps),
            "rounds": self.rounds,
            "expts": 60,
            "start": 0,
            "span": 3.7 * self.cfg.device.qubit.T1[qi],
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            # Flux defaults
            "flux": True,
            "flux_chan": self.cfg.hw.soc.dacs.flux.ch[qi],
            "flux_gain": self.cfg.device.qubit.sweet_spot_ac[qi],
            "flux_readout_wait": 0.1,
            "flux_negative_reset": False,
            # Predistortion parameters (loaded from config if not provided)
            "flux_alphas": [],
            "flux_taus": [],
            "flux_alpha0": 1.0,
            "flux_slow_only": False,
            "flux_tau_threshold": 0.5,
        }

        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 30

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

        self.cfg.expt = params
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
        Sweep wait_time in a Python loop, rebuilding the program per point.

        The arb envelope length is fixed at program load, so each wait time
        gets a fresh T1PredistProgram with a correctly-sized envelope.  The
        span is capped so the longest envelope fits in generator memory.
        """
        q = self.cfg.expt.qubit[0]
        final_delay = self._get_final_delay()

        # Cap span so envelope fits in hardware memory
        if self.cfg.expt.flux:
            flux_ch = self.cfg.expt.flux_chan
            gen_cfg = self.soccfg['gens'][flux_ch]
            fs_mhz = gen_cfg['f_fabric']
            maxlen = gen_cfg['maxlen']
            max_envelope_us = maxlen / fs_mhz
            max_wait = max_envelope_us
            max_span = max_wait - self.cfg.expt.start
            if self.cfg.expt.span > max_span:
                print(f"[T1Predist] Capping span from {self.cfg.expt.span:.2f} "
                      f"to {max_span:.2f} µs (maxlen={maxlen})")
                self.cfg.expt.span = max_span

        # Build wait-time array
        wait_times = np.linspace(
            self.cfg.expt.start,
            self.cfg.expt.start + self.cfg.expt.span,
            self.cfg.expt.expts,
        )

        current_time = get_current_time_string()

        all_avgi = []
        all_avgq = []

        soc = self.im[self.cfg.aliases.soc]
        iterator = tqdm(wait_times) if progress else wait_times

        for wt in iterator:
            # Set scalar wait_time for this point
            self.cfg.expt.wait_time = float(wt)

            # Adjust final_delay for negative reset envelope length
            fd = final_delay
            if self.cfg.expt.flux and self.cfg.expt.flux_negative_reset:
                flux_pulse_length = wt
                fd = max(1.0, final_delay - flux_pulse_length)
            self.cfg.device.readout.final_delay[q] = fd

            # Build a fresh program with the correctly-sized envelope
            for attempt in range(3):
                try:
                    prog = T1PredistProgram(
                        soccfg=self.soccfg, final_delay=fd, cfg=self.cfg)
                    iq_list = prog.acquire(
                        soc,
                        rounds=self.cfg.expt.rounds,
                        threshold=None,
                        progress=False,
                    )
                    break
                except (ValueError, RuntimeError) as e:
                    if attempt < 2:
                        print(f"[T1Predist] Retry {attempt+1} for wt={wt:.3f}: {e}")
                    else:
                        print(f"[T1Predist] Failed after 3 attempts for wt={wt:.3f}: {e}")
                        iq_list = None

            if iq_list is None:
                all_avgi.append(np.nan)
                all_avgq.append(np.nan)
                continue

            # iq_list[0][0] shape: (1, 2) for single-point (reps averaged)
            iq = iq_list[0][0]
            avgi = np.squeeze(iq[..., 0])
            avgq = np.squeeze(iq[..., 1])
            all_avgi.append(float(avgi))
            all_avgq.append(float(avgq))

        all_avgi = np.array(all_avgi)
        all_avgq = np.array(all_avgq)
        amps = np.abs(all_avgi + 1j * all_avgq)
        phases = np.angle(all_avgi + 1j * all_avgq)

        data = {
            "xpts": wait_times,
            "avgi": all_avgi,
            "avgq": all_avgq,
            "amps": amps,
            "phases": phases,
            "start_time": current_time,
        }
        for key in data:
            data[key] = np.array(data[key])

        self.data = data

        try:
            self.scale_ge_config()
        except Exception:
            pass

        return self.data

    def analyze(self, data=None, **kwargs):
        """Fit the decay to an exponential and store the new T1 estimate."""
        if data is None:
            data = self.data

        self.fitfunc = FIT_FUNC
        self.fitterfunc = FITTER_FUNC
        super().analyze(data, **kwargs)

        data["new_t1"] = data["best_fit"][2]
        data["new_t1_i"] = data["fit_avgi"][2]
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
        """Plot the T1 decay with the fitted T1 in the caption."""
        qubit = self.cfg.expt.qubit[0]
        title = f"$T_1$ Q{qubit}"
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
        title += ", predistorted)"

        caption_params = [
            {"index": 2, "format": "$T_1$ fit: {val:.3} $\\pm$ {err:.2} $\\mu$s"},
        ]

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

    def plot_envelope(self, wait_time=None, ax=None):
        """Plot the predistorted flux envelope for a given wait time.

        Args:
            wait_time: Wait time in us.  Defaults to the maximum (start + span).
            ax: Optional matplotlib axis.
        """
        import matplotlib.pyplot as plt
        import scipy.signal
        from ...analysis.flux_filter import step_response_to_tf

        e = self.cfg.expt
        if wait_time is None:
            wait_time = e.start + e.span
        total_length_us = wait_time

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

        alphas = np.asarray(e.flux_alphas)
        taus = np.asarray(e.flux_taus)
        num, den = step_response_to_tf(alphas, taus, e.flux_alpha0)
        b_fwd, a_fwd = scipy.signal.bilinear(num, den, fs=fs_mhz)
        sos_fwd = scipy.signal.tf2sos(b_fwd, a_fwd)

        flux_at_qubit = e.flux_gain * scipy.signal.sosfilt(sos_fwd, env["waveform"])
        const_step = np.ones(env["n_samples"])
        flux_uncorrected = e.flux_gain * scipy.signal.sosfilt(sos_fwd, const_step)

        if ax is None:
            fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        else:
            axes = [ax, ax.twinx()]
            fig = ax.get_figure()

        axes[0].plot(env["time_us"], flux_at_qubit, label="predistorted")
        axes[0].plot(env["time_us"], flux_uncorrected, ls="--", alpha=0.5,
                     label="uncorrected")
        axes[0].axhline(e.flux_gain, color="gray", ls=":", lw=0.8)
        axes[0].set_ylabel("Flux at qubit")
        axes[0].legend(fontsize=8)

        dac_output = e.flux_gain * env["waveform"]
        axes[1].plot(env["time_us"], dac_output)
        axes[1].axhline(e.flux_gain, color="gray", ls=":", lw=0.8)
        axes[1].set_xlabel("Time (µs)")
        axes[1].set_ylabel("DAC output")
        info = (f"gain_scale={env['gain_scale']:.3f}, "
                f"QICK gain={e.flux_gain * env['gain_scale']:.4f}, "
                f"wait_time={wait_time:.2f} µs")
        axes[1].text(0.98, 0.02, info, transform=axes[1].transAxes,
                     ha="right", va="bottom", fontsize=8)

        plt.tight_layout()
        plt.show()
        return env

    def update(self, rng_vals=[0.5, 500], first_time=False, no_final=False, verbose=True):
        """Write the fitted T1 (and derived final_delay) back to the config file."""
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
