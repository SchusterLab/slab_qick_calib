"""
Rabi Oscillation with Flux Pulse Experiment

Variant of the Rabi experiment that applies a constant flux pulse throughout
the qubit drive and readout sequence. All delay_auto calls are replaced with
delay calls so the flux pulse is not waited on — it runs concurrently with
qubit pulses and readout.

Uses sweet_spot_ac gain and f_ge frequency by default.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from qick import *
from qick.asm_v2 import QickSweep1D

from ...analysis import fitting as fitter
from ..general.qick_experiment import (
    QickExperiment,
    QickExperiment2DSimple
)
from ..general.qick_program import QickProgram

from ...exp_handling.datamanagement import AttrDict
from ...helpers import config
FITTER_FUNC = fitter.fitsin
FIT_FUNC = fitter.sinfunc
# ====================================================== #


class RabiFluxProgram(QickProgram):
    """
    Pulse sequence for a Rabi oscillation experiment with a concurrent flux pulse.

    The sequence consists of:
    1. Start flux pulse (runs for the entire body duration)
    2. Wait for flux to settle
    3. Optional pi pulse on |g>-|e> transition (if checking EF transition)
    4. Variable amplitude/length pulse on the qubit (repeated n_pulses times)
    5. Optional second pi pulse on |g>-|e> transition
    6. Measurement (while flux is still on)
    """

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        q = cfg.expt.qubit[0]

        # Initialize with standard readout
        super()._initialize(cfg, readout="standard")

        # Add sweep loop for the experiment
        self.add_loop("sweep_loop", cfg.expt.expts)

        # Create the main qubit pulse
        pulse_params = self._get_pulse_params(cfg)
        super().make_pulse(pulse_params, "qubit_pulse")

        # If checking EF transition and using ge pulse, create a pi pulse
        if (cfg.expt.checkEF and cfg.expt.pulse_ge) or cfg.expt.active_reset:
            super().make_cfg_pulse(q, cfg.device.qubit.f_ge, "pi_ge")

        # Set up flux pulse
        self.flux_ch = cfg.expt.flux_chan
        self.declare_gen(self.flux_ch, nqz=1, mixer_freq=0)

        # Estimate max qubit pulse duration for flux length calculation
        if cfg.expt.sweep == "length":
            max_pulse_dur = cfg.expt.max_length * cfg.expt.n_pulses
        else:
            max_pulse_dur = cfg.expt.length * cfg.expt.n_pulses

        # Flux length covers: settle + qubit pulses + readout + margin
        flux_length = (cfg.expt.flux_settle + max_pulse_dur
                       + self.readout_length + 0.5)

        flux_pulse = {
            "chan": self.flux_ch,
            "freq": 0,
            "phase": 0,
            "gain": cfg.expt.flux_gain,
            "length": flux_length,
            "type": "const",
        }
        super().make_pulse(flux_pulse, "flux_pulse")

    def _get_pulse_params(self, cfg):
        pulse = {
            "freq": cfg.expt.freq,
            "gain": cfg.expt.gain,
            "phase": cfg.expt.phase,
            "type": cfg.expt.pulse_type,
        }

        if cfg.expt.pulse_type == "gauss":
            pulse["sigma"] = cfg.expt.sigma
            pulse["length"] = cfg.expt.length
        elif cfg.expt.pulse_type == "flat_top":
            pulse["length"] = cfg.expt.length
            pulse["ramp_sigma"] = cfg.expt.ramp_sigma
            pulse["ramp_sigma_inc"] = cfg.expt.ramp_sigma_inc
        else:  # const
            pulse["length"] = cfg.expt.length

        return pulse

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)

        # Configure readout
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # Start flux pulse (runs concurrently with everything below)
        self.pulse(ch=self.flux_ch, name="flux_pulse", t=0)
        self.delay(t=cfg.expt.flux_settle, tag="wait_flux_on")

        # If checking EF transition with ge pulse, apply first pi pulse
        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay(t=0.01, tag="wait ef")

        # Apply the main qubit pulse (repeated n_pulses times)
        for i in range(cfg.expt.n_pulses):
            self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)
            if i < cfg.expt.n_pulses - 1:
                self.delay(t=0.01)

        # If checking EF transition with ge pulse, apply second pi pulse
        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            self.delay(t=0.01, tag="wait ef 2")
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        # Add optional end wait time
        if "end_wait" in cfg.expt:
            self.delay(t=cfg.expt.end_wait, tag="end_wait")

        # Perform measurement (flux still on)
        super().measure(cfg)


class RabiFluxExperiment(QickExperiment):
    """
    Rabi oscillation experiment with concurrent flux pulse.

    Same as RabiExperiment but applies a constant flux tone throughout the
    pulse sequence. Defaults to sweet_spot_ac gain and f_ge qubit frequency.
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
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        print=False,
        check_params=True,
    ):
        if prefix is None:
            prefix = self._generate_prefix(params, qi)

        super().__init__(
            cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, check_params=check_params
        )

        params_def = {
            "expts": 60,
            "reps": self.reps,
            "rounds": self.rounds,
            "checkEF": False,
            "pulse_ge": True,
            "num_osc": 2.5,
            "n_pulses": 1,
            "sweep": "amp",
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "loop": False,
            "temp": 40,
            # Flux parameters
            "flux_chan": self.cfg.hw.soc.dacs.flux.ch[qi],
            "flux_gain": self.cfg.device.qubit.sweet_spot_ac[qi],
            "flux_settle": 0.05,
        }

        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 25
        elif style == "temp":
            params_def["reps"] = int(40 * params_def["reps"])
            params_def["rounds"] = int(
                np.ceil(20 * params_def["rounds"] * np.exp(2*(40 / params.get("temp", 40)-1)))
            )
            params_def["pulse_ge"] = False

        params = {**params_def, **params}

        params = self._configure_pulse_params(params, params_def, qi)
        params = {**params_def, **params}

        params = self._configure_sweep_params(params, params_def)

        self.cfg.expt = {**params_def, **params}

        super().check_params(params_def)
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

    def _generate_prefix(self, params, qi):
        if "checkEF" in params and params["checkEF"]:
            ef = "ef_" if params.get("pulse_ge", True) else "ef_no_ge_"
        else:
            ef = ""

        sweep_type = params.get("sweep", "amp")
        name = "length" if sweep_type == "length" else "amp"

        return f"{name}_rabi_flux_{ef}qubit{qi}"

    def _configure_pulse_params(self, params, params_def, qi):
        if params["checkEF"]:
            pulse_config = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            pulse_config = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge_max[qi]

        for key in pulse_config:
            if key not in params:
                params_def[key] = pulse_config[key][qi]

        params_def["pulse_type"] = params_def["type"]
        params = {**params_def, **params}

        # Resolve length from pulse type so downstream code can use direct access
        if "length" not in params or params.get("length") is None:
            if params["pulse_type"] == "gauss":
                params["length"] = params["sigma"] * params["sigma_inc"]
            else:
                params["length"] = params["sigma"]

        # Resolve flat_top ramp defaults
        if params["pulse_type"] == "flat_top":
            if "ramp_sigma" not in params:
                params["ramp_sigma"] = 0.02
            if "ramp_sigma_inc" not in params:
                params["ramp_sigma_inc"] = 3

        return params

    def _configure_sweep_params(self, params, params_def):
        min_gain = 2**-15

        if params["sweep"] == "amp":
            if params["n_pulses"] == 1:
                params_def["start"] = 0.003
                params_def["max_gain"] = params["gain"] * params["num_osc"] * 2
            else:
                gain_range = params["gain"] / params["n_pulses"]
                params_def["start"] = params["gain"] - gain_range
                params_def["max_gain"] = params["gain"] + gain_range

            params = {**params_def, **params}
            params["max_gain"] = min(params["max_gain"], self.cfg.device.qubit.max_gain)

            gain_spacing = (params["max_gain"]-params['start']) / params["expts"]
            if gain_spacing < min_gain:
                span = min_gain * params["expts"]
                params["max_gain"] = params['gain']+span/2
                params["start"] = params['gain']-span/2

        elif params["sweep"] == "length":
            if params["pulse_type"] == "gauss":
                params_def["start"] = self.soccfg.cycles2us(1)
                params_def["max_length"] = 2 * params["num_osc"] * params["sigma"]
            else:
                params_def["start"] = 3 * self.soccfg.cycles2us(1)
                params = {**params_def, **params}
                params_def["max_length"] = 2 * params["num_osc"] * params["length"]

            params = {**params_def, **params}
        return params

    def acquire(self, progress=False, debug=False):
        self.qubit = self.cfg.expt.qubit
        self.param = {"label": "qubit_pulse", "param_type": "pulse"}

        if self.cfg.expt.loop:
            return self._acquire_loop(progress)
        else:
            if self.cfg.expt.pulse_type == 'gauss' and self.cfg.expt.sweep == 'length':
                print('Cannot use qick sweep for a length sweep with Gaussian pulses. Using loop mode instead.')
                return self._acquire_loop(progress)
            return self._acquire_qick_sweep(progress)

    def _acquire_qick_sweep(self, progress):
        if self.cfg.expt.sweep == "amp":
            self.cfg.expt["gain"] = QickSweep1D(
                "sweep_loop", self.cfg.expt.start, self.cfg.expt.max_gain
            )
            self.param['param'] = "gain"

        elif self.cfg.expt.sweep == "length":
            sweep = QickSweep1D("sweep_loop", self.cfg.expt.start, self.cfg.expt.max_length)
            if self.cfg.expt.pulse_type == "gauss":
                self.cfg.expt["sigma"] = sweep
                self.param['param'] = "sigma"
            else:
                self.cfg.expt["length"] = sweep
                self.param['param'] = "total_length"

        super().acquire(RabiFluxProgram, progress=progress)
        return self.data

    def _acquire_loop(self, progress):
        if self.cfg.expt.sweep == "length":
            len_pts = np.linspace(
                self.cfg.expt.start, self.cfg.expt.max_length, self.cfg.expt.expts
            )

            if self.cfg.expt.pulse_type == "gauss":
                x_sweep = [
                    {"pts": len_pts, "var": "sigma"},
                    {"pts": len_pts * self.cfg.expt.sigma_inc, "var": "length"},
                ]
                self.param["param"] = "total_length"
            else:
                x_sweep = [{"pts": len_pts, "var": "length"}]
                self.param["param"] = "total_length"

        else:
            gain_pts = np.linspace(
                self.cfg.expt.start, self.cfg.expt.max_gain, self.cfg.expt.expts
            )
            x_sweep = [{"pts": gain_pts, "var": "gain"}]
            self.param["param"] = "gain"

        self.data = super().run_loop(RabiFluxProgram, x_sweep, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            self.fitterfunc = FITTER_FUNC
            self.fitfunc = FIT_FUNC
            data = super().analyze(fit=fit, **kwargs)

        ydata_lab = ["amps", "avgi", "avgq"]
        for ydata in ydata_lab:
            if f"fit_{ydata}" in data:
                pi_length = fitter.fix_phase(data[f"fit_{ydata}"])
                data[f"pi_length_{ydata}"] = pi_length

        if "best_fit" in data:
            data["pi_length"] = fitter.fix_phase(data["best_fit"])
            data["pi_length_scale_data"] = data.get("pi_length_avgi", data["pi_length"])

        return data

    def update(self, verbose=True):
        qi = self.cfg.expt.qubit[0]
        if "pi_length" not in self.data:
            print("No pi_length found in data, cannot update.")
            return

        pi_val = self.data["pi_length"]
        pulse = "pi_ef" if self.cfg.expt.checkEF else "pi_ge"

        if self.cfg.expt.sweep == "amp":
            config.update_config(
                self.config_file, f"device.qubit.pulses.{pulse}", "gain",
                pi_val, index=qi, verbose=verbose,
            )
        elif self.cfg.expt.sweep == "length":
            if self.cfg.expt.pulse_type == "gauss":
                config.update_config(
                    self.config_file, f"device.qubit.pulses.{pulse}", "sigma",
                    pi_val, index=qi, verbose=verbose,
                )
            else:
                config.update_config(
                    self.config_file, f"device.qubit.pulses.{pulse}", "sigma",
                    pi_val, index=qi, verbose=verbose,
                )

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
        if data is None:
            data = self.data

        title, xlabel = self._get_plot_labels()
        caption_params = [{"index": "pi_length", "format": "$\pi$ length: {val:.3f}"}]

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
            **kwargs,
        )

    def _get_plot_labels(self):
        q = self.cfg.expt.qubit[0]

        if self.cfg.expt.sweep == "amp":
            title = "Amplitude"
            xlabel = "Gain / Max Gain"
            param_name = "sigma"
        else:
            title = "Length"
            xlabel = "Pulse Length ($\mu$s)"
            param_name = "gain"

        param_value = self.cfg.expt[param_name]
        title += f" Rabi Flux Q{q} (Pulse {param_name} {param_value}"

        if self.cfg.expt.checkEF:
            title += ", EF)"
        else:
            title += ")"

        return title, xlabel


class RabiFluxChevronExperiment(QickExperiment2DSimple):
    """
    2D Rabi experiment with flux that sweeps both frequency and amplitude/length.

    Maps out the Rabi chevron pattern with a concurrent flux pulse.
    Uses f_ge_max as the center frequency by default.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        style="",
        prefix=None,
        progress=True,
        live_plot=False,
    ):
        if prefix is None:
            prefix = self._get_prefix(params, qi)

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, live_plot=live_plot)

        params_def = {"span_f": 20, "expts_f": 30, "sweep": "amp"}
        params = {**params_def, **params}

        if params.get("checkEF", False):
            f_qubit = self.cfg.device.qubit.f_ef[qi]
        else:
            f_qubit = self.cfg.device.qubit.f_ge_max[qi]
        params_def["start_f"] = f_qubit - params["span_f"] / 2

        self.expt = RabiFluxExperiment(
            cfg_dict, qi=qi, go=False, params=params, style=style, check_params=False
        )
        params = {**params_def, **params}
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        if go:
            super().run(progress=progress)

    def _get_prefix(self, params, qi):
        sweep_type = params.get("sweep", "amp")
        ef = "ef_" if params.get("checkEF", False) else ""
        return f"{sweep_type}_rabi_flux_chevron_{ef}qubit{qi}"

    def acquire(self, progress=False, debug=False):
        freqpts = np.linspace(
            self.cfg.expt["start_f"],
            self.cfg.expt["start_f"] + self.cfg.expt["span_f"],
            self.cfg.expt["expts_f"],
        )

        ysweep = [{"pts": freqpts, "var": "freq"}]

        self._get_title_and_labels()
        super().acquire(ysweep, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            data = super().analyze(
                fitfunc=FIT_FUNC, fitterfunc=FITTER_FUNC, fit=fit, **kwargs
            )

            if "fit_avgi" in data:
                qubit_freq = self.cfg.device.qubit.f_ge_max[self.cfg.expt.qubit[0]]
                data["chevron_freqs"] = np.array(
                    [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))], dtype=float
                )
                data["chevron_amps"] = np.array(
                    [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))], dtype=float
                )

                valid = np.isfinite(data["chevron_freqs"]) & np.isfinite(data["chevron_amps"])
                if np.any(valid):
                    data["best_freq"] = data["ypts"][valid][np.argmax(data["chevron_amps"][valid])]
                else:
                    data["best_freq"] = data["ypts"][len(data["ypts"]) // 2]

                try:
                    if np.sum(valid) < 3:
                        raise RuntimeError("Not enough valid points for chevron fit")
                    detuning = data["ypts"][valid] - qubit_freq
                    p_freq, _ = curve_fit(chevron_freq, detuning, data["chevron_freqs"][valid])
                    p_amp, _ = curve_fit(chevron_amp, detuning, data["chevron_amps"][valid])
                    data["chevron_freq_fit"] = np.array(p_freq)
                    data["chevron_amp_fit"] = np.array(p_amp)
                except (RuntimeError, TypeError, ValueError):
                    print("Chevron fit failed to converge.")
                    data["chevron_freq_fit"] = np.array([])
                    data["chevron_amp_fit"] = np.array([])

        return data

    def display(self, data=None, fit=True, plot_all=False, **kwargs):
        if data is None:
            data = self.data

        super().display(
            title=self.title,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            data=data,
            fit=fit,
            plot_all=plot_all,
            **kwargs,
        )

        if fit and "chevron_freqs" in data:
            self._plot_chevron_fits(data)

    def _get_title_and_labels(self):
        if self.cfg.expt.sweep == "amp":
            self.xlabel = "Gain / Max Gain"
            param = "sigma" if self.cfg.expt.pulse_type == "gauss" else "length"
        else:
            self.xlabel = "Pulse Length ($\\mu$s)"
            param = "gain"

        self.title = (
            f"Rabi Flux Chevron Q{self.cfg.expt.qubit[0]} "
            f"(Pulse {param} {self.cfg.expt[param]})"
        )
        self.ylabel = "Frequency (MHz)"

    def _plot_chevron_fits(self, data):
        fig, ax = plt.subplots(2, 1, figsize=(6, 6))
        qubit_freq = self.cfg.device.qubit.f_ge_max[self.cfg.expt.qubit[0]]
        detuning = data["ypts"] - qubit_freq

        ax[0].plot(detuning, data["chevron_freqs"], 'o')
        if data.get("chevron_freq_fit") is not None and len(data["chevron_freq_fit"]) > 0:
            ax[0].plot(detuning, chevron_freq(detuning, *data["chevron_freq_fit"]), 'r-')
        ax[0].set_ylabel("Rabi Frequency (MHz)")

        ax[1].plot(detuning, data["chevron_amps"], 'o')
        if data.get("chevron_amp_fit") is not None and len(data["chevron_amp_fit"]) > 0:
            ax[1].plot(detuning, chevron_amp(detuning, *data["chevron_amp_fit"]), 'r-')
        ax[1].set_xlabel("$\\Delta$ Frequency (MHz)")
        ax[1].set_ylabel("Rabi Amplitude")

        plt.tight_layout()
        plt.show()


def chevron_freq(x, w0, x0):
    return np.sqrt(w0**2 + (x - x0)**2)


def chevron_amp(x, w0, a, x0):
    return a / (1 + ((x - x0) / w0) ** 2)
