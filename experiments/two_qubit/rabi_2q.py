import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from qick import *
from qick.asm_v2 import QickSweep1D
from scipy.optimize import curve_fit

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment2DSimple
from ..general.qick_experiment_2q import QickExperiment2Q
from ..general.qick_program import QickProgram2Q
from ...analysis import fitting as fitter


# ====================================================== #


class RabiProgram_2Q(QickProgram2Q):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        super()._initialize(cfg, readout="standard")
        for i, q in enumerate(cfg.expt.qubit):
            pulse = {
                "sigma": cfg.expt.sigma[i],
                "length": cfg.expt.length[i],
                "freq": cfg.expt.freq[i],
                "gain": cfg.expt.gain[i],
                "phase": 0,
                "type": cfg.expt.pulse_type,
            }
            super().make_pulse(i, pulse, f"qubit_pulse_{i}")
            super().make_cfg_pulse(q, i, cfg.device.qubit.f_ge, "pi_ge")

        self.add_loop("sweep_loop", cfg.expt.expts)
        # if cfg.expt.checkEF and cfg.expt.pulse_ge:

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        for q in range(len(cfg.expt.qubit)):
            self.send_readoutconfig(ch=self.adc_ch[q], name=f"readout_{q}", t=0)

        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            for q in range(len(cfg.expt.qubit)):
                self.pulse(ch=self.qubit_ch[q], name=f"pi_ge_{q}", t=0)

            self.delay_auto(t=0.01, tag="wait ef")

        for q in range(len(cfg.expt.qubit)):
            self.pulse(ch=self.qubit_ch[q], name=f"qubit_pulse_{q}", t=0)
        self.delay_auto(t=0.01, tag="wait")

        if cfg.expt.checkEF and cfg.expt.pulse_ge:
            for q in range(len(cfg.expt.qubit)):
                self.pulse(ch=self.qubit_ch[q], name=f"pi_ge_{q}", t=0)
            self.delay_auto(t=0.01, tag="wait ef 2")

        for q in range(len(cfg.expt.qubit)):
            self.pulse(ch=self.res_ch[q], name=f"readout_pulse_{q}", t=0)
            if self.lo_ch[q] is not None:
                self.pulse(ch=self.lo_ch[q], name=f"mix_pulse_{q}", t=0.0)
            self.trigger(
                ros=[self.adc_ch[q]],
                pins=[0],
                t=self.trig_offset[q],
            )
        if cfg.expt.active_reset:
            self.reset(3)

    def reset(self, i):
        super().reset(i)


# ====================================================== #
class Rabi_2Q(QickExperiment2Q):
    """
    - 'expts': Number of experiments to run (default: 60)
    - 'reps': Number of repetitions for each experiment (default: self.reps)
    - 'rounds': Number of rounds for each experiment (default: self.rounds)
    - 'gain': Max gain value for the pulse (default: gain)
    - 'sigma': Standard deviation of the Gaussian pulse (default: sigma)
    - 'checkEF': Boolean flag to check EF interaction (default: False)
    - 'checkCC': Boolean flag to check CC interaction (default: False)
    - 'pulse_ge': Boolean flag to indicate if pulse is for ground to excited state transition (default: True)
    - 'start': Starting point for the experiment (default: 0)
    - 'step': Step size for the gain (calculated as int(params['gain']/params['expts']))
    - 'qubit': List of qubits involved in the experiment (default: [qi])
    - 'pulse_type': Type of pulse used in the experiment (default: 'gauss')
    - 'num_pulses': Number of pulses used in the experiment (default: 1)
    - 'qubit_chan': Channel for the qubit readout (default: self.cfg.hw.soc.adcs.readout.ch[qi])
    Additional keys may be added based on the specific requirements of the experiment.
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
        if prefix is None:
            prefix = "amp_rabi"
            if "checkEF" in params and params["checkEF"]:
                if "pulse_ge" in params and not params["pulse_ge"]:
                    prefix += f"ef_no_ge"
                else:
                    prefix = f"ef"

            for q in qi:
                prefix += f"_Q{q}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)
        params_def = {
            "expts": 60,
            "reps": self.reps,
            "rounds": self.rounds,
            "checkEF": False,
            "pulse_ge": True,
            "type": "amp",
            "pulse_type": "gauss",
            "active_reset": [self.cfg.device.readout.active_reset[q] for q in qi],
            "qubit": qi,
            "qubit_chan": [self.cfg.hw.soc.adcs.readout.ch[q] for q in qi],
        }
        params = {**params_def, **params}

        if params["checkEF"]:
            params_def["sigma"] = self.cfg.device.qubit.pulses.pi_ef.sigma[qi]
            params_def["sigma_inc"] = self.cfg.device.qubit.pulses.pi_ef.sigma_inc[qi]
            params_def["gain"] = self.cfg.device.qubit.pulses.pi_ef.gain[qi]
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            params_def["sigma"] = [
                self.cfg.device.qubit.pulses.pi_ge.sigma[q] for q in qi
            ]
            params_def["sigma_inc"] = [
                self.cfg.device.qubit.pulses.pi_ge.sigma_inc[q] for q in qi
            ]
            params_def["gain"] = [
                self.cfg.device.qubit.pulses.pi_ge.gain[q] for q in qi
            ]
            params_def["freq"] = [self.cfg.device.qubit.f_ge[q] for q in qi]
        params = {**params_def, **params}
        if params["type"] == "amp":
            params_def["max_gain"] = [params["gain"][i] * 4 for i in range(len(qi))]
            params_def["start"] = [0] * len(qi)
        elif params["type"] == "length":
            params_def["sigma"] = 4 * params["sigma"]
            params_def["start"] = 3 / 430

        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 25
        elif style == "temp":
            params_def["reps"] = 40 * params_def["reps"]
            params_def["rounds"] = 40 * params_def["rounds"]
            params_def["pulse_ge"] = False

        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        self.qubit = self.cfg.expt.qubit
        param_pulse = "gain"
        self.cfg.expt["gain"] = [
            QickSweep1D("sweep_loop", self.cfg.expt.start[i], self.cfg.expt.max_gain[i])
            for i in range(len(self.qubit))
        ]
        self.cfg.expt.length = [
            self.cfg.expt.sigma[q] * self.cfg.expt.sigma_inc[q]
            for q in range(len(self.qubit))
        ]
        self.param = []
        for i in range(len(self.qubit)):
            self.param.append(
                {
                    "label": f"qubit_pulse_{i}",
                    "param": param_pulse,
                    "param_type": "pulse",
                }
            )

        super().acquire(RabiProgram_2Q, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            # fitparams=[amp, freq (non-angular), phase (deg), decay time, amp offset]
            fitterfunc = fitter.fitsin
            fitfunc = fitter.sinfunc
            data = super().analyze(
                fitfunc=fitfunc, fitterfunc=fitterfunc, fit=fit, **kwargs
            )

        # # Get pi length from fit
        # ydata_lab = ["amps", "avgi", "avgq"]
        # for ydata in ydata_lab:
        #     for i in range(len(data["amps"])):
        #         pi_length = fitter.fix_phase(data["fit_" + ydata][i])
        #         data["pi_length_" + ydata + str(i)] = pi_length

        #         data["pi_length_"+str(i)] = fitter.fix_phase(data[i]["best_fit"])
        return data

    def display(
        self,
        data=None,
        fit=False,
        plot_all=False,
        ax=None,
        show_hist=True,
        **kwargs,
    ):
        if data is None:
            data = self.data

        q = self.cfg.expt.qubit[0]
        if self.cfg.expt.type == "amp":
            title = "Amplitude"
            param = "sigma"
            xlabel = "Gain / Max Gain"
        else:
            title = "Length"
            param = "gain"
            xlabel = "Pulse Length ($\mu$s)"

        title_list = []
        for i in range(len(self.cfg.expt.qubit)):
            title_list.append(
                title
                + f" Rabi Q{self.cfg.expt.qubit[i]} (Pulse {param} {self.cfg.expt[param][i]})"
            )

        fitfunc = fitter.sinfunc
        # caption_params =[]
        caption_params = [{"index": "pi_length", "format": "$\pi$ length: {val:.3f}"}]

        if self.cfg.expt.checkEF:
            title = title + ", EF)"
        else:
            title = title + ")"

        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title_list,
            xlabel=xlabel,
            fit=fit,
            show_hist=show_hist,
            fitfunc=fitfunc,
            caption_params=caption_params,
        )


# ====================================================== #


class RabiChevron_2Q(QickExperiment2DSimple):
    """
    Amplitude Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz),
        step_f: frequency step (MHz),
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        style="",
        prefix=None,
        progress=None,
    ):
        if "type" in params:
            pre = params["type"]
        else:
            pre = "amp"
        if "checkEF" in params and params["checkEF"]:
            ef = "ef"
        else:
            ef = ""
        prefix = f"{pre}_rabi_chevron_{ef}_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "span_f": 20,
            "expts_f": 30,
        }
        params = {**params_def, **params}
        if "checkEF" in params and params["checkEF"]:
            params_def["start_f"] = (
                self.cfg.device.qubit.f_ef[qi] - params["span_f"] / 2
            )
        else:
            params_def["start_f"] = (
                self.cfg.device.qubit.f_ge[qi] - params["span_f"] / 2
            )

        self.expt = RabiExperiment(cfg_dict, qi, go=False, params=params, style=style, check_params=False)
        params = {**params_def, **params}
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params

        if go:
            super().run()

    def acquire(self, progress=False, debug=False):
        # expand entries in config that are length 1 to fill all qubits

        freqpts = np.linspace(
            self.cfg.expt["start_f"],
            self.cfg.expt["start_f"] + self.cfg.expt["span_f"],
            self.cfg.expt["expts_f"],
        )

        ysweep = [{"pts": freqpts, "var": "freq"}]
        super().acquire(ysweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            fitterfunc = fitter.fitsin
            fitfunc = fitter.sinfunc
            data = super().analyze(
                fitfunc=fitfunc, fitterfunc=fitterfunc, fit=fit, **kwargs
            )
            qubit_freq = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            freq = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]
            amp = [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))]

            p, _ = curve_fit(chevron_freq, data["ypts"] - qubit_freq, freq)
            p2, _ = curve_fit(chevron_amp, data["ypts"] - qubit_freq, amp)
            data["chevron_freq"] = p
            data["chevron_amp"] = p2

    def display(self, data=None, fit=True, plot_both=False, show_hist=False, **kwargs):
        if data is None:
            data = self.data
        if self.cfg.expt.checkEF:
            title = "EF"
        else:
            title = ""
        if self.cfg.expt.type == "amp":
            title = "Amplitude"
            param = "sigma"
            xlabel = "Gain / Max Gain"
        else:
            title = "Length"
            param = "gain"
            xlabel = "Pulse Length ($\mu$s)"
        title_list = []
        for i in range(len(self.cfg.expt.qubit)):
            title_list[i] = (
                title
                + f" Rabi Q{self.cfg.expt.qubit[0]} (Pulse {param} {self.cfg.expt[param]}"
            )

        xlabel = xlabel
        ylabel = "Frequency [MHz]"

        super().display(
            title=title_list,
            xlabel=xlabel,
            ylabel=ylabel,
            data=data,
            fit=fit,
            plot_both=plot_both,
            show_hist=show_hist,
            **kwargs,
        )

        if fit:
            fig, ax = plt.subplots(2, 1, figsize=(6, 6))
            qubit_freq = self.cfg.device.qubit.f_ge[self.cfg.expt.qubit[0]]
            freq = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]
            amp = [data["fit_avgi"][i][0] for i in range(len(data["ypts"]))]
            ax[0].plot(data["ypts"] - qubit_freq, freq)
            ax[1].plot(data["ypts"] - qubit_freq, amp)
            ax[1].set_xlabel("$\Delta$ Frequency (MHz)")
            ax[0].set_ylabel("Frequency (MHz)")
            ax[1].set_ylabel("Amplitude")


def chevron_freq(x, w0):
    return np.sqrt(w0**2 + x**2)


def chevron_amp(x, w0, a):
    return a / (1 + (x / w0) ** 2)
