import matplotlib.pyplot as plt
import numpy as np
from qick import *
import time
from tqdm import tqdm_notebook as tqdm
import copy
import seaborn as sns
from copy import deepcopy

from qick.asm_v2 import QickSweep1D
from ...analysis import fitting as fitter
from ..general.qick_experiment import (
    QickExperiment,
    QickExperiment2DSimple,
    QickExperimentLoop,
)

from ..general.qick_program import QickProgram

from ...exp_handling.datamanagement import AttrDict

from .t1 import T1Program


class T1MultiProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        super()._initialize(cfg, readout="standard")

        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        self.add_loop("wait_loop", cfg.expt.expts)

        if cfg.expt.acStark:
            pulse = {
                "sigma": cfg.expt.wait_time,
                "sigma_inc": 0,
                "freq": cfg.expt.stark_freq,
                "gain": cfg.expt.stark_gain,
                "phase": 0,
                "type": "flat_len",
            }
            super().make_pulse(pulse, "stark_pulse")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # First, the T1 experiment
        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.acStark:
            self.delay_auto(t=0.01, tag="wait_stark")
            self.pulse(ch=self.qubit_ch, name="stark_pulse", t=0)
            self.delay_auto(t=0.01, tag="wait")
        else:
            self.delay_auto(t=cfg.expt["wait_time"] + 0.01, tag="wait")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
        )
        if cfg.expt.active_reset:
            self.reset(3)
        else:
            self.delay_auto(t=0.01, tag="wait_reset")
        # Then, the excited state

        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.acStark:
            self.delay_auto(t=0.01, tag="wait_stark")
            self.pulse(ch=self.qubit_ch, name="stark_pulse", t=0)
            self.delay_auto(t=0.2, tag="wait")
        else:
            self.delay_auto(t=cfg.expt["wait_time"] + 0.01, tag="wait")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
        )
        if cfg.expt.active_reset:
            self.reset(3)
        else:
            self.delay_auto(t=0.01, tag="wait_reset")

        # Then, the "ground" state

    def collect_shots(self, offset=0):
        return super().collect_shots(offset=0)

    def reset(self, i):
        super().reset(i)


class T1StarkExperiment(QickExperiment):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
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
        check_params=True
    ):

        if prefix is None:
            prefix = f"t1_stark_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, check_params=check_params)

        params_def = {
            "reps": 3 * self.reps,
            "rounds": self.rounds,
            "expts": 60,
            "start": 0.05,
            "span": 3.7 * self.cfg.device.qubit.T1[qi],
            "acStark": True,
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "stark_gain": 1,
            "end_wait": 0.5,
            "df": 70,
        }
        params = {**params_def, **params}
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 30

        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]

        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()
        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):
        self.param = {"label": "wait", "param": "t", "param_type": "time"}
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        super().acquire(T1Program, progress=progress)

        return self.data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        fitfunc = fitter.expfunc
        fitterfunc = fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data)

        return self.data

    def display(self, data=None, fit=True, plot_all=False, ax=None, show_hist=False):

        q = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[q]
        xlabel = "Wait Time ($\mu$s)"
        title = f"$T_1$ Stark Q{q} Freq: {df}, Amp: {self.cfg.expt.stark_gain}"
        fitfunc = fitter.expfunc
        caption_params = [
            {"index": 2, "format": "$T_1$ fit: {val:.3} $\pm$ {err:.2} $\mu$s"},
        ]

        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=xlabel,
            fit=fit,
            show_hist=show_hist,
            fitfunc=fitfunc,
            caption_params=caption_params,
        )


class T1StarkPowerExperiment(QickExperiment2DSimple):
    """
    Stark Power Rabi Experiment
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
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=False,
        style="",
        min_r2=None,
        max_err=None,
    ):

        if prefix == "":
            prefix = f"t1_stark_amp_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, qi=qi, prefix=prefix, progress=progress)

        params_def = {
            "end_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 20,
            "start_gain": 0.15,
            "end_wait": 0.5,
            "qubit": [qi],
        }
        self.expt = T1StarkExperiment(
            cfg_dict, qi=qi, go=False, params=params, style=style
        )
        params = {**params_def, **params}
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):

        self.cfg.expt["end_gain"] = np.min(
            [self.cfg.device.qubit.max_gain, self.cfg.expt["end_gain"]]
        )
        gainpts = np.linspace(
            self.cfg.expt["start_gain"],
            self.cfg.expt["end_gain"],
            self.cfg.expt["expts_gain"],
        )

        y_sweep = [{"var": "stark_gain", "pts": gainpts}]
        super().acquire(y_sweep=y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        fitterfunc = fitter.fitexp
        super().analyze(fitfunc=fitter.expfunc, fitterfunc=fitterfunc, data=data)

        data["offset"] = [
            data["fit_avgi"][i][0] for i in range(len(data["stark_gain_pts"]))
        ]
        data["amp"] = [
            data["fit_avgi"][i][1] for i in range(len(data["stark_gain_pts"]))
        ]
        data["t1"] = [
            data["fit_avgi"][i][2] for i in range(len(data["stark_gain_pts"]))
        ]

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data
        qubit = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[qubit]

        title = f"T1 Stark Power Q{qubit} Freq: {df}"
        ylabel = "Gain (DAC units)"
        xlabel = "Wait Time ($\mu$s)"
        super().display(plot_both=plot_both, title=title, xlabel=xlabel, ylabel=ylabel)

        fig, ax = plt.subplots(3, 1, figsize=(6, 8))

        if fit:
            ax[0].plot(data["stark_gain_pts"], data["offset"])
            ax[1].plot(data["stark_gain_pts"], data["amp"])
            ax[2].plot(data["stark_gain_pts"], data["t1"])

            ax[2].set_xlabel("Gain [DAC units]")
            ax[0].set_ylabel("Offset")
            ax[1].set_ylabel("Amplitude")
            ax[2].set_ylabel("T1")
            ax[0].set_title(f"T1 Stark Power Q{qubit} Freq: {df}")
            # print(f'Quadratic Fit: {data['quad_fit'][0]:.3g}x^2 + {data['quad_fit'][1]:.3g}x + {data['quad_fit'][2]:.3g}')
        sns.set_palette("coolwarm", len(data["stark_gain_pts"]))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for i in range(len(data["stark_gain_pts"])):
            ax.plot(
                data["xpts"], data["avgi"][i], linewidth=0.5
            )  # , label=f'Gain {data['stark_gain_pts'][i]}')

        imname = self.fname.split("\\")[-1]
        fig.savefig(
            self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "quad_fit.png"
        )
        plt.show()


class T1StarkFreqExperiment(QickExperiment2DSimple):
    """
    Stark Power Rabi Experiment
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
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=False,
        style="",
        min_r2=None,
        max_err=None,
    ):

        if prefix == "":
            prefix = f"t1_stark_freq_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, qi=qi, prefix=prefix, progress=progress)

        params_def = {
            "span_f": 200,
            "expts_f": 30,
            "start_df": 10,
            "end_wait": 0.5,
        }
        params = {**params_def, **params}
        params["start_f"] = self.cfg.device.qubit.f_ge[qi] + params["start_df"]
        self.expt = T1StarkExperiment(
            cfg_dict, qi=qi, go=False, params=params, style=style, check_params=False
        )
        params = {**params_def, **params}
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):

        freqpts = np.linspace(
            self.cfg.expt["start_f"],
            self.cfg.expt["start_f"] + self.cfg.expt["span_f"],
            self.cfg.expt["expts_f"],
        )

        y_sweep = [{"var": "stark_freq", "pts": freqpts}]
        super().acquire(y_sweep=y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        fitterfunc = fitter.fitexp
        super().analyze(fitfunc=fitter.expfunc, fitterfunc=fitterfunc, data=data)

        data["offset"] = [
            data["fit_avgi"][i][0] for i in range(len(data["stark_freq_pts"]))
        ]
        data["amp"] = [
            data["fit_avgi"][i][1] for i in range(len(data["stark_freq_pts"]))
        ]
        data["t1"] = [
            data["fit_avgi"][i][2] for i in range(len(data["stark_freq_pts"]))
        ]

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data
        qubit = self.cfg.expt.qubit[0]
        gain = self.cfg.expt.stark_gain

        title = f"T1 Stark Freq Q{qubit} Gain: {gain}"
        ylabel = "Frequency [MHz]"
        xlabel = "Wait Time ($\mu$s)"
        super().display(plot_both=False, title=title, xlabel=xlabel, ylabel=ylabel)

        fig, ax = plt.subplots(3, 1, figsize=(6, 8))

        if fit:
            ax[0].plot(data["stark_freq_pts"], data["offset"])
            ax[1].plot(data["stark_freq_pts"], data["amp"])
            ax[2].plot(data["stark_freq_pts"], data["t1"])

            ax[2].set_xlabel("Gain [DAC units]")
            ax[0].set_ylabel("Offset")
            ax[1].set_ylabel("Amplitude")
            ax[2].set_ylabel("T1")
            ax[0].set_title(f"T1 Stark Power Q{qubit} Gain: {gain}")
            # print(f'Quadratic Fit: {data['quad_fit'][0]:.3g}x^2 + {data['quad_fit'][1]:.3g}x + {data['quad_fit'][2]:.3g}')
        sns.set_palette("coolwarm", len(data["stark_freq_pts"]))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for i in range(len(data["stark_freq_pts"])):
            ax.plot(
                data["xpts"], data["avgi"][i], linewidth=0.5
            )  # , label=f'Gain {data['stark_gain_pts'][i]}')

        imname = self.fname.split("\\")[-1]
        fig.savefig(
            self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "quad_fit.png"
        )
        plt.show()


class T1StarkPowerSingle(QickExperiment):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
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
            prefix = f"t1_stark_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "reps": 10 * self.reps,
            "rounds": self.rounds,
            "expts": 200,
            "start": 1,
            "wait_time": self.cfg.device.qubit.T1[qi],
            "acStark": True,
            "active_reset": False,
            "qubit": [qi],
            "max_gain": 1,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "df": 70,
            "end_wait": 0.5,
        }
        params = {**params_def, **params}
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 30

        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]

        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()
        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):
        qi = self.cfg.expt.qubit[0]
        self.param = {"label": "stark_pulse", "param": "gain", "param_type": "pulse"}
        self.cfg.expt.stark_gain = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.max_gain
        )
        super().acquire(T1Program, progress=progress)
        data_t1 = deepcopy(self.data)
        self.cfg.expt.wait_time = 3.3 * self.cfg.device.qubit.T1[qi]
        self.cfg.expt.reps = int(4 * self.reps)
        data_g = super().acquire(T1Program, progress=progress)

        self.cfg.expt.wait_time = 0.025
        self.cfg.expt.reps = int(2.5 * self.reps)
        data_e = super().acquire(T1Program, progress=progress)

        data_types = ["avgi", "avgq", "amps", "phases"]
        for item in data_types:
            self.data[item + "_t1"] = data_t1[item]
            self.data[item + "_e"] = data_e[item]
            self.data[item + "_g"] = data_g[item]

        dv = self.data["avgi_e"] - self.data["avgi_g"]
        norm_data = (self.data["avgi_t1"] - self.data["avgi_g"]) / dv
        t1 = -1 / np.log(norm_data)
        self.data["t1"] = t1
        self.data["dv"] = dv

        return self.data

    def analyze(self, data=None, **kwargs):
        pass

    def display(self, data=None, fit=True, plot_all=False, ax=None, show_hist=True):
        if data is None:
            data = self.data

        q = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[q]
        xlabel = "Gain / Max Gain"
        title = (
            f"$T_1$ Stark Q{q} Freq: {df}, Delay Time: {self.cfg.expt.wait_time} $\mu$s"
        )

        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax = ax.flatten()
        ax[0].plot(data["xpts"], data["avgi_t1"])
        ax[1].plot(data["xpts"], data["avgi_e"])
        ax[2].plot(data["xpts"], data["avgi_g"])
        ax[3].set_xlabel(xlabel)
        ax[0].set_ylabel("I (ADC Units)")

        ax[3].plot(data["xpts"], data["t1"])
        ax[3].set_ylabel("$T_1$ / $T_{1,ave}$")
        fig.tight_layout()


class T1StarkPowerQuadSingle(QickExperimentLoop):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
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
        min_r2=None,
        max_err=None,
    ):

        if prefix is None:
            prefix = f"t1_stark_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "reps": 10 * self.reps,
            "rounds": self.rounds,
            "expts": 200,
            "start": 1,
            "wait_time": self.cfg.device.qubit.T1[qi],
            "acStark": True,
            "active_reset": False,
            "qubit": [qi],
            "max_gain": 1,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "df_pos": 70,
            "df_neg": -70,
            "stop_f": 20,
        }

        conf = self.cfg.stark
        params_def["quad_fit_pos"] = [conf.q[qi], conf.l[qi], conf.o[qi]]
        params_def["quad_fit_neg"] = [conf.qneg[qi], conf.lneg[qi], conf.oneg[qi]]
        params = {**params_def, **params}
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 30

        params_def["stark_freq_pos"] = self.cfg.device.qubit.f_ge[qi] + params["df_pos"]
        params_def["stark_freq_neg"] = self.cfg.device.qubit.f_ge[qi] + params["df_neg"]

        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()
        if go:
            super().run(
                display=display, progress=progress, min_r2=min_r2, max_err=max_err
            )

    def acquire(self, progress=False):
        qi = self.cfg.expt.qubit[0]
        self.param = {"label": "stark_pulse", "param": "gain", "param_type": "pulse"}

        f_pts_pos = np.linspace(0, self.cfg.expt.stop_f, int(self.cfg.expt.expts / 2))
        gain_pos = find_inverse_quad_fit(f_pts_pos, *self.cfg.expt.quad_fit_pos)
        f_pts_neg = np.linspace(-self.cfg.expt.stop_f, 0, int(self.cfg.expt.expts / 2))
        gain_neg = find_inverse_quad_fit(-f_pts_neg, *self.cfg.expt.quad_fit_neg)
        gain_pts = np.concatenate((gain_neg[0:-1], gain_pos))
        f_pts = np.concatenate((f_pts_neg[0:-1], f_pts_pos))
        m = len(f_pts_pos)  # Replace with the desired value of n
        n = len(f_pts_neg) - 1  # Replace with the desired value of m
        stark_freq = np.concatenate(
            (
                np.full(n, self.cfg.expt.stark_freq_neg),
                np.full(m, self.cfg.expt.stark_freq_pos),
            )
        )
        x_sweep = [
            {"var": "stark_gain", "pts": gain_pts},
            {"var": "stark_freq", "pts": stark_freq},
        ]
        self.cfg.expt.expts = 1
        super().acquire(T1Program, x_sweep, progress=progress)
        self.data["f_pts"] = f_pts

        return self.data

    def analyze(self, data=None, **kwargs):
        pass

    def display(self, data=None, fit=True, plot_all=False, ax=None, show_hist=False):
        if data is None:
            data = self.data

        q = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[q]
        xlabel = "Gain / Max Gain"
        title = (
            f"$T_1$ Stark Q{q} Freq: {df}, Delay Time: {self.cfg.expt.wait_time} $\mu$s"
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        # ax = ax.flatten()
        ax.plot(data["f_pts"], data["avgi"])
        # ax[1].plot(data["xpts"], data["avgi_e"])
        # ax[2].plot(data["xpts"], data["avgi_g"])
        # ax[3].set_xlabel(xlabel)
        ax.set_ylabel("I (ADC Units)")

        # ax[3].plot(data["xpts"], data['t1'])
        # ax[3].set_ylabel("$T_1$ / $T_{1,ave}$")
        fig.tight_layout()
        plt.show()

        if show_hist:  # Plot histogram of shots if show_hist is True
            fig2, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.plot(data["bin_centers"], data["hist"] / np.sum(data["hist"]), "o-")
            ax.set_xlabel("I [ADC units]")
            ax.set_ylabel("Probability")


class T1StarkPowerQuadMulti(QickExperimentLoop):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
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
            prefix = f"t1_stark_multi_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "reps": 5 * self.reps,
            "rounds": self.rounds,
            "expts": 200,
            "start": 1,
            "t1": self.cfg.device.qubit.T1[qi],
            "acStark": True,
            "active_reset": False,
            "qubit": [qi],
            "max_gain": 1,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "df_pos": 70,
            "df_neg": -70,
            "stop_f": 20,
        }

        conf = self.cfg.stark
        params_def["quad_fit_pos"] = [conf.q[qi], conf.l[qi], conf.o[qi]]
        params_def["quad_fit_neg"] = [conf.qneg[qi], conf.lneg[qi], conf.oneg[qi]]
        params = {**params_def, **params}
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 30

        params_def["stark_freq_pos"] = self.cfg.device.qubit.f_ge[qi] + params["df_pos"]
        params_def["stark_freq_neg"] = self.cfg.device.qubit.f_ge[qi] + params["df_neg"]

        self.cfg.expt = {**params_def, **params}
        self.cfg.expt.wait_times = [
            0.5 * self.cfg.device.qubit.T1[qi],
            self.cfg.device.qubit.T1[qi],
            1.25 * self.cfg.device.qubit.T1[qi],
        ]

        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()
        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):
        qi = self.cfg.expt.qubit[0]
        self.param = {"label": "stark_pulse", "param": "gain", "param_type": "pulse"}

        f_pts_pos = np.linspace(0, self.cfg.expt.stop_f, int(self.cfg.expt.expts / 2))
        gain_pos = find_inverse_quad_fit(f_pts_pos, *self.cfg.expt.quad_fit_pos)
        print(f"gain_pos: {gain_pos}, shape: {np.shape(gain_pos)}")
        f_pts_neg = np.linspace(
            -self.cfg.expt.stop_f, 0, int(self.cfg.expt.expts / 2)
        )
        gain_neg = find_inverse_quad_fit(-f_pts_neg, *self.cfg.expt.quad_fit_neg)
        gain_pts = np.concatenate((gain_neg[0:-1], gain_pos))
        f_pts = np.concatenate((f_pts_neg[0:-1], f_pts_pos))
        m = len(f_pts_pos)  # Replace with the desired value of n
        n = len(f_pts_neg) - 1  # Replace with the desired value of m
        stark_freq = np.concatenate(
            (
                np.full(n, self.cfg.expt.stark_freq_neg),
                np.full(m, self.cfg.expt.stark_freq_pos),
            )
        )
        x_sweep = [
            {"var": "stark_gain", "pts": gain_pts},
            {"var": "stark_freq", "pts": stark_freq},
        ]
        self.cfg.expt.expts = 1
        labs = ["avgi", "avgq"]
        save_data = {}
        for i, t in enumerate(self.cfg.expt.wait_times):
            self.cfg.expt.wait_time = t

            super().acquire(T1Program, x_sweep, progress=progress)
            for lab in labs:
                save_data[lab + "_" + str(i)] = self.data[lab]

        self.data["f_pts"] = f_pts
        self.data["wait_times"] = self.cfg.expt.wait_times
        self.data.update(save_data)

        return self.data

    def analyze(self, data=None, **kwargs):
        pass

    def display(self, data=None, fit=True, plot_all=False, ax=None, show_hist=False):
        if data is None:
            data = self.data

        q = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[q]
        xlabel = "Gain / Max Gain"
        title = (
            f"$T_1$ Stark Q{q} Freq: {df}, Delay Time: {self.cfg.expt.wait_time} $\mu$s"
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        # ax = ax.flatten()
        ax.plot(data["f_pts"], data["avgi"])
        # ax[1].plot(data["xpts"], data["avgi_e"])
        # ax[2].plot(data["xpts"], data["avgi_g"])
        # ax[3].set_xlabel(xlabel)
        ax.set_ylabel("I (ADC Units)")

        # ax[3].plot(data["xpts"], data['t1'])
        # ax[3].set_ylabel("$T_1$ / $T_{1,ave}$")
        fig.tight_layout()
        plt.show()

        if show_hist:  # Plot histogram of shots if show_hist is True
            fig2, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.plot(data["bin_centers"], data["hist"] / np.sum(data["hist"]), "o-")
            ax.set_xlabel("I [ADC units]")
            ax.set_ylabel("Probability")


class T1StarkPowerContTimeExperiment(QickExperiment2DSimple):
    """
    Stark Power Rabi Experiment
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
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=False,
        style="",
        min_r2=None,
        max_err=None,
    ):

        if prefix == "":
            prefix = f"t1_stark_amp_cont_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "count": 1000,
            "repsT1": 10 * self.reps,
            "repsE": 2 * self.reps,
            "repsG": self.reps,
            "rounds": self.rounds,
            "start_gain": 0,
            "stop_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 200,
            "df": 70,
            "delay_time": self.cfg.device.qubit.T1[qi],
            "start": 0,
            "acStark": True,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params = {**params_def, **params}
        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]

        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubit[0]
        self.update_config(q_ind)

        span_gain = self.cfg.expt.stop_gain - self.cfg.expt.start_gain
        coef = span_gain / np.sqrt(self.cfg.expt["expts_gain"])
        gainpts = self.cfg.expt["start_gain"] + coef * np.sqrt(
            np.arange(self.cfg.expt["expts_gain"])
        )
        data = {
            "xpts": [],
            "time": [],
            "avgi": [],
            "avgq": [],
            "amps": [],
            "phases": [],
            "avgi_e": [],
            "avgq_e": [],
            "amps_e": [],
            "phases_e": [],
            "avgi_g": [],
            "avgq_g": [],
            "amps_g": [],
            "phases_g": [],
        }

        self.cfg.T1expt = copy.deepcopy(self.cfg.expt)
        self.cfg.Eexpt = copy.deepcopy(self.cfg.expt)
        self.cfg.Gexpt = copy.deepcopy(self.cfg.expt)

        self.cfg.Eexpt.reps = self.cfg.expt.repsE
        self.cfg.T1expt.reps = self.cfg.expt.repsT1
        self.cfg.Gexpt.reps = self.cfg.expt.repsG

        self.cfg.Eexpt.length = 0
        self.cfg.Gexpt.length = 0
        self.cfg.T1expt.length = self.cfg.expt.delay_time
        self.cfg.Eexpt.acStark = False
        self.cfg.Gexpt.acStark = False
        for tm in tqdm(np.arange(self.cfg.expt.count)):
            data["time"].append(time.time())
            data["avgi_e"].append([])
            data["avgq_e"].append([])
            data["amps_e"].append([])
            data["phases_e"].append([])

            data["avgi_g"].append([])
            data["avgq_g"].append([])
            data["amps_g"].append([])
            data["phases_g"].append([])

            data["xpts"].append([])
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            for gain in gainpts:
                self.cfg.expt = copy.deepcopy(self.cfg.Gexpt)
                self.cfg.expt.do_exp = False
                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = t1.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                )
                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
                data["avgi_g"][-1].append(avgi)
                data["avgq_g"][-1].append(avgq)
                data["amps_g"][-1].append(amp)
                data["phases_g"][-1].append(phases)

                # Check excited state
                self.cfg.expt = copy.deepcopy(self.cfg.Eexpt)
                self.cfg.expt.do_exp = True

                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = t1.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                )

                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
                data["avgi_e"][-1].append(avgi)
                data["avgq_e"][-1].append(avgq)
                data["amps_e"][-1].append(amp)
                data["phases_e"][-1].append(phases)

                self.cfg.expt = copy.deepcopy(self.cfg.T1expt)
                self.cfg.expt.do_exp = True
                self.cfg.expt.stark_gain = gain

                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = t1.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                )

                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
                data["xpts"][-1].append(gain)
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phases)

        data["xpts"] = data["xpts"][0]
        for k, a in data.items():
            data[k] = np.array(a)

        data["ypts"] = (data["time"] - np.min(data["time"])) / 3600

        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        pass

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data

        dv = data["avgi_e"] - data["avgi_g"]
        norm_data = (data["avgi"] - data["avgi_g"]) / dv
        qubit = self.cfg.expt.qubit
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge
        y_sweep = data["ypts"]
        xlabel = "Gain Sq"
        ylabel = "Time (hrs)"
        title = f"$T_1$ Stark Q{qubit[0]} Freq: {df}"
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.pcolormesh(
            data["xpts"] ** 2 / np.max(data["xpts"] ** 2), y_sweep, norm_data, label="I"
        )
        plt.colorbar()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png")


class T1StarkPowerContTime(QickExperiment2DSimple):
    """
    Stark Power Rabi Experiment add ground state
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
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=False,
        style="",
        min_r2=None,
        max_err=None,
    ):

        if prefix == "":
            prefix = f"t1_stark_amp_cont_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "count": 1000,
            "repsT1": 10 * self.reps,
            "repsE": 2 * self.reps,
            "repsG": self.reps,
            "rounds": self.rounds,
            "expts_f": 200,
            "stop_f": 25,
            "quad_fit_pos": [3e-8, 3e-4, 0],
            "quad_fit_neg": [3e-8, 3e-4, 0],
            "df_pos": 70,
            "df_neg": -70,
            "delay_time": self.cfg.device.qubit.T1[qi],
            "start": 0,
            "acStark": True,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        self.cfg.expt = {**params_def, **params}
        self.cfg.expt["stark_freq_pos"] = (
            self.cfg.device.qubit.f_ge[qi] + params["df_pos"]
        )
        self.cfg.expt["stark_freq_neg"] = (
            self.cfg.device.qubit.f_ge[qi] + params["df_neg"]
        )

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        q_ind = self.cfg.expt.qubit[0]
        self.update_config(q_ind)

        f_pts_pos = np.linspace(0, self.cfg.expt.stop_f, int(self.cfg.expt.expts_f / 2))
        gain_pos = find_inverse_quad_fit(f_pts_pos, *self.cfg.expt.quad_fit_pos)
        f_pts_neg = np.linspace(
            -self.cfg.expt.stop_f, 0, int(self.cfg.expt.expts_f / 2)
        )
        gain_neg = find_inverse_quad_fit(-f_pts_neg, *self.cfg.expt.quad_fit_neg)
        gain_pts = np.concatenate((gain_neg[0:-1], gain_pos))
        f_pts = np.concatenate((f_pts_neg[0:-1], f_pts_pos))

        data = {
            "xpts": [],
            "time": [],
            "avgi": [],
            "avgq": [],
            "amps": [],
            "phases": [],
            "avgi_e": [],
            "avgq_e": [],
            "amps_e": [],
            "phases_e": [],
            "avgi_g": [],
            "avgq_g": [],
            "amps_g": [],
            "phases_g": [],
        }

        self.cfg.T1expt = copy.deepcopy(self.cfg.expt)
        self.cfg.Eexpt = copy.deepcopy(self.cfg.expt)
        self.cfg.Gexpt = copy.deepcopy(self.cfg.expt)

        self.cfg.Eexpt.reps = self.cfg.expt.repsE
        self.cfg.T1expt.reps = self.cfg.expt.repsT1
        self.cfg.Gexpt.reps = self.cfg.expt.repsG

        self.cfg.Eexpt.length = 0
        self.cfg.Gexpt.length = 0
        self.cfg.T1expt.length = self.cfg.expt.delay_time
        self.cfg.Eexpt.acStark = False
        self.cfg.Gexpt.acStark = False
        for tm in tqdm(np.arange(self.cfg.expt.count)):
            self.cfg.expt = copy.deepcopy(self.cfg.Gexpt)
            self.cfg.expt.do_exp = False
            t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = t1.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
            )
            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
            phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
            data["avgi_g"].append(avgi)
            data["avgq_g"].append(avgq)
            data["amps_g"].append(amp)
            data["phases_g"].append(phases)

            # Check excited state
            self.cfg.expt = copy.deepcopy(self.cfg.Eexpt)
            self.cfg.expt.do_exp = True

            t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
            avgi, avgq = t1.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=False,
            )

            avgi = avgi[0][0]
            avgq = avgq[0][0]
            amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
            phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
            data["avgi_e"].append(avgi)
            data["avgq_e"].append(avgq)
            data["amps_e"].append(amp)
            data["phases_e"].append(phases)

            self.cfg.expt = copy.deepcopy(self.cfg.T1expt)
            self.cfg.expt.do_exp = True

            data["time"].append(time.time())
            data["xpts"].append([])
            data["avgi"].append([])
            data["avgq"].append([])
            data["amps"].append([])
            data["phases"].append([])
            for i in range(len(gain_pts)):
                if f_pts[i] < 0:
                    self.cfg.expt.stark_freq = self.cfg.expt.stark_freq_neg
                else:
                    self.cfg.expt.stark_freq = self.cfg.expt.stark_freq_pos

                self.cfg.expt.stark_gain = gain_pts[i]
                t1 = T1StarkProgram(soccfg=self.soccfg, cfg=self.cfg)
                avgi, avgq = t1.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    load_pulses=True,
                    progress=False,
                )

                avgi = avgi[0][0]
                avgq = avgq[0][0]
                amp = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
                phases = np.angle(avgi + 1j * avgq)  # Calculating the phase
                data["xpts"][-1].append(gain_pts[i])
                data["avgi"][-1].append(avgi)
                data["avgq"][-1].append(avgq)
                data["amps"][-1].append(amp)
                data["phases"][-1].append(phases)

        data["xpts"] = data["xpts"][0]
        for k, a in data.items():
            data[k] = np.array(a)
        data["fpts"] = f_pts
        data["ypts"] = (data["time"] - np.min(data["time"])) / 3600

        self.data = data
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        pass

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data

        dv = data["avgi_e"] - data["avgi_g"]
        norm_data = (data["avgi"] - data["avgi_g"][:, np.newaxis]) / dv[:, np.newaxis]
        qubit = self.cfg.expt.qubit
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge
        y_sweep = data["ypts"]
        xlabel = "Gain Sq"
        ylabel = "Time (hrs)"
        title = f"$T_1$ Stark Q{qubit[0]} Freq: {df}"
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        plt.pcolormesh(data["fpts"], y_sweep, norm_data, label="I")
        plt.colorbar()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        plt.tight_layout()
        plt.show()

        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png")


def find_inverse_quad_fit(y, a, b, c):
    rt = []
    for yt in y:
        if yt == 0:
            rt.append(0.0)
            continue
        # Solving the quadratic equation a*x^2 + b*x + (c - y) = 0
        discriminant = b**2 - 4 * a * (c - yt)
        #print("discriminant: ",discriminant)
        if discriminant < 0:
            print("discriminant < 0, no real roots")
            print("yt: ", yt, " a: ", a, " b: ", b, " c: ", c)
            return None  # No real roots
        elif discriminant == 0:
            return -b / (2 * a)  # One real root
        else:
            root1 = (-b + np.sqrt(discriminant)) / (2 * a)
            root2 = (-b - np.sqrt(discriminant)) / (2 * a)
        rt.append(root1)
    return rt
