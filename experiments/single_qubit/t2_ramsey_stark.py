import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from qick import *
from qick.asm_v2 import QickSweep1D

from ..general.qick_experiment import QickExperiment2DSimple, QickExperiment
from .t2 import T2Program
from ...analysis import fitting as fitter


class RamseyStarkExperiment(QickExperiment):
    """
    Ramsey experiment
    Experimental Config:
    expt = dict(
        start: wait time start sweep [us]
        step: wait time step - make sure nyquist freq = 0.5 * (1/step) > ramsey (signal) freq!
        expts: number experiments stepping from start
        ramsey_freq: frequency by which to advance phase [MHz]
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
        checkEF: does ramsey on the EF transition instead of ge
        qubits: 
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
            prefix = f"ramsey_stark_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, check_params=check_params)

        params_def = {
            "expts": 200,
            "reps": 2 * self.reps,
            "rounds": 2 * self.rounds,
            "start": 0.1,
            "ramsey_freq": "smart",
            "stark_gain": 0.5,
            "step": cfg_dict['soc'].cycles2us(1)+0.001,
            "df": 70,
            "acStark": True,
            "checkEF": False,
            'active_reset': self.cfg.device.readout.active_reset[qi],
            'experiment_type': 'ramsey',
            "num_pi":0, 
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params = {**params_def, **params}
        if params["checkEF"]:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ef
            params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
        else:
            cfg_qub = self.cfg.device.qubit.pulses.pi_ge
            params_def["freq"] = self.cfg.device.qubit.f_ge[qi]
        for key in cfg_qub:
            params_def[key] = cfg_qub[key][qi]
        if params["ramsey_freq"] == "smart":
            params["ramsey_freq"] = np.pi / 2 / self.cfg.device.qubit.T2r[qi]

        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 30
        params = {**params_def, **params}
        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]
        self.cfg.expt = params

        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):
        self.param = {"label": "waiting", "param": "t", "param_type": "time"}
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop",
            self.cfg.expt.start,
            self.cfg.expt.start + self.cfg.expt.step * self.cfg.expt.expts,
        )

        data = super().acquire(T2Program, progress=progress)
        return data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data
        fitterfunc = fitter.fitdecaysin
        fitfunc = fitter.decaysin
        if fit:
            super().analyze(fitfunc=fitfunc, fitterfunc=fitterfunc, data=data)

        return self.data

    def display(
        self,
        data=None,
        fit=True,
        debug=False,
        plot_all=False,
        ax=None,
        show_hist=True,
        **kwargs,
    ):
        qubit = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[qubit]
        title = (
            f"$T_2$ Ramsey Stark Q{qubit} Freq: {df}, Amp: {self.cfg.expt.stark_gain}"
        )
        xlabel = "Wait Time ($\mu$s)"

        fitfunc = fitter.decaysin

        caption_params = [
            {"index": 3, "format": "$T_2$ Ramsey : {val:.4} $\pm$ {err:.2g} $\mu$s"},
            {"index": 1, "format": "Freq. : {val:.3} $\pm$ {err:.1} MHz"},
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

        return data


class RamseyStarkPowerExperiment(QickExperiment2DSimple):
    """
    Initialize the T2 Ramsey Stark experiment.
    self.cfg.expt:
        start (float): Wait time tau in microseconds.
        step (float): Step size in microseconds.
        expts (int): Number of experiments.
        start_gain (int): Starting gain value.
        end_gain (int): Ending gain value.
        expts_gain (int): Gain value for experiments.
        ramsey_freq (float): Ramsey frequency in MHz.
        reps (int): Number of repetitions.
        rounds (int): Number of rounds.
        qubit (list): List containing the qubit index.
        stark_freq (float): Stark frequency.
        checkEF (bool): Flag to check EF interaction.
        qubit_chan (int): Qubit channel for readout.
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
            prefix = f"ramsey_stark_amp_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, qi=qi, prefix=prefix, progress=progress)

        params_def = {
            "end_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 20,
            "start_gain": 0.15,
            "qubit": [qi],
        }
        exp_name = RamseyStarkExperiment
        self.expt = exp_name(cfg_dict, qi=qi, go=False, params=params, check_params=False)
        params = {**params_def, **params}
        params.end_gain = np.min(
            [self.cfg.device.qubit.max_gain, params.end_gain]
        )
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        if go:
            super().run(min_r2=min_r2, max_err=max_err, progress=progress)

    def acquire(self, progress=False):
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

        fitterfunc = fitter.fitsin
        super().analyze(fitfunc=fitter.decaysin, fitterfunc=fitterfunc, data=data)

        freq = [data["fit_avgi"][i][1] for i in range(len(data["stark_gain_pts"]))]
        popt, pcov = curve_fit(quad_fit, data["stark_gain_pts"], freq)
        data["quad_fit"] = popt

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data
        qubit = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[qubit]

        title = f"Stark Power Ramsey Q{qubit} Freq: {df}"
        ylabel = "Gain [DAC units]"
        xlabel = "Wait Time ($\mu$s)"
        super().display(plot_both=False, title=title, xlabel=xlabel, ylabel=ylabel)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax = [ax]
        if fit:
            freq = [data["fit_avgi"][i][1] for i in range(len(data["stark_gain_pts"]))]
            ax[0].plot(data["stark_gain_pts"], freq, "o")

            ax[0].plot(
                data["stark_gain_pts"],
                quad_fit(data["stark_gain_pts"], *data["quad_fit"]),
                label="Fit: {:.3g}$x^2$+{:.3g}$x$+{:.3g}".format(*data["quad_fit"]),
            )
            ax[0].set_xlabel("Gain [DAC units]")
            ax[0].set_ylabel("Frequency [MHz]")
            ax[0].legend()
            ax[0].set_title(f"Stark Power Ramsey Q{qubit} Freq: {df}")
            # print(f'Quadratic Fit: {data['quad_fit'][0]:.3g}x^2 + {data['quad_fit'][1]:.3g}x + {data['quad_fit'][2]:.3g}')

        # Plot raw data
        fig3, ax = plt.subplots(1, 1, figsize=(6, 8))
        off = 0
        for i in range(len(data["stark_gain_pts"])):

            ax.plot(
                data["xpts"], data["avgi"][i] + off
            )  # , label=f'Gain {data['stark_gain_pts'][i]}')
            off += 2 * data["fit_avgi"][i][0]

        imname = self.fname.split("\\")[-1]
        fig.savefig(
            self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "quad_fit.png"
        )
        plt.show()


class RamseyStarkFreqExperiment(QickExperiment2DSimple):
    """
    Initialize the T2 Ramsey Stark experiment.
    self.cfg.expt:
        start (float): Wait time tau in microseconds.
        step (float): Step size in microseconds.
        expts (int): Number of experiments.
        start_gain (int): Starting gain value.
        end_gain (int): Ending gain value.
        expts_gain (int): Gain value for experiments.
        ramsey_freq (float): Ramsey frequency in MHz.
        reps (int): Number of repetitions.
        rounds (int): Number of rounds.
        qubit (list): List containing the qubit index.
        stark_freq (float): Stark frequency.
        checkZZ (bool): Flag to check ZZ interaction.
        checkEF (bool): Flag to check EF interaction.
        qubit_chan (int): Qubit channel for readout.
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
            prefix = f"ramsey_stark_freq_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, qi=qi, prefix=prefix, progress=progress)

        exp_name = RamseyStarkExperiment

        params_def = {
            "end_df": 200,
            "expts_df": 20,
            "start_df": 5,
            "qubit": [qi],
        }
        self.expt = exp_name(cfg_dict, qi=qi, go=False, params=params, check_params=False)
        params = {**params_def, **params}
        params["start_freq"] = self.cfg.device.qubit.f_ge[qi] + params["start_df"]
        params["end_freq"] = self.cfg.device.qubit.f_ge[qi] + params["end_df"]
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):

        freq_pts = np.linspace(
            self.cfg.expt["start_freq"],
            self.cfg.expt["end_freq"],
            self.cfg.expt["expts_df"],
        )

        y_sweep = [{"var": "stark_freq", "pts": freq_pts}]
        super().acquire(y_sweep=y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        fitterfunc = fitter.fitsin
        super().analyze(fitfunc=fitter.sinfunc, fitterfunc=fitterfunc, data=data)
        self.data["freq"] = [data["fit_avgi"][i][1] for i in range(len(data["ypts"]))]

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data
        qubit = self.cfg.expt.qubit[0]
        gain = self.cfg.expt.stark_gain

        title = f"Stark Freq Ramsey Q{qubit} Gain: {gain}"
        ylabel = "Frequency (MHz)"
        xlabel = "Wait Time ($\mu$s)"
        super().display(plot_both=False, title=title, xlabel=xlabel, ylabel=ylabel)

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax = [ax]
        if fit:
            self.data["freq"] = [
                data["fit_avgi"][i][1] for i in range(len(data["ypts"]))
            ]
            ax[0].plot(data["ypts"], self.data["freq"], "o")
            alpha = (
                self.cfg.device.qubit.f_ge[qubit] - self.cfg.device.qubit.f_ef[qubit]
            )
            df = data["ypts"] - self.cfg.device.qubit.f_ge[qubit]
            df2 = data["ypts"] - self.cfg.device.qubit.f_ef[qubit]
            exp_val = np.abs(alpha * gain**2 / df / df2) * 500
            inds = exp_val < 30
            ax[0].plot(data["ypts"][inds], exp_val[inds], ".")

        # Plot raw data
        fig3, ax = plt.subplots(1, 1, figsize=(6, 8))
        off = 0
        for i in range(len(data["ypts"])):

            ax.plot(
                data["xpts"], data["avgi"][i] + off
            )  # , label=f'Gain {data['stark_gain_pts'][i]}')
            off += 2 * data["fit_avgi"][i][0]

        # imname = self.fname.split("\\")[-1]
        # fig.savefig(
        #     self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "quad_fit.png"
        # )
        # plt.show()


def quad_fit(x, a, b, c):
    return a * x**2 + b * x + c
