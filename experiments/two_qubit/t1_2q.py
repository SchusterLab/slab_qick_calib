import numpy as np
from qick import *

from ...exp_handling.datamanagement import AttrDict
from datetime import datetime
from ...analysis import fitting as fitter
from ..general.qick_experiment_2q import QickExperiment2Q
from ..general.qick_program import QickProgram2Q
from qick.asm_v2 import QickSweep1D


class T1_2Q_Program(QickProgram2Q):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        super()._initialize(cfg, readout="standard")
        for i, q in enumerate(cfg.expt.qubit):
            super().make_cfg_pulse(cfg.expt.qubit[i], i, cfg.device.qubit.f_ge, "pi_ge")

            if cfg.expt.acStark:
                pulse = {
                    "sigma": cfg.expt[f"wait_time_stark_{i}"],
                    "sigma_inc": 0,
                    "freq": cfg.expt.stark_freq[i],
                    "gain": cfg.expt.stark_gain[i],
                    "phase": 0,
                    "type": "const",
                }
                super().make_pulse(pulse, f"stark_pulse_{i}")

        self.add_loop("wait_loop", cfg.expt.expts)

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        for q in range(len(cfg.expt.qubit)):
            self.send_readoutconfig(ch=self.adc_ch[q], name=f"readout_{q}", t=0)
        if cfg.expt.span[0] > cfg.expt.span[1]:
            q_order = [0, 1]
        else:
            q_order = [1, 0]
        # check if delay vs delay_auto handle pulse time differently -- I think they should be different
        q = q_order[0]
        self.pulse(ch=self.qubit_ch[q], name=f"pi_ge_{q}", t=0)

        if cfg.expt.acStark:
            q = q_order[0]
            self.delay(
                t=cfg.device.qubit.pulses.pi_ge.sigma[cfg.expt.qubit[q]]
                * cfg.device.qubit.pulses.pi_ge.sigma_inc[cfg.expt.qubit[q]],
                tag=f"wait_stark_{q}",
            )
            self.pulse(ch=self.qubit_ch, name=f"stark_pulse_{q}", t=0)  # length is T1 0
            self.delay(t=cfg.expt[f"wait_time_{q}"], tag=f"wait_{q}")
            q = q_order[1]
            self.pulse(ch=self.qubit_ch[q], name=f"pi_ge_{q}", t=0)
            self.delay(
                t=cfg.device.qubit.pulses.pi_ge.sigma[cfg.expt.qubit[q]]
                * cfg.device.qubit.pulses.pi_ge.sigma_inc[cfg.expt.qubit[q]],
                tag=f"wait_stark_{q}",
            )
            self.pulse(ch=self.qubit_ch, name=f"stark_pulse_{q}", t=0)  # length is T1 0
            self.delay(t=cfg.expt[f"wait_time_{q}"], tag=f"wait_{q}")
        else:
            self.delay(t=cfg.expt[f"wait_time_{q}"] + 0.01, tag=f"wait_{q}")
            q = q_order[1]
            self.pulse(ch=self.qubit_ch[q], name=f"pi_ge_{q}", t=0)
            self.delay_auto(t=cfg.expt[f"wait_time_{q}"] + 0.01, tag=f"wait_{q}")

        for q in q_order:
            self.pulse(ch=self.res_ch[q], name=f"readout_pulse_{q}", t=0)
            if self.lo_ch[q] is not None:
                self.pulse(ch=self.lo_ch[q], name=f"mix_pulse_{q}", t=0.0)
            self.trigger(ros=[self.adc_ch[q]], pins=[0], t=self.trig_offset[q])
        if cfg.expt.active_reset:
            self.reset(3)

    def collect_shots(self, offset=[0, 0]):
        return super().collect_shots(offset=offset)

    def reset(self, i):
        super().reset(i)


class T1_2Q(QickExperiment2Q):
    """
    self.cfg.expt: dict
        A dictionary containing the configuration parameters for the T1 experiment. The keys and their descriptions are as follows:
        - span (float): The total span of the wait time sweep in microseconds.
        - expts (int): The number of experiments to be performed.
        - reps (int): The number of repetitions for each experiment (inner loop)
        - rounds (int): The number of rounds for the experiment (outer loop)
        - qubit (int): The index of the qubit being used in the experiment.
        - qubit_chan (int): The channel of the qubit being read out.
    """

    def __init__(
        self,
        cfg_dict,
        qi=[0, 1],
        go=True,
        params={},
        prefix=None,
        progress=None,
        style="",
        min_r2=None,
        max_err=None,
    ):

        if prefix is None:
            prefix = "t1"
            for q in qi:
                prefix += f"_Q{q}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "reps": 2 * self.reps,
            "rounds": self.rounds,
            "expts": 60,
            "start": 1,
            "span": [3.7 * self.cfg.device.qubit.T1[q] for q in qi],
            "acStark": False,
            "active_reset": [self.cfg.device.readout.active_reset[q] for q in qi],
            "qubit": qi,
            "qubit_chan": [self.cfg.hw.soc.adcs.readout.ch[q] for q in qi],
        }
        params_def["active_reset"] = np.all(params_def["active_reset"])
        # We assume the first T1 is longer
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 30

        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()
        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False, debug=False):
        self.param = []
        for i in range(len(self.cfg.expt.qubit)):
            self.param.append(
                {"label": f"wait_{i}", "param": "t", "param_type": "time"}
            )
        span_diff = self.cfg.expt.span[0] - self.cfg.expt.span[1]
        # self.cfg.device.qubit.pulses.pi_ge.sigma[self.cfg.expt.qubit[1]]*self.cfg.device.qubit.pulses.pi_ge.sigma_inc[self.cfg.expt.qubit[1]]
        if span_diff > 0:
            pulse_wait_time = (
                self.cfg.device.qubit.pulses.pi_ge.sigma[self.cfg.expt.qubit[1]]
                * self.cfg.device.qubit.pulses.pi_ge.sigma_inc[self.cfg.expt.qubit[1]]
            )
            self.cfg.expt.wait_time_0 = QickSweep1D(
                "wait_loop",
                self.cfg.expt.start,
                self.cfg.expt.start + np.absolute(span_diff) - pulse_wait_time,
            )
            self.cfg.expt.wait_time_1 = QickSweep1D(
                "wait_loop",
                self.cfg.expt.start,
                self.cfg.expt.start + self.cfg.expt.span[1],
            )
        else:
            pulse_wait_time = (
                self.cfg.device.qubit.pulses.pi_ge.sigma[self.cfg.expt.qubit[0]]
                * self.cfg.device.qubit.pulses.pi_ge.sigma_inc[self.cfg.expt.qubit[0]]
            )
            self.cfg.expt.wait_time_1 = QickSweep1D(
                "wait_loop",
                self.cfg.expt.start,
                self.cfg.expt.start + np.absolute(span_diff) - pulse_wait_time,
            )
            self.cfg.expt.wait_time_0 = QickSweep1D(
                "wait_loop",
                self.cfg.expt.start,
                self.cfg.expt.start + self.cfg.expt.span[0],
            )
        data = super().acquire(T1_2Q_Program, progress=progress)
        if span_diff > 0:
            data["xpts"][0] = (
                data["xpts"][0]
                + data["xpts"][1]
                + self.cfg.device.qubit.pulses.pi_ge.sigma[self.cfg.expt.qubit[1]]
                * self.cfg.device.qubit.pulses.pi_ge.sigma_inc[self.cfg.expt.qubit[1]]
            )
        else:
            data["xpts"][1] = (
                data["xpts"][0]
                + data["xpts"][1]
                + self.cfg.device.qubit.pulses.pi_ge.sigma[self.cfg.expt.qubit[0]]
                * self.cfg.device.qubit.pulses.pi_ge.sigma_inc[self.cfg.expt.qubit[0]]
            )

        return self.data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        # fitparams=[y-offset, amp, x-offset, decay rate]
        fitfunc = fitter.expfunc
        fitterfunc = fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data, **kwargs)
        # data["new_t1"] = data["best_fit"][2]
        # data["new_t1_i"] = data["fit_avgi"][2]
        return data

    def display(
        self, data=None, fit=True, plot_all=False, ax=None, show_hist=True, **kwargs
    ):

        title = [f"$T_1$ Q{q}" for q in self.cfg.expt.qubit]
        xlabel = "Wait Time ($\mu$s)"

        caption_params = [
            {"index": 2, "format": "$T_1$ fit: {val:.3} $\pm$ {err:.2} $\mu$s"},
        ]
        fitfunc = fitter.expfunc

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
            **kwargs,
        )


class T1_2Q_Continuous(QickExperiment2Q):
    """
    T1 Continuous
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
        soccfg=None,
        path="",
        prefix="T1Continuous",
        config_file=None,
        progress=None,
    ):
        super().__init__(
            soccfg=soccfg,
            path=path,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
        )

    def acquire(self, progress=False, debug=False):

        self.update_config(q_ind=self.cfg.expt.qubit)
        t1 = T1Program(soccfg=self.soccfg, cfg=self.cfg)
        x_pts, avgi, avgq = t1.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            load_pulses=True,
            progress=progress,
            debug=debug,
        )

        shots_i, shots_q = t1.collect_shots()

        avgi = avgi[0][0]
        avgq = avgq[0][0]
        amps = np.abs(avgi + 1j * avgq)  # Calculating the magnitude
        phases = np.angle(avgi + 1j * avgq)  # Calculating the phase

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        data = {
            "xpts": x_pts,
            "avgi": avgi,
            "avgq": avgq,
            "amps": amps,
            "phases": phases,
            "time": current_time,
            "raw_i": shots_i,
            "raw_q": shots_q,
            "raw_amps": np.abs(shots_i + 1j * shots_q),
        }

        self.data = data
        return data

    def analyze(self, data=None, **kwargs):
        pass

    def display(self, data=None, fit=True, show=False, **kwargs):
        pass
