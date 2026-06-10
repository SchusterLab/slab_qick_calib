"""
Shot-resolved continuous two-qubit T1 measurement (test version).

Each program repetition interleaves n_g ground, n_e excited, and n_t1
T1-wait measurements on both qubits, giving per-shot calibration data.
"""

import numpy as np
from pathlib import Path
from qick import *

from ...exp_handling.datamanagement import AttrDict
from datetime import datetime
from ...analysis import fitting as fitter
from ..general.qick_experiment_2q import QickExperiment2Q
from ..general.qick_program import QickProgram2Q
from qick.asm_v2 import QickSweep1D
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


class T1Cont2QProgram(QickProgram2Q):
    """Pulse sequence interleaving ground, excited, and T1-wait measurements on two qubits."""

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """Set up readout and pi pulses for both qubits, looping over shots."""
        cfg = AttrDict(self.cfg)
        self.add_loop("shot_loop", cfg.expt.shots)

        super()._initialize(cfg, readout="standard")
        for i, q in enumerate(cfg.expt.qubit):
            super().make_cfg_pulse(cfg.expt.qubit[i], i, cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        """Play n_g ground, n_e excited, and n_t1 T1-wait measurement blocks."""
        cfg = AttrDict(self.cfg)

        # Configure readout for both qubits
        for q in range(len(cfg.expt.qubit)):
            self.send_readoutconfig(ch=self.adc_ch[q], name=f"readout_{q}", t=0)

        # Ground state measurements
        self.delay_auto(t=0.01, tag="readout0_delay_1")
        for i in range(cfg.expt.n_g):
            self.measure(cfg)
        self.delay_auto(t=cfg.expt["readout"] + 0.01, tag=f"readout_delay_1_{i}")
        self.delay_auto(t=cfg.expt["final_delay"] + 0.01, tag=f"final_delay_g_{i}")

        # Excited state measurements
        if cfg.expt.pulse_length[0] > cfg.expt.pulse_length[1]:
            q_order = [0, 1]
        else:
            q_order = [1, 0]
        for i in range(cfg.expt.n_e):
            q = q_order[0]
            self.pulse(ch=self.qubit_ch[q], name=f"pi_ge_{q}", t=0)
            self.delay(
                t=np.absolute(cfg.expt.pulse_length[0] - cfg.expt.pulse_length[1]),
                tag=f"wait0_{i}",
            )
            q = q_order[1]
            self.pulse(ch=self.qubit_ch[q], name=f"pi_ge_{q}", t=0)
            self.delay_auto(0.01, tag=f"readout_ge_delay_0_{i}")
            self.measure(cfg)
            if cfg.expt.active_reset:
                self.reset(3, 0, i)
                self.delay_auto(t=cfg.expt["readout"] + 0.01, tag=f"final_delay_0_{i}")
            else:
                self.delay_auto(
                    t=cfg.expt["final_delay"] + 0.01, tag=f"final_delay_1_{i}"
                )

        # T1 measurements
        if cfg.expt.span[0] > cfg.expt.span[1]:
            q_order = [0, 1]
        else:
            q_order = [1, 0]

        for i in range(cfg.expt.n_t1):
            q = q_order[0]
            self.pulse(ch=self.qubit_ch[q], name=f"pi_ge_{q}", t=0)
            q = q_order[1]
            self.delay_auto(t=cfg.expt[f"wait_time_{q}"] + 0.01, tag=f"wait_{i}")
            self.pulse(ch=self.qubit_ch[q], name=f"pi_ge_{q}", t=0)
            self.delay(t=cfg.expt.span[q], tag=f"wait_t1{q}_{i}")
            self.measure(cfg)
            if cfg.expt.active_reset:
                self.reset(3, 1, i)
                self.delay_auto(t=cfg.expt["readout"] + 0.01, tag=f"final_delay_{i}")
            else:
                self.delay_auto(
                    t=cfg.expt["final_delay"] + 0.01, tag=f"final_delay_{i}"
                )

    def measure(self, cfg):
        """Play both readout pulses and trigger both ADCs."""
        for q in range(len(cfg.expt.qubit)):
            self.pulse(ch=self.res_ch[q], name=f"readout_pulse_{q}", t=0)
            if self.lo_ch[q] is not None:
                self.pulse(ch=self.lo_ch[q], name=f"mix_pulse_{q}", t=0.0)
            self.trigger(ros=[self.adc_ch[q]], pins=[0], t=self.trig_offset[q])


class T1Cont2QExperiment(QickExperiment2Q):
    """Continuously measure T1 on two qubits with inline g/e calibration shots."""

    def __init__(
        self,
        cfg_dict,
        qi=[0, 1],
        go=True,
        params={},
        prefix=None,
        progress=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        display=True,
    ):
        if prefix is None:
            prefix = "t1_cont"
            for q in qi:
                prefix += f"_Q{q}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "shots": 50000,
            "reps": 1,
            "rounds": self.rounds,
            "wait_time": max([self.cfg.device.qubit.T1[q] for q in qi]),
            "span": [self.cfg.device.qubit.T1[q] for q in qi],
            "active_reset": np.all(
                [self.cfg.device.readout.active_reset[q] for q in qi]
            ),
            "final_delay": max([self.cfg.device.qubit.T1[q] for q in qi]) * 6,
            "readout": max([self.cfg.device.readout.readout_length[q] for q in qi]),
            "n_g": 1,
            "n_e": 2,
            "n_t1": 7,
            "qubit": qi,
            "qubit_chan": [self.cfg.hw.soc.adcs.readout.ch[q] for q in qi],
            "pulse_length": [
                self.cfg.device.qubit.pulses.pi_ge.sigma[q]
                * self.cfg.device.qubit.pulses.pi_ge.sigma_inc[q]
                for q in qi
            ],
            "wait_time_0": 0,
            "wait_time_1": 0,
        }

        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()

        if go:
            super().run(
                display=display,
                progress=progress,
                min_r2=min_r2,
                max_err=max_err,
            )

    def acquire(self, progress=False, get_hist=True):
        """Acquire shot-resolved data for the interleaved measurement blocks."""
        self.param = {"label": "wait_0", "param": "t", "param_type": "time"}

        if "active_reset" in self.cfg.expt and self.cfg.expt.active_reset:
            final_delay = max(
                [self.cfg.device.readout.readout_length[q] for q in self.cfg.expt.qubit]
            )
        else:
            final_delay = max(
                [self.cfg.device.readout.final_delay[q] for q in self.cfg.expt.qubit]
            )

        prog = T1Cont2QProgram(
            soccfg=self.soccfg, final_delay=final_delay, cfg=self.cfg
        )

        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        # pass over the pulse length for each qubit
        # pulse_length0 = self.cfg.device.qubit.pulses.pi_ge.sigma[self.cfg.expt.qubit[0]]*self.cfg.device.qubit.pulses.pi_ge.sigma_inc[self.cfg.expt.qubit[0]]
        # pulse_length1 = self.cfg.device.qubit.pulses.pi_ge.sigma[self.cfg.expt.qubit[1]]*self.cfg.device.qubit.pulses.pi_ge.sigma_inc[self.cfg.expt.qubit[1]]
        # self.cfg.expt.pulse_length_0 = pulse_length0
        # self.cfg.expt.pulse_length_1 = pulse_length1

        # pass over the wait time for each qubit for t1 measurements
        span_diff = self.cfg.expt.span[0] - self.cfg.expt.span[1]
        if span_diff > 0:
            self.cfg.expt.wait_time_0 = 0
            self.cfg.expt.wait_time_1 = (
                np.absolute(span_diff) - self.cfg.expt.pulse_length[1]
            )
        else:
            self.cfg.expt.wait_time_0 = (
                np.absolute(span_diff) - self.cfg.expt.pulse_length[0]
            )
            self.cfg.expt.wait_time_1 = 0

        iq_list = prog.acquire(
            self.im[self.cfg.aliases.soc],
            rounds=self.cfg.expt.rounds,
            threshold=None,
            load_pulses=True,
            progress=progress,
        )
        xpts = self.get_params(prog)

        if get_hist:
            v, hist = self.make_hist(prog)

        data = {
            "xpts": xpts,
            "start_time": current_time,
        }

        nms = ["g", "e", "t1"]
        if self.cfg.expt.active_reset:
            start_ind = [
                0,
                self.cfg.expt.n_g,
                self.cfg.expt.n_g + self.cfg.expt.n_e * 3,
                len(iq_list[0]),
            ]
        else:
            start_ind = [
                0,
                self.cfg.expt.n_g,
                self.cfg.expt.n_g + self.cfg.expt.n_e,
                len(iq_list[0]),
            ]

        # Process data for each qubit
        for q in range(len(self.cfg.expt.qubit)):
            iq_array = np.array(iq_list[q])
            for i in range(3):
                nm = nms[i]
                if self.cfg.expt.active_reset:
                    inds = np.arange(start_ind[i], start_ind[i + 1], 3)
                else:
                    inds = np.arange(start_ind[i], start_ind[i + 1])

                data[f"avgi_{nm}_{q}"] = iq_array[inds, :, 0]
                data[f"avgq_{nm}_{q}"] = iq_array[inds, :, 1]

        if get_hist:
            data["bin_centers"] = v
            data["hist"] = hist

        for key in data:
            data[key] = np.array(data[key])
        self.data = data

        return self.data

    def analyze(self, data=None, **kwargs):
        """No-op: shot data is processed downstream."""
        if data is None:
            data = self.data
        return data

    def display(
        self,
        data=None,
        fit=True,
        plot_all=False,
        ax=None,
        show_hist=True,
        rescale=False,
        savefig=True,
        **kwargs,
    ):
        """Plot histograms of the interleaved ground/excited/T1 measurement blocks."""
        if data is None:
            data = self.data

        nexp = self.cfg.expt.n_g + self.cfg.expt.n_e + self.cfg.expt.n_t1
        n_t1 = self.cfg.expt.n_t1
        pi_time = 0.4  # Fix me

        if self.cfg.expt.active_reset:
            n_reset = 3
            pulse_length = (
                self.cfg.expt.readout
                * (self.cfg.expt.n_g + n_reset * (self.cfg.expt.n_e + n_t1))
                + self.cfg.expt.wait_time * n_t1
                + nexp * pi_time
            )
        else:
            pulse_length = (
                self.cfg.expt.readout * nexp
                + self.cfg.expt.wait_time * n_t1
                + self.cfg.expt.final_delay * (self.cfg.expt.n_e + n_t1)
                + nexp * pi_time
            )
        pulse_length = pulse_length / 1e6
        navg = 100
        nred = int(np.floor(navg / 10))
        m = 0.5  # marker size
        alpha = min(1, 250 / (self.cfg.expt.shots) ** (2 / 3))

        # Create histograms for both qubits
        if show_hist:
            fig_hist, axes_hist = plt.subplots(1, 2, figsize=(8, 3))
            for q in range(len(self.cfg.expt.qubit)):
                axes_hist[q].hist(
                    data[f"avgi_e_{q}"].flatten(),
                    bins=50,
                    alpha=0.6,
                    color="r",
                    label="Excited State",
                    density=True,
                )
                axes_hist[q].hist(
                    data[f"avgi_g_{q}"].flatten(),
                    bins=50,
                    alpha=0.6,
                    color="b",
                    label="Ground State",
                    density=True,
                )
                axes_hist[q].set_xlabel("I [ADC units]")
                axes_hist[q].set_ylabel("Probability")
                axes_hist[q].set_title(f"Q{self.cfg.expt.qubit[q]}")
                axes_hist[q].legend()

        # Raw IQ data plot
        fig_raw, axes_raw = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        q = 0
        for q in range(len(self.cfg.expt.qubit)):

            for i in range(len(data[f"avgi_e_{q}"])):

                # Plot I quadrature
                axes_raw[q, 0].plot(
                    data[f"avgi_e_{q}"][i], "b.", markersize=m, alpha=alpha
                )
                axes_raw[q, 1].plot(
                    data[f"avgq_e_{q}"][i], "b.", markersize=m, alpha=alpha
                )

            # Plot Q quadrature
            for i in range(len(data[f"avgq_g_{q}"])):
                axes_raw[q, 0].plot(
                    data[f"avgi_g_{q}"][i], "k.", markersize=m, alpha=alpha
                )
                axes_raw[q, 1].plot(
                    data[f"avgq_g_{q}"][i], "k.", markersize=m, alpha=alpha
                )
            for i in range(len(data[f"avgi_t1_{q}"])):
                axes_raw[q, 0].plot(
                    data[f"avgi_t1_{q}"][i], "r.", markersize=m, alpha=alpha
                )
                axes_raw[q, 1].plot(
                    data[f"avgq_t1_{q}"][i], "r.", markersize=m, alpha=alpha
                )

            axes_raw[q, 0].set_ylabel(f"I (ADC) Q{self.cfg.expt.qubit[q]}")
            axes_raw[q, 1].set_ylabel(f"Q (ADC) Q{self.cfg.expt.qubit[q]}")

        # Processed data plots for each qubit
        t1_data_both = []

        for q in range(len(self.cfg.expt.qubit)):
            t1_data = data[f"avgi_t1_{q}"].transpose().flatten()
            g_data = data[f"avgi_g_{q}"].transpose().flatten()
            e_data = data[f"avgi_e_{q}"].transpose().flatten()

            fig_proc, axes_proc = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

            # Smooth and plot T1 data
            smoothed_t1_data = uniform_filter1d(t1_data, size=navg * self.cfg.expt.n_t1)
            smoothed_t1_data = smoothed_t1_data[:: nred * self.cfg.expt.n_t1]

            # Smooth and plot ground state data
            smoothed_g_data = uniform_filter1d(g_data, size=navg * self.cfg.expt.n_g)
            smoothed_g_data = smoothed_g_data[:: nred * self.cfg.expt.n_g]

            # Smooth and plot excited state data
            smoothed_e_data = uniform_filter1d(e_data, size=navg * self.cfg.expt.n_e)
            smoothed_e_data = smoothed_e_data[:: nred * self.cfg.expt.n_e]

            npts = len(smoothed_t1_data)
            times = np.arange(npts) * pulse_length * nred

            axes_proc[0].plot(
                times,
                smoothed_t1_data,
                "k.-",
                linewidth=0.1,
                markersize=m,
                label="Smoothed T1 Data",
            )
            axes_proc[1].plot(
                times,
                smoothed_g_data,
                "k.-",
                linewidth=0.1,
                markersize=m,
                label="Smoothed g Data",
            )
            axes_proc[2].plot(
                times,
                smoothed_e_data,
                "k.-",
                linewidth=0.1,
                markersize=m,
                label="Smoothed e Data",
            )

            # Calculate and plot normalized T1 decay
            dv = smoothed_e_data - smoothed_g_data
            pt1 = (smoothed_t1_data - smoothed_g_data) / dv
            axes_proc[3].plot(
                times, pt1, "k.-", linewidth=0.1, markersize=m, label="Normalized T1"
            )
            axes_proc[3].axhline(
                np.exp(-1), color="r", linestyle="--", label="$e^{-1}$"
            )

            axes_proc[0].set_ylabel("I (ADC), $T=T_1$")
            axes_proc[1].set_ylabel("I (ADC), $g$ state")
            axes_proc[2].set_ylabel("I (ADC), $e$ state")
            axes_proc[3].set_ylabel("$(v_{t1}-v_g)/(v_e-v_g)$")
            axes_proc[3].set_xlabel("Time (s)")

            fig_proc.suptitle(f"Q{self.cfg.expt.qubit[q]} T1 Analysis")

            # T1 time constant plot
            fig_t1, ax_t1 = plt.subplots(1, 1, figsize=(15, 4))
            t1m = -1 / np.log(pt1)
            ax_t1.plot(times, t1m, "k.-", linewidth=0.1, markersize=m, label="T1 Data")
            ax_t1.set_xlabel("Time (s)")
            ax_t1.set_ylabel("$T_1/\tau$")
            ax_t1.set_title(f"Q{self.cfg.expt.qubit[q]} T1 Time Evolution")

            # Combined plot with twin axes
            fig_comb, ax_comb = plt.subplots(1, 1, figsize=(14, 4))
            ax_comb.plot(
                times,
                smoothed_t1_data,
                "k.-",
                linewidth=0.1,
                markersize=m,
                label="T1 Data",
            )
            ax_e = ax_comb.twinx()
            ax_e.plot(
                times,
                smoothed_e_data,
                "b.-",
                linewidth=0.1,
                markersize=m,
                label="e Data",
            )
            ax_e.legend()
            ax_g = ax_comb.twinx()
            ax_g.plot(
                times,
                smoothed_g_data,
                "r.-",
                linewidth=0.1,
                markersize=m,
                label="g Data",
            )
            ax_g.legend()
            ax_comb.set_title(f"Smoothed Offsetted T1 Data Q{self.cfg.expt.qubit[q]}")
            ax_comb.legend()
            ax_comb.set_xlabel("Time (s)")

            fig_comb_no_offset, ax_comb_no_offset = plt.subplots(1, 1, figsize=(14, 4))
            ax_comb_no_offset.plot(
                times,
                smoothed_t1_data,
                "k.-",
                linewidth=0.1,
                markersize=m,
                label="T1 Data",
            )
            ax_comb_no_offset.plot(
                times,
                smoothed_e_data,
                "b.-",
                linewidth=0.1,
                markersize=m,
                label="e Data",
            )
            ax_comb_no_offset.plot(
                times,
                smoothed_g_data,
                "r.-",
                linewidth=0.1,
                markersize=m,
                label="g Data",
            )
            ax_comb_no_offset.legend()
            ax_comb_no_offset.set_title(
                f"Smoothed No Offset T1 Data Q{self.cfg.expt.qubit[q]}"
            )
            ax_comb_no_offset.set_xlabel("Time (s)")

            fig_comb_raw, ax_comb_raw = plt.subplots(1, 1, figsize=(14, 4))
            ax_comb_raw.plot(
                np.arange(len(g_data)),
                g_data,
                "r.-",
                linewidth=0.005,
                markersize=m,
                label="g Data",
                alpha=0.1,
            )
            ax_comb_raw.plot(
                np.arange(len(e_data)),
                e_data,
                "b.-",
                linewidth=0.05,
                markersize=m,
                label="e Data",
                alpha=0.1,
            )
            ax_comb_raw.plot(
                np.arange(len(t1_data)),
                t1_data,
                "k.-",
                linewidth=0.05,
                markersize=m,
                label="T1 Data",
                alpha=0.1,
            )
            ax_comb_raw.legend()
            ax_comb_raw.set_title(f"Raw T1 Data Q{self.cfg.expt.qubit[q]}")
            ax_comb_raw.set_xlabel("npts")
            date_time = datetime.now().strftime("%Y%m%d%H%M")
            t1_data_both.append(t1_data)
            if savefig:
                i = 0
                for fig in [
                    fig_proc,
                    fig_t1,
                    fig_comb,
                    fig_comb_no_offset,
                    fig_comb_raw,
                ]:
                    fig.tight_layout()
                    imname = Path(self.fname).stem
                    fig.savefig(
                        Path(self.path) / "images" / f"{imname}_Q{self.cfg.expt.qubit[q]}_{date_time}_{i}.png"
                    )
                    i += 1

        plt.show()
        if savefig:
            self.save_config()
