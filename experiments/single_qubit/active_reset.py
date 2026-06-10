import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erf
from copy import deepcopy
import copy
import seaborn as sns

from qick import *
from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment
from ..general.qick_program import QickProgram
from ...calib import readout_helpers as helpers


blue = "#4053d3"
red = "#b51d14"
int_rgain = True

class MemoryProgram(QickProgram):
    # This checks that the two memories are getting the same values
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.add_loop("shotloop", cfg.expt.shots)  # number of total shots

        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        self.phase = cfg.device.readout.phase[cfg.expt.qubit[0]]
        self.readout_length = cfg.expt.readout_length
        super()._initialize(cfg, readout="")

        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
        self.delay_auto(t=0.01, tag="wait")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
        self.wait_auto(cfg.expt.read_wait)
        self.read_input(ro_ch=self.adc_ch)
        self.write_dmem(addr=0, src="s_port_l")
        self.write_dmem(addr=1, src="s_port_h")

        if cfg.expt.active_reset:
            self.reset(5)

    def reset(self, i):

        # Perform active reset i times
        cfg = AttrDict(self.cfg)
        for n in range(i):
            self.wait_auto(cfg.expt.read_wait)
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)

            # read the input, test a threshold, and jump if it is met [so, if i<threshold, doesn't do pi pulse]
            self.read_and_jump(
                ro_ch=self.adc_ch,
                component="I",
                threshold=cfg.expt.threshold,
                test="<",
                label=f"NOPULSE{n}",
            )

            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(0.01)
            self.label(f"NOPULSE{n}")

            if n < i - 1:
                self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
                self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
                if self.lo_ch is not None:
                    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)

    def collect_shots(self, offset=0):

        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            # nsamp = rocfg["length"]
            iq_raw = self.get_raw()
            i_shots = iq_raw[i][:, :, 0, 0]  # / nsamp - offset
            i_shots = i_shots.flatten()
            q_shots = iq_raw[i][:, :, 0, 1]  # / nsamp - offset
            q_shots = q_shots.flatten()
        return i_shots, q_shots

class RepMeasProgram(QickProgram):

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.add_loop("shotloop", cfg.expt.shots)  # number of total shots

        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        self.phase = cfg.device.readout.phase[cfg.expt.qubit[0]]
        self.readout_length = cfg.expt.readout_length
        super()._initialize(cfg)

        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")


    def _body(self, cfg):

        cfg = AttrDict(self.cfg)
        self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
        self.delay_auto(t=0.01, tag="wait")

        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)

        if cfg.expt.active_reset:
            self.reset(5)

    def reset(self, i):

        # Perform active reset i times
        cfg = AttrDict(self.cfg)
        for n in range(i):
            self.wait_auto(cfg.expt.read_wait)
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)

            if n < i - 1:
                self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
                self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
                if self.lo_ch is not None:
                    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
                self.delay_auto(0.01)

    def collect_shots(self, offset=0):

        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            # nsamp = rocfg["length"]
            iq_raw = self.get_raw()
            i_shots = iq_raw[i][:, :, 0, 0]  # / nsamp - offset
            i_shots = i_shots.flatten()
            q_shots = iq_raw[i][:, :, 0, 1]  # / nsamp - offset
            q_shots = q_shots.flatten()
        return i_shots, q_shots

class MemoryExperiment(QickExperiment):
    """
    Histogram Experiment
    expt = dict(
        shots: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    """

    def __init__(
        self,
        cfg_dict,
        prefix=None,
        progress=False,
        qi=0,
        go=True,
        check_f=False,
        params={},
        style="",
        display=True,
    ):
        """
        This experiment does not do any fitting. It is designed to test the memory and the active reset.
        Default parameters are defined in this function, and can be overwritten by config_dict['expt']
        - shots: 10000
        - reps: 1
        - expts: 100
        - rounds: 1
        - readout_length: from device config
        - frequency: from device config
        - gain: from device config
        - active_reset: False
        - check_e: True
        - check_f: False
        - read_wait: 0.2
        - qubit: [qi]
        - qubit_chan: from hardware config
        """

        if prefix is None:
            prefix = f"single_shot_qubit_{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = dict(
            shots=1,
            reps=1,
            expts=100,
            rounds=1,
            readout_length=self.cfg.device.readout.readout_length[qi],
            frequency=self.cfg.device.readout.frequency[qi],
            gain=self.cfg.device.readout.gain[qi],
            active_reset=False,
            check_e=True,
            check_f=check_f,
            read_wait=0.2,
            qubit=[qi],
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[qi],
        )

        self.cfg.expt = {**params_def, **params}
        if self.cfg.expt.active_reset:
            super().configure_reset()

        if go:
            self.go(analyze=True, display=False, progress=progress, save=True)

    def acquire(self, progress=False, debug=False):

        data = dict()
        if "setup_reset" in self.cfg.expt and self.cfg.expt.setup_reset:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        elif self.cfg.expt.active_reset:
            final_delay = self.cfg.expt.readout_length
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Ground state shots

        cfg2 = copy.deepcopy(dict(self.cfg))
        cfg = AttrDict(cfg2)
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False

        ig, qg, ie, qe, g_phase, e_phase, g_norm, e_norm = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i in range(cfg.expt.expts):
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = False
            cfg.expt.pulse_f = False
            histpro = MemoryProgram(
                soccfg=self.soccfg, final_delay=final_delay, cfg=cfg
            )
            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                progress=progress,
            )
            data["Ig"] = iq_list[0][0][:, 0]
            data["Qg"] = iq_list[0][0][:, 1]
            if self.cfg.expt.active_reset:
                data["Igr"] = iq_list[0][1:, :, 0]

            irawg, qrawg = histpro.collect_shots()

            rawd = [irawg[-1], qrawg[-1]]
            print("buffered readout:", rawd)
            dd = self.soc.read_mem(2, "dmem")
            dd_ang = np.arctan2(dd[1], dd[0]) * 180 / np.pi
            print("feedback readout:", dd)
            #print("feedback angle:", dd_ang)
            dd_sz = np.sqrt(dd[0] ** 2 + dd[1] ** 2)
            # print("g size:", dd_sz)
            ig.append(dd[0])
            qg.append(dd[1])
            g_phase.append(dd_ang)
            g_norm.append(dd_sz)

            # Excited state shots
            if self.cfg.expt.check_e:
                cfg = AttrDict(self.cfg.copy())
                cfg.expt.pulse_e = True
                cfg.expt.pulse_f = False
                histpro = MemoryProgram(
                    soccfg=self.soccfg, final_delay=final_delay, cfg=cfg
                )
                iq_list = histpro.acquire(
                    self.im[self.cfg.aliases.soc],
                    threshold=None,
                    progress=progress,
                )

                data["Ie"] = iq_list[0][0][:, 0]
                data["Qe"] = iq_list[0][0][:, 1]
                irawe, qrawe = histpro.collect_shots()
                rawd = [irawe[-1], qrawe[-1]]
                dd = self.soc.read_mem(2, "dmem")
                dd_ang = np.arctan2(dd[1], dd[0]) * 180 / np.pi
                print("buffered readout:", rawd)
                print("feedback readout:", dd)
                #print("feedback angle:", dd_ang)
                dd_sz = np.sqrt(dd[0] ** 2 + dd[1] ** 2)
                ie.append(dd[0])
                qe.append(dd[1])
                e_phase.append(dd_ang)
                e_norm.append(dd_sz)
                # print("e size:", dd_sz)
                if self.cfg.expt.active_reset:
                    data["Ier"] = iq_list[0][1:, :, 0]
                # print(f"{np.mean(irawg)} mean raw g, {np.mean(irawe)} mean raw e")
        data = {
            "ie": ie,
            "qe": qe,
            "ig": ig,
            "qg": qg,
            "g_phase": g_phase,
            "e_phase": e_phase,
            "g_norm": g_norm,
            "e_norm": e_norm,
        }

        keys_list = data.keys()
        for key in keys_list:
            data[key] = np.array(data[key])

        mean_data = {key + "_mean": np.mean(data[key]) for key in keys_list}
        std_data = {key + "_std": np.std(data[key]) for key in keys_list}

        data.update(mean_data)
        data.update(std_data)

        qi = self.cfg.expt.qubit[0]
        ie = data["ie_mean"] / (
            self.cfg.device.readout.readout_length[qi] / 0.0032552083333333335
        )
        print(ie)
        ig = data["ig_mean"] / (
            self.cfg.device.readout.readout_length[qi] / 0.0032552083333333335
        )
        print(ig)
        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=False, **kwargs):
        if data is None:
            data = self.data

        return data

    def display(
        self,
        data=None,
        span=None,
        verbose=False,
        plot_e=True,
        plot_f=False,
        ax=None,
        plot=True,
        **kwargs,
    ):
        if data is None:
            data = self.data

    def check_reset(self):
        nbins = 75
        fig, ax = plt.subplots(2, 1, figsize=(6, 7))
        fig.suptitle(f"Q{self.cfg.expt.qubit[0]}")
        vg, histg = make_hist(self.data["Ig"], nbins=nbins)
        ax[0].semilogy(vg, histg, color=blue, linewidth=2)
        ax[1].semilogy(vg, histg, color=blue, linewidth=2)
        b = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(self.data["Igr"]))
        ve, histe = make_hist(self.data["Ie"], nbins=nbins)
        ax[1].semilogy(ve, histe, color=red, linewidth=2)
        for i in range(len(self.data["Igr"])):
            v, hist = make_hist(self.data["Igr"][i], nbins=nbins)
            ax[0].semilogy(v, hist, color=b[i], linewidth=1, label=f"{i+1}")
            v, hist = make_hist(self.data["Ier"][i], nbins=nbins)
            ax[1].semilogy(v, hist, color=b[i], linewidth=1, label=f"{i+1}")

        def find_bin_closest_to_value(bins, value):
            return np.argmin(np.abs(bins - value))

        ind = find_bin_closest_to_value(v, self.data["ie"])
        ind_e = find_bin_closest_to_value(ve, self.data["ie"])
        ind_g = find_bin_closest_to_value(vg, self.data["ie"])

        reset_level = hist[ind]
        e_level = histe[ind_e]
        g_level = histg[ind_g]

        print(
            f"Reset is {reset_level/e_level:3g} of e and {reset_level/g_level:3g} of g"
        )

        self.data["reset_e"] = reset_level / e_level
        self.data["reset_g"] = reset_level / g_level

        ax[0].legend()

        ax[0].set_title("Ground state")
        ax[1].set_title("Excited state")
        plt.show()


class RepMeasExperiment(QickExperiment):
    """
    Histogram Experiment
    expt = dict(
        shots: number of shots per expt
        check_e: whether to test the e state blob (true if unspecified)
        check_f: whether to also test the f state blob
    )
    This just does the measurement repeatedly, not doing actual active reset. If the readout power is too high, we might expect that the ground state is being excited during the measurement. 
    """

    def __init__(
        self,
        cfg_dict,
        prefix=None,
        progress=True,
        qi=0,
        go=True,
        check_f=False,
        params={},
        style="",
        display=True,
    ):

        if prefix is None:
            prefix = f"single_shot_qubit_{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = dict(
            shots=10000,
            reps=1,
            rounds=1,
            readout_length=self.cfg.device.readout.readout_length[qi],
            frequency=self.cfg.device.readout.frequency[qi],
            gain=self.cfg.device.readout.gain[qi],
            active_reset=False,
            check_e=True,
            check_f=check_f,
            qubit=[qi],
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[qi],
        )

        self.cfg.expt = {**params_def, **params}
        if self.cfg.expt.active_reset:
            super().configure_reset()

        if go:
            self.go(analyze=True, display=display, progress=progress, save=True)

    def acquire(self, progress=False, debug=False):

        data = dict()
        if "setup_reset" in self.cfg.expt and self.cfg.expt.setup_reset:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        elif self.cfg.expt.active_reset:
            final_delay = self.cfg.expt.readout_length
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Ground state shots
        cfg2 = copy.deepcopy(dict(self.cfg))
        cfg = AttrDict(cfg2)
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False

        histpro = RepMeasProgram(soccfg=self.soccfg, final_delay=final_delay, cfg=cfg)
        iq_list = histpro.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            progress=progress,
        )
        data["Ig"] = iq_list[0][0][:, 0]
        data["Qg"] = iq_list[0][0][:, 1]
        if self.cfg.expt.active_reset:
            data["Igr"] = iq_list[0][1:, :, 0]

        irawg, qrawg = histpro.collect_shots()

        rawd = [irawg[-1], qrawg[-1]]
        # print("buffered readout:", rawd)

        # Excited state shots
        if self.cfg.expt.check_e:
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = False
            histpro = RepMeasProgram(
                soccfg=self.soccfg, final_delay=final_delay, cfg=cfg
            )
            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                progress=progress,
            )

            data["Ie"] = iq_list[0][0][:, 0]
            data["Qe"] = iq_list[0][0][:, 1]
            irawe, qraw = histpro.collect_shots()
            # rawd = [iraw[-1], qraw[-1]]
            # print("buffered readout:", rawd)
            # print("feedback readout:", self.soc.read_mem(2,'dmem'))
            if self.cfg.expt.active_reset:
                data["Ier"] = iq_list[0][1:, :, 0]
            # print(f"{np.mean(irawg)} mean raw g, {np.mean(irawe)} mean raw e")

        # Excited state shots

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=False, **kwargs):
        if data is None:
            data = self.data

        params, _ = helpers.hist(data=data, plot=False, span=span, verbose=verbose)
        data.update(params)
        try:
            data2, p, paramsg, paramse2 = helpers.fit_single_shot(data, plot=False)
            data.update(p)
            data["vhg"] = data2["vhg"]
            data["histg"] = data2["histg"]
            data["vhe"] = data2["vhe"]
            data["histe"] = data2["histe"]
            data["paramsg"] = paramsg
            data["shots"] = self.cfg.expt.shots
        except:
            print("Fits failed")

        return data

    def display(
        self,
        data=None,
        span=None,
        verbose=False,
        plot_e=True,
        plot_f=False,
        ax=None,
        plot=True,
        **kwargs,
    ):
        if data is None:
            data = self.data

        if ax is not None:
            savefig = False
        else:
            savefig = True

        params, fig = helpers.hist(
            data=data,
            plot=plot,
            verbose=verbose,
            span=span,
            ax=ax,
            qubit=self.cfg.expt.qubit[0],
        )
        fids = params["fids"]
        thresholds = params["thresholds"]
        angle = params["angle"]
        print(f"ge Fidelity (%): {100*fids[0]:.3f}")
        if "expt" not in self.cfg:
            self.cfg.expt.check_e = plot_e
            self.cfg.expt.check_f = plot_f
        if self.cfg.expt.check_f:
            print(f"gf Fidelity (%): {100*fids[1]:.3f}")
            print(f"ef Fidelity (%): {100*fids[2]:.3f}")
        print(f"Rotation angle (deg): {angle:.3f}")
        print(f"Threshold ge: {thresholds[0]:.3f}")
        if self.cfg.expt.check_f:
            print(f"Threshold gf: {thresholds[1]:.3f}")
            print(f"Threshold ef: {thresholds[2]:.3f}")
        if savefig:
            plt.show()
            self.save_fig(fig)
            self.save_config()

    def check_reset(self):
        nbins = 75
        fig, ax = plt.subplots(2, 1, figsize=(6, 7))
        fig.suptitle(f"Q{self.cfg.expt.qubit[0]}")
        vg, histg = helpers.make_hist(self.data["Ig"], nbins=nbins)
        ax[0].semilogy(vg, histg, color=blue, linewidth=2)
        ax[1].semilogy(vg, histg, color=blue, linewidth=2)
        b = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(self.data["Igr"]))
        ve, histe = helpers.make_hist(self.data["Ie"], nbins=nbins)
        ax[1].semilogy(ve, histe, color=red, linewidth=2)
        for i in range(len(self.data["Igr"])):
            v, hist = helpers.make_hist(self.data["Igr"][i], nbins=nbins)
            ax[0].semilogy(v, hist, color=b[i], linewidth=1, label=f"{i+1}")
            v, hist = helpers.make_hist(self.data["Ier"][i], nbins=nbins)
            ax[1].semilogy(v, hist, color=b[i], linewidth=1, label=f"{i+1}")

        def find_bin_closest_to_value(bins, value):
            return np.argmin(np.abs(bins - value))

        ind = find_bin_closest_to_value(v, self.data["ie"])
        ind_e = find_bin_closest_to_value(ve, self.data["ie"])
        ind_g = find_bin_closest_to_value(vg, self.data["ie"])

        reset_level = hist[ind]
        e_level = histe[ind_e]
        g_level = histg[ind_g]

        print(
            f"Reset is {reset_level/e_level:3g} of e and {reset_level/g_level:3g} of g"
        )

        self.data["reset_e"] = reset_level / e_level
        self.data["reset_g"] = reset_level / g_level

        ax[0].legend()

        ax[0].set_title("Ground state")
        ax[1].set_title("Excited state")
        plt.show()


