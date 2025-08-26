import matplotlib.pyplot as plt
import numpy as np
from qick import *
import seaborn as sns
from tqdm import tqdm_notebook as tqdm

from ..general.qick_experiment import QickExperiment
from .single_shot import HistogramExperiment
from ...helpers import config

blue = "#4053d3"
red = "#b51d14"
int_rgain = True


class SingleShotOptExperiment(QickExperiment):
    """
    A class for optimizing single-shot readout experiments by sweeping frequency, gain, and readout length.

    This experiment iterates through a parameter space to find the optimal
    combination of readout frequency, gain, and length that maximizes readout fidelity.

    The parameters for this experiment can be configured via the `params` dictionary.
    If a parameter is not provided, a default value will be used.

    Default Parameters
    ------------------
    - `span_f`: Readout frequency span, defaults to `0.8 * kappa` from the device config.
    - `expts_f`: Number of frequency points, defaults to 5.
    - `expts_gain`: Number of gain points, defaults to 5.
    - `expts_len`: Number of readout length points, defaults to 5.
    - `shots`: Number of shots per measurement, defaults to 10000.
    - `check_f`: Boolean to check the f-state, defaults to `False`.
    - `qubit`: Qubit index, defaults to the one specified in `qi`.
    - `save_data`: Boolean to save the raw data, defaults to `True`.
    - `qubit_chan`: Readout channel, defaults to the one from the hardware config.

    The starting points for frequency, gain, and length are determined based on the
    device configuration and the number of experiment points. If `expts_f`, `expts_gain`,
    or `expts_len` is 1, the starting value is taken directly from the device config.
    Otherwise, it is calculated to center the sweep around the config value.
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
            prefix = f"single_shot_opt_qubit_{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)
        self.im = cfg_dict["im"]
        self.soccfg = cfg_dict["soc"]
        self.config_file = cfg_dict["cfg_file"]
        self.cfg_dict = cfg_dict

        params_def = {
            "span_f": self.cfg.device.readout.kappa[qi] * 0.8,
            "expts_f": 5,
            "expts_gain": 5,
            "expts_len": 5,
            "shots": 10000,
            "check_f": False,
            "qubit": [qi],
            "save_data": True,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        params = {**params_def, **params}

        # Start vals
        if params["expts_f"] == 1:
            params_def["start_f"] = self.cfg.device.readout.frequency[qi]
        else:
            params_def["start_f"] = (
                self.cfg.device.readout.frequency[qi] - 0.5 * params["span_f"]
            )

        if params["expts_gain"] == 1:
            params_def["start_gain"] = self.cfg.device.readout.gain[qi]
            params_def["span_gain"] = 0
        else:
            if style == "fine":
                params_def["start_gain"] = self.cfg.device.readout.gain[qi] * 0.8
                params_def["span_gain"] = 0.4 * self.cfg.device.readout.gain[qi]
            else:
                params_def["start_gain"] = self.cfg.device.readout.gain[qi] * 0.3
                params_def["span_gain"] = 1.8 * self.cfg.device.readout.gain[qi]

        if params["expts_len"] == 1:
            params_def["start_len"] = self.cfg.device.readout.readout_length[qi]
        else:
            if style == "fine":
                params_def["start_len"] = (
                    self.cfg.device.readout.readout_length[qi] * 0.8
                )
                params_def["span_len"] = (
                    0.4 * self.cfg.device.readout.readout_length[qi]
                )
            else:
                params_def["start_len"] = (
                    self.cfg.device.readout.readout_length[qi] * 0.3
                )
                params_def["span_len"] = (
                    1.8 * self.cfg.device.readout.readout_length[qi]
                )

        params = {**params_def, **params}
        if params["expts_f"] == 1:
            params_def["step_f"] = 0
        else:
            params_def["step_f"] = params["span_f"] / (params["expts_f"] - 1)

        if params["expts_gain"] == 1:
            params_def["step_gain"] = 0
            params_def["span_gain"] = 0
        else:
            params_def["step_gain"] = params["span_gain"] / (params["expts_gain"] - 1)

        if params["expts_len"] == 1:
            params_def["step_len"] = 0
        else:
            params_def["step_len"] = params["span_len"] / (params["expts_len"] - 1)

        if params["span_gain"] + params["start_gain"] > self.cfg.device.qubit.max_gain:
            params_def["span_gain"] = (
                self.cfg.device.qubit.max_gain - params["start_gain"]
            )
        self.cfg.expt = {**params_def, **params}

        # Check for unexpected parameters
        super().check_params(params)
        if print:
            super().print()
            go = False
        if go:
            self.go(analyze=False, display=False, progress=False, save=True)
            self.analyze()
            self.display()

    def acquire(self, progress=False, debug=False):
        fpts = self.cfg.expt["start_f"] + self.cfg.expt["step_f"] * np.arange(
            self.cfg.expt["expts_f"]
        )

        max_gain = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"] * (
            self.cfg.expt["expts_gain"] - 1
        )
        if max_gain > self.cfg.device.qubit.max_gain:
            self.cfg.expt["step_gain"] = (
                self.cfg.device.qubit.max_gain - self.cfg.expt["start_gain"]
            ) / (self.cfg.expt["expts_gain"] - 1)
        gainpts = self.cfg.expt["start_gain"] + self.cfg.expt["step_gain"] * np.arange(
            self.cfg.expt["expts_gain"]
        )

        lenpts = self.cfg.expt["start_len"] + self.cfg.expt["step_len"] * np.arange(
            self.cfg.expt["expts_len"]
        )

        if "save_data" not in self.cfg.expt:
            self.cfg.expt.save_data = False

        fid = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        threshold = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        angle = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        tm = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        sigma = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        if "check_f" not in self.cfg.expt:
            check_f = False
        else:
            check_f = self.cfg.expt.check_f
        Ig, Ie, Qg, Qe = [], [], [], []
        if check_f:
            If, Qf = [], []
        gprog = False
        fprog = False
        lprog = False
        if len(fpts) > 1:
            fprog = True
        else:
            fprog = False
            if len(gainpts) > 1:
                gprog = True
            else:
                gprog = False
                if len(lenpts) > 1:
                    lprog = True
                else:
                    lprog = False

        for f_ind, f in enumerate(tqdm(fpts, disable=not fprog)):
            Ig.append([])
            Ie.append([])
            Qg.append([])
            Qe.append([])
            if check_f:
                If.append([])
                Qf.append([])
            for g_ind, gain in enumerate(tqdm(gainpts, disable=not gprog)):
                Ig[-1].append([])
                Ie[-1].append([])
                Qg[-1].append([])
                Qe[-1].append([])
                if check_f:
                    If[-1].append([])
                    Qf[-1].append([])
                for l_ind, l in enumerate(tqdm(lenpts, disable=not lprog)):
                    shot = HistogramExperiment(
                        self.cfg_dict,
                        go=False,
                        progress=False,
                        qi=self.cfg.expt.qubit[0],
                        params=dict(
                            frequency=f,
                            gain=gain,
                            readout_length=l,
                            reps=1,
                            check_e=True,
                            check_f=check_f,
                            shots=self.cfg.expt.shots,
                            save_data=self.cfg.expt.save_data,
                            qubit_chan=self.cfg.expt.qubit_chan,
                        ),
                    )
                    # shot.cfg = self.cfg

                    shot.go(analyze=False, display=False, progress=progress, save=False)
                    Ig[-1][-1].append(shot.data["Ig"])
                    Ie[-1][-1].append(shot.data["Ie"])
                    Qg[-1][-1].append(shot.data["Qg"])
                    Qe[-1][-1].append(shot.data["Qe"])
                    if check_f:
                        If[-1][-1].append(shot.data["If"])
                        Qf[-1][-1].append(shot.data["Qf"])
                    results = shot.analyze(verbose=False)
                    fid[f_ind, g_ind, l_ind] = (
                        results["fids"][0] if not check_f else results["fids"][1]
                    )
                    threshold[f_ind, g_ind, l_ind] = (
                        results["thresholds"][0]
                        if not check_f
                        else results["thresholds"][1]
                    )
                    try:
                        tm[f_ind, g_ind, l_ind] = results["tm"]
                        sigma[f_ind, g_ind, l_ind] = results["sigma"]
                    except:
                        pass
                    angle[f_ind, g_ind, l_ind] = results["angle"]
                    # print(f'freq: {f}, gain: {gain}, len: {l}')
                    # print(f'\tfid ge [%]: {100*results["fids"][0]}')
                    # if check_f:
                    #     print(f'\tfid gf [%]: {100*results["fids"][1]:.3f}')

        if check_f:
            self.data["If"] = np.array(If)
            self.data["Qf"] = np.array(Qf)
        if self.cfg.expt.save_data:
            self.data = dict(
                fpts=fpts,
                gainpts=gainpts,
                lenpts=lenpts,
                fid=fid,
                threshold=threshold,
                angle=angle,
                Ig=Ig,
                Ie=Ie,
                Qg=Qg,
                Qe=Qe,
                tm=tm,
                sigma=sigma,
            )
            if check_f:
                self.data["If"] = If
                self.data["Qf"] = Qf
        else:
            self.data = dict(
                fpts=fpts,
                gainpts=gainpts,
                lenpts=lenpts,
                fid=fid,
                threshold=threshold,
                angle=angle,
                tm=tm,
                sigma=sigma,
            )

        for key in self.data.keys():
            self.data[key] = np.array(self.data[key])
        return self.data

    def analyze(self, data=None, fit=True, low_gain=True, **kwargs):
        if data == None:
            data = self.data
        fid = data["fid"]
        fpts = data["fpts"]
        gainpts = data["gainpts"]
        lenpts = data["lenpts"]

        imax = np.unravel_index(np.argmax(fid), shape=fid.shape)
        perc_fid = 0.95
        max_fid = np.max(fid)
        print(f"Max fidelity {100*max_fid:.3f} %")

        print(
            f"Optimal params: \n Freq (MHz) {fpts[imax[0]]:.3f} \n Gain (DAC units) {gainpts[imax[1]]:.3f} \n Readout length (us) {lenpts[imax[2]]:.3f}"
        )
        self.do_more = self.check_edges()
        if low_gain and not self.do_more:
            min_accept = max_fid * perc_fid

            # Find values above threshold
            above_threshold = fid >= min_accept

            # For each row, get first index that's above threshold
            freq_indices = np.where(above_threshold)[0]
            gain_indices = np.where(above_threshold)[1]
            time_indices = np.where(above_threshold)[2]
            min_ind = gain_indices + time_indices
            a = np.argmin(min_ind)

            # Get the first occurrence
            imax = (freq_indices[a], gain_indices[a], time_indices[a])
            print(f"Set fidelity: {100*fid[imax]:.3f} %")
            print(
                f"Set params: \n Freq (MHz) {fpts[imax[0]]:.3f} \n Gain (DAC units) {gainpts[imax[1]]:.3f} \n Readout length (us) {lenpts[imax[2]]:.3f}"
            )

        self.data["freq"] = fpts[imax[0]]
        self.data["gain"] = gainpts[imax[1]]
        self.data["length"] = lenpts[imax[2]]

        if self.data["gain"] == 1:  # change to max_gain
            self.do_more = False

        return imax

    def display(self, data=None, plot_pars=False, **kwargs):
        if data is None:
            data = self.data

        fid = data["fid"]

        fpts = data["fpts"]  # outer sweep, index 0
        gainpts = data["gainpts"]  # middle sweep, index 1
        lenpts = data["lenpts"]  # inner sweep, index 2
        ndims = 0
        npts = []
        inds = []
        sweep_var = []
        labs = ["Freq. (MHz)", "Gain", "Readout Length ($\mu$s)"]
        if len(fpts) > 1:
            ndims += 1
            sweep_var.append("fpts")
            npts.append(len(fpts))
            inds.append(0)
        if len(gainpts) > 1:
            ndims += 1
            sweep_var.append("gainpts")
            npts.append(len(gainpts))
            inds.append(1)
        if len(lenpts) > 1:
            ndims += 1
            sweep_var.append("lenpts")
            npts.append(len(lenpts))
            inds.append(2)

        def smart_ax(n):
            row = int(np.ceil(n / 5))
            if n < 5:
                col = n
            else:
                col = 5
            return row, col

        title = f"Single Shot Optimization Q{self.cfg.expt.qubit[0]}"

        def return_dim(data, dim, i):
            if len(dim) == 1:
                if dim[0] == 0:
                    return data[i, :, :].reshape(-1)
                elif dim[0] == 1:
                    return data[:, i, :].reshape(-1)
                elif dim[0] == 2:
                    return data[:, :, i].reshape(-1)
            elif len(dim) == 2:
                if dim == [0, 1]:
                    return data[i[0], i[1], :].reshape(-1)
                if dim == [0, 2]:
                    return data[i[0], :, i[1]].reshape(-1)
                if dim == [1, 2]:
                    return data[:, i[0], i[1]].reshape(-1)

        m = 0.5
        imname = self.fname.split("\\")[-1]
        folder = self.fname[0 : -len(imname)]
        imname = imname[0:-3]

        if ndims == 1:
            row, col = smart_ax(npts[0])
            fig, ax = plt.subplots(row, col, figsize=(col * 3, row * 3))
            ax = ax.flatten()
            for i in range(npts[0]):

                ax[i].plot(
                    return_dim(self.data["Ig"], inds, i),
                    return_dim(self.data["Qg"], inds, i),
                    ".",
                    color=blue,
                    alpha=0.2,
                    markersize=m,
                )
                ax[i].plot(
                    return_dim(self.data["Ie"], inds, i),
                    return_dim(self.data["Qe"], inds, i),
                    ".",
                    color=red,
                    alpha=0.2,
                    markersize=m,
                )

                ax[i].set_title(f"{labs[inds[0]]} {data[sweep_var[0]][i]:.2f}")
            fig.savefig(folder + "images\\" + f"{imname}_raw_{k}.png")

        elif ndims == 2:
            fig, ax = plt.subplots(npts[0], npts[1], figsize=(npts[1] * 3, npts[0] * 3))

            for i in range(npts[0]):
                for j in range(npts[1]):
                    ax[i, j].plot(
                        return_dim(self.data["Ig"], inds, [i, j]),
                        return_dim(self.data["Qg"], inds, [i, j]),
                        ".",
                        color=blue,
                        alpha=0.2,
                        markersize=m,
                    )
                    ax[i, j].plot(
                        return_dim(self.data["Ie"], inds, [i, j]),
                        return_dim(self.data["Qe"], inds, [i, j]),
                        ".",
                        color=red,
                        alpha=0.2,
                        markersize=m,
                    )

                    if i == npts[0] - 1:
                        ax[i, j].set_xlabel(np.round(self.data[sweep_var[1]][j], 2))
                    if j == 0:
                        ax[i, j].set_ylabel(np.round(self.data[sweep_var[0]][i], 2))
            plt.figtext(0.5, 0.0, labs[inds[1]], horizontalalignment="center")
            plt.figtext(
                0.0, 0.5, labs[inds[0]], verticalalignment="center", rotation="vertical"
            )
            fig.savefig(folder + "images\\" + f"{imname}_raw.png")
        else:
            for k in range(npts[2]):
                fig, ax = plt.subplots(
                    npts[0], npts[1], figsize=(npts[1] * 3, npts[0] * 3)
                )
                for i in range(npts[0]):
                    for j in range(npts[1]):
                        ax[i, j].plot(
                            self.data["Ig"][i, j, k, :],
                            self.data["Qg"][i, j, k],
                            ".",
                            color=blue,
                            alpha=0.2,
                            markersize=m,
                        )
                        ax[i, j].plot(
                            self.data["Ie"][i, j, k, :],
                            self.data["Qe"][i, j, k],
                            ".",
                            color=red,
                            alpha=0.2,
                            markersize=m,
                        )
                fig.suptitle(title)
                fig.tight_layout()
                fig.savefig(folder + "images\\" + f"{imname}_raw_{k}.png")

        title = f"Single Shot Optimization Q{self.cfg.expt.qubit[0]}"
        fig = plt.figure(figsize=(9, 5.5))
        plt.title(title)
        if len(fpts) > 1:
            xval = fpts
            xlabel = "Frequency (MHz)"
            var1 = gainpts
            var2 = lenpts
            npts = len(var1) * len(var2)
            bb = sns.color_palette("coolwarm", npts)
            leg_title = "Gain, Len ($\mu$s)"
            for v1_ind, v1 in enumerate(var1):
                for v2_ind, v2 in enumerate(var2):
                    plt.plot(
                        xval,
                        100 * fid[:, v1_ind, v2_ind],
                        "o-",
                        label=f"{v1:.2f}, {v2:.2f}",
                        color=bb[v1_ind * len(var2) + v2_ind],
                    )
        elif len(gainpts) > 1:
            xval = gainpts
            xlabel = "Gain/Max Gain"
            var1 = fpts
            var2 = lenpts
            npts = len(var1) * len(var2)
            bb = sns.color_palette("coolwarm", npts)
            leg_title = "Freq (MHz), Len ($\mu$s)"
            for v1_ind, v1 in enumerate(var1):
                for v2_ind, v2 in enumerate(var2):
                    plt.plot(
                        xval,
                        100 * fid[v1_ind, :, v2_ind],
                        "o-",
                        label=f"{v1:.2f}, {v2:.2f}",
                        color=bb[v1_ind * len(var2) + v2_ind],
                    )
        else:
            xval = lenpts
            xlabel = "Readout length ($\mu$s)"
            var1 = fpts
            var2 = gainpts
            npts = len(var1) * len(var2)
            bb = sns.color_palette("coolwarm", npts)
            leg_title = "Freq (MHz), Gain"
            for v1_ind, v1 in enumerate(var1):
                for v2_ind, v2 in enumerate(var2):
                    plt.plot(
                        xval,
                        100 * fid[v1_ind, v2_ind, :],
                        "o-",
                        label=f"{v2:1.0f},  {v1:.2f}",
                        color=bb[v1_ind * len(var2) + v2_ind],
                    )

        plt.xlabel(xlabel)
        plt.ylabel(f"Fidelity [%]")
        plt.legend(title=leg_title)
        fig.savefig(folder + "images\\" + imname + ".png")
        plt.show()

        if plot_pars:
            tmv = self.data["tm"][0]
            tmv[tmv < 0.001] = np.nan
            sns.set_palette("coolwarm", len(tmv))
            fig, ax = plt.subplots(2, 1, figsize=(8, 6))
            # tm = np.transpose(tmv)
            for i, tm_arr in enumerate(tmv):
                gain = self.data["gainpts"][i]
                ax[0].plot(
                    self.data["lenpts"],
                    self.data["lenpts"] / tm_arr,
                    "o-",
                    label=f"{gain:.2f}",
                )
            ax[0].set_xlabel("Readout Length")
            ax[0].axhline(
                y=self.cfg.device.qubit.T1[self.cfg.expt.qubit[0]],
                color="k",
                linestyle="--",
                label="T1",
            )
            ax[0].set_ylabel("$T_m/(T_m/T_1)$")
            ax[0].legend()
            sigma = self.data["sigma"][0]

            for i, s in enumerate(sigma):
                gain = self.data["gainpts"][i]
                ax[1].loglog(self.data["lenpts"], s, "o-", label=f"{gain:.2f}")
            ax[1].legend()
            ax[1].set_xlabel("Readout Length")
            ax[1].set_ylabel("$\sigma$")
            fig.tight_layout()

    def check_edges(self):
        do_more = False
        fid = self.data["fid"]
        fid_expts = fid.shape
        if all(dim % 2 != 0 for dim in fid_expts):
            old_fid = fid[(fid_expts[0] // 2), (fid_expts[1] // 2), (fid_expts[2] // 2)]
            max_fid = np.max(fid)
            if (max_fid - old_fid) / old_fid > 0.1:
                print("Fidelity is not maximized at the center of the sweep.")
                max_indices = np.unravel_index(np.argmax(fid), fid.shape)
                print(f"Max fidelity found at indices: {max_indices}")
                if (
                    max_indices[1] == 0
                    or max_indices[1] == fid_expts[1] - 1
                    or max_indices[2] == 0
                    or max_indices[2] == fid_expts[2] - 1
                ):
                    do_more = True
        else:
            print("Not all elements in fid_expts are odd.")
        return do_more

    def update(self, verbose=True):
        qi = self.cfg.expt.qubit[0]
        cfg_file= self.config_file
        config.update_readout(cfg_file, "gain", self.data["gain"], qi, verbose=verbose)
        config.update_readout(
            cfg_file, "readout_length", self.data["length"], qi, verbose=verbose
        )
        config.update_readout(
            cfg_file, "frequency", self.data["freq"], qi, verbose=verbose
        )
