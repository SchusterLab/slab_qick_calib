"""
Single-shot readout optimization.

Sweeps readout frequency, gain, and length, running a HistogramExperiment at
each point and mapping the readout fidelity over the 3D parameter space.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from qick import *
import seaborn as sns
from tqdm import tqdm_notebook as tqdm

from ..general.qick_experiment import QickExperiment
from .single_shot import HistogramExperiment
from ...helpers import config
from ...calib import readout_helpers as helpers

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
        check_f=False,
        params={},
        prefix=None,
        fname=None,
        progress=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        display=True,
        live_plot=False,
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
            "check_f": check_f,
            "qubit": [qi],
            "save_data": True,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            # Forwarded to every inner HistogramExperiment, so the readout
            # is optimized under the same reset conditions it will be used
            # with. configure_reset() runs inside that inner class.
            "reset_mode": "passive",
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            # Calibration only: re-measure after the final pi pulse so the
            # last recorded round is the true post-reset state. Costs one
            # readout per shot and nothing acts on it, so leave it off in use.
            "reset_verify": False,
            # Add a pi_ef ahead of each pi_ge, for sequences that can leave
            # population in |f>. Needs reset >= 2: |f> clears in one round but
            # |e> is driven to |f> first and clears on the next.
            "reset_f": False,
            # "while excited" semantics at a fixed trip count: once a round
            # reads ground, a latch suppresses every later pi pulse. Measures
            # every round, so the trigger count stays fixed.
            "reset_latch": False,
            # Mean rounds the while loop takes, for the cycle-time estimate
            # only; `reset` is its hard cap. See CLAUDE.md.
            "reset_while_typical": 1,
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
        self.live_plot = live_plot

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
        """Run a HistogramExperiment at every (frequency, gain, length) point and record fidelities."""
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
        fit_fid = np.full(shape=(len(fpts), len(gainpts), len(lenpts)), fill_value=np.nan)
        pop_ratio = np.full(shape=(len(fpts), len(gainpts), len(lenpts)), fill_value=np.nan)
        threshold = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        angle = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        tm = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        sigma = np.zeros(shape=(len(fpts), len(gainpts), len(lenpts)))
        if self.live_plot:
            self._init_live_plots(fpts, gainpts)
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
                            reset_mode=self.cfg.expt.reset_mode,
                            reset=self.cfg.expt.reset,
                            setup_reset=self.cfg.expt.setup_reset,
                            reset_verify=self.cfg.expt.reset_verify,
                            reset_f=self.cfg.expt.reset_f,
                            reset_latch=self.cfg.expt.reset_latch,
                            reset_while_typical=self.cfg.expt.reset_while_typical,
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
                    fit_fid_val = results.get("fit_fidelity")
                    if fit_fid_val is not None:
                        fit_fid[f_ind, g_ind, l_ind] = fit_fid_val
                    # Residual excitation ratio without a pi pulse: excited/ground
                    me = results.get("mag_excited_g")
                    if me is not None and me < 1:
                        pop_ratio[f_ind, g_ind, l_ind] = me / (1 - me)
                    angle[f_ind, g_ind, l_ind] = results["angle"]
                    if self.live_plot:
                        self._update_live_plots(
                            f_ind, g_ind, l_ind, fid, fit_fid, tm, pop_ratio,
                            results, shot.data,
                        )
                    # print(f'freq: {f}, gain: {gain}, len: {l}')
                    # print(f'\tfid ge [%]: {100*results["fids"][0]}')
                    # if check_f:
                    #     print(f'\tfid gf [%]: {100*results["fids"][1]:.3f}')

        if self.cfg.expt.save_data:
            self.data = dict(
                fpts=fpts,
                gainpts=gainpts,
                lenpts=lenpts,
                fid=fid,
                fit_fid=fit_fid,
                pop_ratio=pop_ratio,
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
                fit_fid=fit_fid,
                pop_ratio=pop_ratio,
                threshold=threshold,
                angle=angle,
                tm=tm,
                sigma=sigma,
            )

        for key in self.data.keys():
            self.data[key] = np.array(self.data[key])
        return self.data

    def _init_live_plots(self, fpts, gainpts):
        self._live_metrics_fig, self._live_metrics_ax = plt.subplots(2, 2, figsize=(12, 8))
        shape = (len(fpts), len(gainpts))
        extent = (gainpts.min(), gainpts.max(), fpts.min(), fpts.max())
        self._live_metric_images = []
        for axis, title in zip(
            self._live_metrics_ax.ravel(),
            ("Fidelity", "Fit fidelity", "Tm/T1", "Excited/Ground"),
        ):
            image = axis.imshow(
                np.full(shape, np.nan), origin="lower", aspect="auto",
                interpolation="none", extent=extent,
            )
            axis.set_title(title)
            axis.set_xlabel("Readout gain")
            axis.set_ylabel("Frequency (MHz)")
            self._live_metrics_fig.colorbar(image, ax=axis)
            self._live_metric_images.append(image)
        self._live_metrics_fig.tight_layout()
        self._live_metrics_fig.show()

        self._live_shot_fig, self._live_shot_ax = plt.subplots(1, 2, figsize=(10, 4))
        self._live_hist_g, = self._live_shot_ax[0].plot([], [], label="g")
        self._live_hist_e, = self._live_shot_ax[0].plot([], [], label="e/f")
        self._live_blob_g, = self._live_shot_ax[1].plot([], [], ".", alpha=0.2, label="g")
        self._live_blob_e, = self._live_shot_ax[1].plot([], [], ".", alpha=0.2, label="e/f")
        self._live_shot_ax[0].set_title("Latest rotated-I histogram")
        self._live_shot_ax[1].set_title("Latest IQ blobs")
        self._live_shot_ax[0].legend()
        self._live_shot_ax[1].legend()
        self._live_shot_fig.tight_layout()
        self._live_shot_fig.show()

    def _update_live_plots(self, f_ind, g_ind, l_ind, fid, fit_fid, tm, pop_ratio, results, shot_data):
        for image, metric in zip(
            self._live_metric_images, (fid, fit_fid, tm, pop_ratio)
        ):
            values = np.asarray(image.get_array()).copy()
            values[:] = np.nan
            values[f_ind, g_ind] = metric[f_ind, g_ind, l_ind]
            image.set_data(values)

        angle = np.deg2rad(results["angle"])
        ig = np.asarray(shot_data["Ig"])
        qg = np.asarray(shot_data["Qg"])
        excited_i = "If" if self.cfg.expt.check_f and "If" in shot_data else "Ie"
        excited_q = "Qf" if self.cfg.expt.check_f and "Qf" in shot_data else "Qe"
        ie = np.asarray(shot_data[excited_i])
        qe = np.asarray(shot_data[excited_q])
        rotated_g = ig * np.cos(angle) + qg * np.sin(angle)
        rotated_e = ie * np.cos(angle) + qe * np.sin(angle)
        bins = np.linspace(
            min(rotated_g.min(), rotated_e.min()),
            max(rotated_g.max(), rotated_e.max()),
            60,
        )
        centers = (bins[:-1] + bins[1:]) / 2
        self._live_hist_g.set_data(centers, np.histogram(rotated_g, bins=bins, density=True)[0])
        self._live_hist_e.set_data(centers, np.histogram(rotated_e, bins=bins, density=True)[0])
        self._live_blob_g.set_data(ig, qg)
        self._live_blob_e.set_data(ie, qe)
        self._live_metrics_fig.canvas.draw_idle()
        self._live_shot_fig.canvas.draw_idle()
        self._live_metrics_fig.canvas.flush_events()
        self._live_shot_fig.canvas.flush_events()
        plt.pause(0.01)

    def analyze(self, data=None, fit=True, low_gain=True, **kwargs):
        """Find the fidelity-maximizing point (preferring lower gain when nearly equal) and store it."""
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
        """Plot fidelity vs frequency for each (gain, length) combination, plus optional fit-parameter maps."""
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
            """Choose a subplot grid (rows, cols) for n panels, max 5 per row."""
            row = int(np.ceil(n / 5))
            if n < 5:
                col = n
            else:
                col = 5
            return row, col

        title = f"Single Shot Optimization Q{self.cfg.expt.qubit[0]}"

        def return_dim(data, dim, i):
            """Slice the 3D fidelity array along the non-swept dimensions at index i."""
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

        def rotated_i(i_data, q_data, angle_deg):
            """Rotate IQ data by angle_deg (as fit by HistogramExperiment.analyze) and return I."""
            theta = np.pi * angle_deg / 180
            i_rot, _ = helpers.rotate_coordinates(np.asarray(i_data), np.asarray(q_data), theta)
            return i_rot

        def plot_hist(axis, ig, qg, ie, qe, angle_deg, exc_label="e"):
            """Plot rotated-I histograms of the g and excited (e or f) clouds, overlaying
            the two total fits and the four fit components (ground-level and excited-level).
            When exc_label=="f" the passed ie/qe are the f-state shots, so this shows g-f."""
            d = {
                "Ig": np.asarray(ig), "Qg": np.asarray(qg),
                "Ie": np.asarray(ie), "Qe": np.asarray(qe),
            }
            try:
                dr, _, _, _ = helpers.fit_single_shot(d, plot=False)
            except Exception:
                # Fall back to plain rotated histograms if the fit fails
                axis.hist(rotated_i(ig, qg, angle_deg), bins=50, color=blue, alpha=0.5, label="g")
                axis.hist(rotated_i(ie, qe, angle_deg), bins=50, color=red, alpha=0.5, label=exc_label)
                return

            # Rotated histograms (density scale, matching the fit curves)
            axis.fill_between(dr["vhg"], dr["histg"], color=blue, alpha=0.3, label="g")
            axis.fill_between(dr["vhe"], dr["histe"], color=red, alpha=0.3, label=exc_label)

            # Total fits
            axis.plot(dr["vhg"], helpers.combined_state_model(dr["vhg"], *dr["paramsg2"]), "k", linewidth=1)
            axis.plot(
                dr["vhe"],
                helpers.combined_state_model(dr["vhe"], dr["mag_ground_e"], dr["vg"], dr["ve"], dr["sigma"], dr["tm"]),
                "k", linewidth=1,
            )

            # Four components: ground-level and excited-level of each fit
            # (g-shot uses its own tm so components match the total)
            for comp, ccol in helpers.combined_fit_components(dr["vhg"], dr["paramsg2"][0], dr, tm=dr["paramsg2"][4]):
                axis.plot(dr["vhg"], comp, color=ccol, linewidth=1, linestyle="--")
            for comp, ccol in helpers.combined_fit_components(dr["vhe"], dr["mag_ground_e"], dr):
                axis.plot(dr["vhe"], comp, color=ccol, linewidth=1, linestyle="--")

        def plot_blob(axis, ig, qg, ie, qe):
            """Scatter every g and e shot in the IQ plane on axis."""
            axis.plot(ig, qg, ".", color=blue, alpha=0.2, markersize=0.5, rasterized=True)
            axis.plot(ie, qe, ".", color=red, alpha=0.2, markersize=0.5, rasterized=True)

        imname = Path(self.fname).stem
        folder = Path(getattr(self, "metadata_path", None) or Path(self.path) / "images")
        folder.mkdir(parents=True, exist_ok=True)

        # When the f state was measured, show the g-f pair (matches the recorded
        # fidelity/metrics); otherwise g-e.
        check_f = "If" in self.data
        exc_i_key = "If" if check_f else "Ie"
        exc_q_key = "Qf" if check_f else "Qe"
        exc_label = "f" if check_f else "e"

        if ndims == 1:
            for mode in ("hist", "blob"):
                row, col = smart_ax(npts[0])
                fig, ax = plt.subplots(row, col, figsize=(col * 3, row * 3))
                ax = ax.flatten()
                for i in range(npts[0]):
                    ig = return_dim(self.data["Ig"], inds, i)
                    qg = return_dim(self.data["Qg"], inds, i)
                    ie = return_dim(self.data[exc_i_key], inds, i)
                    qe = return_dim(self.data[exc_q_key], inds, i)
                    if mode == "hist":
                        angle_i = return_dim(self.data["angle"], inds, i)[0]
                        plot_hist(ax[i], ig, qg, ie, qe, angle_i, exc_label=exc_label)
                    else:
                        plot_blob(ax[i], ig, qg, ie, qe)
                    ax[i].set_title(f"{labs[inds[0]]} {data[sweep_var[0]][i]:.2f}")
                fig.savefig(folder / f"{imname}_raw_{mode}.png")

        elif ndims == 2:
            for mode in ("hist", "blob"):
                fig, ax = plt.subplots(npts[0], npts[1], figsize=(npts[1] * 3, npts[0] * 3))
                for i in range(npts[0]):
                    for j in range(npts[1]):
                        ig = return_dim(self.data["Ig"], inds, [i, j])
                        qg = return_dim(self.data["Qg"], inds, [i, j])
                        ie = return_dim(self.data[exc_i_key], inds, [i, j])
                        qe = return_dim(self.data[exc_q_key], inds, [i, j])
                        if mode == "hist":
                            angle_ij = return_dim(self.data["angle"], inds, [i, j])[0]
                            plot_hist(ax[i, j], ig, qg, ie, qe, angle_ij, exc_label=exc_label)
                        else:
                            plot_blob(ax[i, j], ig, qg, ie, qe)
                        if i == npts[0] - 1:
                            ax[i, j].set_xlabel(np.round(self.data[sweep_var[1]][j], 2))
                        if j == 0:
                            ax[i, j].set_ylabel(np.round(self.data[sweep_var[0]][i], 2))
                plt.figtext(0.5, 0.0, labs[inds[1]], horizontalalignment="center")
                plt.figtext(
                    0.0, 0.5, labs[inds[0]], verticalalignment="center", rotation="vertical"
                )
                fig.savefig(folder / f"{imname}_raw_{mode}.png")
        else:
            for k in range(npts[2]):
                for mode in ("hist", "blob"):
                    fig, ax = plt.subplots(
                        npts[0], npts[1], figsize=(npts[1] * 3, npts[0] * 3)
                    )
                    for i in range(npts[0]):
                        for j in range(npts[1]):
                            ig = self.data["Ig"][i, j, k, :]
                            qg = self.data["Qg"][i, j, k, :]
                            ie = self.data[exc_i_key][i, j, k, :]
                            qe = self.data[exc_q_key][i, j, k, :]
                            if mode == "hist":
                                angle_ijk = self.data["angle"][i, j, k]
                                plot_hist(ax[i, j], ig, qg, ie, qe, angle_ijk, exc_label=exc_label)
                            else:
                                plot_blob(ax[i, j], ig, qg, ie, qe)
                            if i == npts[0] - 1:
                                ax[i, j].set_xlabel(np.round(self.data[sweep_var[1]][j], 2))
                            if j == 0:
                                ax[i, j].set_ylabel(np.round(self.data[sweep_var[0]][i], 2))

                    # Add figure title with constant parameter value
                    const_param_value = data[sweep_var[2]][k]
                    fig.suptitle(f"{title}, {labs[inds[2]]} = {const_param_value:.2f}")

                    # Add figure-level axis labels
                    plt.figtext(0.5, 0.0, labs[inds[1]], horizontalalignment="center")
                    plt.figtext(
                        0.0, 0.5, labs[inds[0]], verticalalignment="center", rotation="vertical"
                    )

                    fig.tight_layout()
                    fig.savefig(folder / f"{imname}_raw_{mode}_{k}.png")

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

        self.plot_metrics(data)

        self.save_config()

    def plot_metrics(self, data=None):
        """Plot fidelity, fit fidelity, Tm/T1, and the no-pi excited/ground ratio
        against the swept parameter(s).

        The layout adapts to how many of expts_f / expts_gain / expts_len are > 1:
        - 1 swept param  -> four line plots (metric vs param)
        - 2 swept params -> four 2D color maps (metric as color vs the two params)
        - 3 swept params -> four 3D scatter heatmaps (metric as color vs the three params)
        """
        if data is None:
            data = self.data

        # Identify which parameters were actually swept, in (freq, gain, length) order
        sweeps = [
            ("fpts", "Frequency (MHz)"),
            ("gainpts", "Gain"),
            ("lenpts", "Readout length ($\mu$s)"),
        ]
        active = [(key, lab) for key, lab in sweeps if len(data[key]) > 1]
        ndims = len(active)
        if ndims == 0:
            print("No parameter swept (all expts == 1); nothing to plot in plot_metrics.")
            return

        # Older saved data may predate these keys; fall back to NaNs so the plot still renders
        if "fit_fid" not in data:
            data["fit_fid"] = np.full_like(data["fid"], np.nan)
        if "pop_ratio" not in data:
            data["pop_ratio"] = np.full_like(data["fid"], np.nan)

        metrics = [
            (100 * data["fid"], "Fidelity [%]"),
            (100 * data["fit_fid"], "Fit fidelity [%]"),
            (data["tm"], "$T_m/T_1$"),
            (data["pop_ratio"], "Excited/Ground (no $\\pi$)"),
        ]

        title = f"Single Shot Optimization Q{self.cfg.expt.qubit[0]}"
        imname = Path(self.fname).stem
        folder = Path(getattr(self, "metadata_path", None) or Path(self.path) / "images")
        folder.mkdir(parents=True, exist_ok=True)

        if ndims == 1:
            xkey, xlab = active[0]
            xval = data[xkey]
            fig, axes = plt.subplots(1, 4, figsize=(20, 4))
            for ax, (m, mlab) in zip(axes, metrics):
                ax.plot(xval, np.squeeze(m), "o-", color=blue)
                ax.set_xlabel(xlab)
                ax.set_ylabel(mlab)
        elif ndims == 2:
            (xkey, xlab), (ykey, ylab) = active
            xval = data[xkey]
            yval = data[ykey]
            fig, axes = plt.subplots(1, 4, figsize=(21, 4.5))
            for ax, (m, mlab) in zip(axes, metrics):
                # squeezed shape is (len(xval), len(yval)); pcolormesh expects (len(yval), len(xval))
                C = np.squeeze(m)
                pc = ax.pcolormesh(xval, yval, C.T, shading="auto", cmap="viridis")
                fig.colorbar(pc, ax=ax, label=mlab)
                ax.set_xlabel(xlab)
                ax.set_ylabel(ylab)
        else:
            (xkey, xlab), (ykey, ylab), (zkey, zlab) = active
            xval, yval, zval = data[xkey], data[ykey], data[zkey]
            X, Y, Z = np.meshgrid(xval, yval, zval, indexing="ij")
            fig = plt.figure(figsize=(22, 5))
            for i, (m, mlab) in enumerate(metrics):
                ax = fig.add_subplot(1, 4, i + 1, projection="3d")
                sc = ax.scatter(
                    X.ravel(), Y.ravel(), Z.ravel(), c=np.asarray(m).ravel(), cmap="viridis"
                )
                fig.colorbar(sc, ax=ax, label=mlab, shrink=0.6)
                ax.set_xlabel(xlab)
                ax.set_ylabel(ylab)
                ax.set_zlabel(zlab)

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(folder / f"{imname}_metrics.png")
        plt.show()

    def check_edges(self):
        """Return True when the fidelity maximum sits on the sweep edge, i.e. the sweep should be extended."""
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
        """Write the optimal readout gain, length, and frequency back to the config file."""
        qi = self.cfg.expt.qubit[0]
        cfg_file= self.config_file
        config.update_readout(cfg_file, "gain", self.data["gain"], qi, verbose=verbose)
        config.update_readout(
            cfg_file, "readout_length", self.data["length"], qi, verbose=verbose
        )
        config.update_readout(
            cfg_file, "frequency", self.data["freq"], qi, verbose=verbose
        )
