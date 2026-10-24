"""JPA DC-bias scan experiments."""

import time

import matplotlib.pyplot as plt
import numpy as np

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment
from ..single_qubit.resonator_spectroscopy import ResSpec
from .config_update import should_update_yaml, update_jpa_config


XLABEL = "Waste Readout Frequency (MHz)"


def _phase_with_linear_slope_removed(phases, xpts=None, fit_fraction=None):
    phases = np.unwrap(np.asarray(phases), axis=-1)
    xpts = np.arange(phases.shape[-1]) if xpts is None else np.asarray(xpts)
    nfit = len(xpts) if fit_fraction is None else max(2, int(len(xpts) * fit_fraction))
    corrected = np.empty_like(phases, dtype=float)

    for ind in np.ndindex(phases.shape[:-1]):
        row = phases[ind]
        coeff = np.polyfit(xpts[:nfit], row[:nfit], 1)
        corrected[ind] = row - np.polyval(coeff, xpts)

    return corrected


class JPABiasScan(QickExperiment):
    """
    Sweep the JPA DC bias and measure the waste/readout resonator response.

    This is the class-based version of the old notebook DC-bias scan: for each
    bias voltage it sets the RF board DC output, waits, and runs a 1D ResSpec
    around the waste resonance.
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=True,
        display=True,
        save=True,
        analyze=True,
        qi=1,
        go=True,
        params=None,
        style="fine",
        check_params=True,
    ):
        if params is None:
            params = {}
        if not prefix:
            prefix = f"jpa_bias_scan_qubit{qi}"

        super().__init__(
            cfg_dict=cfg_dict,
            prefix=prefix,
            progress=progress,
            qi=qi,
            check_params=check_params,
        )
        self.cfg_dict = cfg_dict
        self.qi = qi

        inner_defaults = {
            "qubit": [qi],
            "center": self.cfg.device.readout.frequency[qi],
            "span": 5.0,
            "expts": 220,
            "length": self.cfg.device.readout.readout_length[qi],
            "gain": self.cfg.device.readout.gain[qi],
            "reps": self.reps,
            "rounds": self.rounds,
            "final_delay": 5,
            "pulse_e": False,
            "pulse_f": False,
            "long_pulse": False,
            "loop": False,
            "phase_const": False,
            "reset_mode": "passive",
            "flux": False,
        }
        dc_defaults = {
            "dc_start": 112e-3,
            "dc_stop": 122e-3,
            "expts_dc": 21,
            # The JPA has its own DC bias channel; it does not use the qubit
            # flux line, which may not exist at all.
            "bias_chan": self.cfg.hw.soc.dacs.jpa.jpa_dc_ch[qi],
            "jpa_dc_val": 117e-3,
            "dc_settle": 0.25,
            "dc_verify_print": True,
            "reset_bias_after": True,
            "reset_bias": 0.0,
            "remove_phase_slope": True,
            "rspec_update": False,
            "update": False,
            "rspec_update_verbose": True,
            "pump_ch": 13,
            "pump_nqz": 2,
            "pump_type": "int",
            "pump_freq_mhz": self.cfg.device.readout.frequency[qi] + 2.0,
            "pump_if_mhz": 400.0,
            "pump_filter_bw_mhz": 1000.0,
            "pump_filter_type": "bandpass",
            "pump_attn1_db": 15.0,
            "pump_attn2_db": 0.0,
            "pump_gain": np.nan,
        }

        p = {**inner_defaults, **dc_defaults, **params}
        p["pump_freq"] = float(p.get("pump_freq_mhz", p["pump_freq_mhz"]))
        p["pump_freq_mhz"] = float(p["pump_freq"])
        p["pump_mixer_freq"] = float(p.get("pump_mixer_freq", p["pump_freq_mhz"] - p["pump_if_mhz"]))
        p["pump_mixer_freq_mhz"] = float(p["pump_mixer_freq"])
        if "center" in p:
            p["start"] = p["center"] - p["span"] / 2.0

        self.cfg.expt = AttrDict(p)
        self.xlabel = XLABEL
        self.ylabel = "JPA DC Bias (mV)"

        if go:
            self.run(progress=progress, analyze=analyze, display=display, save=save)

    def _res_params(self):
        p = dict(self.cfg.expt)
        for key in (
            "dc_start",
            "dc_stop",
            "expts_dc",
            "bias_chan",
            "dc_settle",
            "dc_verify_print",
            "reset_bias_after",
            "reset_bias",
            "remove_phase_slope",
            "rspec_update",
            "update",
            "rspec_update_verbose",
            "jpa_dc_val",
            "pump_ch",
            "pump_nqz",
            "pump_type",
            "pump_freq",
            "pump_freq_mhz",
            "pump_if_mhz",
            "pump_mixer_freq",
            "pump_mixer_freq_mhz",
            "pump_filter_bw_mhz",
            "pump_filter_type",
            "pump_attn1_db",
            "pump_attn2_db",
            "pump_gain",
        ):
            p.pop(key, None)
        p["start"] = p["center"] - p["span"] / 2.0 if "center" in p else p["start"]
        return p

    def acquire(self, progress=False):
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = lambda x, disable=False: x

        soc = self.im[self.cfg.aliases.soc]
        bias_chan = int(self.cfg.expt.bias_chan)
        dc_biases = np.linspace(
            float(self.cfg.expt.dc_start),
            float(self.cfg.expt.dc_stop),
            int(self.cfg.expt.expts_dc),
        )

        rows = []
        readbacks = []
        times = []
        res_params = self._res_params()

        try:
            for dc_val in tqdm(dc_biases, disable=not progress):
                dc_val = float(dc_val)
                soc.rfb_set_bias(bias_chan, dc_val)
                if self.cfg.expt.dc_settle > 0:
                    time.sleep(float(self.cfg.expt.dc_settle))

                rb = np.nan
                try:
                    rb = float(soc.rfb_get_bias(bias_chan))
                except Exception:
                    pass
                if self.cfg.expt.dc_verify_print:
                    print(f"[JPA bias] ch {bias_chan}: set {dc_val*1e3:.3f} mV, read {rb*1e3:.3f} mV")

                row_expt = ResSpec(
                    self.cfg_dict,
                    qi=self.qi,
                    go=False,
                    params={**res_params, "jpa_dc_bias": dc_val},
                    style="fine",
                    check_params=False,
                )
                row_expt.acquire(progress=False)
                row_expt.analyze(fit=False)
                rows.append(row_expt.data)
                readbacks.append(rb)
                times.append(time.time())
        finally:
            if self.cfg.expt.reset_bias_after:
                soc.rfb_set_bias(bias_chan, float(self.cfg.expt.reset_bias))

        data = {
            "dc_biases_v": dc_biases,
            "dc_biases_mv": 1e3 * dc_biases,
            "dc_readback_v": np.asarray(readbacks),
            "time": np.asarray(times),
        }
        for key in rows[0]:
            data[key] = np.asarray([row[key] for row in rows])

        data["freq_mhz"] = np.asarray(data["xpts"][0])
        data["xpts"] = data["freq_mhz"]
        data["amps_db"] = 20 * np.log10(np.maximum(np.asarray(data["amps"]), 1e-12))
        if "phase_fix" in data:
            data["freq_phase_mhz"] = data["freq_mhz"][1:-1]
            data["phase_fix_corrected"] = _phase_with_linear_slope_removed(
                data["phase_fix"],
                xpts=data["freq_phase_mhz"],
                fit_fraction=0.1,
            )
        if self.cfg.expt.remove_phase_slope:
            data["phases_corrected"] = _phase_with_linear_slope_removed(data["phases"])

        self.data = data
        if should_update_yaml(self.cfg.expt):
            update_jpa_config(
                self.config_file,
                self.cfg.expt,
                data={"best_dc_bias_v": float(self.cfg.expt.jpa_dc_val)},
                verbose=bool(self.cfg.expt.rspec_update_verbose),
            )
        return data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data
        amp = np.asarray(data["amps"])
        imin = np.nanargmin(amp, axis=1)
        data["res_freq_mhz"] = data["freq_mhz"][imin]
        data["res_amp"] = amp[np.arange(len(imin)), imin]
        self.data = data
        return data

    def display(self, data=None, plot="phase", cmap=None, **kwargs):
        if data is None:
            data = self.data

        if plot == "amp":
            z = data["amps_db"]
            x = data["freq_mhz"]
            cbar = "Amplitude (dB)"
            cmap = "viridis" if cmap is None else cmap
        elif "phase_fix_corrected" in data:
            z = data["phase_fix_corrected"]
            x = data["freq_phase_mhz"]
            cbar = "Phase (rad)"
            cmap = "RdBu" if cmap is None else cmap
        else:
            z = data.get("phases_corrected", data["phases"])
            x = data["freq_mhz"]
            cbar = "Phase (rad)"
            cmap = "RdBu" if cmap is None else cmap

        fig, ax = plt.subplots(figsize=(10, 6))
        mesh = ax.pcolormesh(
            x,
            data["dc_biases_mv"],
            z,
            shading="auto",
            cmap=cmap,
        )
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_title("JPA DC Bias Scan")
        fig.colorbar(mesh, ax=ax, label=cbar)
        fig.tight_layout()
        plt.show()
        return fig, ax
