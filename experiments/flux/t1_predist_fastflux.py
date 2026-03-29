"""
T1 vs flux gain sweep with predistorted flux pulses.

Runs a T1Predist measurement at each flux gain point, collecting full T1
decay curves with predistorted arb envelopes.  Mirrors the interface of
T1FastFlux but uses T1Predist as the inner experiment so each wait-time
point gets a correctly-sized predistorted envelope.

Classes:
- T1PredistFastFlux: Sweep flux gain, run full T1 decay at each point
- T1PredistFastFluxRepeated: Repeatedly run T1PredistFastFlux sweeps and build
  a T1(frequency, time) heatmap
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ...analysis import fitting as fitter
from ..general.qick_experiment import QickExperiment2DSimple
from .t1_predistorted import T1Predist


class T1PredistFastFlux(QickExperiment2DSimple):
    """
    Sweep T1 across flux gain values using predistorted flux pulses.

    Runs a T1Predist at each flux gain point, collecting T1 vs flux gain
    (and optionally vs frequency). Saves a single HDF5 file with
    save_interim after each gain point. Performs adaptive span adjustment
    between scans.

    Sweep parameters (passed via params dict):
        gain_start: Starting flux gain value (default: sweet_spot_ac)
        gain_stop: Ending flux gain value (default: sweet_spot_ac + 0.4)
        direction: 'pos' or 'neg' — sweep direction from sweet spot
        freq_span: Frequency span in MHz. Overrides gain_stop when a flux
            model is available.
        expts_gain: Number of gain points in the sweep
        lin_freq: If True and flux model available, space points linearly
            in frequency instead of linearly in gain.
        start_t1: Initial T1 estimate for setting the first span.

    All other params are forwarded to T1Predist (flux_alphas, flux_taus, etc.).
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
    ):
        if prefix is None:
            prefix = f"t1_predist_fastflux_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, qi=qi)

        # Load flux model from config
        cfg_flux = getattr(
            getattr(getattr(getattr(self.cfg, "hw", None), "soc", None), "dacs", None),
            "flux", None)
        cfg_qubit = getattr(getattr(self.cfg, "device", None), "qubit", None)

        sweet_spot = 0.0
        if cfg_qubit is not None and hasattr(cfg_qubit, "sweet_spot_ac"):
            sweet_spot = cfg_qubit.sweet_spot_ac[qi]

        direction = params.get("direction", "pos")
        sign = 1 if direction == "pos" else -1

        converter = params.get("flux_converter", None)
        if converter is None:
            converter = fitter.flux_converter_from_config(cfg_flux, cfg_qubit, qi, direction)
        freq_span = params.get("freq_span", None)

        if freq_span is not None and converter is not None and "gain_stop" not in params:
            f_sweet = converter.gain_to_freq(sweet_spot)
            f_target = f_sweet - freq_span
            gain_stop = float(converter.freq_to_gain(f_target))
        elif "gain_stop" not in params:
            gain_stop = sweet_spot + sign * 0.4
        else:
            gain_stop = params["gain_stop"]

        start_t1 = params.pop("start_t1", None)

        params_def = {
            "gain_start": sweet_spot,
            "gain_stop": gain_stop,
            "start": 0.05,
            "expts": 25,
            "expts_gain": 50,
            "freq_span": freq_span,
            "direction": direction,
            "lin_freq": True,
            "t1_max": float("inf"),
        }

        if start_t1 is not None:
            params_def["span"] = 4.1 * start_t1
        params = {**params_def, **params, "flux": True}

        # Create inner T1Predist experiment (go=False)
        self.expt = T1Predist(cfg_dict, qi, go=False, params=params, check_params=False)
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = {**params_def, **params}
        self.expt.cfg.expt = {**self.expt.cfg.expt, **params}

        self._qi = qi
        self._cfg_dict = cfg_dict
        self._flux_converter = converter

        # Build gain_pts and freq_pts
        cfg_e = self.cfg.expt
        if cfg_e["lin_freq"] and converter is not None:
            freq_start = converter.gain_to_freq(cfg_e["gain_start"])
            freq_stop = converter.gain_to_freq(cfg_e["gain_stop"])
            self.freq_pts = np.linspace(float(freq_start), float(freq_stop), cfg_e["expts_gain"])
            self.gain_pts = converter.freq_to_gain(self.freq_pts)
        else:
            self.gain_pts = np.linspace(cfg_e["gain_start"], cfg_e["gain_stop"], cfg_e["expts_gain"])
            if converter is not None:
                self.freq_pts = converter.gain_to_freq(self.gain_pts)
            else:
                self.freq_pts = None

        if go:
            self.acquire(progress=progress, display=display)
            self.display()

    def acquire(self, progress=True, display=True):
        data = {"time": []}
        t1_list = []
        offset_list = []
        amp_list = []
        q_offset_list = []
        q_amp_list = []
        s_offset_list = []
        s_amp_list = []
        timestamp_list = []
        elapsed_list = []
        t_acq_start = time.time()
        csv_path = Path(self.fname).with_suffix(".csv")

        for i, g in enumerate(tqdm(self.gain_pts, disable=not progress)):
            self.expt.cfg.expt["flux_gain"] = float(g)

            data_new = self.expt.acquire(progress=False)
            self.expt.analyze(data=data_new, verbose=False, rescale=True)
            if display:
                self.expt.display(data=data_new)

            for key in ("avgi", "avgq", "amps", "phases", "xpts", "scale_data"):
                if key in data_new:
                    if i == 0:
                        data[key] = []
                    data[key].append(data_new[key])
            now = time.time()
            data["time"].append(now)
            timestamp_list.append(datetime.fromtimestamp(now).isoformat())
            elapsed_list.append(now - t_acq_start)

            best_fit = data_new.get("best_fit", [np.nan, np.nan, np.nan])
            t1_val = best_fit[2]
            t1_list.append(t1_val)
            offset_list.append(best_fit[0])
            amp_list.append(best_fit[1])
            fit_q = data_new.get("fit_avgq", [np.nan, np.nan, np.nan])
            q_offset_list.append(fit_q[0])
            q_amp_list.append(fit_q[1])
            fit_s = data_new.get("fit_scale_data", [np.nan, np.nan, np.nan])
            if fit_s is not None:
                s_offset_list.append(fit_s[0])
                s_amp_list.append(fit_s[1])
            else:
                s_offset_list.append(np.nan)
                s_amp_list.append(np.nan)

            if np.isfinite(t1_val) and t1_val > 0:
                new_span = 3.7 * min(t1_val, self.cfg.expt.t1_max)
                self.expt.cfg.expt["span"] = max(0.1, new_span)

            data["gain_pts"] = self.gain_pts
            if self.freq_pts is not None:
                data["freq_pts"] = self.freq_pts
            data["t1_list"] = np.array(t1_list)
            data["offset_list"] = np.array(offset_list)
            data["amp_list"] = np.array(amp_list)
            data["q_offset_list"] = np.array(q_offset_list)
            data["q_amp_list"] = np.array(q_amp_list)
            data["s_offset_list"] = np.array(s_offset_list)
            data["s_amp_list"] = np.array(s_amp_list)

            if self.save_interim:
                self.save_data(data=data)

            gains_so_far = self.gain_pts[: i + 1]
            if self.freq_pts is not None:
                num_cols = np.column_stack((gains_so_far, self.freq_pts[: i + 1],
                    data["t1_list"], data["offset_list"], data["amp_list"],
                    data["q_offset_list"], data["q_amp_list"],
                    np.array(elapsed_list)))
                header = "timestamp,gain,freq,t1,offset,amplitude,q_offset,q_amplitude,elapsed_s"
            else:
                num_cols = np.column_stack((gains_so_far,
                    data["t1_list"], data["offset_list"], data["amp_list"],
                    data["q_offset_list"], data["q_amp_list"],
                    np.array(elapsed_list)))
                header = "timestamp,gain,t1,offset,amplitude,q_offset,q_amplitude,elapsed_s"
            with open(csv_path, "w") as f_csv:
                f_csv.write(header + "\n")
                for row_idx in range(num_cols.shape[0]):
                    ts = timestamp_list[row_idx]
                    nums = ",".join(f"{v}" for v in num_cols[row_idx])
                    f_csv.write(f"{ts},{nums}\n")

        for k, a in data.items():
            data[k] = np.array(a)
        self.data = data
        return data

    def display(self, data=None, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        gain_pts = data["gain_pts"]
        t1_list = data["t1_list"]

        fig, ax = plt.subplots()
        t1_arr = np.array(t1_list, dtype=float)
        t1_mask = np.isfinite(t1_arr)
        if t1_mask.sum() > 2:
            q1, q3 = np.percentile(t1_arr[t1_mask], [25, 75])
            iqr = q3 - q1
            t1_inlier = t1_mask & (t1_arr >= q1 - 1.5 * iqr) & (t1_arr <= q3 + 1.5 * iqr)
            ax.plot(gain_pts[t1_inlier], t1_arr[t1_inlier], ".-")
            ax.plot(gain_pts[~t1_inlier & t1_mask], t1_arr[~t1_inlier & t1_mask], "rx", ms=6)
            inlier_vals = t1_arr[t1_inlier]
            if len(inlier_vals) > 0:
                ymin, ymax = inlier_vals.min(), inlier_vals.max()
                margin = (ymax - ymin) * 0.1 if ymax > ymin else abs(ymax) * 0.1 or 0.1
                ax.set_ylim(ymin - margin, ymax + margin)
        else:
            ax.plot(gain_pts, t1_list, ".-")
        ax.set_xlabel("Flux Gain")
        ax.set_ylabel("$T_1$ ($\\mu$s)")
        self.save_fig(fig, suffix="_t1_vs_gain")

        if "freq_pts" in data:
            freq_pts = data["freq_pts"]

            fig, ax = plt.subplots()
            t1_arr = np.array(t1_list, dtype=float)
            t1_mask = np.isfinite(t1_arr)
            if t1_mask.sum() > 2:
                q1, q3 = np.percentile(t1_arr[t1_mask], [25, 75])
                iqr = q3 - q1
                t1_inlier = t1_mask & (t1_arr >= q1 - 1.5 * iqr) & (t1_arr <= q3 + 1.5 * iqr)
                ax.plot(freq_pts[t1_inlier], t1_arr[t1_inlier], ".-")
                ax.plot(freq_pts[~t1_inlier & t1_mask], t1_arr[~t1_inlier & t1_mask], "rx", ms=6)
                inlier_vals = t1_arr[t1_inlier]
                if len(inlier_vals) > 0:
                    ymin, ymax = inlier_vals.min(), inlier_vals.max()
                    margin = (ymax - ymin) * 0.1 if ymax > ymin else abs(ymax) * 0.1 or 0.1
                    ax.set_ylim(ymin - margin, ymax + margin)
            else:
                ax.plot(freq_pts, t1_list, ".-")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("$T_1$ ($\\mu$s)")
            self.save_fig(fig, suffix="_t1_vs_freq")

            fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
            panels = [
                (axes[0, 0], data["amp_list"], "I Amplitude"),
                (axes[0, 1], data["offset_list"], "I Offset"),
                (axes[1, 0], data["q_amp_list"], "Q Amplitude"),
                (axes[1, 1], data["q_offset_list"], "Q Offset"),
                (axes[2, 0], data.get("s_amp_list", []), "Rescaled Amplitude"),
                (axes[2, 1], data.get("s_offset_list", []), "Rescaled Offset"),
            ]
            for ax, vals, label in panels:
                vals = np.array(vals, dtype=float)
                if vals.size == 0:
                    ax.set_ylabel(label)
                    continue
                mask = np.isfinite(vals)
                if mask.sum() > 2:
                    q1, q3 = np.percentile(vals[mask], [25, 75])
                    iqr = q3 - q1
                    inlier = mask & (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
                    ax.plot(freq_pts[inlier], vals[inlier], ".-")
                    ax.plot(freq_pts[~inlier & mask], vals[~inlier & mask], "rx", ms=6)
                    inlier_vals = vals[inlier]
                    if len(inlier_vals) > 0:
                        ymin, ymax = inlier_vals.min(), inlier_vals.max()
                        margin = (ymax - ymin) * 0.1 if ymax > ymin else abs(ymax) * 0.1 or 0.1
                        ax.set_ylim(ymin - margin, ymax + margin)
                else:
                    ax.plot(freq_pts, vals, ".-")
                ax.set_ylabel(label)
            axes[2, 0].set_xlabel("Frequency (MHz)")
            axes[2, 1].set_xlabel("Frequency (MHz)")
            self.save_fig(fig, suffix="_fit_params")

        # Waterfall plot
        if "scale_data" in data:
            curves = data["scale_data"]
        elif "avgi" in data:
            curves = data["avgi"]
        else:
            curves = None

        if curves is not None and len(curves) > 0:
            xpts_list = data.get("xpts", None)
            use_freq = "freq_pts" in data and data["freq_pts"] is not None
            label_pts = data["freq_pts"] if use_freq else gain_pts

            fig, ax = plt.subplots(figsize=(8, max(6, len(curves) * 0.25)))
            for i, curve in enumerate(curves):
                y = np.array(curve, dtype=float)
                x = np.array(xpts_list[i], dtype=float) if xpts_list is not None else np.arange(len(y))
                ymin, ymax = np.nanmin(y), np.nanmax(y)
                if ymax > ymin:
                    y_norm = (y - ymin) / (ymax - ymin)
                else:
                    y_norm = np.zeros_like(y)
                ax.plot(x, y_norm + i, "-", lw=0.8)
            ax.set_xlabel("Time ($\\mu$s)")
            ax.set_ylabel("Frequency (MHz)" if use_freq else "Flux Gain")
            n = len(curves)
            max_ticks = 20
            step = max(1, n // max_ticks)
            tick_idx = np.arange(0, n, step)
            ax.set_yticks(tick_idx)
            ax.set_yticklabels([f"{label_pts[j]:.1f}" for j in tick_idx])
            ax.set_ylim(-0.5, n - 0.5)
            self.save_fig(fig, suffix="_waterfall")


class T1PredistFastFluxRepeated(T1PredistFastFlux):
    """
    Repeatedly run T1PredistFastFlux sweeps and build a T1(frequency, time) heatmap.

    Runs T1PredistFastFlux N times (or for a specified duration), accumulating
    T1 vs frequency data. After each sweep, updates a live 2D heatmap
    (x=frequency, y=elapsed time, color=T1) and appends to a master CSV.

    Extra parameters (passed via params dict, in addition to T1PredistFastFlux params):
        n_repeats: Number of sweeps to run (default: None = run forever)
        duration_hours: Max duration in hours (default: None = no limit).
            If both n_repeats and duration_hours are given, stops at whichever
            comes first. If both None, runs until KeyboardInterrupt.
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
    ):
        self._n_repeats = params.pop("n_repeats", None)
        self._duration_hours = params.pop("duration_hours", None)

        if prefix is None:
            prefix = f"t1_predist_fastflux_repeated_qubit{qi}"

        super().__init__(
            cfg_dict=cfg_dict, qi=qi, go=False, params=params,
            prefix=prefix, progress=progress, display=display,
        )

        if go:
            self.acquire(progress=progress, display=display)
            self.display()

    def acquire(self, progress=False, display=True):
        try:
            from IPython.display import display as ipy_display, clear_output
            _has_ipy = True
        except ImportError:
            _has_ipy = False

        self.t1_list = []
        self.timestamps = []
        t_start = time.time()
        sweep_idx = 0

        csv_dir = Path(self.fname).with_suffix("") / "csvs"
        csv_dir.mkdir(parents=True, exist_ok=True)

        _repeated_keys = ("n_repeats", "duration_hours")
        sweep_params = {k: v for k, v in self.cfg.expt.items() if k not in _repeated_keys}

        try:
            while True:
                if self._n_repeats is not None and sweep_idx >= self._n_repeats:
                    break
                elapsed_h = (time.time() - t_start) / 3600
                if self._duration_hours is not None and elapsed_h > self._duration_hours:
                    break

                inner_progress = progress and not _has_ipy
                sweep = T1PredistFastFlux(
                    self._cfg_dict, qi=self._qi, go=False, params={**sweep_params},
                    progress=inner_progress, display=False,
                )
                sweep.gain_pts = self.gain_pts
                if self.freq_pts is not None:
                    sweep.freq_pts = self.freq_pts
                sweep.save_interim = False
                ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
                sweep.fname = str(csv_dir / f"sweep_{sweep_idx:04d}_{ts_label}")
                if _has_ipy:
                    print(f"Sweep {sweep_idx + 1}" +
                          (f"/{self._n_repeats}" if self._n_repeats else "") +
                          f" — {len(self.gain_pts)} gain points ...")
                data = sweep.acquire(progress=inner_progress, display=False)

                elapsed_h = (time.time() - t_start) / 3600
                self.t1_list.append(data["t1_list"].copy())
                self.timestamps.append(elapsed_h)

                # Carry adaptive span to next sweep
                last_t1 = data["t1_list"]
                finite_t1s = last_t1[np.isfinite(last_t1) & (last_t1 > 0)]
                if len(finite_t1s) > 0:
                    median_t1 = np.median(finite_t1s)
                    t1_max = self.cfg.expt.t1_max
                    sweep_params["span"] = 4.1 * min(median_t1, t1_max)

                self.data = {
                    "freq_pts": self.freq_pts,
                    "gain_pts": self.gain_pts,
                    "t1_list": np.array(self.t1_list),
                    "timestamps": np.array(self.timestamps),
                }
                self.save_data()

                if _has_ipy:
                    self._update_heatmap(clear_output, ipy_display)

                sweep_idx += 1

        except KeyboardInterrupt:
            print(f"\nInterrupted after {sweep_idx} sweep(s).")

        self.data = {
            "freq_pts": self.freq_pts,
            "gain_pts": self.gain_pts,
            "t1_list": np.array(self.t1_list) if self.t1_list else np.array([]).reshape(0, 0),
            "timestamps": np.array(self.timestamps),
        }
        return self.data

    def _update_heatmap(self, clear_output, ipy_display):
        import sys
        import io
        import matplotlib.pyplot as plt
        from IPython.display import Image

        try:
            t1_2d = np.array(self.t1_list)
            freq = self.freq_pts if self.freq_pts is not None else self.gain_pts
            times = np.array(self.timestamps)

            fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
            finite_vals = t1_2d[np.isfinite(t1_2d)]
            vmin, vmax = None, None
            if len(finite_vals) > 2:
                q1, q3 = np.percentile(finite_vals, [25, 75])
                iqr = q3 - q1
                inlier_vals = finite_vals[(finite_vals >= q1 - 1.5 * iqr) & (finite_vals <= q3 + 1.5 * iqr)]
                if len(inlier_vals) > 0:
                    vmin, vmax = inlier_vals.min(), inlier_vals.max()
            mesh = ax.pcolormesh(freq, times, t1_2d, cmap="viridis", shading="auto",
                                 vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_xlabel("Frequency (MHz)" if self.freq_pts is not None else "Flux Gain")
            ax.set_ylabel("Elapsed Time (hours)")
            plt.colorbar(mesh, ax=ax, label="$T_1$ ($\\mu$s)")
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            sys.stdout.flush()
            sys.stderr.flush()
            clear_output(wait=True)
            ipy_display(Image(buf.read()))
            print(f"Completed {len(self.t1_list)} sweep(s), {times[-1]:.2f} hours elapsed")
        except Exception as e:
            print(f"[LivePlot] Heatmap update failed: {e}")

    def display(self, data=None, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        t1_2d = data["t1_list"]
        if t1_2d.size == 0:
            print("No sweep data to display.")
            return

        freq = data.get("freq_pts", data["gain_pts"])
        times = data["timestamps"]

        fig, ax = plt.subplots(figsize=(10, 6))
        finite_vals = t1_2d[np.isfinite(t1_2d)]
        vmin, vmax = None, None
        if len(finite_vals) > 2:
            q1, q3 = np.percentile(finite_vals, [25, 75])
            iqr = q3 - q1
            inlier_vals = finite_vals[(finite_vals >= q1 - 1.5 * iqr) & (finite_vals <= q3 + 1.5 * iqr)]
            if len(inlier_vals) > 0:
                vmin, vmax = inlier_vals.min(), inlier_vals.max()
        mesh = ax.pcolormesh(freq, times, t1_2d, cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Frequency (MHz)" if "freq_pts" in data and data["freq_pts"] is not None else "Flux Gain")
        ax.set_ylabel("Elapsed Time (hours)")
        plt.colorbar(mesh, ax=ax, label="$T_1$ ($\\mu$s)")
        self.save_fig(fig, suffix="_heatmap")
