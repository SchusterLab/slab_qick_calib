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

from ..single_qubit.t1 import T1FastFlux
from .t1_predistorted import T1Predist


class T1PredistFastFlux(T1FastFlux):
    """
    Sweep T1 across flux gain values using predistorted flux pulses.

    Identical to T1FastFlux except the inner experiment is T1Predist instead
    of T1Experiment, so each wait-time point gets a correctly-sized
    predistorted arb envelope.

    All constructor parameters and sweep parameters are the same as T1FastFlux.
    All other params are forwarded to T1Predist (flux_alphas, flux_taus, etc.).
    """

    _inner_expt_class = T1Predist

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
        super().__init__(
            cfg_dict=cfg_dict, qi=qi, go=go, params=params,
            prefix=prefix, progress=progress, display=display,
        )


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
        """
        Run repeated T1PredistFastFlux sweeps until n_repeats / duration_hours / interrupt.

        Each iteration runs a fresh inner sweep, appends the fitted T1 row to
        the heatmap data, live-plots it, writes a per-sweep CSV, and saves the
        master HDF5.

        Returns:
            dict: Data with t1_list (2D: sweeps × gain points), timestamps
            (hours elapsed), gain_pts, and freq_pts
        """
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
        """Refresh the live Jupyter heatmap (frequency × elapsed time, color = T1) after a sweep."""
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
        """Plot the accumulated T1(frequency, time) heatmap with robust color limits."""
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
        self.save_config()
