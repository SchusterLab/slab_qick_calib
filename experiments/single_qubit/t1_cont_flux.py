import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

from ...analysis import fitting as fitter
from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperimentLoop, DataProcessor
from ..general.qick_program import QickProgram


class T1ContFluxProgram(QickProgram):
    """
    Pulse sequence for T1 measurement with inline calibration at a fixed flux point.

    Per rep, executes:
    1. n_e short-wait measurements (pi + short flux pulse + readout) -> excited state proxy
    2. n_g long-wait measurements (pi + long flux pulse + readout) -> ground state proxy
    3. n_t1 T1 measurements (pi + flux pulse at wait_time + readout) -> T1 data
    """

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        # Standard readout and pi pulse
        super()._initialize(cfg, readout="standard")
        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        # Flux generator and pulses
        self.declare_gen(cfg.expt.flux_chan, nqz=1, mixer_freq=0)

        flux_base = {
            "chan": cfg.expt.flux_chan,
            "freq": 0,
            "phase": 0,
            "gain": cfg.expt.flux_gain,
            "type": "const",
        }

        super().make_pulse({**flux_base, "length": cfg.expt.calib_wait_short}, "flux_pulse_short")
        super().make_pulse({**flux_base, "length": cfg.expt.calib_wait_long}, "flux_pulse_long")
        super().make_pulse({**flux_base, "length": cfg.expt.wait_time}, "flux_pulse_t1")

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)

        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # n_e short-wait calibration measurements (excited state proxy)
        for i in range(cfg.expt.n_e):
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag=f"wait_flux_e_{i}")
            self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse_short", t=0)
            self.delay_auto(t=cfg.expt.flux_readout_wait, tag=f"wait_post_e_{i}")
            self.measure(cfg)
            self.delay_auto(t=cfg.expt["final_delay"] + 0.01, tag=f"final_delay_e_{i}")

        # n_g long-wait calibration measurements (ground state proxy)
        for i in range(cfg.expt.n_g):
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag=f"wait_flux_g_{i}")
            self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse_long", t=0)
            self.delay_auto(t=cfg.expt.flux_readout_wait, tag=f"wait_post_g_{i}")
            self.measure(cfg)
            self.delay_auto(t=cfg.expt["final_delay"] + 0.01, tag=f"final_delay_g_{i}")

        # n_t1 T1 measurements
        for i in range(cfg.expt.n_t1):
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag=f"wait_flux_t1_{i}")
            self.pulse(ch=cfg.expt.flux_chan, name="flux_pulse_t1", t=0)
            self.delay_auto(t=cfg.expt.flux_readout_wait, tag=f"wait_post_t1_{i}")
            self.measure(cfg)
            self.delay_auto(t=cfg.expt["final_delay"] + 0.01, tag=f"final_delay_t1_{i}")

    def measure(self, cfg):
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
        self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)


class T1ContFluxExperiment(QickExperimentLoop):
    """
    T1 vs flux measurement with inline calibration.

    At each flux gain point, runs T1ContFluxProgram which performs
    short-wait (excited), long-wait (ground), and T1-wait measurements
    in a single program execution. This provides per-point calibration
    without separate calibration sweeps.

    Sweeps flux_gain on the x-axis, like T1FastFluxLoop.
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
            prefix = f"t1_cont_flux_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, qi=qi, check_params=False)

        # Load flux model from config
        cfg_flux = getattr(getattr(getattr(getattr(self.cfg, "hw", None), "soc", None), "dacs", None), "flux", None)
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

        params_def = {
            "gain_start": sweet_spot,
            "gain_stop": gain_stop,
            "expts_gain": 50,
            "reps": int(1.5 * self.reps),
            "rounds": self.rounds,
            "n_e": 3,
            "n_g": 2,
            "n_t1": 8,
            "calib_wait_short": 0.04,
            "calib_wait_long": 15.0,
            "flux": True,
            "flux_chan": self.cfg.hw.soc.dacs.flux.ch[qi],
            "flux_gain": 0.0,
            "flux_readout_wait": 0.1,
            "freq_span": freq_span,
            "direction": direction,
            "lin_freq": True,
            "t1_max": float("inf"),
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "wait_time": 0.0,  # set dynamically in acquire
            "final_delay": self.cfg.device.qubit.T1[qi] * 6,
        }
        self.cfg.expt = {**params_def, **params}

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

    def acquire(self, progress=True, display=False):
        from pathlib import Path

        final_delay = self._get_final_delay()
        qi = self._qi
        n_e = self.cfg.expt["n_e"]
        n_g = self.cfg.expt["n_g"]
        n_t1 = self.cfg.expt["n_t1"]

        # Initial T1 estimate from config
        current_t1 = self.cfg.device.qubit.T1[qi]
        t1_max = self.cfg.expt.t1_max

        data = {
            "avgi": [], "avgq": [], "amps": [], "phases": [],
            "gain_pts": self.gain_pts,
            "t1_list": [], "population": [], "wait_times": [],
            "ve_list": [], "vg_list": [],
        }
        if self.freq_pts is not None:
            data["freq_pts"] = self.freq_pts

        timestamp_list = []
        elapsed_list = []
        t_acq_start = time.time()
        csv_path = Path(self.fname).with_suffix(".csv")

        for i, g in enumerate(tqdm(self.gain_pts, disable=not progress)):
            self.cfg.expt["flux_gain"] = float(g)

            # Adaptive wait time at T1/2
            wait_time = current_t1 / 2.0
            self.cfg.expt["wait_time"] = float(wait_time)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    prog = T1ContFluxProgram(
                        soccfg=self.soccfg, final_delay=final_delay, cfg=self.cfg
                    )
                    iq_list = prog.acquire(
                        self.im[self.cfg.aliases.soc],
                        rounds=self.cfg.expt.rounds,
                        threshold=None,
                        progress=False,
                    )
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"  Scan failed at gain={g:.4f} (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                        time.sleep(0.5)
                    else:
                        raise

            # iq_list[0] shape: [n_measurements, 2] (averaged over reps/rounds)
            iq_array = np.array(iq_list)

            # Extract I values for each measurement type
            # Short-wait (excited cal): indices [0:n_e]
            ve = float(np.mean(iq_array[0, 0:n_e, 0]))
            # Long-wait (ground cal): indices [n_e:n_e+n_g]
            vg = float(np.mean(iq_array[0, n_e:n_e+n_g, 0]))
            # T1 data: indices [n_e+n_g:n_e+n_g+n_t1]
            vi_t1 = float(np.mean(iq_array[0, n_e+n_g:n_e+n_g+n_t1, 0]))

            data["ve_list"].append(ve)
            data["vg_list"].append(vg)

            # Also store overall avgi/avgq (mean of T1 measurements)
            avgi_mean = vi_t1
            avgq_mean = float(np.mean(iq_array[0, n_e+n_g:n_e+n_g+n_t1, 1]))
            data["avgi"].append(avgi_mean)
            data["avgq"].append(avgq_mean)
            data["amps"].append(float(np.sqrt(avgi_mean**2 + avgq_mean**2)))
            data["phases"].append(float(np.arctan2(avgq_mean, avgi_mean)))

            # Normalize population
            dv = ve - vg
            if abs(dv) > 1e-10:
                pop = (vi_t1 - vg) / dv
            else:
                # Fallback to config means
                g_mean = self.cfg.device.readout.g_mean[qi]
                e_mean = self.cfg.device.readout.e_mean[qi]
                pop = (vi_t1 - g_mean) / (e_mean - g_mean)
            data["population"].append(pop)
            data["wait_times"].append(wait_time)

            # Extract T1: P = exp(-t/T1) => T1 = -t/ln(P)
            if 0.01 < pop < 0.99 and wait_time > 0:
                t1_new = -wait_time / np.log(pop)
                t1_new = min(t1_new, t1_max)
                if t1_new > 0:
                    current_t1 = t1_new
            t1_val = current_t1
            data["t1_list"].append(t1_val)

            # Timestamps
            now = time.time()
            timestamp_list.append(datetime.fromtimestamp(now).isoformat())
            elapsed_list.append(now - t_acq_start)

            # Interim save
            save_data = {k: np.array(v) for k, v in data.items()}
            self.data = save_data
            self.save_data(data=save_data)

            # Incremental CSV
            n = i + 1
            gains_so_far = self.gain_pts[:n]
            t1_arr = np.array(data["t1_list"])
            pop_arr = np.array(data["population"])
            wt_arr = np.array(data["wait_times"])
            ve_arr = np.array(data["ve_list"])
            vg_arr = np.array(data["vg_list"])
            elapsed_arr = np.array(elapsed_list)
            if self.freq_pts is not None:
                num_cols = np.column_stack((
                    gains_so_far, self.freq_pts[:n], wt_arr, pop_arr, t1_arr,
                    ve_arr, vg_arr, elapsed_arr
                ))
                header = "timestamp,gain,freq,wait_time,population,t1,ve,vg,elapsed_s"
            else:
                num_cols = np.column_stack((
                    gains_so_far, wt_arr, pop_arr, t1_arr,
                    ve_arr, vg_arr, elapsed_arr
                ))
                header = "timestamp,gain,wait_time,population,t1,ve,vg,elapsed_s"
            with open(csv_path, "w") as f_csv:
                f_csv.write(header + "\n")
                for row_idx in range(num_cols.shape[0]):
                    ts = timestamp_list[row_idx]
                    nums = ",".join(f"{v}" for v in num_cols[row_idx])
                    f_csv.write(f"{ts},{nums}\n")

        # Finalize
        for k, v in data.items():
            data[k] = np.array(v)
        self.data = data
        return data

    def display(self, data=None, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        gain_pts = data["gain_pts"]
        t1_list = data["t1_list"]
        has_freq = "freq_pts" in data and data["freq_pts"] is not None

        t1_arr = np.array(t1_list, dtype=float)
        t1_mask = np.isfinite(t1_arr)

        # Compute outlier-aware limits
        t1_inlier = t1_mask
        ymin, ymax, margin = None, None, None
        if t1_mask.sum() > 2:
            q1, q3 = np.percentile(t1_arr[t1_mask], [25, 75])
            iqr = q3 - q1
            t1_inlier = t1_mask & (t1_arr >= q1 - 1.5 * iqr) & (t1_arr <= q3 + 1.5 * iqr)
            inlier_vals = t1_arr[t1_inlier]
            if len(inlier_vals) > 0:
                ymin, ymax = inlier_vals.min(), inlier_vals.max()
                margin = (ymax - ymin) * 0.1 if ymax > ymin else abs(ymax) * 0.1 or 0.1

        # T1 vs gain
        fig, ax = plt.subplots()
        if t1_mask.sum() > 2:
            ax.plot(gain_pts[t1_inlier], t1_arr[t1_inlier], ".-")
            ax.plot(gain_pts[~t1_inlier & t1_mask], t1_arr[~t1_inlier & t1_mask], "rx", ms=6)
            if ymin is not None:
                ax.set_ylim(ymin - margin, ymax + margin)
        else:
            ax.plot(gain_pts, t1_list, ".-")
        ax.set_xlabel("Flux Gain")
        ax.set_ylabel("$T_1$ ($\\mu$s)")
        self.save_fig(fig, suffix="_t1_vs_gain")

        # T1 vs frequency
        if has_freq:
            freq_pts = data["freq_pts"]
            fig, ax = plt.subplots()
            if t1_mask.sum() > 2:
                ax.plot(freq_pts[t1_inlier], t1_arr[t1_inlier], ".-")
                ax.plot(freq_pts[~t1_inlier & t1_mask], t1_arr[~t1_inlier & t1_mask], "rx", ms=6)
                if ymin is not None:
                    ax.set_ylim(ymin - margin, ymax + margin)
            else:
                ax.plot(freq_pts, t1_list, ".-")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("$T_1$ ($\\mu$s)")
            self.save_fig(fig, suffix="_t1_vs_freq")

        # Population vs gain
        if "population" in data:
            fig, ax = plt.subplots()
            ax.plot(gain_pts, data["population"], ".-")
            ax.set_xlabel("Flux Gain")
            ax.set_ylabel("Population")
            self.save_fig(fig, suffix="_pop_vs_gain")

        # Calibration ve/vg vs gain
        if "ve_list" in data and "vg_list" in data:
            fig, ax = plt.subplots()
            ax.plot(gain_pts, data["ve_list"], ".-", label="$v_e$ (short wait)")
            ax.plot(gain_pts, data["vg_list"], ".-", label="$v_g$ (long wait)")
            ax.set_xlabel("Flux Gain")
            ax.set_ylabel("Readout I (a.u.)")
            ax.legend()
            self.save_fig(fig, suffix="_calib_vs_gain")


class T1ContFluxRepeated(T1ContFluxExperiment):
    """
    Repeatedly run T1ContFluxExperiment sweeps and build a T1(frequency, time) heatmap.

    Each sweep already includes inline calibration (no separate calibration passes needed).

    Extra parameters (passed via params dict):
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
            prefix = f"t1_cont_flux_repeated_qubit{qi}"

        # Init parent (sets up gain_pts, freq_pts) but don't run
        super().__init__(
            cfg_dict=cfg_dict, qi=qi, go=False, params=params,
            prefix=prefix, progress=progress, display=display,
        )

        if go:
            self.acquire(progress=progress, display=display)
            self.display()

    def acquire(self, progress=False, display=True):
        from pathlib import Path

        try:
            from IPython.display import display as ipy_display, clear_output
            _has_ipy = True
        except ImportError:
            _has_ipy = False

        data = {
            "freq_pts": self.freq_pts,
            "gain_pts": self.gain_pts,
            "timestamps": [],
        }
        static_keys = {"gain_pts", "freq_pts"}

        t_start = time.time()
        sweep_idx = 0

        csv_dir = Path(self.fname).with_suffix("") / "csvs"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Build inner sweep params, excluding repeated-specific keys
        _repeated_keys = ("n_repeats", "duration_hours")
        sweep_params = {k: v for k, v in self.cfg.expt.items() if k not in _repeated_keys}

        inner_progress = progress and not _has_ipy
        self.expt = T1ContFluxExperiment(
            self._cfg_dict, qi=self._qi, go=False, params={**sweep_params},
            progress=inner_progress, display=False,
        )
        self.expt.gain_pts = self.gain_pts
        self.expt.freq_pts = self.freq_pts

        try:
            while True:
                if self._n_repeats is not None and sweep_idx >= self._n_repeats:
                    break
                elapsed_h = (time.time() - t_start) / 3600
                if self._duration_hours is not None and elapsed_h > self._duration_hours:
                    break

                ts_label = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.expt.fname = str(csv_dir / f"sweep_{sweep_idx:04d}_{ts_label}")
                if _has_ipy:
                    print(f"Sweep {sweep_idx + 1}" +
                          (f"/{self._n_repeats}" if self._n_repeats else "") +
                          f" — {len(self.gain_pts)} gain points ...")

                max_retries = 3
                data_new = None
                for attempt in range(max_retries):
                    try:
                        data_new = self.expt.acquire(progress=inner_progress, display=False)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"  Sweep {sweep_idx + 1} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                            time.sleep(1.0)
                            self.expt = T1ContFluxExperiment(
                                self._cfg_dict, qi=self._qi, go=False, params={**sweep_params},
                                progress=inner_progress, display=False,
                            )
                            self.expt.gain_pts = self.gain_pts
                            self.expt.freq_pts = self.freq_pts
                            self.expt.fname = str(csv_dir / f"sweep_{sweep_idx:04d}_{ts_label}")
                        else:
                            print(f"  Sweep {sweep_idx + 1} failed after {max_retries} attempts: {e}. Skipping.")
                            break

                if data_new is None:
                    sweep_idx += 1
                    continue

                for key in data_new:
                    if key in static_keys:
                        continue
                    if key not in data:
                        data[key] = []
                    data[key].append(np.array(data_new[key]).copy())

                elapsed_h = (time.time() - t_start) / 3600
                data["timestamps"].append(elapsed_h)

                save_data = {k: np.array(v) if isinstance(v, list) else v for k, v in data.items()}
                self.save_data(data=save_data)

                if _has_ipy:
                    self._update_heatmap(data)

                sweep_idx += 1

        except KeyboardInterrupt:
            print(f"\nInterrupted after {sweep_idx} sweep(s).")

        for k, v in data.items():
            if isinstance(v, list):
                data[k] = np.array(v) if v else np.array([]).reshape(0, 0)
        self.data = data
        return self.data

    def _update_heatmap(self, data):
        import sys
        import io
        import matplotlib.pyplot as plt
        from IPython.display import display as ipy_display, clear_output, Image

        try:
            t1_2d = np.array(data["t1_list"])
            freq = self.freq_pts if self.freq_pts is not None else self.gain_pts
            times = np.array(data["timestamps"])
            has_freq = self.freq_pts is not None
            x_label = "Frequency (MHz)" if has_freq else "Flux Gain"

            if len(times) == 1:
                times = np.array([times[0], times[0] + 0.01])
                t1_2d = np.vstack([t1_2d, t1_2d])

            # T1 heatmap + calibration panel
            ve_2d = np.array(data["ve_list"])
            vg_2d = np.array(data["vg_list"])
            n_plots = 2 if ve_2d.size > 0 else 1
            fig, axes = plt.subplots(1, n_plots, figsize=(5 + 5 * n_plots, 5), dpi=100)
            if n_plots == 1:
                axes = [axes]

            ax = axes[0]
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
            ax.set_xlabel(x_label)
            ax.set_ylabel("Elapsed Time (hours)")
            plt.colorbar(mesh, ax=ax, label="$T_1$ ($\\mu$s)")

            if n_plots > 1:
                ax = axes[1]
                for i in range(len(ve_2d)):
                    kw = dict(lw=0.5, alpha=0.6)
                    label_e = "$v_e$ (short wait)" if i == 0 else None
                    label_g = "$v_g$ (long wait)" if i == 0 else None
                    ax.plot(freq, ve_2d[i], "-", color="C0", label=label_e, **kw)
                    ax.plot(freq, vg_2d[i], "-", color="C1", label=label_g, **kw)
                ax.set_xlabel(x_label)
                ax.set_ylabel("Readout I (a.u.)")
                ax.legend(fontsize=8)

            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)

            sys.stdout.flush()
            sys.stderr.flush()
            clear_output(wait=True)
            ipy_display(Image(buf.read()))
            n_sweeps = len(data["t1_list"])
            print(f"Completed {n_sweeps} sweep(s), {times[-1]:.2f} hours elapsed")
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
        has_freq = "freq_pts" in data and data["freq_pts"] is not None
        x_label = "Frequency (MHz)" if has_freq else "Flux Gain"

        # T1 heatmap
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
        ax.set_xlabel(x_label)
        ax.set_ylabel("Elapsed Time (hours)")
        plt.colorbar(mesh, ax=ax, label="$T_1$ ($\\mu$s)")
        self.save_fig(fig, suffix="_heatmap")

        # Calibration drift: ve/vg heatmaps
        ve_2d = data.get("ve_list", np.array([]))
        vg_2d = data.get("vg_list", np.array([]))
        if ve_2d.ndim == 2 and ve_2d.shape[0] > 0:
            x = data["freq_pts"] if has_freq else data["gain_pts"]

            fig, ax = plt.subplots()
            ax.plot(x, ve_2d[-1], ".-", label="$v_e$ (short wait)")
            ax.plot(x, vg_2d[-1], ".-", label="$v_g$ (long wait)")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Readout I (a.u.)")
            ax.legend()
            self.save_fig(fig, suffix="_calib_latest")

            if ve_2d.shape[0] > 1:
                for arr, label, suffix in [
                    (ve_2d, "$v_e$", "_calib_ve"),
                    (vg_2d, "$v_g$", "_calib_vg"),
                ]:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mesh = ax.pcolormesh(x, times, arr, cmap="viridis", shading="auto")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel("Elapsed Time (hours)")
                    plt.colorbar(mesh, ax=ax, label=f"{label} (a.u.)")
                    self.save_fig(fig, suffix=suffix)
