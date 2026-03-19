import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

from ...analysis import fitting as fitter
from ..general.qick_experiment import QickExperimentLoop, DataProcessor
from .t1 import T1Program
from .single_shot import HistogramExperiment


class T1FastFluxLoop(QickExperimentLoop):
    """
    Fast T1 vs flux measurement using single-point population readout.

    Instead of running a full T1 decay curve at each flux point (like T1FastFlux),
    this measures population at a single wait time = T1/2 per gain point. T1 is
    extracted from: T1 = -wait_time / ln(population).

    The wait time adapts as the sweep progresses: after each measurement, the T1
    estimate is updated and the next wait_time is set to new_T1/2.

    Inherits from QickExperimentLoop. Sweeps flux_gain on the x-axis.

    Sweep parameters (passed via params dict):
        gain_start: Starting flux gain value (default: sweet_spot_ac)
        gain_stop: Ending flux gain value
        direction: 'pos' or 'neg' sweep direction from sweet spot
        freq_span: Frequency span in MHz (overrides gain_stop when flux model available)
        expts_gain: Number of gain points in the sweep (default: 50)
        lin_freq: If True, space points linearly in frequency
        flux_converter: Optional FluxConverter instance
        t1_max: Upper bound on T1 for adaptive tracking
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
            prefix = f"t1_fastflux_loop_qubit{qi}"

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
            "expts": 1,
            "reps": int(1.5 * self.reps),
            "rounds": self.rounds,
            "start": 0.05,
            "flux": True,
            "flux_chan": self.cfg.hw.soc.dacs.flux.ch[qi],
            "flux_gain": 0.0,
            "flux_readout_wait": 0.1,
            "freq_span": freq_span,
            "direction": direction,
            "lin_freq": True,
            "t1_max": float("inf"),
            "acStark": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "wait_time": 0.0,  # set dynamically in acquire
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

    @classmethod
    def from_h5file(cls, fname):
        """Load from h5, restoring _qi from the saved config."""
        self = super().from_h5file(fname)
        # Recover qubit index from config; fall back to parsing the filename
        try:
            self._qi = self.cfg.expt.qubit[0]
        except Exception:
            import re
            m = re.search(r'qubit(\d+)', str(fname))
            self._qi = int(m.group(1)) if m else 0
        self._cfg_dict = None
        self._flux_converter = None
        return self

    def acquire(self, progress=True, display=False, ve=None, vg=None):
        from pathlib import Path

        final_delay = self._get_final_delay()
        qi = self._qi

        # Initial T1 estimate from config
        current_t1 = self.cfg.device.qubit.T1[qi]
        t1_max = self.cfg.expt.t1_max

        # g/e scaling — use passed-in calibration values or fall back to config
        # ve/vg can be scalars or per-gain-point arrays
        g_mean = self.cfg.device.readout.g_mean[qi]
        e_mean = self.cfg.device.readout.e_mean[qi]
        dv = e_mean - g_mean
        if ve is None:
            ve = np.full(len(self.gain_pts), e_mean)
        elif np.ndim(ve) == 0:
            ve = np.full(len(self.gain_pts), float(ve))
        if vg is None:
            vg = np.full(len(self.gain_pts), g_mean)
        elif np.ndim(vg) == 0:
            vg = np.full(len(self.gain_pts), float(vg))

        data = {
            "avgi": [], "avgq": [], "amps": [], "phases": [],
            "gain_pts": self.gain_pts,
            "t1_list": [], "population": [], "wait_times": [],
        }
        if self.freq_pts is not None:
            data["freq_pts"] = self.freq_pts

        timestamp_list = []
        elapsed_list = []
        t_acq_start = time.time()
        csv_path = Path(self.fname).with_suffix(".csv")

        for i, g in enumerate(tqdm(self.gain_pts, disable=not progress)):
            self.cfg.expt["flux_gain"] = float(g)

            # Main measurement at T1/2
            wait_time = current_t1 / 2.0
            self.cfg.expt["wait_time"] = float(wait_time)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    prog = T1Program(soccfg=self.soccfg, final_delay=final_delay, cfg=self.cfg)
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

            # Extract I/Q (mean over expts loop points)
            processed = DataProcessor.process_iq_data(iq_list)
            avgi_mean = float(np.mean(processed["avgi"]))
            avgq_mean = float(np.mean(processed["avgq"]))
            data["avgi"].append(avgi_mean)
            data["avgq"].append(avgq_mean)
            data["amps"].append(float(np.mean(processed["amps"])))
            data["phases"].append(float(np.mean(processed["phases"])))

            # Normalize population using per-point ve/vg calibration
            raw_i = avgi_mean
            dv_local = ve[i] - vg[i]
            if abs(dv_local) > 1e-10:
                pop = (raw_i - vg[i]) / dv_local
            else:
                pop = float((raw_i - g_mean) / dv)
            data["population"].append(pop)
            data["wait_times"].append(wait_time)

            # Extract T1 from population: P = exp(-t/T1) => T1 = -t/ln(P)
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

            # if display:
            #     print(f"  gain={g:.4f}  wait={wait_time:.2f}us  pop={pop:.3f}  T1={t1_val:.2f}us")

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
            elapsed_arr = np.array(elapsed_list)
            if self.freq_pts is not None:
                num_cols = np.column_stack((gains_so_far, self.freq_pts[:n], wt_arr, pop_arr, t1_arr, elapsed_arr))
                header = "timestamp,gain,freq,wait_time,population,t1,elapsed_s"
            else:
                num_cols = np.column_stack((gains_so_far, wt_arr, pop_arr, t1_arr, elapsed_arr))
                header = "timestamp,gain,wait_time,population,t1,elapsed_s"
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

        # T1 vs gain
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
        ax.set_title("T1 Fast Flux Loop")
        self.save_fig(fig, suffix="_t1_vs_gain")

        # Population vs gain
        if "population" in data:
            fig, ax = plt.subplots()
            ax.plot(gain_pts, data["population"], ".-")
            ax.set_xlabel("Flux Gain")
            ax.set_ylabel("Population")
            ax.set_title("Excited State Population at $T_1/2$")
            self.save_fig(fig, suffix="_pop_vs_gain")

        # T1 vs frequency
        if "freq_pts" in data:
            freq_pts = data["freq_pts"]
            fig, ax = plt.subplots()
            if t1_mask.sum() > 2:
                ax.plot(freq_pts[t1_inlier], t1_arr[t1_inlier], ".-")
                ax.plot(freq_pts[~t1_inlier & t1_mask], t1_arr[~t1_inlier & t1_mask], "rx", ms=6)
                if len(inlier_vals) > 0:
                    ax.set_ylim(ymin - margin, ymax + margin)
            else:
                ax.plot(freq_pts, t1_list, ".-")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("$T_1$ ($\\mu$s)")
            ax.set_title("T1 Fast Flux Loop")
            self.save_fig(fig, suffix="_t1_vs_freq")


class T1FastFluxLoopRepeated(T1FastFluxLoop):
    """
    Repeatedly run T1FastFluxLoop sweeps and build a T1(frequency, time) heatmap.

    Runs T1FastFluxLoop N times (or for a specified duration), accumulating
    T1 vs frequency data. After each sweep, updates a live 2D heatmap
    (x=frequency, y=elapsed time, color=T1) and saves to a master HDF5.

    Extra parameters (passed via params dict, in addition to T1FastFluxLoop params):
        n_repeats: Number of sweeps to run (default: None = run forever)
        duration_hours: Max duration in hours (default: None = no limit).
            If both n_repeats and duration_hours are given, stops at whichever
            comes first. If both None, runs until KeyboardInterrupt.
        calibration: If True (default), run periodic g/e calibration sweeps.
            If "single_shot", run a HistogramExperiment at the start of each
            sweep and use mean(Ig)/mean(Ie) as vg/ve for the entire sweep.
            If False, use static mean_g/mean_e from config for all sweeps.
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
        # Pop repeated-specific params before passing to parent
        self._n_repeats = params.pop("n_repeats", None)
        self._duration_hours = params.pop("duration_hours", None)
        self._calibration = params.pop("calibration", True)

        if prefix is None:
            prefix = f"t1_fastflux_loop_repeated_qubit{qi}"

        # Init parent (sets up gain_pts, freq_pts) but don't run
        super().__init__(
            cfg_dict=cfg_dict, qi=qi, go=False, params=params,
            prefix=prefix, progress=progress, display=display,
        )

        if go:
            self.acquire(progress=progress, display=display)
            self.display()

    def _run_calibration(self, progress=False):
        """Run short/long wait calibration sweeps across all gain points to get per-point ve/vg."""
        final_delay = self._get_final_delay()

        calib_wait_short = 0.04   # 40 ns — short wait ≈ excited state
        calib_wait_long = 15.0    # 15 us — long wait ≈ ground state

        ve_arr = np.empty(len(self.gain_pts))
        vg_arr = np.empty(len(self.gain_pts))

        for i, g in enumerate(tqdm(self.gain_pts, disable=not progress, desc="Calibration")):
            self.cfg.expt["flux_gain"] = float(g)
            for calib_wait, arr in [
                (calib_wait_short, ve_arr),
                (calib_wait_long, vg_arr),
            ]:
                self.cfg.expt["wait_time"] = float(calib_wait)
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        prog = T1Program(soccfg=self.soccfg, final_delay=final_delay, cfg=self.cfg)
                        iq_list = prog.acquire(
                            self.im[self.cfg.aliases.soc],
                            rounds=self.cfg.expt.rounds,
                            threshold=None,
                            progress=False,
                        )
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"  Calibration scan failed at gain={g:.4f} (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                            time.sleep(0.5)
                        else:
                            raise
                calib_proc = DataProcessor.process_iq_data(iq_list)
                arr[i] = float(np.mean(calib_proc["avgi"]))

        return ve_arr, vg_arr

    def _run_single_shot_calibration(self, progress=False):
        """Run a single-shot (histogram) experiment and return scalar vg, ve from mean(Ig), mean(Ie)."""
        hist = HistogramExperiment(
            cfg_dict=self._cfg_dict, qi=self._qi, go=False,
            params=dict(calibrate=True, shots=25000),
        )
        data = hist.acquire(progress=progress)
        vg = float(np.mean(data["Ig"]))
        ve = float(np.mean(data["Ie"]))
        print(f"  Single-shot calibration: vg={vg:.4f}, ve={ve:.4f}")
        return vg, ve

    def acquire(self, progress=False, display=True):
        from pathlib import Path

        try:
            from IPython.display import display as ipy_display, clear_output
            _has_ipy = True
        except ImportError:
            _has_ipy = False

        # Initialize data dict (like QickExperiment2DSimple)
        data = {
            "freq_pts": self.freq_pts,
            "gain_pts": self.gain_pts,
            "timestamps": [],
            "calib_ve": [],
            "calib_vg": [],
        }
        static_keys = {"gain_pts", "freq_pts"}

        t_start = time.time()
        sweep_idx = 0

        # Create a dedicated folder for this run's CSVs
        csv_dir = Path(self.fname).with_suffix("") / "csvs"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Build inner T1FastFluxLoop params from cfg.expt, excluding repeated-specific keys
        _repeated_keys = ("n_repeats", "duration_hours", "calibration")
        sweep_params = {k: v for k, v in self.cfg.expt.items() if k not in _repeated_keys}

        # Calibration interval: run g/e calibration scans every N sweeps
        use_single_shot_calib = self._calibration == "single_shot"
        calib_interval = 10 if (self._calibration and not use_single_shot_calib) else 50

        # Initialize ve/vg arrays from config
        qi = self._qi
        n_pts = len(self.gain_pts)
        ve = np.full(n_pts, self.cfg.device.readout.e_mean[qi])
        vg = np.full(n_pts, self.cfg.device.readout.g_mean[qi])

        # Create inner experiment (like QickExperiment2DSimple.expt)
        inner_progress = progress and not _has_ipy
        self.expt = T1FastFluxLoop(
            self._cfg_dict, qi=self._qi, go=False, params={**sweep_params},
            progress=inner_progress, display=False,
        )
        self.expt.gain_pts = self.gain_pts
        self.expt.freq_pts = self.freq_pts

        try:
            while True:
                # Check stopping conditions
                if self._n_repeats is not None and sweep_idx >= self._n_repeats:
                    break
                elapsed_h = (time.time() - t_start) / 3600
                if self._duration_hours is not None and elapsed_h > self._duration_hours:
                    break

                # Run calibration
                if use_single_shot_calib:
                    # Single-shot calibration: run at the start of every sweep
                    ss_vg, ss_ve = self._run_single_shot_calibration(progress=inner_progress)
                    ve = np.full(n_pts, ss_ve)
                    vg = np.full(n_pts, ss_vg)
                    data["calib_ve"].append(ss_ve)
                    data["calib_vg"].append(ss_vg)
                elif sweep_idx % calib_interval == 0:
                    cal_ve, cal_vg = self._run_calibration(progress=inner_progress)
                    data["calib_ve"].append(cal_ve.copy())
                    data["calib_vg"].append(cal_vg.copy())
                    if self._calibration:
                        ve, vg = cal_ve, cal_vg

                # Redirect CSV into the dedicated folder
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
                        data_new = self.expt.acquire(progress=inner_progress, display=False,
                                             ve=ve, vg=vg)
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"  Sweep {sweep_idx + 1} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying...")
                            time.sleep(1.0)
                            # Re-create inner experiment for clean state
                            self.expt = T1FastFluxLoop(
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

                # Generic data accumulation (like QickExperiment2DSimple)
                for key in data_new:
                    if key in static_keys:
                        continue
                    if key not in data:
                        data[key] = []
                    data[key].append(np.array(data_new[key]).copy())

                elapsed_h = (time.time() - t_start) / 3600
                data["timestamps"].append(elapsed_h)

                # Interim save
                save_data = {k: np.array(v) if isinstance(v, list) else v for k, v in data.items()}
                self.save_data(data=save_data)

                # Update live heatmap
                if _has_ipy:
                    self._update_heatmap(data)

                sweep_idx += 1

        except KeyboardInterrupt:
            print(f"\nInterrupted after {sweep_idx} sweep(s).")

        # Finalize: convert lists to arrays
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

            # pcolormesh needs at least 2 y-values to draw cells;
            # duplicate the single row so the heatmap renders after the first sweep
            if len(times) == 1:
                times = np.array([times[0], times[0] + 0.01])
                t1_2d = np.vstack([t1_2d, t1_2d])

            calib_ve = data["calib_ve"]
            calib_vg = data["calib_vg"]
            n_calib = len(calib_ve)
            n_plots = 1 + (1 if n_calib > 0 else 0)
            fig, axes = plt.subplots(1, n_plots, figsize=(5 + 5 * n_plots, 5), dpi=100)
            if n_plots == 1:
                axes = [axes]

            # T1 heatmap
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
            ax.set_title(f"$T_1$ — Q{self._qi}")
            plt.colorbar(mesh, ax=ax, label="$T_1$ ($\\mu$s)")

            # Calibration panel
            if n_calib > 0:
                ax = axes[1]
                scalar_calib = np.ndim(calib_ve[0]) == 0
                if scalar_calib:
                    # Single-shot calibration: plot scalar ve/vg vs time
                    calib_times = times[:n_calib] if n_calib <= len(times) else np.arange(n_calib)
                    ax.plot(calib_times, calib_ve, ".-", color="C0", label="$v_e$")
                    ax.plot(calib_times, calib_vg, ".-", color="C1", label="$v_g$")
                    ax.set_xlabel("Elapsed Time (hours)")
                    ax.set_ylabel("Readout I (a.u.)")
                else:
                    # Per-point calibration: overlay ve/vg vs frequency
                    for i in range(n_calib):
                        kw = dict(lw=0.5, alpha=0.6)
                        label_e = "$v_e$ (short wait)" if i == 0 else None
                        label_g = "$v_g$ (long wait)" if i == 0 else None
                        ax.plot(freq, calib_ve[i], "-", color="C0", label=label_e, **kw)
                        ax.plot(freq, calib_vg[i], "-", color="C1", label=label_g, **kw)
                    ax.set_xlabel(x_label)
                    ax.set_ylabel("Readout I (a.u.)")
                ax.legend(fontsize=8)

            fig.tight_layout()

            # Render to PNG and close figure immediately
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
        ax.set_title(f"$T_1$ vs Frequency and Time (Loop) — Q{self._qi}")
        plt.colorbar(mesh, ax=ax, label="$T_1$ ($\\mu$s)")
        self.save_fig(fig, suffix="_heatmap")

        # Calibration plots
        calib_ve = data.get("calib_ve", np.array([]))
        calib_vg = data.get("calib_vg", np.array([]))
        if calib_ve.ndim == 1 and calib_ve.shape[0] > 0 and np.ndim(calib_ve[0]) == 0:
            # Single-shot calibration: scalar ve/vg per sweep — plot vs time
            fig, ax = plt.subplots()
            ax.plot(times, calib_ve, ".-", label="$v_e$")
            ax.plot(times, calib_vg, ".-", label="$v_g$")
            ax.set_xlabel("Elapsed Time (hours)")
            ax.set_ylabel("Readout I (a.u.)")
            ax.legend()
            self.save_fig(fig, suffix="_calib_vs_time")
        elif calib_ve.ndim == 2 and calib_ve.shape[0] > 0:
            # Per-point calibration: ve/vg arrays per sweep
            has_freq = "freq_pts" in data and data["freq_pts"] is not None
            x = data["freq_pts"] if has_freq else data["gain_pts"]
            x_label = "Frequency (MHz)" if has_freq else "Flux Gain"

            # Latest ve/vg vs frequency/gain
            fig, ax = plt.subplots()
            ax.plot(x, calib_ve[-1], ".-", label="$v_e$ (short wait)")
            ax.plot(x, calib_vg[-1], ".-", label="$v_g$ (long wait)")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Readout I (a.u.)")
            ax.legend()
            self.save_fig(fig, suffix="_calib_latest")

            # ve/vg heatmaps over time (if multiple calibrations)
            if calib_ve.shape[0] > 1:
                n_calib = calib_ve.shape[0]
                calib_indices = np.arange(0, len(times), 10)[:n_calib]
                if len(calib_indices) < n_calib:
                    calib_times = np.linspace(times[0], times[-1], n_calib)
                else:
                    calib_times = times[calib_indices]
                for arr, label, suffix in [
                    (calib_ve, "$v_e$", "_calib_ve"),
                    (calib_vg, "$v_g$", "_calib_vg"),
                ]:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    mesh = ax.pcolormesh(x, calib_times, arr, cmap="viridis", shading="auto")
                    ax.set_xlabel(x_label)
                    ax.set_ylabel("Elapsed Time (hours)")
                    ax.set_title(f"{label} Calibration Drift — Q{self._qi}")
                    plt.colorbar(mesh, ax=ax, label=f"{label} (a.u.)")
                    self.save_fig(fig, suffix=suffix)
