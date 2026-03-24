"""
Flux-dependent Pulse Probe Spectroscopy Experiments

This module contains spectroscopy experiments that sweep qubit frequency
as a function of flux (DC bias or fast flux gain).

Classes:
- QubitSpecFlux: 2D spectroscopy: frequency × DC bias
- QubitSpecFastFlux: Spectroscopy vs fast flux gain
- QubitSpecFluxSweep: Spectroscopy vs arbitrary parameter
"""

import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

from ...helpers import config
from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment2DSimple

from ...analysis import fitting as fitter
from ..single_qubit.pulse_probe_spectroscopy import QubitSpec, XLABEL, YLABEL, DCYLABEL, FITTER_FUNC, FIT_FUNC
from ..single_qubit.resonator_spectroscopy import ResSpec
from .qubit_spec_predistorted import QubitSpecPredist


class QubitSpecFlux(QickExperiment2DSimple):
    """
    2D pulse probe spectroscopy: frequency (X) × DC bias (Y).

    Sweeps a fixed frequency span for each DC bias setpoint using the SoC bias port.
    Hardware DC is set per-row via soc.rfb_set_bias(bias_chan, volts).

    Parameters (in params dict):
      # Spectroscopy (inner 1D, same as QubitSpec)
      - 'span' (MHz), 'expts' (# points), 'reps', 'gain', 'pulse_type', 'checkEF', etc.

      # Flux/DC sweep (outer 2D)
      - 'dc_start' (V): start of DC sweep
      - 'dc_stop'  (V): end of DC sweep
      - 'expts_dc' (# rows)
      - 'bias_chan' (int): SoC DC channel to use
      - 'dc_settle' (s): sleep after setting bias (optional)
      - 'dc_verify_print' (bool): print readback line per row (optional)
     optional table to update readout per bias
      - 'bias_table_V': 1D list/array of DC bias values (V)
      - 'f0_table_MHz': 1D list/array of fitted resonator f0 (MHz), same length as bias_table_V
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=True,
        qi=0,
        go=True,
        params={},
        style="",
        display=True,
        min_r2=None,
        max_err=None,
        live_plot=False,
    ):
        # Prefix like other classes
        ef = "ef_" if params.get("checkEF", False) else ""
        prefix = f"qubit_spectroscopy_flux_{ef}{style}_qubit{qi}" if not prefix else prefix
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, live_plot=live_plot)
        self.cfg_dict = cfg_dict

        # ---- style presets for the X (frequency) axis ----
        if style == "coarse":
            x_defaults = {"span": 800, "expts": 500}
        elif style == "fine":
            x_defaults = {"span": 40, "expts": 100}
        else:
            x_defaults = {"span": 120, "expts": 200}

        # ---- defaults for inner 1D and outer DC sweep ----
        inner_defaults = {
            "reps": 2 * self.reps,
            "gain": self.cfg.device.qubit.low_gain * self.cfg.device.qubit.spec_gain[qi],
            "pulse_type": "const",
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "checkEF": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "sep_readout": True,
            "active_reset": False,
        }

        # Outer DC sweep defaults
        dc_defaults = {
            "dc_start": 0.0,
            "dc_stop":  0.50,
            "expts_dc":  41,
            "bias_chan": 0,
            "dc_settle": 0.0,
            "dc_verify_print": False,
            "recenter_res": False,        # run ResSpec at each bias to recenter readout
            "recenter_res_span": 5,       # MHz span for the ResSpec sweep
            "recenter_res_expts": 100,    # number of points for ResSpec
        }

        # Merge defaults
        params_def = {**x_defaults, **inner_defaults, **dc_defaults}
        params = {**params_def, **params}

        # Determine start frequency around f_ge or f_ef
        if params.get("checkEF", False):
            params["start"] = self.cfg.device.qubit.f_ef[qi] - params["span"] / 2
        else:
            params["start"] = self.cfg.device.qubit.f_ge[qi] - params["span"] / 2

        # Accept an optional bias→f0 table and store sorted copies
        bias_tbl = params.get("bias_table_V", None)
        f0_tbl   = params.get("f0_table_MHz", None)
        if bias_tbl is not None and f0_tbl is not None:
            bt = np.asarray(bias_tbl, dtype=float)
            ft = np.asarray(f0_tbl, dtype=float)
            if bt.shape != ft.shape:
                raise ValueError("bias_table_V and f0_table_MHz must have the same shape.")
            order = np.argsort(bt)
            self.bias_table_V = bt[order]
            self.f0_table_MHz = ft[order]
            # store also in cfg for provenance/saving
            params["bias_table_V"]  = self.bias_table_V.tolist()
            params["f0_table_MHz"]  = self.f0_table_MHz.tolist()

        # Build inner 1D experiment but don't run yet
        self.expt = QubitSpec(cfg_dict, qi=qi, go=False, params=params, check_params=False)

        # Copy params back to outer cfg so both sides agree
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        # Axis labels for plots
        self.ylabel = XLABEL
        self.xlabel = DCYLABEL

        if go:
            self.run(progress=progress, display=display, analyze=False, min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=True):
        """
        Per-row:
          - set dc → soc.rfb_set_bias(bias_chan, dc)
          - optional settle + readback print
          - run inner 1D spectroscopy and stack results
        """
        # Build DC sweep points
        dcpts = np.linspace(self.cfg.expt.dc_start, self.cfg.expt.dc_stop, int(self.cfg.expt.expts_dc))

        # Prepare containers (same keys as QickExperiment2DSimple.acquire)
        data = {}
        yvals = np.arange(len(dcpts))
        data["time"] = []

         # Precompute if we have a bias→f0 table & container for used f0
        have_table = hasattr(self, "bias_table_V") and hasattr(self, "f0_table_MHz")
        if not have_table:
            # also accept from cfg if class attributes not present (e.g. reload)
            if "bias_table_V" in self.cfg.expt and "f0_table_MHz" in self.cfg.expt:
                bt = np.asarray(self.cfg.expt["bias_table_V"], dtype=float)
                ft = np.asarray(self.cfg.expt["f0_table_MHz"], dtype=float)
                if bt.shape == ft.shape:
                    order = np.argsort(bt)
                    self.bias_table_V  = bt[order]
                    self.f0_table_MHz  = ft[order]
                    have_table = True
        f0_used_each_row = []  # record per-row readout used

        # For each DC bias row
        for i in tqdm(yvals, disable=not progress):
            dc_val = float(dcpts[i])

            # 1) write into the inner experiment config
            self.expt.cfg.expt["dc"] = dc_val

            # 2) set DC on hardware
            soc = self.im[self.cfg.aliases.soc]
            chan = int(self.cfg.expt["bias_chan"])
            soc.rfb_set_bias(chan, dc_val)

            # optional settle and readback
            if self.cfg.expt.dc_settle > 0:
                time.sleep(self.cfg.expt.dc_settle)
            readback = soc.rfb_get_bias(chan)
            if self.cfg.expt.dc_verify_print:
                print(f"[DC Bias] ch {chan}: set {dc_val:.6f} V, read {readback:.6f} V")

            # Recenter readout by running a quick ResSpec at this bias
            q = self.expt.cfg.expt["qubit"][0]
            if self.cfg.expt.recenter_res:
                res_params = {
                    "center": self.cfg.device.readout.frequency[q],
                    "span": self.cfg.expt.recenter_res_span,
                    "expts": self.cfg.expt.recenter_res_expts,
                    "qubit": [q],
                    "qubit_chan": self.cfg.expt["qubit_chan"],
                }
                res = ResSpec(self.cfg_dict, qi=q, go=False, params=res_params, check_params=False)
                res.acquire(progress=False)
                res.analyze(fit=True, hanger=True)
                if res.data.get("freq_fit") is not None:
                    f0_row = float(res.data["freq_fit"][0])
                    self.expt.cfg.device.readout.frequency[q] = f0_row
                    self.cfg.device.readout.frequency[q]      = f0_row
                    f0_used_each_row.append(f0_row)
            # If we have a bias→f0 table, interpolate and update readout frequency
            elif have_table:
                f0_row = float(np.interp(dc_val, self.bias_table_V, self.f0_table_MHz))
                # update both the inner expt cfg and outer cfg so QickProgram picks it up
                self.expt.cfg.device.readout.frequency[q] = f0_row
                self.cfg.device.readout.frequency[q]      = f0_row
                f0_used_each_row.append(f0_row)

            # 3) run the inner 1D experiment and append
            try:
                data_new = self.expt.acquire(progress=False)
            except RuntimeError as e:
                # Readout loop error — retry once after tproc reset
                print(f"[QubitSpecFlux] RuntimeError at bias {dc_val:.4f} V: {e}")
                try:
                    soc.tproc.reset()
                    print("[QubitSpecFlux] tproc reset, retrying...")
                    time.sleep(0.5)
                    data_new = self.expt.acquire(progress=False)
                except RuntimeError:
                    print(f"[QubitSpecFlux] Retry failed. Saving {len(data.get('time',[]))} rows and stopping.")
                    break
            for key in data_new:
                if i == 0:
                    data[key] = []
                data[key].append(data_new[key])
            data["time"].append(time.time())

            # Optional live update/auto-save from base class
            if getattr(self, "save_interim", False):
                super().save_data(data=data)

        freq = data["xpts"][0] if np.ndim(data["xpts"]) > 1 else data["xpts"]
        n_completed = len(data.get("time", []))
        bias = dcpts[:n_completed]

        # overwrite axes
        data["xpts"] = bias
        data["ypts"] = freq
        data["dc_pts"] = bias

        # transpose 2D data so shape matches (len(freq), len(bias))
        for key in ["avgi", "avgq", "amps", "phases"]:
            if key in data:
                data[key] = np.array(data[key]).T  # was (Nbias, Nfreq), make (Nfreq, Nbias)

        # convert lists → numpy arrays
        for k, a in data.items():
            if not isinstance(a, np.ndarray):
                data[k] = np.array(a)

        # Store the readout frequency actually used at each bias (if available)
        if len(f0_used_each_row) > 0:
            data["f0_used_MHz"] = np.asarray(f0_used_each_row, dtype=float)

        if "amps" in data:
            A = np.array(data["amps"])
            data["amps_raw"] = A.copy()
            # subtract per-column minimum (each bias)
            A_min = np.min(A, axis=0, keepdims=True)
            A_sub = A - A_min
            data["amps"] = A_sub
        else:
            A_sub = None

        self.data = data
        return data

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):
        title = f"Spectroscopy vs DC Bias Q{self.cfg.expt.qubit[0]}"
        if self.cfg.expt.checkEF:
            title = "EF " + title
        super().display(
            data=data,
            ax=ax,
            plot_both=False,
            plot_amps=plot_amps,
            title=title,
            xlabel=DCYLABEL,
            ylabel=XLABEL,
            **kwargs,
        )


class QubitSpecFastFlux(QickExperiment2DSimple):
    """
    QubitSpecFastFlux: Sweep qubit spectroscopy across fast flux gain values.

    Runs a QubitSpec at each flux gain point to measure qubit frequency vs flux gain.
    Fits a quadratic to the resulting freq(gain) curve. Saves a single HDF5 file
    with interim saves after each gain point, plus an incremental CSV.

    Sweep parameters (passed via params dict):
        gain_start: Starting flux gain value (default: sweet_spot_ac)
        gain_stop: Ending flux gain value (default: 0.4)
        expts_gain: Number of gain points in the sweep
        update: Whether to update config with f_ge and kappa after each point
        lin_freq: If True and a flux model is available, space points linearly in
            frequency instead of linearly in gain.
        flux_converter: Optional FluxConverter instance. If not provided, loads from config
            (prefers transmon model, falls back to quadratic).
        freq_span: Frequency span in MHz. When provided with a flux model, converts
            to a gain range. Overrides gain_stop.
        direction: 'pos' or 'neg' — sweep direction from sweet spot (default: 'pos').
            Used for picking the correct quadratic root and setting gain_stop from freq_span.
        fit_model: 'quadratic' (default) or 'transmon'. When 'transmon', additionally
            fits the transmon SQUID model f(g) = (f_max+E_C)*(cos^2(phi)+d^2*sin^2(phi))^0.25 - E_C
            with E_C fixed from config. The quadratic fit is always performed.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        style="fine",
        display=True,
        min_r2=None,
        max_err=None,
    ):
        prefix = prefix or f"qspec_fastflux_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)
        self._cfg_dict = cfg_dict
        self._qi = qi
        self._style = style

        # Load flux model from config (prefers transmon, falls back to quadratic)
        cfg_flux = getattr(getattr(getattr(getattr(self.cfg, "hw", None), "soc", None), "dacs", None), "flux", None)
        cfg_qubit = self.cfg.device.qubit

        sweet_spot = 0.0
        if cfg_qubit is not None and hasattr(cfg_qubit, "sweet_spot_ac"):
            sweet_spot = cfg_qubit.sweet_spot_ac[qi]

        # Determine sweep direction and gain range
        direction = params.get("direction", "pos")
        sign = 1 if direction == "pos" else -1

        converter = params.get("flux_converter", None)
        if converter is None:
            converter = fitter.flux_converter_from_config(cfg_flux, cfg_qubit, qi, direction)
        freq_span = params.get("freq_span", None)

        # Convert freq_span to gain_stop using the flux model
        if freq_span is not None and converter is not None and "gain_stop" not in params:
            f_sweet = converter.gain_to_freq(sweet_spot)
            f_target = f_sweet - freq_span  # frequency decreases away from sweet spot
            gain_stop = float(converter.freq_to_gain(f_target))
        else:
            gain_stop = None  # will use params or default

        # Default parameters for the fast flux sweep
        flux_defaults = {
            "gain_start": sweet_spot if converter is not None else -0.4,
            "gain_stop": gain_stop if gain_stop is not None else 0.4,
            "expts_gain": 50,
            "update": True,
            "start_center_freq": None,  # qubit freq at gain_start; None = use config f_ge
            "lin_freq": False,
            "freq_span": freq_span,
            "direction": direction,
            "fit_model": "quadratic",
        }

        # Merge defaults with user params
        params_def = {**flux_defaults}
        params = {**params_def, **params, "flux": True}

        # Create inner QubitSpec (go=False) to inherit its config
        self.expt = QubitSpec(cfg_dict, qi, go=False, params=params, style=style, check_params=False)
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params
        self._flux_converter = converter

        # Build gain_pts and freq_pts from config
        cfg_e = self.cfg.expt
        if cfg_e["lin_freq"] and converter is not None:
            freq_start = converter.gain_to_freq(cfg_e["gain_start"])
            freq_stop = converter.gain_to_freq(cfg_e["gain_stop"])
            self.freq_pts = np.linspace(float(freq_start), float(freq_stop), int(cfg_e["expts_gain"]))
            self.gain_pts = converter.freq_to_gain(self.freq_pts)
        else:
            self.gain_pts = np.linspace(
                cfg_e["gain_start"], cfg_e["gain_stop"], int(cfg_e["expts_gain"]),
            )
            if converter is not None:
                self.freq_pts = converter.gain_to_freq(self.gain_pts)
            else:
                self.freq_pts = None

        if go:
            self.run(progress=progress, display=display, analyze=True,
                     min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=True, display=False):
        from pathlib import Path

        data = {"time": []}
        qubit_freq_list = []
        kappa_list = []
        fit_params_list = []
        r2_list = []
        timestamp_list = []
        elapsed_list = []
        t_acq_start = time.time()
        csv_path = Path(self.fname).with_suffix(".csv")

        # Initialize center frequency from param or None (first scan uses config default)
        center_freq = self.cfg.expt.start_center_freq

        max_retries = 3
        first_data = None  # template for NaN-filling on failure

        for i, g in enumerate(tqdm(self.gain_pts, disable=not progress)):
            self.expt.cfg.expt["flux_gain"] = float(g)

            # Use previous fit frequency to center the next scan
            if center_freq is not None:
                self.expt.cfg.expt["start"] = center_freq - self.expt.cfg.expt["span"] / 2

            # Run inner QubitSpec with retry on QICK errors
            data_new = None
            for attempt in range(max_retries):
                try:
                    data_new = self.expt.acquire(progress=False)
                    self.expt.analyze(data=data_new, verbose=False)
                    if first_data is None:
                        first_data = data_new
                    break
                except RuntimeError as e:
                    if attempt < max_retries - 1:
                        print(f"  Retry {attempt+1}/{max_retries} for gain={g:.4f}: {e}")
                        time.sleep(1)
                    else:
                        print(f"  SKIP gain={g:.4f} after {max_retries} retries: {e}")
                        # Create NaN-filled data matching first successful scan
                        if first_data is not None:
                            data_new = {}
                            for k, v in first_data.items():
                                if isinstance(v, np.ndarray):
                                    data_new[k] = np.full_like(v, np.nan)
                                else:
                                    data_new[k] = v
                            data_new["best_fit"] = [np.nan, np.nan, np.nan, np.nan]
                        else:
                            raise  # no template yet, can't continue

            if display:
                self.expt.display(data=data_new)

            # Accumulate raw 2D data
            for key in ("avgi", "avgq", "amps", "phases", "xpts"):
                if key in data_new:
                    if i == 0:
                        data[key] = []
                    data[key].append(data_new[key])
            now = time.time()
            data["time"].append(now)
            timestamp_list.append(datetime.fromtimestamp(now).isoformat())
            elapsed_list.append(now - t_acq_start)

            # Extract fit results
            best_fit = data_new.get("best_fit", [np.nan, np.nan, np.nan, np.nan])
            qubit_freq_list.append(best_fit[2])
            kappa_list.append(best_fit[3])
            fit_params_list.append(list(best_fit))
            r2_list.append(data_new.get("r2", np.nan))

            # Update center frequency from fit for next scan
            if self.expt.status:
                center_freq = best_fit[2]

            # Store analysis summaries in data for interim save
            data["gain_pts"] = self.gain_pts
            if self.freq_pts is not None:
                data["freq_pts"] = self.freq_pts
            data["qubit_freq_list"] = np.array(qubit_freq_list)
            data["kappa_list"] = np.array(kappa_list)
            data["fit_params_list"] = np.array(fit_params_list)
            data["r2_list"] = np.array(r2_list)

            if self.save_interim:
                self.save_data(data=data)

            # Save CSV incrementally
            gains_so_far = self.gain_pts[:i + 1]
            num_cols = [gains_so_far]
            header_parts = ["timestamp", "gain"]
            if self.freq_pts is not None:
                num_cols.append(self.freq_pts[:i + 1])
                header_parts.append("freq")
            num_cols.extend([
                data["qubit_freq_list"], data["kappa_list"],
                np.array(elapsed_list),
            ])
            header_parts.extend(["qubit_freq", "kappa", "elapsed_s"])
            num_cols = np.column_stack(num_cols)
            header = ",".join(header_parts)
            with open(csv_path, "w") as f_csv:
                f_csv.write(header + "\n")
                for row_idx in range(num_cols.shape[0]):
                    ts = timestamp_list[row_idx]
                    nums = ",".join(f"{v}" for v in num_cols[row_idx])
                    f_csv.write(f"{ts},{nums}\n")

        # Convert lists to arrays
        for k, a in data.items():
            if not isinstance(a, np.ndarray):
                data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit_model='quadratic', Ec=None, **kwargs):
        if data is None:
            data = self.data

        gain_pts = data["gain_pts"]
        qubit_freq_list = data["qubit_freq_list"]

        # Fit quadratic: freq = a*g^2 + b*g + c (always performed)
        mask = ~np.isnan(qubit_freq_list)
        if np.sum(mask) >= 3:
            coeffs = np.polyfit(gain_pts[mask], qubit_freq_list[mask], 2)
            data["freq_coeffs"] = tuple(coeffs)
            data["freq_fit"] = np.polyval(coeffs, gain_pts)
            a, b, c = coeffs
            data["gain_sweet_spot"] = -b / (2 * a)
            data["f_ge_max"] = c - b**2 / (4 * a)
        else:
            data["freq_coeffs"] = None
            data["freq_fit"] = None
            data["gain_sweet_spot"] = np.nan
            data["f_ge_max"] = np.nan

        # Optional transmon SQUID model fit

        data["transmon_params"] = np.array([])
        data["transmon_fit"] = np.array([])
        if fit_model == "transmon" and np.sum(mask) >= 5:
            qi = self.cfg.expt["qubit"][0]
            cfg_qubit = self.cfg.device.qubit
            if Ec is None:
                E_C = 200.0
                try:
                    model_path = self.config_file[0:-4] + "_model.yml"
                    model_cfg = config.load(model_path)
                    if hasattr(model_cfg, "Ec") and model_cfg.Ec[qi] is not None:
                        E_C = model_cfg.Ec[qi] * 1000  # GHz -> MHz
                except (FileNotFoundError, OSError):
                    pass
            else:
                E_C = float(Ec)
            pOpt, pCov, _ = fitter.fit_transmon_flux(
                gain_pts[mask], qubit_freq_list[mask], E_C,
            )
            if np.isfinite(pCov).all():
                data["transmon_params"] = np.array(pOpt)
                data["transmon_fit"] = fitter.transmon_flux(gain_pts, *pOpt)
                # Use transmon sweet spot and f_max if fit succeeded
                data["gain_sweet_spot"] = pOpt[3]  # g_offset
                data["f_ge_max"] = pOpt[0]  # f_max

        return data

    def display(self, data=None, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        gain_pts = data["gain_pts"]
        qubit_freq_list = data["qubit_freq_list"]
        qi = self.cfg.expt["qubit"][0]

        # --- Freq vs gain plot ---
        fig, ax = plt.subplots()
        ax.plot(gain_pts, qubit_freq_list, "o", label="data")
        if data.get("freq_fit") is not None:
            ax.plot(gain_pts, data["freq_fit"], "-", label="quadratic fit")
        if len(data.get("transmon_fit", [])) > 0:
            gain_dense = np.linspace(gain_pts.min(), gain_pts.max(), 200)
            ax.plot(gain_dense, fitter.transmon_flux(gain_dense, *data["transmon_params"]),
                    "--", label="transmon fit")
        if data.get("freq_fit") is not None or len(data.get("transmon_fit", [])) > 0:
            ax.legend()
        ax.set_xlabel("Flux Gain")
        ax.set_ylabel("Qubit Frequency (MHz)")
        ax.set_title(f"Qubit Spectroscopy vs Flux Gain Q{qi}")
        self.save_fig(fig, suffix="_qspec_vs_gain")

        # --- Waterfall plot of all 1D spectra (skip if too many traces) ---
        amps_all = data.get("amps")
        xpts_all = data.get("xpts")
        if amps_all is not None and xpts_all is not None and len(gain_pts) <= 300:
            fig, ax = plt.subplots(figsize=(10, 8))
            # Compute vertical offset from median signal range
            ranges = []
            for sig in amps_all:
                s = np.asarray(sig)
                if len(s) > 2:
                    ranges.append(np.ptp(s[1:-1]))
            offset_scale = np.median(ranges) * 0.8 if ranges else 1

            for i, (xpts, signal) in enumerate(zip(xpts_all, amps_all)):
                xpts = np.asarray(xpts)
                signal = np.asarray(signal)
                ax.plot(xpts[1:-1], signal[1:-1] + i * offset_scale,
                        label=f"gain={gain_pts[i]:.3f}")

            ax.set_xlabel("Qubit Frequency (MHz)")
            ax.set_ylabel("Signal (offset)")
            ax.set_title(f"Qubit Spec vs Flux Gain Q{qi} (waterfall)")
            if len(gain_pts) <= 20:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
            self.save_fig(fig, suffix="_waterfall")

            # --- Color plot of amps ---
            # Each gain point may sweep a different frequency range, so
            # interpolate all spectra onto a common frequency grid.
            from scipy.interpolate import interp1d

            avgi_all = data.get("avgi")
            freq_trimmed = [np.asarray(x) for x in xpts_all]
            avgi_trimmed = [np.asarray(a) for a in avgi_all]
            f_min = min(f.min() for f in freq_trimmed)
            f_max = max(f.max() for f in freq_trimmed)
            n_common = len(freq_trimmed[0])
            common_freq = np.linspace(f_min, f_max, n_common)

            amps_interp = np.full((len(gain_pts), n_common), np.nan)
            for i, (f, a) in enumerate(zip(freq_trimmed, avgi_trimmed)):
                interp_fn = interp1d(f, a, kind="linear",
                                     bounds_error=False, fill_value=np.nan)
                mask = (common_freq >= f.min()) & (common_freq <= f.max())
                amps_interp[i, mask] = interp_fn(common_freq[mask])

            n_gain = len(gain_pts)
            fig_w = max(8, n_gain / 40)
            fig, ax = plt.subplots(figsize=(fig_w, 6))
            pcm = ax.pcolormesh(gain_pts, common_freq, amps_interp.T,
                                cmap="viridis", shading="nearest",
                                rasterized=True)
            plt.colorbar(pcm, ax=ax, label="I (ADC Units)")
            ax.set_xlabel("Flux Gain")
            ax.set_ylabel("Qubit Frequency (MHz)")
            self.save_fig(fig, suffix="_colorplot")

            # --- Color plot of Q channel ---
            avgq_all = data.get("avgq")
            if avgq_all is not None:
                avgq_trimmed = [np.asarray(a) for a in avgq_all]
                avgq_interp = np.full((len(gain_pts), n_common), np.nan)
                for i, (f, a) in enumerate(zip(freq_trimmed, avgq_trimmed)):
                    interp_fn = interp1d(f, a, kind="linear",
                                         bounds_error=False, fill_value=np.nan)
                    mask = (common_freq >= f.min()) & (common_freq <= f.max())
                    avgq_interp[i, mask] = interp_fn(common_freq[mask])

                fig, ax = plt.subplots(figsize=(fig_w, 6))
                pcm = ax.pcolormesh(gain_pts, common_freq, avgq_interp.T,
                                    cmap="viridis", shading="nearest",
                                    rasterized=True)
                plt.colorbar(pcm, ax=ax, label="Q (ADC Units)")
                ax.set_xlabel("Flux Gain")
                ax.set_ylabel("Qubit Frequency (MHz)")
                self.save_fig(fig, suffix="_colorplot_q")

    def update(self, verbose=True):
        qi = self.cfg.expt["qubit"][0]
        data = self.data
        if data.get("freq_coeffs") is not None:
            config.update_qubit(
                self.config_file, "f_ge_max", data["f_ge_max"], qi, verbose=verbose,
            )
            config.update_qubit(
                self.config_file, "sweet_spot_ac", data["gain_sweet_spot"], qi, verbose=verbose,
            )
            a, b, c = data["freq_coeffs"]
            config.update_config(
                self.config_file, "hw.soc.dacs.flux", "quad_a", a, qi, verbose=verbose,
            )
            config.update_config(
                self.config_file, "hw.soc.dacs.flux", "quad_b", b, qi, verbose=verbose,
            )
            config.update_config(
                self.config_file, "hw.soc.dacs.flux", "quad_c", c, qi, verbose=verbose,
            )
        if len(data.get("transmon_params", [])) > 0:
            f_max, E_C, g_period, g_offset, d = data["transmon_params"]
            config.update_config(
                self.config_file, "hw.soc.dacs.flux", "transmon_f_max", f_max, qi, verbose=verbose,
            )
            config.update_config(
                self.config_file, "hw.soc.dacs.flux", "transmon_E_C", E_C, qi, verbose=verbose,
            )
            config.update_config(
                self.config_file, "hw.soc.dacs.flux", "transmon_g_period", g_period, qi, verbose=verbose,
            )
            config.update_config(
                self.config_file, "hw.soc.dacs.flux", "transmon_g_offset", g_offset, qi, verbose=verbose,
            )
            config.update_config(
                self.config_file, "hw.soc.dacs.flux", "transmon_d", d, qi, verbose=verbose,
            )


class QubitSpecFluxSweep(QickExperiment2DSimple):
    """
    QubitSpecFluxSweep: Sweep qubit spectroscopy across an arbitrary parameter.

    Like QubitSpecFastFlux, runs a QubitSpec at each sweep point and fits each
    scan's qubit frequency to center the next scan on it. The swept parameter
    is specified via ``sweep_var``.

    Sweep parameters (passed via params dict):
        sweep_var: Name of the inner-experiment parameter to vary (e.g. "flux_gain",
            "length", "gain"). Required.
        sweep_pts: Explicit array of sweep values. If provided, overrides
            sweep_start/sweep_stop/expts_sweep.
        sweep_start, sweep_stop: Start and stop values for the sweep.
        expts_sweep: Number of sweep points (default: 50).
        sweep_label: Y-axis label for plots (default: sweep_var).
        update: Whether to update inner config with fit results after each point.
        start_center_freq: Initial qubit frequency center; None = use config f_ge.
        predistorted: If True, use QubitSpecPredist (predistorted flux pulse)
            as the inner experiment instead of QubitSpec. Requires flux_alphas,
            flux_taus, and flux_alpha0 in params.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        style="fine",
        display=True,
        min_r2=None,
        max_err=None,
    ):
        sweep_var = params.get("sweep_var", "flux_gain")
        prefix = prefix or f"qspec_sweep_{sweep_var}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)
        self._cfg_dict = cfg_dict
        self._qi = qi
        self._style = style

        predistorted = params.get("predistorted", False)

        params_def = {
            "sweep_var": sweep_var,
            "sweep_pts": None,
            "sweep_start": 0.0,
            "sweep_stop": 1.0,
            "expts_sweep": 50,
            "sweep_label": sweep_var,
            "log_sweep": False,
            "update": True,
            "start_center_freq": None,
            "predistorted": predistorted,
        }

        params = {**params_def, **params, "flux": True}

        # Load predistortion params from config if not provided explicitly
        if predistorted:
            cfg_flux = self.cfg.hw.soc.dacs.flux
            if "flux_alphas" not in params or not params["flux_alphas"]:
                params["flux_alphas"] = list(cfg_flux.step_alphas[qi])
            if "flux_taus" not in params or not params["flux_taus"]:
                params["flux_taus"] = list(cfg_flux.step_taus[qi])
            if "flux_alpha0" not in params:
                step_alpha0 = getattr(cfg_flux, "step_alpha0", None)
                if step_alpha0 is not None:
                    val = step_alpha0[qi]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        params["flux_alpha0"] = val

        # Create inner experiment (go=False) to inherit its config
        if predistorted:
            self.expt = QubitSpecPredist(
                cfg_dict, qi, go=False, params=params, style=style, check_params=False)
        else:
            self.expt = QubitSpec(
                cfg_dict, qi, go=False, params=params, style=style, check_params=False)
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params

        # Build sweep_pts
        cfg_e = self.cfg.expt
        if cfg_e["sweep_pts"] is not None:
            self.sweep_pts = np.asarray(cfg_e["sweep_pts"])
        else:
            self.sweep_pts = np.linspace(
                cfg_e["sweep_start"], cfg_e["sweep_stop"], int(cfg_e["expts_sweep"]),
            )

        self._sweep_var = cfg_e["sweep_var"]
        self._sweep_label = cfg_e["sweep_label"]
        self._log_sweep = cfg_e["log_sweep"]

        if go:
            self.run(progress=progress, display=display, analyze=True,
                     min_r2=min_r2, max_err=max_err)

    @classmethod
    def from_h5file(cls, fname):
        self = super().from_h5file(fname)
        cfg_e = self.cfg.expt
        self._sweep_var = cfg_e.get("sweep_var", "flux_gain")
        self._sweep_label = cfg_e.get("sweep_label", self._sweep_var)
        self._log_sweep = cfg_e.get("log_sweep", False)
        return self

    def acquire(self, progress=True, display=False):
        from pathlib import Path

        data = {"time": []}
        qubit_freq_list = []
        kappa_list = []
        fit_params_list = []
        r2_list = []
        timestamp_list = []
        elapsed_list = []
        t_acq_start = time.time()
        csv_path = Path(self.fname).with_suffix(".csv")

        center_freq = self.cfg.expt.start_center_freq

        max_retries = 3
        first_data = None

        for i, val in enumerate(tqdm(self.sweep_pts, disable=not progress)):
            self.expt.cfg.expt[self._sweep_var] = float(val)

            # Use previous fit frequency to center the next scan
            if center_freq is not None:
                self.expt.cfg.expt["start"] = center_freq - self.expt.cfg.expt["span"] / 2

            # Run inner QubitSpec with retry on QICK errors
            data_new = None
            for attempt in range(max_retries):
                try:
                    data_new = self.expt.acquire(progress=False)
                    self.expt.analyze(data=data_new, verbose=False)
                    if first_data is None:
                        first_data = data_new
                    break
                except RuntimeError as e:
                    if attempt < max_retries - 1:
                        print(f"  Retry {attempt+1}/{max_retries} for {self._sweep_var}={val:.4f}: {e}")
                        time.sleep(1)
                    else:
                        print(f"  SKIP {self._sweep_var}={val:.4f} after {max_retries} retries: {e}")
                        if first_data is not None:
                            data_new = {}
                            for k, v in first_data.items():
                                if isinstance(v, np.ndarray):
                                    data_new[k] = np.full_like(v, np.nan)
                                else:
                                    data_new[k] = v
                            data_new["best_fit"] = [np.nan, np.nan, np.nan, np.nan]
                        else:
                            raise

            if display:
                self.expt.display(data=data_new)

            # Accumulate raw 2D data
            for key in ("avgi", "avgq", "amps", "phases", "xpts"):
                if key in data_new:
                    if i == 0:
                        data[key] = []
                    data[key].append(data_new[key])
            now = time.time()
            data["time"].append(now)
            timestamp_list.append(datetime.fromtimestamp(now).isoformat())
            elapsed_list.append(now - t_acq_start)

            # Extract fit results
            best_fit = data_new.get("best_fit", [np.nan, np.nan, np.nan, np.nan])
            qubit_freq_list.append(best_fit[2])
            kappa_list.append(best_fit[3])
            fit_params_list.append(list(best_fit))
            r2_list.append(data_new.get("r2", np.nan))

            # Update center frequency from fit for next scan
            if self.expt.status:
                center_freq = best_fit[2]

            # Store analysis summaries in data for interim save
            data["sweep_pts"] = self.sweep_pts
            data["qubit_freq_list"] = np.array(qubit_freq_list)
            data["kappa_list"] = np.array(kappa_list)
            data["fit_params_list"] = np.array(fit_params_list)
            data["r2_list"] = np.array(r2_list)

            if self.save_interim:
                self.save_data(data=data)

            # Save CSV incrementally
            sweep_so_far = self.sweep_pts[:i + 1]
            num_cols = np.column_stack([
                sweep_so_far,
                data["qubit_freq_list"], data["kappa_list"],
                np.array(elapsed_list),
            ])
            header = f"timestamp,{self._sweep_var},qubit_freq,kappa,elapsed_s"
            with open(csv_path, "w") as f_csv:
                f_csv.write(header + "\n")
                for row_idx in range(num_cols.shape[0]):
                    ts = timestamp_list[row_idx]
                    nums = ",".join(f"{v}" for v in num_cols[row_idx])
                    f_csv.write(f"{ts},{nums}\n")

        # Convert lists to arrays
        for k, a in data.items():
            if not isinstance(a, np.ndarray):
                data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, n_exp=3, f_inf=None, t_range=None,
                flux_converter="auto", flux_gain_target=None,
                alpha0=None, fitparams=None, outlier_sigma=None, **kwargs):
        """
        Analyze sweep data. For flux_lead_time sweeps, fits a sum of exponentials
        to characterize the step response (arXiv:2503.04610, Figure 4).

        Two fitting modes:
        1. **Flux space** (default when transmon/quadratic model is in config):
           converts to normalized flux s(t) = α₀ + Σ αᵢ exp(-t/τᵢ), which
           linearizes the nonlinear frequency–flux mapping.
        2. **Frequency space** (fallback): fits f(t) = f_inf + Σ aᵢ exp(-t/τᵢ).
           Use `f_inf` to fix the steady-state frequency when the measurement
           window is shorter than the longest time constant.

        Args:
            data: Data dict (default: self.data)
            n_exp: Number of exponential terms for step response fit
            f_inf: Fix the steady-state frequency (MHz) for freq-space fitting.
            t_range: Tuple (t_min, t_max) in µs to restrict the fit range.
            flux_converter: FluxConverter for flux-space fitting. "auto"
                (default) builds one from the config if transmon/quadratic
                model params are available. Pass None to force freq-space
                fitting, or a FluxConverter instance to use it directly.
            flux_gain_target: Target flux gain used in the experiment. If None,
                uses the experiment's flux_gain parameter.
            alpha0: Steady-state value α₀ for flux-space fitting. If None
                (default), fit as a free parameter. Use 0.0 for high-pass
                bias tee.
            fitparams: Optional initial parameters for the fit.
            outlier_sigma: If set, mark data points whose fit residual exceeds
                this many standard deviations as outliers (stored in
                data["outlier_mask"], True = outlier).
        """
        if data is None:
            data = self.data

        if self._sweep_var == "flux_lead_time":
            # Auto-build flux_converter from config if available
            if flux_converter == "auto":
                from ...analysis.fitting import flux_converter_from_config
                qi = self.cfg.expt["qubit"][0]
                # Determine direction from flux_gain vs sweet spot
                flux_gain = self.cfg.expt.get("flux_gain", -0.35)
                sweet_spot = self.cfg.device.qubit.sweet_spot_ac[qi]
                direction = "neg" if flux_gain < sweet_spot else "pos"
                flux_converter = flux_converter_from_config(
                    self.cfg.hw.soc.dacs.flux, self.cfg.device.qubit, qi,
                    direction=direction)

            mask = ~np.isnan(data["qubit_freq_list"])
            sweep_pts = data["sweep_pts"][mask]
            freq_list = data["qubit_freq_list"][mask]

            if flux_converter is not None:
                # Flux-space fitting (fit on masked data, evaluate on all points)
                from ...analysis.fitting import fit_step_response_flux

                if flux_gain_target is None:
                    flux_gain_target = self.cfg.expt.get("flux_gain", -0.35)

                params, s_data, s_fit = fit_step_response_flux(
                    data["sweep_pts"], data["qubit_freq_list"],
                    flux_converter, flux_gain_target,
                    n_exp=n_exp, alpha0=alpha0,
                    fitparams=fitparams, t_range=t_range,
                )
                data["step_params"] = params
                data["step_s_data"] = s_data
                data["step_s_fit"] = s_fit
                data["step_fit_mode"] = "flux"

                # Convert fit back to frequency space for display
                delta = params["delta_gain"]
                sweet = params["gain_sweet"]
                data["step_fit_curve"] = flux_converter.gain_to_freq(
                    s_fit * delta + sweet
                )

            else:
                # Frequency-space fitting
                from ...analysis.fitting import (
                    fit_step_response,
                    sum_exp_func,
                    extract_step_response_params,
                )

                pOpt, pCov, pInit = fit_step_response(
                    sweep_pts, freq_list, n_exp=n_exp,
                    fitparams=fitparams, f_inf=f_inf, t_range=t_range,
                )
                data["step_fit"] = pOpt
                data["step_fit_err"] = pCov
                data["step_params"] = extract_step_response_params(pOpt)
                data["step_fit_curve"] = sum_exp_func(data["sweep_pts"], *pOpt)
                data["step_fit_mode"] = "freq"

            # Compute outlier mask based on fit residuals
            if outlier_sigma is not None and "step_fit_curve" in data:
                residuals = data["qubit_freq_list"] - data["step_fit_curve"]
                valid = ~np.isnan(residuals)
                std = np.nanstd(residuals)
                data["outlier_mask"] = np.abs(residuals) > outlier_sigma * std
            else:
                data["outlier_mask"] = np.zeros(len(data["sweep_pts"]), dtype=bool)

        return data

    def update(self, verbose=True):
        """Save fitted step response params (alpha0, alphas, taus) to the config."""
        qi = self.cfg.expt["qubit"][0]
        data = self.data
        step_params = data.get("step_params")
        if step_params is None:
            print("No step_params found in data — run analyze() first.")
            return

        cfg = config.load(self.config_file)
        flux_sec = cfg["hw"]["soc"]["dacs"]["flux"]
        n = len(flux_sec.get("ch", [0, 0, 0, 0]))

        # alpha0 (flux-space fit) or f_inf (freq-space fit)
        if "alpha0" in step_params:
            config.update_config(
                self.config_file, "hw.soc.dacs.flux", "step_alpha0",
                float(step_params["alpha0"]), qi, verbose=verbose,
            )
            # Reload after update_config's save
            cfg = config.load(self.config_file)
            flux_sec = cfg["hw"]["soc"]["dacs"]["flux"]
        elif "f_inf" in step_params:
            config.update_config(
                self.config_file, "hw.soc.dacs.flux", "step_f_inf",
                float(step_params["f_inf"]), qi, verbose=verbose,
            )
            cfg = config.load(self.config_file)
            flux_sec = cfg["hw"]["soc"]["dacs"]["flux"]

        # alphas and taus are variable-length lists per qubit — update_config
        # doesn't handle list values, so we load/save manually.
        for key, vals in [
            ("step_alphas", [round(float(a), 6) for a in step_params["alphas"]]),
            ("step_taus", [round(float(t), 6) for t in step_params["taus"]]),
        ]:
            if key not in flux_sec:
                flux_sec[key] = [[] for _ in range(n)]
            old = flux_sec[key][qi]
            flux_sec[key][qi] = vals
            if verbose:
                print(f"*Set cfg hw.soc.dacs.flux {qi} {key} to {vals} from {old}*")

        config.save(cfg, self.config_file)

    def display(self, data=None, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        sweep_pts = data["sweep_pts"]
        qubit_freq_list = data["qubit_freq_list"]
        qi = self.cfg.expt["qubit"][0]
        sweep_label = self._sweep_label
        outliers = data.get("outlier_mask", np.zeros(len(sweep_pts), dtype=bool))
        show = ~outliers

        # --- Freq vs sweep parameter plot (with optional step response fit) ---
        has_fit = "step_fit_curve" in data and data["step_fit_curve"] is not None
        if has_fit:
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
            ax = axes[0]
        else:
            fig, ax = plt.subplots()

        ax.plot(sweep_pts[show], qubit_freq_list[show], "o", ms=4, label="data")
        if has_fit:
            # Plot fit curve
            ax.plot(sweep_pts, data["step_fit_curve"], "-", color="C3", lw=2, label="fit")
            ax.legend(fontsize=9)

            # Print fitted parameters as text
            p = data["step_params"]
            fit_mode = data.get("step_fit_mode", "freq")
            lines = []
            if "f_inf" in p:
                lines.append(f"$f_\\infty$ = {p['f_inf']:.2f} MHz")
            if "alpha0" in p:
                lines.append(f"$\\alpha_0$ = {p['alpha0']:.4f}")
            for i, (a, tau) in enumerate(zip(p["alphas"], p["taus"])):
                if fit_mode == "flux":
                    lines.append(f"$\\alpha_{{{i+1}}}$ = {a:.4f}, $\\tau_{{{i+1}}}$ = {tau:.3f} µs")
                else:
                    lines.append(f"$a_{{{i+1}}}$ = {a:.2f} MHz, $\\tau_{{{i+1}}}$ = {tau:.3f} µs")
            ax.text(0.02, 0.97, "\n".join(lines), transform=ax.transAxes,
                    fontsize=10, va="top", ha="left",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8))

            # Residuals subplot
            ax_r = axes[1]
            residuals = qubit_freq_list - data["step_fit_curve"]
            ax_r.plot(sweep_pts[show], residuals[show], "o", ms=3, color="C2")
            ax_r.axhline(0, color="k", lw=0.5, ls="--")
            ax_r.set_ylabel("Residual (MHz)")
            ax_r.set_xlabel(sweep_label)
            if self._log_sweep:
                ax_r.set_xscale("log")

        ax.set_ylabel("Qubit Frequency (MHz)")
        if not has_fit:
            ax.set_xlabel(sweep_label)
        if self._log_sweep:
            ax.set_xscale("log")
        self.save_fig(fig, suffix="_qspec_vs_sweep")

        # --- Waterfall plot of all 1D spectra ---
        amps_all = data.get("amps")
        xpts_all = data.get("xpts")
        if amps_all is not None and xpts_all is not None and len(sweep_pts) <= 300:
            fig, ax = plt.subplots(figsize=(10, 8))
            ranges = []
            for sig in amps_all:
                s = np.asarray(sig)
                if len(s) > 2:
                    ranges.append(np.ptp(s[1:-1]))
            offset_scale = np.median(ranges) * 0.8 if ranges else 1

            for i, (xpts, signal) in enumerate(zip(xpts_all, amps_all)):
                xpts = np.asarray(xpts)
                signal = np.asarray(signal)
                ax.plot(xpts[1:-1], signal[1:-1] + i * offset_scale,
                        label=f"{self._sweep_var}={sweep_pts[i]:.3f}")

            ax.set_xlabel("Qubit Frequency (MHz)")
            ax.set_ylabel("Signal (offset)")
            if len(sweep_pts) <= 20:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
            self.save_fig(fig, suffix="_waterfall")

            # --- Color plot of I channel ---
            from scipy.interpolate import interp1d

            avgi_all = data.get("avgi")
            freq_trimmed = [np.asarray(x) for x in xpts_all]
            avgi_trimmed = [np.asarray(a) for a in avgi_all]
            f_min = min(f.min() for f in freq_trimmed)
            f_max = max(f.max() for f in freq_trimmed)
            n_common = len(freq_trimmed[0])
            common_freq = np.linspace(f_min, f_max, n_common)

            amps_interp = np.full((len(sweep_pts), n_common), np.nan)
            for i, (f, a) in enumerate(zip(freq_trimmed, avgi_trimmed)):
                interp_fn = interp1d(f, a, kind="linear",
                                     bounds_error=False, fill_value=np.nan)
                mask = (common_freq >= f.min()) & (common_freq <= f.max())
                amps_interp[i, mask] = interp_fn(common_freq[mask])

            n_sweep = len(sweep_pts)
            fig_w = max(8, n_sweep / 40)
            fig, ax = plt.subplots(figsize=(fig_w, 6))
            pcm = ax.pcolormesh(sweep_pts, common_freq, amps_interp.T,
                                cmap="viridis", shading="nearest",
                                rasterized=True)
            plt.colorbar(pcm, ax=ax, label="I (ADC Units)")
            ax.set_xlabel(sweep_label)
            ax.set_ylabel("Qubit Frequency (MHz)")
            if self._log_sweep:
                ax.set_xscale("log")
            self.save_fig(fig, suffix="_colorplot")

            # --- Color plot of Q channel ---
            avgq_all = data.get("avgq")
            if avgq_all is not None:
                avgq_trimmed = [np.asarray(a) for a in avgq_all]
                avgq_interp = np.full((len(sweep_pts), n_common), np.nan)
                for i, (f, a) in enumerate(zip(freq_trimmed, avgq_trimmed)):
                    interp_fn = interp1d(f, a, kind="linear",
                                         bounds_error=False, fill_value=np.nan)
                    mask = (common_freq >= f.min()) & (common_freq <= f.max())
                    avgq_interp[i, mask] = interp_fn(common_freq[mask])

                fig, ax = plt.subplots(figsize=(fig_w, 6))
                pcm = ax.pcolormesh(sweep_pts, common_freq, avgq_interp.T,
                                    cmap="viridis", shading="nearest",
                                    rasterized=True)
                plt.colorbar(pcm, ax=ax, label="Q (ADC Units)")
                ax.set_xlabel(sweep_label)
                ax.set_ylabel("Qubit Frequency (MHz)")
                if self._log_sweep:
                    ax.set_xscale("log")
                self.save_fig(fig, suffix="_colorplot_q")
