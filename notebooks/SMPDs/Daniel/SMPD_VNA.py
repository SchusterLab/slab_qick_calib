"""
Class-based VNA data taking helpers for SMPD measurements.

The classes here keep the notebook focused on experiment parameters while the
instrument loops, HDF5 formats, plotting, and fitres compatibility live in one
place.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import h5py
import numpy as np


@dataclass
class VNASweepConfig:
    start_ghz: float
    stop_ghz: float
    step_ghz: float
    points_per_segment: int
    measurement: str = "S43"
    rf_power_dbm: float = -60.0
    if_bw_hz: float = 10_000.0
    avg_time_s: float = 50.0
    phase_units: str = "deg"
    phase_fit_edge_fraction: float = 0.1
    extra_settings: dict[str, Any] = field(default_factory=dict)


@dataclass
class SweepResult:
    freq_ghz: np.ndarray
    mag_db: np.ndarray
    phase_raw: np.ndarray
    phase_rad: np.ndarray
    config: Optional[VNASweepConfig] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def s21(self, corrected_phase: bool = True) -> np.ndarray:
        phase = self.phase_rad if corrected_phase else phase_to_rad(self.phase_raw, self.config)
        return db_phase_to_complex(self.mag_db, phase)

    def save_h5(self, path: str | os.PathLike[str]) -> Path:
        path = ensure_h5_path(path)
        with h5py.File(path, "w") as handle:
            handle.attrs["smpd_result_type"] = "sweep"
            handle.attrs["config_json"] = _to_json_attr(self.config)
            handle.attrs["metadata_json"] = _to_json_attr(self.metadata)
            handle.create_dataset("freq_ghz", data=np.asarray(self.freq_ghz, dtype=float))
            handle.create_dataset("mag_db", data=np.asarray(self.mag_db, dtype=float))
            handle.create_dataset("phase_raw", data=np.asarray(self.phase_raw, dtype=float))
            handle.create_dataset("phase_rad", data=np.asarray(self.phase_rad, dtype=float))
        return path

    def save_pickle(self, path: str | os.PathLike[str], legacy_tuple: bool = True) -> Path:
        if _is_h5_path(path):
            return self.save_h5(path)
        path = ensure_pickle_path(path)
        payload: Any = (self.freq_ghz, self.mag_db, self.phase_raw) if legacy_tuple else self
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path


@dataclass
class ResonanceFit:
    params: tuple[Any, ...]
    initial_guess_ghz: float
    fit_type: str = "transmission"

    @property
    def fr_ghz(self) -> float:
        return float(self.params[0])

    @property
    def qr(self) -> float:
        return float(self.params[1])

    @property
    def qc(self) -> float:
        return float(self.params[6])


@dataclass
class PumpBiasScanResult:
    freq_ghz: np.ndarray
    gain_db: np.ndarray
    pump_powers_dbm: np.ndarray
    dc_biases_mv: np.ndarray
    waste_resonance_ghz: float
    baseline_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def gain_at_resonance(self, window_hz: float = 0.5e6) -> np.ndarray:
        half_window_ghz = 0.5 * window_hz / 1e9
        mask = np.abs(self.freq_ghz - self.waste_resonance_ghz) <= half_window_ghz
        if not np.any(mask):
            idx = int(np.argmin(np.abs(self.freq_ghz - self.waste_resonance_ghz)))
            mask = np.zeros_like(self.freq_ghz, dtype=bool)
            mask[idx] = True
        return np.nanmean(self.gain_db[:, :, mask], axis=2)

    def save_h5(self, path: str | os.PathLike[str]) -> Path:
        path = ensure_h5_path(path)
        with h5py.File(path, "w") as handle:
            handle.attrs["smpd_result_type"] = "pump_bias_scan"
            handle.attrs["waste_resonance_ghz"] = float(self.waste_resonance_ghz)
            handle.attrs["baseline_path"] = "" if self.baseline_path is None else str(self.baseline_path)
            handle.attrs["metadata_json"] = _to_json_attr(self.metadata)
            handle.create_dataset("freq_ghz", data=np.asarray(self.freq_ghz, dtype=float))
            handle.create_dataset("gain_db", data=np.asarray(self.gain_db, dtype=float))
            handle.create_dataset("pump_powers_dbm", data=np.asarray(self.pump_powers_dbm, dtype=float))
            handle.create_dataset("dc_biases_mv", data=np.asarray(self.dc_biases_mv, dtype=float))
        return path

    def save_pickle(self, path: str | os.PathLike[str], legacy_gain_only: bool = False) -> Path:
        if _is_h5_path(path):
            return self.save_h5(path)
        path = ensure_pickle_path(path)
        payload: Any = self.gain_db if legacy_gain_only else self
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path


@dataclass
class PowerScanResult:
    freq_pow_sweep: np.ndarray
    mag_pow_sweep: np.ndarray
    phase_pow_sweep: np.ndarray
    powers_dbm: np.ndarray
    config: Optional[VNASweepConfig] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def freq_axis(self) -> np.ndarray:
        if np.ndim(self.freq_pow_sweep) == 2:
            return self.freq_pow_sweep[0, :]
        return self.freq_pow_sweep

    def save_h5(self, path: str | os.PathLike[str]) -> Path:
        path = ensure_h5_path(path)
        with h5py.File(path, "w") as handle:
            handle.attrs["smpd_result_type"] = "power_scan"
            handle.attrs["config_json"] = _to_json_attr(self.config)
            handle.attrs["metadata_json"] = _to_json_attr(self.metadata)
            handle.create_dataset("freq_pow_sweep", data=np.asarray(self.freq_pow_sweep, dtype=float))
            handle.create_dataset("mag_pow_sweep", data=np.asarray(self.mag_pow_sweep, dtype=float))
            handle.create_dataset("phase_pow_sweep", data=np.asarray(self.phase_pow_sweep, dtype=float))
            handle.create_dataset("powers_dbm", data=np.asarray(self.powers_dbm, dtype=float))
        return path

    def save_pickle(self, path: str | os.PathLike[str]) -> Path:
        if _is_h5_path(path):
            return self.save_h5(path)
        path = ensure_pickle_path(path)
        with path.open("wb") as handle:
            pickle.dump(
                (
                    self.freq_pow_sweep,
                    self.mag_pow_sweep,
                    self.phase_pow_sweep,
                    list(self.powers_dbm),
                ),
                handle,
            )
        return path


@dataclass
class PowerFitResult:
    power: float
    fr: float
    Q: float
    Qc: float
    Qi: float
    Qc_hat_mag: float
    a: complex
    phi: float
    tau: complex
    kappa: float
    kappa_c: float
    kappa_i: float
    SNR: float
    power_index: int
    range_label: int = 0


@dataclass
class NoiseTimeStreamResult:
    readout_power_dbm: float
    fr_ghz: float
    q_total: float
    ringup_s: float
    sample_interval_s: float
    sigma_i: float
    sigma_q: float
    blob_radius: float
    blob_area_1sigma: float


@dataclass
class NoiseTimeStreamStudyResult:
    power_scan: PowerScanResult
    power_fits: list[PowerFitResult]
    timestreams: list[NoiseTimeStreamResult]

    @property
    def powers_dbm(self) -> np.ndarray:
        return np.array([fit.power for fit in self.power_fits], dtype=float)


@dataclass
class NoiseTimeStreamGridResult:
    path: Path
    readout_powers_dbm: np.ndarray
    if_bandwidths_hz: np.ndarray
    center_ghz: float
    n_samples: int
    n_repeats: int
    center_freqs_ghz: Optional[np.ndarray] = None


class FitResAdapter:
    """Small compatibility wrapper for old and new fitres.py modules."""

    def __init__(
        self,
        fitres_module: Any,
        fit_type: str = "transmission",
        hpd: bool | dict[str, Any] = False,
        phase_first: bool = False,
        retry_without_new_options: bool = True,
    ) -> None:
        self.fitres = fitres_module
        self.fit_type = fit_type
        self.hpd = hpd
        self.phase_first = phase_first
        self.retry_without_new_options = retry_without_new_options

    def finefit(
        self,
        f_ghz: Sequence[float],
        z: Sequence[complex],
        fr_0: Any,
        p0: Optional[Sequence[float]] = None,
    ) -> tuple[Any, ...]:
        f = np.asarray(f_ghz, dtype=float)
        z = np.asarray(z, dtype=complex)
        guess = scalar_frequency(fr_0)
        kwargs = self._supported_fit_kwargs(include_new_options=True)
        if p0 is not None and "p0" in inspect.signature(self.fitres.finefit).parameters:
            kwargs["p0"] = p0
        try:
            return self.fitres.finefit(f, z, guess, **kwargs)
        except TypeError:
            kwargs = self._supported_fit_kwargs(include_new_options=False)
            if p0 is not None and "p0" in inspect.signature(self.fitres.finefit).parameters:
                kwargs["p0"] = p0
            return self.fitres.finefit(f, z, guess, **kwargs)
        except Exception:
            if not self.retry_without_new_options:
                raise
            kwargs = self._supported_fit_kwargs(include_new_options=False)
            if p0 is not None and "p0" in inspect.signature(self.fitres.finefit).parameters:
                kwargs["p0"] = p0
            return self.fitres.finefit(f, z, guess, **kwargs)

    def resfunc3(self, f_ghz: Sequence[float] | float, params: Sequence[Any]) -> np.ndarray:
        kwargs = {}
        if self._function_accepts(self.fitres.resfunc3, "fit_type"):
            kwargs["fit_type"] = self.fit_type
        return self.fitres.resfunc3(f_ghz, *params[:6], **kwargs)

    def fit_result(
        self,
        sweep: SweepResult,
        initial_guess_ghz: Optional[float] = None,
        corrected_phase: bool = True,
    ) -> ResonanceFit:
        guess = initial_guess_ghz
        if guess is None:
            guess = estimate_resonance_from_phase(sweep.freq_ghz, sweep.phase_rad)
        params = self.finefit(sweep.freq_ghz, sweep.s21(corrected_phase=corrected_phase), guess)
        return ResonanceFit(params=params, initial_guess_ghz=float(guess), fit_type=self.fit_type)

    def _supported_fit_kwargs(self, include_new_options: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        sig = inspect.signature(self.fitres.finefit)
        if "fit_type" in sig.parameters:
            kwargs["fit_type"] = self.fit_type
        if include_new_options and "hpd" in sig.parameters:
            kwargs["hpd"] = self.hpd
        if include_new_options and "phase_first" in sig.parameters:
            kwargs["phase_first"] = self.phase_first
        return kwargs

    @staticmethod
    def _function_accepts(func: Callable[..., Any], name: str) -> bool:
        return name in inspect.signature(func).parameters


class SMPDVNA:
    def __init__(
        self,
        vna: Any,
        clear_output_func: Optional[Callable[..., Any]] = None,
        sleep_s: float = 0.1,
    ) -> None:
        self.vna = vna
        self.clear_output = clear_output_func
        self.sleep_s = sleep_s

    def measure_sweep(
        self,
        config: VNASweepConfig,
        live_plot: bool = False,
        progress: bool = True,
    ) -> SweepResult:
        freq_all: list[np.ndarray] = []
        mag_all: list[np.ndarray] = []
        phase_all: list[np.ndarray] = []

        current = float(config.start_ghz)
        stop = float(config.stop_ghz)
        step = float(config.step_ghz)
        if step <= 0:
            raise ValueError("VNASweepConfig.step_ghz must be positive.")

        while current < stop - 1e-12:
            segment_stop = min(current + step, stop)
            settings = self._settings_for_segment(config, current, segment_stop)
            data = self._trans_meas(settings)

            freq = to_float_array(data["xaxis"]) / 1e9
            mag = to_float_array(data["mag"])
            phase = to_float_array(data["phase"])

            freq_all.append(freq)
            mag_all.append(mag)
            phase_all.append(phase)

            current += float(config.step_ghz)

            if progress or live_plot:
                sweep = build_sweep_result(freq_all, mag_all, phase_all, config)
                self._show_progress(sweep, config, current, live_plot)

        return build_sweep_result(freq_all, mag_all, phase_all, config)

    def _settings_for_segment(
        self,
        config: VNASweepConfig,
        start_ghz: float,
        stop_ghz: float,
    ) -> dict[str, Any]:
        if hasattr(self.vna, "reset"):
            self.vna.reset()
        settings = self.vna.trans_default_settings()
        settings.update(
            {
                "measurement": config.measurement,
                "start_freq": start_ghz * 1e9,
                "stop_freq": stop_ghz * 1e9,
                "RFpower": config.rf_power_dbm,
                "ifBW": config.if_bw_hz,
                "avg_time": config.avg_time_s,
                "freq_points": config.points_per_segment,
            }
        )
        settings.update(config.extra_settings)
        return settings

    def _trans_meas(self, settings: dict[str, Any]) -> dict[str, Any]:
        return self.vna.trans_meas(settings)

    def _show_progress(
        self,
        sweep: SweepResult,
        config: VNASweepConfig,
        current_ghz: float,
        live_plot: bool,
    ) -> None:
        if self.clear_output is not None:
            self.clear_output(wait=True)
        measured = current_ghz - config.start_ghz
        total = config.stop_ghz - config.start_ghz
        print(f"Measured {measured:.4f} GHz / {total:.4f} GHz", end="\r")
        if live_plot:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
            ax[0].plot(sweep.freq_ghz, sweep.mag_db)
            ax[0].set_ylabel("Magnitude (dB)")
            ax[0].set_title(f"{config.measurement}, {config.rf_power_dbm:g} dBm")
            ax[1].plot(sweep.freq_ghz, sweep.phase_rad)
            ax[1].set_xlabel("Frequency (GHz)")
            ax[1].set_ylabel("Phase (rad)")
            plt.tight_layout()
            plt.show()
        if self.sleep_s:
            time.sleep(self.sleep_s)


class WasteResonanceZoom:
    def __init__(self, vna_controller: SMPDVNA, fitres_adapter: FitResAdapter) -> None:
        self.vna_controller = vna_controller
        self.fitres = fitres_adapter

    def measure_and_fit(
        self,
        config: VNASweepConfig,
        initial_guess_ghz: Optional[float] = None,
        save_path: Optional[str | os.PathLike[str]] = None,
        live_plot: bool = True,
    ) -> tuple[SweepResult, ResonanceFit]:
        sweep = self.vna_controller.measure_sweep(config, live_plot=live_plot)
        fit = self.fitres.fit_result(sweep, initial_guess_ghz=initial_guess_ghz)
        if save_path is not None:
            sweep.save_h5(save_path)
        return sweep, fit

    def plot_fit(self, sweep: SweepResult, fit: ResonanceFit) -> None:
        import matplotlib.pyplot as plt

        s21 = sweep.s21(corrected_phase=True)
        fit_curve = self.fitres.resfunc3(sweep.freq_ghz, fit.params)
        fit_at_fr = self.fitres.resfunc3(fit.fr_ghz, fit.params)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(sweep.freq_ghz, sweep.mag_db, ".", ms=3)
        ax[0].axvline(fit.fr_ghz, color="k", lw=1)
        ax[0].set_xlabel("Frequency (GHz)")
        ax[0].set_ylabel("Magnitude (dB)")
        ax[0].set_title(f"fr = {fit.fr_ghz:.7f} GHz")

        ax[1].plot(s21.real, s21.imag, ".", ms=3, label="data")
        ax[1].plot(fit_curve.real, fit_curve.imag, "-", label="fit")
        ax[1].plot(np.real(fit_at_fr), np.imag(fit_at_fr), "kx", label="fr")
        ax[1].set_aspect("equal", adjustable="datalim")
        ax[1].set_xlabel("I")
        ax[1].set_ylabel("Q")
        ax[1].grid(True)
        ax[1].legend()
        plt.tight_layout()


class WasteBaseline:
    def __init__(self, vna_controller: SMPDVNA) -> None:
        self.vna_controller = vna_controller

    def measure(
        self,
        config: VNASweepConfig,
        save_path: str | os.PathLike[str],
        image_path: Optional[str | os.PathLike[str]] = None,
        live_plot: bool = False,
    ) -> SweepResult:
        sweep = self.vna_controller.measure_sweep(config, live_plot=live_plot)
        sweep.save_h5(save_path)
        if image_path is not None:
            plot_mag_phase(sweep, title=f"readout power {config.rf_power_dbm:g} dBm", save_path=image_path)
        return sweep


class RFSwitchChannelSurvey:
    def __init__(self, vna_controller: SMPDVNA, switch_controller: Any) -> None:
        self.vna_controller = vna_controller
        self.switch_controller = switch_controller

    def measure_channels(
        self,
        channel_map: dict[int, str],
        config: VNASweepConfig,
        save_dir: str | os.PathLike[str],
        file_prefix: str = "BFG_",
        switch_names: Sequence[str] = ("A", "B"),
        settle_s: float = 1.0,
        live_plot: bool = True,
        connect: bool = True,
        disconnect: bool = True,
    ) -> dict[str, SweepResult]:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, SweepResult] = {}

        if connect and hasattr(self.switch_controller, "connect"):
            self.switch_controller.connect()
        try:
            for switch_index, channel_label in channel_map.items():
                for switch_name in switch_names:
                    self.switch_controller.switch(switch_name, switch_index)
                if settle_s:
                    time.sleep(settle_s)

                sweep = self.vna_controller.measure_sweep(config, live_plot=live_plot)
                sweep.metadata.update(
                    {
                        "switch_index": switch_index,
                        "channel_label": channel_label,
                        "switch_names": tuple(switch_names),
                    }
                )
                sweep.save_h5(save_dir / f"{file_prefix}{channel_label}.h5")
                results[channel_label] = sweep
        finally:
            if disconnect and hasattr(self.switch_controller, "disconnect"):
                self.switch_controller.disconnect()

        return results


class PowerScanAcquisition:
    def __init__(self, vna_controller: SMPDVNA) -> None:
        self.vna_controller = vna_controller

    def acquire(
        self,
        powers_dbm: Sequence[float],
        config: VNASweepConfig,
        save_path: Optional[str | os.PathLike[str]] = None,
        live_plot: bool = True,
    ) -> PowerScanResult:
        powers = np.asarray(powers_dbm, dtype=float)
        sweeps: list[SweepResult] = []

        for power in powers:
            power_config = VNASweepConfig(
                start_ghz=config.start_ghz,
                stop_ghz=config.stop_ghz,
                step_ghz=config.step_ghz,
                points_per_segment=config.points_per_segment,
                measurement=config.measurement,
                rf_power_dbm=float(power),
                if_bw_hz=config.if_bw_hz,
                avg_time_s=config.avg_time_s,
                phase_units=config.phase_units,
                phase_fit_edge_fraction=config.phase_fit_edge_fraction,
                extra_settings=dict(config.extra_settings),
            )
            sweep = self.vna_controller.measure_sweep(power_config, live_plot=live_plot)
            sweep.metadata["readout_power_dbm"] = float(power)
            sweeps.append(sweep)

        if not sweeps:
            raise ValueError("No readout powers supplied.")

        n_points = len(sweeps[0].freq_ghz)
        freq_pow_sweep = np.zeros((len(sweeps), n_points))
        mag_pow_sweep = np.zeros((len(sweeps), n_points))
        phase_pow_sweep = np.zeros((len(sweeps), n_points))

        for idx, sweep in enumerate(sweeps):
            if len(sweep.freq_ghz) != n_points:
                raise ValueError("All power-sweep traces must have the same number of frequency points.")
            freq_pow_sweep[idx, :] = sweep.freq_ghz
            mag_pow_sweep[idx, :] = sweep.mag_db
            phase_pow_sweep[idx, :] = sweep.phase_rad

        result = PowerScanResult(
            freq_pow_sweep=freq_pow_sweep,
            mag_pow_sweep=mag_pow_sweep,
            phase_pow_sweep=phase_pow_sweep,
            powers_dbm=powers,
            config=config,
        )
        if save_path is not None:
            result.save_h5(save_path)
        return result


class PowerScanAnalyzer:
    def __init__(
        self,
        result: PowerScanResult,
        fitres_adapter: Optional[FitResAdapter] = None,
        fit_type: Optional[str] = None,
    ) -> None:
        self.result = result
        self.fitres_adapter = fitres_adapter
        if fitres_adapter is not None:
            if fit_type is not None and fit_type != fitres_adapter.fit_type:
                raise ValueError(
                    "PowerScanAnalyzer.fit_type must match fitres_adapter.fit_type. "
                    "Set fit_type only on FitResAdapter."
                )
            self.fit_type = fitres_adapter.fit_type
        else:
            self.fit_type = "reflection" if fit_type is None else fit_type

    def plot_phase_heatmap(
        self,
        use_norm: bool = False,
        use_fixed_limits: bool = False,
        vmin: float = -8,
        vmax: float = 8,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm

        plot_kwargs: dict[str, Any] = {"shading": "auto", "cmap": "twilight_shifted"}
        if use_norm:
            plot_kwargs["norm"] = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        elif use_fixed_limits:
            plot_kwargs["vmin"] = vmin
            plot_kwargs["vmax"] = vmax

        plt.figure(figsize=(12, 6))
        pcm = plt.pcolormesh(
            self.result.freq_axis,
            self.result.powers_dbm,
            self.result.phase_pow_sweep,
            **plot_kwargs,
        )
        cbar = plt.colorbar(pcm, label=r"Phase ($\pi$ rad)")
        tick_vals = np.array([-2*np.pi, -np.pi, -np.pi/2, 0, np.pi/2, np.pi, 2*np.pi])
        tick_lbls = [r"$-2\pi$", r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$", r"$2\pi$"]
        if use_norm or use_fixed_limits:
            keep = (tick_vals >= vmin) & (tick_vals <= vmax)
        else:
            keep = (
                (tick_vals >= np.nanmin(self.result.phase_pow_sweep))
                & (tick_vals <= np.nanmax(self.result.phase_pow_sweep))
            )
        cbar.set_ticks(tick_vals[keep])
        cbar.set_ticklabels(np.array(tick_lbls)[keep])
        plt.xlabel("frequency (GHz)")
        plt.ylabel("Readout Power (dBm)")
        plt.title("Raw Phase vs Frequency for Different Readout Powers")
        plt.tight_layout()
        plt.show()

    def plot_phase_trace(self, target_power_dbm: float) -> None:
        import matplotlib.pyplot as plt

        power_idx = self.nearest_power_index(target_power_dbm)
        selected_power = self.result.powers_dbm[power_idx]
        plt.figure(figsize=(12, 5))
        plt.plot(self.result.freq_pow_sweep[power_idx, :], self.result.phase_pow_sweep[power_idx, :], lw=1.2)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Phase (rad)")
        plt.title(f"Raw Phase vs Frequency at {selected_power:.1f} dBm")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_wrapped_phase_heatmap(self) -> None:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        phase_wrap = -np.mod(-self.result.phase_pow_sweep, 2*np.pi)
        norm = mpl.colors.Normalize(vmin=-2*np.pi, vmax=0)
        plt.figure(figsize=(12, 6))
        pcm = plt.pcolormesh(
            self.result.freq_axis,
            self.result.powers_dbm,
            phase_wrap,
            shading="auto",
            cmap=twilight_shifted_cream_black(),
            norm=norm,
        )
        cbar = plt.colorbar(pcm, label="Phase (wrapped, rad)")
        cbar.set_ticks([0, -np.pi/2, -np.pi, -3*np.pi/2, -2*np.pi])
        cbar.set_ticklabels([r"$0$", r"$-\pi/2$", r"$-\pi$", r"$-3\pi/2$", r"$-2\pi$"])
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Readout Power (dBm)")
        plt.title(r"Phase vs Frequency (cyclic $0$ to $-2\pi$)")
        plt.tight_layout()
        plt.show()

    def plot_magnitude_heatmap(self) -> None:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        pcm = plt.pcolormesh(
            self.result.freq_axis * 1000.0,
            self.result.powers_dbm,
            self.result.mag_pow_sweep,
            shading="auto",
            cmap="magma",
        )
        plt.colorbar(pcm, label="S21 Magnitude (dB)")
        plt.xlabel("frequency (MHz)")
        plt.ylabel("Readout Power (dBm)")
        plt.title("S21 Magnitude vs Frequency for Different Readout Powers")
        plt.tight_layout()
        plt.show()

    def fit_single_power(
        self,
        target_power_dbm: float,
        p0: Sequence[float] = (0, 0, 0),
        plot: bool = True,
    ) -> PowerFitResult:
        power_idx = self.nearest_power_index(target_power_dbm)
        fit = self._fit_power_index(power_idx, p0=p0, range_label=0)
        if plot:
            self.plot_single_fit(fit)
        return fit

    def fit_powers(
        self,
        target_powers_dbm: Sequence[float],
        p0: Sequence[float] = (0, 0, 0),
        require_exact: bool = True,
        atol_db: float = 1e-6,
        plot: bool = False,
    ) -> list[PowerFitResult]:
        requested = np.asarray(target_powers_dbm, dtype=float)
        results: list[PowerFitResult] = []
        for target_power in requested:
            power_idx = self.nearest_power_index(float(target_power))
            measured_power = float(self.result.powers_dbm[power_idx])
            if require_exact and not np.isclose(measured_power, target_power, atol=atol_db, rtol=0):
                raise ValueError(
                    f"Requested {target_power:g} dBm, but the nearest measured power is "
                    f"{measured_power:g} dBm. Change target_powers_dbm or set require_exact=False."
                )
            try:
                fit = self._fit_power_index(power_idx, p0=p0, range_label=0)
            except Exception as exc:
                print(f"  Fit failed at {measured_power:.1f} dBm: {exc}")
                continue
            results.append(fit)
            print(
                f"Fit P={fit.power:g} dBm: fr={fit.fr:.7f} GHz, "
                f"Q={fit.Q:.0f}, Qc={fit.Qc:.0f}"
            )
            if plot:
                self.plot_single_fit(fit)
        print(f"\nSuccessful fits: {len(results)} / {len(requested)}")
        return results

    def fit_power_ranges(
        self,
        power_range_1: tuple[float, float] = (-15, 0),
        power_range_2: tuple[float, float] = (-60, -45),
        p0: Sequence[float] = (0, 0, 0),
    ) -> list[PowerFitResult]:
        powers = self.result.powers_dbm

        def indices_in_range(pmin: float, pmax: float) -> np.ndarray:
            return np.where((powers >= pmin) & (powers <= pmax))[0]

        range1_idx = indices_in_range(*power_range_1)
        range2_idx = indices_in_range(*power_range_2)
        all_idx = np.concatenate([range1_idx, range2_idx])
        print(f"Range 1: {len(range1_idx)} powers in [{power_range_1[0]}, {power_range_1[1]}] dBm")
        print(f"Range 2: {len(range2_idx)} powers in [{power_range_2[0]}, {power_range_2[1]}] dBm")

        results: list[PowerFitResult] = []
        for power_idx in all_idx:
            range_label = 1 if power_idx in range1_idx else 2
            try:
                results.append(self._fit_power_index(int(power_idx), p0=p0, range_label=range_label))
            except Exception as exc:
                print(f"  Fit failed at {powers[power_idx]:.1f} dBm: {exc}")
        print(f"\nSuccessful fits: {len(results)} / {len(all_idx)}")
        return results

    def plot_fr_vs_power(
        self,
        fits: Sequence[PowerFitResult],
        color_quantity: str = "kappa_i",
        power_range_1: tuple[float, float] = (-15, 0),
        power_range_2: tuple[float, float] = (-60, -45),
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        if not fits:
            print("No fit results available.")
            return

        fr_arr = np.array([fit.fr for fit in fits])
        pow_arr = np.array([fit.power for fit in fits])
        color_arr = np.array([getattr(fit, color_quantity) for fit in fits])
        label_arr = np.array([fit.range_label for fit in fits])
        mask_finite = np.isfinite(color_arr)
        norm = Normalize(vmin=np.nanmin(color_arr[mask_finite]), vmax=np.nanmax(color_arr[mask_finite]))

        fig, ax = plt.subplots(figsize=(10, 6))
        for range_label, marker, label in [
            (1, "o", f"Range 1: {power_range_1}"),
            (2, "s", f"Range 2: {power_range_2}"),
        ]:
            mask = (label_arr == range_label) & mask_finite
            ax.scatter(
                fr_arr[mask],
                pow_arr[mask],
                c=color_arr[mask],
                marker=marker,
                s=60,
                edgecolors="k",
                linewidths=0.5,
                norm=norm,
                cmap="viridis",
                label=label,
                picker=True,
            )
        units = "MHz" if color_quantity.startswith("kappa") else ""
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax)
        cbar.set_label(f"{color_quantity} ({units})" if units else color_quantity)
        ax.set_xlabel("Fitted $f_r$ (GHz)")
        ax.set_ylabel("Readout Power (dBm)")
        ax.set_title("Circle-fit $f_r$ vs Power")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_kappa_c_vs_power(self, fits: Sequence[PowerFitResult]) -> None:
        import matplotlib.pyplot as plt

        if not fits:
            print("No fit results available.")
            return
        pow_all = np.array([fit.power for fit in fits])
        kap_all = np.array([fit.kappa_c for fit in fits])
        fr_all = np.array([fit.fr for fit in fits])
        rng_all = np.array([fit.range_label for fit in fits])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        for ax, range_label, title in zip(axes, [1, 2], ["Range 1", "Range 2"]):
            mask = (rng_all == range_label) & np.isfinite(kap_all) & np.isfinite(fr_all)
            if np.any(mask):
                sc = ax.scatter(
                    pow_all[mask],
                    kap_all[mask],
                    c=fr_all[mask],
                    cmap="viridis",
                    s=70,
                    edgecolors="k",
                    linewidths=0.5,
                )
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label("Fitted $f_r$ (GHz)")
            else:
                ax.text(0.5, 0.5, "No valid fits in this range", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel("Readout Power (dBm)")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel(r"$\kappa_c$ (MHz)")
        fig.suptitle(r"$\kappa_c$ vs Power (color = fitted $f_r$)", fontsize=13)
        fig.tight_layout()
        plt.show()

    def plot_single_fit(self, fit: PowerFitResult, detail_window_mhz: Optional[float] = None) -> None:
        import matplotlib.pyplot as plt

        freq = self.result.freq_pow_sweep[fit.power_index, :]
        mag_db = self.result.mag_pow_sweep[fit.power_index, :]
        phase_rad = self.result.phase_pow_sweep[fit.power_index, :]
        if detail_window_mhz is not None:
            mask = np.abs(freq - fit.fr) <= (detail_window_mhz / 1e3)
            freq = freq[mask]
            mag_db = mag_db[mask]
            phase_rad = phase_rad[mask]

        iq_data = 10 ** (mag_db / 20) * np.exp(1j * phase_rad)
        fit_curve = self._resfunc3(freq, fit)

        plt.figure(figsize=(12, 6))
        plt.plot(freq, mag_db)
        plt.xlabel("frequency (GHz)")
        plt.ylabel("Magnitude (dB)")
        plt.title(f"Magnitude vs Frequency for {fit.power:g} dBm Readout Power")

        plt.figure(figsize=(12, 6))
        plt.plot(freq, phase_rad)
        plt.xlabel("frequency (GHz)")
        plt.ylabel("Phase (radians)")
        plt.title(f"Phase vs Frequency for {fit.power:g} dBm Readout Power")

        plt.figure(figsize=(6, 6))
        plt.plot(iq_data.real, iq_data.imag, marker=".", linestyle="None", label="data")
        plt.plot(
            fit_curve.real,
            fit_curve.imag,
            color="red",
            label=(
                r"$f_r$={:.3f} GHz, $\kappa$={:.3f} MHz, "
                r"$\kappa_c$={:.3f} MHz, $\kappa_i$={:.3f} MHz"
            ).format(fit.fr, fit.kappa, fit.kappa_c, fit.kappa_i),
        )
        plt.axis("equal")
        plt.axhline(0, color="k", linewidth=0.8)
        plt.axvline(0, color="k", linewidth=0.8)
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.title(f"Reflection Circle Fit at {fit.power:g} dBm (SNR={fit.SNR:.0f})")
        plt.legend()
        plt.show()

    def nearest_power_index(self, target_power_dbm: float) -> int:
        return int(np.argmin(np.abs(self.result.powers_dbm - target_power_dbm)))

    def _fit_power_index(
        self,
        power_idx: int,
        p0: Sequence[float] = (0, 0, 0),
        range_label: int = 0,
    ) -> PowerFitResult:
        if self.fitres_adapter is None:
            raise ValueError("PowerScanAnalyzer needs a FitResAdapter for fitting.")
        freq = self.result.freq_pow_sweep[power_idx, :]
        mag_db = self.result.mag_pow_sweep[power_idx, :]
        phase_rad = self.result.phase_pow_sweep[power_idx, :]
        iq_data = 10 ** (mag_db / 20) * np.exp(1j * phase_rad)
        fr_guess = freq[np.argmin(mag_db)]
        fr, Qr, Qc_hat_mag, a, phi, tau, Qc, _ = self.fitres_adapter.finefit(
            freq,
            iq_data,
            fr_guess,
            p0=p0,
        )
        qc_complex_inv = (1.0 / Qc_hat_mag) * np.exp(1j * phi)
        qi_inv = (1.0 / Qr) - np.real(qc_complex_inv)
        Qi = 1.0 / qi_inv if qi_inv != 0 else np.nan
        kappa = fr / Qr * 1e3
        kappa_c = fr / Qc * 1e3
        kappa_i = fr / Qi * 1e3 if np.isfinite(Qi) else np.nan
        snr = self._calc_snr(iq_data)
        return PowerFitResult(
            power=float(self.result.powers_dbm[power_idx]),
            fr=float(fr),
            Q=float(Qr),
            Qc=float(Qc),
            Qi=float(Qi),
            Qc_hat_mag=float(Qc_hat_mag),
            a=a,
            phi=float(phi),
            tau=tau,
            kappa=float(kappa),
            kappa_c=float(kappa_c),
            kappa_i=float(kappa_i),
            SNR=float(snr),
            power_index=power_idx,
            range_label=range_label,
        )

    def _calc_snr(self, iq_data: np.ndarray) -> float:
        fitres = self.fitres_adapter.fitres if self.fitres_adapter is not None else None
        if fitres is None or not hasattr(fitres, "circle2") or not hasattr(fitres, "calc_snr"):
            return np.nan
        _, zc, r0 = fitres.circle2(iq_data)
        return float(fitres.calc_snr(iq_data, zc.real, zc.imag, r0))

    def _resfunc3(self, freq: np.ndarray, fit: PowerFitResult) -> np.ndarray:
        if self.fitres_adapter is None:
            raise ValueError("PowerScanAnalyzer needs a FitResAdapter for plotting fits.")
        return self.fitres_adapter.resfunc3(freq, (fit.fr, fit.Q, fit.Qc_hat_mag, fit.a, fit.phi, fit.tau))


class NoiseReadoutStudy:
    """IF-bandwidth and fixed-frequency IQ-noise measurements."""

    MAX_VNA_IF_BW_HZ = 1_000_000.0

    def __init__(
        self,
        vna_controller: SMPDVNA,
        fitres_adapter: Optional[FitResAdapter] = None,
        t1_s: float = 20e-6,
    ) -> None:
        self.vna_controller = vna_controller
        self.fitres_adapter = fitres_adapter
        self.t1_s = float(t1_s)

    def scan_if_bandwidths(
        self,
        readout_powers_dbm: Sequence[float],
        if_bandwidths_hz: Sequence[float],
        base_config: VNASweepConfig,
        points_per_segment: Optional[int] = None,
        avg_time_s: Optional[float] = None,
        live_plot: bool = True,
    ) -> list[dict[str, Any]]:
        if self.fitres_adapter is None:
            raise ValueError("NoiseReadoutStudy needs a FitResAdapter for IF-bandwidth fits.")

        rows: list[dict[str, Any]] = []
        for readout_power in np.asarray(readout_powers_dbm, dtype=float):
            for if_bw in np.asarray(if_bandwidths_hz, dtype=float):
                self._validate_if_bandwidth(if_bw)
                cfg = VNASweepConfig(
                    start_ghz=base_config.start_ghz,
                    stop_ghz=base_config.stop_ghz,
                    step_ghz=base_config.step_ghz,
                    points_per_segment=(
                        base_config.points_per_segment
                        if points_per_segment is None
                        else int(points_per_segment)
                    ),
                    measurement=base_config.measurement,
                    rf_power_dbm=float(readout_power),
                    if_bw_hz=float(if_bw),
                    avg_time_s=base_config.avg_time_s if avg_time_s is None else float(avg_time_s),
                    phase_units=base_config.phase_units,
                    phase_fit_edge_fraction=base_config.phase_fit_edge_fraction,
                    extra_settings=dict(base_config.extra_settings),
                )
                sample_interval_s = 1.0 / float(if_bw)
                sweep = self.vna_controller.measure_sweep(cfg, live_plot=live_plot)
                try:
                    fit = self.fit_sweep(sweep)
                except Exception as exc:
                    row = {
                        "readout_power_dbm": float(readout_power),
                        "if_bw_hz": float(if_bw),
                        "fr_ghz": np.nan,
                        "Q": np.nan,
                        "Qc": np.nan,
                        "linewidth_hz": np.nan,
                        "ringup_s": np.nan,
                        "sample_interval_s": sample_interval_s,
                        "timing_ok": False,
                        "fit": None,
                        "sweep": sweep,
                        "fit_error": repr(exc),
                    }
                    rows.append(row)
                    print(
                        f"P={readout_power:g} dBm, IFBW={if_bw:g} Hz: fit failed; "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue

                timing = self.resonator_timing(fit.fr_ghz, fit.qr)
                row = {
                    "readout_power_dbm": float(readout_power),
                    "if_bw_hz": float(if_bw),
                    "fr_ghz": fit.fr_ghz,
                    "Q": fit.qr,
                    "Qc": fit.qc,
                    "linewidth_hz": timing["linewidth_hz"],
                    "ringup_s": timing["ringup_s"],
                    "sample_interval_s": sample_interval_s,
                    "timing_ok": bool(timing["ringup_s"] < sample_interval_s < self.t1_s),
                    "fit": fit,
                    "sweep": sweep,
                }
                rows.append(row)
                print(
                    f"P={readout_power:g} dBm, IFBW={if_bw:g} Hz: "
                    f"fr={fit.fr_ghz:.7f} GHz, Q={fit.qr:.0f}, "
                    f"Q/w={timing['ringup_s']*1e6:.2f} us, dt~{sample_interval_s*1e6:.2f} us"
                )
        return rows

    def fit_sweep(self, sweep: SweepResult) -> ResonanceFit:
        if self.fitres_adapter is None:
            raise ValueError("NoiseReadoutStudy needs a FitResAdapter for fitting.")
        guess_ghz = float(sweep.freq_ghz[np.argmin(sweep.mag_db)])
        try:
            return self.fitres_adapter.fit_result(sweep, initial_guess_ghz=guess_ghz, corrected_phase=True)
        except Exception:
            return self.fitres_adapter.fit_result(sweep, initial_guess_ghz=guess_ghz, corrected_phase=False)

    def measure_zero_span_timestream(
        self,
        center_ghz: float,
        readout_power_dbm: float,
        if_bw_hz: float,
        n_samples: int,
        measurement: str = "S43",
        phase_units: str = "deg",
        avg_time_s: float = 0.0,
        extra_settings: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._validate_if_bandwidth(if_bw_hz)
        if hasattr(self.vna_controller.vna, "reset"):
            self.vna_controller.vna.reset()
        settings = self.vna_controller.vna.trans_default_settings()
        settings.update(
            {
                "measurement": measurement,
                "start_freq": float(center_ghz) * 1e9,
                "stop_freq": float(center_ghz) * 1e9,
                "span": 0,
                "RFpower": float(readout_power_dbm),
                "ifBW": float(if_bw_hz),
                "avg_time": float(avg_time_s),
                "freq_points": int(n_samples),
            }
        )
        if extra_settings:
            settings.update(extra_settings)

        data = self.vna_controller.vna.trans_meas(settings)
        mag_db = to_float_array(data["mag"])
        phase_cfg = VNASweepConfig(
            start_ghz=float(center_ghz),
            stop_ghz=float(center_ghz),
            step_ghz=1.0,
            points_per_segment=int(n_samples),
            measurement=measurement,
            rf_power_dbm=float(readout_power_dbm),
            if_bw_hz=float(if_bw_hz),
            avg_time_s=float(avg_time_s),
            phase_units=phase_units,
        )
        phase_rad = phase_to_rad(to_float_array(data["phase"]), phase_cfg)
        z = db_phase_to_complex(mag_db, phase_rad)
        time_s = np.arange(len(z), dtype=float) / float(if_bw_hz)
        return time_s, z, mag_db, phase_rad

    def acquire_timestreams(
        self,
        readout_powers_dbm: Sequence[float],
        if_bw_hz: float,
        n_samples: int,
        base_config: VNASweepConfig,
        save_path: str | os.PathLike[str],
        power_fits: Optional[Sequence[PowerFitResult]] = None,
        power_scan: Optional[PowerScanResult] = None,
        avg_time_s: float = 0.0,
    ) -> list[NoiseTimeStreamResult]:
        self._validate_if_bandwidth(if_bw_hz)
        path = ensure_h5_path(save_path)
        results: list[NoiseTimeStreamResult] = []
        sample_interval_s = 1.0 / float(if_bw_hz)

        with h5py.File(path, "w") as handle:
            handle.attrs["smpd_result_type"] = "noise_timestreams"
            handle.attrs["if_bw_hz"] = float(if_bw_hz)
            handle.attrs["sample_interval_s"] = sample_interval_s
            handle.attrs["t1_s"] = self.t1_s
            handle.attrs["avg_time_s"] = float(avg_time_s)
            handle.attrs["base_config_json"] = _to_json_attr(base_config)

            for readout_power in np.asarray(readout_powers_dbm, dtype=float):
                fr_ghz, q_total = self.resonance_for_power(readout_power, power_fits, power_scan)
                ringup_s = (
                    self.resonator_timing(fr_ghz, q_total)["ringup_s"]
                    if np.isfinite(q_total)
                    else np.nan
                )
                time_s, z, mag_db, phase_rad = self.measure_zero_span_timestream(
                    center_ghz=fr_ghz,
                    readout_power_dbm=float(readout_power),
                    if_bw_hz=float(if_bw_hz),
                    n_samples=int(n_samples),
                    measurement=base_config.measurement,
                    phase_units=base_config.phase_units,
                    avg_time_s=avg_time_s,
                    extra_settings=dict(base_config.extra_settings),
                )
                metrics = self.iq_blob_metrics(z)
                result = NoiseTimeStreamResult(
                    readout_power_dbm=float(readout_power),
                    fr_ghz=fr_ghz,
                    q_total=q_total,
                    ringup_s=ringup_s,
                    sample_interval_s=sample_interval_s,
                    **metrics,
                )
                results.append(result)

                attrs = asdict(result)
                attrs["Q"] = result.q_total
                group = handle.create_group(f"power_{readout_power:g}_dBm")
                group.attrs.update(attrs)
                group.create_dataset("time_s", data=time_s)
                group.create_dataset("i", data=np.real(z))
                group.create_dataset("q", data=np.imag(z))
                group.create_dataset("mag_db", data=mag_db)
                group.create_dataset("phase_rad", data=phase_rad)
                print(
                    f"P={readout_power:g} dBm at {fr_ghz:.7f} GHz: "
                    f"blob radius={result.blob_radius:.3e}, area={result.blob_area_1sigma:.3e}"
                )

        print(path)
        return results

    def acquire_timestreams_from_power_scan(
        self,
        power_scan_path: str | os.PathLike[str],
        base_config: VNASweepConfig,
        fitres_adapter: FitResAdapter,
        fit_selection: str,
        if_bw_hz: float,
        n_samples: int,
        save_path: str | os.PathLike[str],
        readout_powers_dbm: Optional[Sequence[float]] = None,
        power_range_1: tuple[float, float] = (-15, 0),
        power_range_2: tuple[float, float] = (-60, -45),
        fit_p0: Sequence[float] = (0, 0, 0),
        require_exact_power: bool = True,
        plot_each_power_fit: bool = False,
        avg_time_s: float = 0.0,
    ) -> NoiseTimeStreamStudyResult:
        power_scan = load_power_scan_result(power_scan_path, config=base_config)
        analyzer = PowerScanAnalyzer(power_scan, fitres_adapter=fitres_adapter)
        power_fits = self._fit_power_scan_selection(
            analyzer=analyzer,
            fit_selection=fit_selection,
            readout_powers_dbm=readout_powers_dbm,
            power_range_1=power_range_1,
            power_range_2=power_range_2,
            fit_p0=fit_p0,
            require_exact_power=require_exact_power,
            plot_each_power_fit=plot_each_power_fit,
        )
        if not power_fits:
            raise RuntimeError("No time-stream power fits succeeded; check the selected powers and fit settings.")

        power_fits = sorted(power_fits, key=lambda fit: fit.power)
        fitted_powers = np.array([fit.power for fit in power_fits], dtype=float)
        print("Time-stream powers from fresh fits:", fitted_powers)

        timestreams = self.acquire_timestreams(
            readout_powers_dbm=fitted_powers,
            if_bw_hz=if_bw_hz,
            n_samples=n_samples,
            base_config=base_config,
            save_path=save_path,
            power_fits=power_fits,
            power_scan=None,
            avg_time_s=avg_time_s,
        )
        return NoiseTimeStreamStudyResult(
            power_scan=power_scan,
            power_fits=power_fits,
            timestreams=timestreams,
        )

    def acquire_fixed_frequency_grid(
        self,
        center_ghz: float | Sequence[float],
        readout_powers_dbm: Sequence[float],
        if_bandwidths_hz: Sequence[float],
        n_samples: int,
        n_repeats: int,
        save_path: str | os.PathLike[str],
        base_config: Optional[VNASweepConfig] = None,
        measurement: str = "S43",
        phase_units: str = "deg",
        avg_time_s: float = 0.0,
        extra_settings: Optional[dict[str, Any]] = None,
    ) -> NoiseTimeStreamGridResult:
        if base_config is not None:
            measurement = base_config.measurement
            phase_units = base_config.phase_units
            if extra_settings is None:
                extra_settings = dict(base_config.extra_settings)
        center_freqs = np.atleast_1d(np.asarray(center_ghz, dtype=float))
        if center_freqs.size < 1:
            raise ValueError("At least one center frequency is required.")
        powers = np.asarray(readout_powers_dbm, dtype=float)
        ifbws = np.asarray(if_bandwidths_hz, dtype=float)
        for if_bw in ifbws:
            self._validate_if_bandwidth(if_bw)
        if n_repeats < 1:
            raise ValueError("n_repeats must be at least 1.")

        path = ensure_h5_path(save_path)
        with h5py.File(path, "w") as handle:
            handle.attrs["smpd_result_type"] = "noise_timestream_grid"
            handle.attrs["center_ghz"] = float(center_freqs[0])
            handle.attrs["n_samples"] = int(n_samples)
            handle.attrs["n_repeats"] = int(n_repeats)
            handle.attrs["avg_time_s"] = float(avg_time_s)
            handle.attrs["measurement"] = measurement
            handle.attrs["phase_units"] = phase_units
            handle.attrs["extra_settings_json"] = _to_json_attr(extra_settings or {})
            handle.create_dataset("center_freqs_ghz", data=center_freqs)
            handle.create_dataset("readout_powers_dbm", data=powers)
            handle.create_dataset("if_bandwidths_hz", data=ifbws)

            for if_bw in ifbws:
                if_group = handle.create_group(self._ifbw_group_name(if_bw))
                if_group.attrs["if_bw_hz"] = float(if_bw)
                if_group.attrs["sample_interval_s"] = 1.0 / float(if_bw)
                for power in powers:
                    power_group = if_group.create_group(self._power_group_name(power))
                    power_group.attrs["readout_power_dbm"] = float(power)
                    power_group.attrs["center_ghz"] = float(center_freqs[0])
                    for freq_ghz in center_freqs:
                        freq_group = power_group.create_group(self._freq_group_name(freq_ghz))
                        freq_group.attrs["center_ghz"] = float(freq_ghz)
                        for repeat_idx in range(int(n_repeats)):
                            time_s, z, mag_db, phase_rad = self.measure_zero_span_timestream(
                                center_ghz=float(freq_ghz),
                                readout_power_dbm=float(power),
                                if_bw_hz=float(if_bw),
                                n_samples=int(n_samples),
                                measurement=measurement,
                                phase_units=phase_units,
                                avg_time_s=avg_time_s,
                                extra_settings=extra_settings,
                            )
                            metrics = self.iq_blob_metrics(z)
                            repeat_group = freq_group.create_group(f"repeat_{repeat_idx:03d}")
                            repeat_group.attrs.update(metrics)
                            repeat_group.attrs["repeat_index"] = repeat_idx
                            repeat_group.attrs["center_ghz"] = float(freq_ghz)
                            repeat_group.create_dataset("time_s", data=time_s)
                            repeat_group.create_dataset("i", data=np.real(z))
                            repeat_group.create_dataset("q", data=np.imag(z))
                            repeat_group.create_dataset("mag_db", data=mag_db)
                            repeat_group.create_dataset("phase_rad", data=phase_rad)
                            print(
                                f"IFBW={if_bw:g} Hz, P={power:g} dBm, f={freq_ghz:.9f} GHz, "
                                f"repeat {repeat_idx + 1}/{n_repeats}: "
                                f"blob radius={metrics['blob_radius']:.3e}"
                            )

        print(path)
        return NoiseTimeStreamGridResult(
            path=path,
            readout_powers_dbm=powers,
            if_bandwidths_hz=ifbws,
            center_ghz=float(center_freqs[0]),
            n_samples=int(n_samples),
            n_repeats=int(n_repeats),
            center_freqs_ghz=center_freqs,
        )

    def _fit_power_scan_selection(
        self,
        analyzer: PowerScanAnalyzer,
        fit_selection: str,
        readout_powers_dbm: Optional[Sequence[float]],
        power_range_1: tuple[float, float],
        power_range_2: tuple[float, float],
        fit_p0: Sequence[float],
        require_exact_power: bool,
        plot_each_power_fit: bool,
    ) -> list[PowerFitResult]:
        if fit_selection == "ranges":
            power_fits = analyzer.fit_power_ranges(
                power_range_1=power_range_1,
                power_range_2=power_range_2,
                p0=fit_p0,
            )
            if plot_each_power_fit:
                for fit in power_fits:
                    analyzer.plot_single_fit(fit)
            return power_fits
        if fit_selection == "powers":
            if readout_powers_dbm is None:
                raise ValueError('readout_powers_dbm is required when fit_selection is "powers".')
            return analyzer.fit_powers(
                target_powers_dbm=readout_powers_dbm,
                p0=fit_p0,
                require_exact=require_exact_power,
                plot=plot_each_power_fit,
            )
        raise ValueError('fit_selection must be "ranges" or "powers".')

    def resonance_for_power(
        self,
        readout_power_dbm: float,
        power_fits: Optional[Sequence[PowerFitResult]] = None,
        power_scan: Optional[PowerScanResult] = None,
    ) -> tuple[float, float]:
        if power_fits is not None and len(power_fits) > 0:
            fit_by_power = {round(float(fit.power), 6): fit for fit in power_fits}
            fit = fit_by_power.get(round(float(readout_power_dbm), 6))
            if fit is not None:
                return float(fit.fr), float(fit.Q)

        if power_scan is None:
            raise ValueError(
                "No fitted resonance was found for this power. "
                "Provide power_fits or a power_scan fallback."
            )

        power_idx = int(np.argmin(np.abs(power_scan.powers_dbm - float(readout_power_dbm))))
        mag_idx = int(np.argmin(power_scan.mag_pow_sweep[power_idx, :]))
        fr_ghz = float(power_scan.freq_pow_sweep[power_idx, mag_idx])
        return fr_ghz, np.nan

    def plot_if_bandwidth_results(
        self,
        rows: Sequence[dict[str, Any]],
        readout_powers_dbm: Optional[Sequence[float]] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        if not rows:
            print("No IF-bandwidth results available.")
            return

        powers = (
            np.asarray(readout_powers_dbm, dtype=float)
            if readout_powers_dbm is not None
            else np.unique([float(row["readout_power_dbm"]) for row in rows])
        )
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
        for readout_power in powers:
            selected = [row for row in rows if row["readout_power_dbm"] == float(readout_power)]
            if not selected:
                continue
            x = np.array([row["if_bw_hz"] for row in selected], dtype=float)
            axes[0, 0].semilogx(x, [row["fr_ghz"] for row in selected], "o-", label=f"{readout_power:g} dBm")
            axes[0, 1].loglog(x, [row["Q"] for row in selected], "o-", label=f"{readout_power:g} dBm")
            axes[1, 0].loglog(
                x,
                np.array([row["ringup_s"] for row in selected], dtype=float) * 1e6,
                "o-",
                label=f"{readout_power:g} dBm",
            )
            axes[1, 1].loglog(
                x,
                np.array([row["sample_interval_s"] for row in selected], dtype=float) * 1e6,
                "o-",
                label=f"{readout_power:g} dBm",
            )

        axes[0, 0].set_ylabel("Fitted fr (GHz)")
        axes[0, 1].set_ylabel("Fitted Q")
        axes[1, 0].set_ylabel("Q/w ring-up (us)")
        axes[1, 1].set_ylabel("sample interval ~1/IFBW (us)")
        axes[1, 0].axhline(self.t1_s * 1e6, color="k", lw=1, ls="--")
        axes[1, 1].axhline(self.t1_s * 1e6, color="k", lw=1, ls="--", label="T1")
        for ax in axes.ravel():
            ax.set_xlabel("IF bandwidth (Hz)")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
        fig.tight_layout()
        plt.show()

    def plot_if_bandwidth_fit_grid(
        self,
        rows: Sequence[dict[str, Any]],
        unit_mode: str = "s21",
        fitres_adapter: Optional[FitResAdapter] = None,
        max_points: Optional[int] = 2500,
    ) -> None:
        import matplotlib.pyplot as plt

        if not rows:
            print("No IF-bandwidth results available.")
            return
        adapter = fitres_adapter or self.fitres_adapter
        if adapter is None:
            raise ValueError("A FitResAdapter is required to plot circle-fit curves.")

        powers = np.unique([float(row["readout_power_dbm"]) for row in rows])
        ifbws = np.unique([float(row["if_bw_hz"]) for row in rows])
        fig, axes = plt.subplots(
            len(ifbws),
            len(powers),
            figsize=(3.0 * len(powers), 2.8 * len(ifbws)),
            squeeze=False,
        )
        for row_idx, if_bw in enumerate(ifbws):
            for col_idx, power in enumerate(powers):
                ax = axes[row_idx, col_idx]
                match = [
                    row for row in rows
                    if np.isclose(float(row["readout_power_dbm"]), power)
                    and np.isclose(float(row["if_bw_hz"]), if_bw)
                ]
                if not match:
                    ax.axis("off")
                    continue
                item = match[0]
                sweep = item.get("sweep")
                fit = item.get("fit")
                if sweep is None or fit is None:
                    ax.text(0.5, 0.5, "missing sweep/fit", ha="center", va="center", transform=ax.transAxes)
                    ax.axis("off")
                    continue

                z_data = sweep.s21(corrected_phase=True)
                z_fit = adapter.resfunc3(sweep.freq_ghz, self._fit_resfunc_params(fit))
                if unit_mode.lower() in {"voltage", "raw_voltage"}:
                    scale = self.readout_voltage_v(power)
                    z_data = z_data * scale
                    z_fit = z_fit * scale
                elif unit_mode.lower() not in {"s21", "raw_s21"}:
                    raise ValueError('IFBW fit grid unit_mode must be "s21" or "voltage".')

                if max_points is not None and len(z_data) > max_points:
                    stride = int(np.ceil(len(z_data) / max_points))
                    z_data = z_data[::stride]
                    z_fit = z_fit[::stride]

                ax.plot(np.real(z_data), np.imag(z_data), ".", ms=1.5, alpha=0.35)
                ax.plot(np.real(z_fit), np.imag(z_fit), "-", lw=1.0, color="k")
                ax.set_aspect("equal", adjustable="datalim")
                ax.grid(True, alpha=0.2)
                if row_idx == 0:
                    ax.set_title(f"{power:g} dBm")
                if col_idx == 0:
                    ax.set_ylabel(f"{if_bw/1e3:g} kHz")
                if row_idx == len(ifbws) - 1:
                    ax.set_xlabel(self._unit_axis_label(unit_mode))
        fig.suptitle(f"IFBW circle fits: {unit_mode}", y=1.01)
        fig.tight_layout()
        plt.show()

    def plot_iq_grid(
        self,
        grid: NoiseTimeStreamGridResult | str | os.PathLike[str],
        unit_mode: str = "s21",
        power_fits: Optional[Sequence[PowerFitResult]] = None,
        reference_fit: Optional[Any] = None,
        fitres_adapter: Optional[FitResAdapter] = None,
        max_points: Optional[int] = 2500,
    ) -> None:
        import matplotlib.pyplot as plt

        path, powers, ifbws, center_freqs = self._grid_info(grid)
        rows = list(ifbws)
        cols = list(powers)
        cmap = plt.get_cmap("tab10", max(len(center_freqs), 1))
        fig, axes = plt.subplots(
            len(rows),
            len(cols),
            figsize=(3.0 * len(cols), 2.8 * len(rows)),
            squeeze=False,
            sharex=False,
            sharey=False,
        )
        with h5py.File(path, "r") as handle:
            for row_idx, if_bw in enumerate(rows):
                for col_idx, power in enumerate(cols):
                    ax = axes[row_idx, col_idx]
                    traces = self._load_grid_traces(handle, power, if_bw)
                    fit_for_calibration = self._fit_for_power(
                        power,
                        power_fits=power_fits,
                        reference_fit=reference_fit,
                        required=False,
                    )
                    adapter = fitres_adapter or self.fitres_adapter
                    s21_calibration = self._trace_fit_calibration(
                        traces,
                        fit_for_calibration,
                        adapter,
                        center_freqs[0] if len(center_freqs) else None,
                    )
                    seen_freqs: set[float] = set()
                    for trace, freq_ghz in traces:
                        z = self._transform_trace(
                            trace,
                            unit_mode=unit_mode,
                            readout_power_dbm=power,
                            power_fits=power_fits,
                            reference_fit=reference_fit,
                            fitres_adapter=fitres_adapter,
                            freq_ghz=freq_ghz,
                            s21_calibration=s21_calibration,
                        )
                        if max_points is not None and len(z) > max_points:
                            stride = int(np.ceil(len(z) / max_points))
                            z = z[::stride]
                        color_idx = int(np.argmin(np.abs(center_freqs - freq_ghz))) if len(center_freqs) else 0
                        freq_key = round(float(freq_ghz), 9)
                        label = None
                        if freq_key not in seen_freqs:
                            offset_mhz = (float(freq_ghz) - float(center_freqs[0])) * 1e3
                            label = f"{offset_mhz:+.3f} MHz"
                            if np.isclose(offset_mhz, 0.0, atol=1e-6):
                                label += " (waste)"
                            seen_freqs.add(freq_key)
                        ax.plot(
                            np.real(z),
                            np.imag(z),
                            ".",
                            ms=1.5,
                            alpha=0.28,
                            color=cmap(color_idx),
                            label=label,
                        )

                    fit_curve = self._fit_curve_for_power(
                        power,
                        power_fits=power_fits,
                        reference_fit=reference_fit,
                        fitres_adapter=fitres_adapter,
                        unit_mode=unit_mode,
                        s21_calibration=s21_calibration,
                    )
                    if fit_curve is not None:
                        ax.plot(np.real(fit_curve), np.imag(fit_curve), "-", lw=1.0, color="k", alpha=0.75)

                    ax.set_aspect("equal", adjustable="datalim")
                    ax.grid(True, alpha=0.2)
                    if row_idx == 0:
                        ax.set_title(f"{power:g} dBm")
                    if col_idx == 0:
                        ax.set_ylabel(f"{if_bw/1e3:g} kHz")
                    if row_idx == len(rows) - 1:
                        ax.set_xlabel(self._unit_axis_label(unit_mode))
                    if row_idx == 0 and col_idx == len(cols) - 1 and len(center_freqs) > 1:
                        ax.legend(fontsize=7, loc="best", markerscale=3)
        fig.suptitle(f"IQ blobs: {unit_mode}", y=1.01)
        fig.tight_layout()
        plt.show()

    def plot_mean_iq_grid(
        self,
        grid: NoiseTimeStreamGridResult | str | os.PathLike[str],
        unit_mode: str = "s21",
        power_fits: Optional[Sequence[PowerFitResult]] = None,
        reference_fit: Optional[Any] = None,
        fitres_adapter: Optional[FitResAdapter] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        path, powers, ifbws, center_freqs = self._grid_info(grid)
        rows = list(ifbws)
        cols = list(powers)
        cmap = plt.get_cmap("tab10", max(len(center_freqs), 1))
        fig, axes = plt.subplots(
            len(rows),
            len(cols),
            figsize=(3.0 * len(cols), 2.8 * len(rows)),
            squeeze=False,
            sharex=False,
            sharey=False,
        )
        with h5py.File(path, "r") as handle:
            for row_idx, if_bw in enumerate(rows):
                for col_idx, power in enumerate(cols):
                    ax = axes[row_idx, col_idx]
                    traces = self._load_grid_traces(handle, power, if_bw)
                    fit_for_calibration = self._fit_for_power(
                        power,
                        power_fits=power_fits,
                        reference_fit=reference_fit,
                        required=False,
                    )
                    adapter = fitres_adapter or self.fitres_adapter
                    s21_calibration = self._trace_fit_calibration(
                        traces,
                        fit_for_calibration,
                        adapter,
                        center_freqs[0] if len(center_freqs) else None,
                    )

                    means_by_freq: dict[float, list[complex]] = {}
                    for trace, freq_ghz in traces:
                        z = self._transform_trace(
                            trace,
                            unit_mode=unit_mode,
                            readout_power_dbm=power,
                            power_fits=power_fits,
                            reference_fit=reference_fit,
                            fitres_adapter=fitres_adapter,
                            freq_ghz=freq_ghz,
                            s21_calibration=s21_calibration,
                        )
                        freq_key = round(float(freq_ghz), 9)
                        means_by_freq.setdefault(freq_key, []).append(complex(np.mean(z)))

                    fit_curve = self._fit_curve_for_power(
                        power,
                        power_fits=power_fits,
                        reference_fit=reference_fit,
                        fitres_adapter=fitres_adapter,
                        unit_mode=unit_mode,
                        s21_calibration=s21_calibration,
                    )
                    if fit_curve is not None:
                        ax.plot(np.real(fit_curve), np.imag(fit_curve), "-", lw=1.0, color="k", alpha=0.75)

                    for freq_key, means in means_by_freq.items():
                        freq_ghz = float(freq_key)
                        mean_z = complex(np.mean(means))
                        color_idx = int(np.argmin(np.abs(center_freqs - freq_ghz))) if len(center_freqs) else 0
                        offset_mhz = (freq_ghz - float(center_freqs[0])) * 1e3 if len(center_freqs) else 0.0
                        label = f"{offset_mhz:+.3f} MHz"
                        if np.isclose(offset_mhz, 0.0, atol=1e-6):
                            label += " (waste)"
                        ax.plot(
                            np.real(mean_z),
                            np.imag(mean_z),
                            marker="x",
                            linestyle="None",
                            ms=8,
                            mew=2.0,
                            color=cmap(color_idx),
                            label=label,
                        )

                    ax.set_aspect("equal", adjustable="datalim")
                    ax.grid(True, alpha=0.2)
                    if row_idx == 0:
                        ax.set_title(f"{power:g} dBm")
                    if col_idx == 0:
                        ax.set_ylabel(f"{if_bw/1e3:g} kHz")
                    if row_idx == len(rows) - 1:
                        ax.set_xlabel(self._unit_axis_label(unit_mode))
                    if row_idx == 0 and col_idx == len(cols) - 1:
                        ax.legend(fontsize=7, loc="best", markerscale=1.2)
        fig.suptitle(f"Mean IQ blobs: {unit_mode}", y=1.01)
        fig.tight_layout()
        plt.show()

    def plot_psd_grid(
        self,
        grid: NoiseTimeStreamGridResult | str | os.PathLike[str],
        unit_mode: str = "s21",
        component: str = "both",
        power_fits: Optional[Sequence[PowerFitResult]] = None,
        reference_fit: Optional[Any] = None,
        fitres_adapter: Optional[FitResAdapter] = None,
        remove_mean: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        path, powers, ifbws, center_freqs = self._grid_info(grid)
        rows = list(ifbws)
        cols = list(powers)
        fig, axes = plt.subplots(
            len(rows),
            len(cols),
            figsize=(3.2 * len(cols), 2.8 * len(rows)),
            squeeze=False,
            sharex=True,
            sharey=False,
        )
        with h5py.File(path, "r") as handle:
            for row_idx, if_bw in enumerate(rows):
                for col_idx, power in enumerate(cols):
                    ax = axes[row_idx, col_idx]
                    raw_traces = self._load_grid_traces(handle, power, if_bw)
                    fit_for_calibration = self._fit_for_power(
                        power,
                        power_fits=power_fits,
                        reference_fit=reference_fit,
                        required=False,
                    )
                    s21_calibration = self._trace_fit_calibration(
                        raw_traces,
                        fit_for_calibration,
                        fitres_adapter or self.fitres_adapter,
                        center_freqs[0] if len(center_freqs) else None,
                    )
                    traces = [
                        self._transform_trace(
                            trace,
                            unit_mode=unit_mode,
                            readout_power_dbm=power,
                            power_fits=power_fits,
                            reference_fit=reference_fit,
                            fitres_adapter=fitres_adapter,
                            freq_ghz=freq_ghz,
                            s21_calibration=s21_calibration,
                        )
                        for trace, freq_ghz in raw_traces
                    ]
                    psd_items = self.average_psd(
                        traces,
                        sample_rate_hz=float(if_bw),
                        component=component,
                        remove_mean=remove_mean,
                    )
                    for label, freq_hz, psd in psd_items:
                        if len(freq_hz) > 1:
                            ax.loglog(freq_hz[1:], psd[1:], lw=1.0, label=label)
                    ax.grid(True, which="both", alpha=0.2)
                    if row_idx == 0:
                        ax.set_title(f"{power:g} dBm")
                    if col_idx == 0:
                        ax.set_ylabel(f"{if_bw/1e3:g} kHz\nPSD")
                    if row_idx == len(rows) - 1:
                        ax.set_xlabel("Noise frequency (Hz)")
                    if row_idx == 0 and col_idx == len(cols) - 1:
                        ax.legend(fontsize=8)
        fig.suptitle(f"Averaged PSD: {unit_mode}", y=1.01)
        fig.tight_layout()
        plt.show()

    def average_psd(
        self,
        traces: Sequence[np.ndarray],
        sample_rate_hz: float,
        component: str = "both",
        remove_mean: bool = True,
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        if not traces:
            return []
        arrays = [np.asarray(trace) for trace in traces]
        components = self._components_for_psd(arrays, component)
        averaged: list[tuple[str, np.ndarray, np.ndarray]] = []
        for label, series_list in components:
            freq_ref: Optional[np.ndarray] = None
            psds: list[np.ndarray] = []
            for series in series_list:
                freq_hz, psd = self._single_sided_psd(series, sample_rate_hz, remove_mean=remove_mean)
                if freq_ref is None:
                    freq_ref = freq_hz
                psds.append(psd)
            if freq_ref is not None and psds:
                averaged.append((label, freq_ref, np.nanmean(np.vstack(psds), axis=0)))
        return averaged

    def plot_timestream_summary(self, results: Sequence[NoiseTimeStreamResult]) -> None:
        import matplotlib.pyplot as plt

        if not results:
            print("No time-stream results available.")
            return

        powers = np.array([result.readout_power_dbm for result in results], dtype=float)
        blob_radius = np.array([result.blob_radius for result in results], dtype=float)
        blob_area = np.array([result.blob_area_1sigma for result in results], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(powers, blob_radius, "o-")
        axes[0].set_xlabel("Readout power (dBm)")
        axes[0].set_ylabel("IQ blob radius")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(powers, blob_area, "o-")
        axes[1].set_xlabel("Readout power (dBm)")
        axes[1].set_ylabel("IQ blob area (1 sigma)")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show()

    def plot_iq_clouds(
        self,
        timestream_path: str | os.PathLike[str],
        readout_powers_dbm: Sequence[float],
    ) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        with h5py.File(timestream_path, "r") as handle:
            for readout_power in np.asarray(readout_powers_dbm, dtype=float):
                group = handle[f"power_{readout_power:g}_dBm"]
                ax.plot(group["i"][:], group["q"][:], ".", ms=2, alpha=0.35, label=f"{readout_power:g} dBm")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        plt.show()

    def _grid_info(
        self,
        grid: NoiseTimeStreamGridResult | str | os.PathLike[str],
    ) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
        path = grid.path if isinstance(grid, NoiseTimeStreamGridResult) else Path(grid)
        with h5py.File(path, "r") as handle:
            powers = np.asarray(handle["readout_powers_dbm"], dtype=float)
            ifbws = np.asarray(handle["if_bandwidths_hz"], dtype=float)
            if "center_freqs_ghz" in handle:
                center_freqs = np.asarray(handle["center_freqs_ghz"], dtype=float)
            else:
                center_freqs = np.array([float(handle.attrs["center_ghz"])], dtype=float)
        return path, powers, ifbws, center_freqs

    def _load_grid_traces(self, handle: h5py.File, power_dbm: float, if_bw_hz: float) -> list[tuple[np.ndarray, float]]:
        power_group = handle[self._ifbw_group_name(if_bw_hz)][self._power_group_name(power_dbm)]
        traces: list[tuple[np.ndarray, float]] = []
        if any(name.startswith("repeat_") for name in power_group.keys()):
            freq_ghz = float(power_group.attrs["center_ghz"])
            for repeat_name in sorted(power_group.keys()):
                if not repeat_name.startswith("repeat_"):
                    continue
                repeat_group = power_group[repeat_name]
                trace = np.asarray(repeat_group["i"], dtype=float) + 1j * np.asarray(repeat_group["q"], dtype=float)
                traces.append((trace, freq_ghz))
            return traces

        freq_items = sorted(
            (
                (float(power_group[name].attrs["center_ghz"]), name)
                for name in power_group.keys()
                if name.startswith("freq_")
            ),
            key=lambda item: self._freq_order_key(handle, item[0]),
        )
        for freq_ghz, freq_name in freq_items:
            freq_group = power_group[freq_name]
            for repeat_name in sorted(freq_group.keys()):
                repeat_group = freq_group[repeat_name]
                trace = np.asarray(repeat_group["i"], dtype=float) + 1j * np.asarray(repeat_group["q"], dtype=float)
                traces.append((trace, freq_ghz))
        return traces

    @staticmethod
    def _freq_order_key(handle: h5py.File, freq_ghz: float) -> tuple[int, float]:
        if "center_freqs_ghz" not in handle:
            return (0, float(freq_ghz))
        center_freqs = np.asarray(handle["center_freqs_ghz"], dtype=float)
        if center_freqs.size == 0:
            return (0, float(freq_ghz))
        idx = int(np.argmin(np.abs(center_freqs - float(freq_ghz))))
        return (idx, abs(float(center_freqs[idx]) - float(freq_ghz)))

    def _trace_fit_calibration(
        self,
        traces: Sequence[tuple[np.ndarray, float]],
        fit: Optional[Any],
        fitres_adapter: Optional[FitResAdapter],
        reference_freq_ghz: Optional[float],
    ) -> Optional[complex]:
        if fit is None or fitres_adapter is None or not traces:
            return None
        if reference_freq_ghz is None:
            selected_freq = float(traces[0][1])
        else:
            selected_freq = min(traces, key=lambda item: abs(float(item[1]) - float(reference_freq_ghz)))[1]
        selected = [
            np.asarray(trace, dtype=complex)
            for trace, freq_ghz in traces
            if np.isclose(float(freq_ghz), float(selected_freq), atol=1e-9, rtol=0)
        ]
        if not selected:
            return None
        trace_mean = complex(np.mean(np.concatenate(selected)))
        fit_value = complex(fitres_adapter.resfunc3(float(selected_freq), self._fit_resfunc_params(fit)))
        if abs(fit_value) == 0:
            return None
        return trace_mean / fit_value

    def _transform_trace(
        self,
        z: np.ndarray,
        unit_mode: str,
        readout_power_dbm: float,
        power_fits: Optional[Sequence[PowerFitResult]] = None,
        reference_fit: Optional[Any] = None,
        fitres_adapter: Optional[FitResAdapter] = None,
        freq_ghz: Optional[float] = None,
        s21_calibration: Optional[complex] = None,
    ) -> np.ndarray:
        unit = unit_mode.lower()
        if unit in {"s21", "raw_s21"}:
            return np.asarray(z, dtype=complex)
        if unit in {"voltage", "raw_voltage"}:
            return np.asarray(z, dtype=complex) * self.readout_voltage_v(readout_power_dbm)
        if unit in {"resonator_s21", "resonator_basis_s21"}:
            fit = self._fit_for_power(readout_power_dbm, power_fits, reference_fit=reference_fit)
            z_arr = np.asarray(z, dtype=complex)
            if s21_calibration is not None and abs(s21_calibration) > 0:
                z_arr = z_arr / complex(s21_calibration)
            return self.resonator_basis(
                z_arr,
                fit,
                fitres_adapter or self.fitres_adapter,
                freq_ghz=freq_ghz,
            )
        if unit in {"resonator_voltage", "resonator_basis_voltage"}:
            fit = self._fit_for_power(readout_power_dbm, power_fits, reference_fit=reference_fit)
            z_arr = np.asarray(z, dtype=complex)
            if s21_calibration is not None and abs(s21_calibration) > 0:
                z_arr = z_arr / complex(s21_calibration)
            basis = self.resonator_basis(
                z_arr,
                fit,
                fitres_adapter or self.fitres_adapter,
                freq_ghz=freq_ghz,
            )
            return basis * self.readout_voltage_v(readout_power_dbm)
        raise ValueError(
            "unit_mode must be one of: s21, voltage, resonator_s21, resonator_voltage."
        )

    def _fit_curve_for_power(
        self,
        readout_power_dbm: float,
        power_fits: Optional[Sequence[Any]],
        reference_fit: Optional[Any],
        fitres_adapter: Optional[FitResAdapter],
        unit_mode: str,
        n_points: int = 1001,
        s21_calibration: Optional[complex] = None,
    ) -> Optional[np.ndarray]:
        fit = self._fit_for_power(
            readout_power_dbm,
            power_fits,
            reference_fit=reference_fit,
            required=False,
        )
        adapter = fitres_adapter or self.fitres_adapter
        if fit is None or adapter is None:
            return None
        fr_ghz = self._fit_fr_ghz(fit)
        q_total = self._fit_q_total(fit)
        span_ghz = max(5.0 * fr_ghz / q_total, 1e-6)
        freq_ghz = np.linspace(fr_ghz - span_ghz, fr_ghz + span_ghz, n_points)
        curve = adapter.resfunc3(freq_ghz, self._fit_resfunc_params(fit))
        unit = unit_mode.lower()
        if unit in {"voltage", "raw_voltage"}:
            if s21_calibration is not None and abs(s21_calibration) > 0:
                curve = curve * complex(s21_calibration)
            curve = curve * self.readout_voltage_v(readout_power_dbm)
        elif unit in {"s21", "raw_s21"}:
            if s21_calibration is not None and abs(s21_calibration) > 0:
                curve = curve * complex(s21_calibration)
        elif unit in {"resonator_s21", "resonator_basis_s21"}:
            curve = self.resonator_basis(
                curve,
                fit,
                adapter,
                freq_ghz=freq_ghz,
            )
        elif unit in {"resonator_voltage", "resonator_basis_voltage"}:
            curve = self.resonator_basis(
                curve,
                fit,
                adapter,
                freq_ghz=freq_ghz,
            ) * self.readout_voltage_v(readout_power_dbm)
        return curve

    def _fit_for_power(
        self,
        readout_power_dbm: float,
        power_fits: Optional[Sequence[Any]],
        reference_fit: Optional[Any] = None,
        required: bool = True,
    ) -> Optional[Any]:
        if power_fits:
            for item in power_fits:
                if isinstance(item, dict):
                    item_power = item.get("readout_power_dbm")
                    item_fit = item.get("fit")
                else:
                    item_power = getattr(item, "power", None)
                    item_fit = item
                if item_power is not None and np.isclose(float(item_power), float(readout_power_dbm), atol=1e-6, rtol=0):
                    return item_fit
        if reference_fit is not None:
            return reference_fit
        if required:
            raise ValueError(
                f"No power fit available for {readout_power_dbm:g} dBm. "
                "Use unit_mode='s21' or 'voltage' for raw time-stream plots, "
                "pass reference_fit=waste_fit, or run the IF-bandwidth fit-check with this readout power before "
                "using 'resonator_s21' or 'resonator_voltage'."
            )
        return None

    def resonator_basis(
        self,
        z: np.ndarray,
        fit: Any,
        fitres_adapter: Optional[FitResAdapter],
        freq_ghz: Optional[Sequence[float] | float] = None,
    ) -> np.ndarray:
        if fitres_adapter is None:
            raise ValueError("A FitResAdapter is required for resonator-basis conversion.")
        fr_ghz = self._fit_fr_ghz(fit)
        params = self._fit_resfunc_params(fit)
        _, _, _, a, _, tau = params[:6]

        if freq_ghz is None:
            freq = fr_ghz
        else:
            freq = np.asarray(freq_ghz, dtype=float)

        background = a * np.exp(-2j * np.pi * (freq - fr_ghz) * tau)
        if np.any(np.asarray(background) == 0):
            raise ValueError("Cannot normalize resonator basis with zero background.")

        # Normalized resonator response: off resonance is near +1. In transmission this
        # is the S21 notch response; in reflection this is the S11 response.
        return np.asarray(z, dtype=complex) / background

    @staticmethod
    def resonator_circle_blob_frequencies(
        center_ghz: float,
        fit: Any,
        n_blobs: int,
        epsilon_rad: float = 0.01,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_blobs = int(n_blobs)
        if n_blobs < 1:
            raise ValueError("n_blobs must be at least 1.")
        center = float(center_ghz)
        if n_blobs == 1:
            return np.array([center], dtype=float), np.array([0.0], dtype=float)
        if fit is None:
            raise ValueError("A resonator fit is needed to convert evenly spaced circle angles to frequencies.")

        q_total = NoiseReadoutStudy._fit_q_total(fit)
        angles = 2 * np.pi * np.arange(n_blobs, dtype=float) / n_blobs
        singular = np.isclose(np.mod(angles, 2 * np.pi), np.pi, atol=1e-12)
        angles[singular] = np.pi - float(epsilon_rad)

        u_targets = 0.5 + 0.5 * np.exp(1j * angles)
        detuning_ghz = 0.5 * np.imag(1.0 / u_targets) * center / q_total
        return center + detuning_ghz, detuning_ghz * 1e3

    @staticmethod
    def fits_from_ifbw_results(
        rows: Sequence[dict[str, Any]],
        target_if_bw_hz: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        if not rows:
            return []
        selected_rows: list[dict[str, Any]] = []
        powers = np.unique([float(row["readout_power_dbm"]) for row in rows])
        for power in powers:
            power_rows = [row for row in rows if np.isclose(float(row["readout_power_dbm"]), power)]
            if target_if_bw_hz is None:
                row = power_rows[0]
            else:
                row = min(power_rows, key=lambda r: abs(float(r["if_bw_hz"]) - float(target_if_bw_hz)))
            if "fit" in row:
                selected_rows.append(row)
        return selected_rows

    @staticmethod
    def _fit_fr_ghz(fit: Any) -> float:
        if isinstance(fit, PowerFitResult):
            return float(fit.fr)
        if isinstance(fit, ResonanceFit):
            return float(fit.fr_ghz)
        if hasattr(fit, "fr"):
            return float(fit.fr)
        if hasattr(fit, "fr_ghz"):
            return float(fit.fr_ghz)
        raise TypeError(f"Unsupported fit object: {type(fit)!r}")

    @staticmethod
    def _fit_q_total(fit: Any) -> float:
        if isinstance(fit, PowerFitResult):
            return float(fit.Q)
        if isinstance(fit, ResonanceFit):
            return float(fit.qr)
        if hasattr(fit, "Q"):
            return float(fit.Q)
        if hasattr(fit, "qr"):
            return float(fit.qr)
        raise TypeError(f"Unsupported fit object: {type(fit)!r}")

    @staticmethod
    def _fit_resfunc_params(fit: Any) -> tuple[Any, ...]:
        if isinstance(fit, PowerFitResult):
            return (fit.fr, fit.Q, fit.Qc_hat_mag, fit.a, fit.phi, fit.tau)
        if isinstance(fit, ResonanceFit):
            return tuple(fit.params[:6])
        if hasattr(fit, "params"):
            return tuple(fit.params[:6])
        if all(hasattr(fit, name) for name in ("fr", "Q", "Qc_hat_mag", "a", "phi", "tau")):
            return (fit.fr, fit.Q, fit.Qc_hat_mag, fit.a, fit.phi, fit.tau)
        raise TypeError(f"Unsupported fit object: {type(fit)!r}")

    def _components_for_psd(
        self,
        traces: Sequence[np.ndarray],
        component: str,
    ) -> list[tuple[str, list[np.ndarray]]]:
        comp = component.lower()
        if comp == "both":
            return [
                ("x/I", [np.real(trace) for trace in traces]),
                ("y/Q", [np.imag(trace) for trace in traces]),
            ]
        if comp in {"i", "x", "real"}:
            return [("x/I", [np.real(trace) for trace in traces])]
        if comp in {"q", "y", "imag"}:
            return [("y/Q", [np.imag(trace) for trace in traces])]
        if comp in {"magnitude", "abs", "amplitude"}:
            return [("|z|", [np.abs(trace) for trace in traces])]
        if comp in {"phase", "angle"}:
            return [("angle", [np.unwrap(np.angle(trace)) for trace in traces])]
        raise ValueError("component must be both, i, q, magnitude, or phase.")

    @staticmethod
    def _single_sided_psd(
        series: Sequence[float],
        sample_rate_hz: float,
        remove_mean: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(series, dtype=float)
        if x.size < 2:
            raise ValueError("At least two samples are required for a PSD.")
        if remove_mean:
            x = x - np.nanmean(x)
        window = np.hanning(x.size)
        window_power = np.sum(window**2)
        fft = np.fft.rfft(x * window)
        freq_hz = np.fft.rfftfreq(x.size, d=1.0 / float(sample_rate_hz))
        psd = (np.abs(fft) ** 2) / (float(sample_rate_hz) * window_power)
        if x.size > 2:
            psd[1:-1] *= 2.0
        return freq_hz, psd

    @staticmethod
    def readout_voltage_v(readout_power_dbm: float, impedance_ohm: float = 50.0) -> float:
        power_w = 10.0 ** ((float(readout_power_dbm) - 30.0) / 10.0)
        return float(np.sqrt(power_w * float(impedance_ohm)))

    @staticmethod
    def _unit_axis_label(unit_mode: str) -> str:
        unit = unit_mode.lower()
        if "voltage" in unit:
            return "Real / Imag (V)"
        return "Real / Imag"

    @classmethod
    def _validate_if_bandwidth(cls, if_bw_hz: float) -> None:
        if float(if_bw_hz) <= 0:
            raise ValueError("IF bandwidth must be positive.")
        if float(if_bw_hz) > cls.MAX_VNA_IF_BW_HZ:
            raise ValueError(
                f"IF bandwidth {if_bw_hz:g} Hz exceeds the VNA max of "
                f"{cls.MAX_VNA_IF_BW_HZ:g} Hz."
            )

    @staticmethod
    def _power_group_name(power_dbm: float) -> str:
        return f"power_{float(power_dbm):g}_dBm"

    @staticmethod
    def _ifbw_group_name(if_bw_hz: float) -> str:
        return f"ifbw_{float(if_bw_hz):g}_Hz"

    @staticmethod
    def _freq_group_name(freq_ghz: float) -> str:
        return f"freq_{float(freq_ghz):.9f}_GHz"

    @staticmethod
    def resonator_timing(fr_ghz: float, q_total: float) -> dict[str, float]:
        fr_hz = float(fr_ghz) * 1e9
        q_value = float(q_total)
        linewidth_hz = fr_hz / q_value
        ringup_s = q_value / (2 * np.pi * fr_hz)
        return {"linewidth_hz": linewidth_hz, "ringup_s": ringup_s}

    @staticmethod
    def iq_blob_metrics(z: Sequence[complex]) -> dict[str, float]:
        z_arr = np.asarray(z, dtype=complex)
        if z_arr.size < 2:
            raise ValueError("At least two IQ samples are needed to estimate blob metrics.")
        iq = np.column_stack((np.real(z_arr), np.imag(z_arr)))
        centered = iq - np.mean(iq, axis=0)
        cov = np.cov(centered, rowvar=False)
        eig = np.linalg.eigvalsh(cov)
        eig = np.clip(eig, 0, None)
        return {
            "sigma_i": float(np.sqrt(cov[0, 0])),
            "sigma_q": float(np.sqrt(cov[1, 1])),
            "blob_radius": float(np.sqrt(np.sum(eig))),
            "blob_area_1sigma": float(np.pi * np.sqrt(np.prod(eig))),
        }


class PumpBiasScan:
    def __init__(
        self,
        vna_controller: SMPDVNA,
        pump_source: Any,
        bias_source: Any,
        bias_channel: int,
    ) -> None:
        self.vna_controller = vna_controller
        self.pump_source = pump_source
        self.bias_source = bias_source
        self.bias_channel = bias_channel

    def scan(
        self,
        pump_powers_dbm: Sequence[float],
        dc_biases_mv: Sequence[float],
        sweep_config: VNASweepConfig,
        waste_resonance_ghz: float,
        baseline: SweepResult | str | os.PathLike[str],
        output_path: Optional[str | os.PathLike[str]] = None,
        window_hz: float = 0.5e6,
        live_heatmap: bool = True,
        save_legacy_gain_only: bool = False,
    ) -> PumpBiasScanResult:
        baseline_sweep = load_sweep_result(baseline, config=sweep_config)
        pump_powers = np.asarray(pump_powers_dbm, dtype=float)
        dc_biases = np.asarray(dc_biases_mv, dtype=float)

        n_pump = len(pump_powers)
        n_bias = len(dc_biases)
        gain_data: Optional[np.ndarray] = None
        result: Optional[PumpBiasScanResult] = None

        try:
            for i, pump_power in enumerate(pump_powers):
                self._set_pump_power(pump_power)
                self._set_pump_enabled(True)

                for j, dc_bias in enumerate(dc_biases):
                    self._set_dc_bias(dc_bias)
                    sweep = self.vna_controller.measure_sweep(
                        sweep_config,
                        live_plot=False,
                        progress=True,
                    )
                    gain_db = gain_from_baseline(sweep, baseline_sweep)
                    if gain_data is None:
                        gain_data = np.full((n_pump, n_bias, len(sweep.freq_ghz)), np.nan)
                    gain_data[i, j, :] = gain_db

                    result = PumpBiasScanResult(
                        freq_ghz=sweep.freq_ghz,
                        gain_db=gain_data,
                        pump_powers_dbm=pump_powers,
                        dc_biases_mv=dc_biases,
                        waste_resonance_ghz=waste_resonance_ghz,
                        baseline_path=str(baseline) if not isinstance(baseline, SweepResult) else None,
                        metadata={"window_hz": window_hz, "sweep_config": sweep_config},
                    )
                    if live_heatmap:
                        plot_gain_heatmap(result, window_hz=window_hz)

            if result is None:
                raise RuntimeError("Pump-bias scan did not acquire any points.")
            if output_path is not None:
                result.save_h5(output_path)
            return result
        finally:
            self._set_pump_enabled(False)

    def _set_pump_power(self, power_dbm: float) -> None:
        self.pump_source.set_power(float(power_dbm))

    def _set_pump_enabled(self, enabled: bool) -> None:
        self.pump_source.set_output_state(enable=bool(enabled))

    def _set_dc_bias(self, voltage_mv: float) -> None:
        self.bias_source.set_voltage_mV(self.bias_channel, voltage_mv=float(voltage_mv))
        self.bias_source.enable_output(self.bias_channel)


def load_sweep_result(
    source: SweepResult | str | os.PathLike[str],
    config: Optional[VNASweepConfig] = None,
) -> SweepResult:
    if isinstance(source, SweepResult):
        return source
    path = Path(source)
    if _is_h5_path(path):
        return _load_sweep_h5(path, config=config)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, SweepResult):
        return payload
    if isinstance(payload, tuple) and len(payload) >= 3:
        freq, mag, phase = payload[:3]
        phase_rad = detrended_phase(freq, phase, config=config)
        return SweepResult(
            freq_ghz=np.asarray(freq, dtype=float),
            mag_db=np.asarray(mag, dtype=float),
            phase_raw=np.asarray(phase, dtype=float),
            phase_rad=phase_rad,
            config=config,
        )
    raise TypeError(f"Unsupported sweep payload: {type(payload)!r}")


def load_pump_bias_scan_result(path: str | os.PathLike[str]) -> PumpBiasScanResult:
    path = Path(path)
    if _is_h5_path(path):
        return _load_pump_bias_scan_h5(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, PumpBiasScanResult):
        return payload
    raise TypeError(
        "This file does not contain a PumpBiasScanResult. "
        "Legacy gain-only arrays need pump powers, DC biases, and frequency data supplied separately."
    )


def load_power_scan_result(
    path: str | os.PathLike[str],
    config: Optional[VNASweepConfig] = None,
) -> PowerScanResult:
    path = Path(path)
    if _is_h5_path(path):
        return _load_power_scan_h5(path, config=config)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, PowerScanResult):
        return payload
    if isinstance(payload, tuple) and len(payload) >= 4:
        freq_pow_sweep, mag_pow_sweep, phase_pow_sweep, powers_dbm = payload[:4]
        return PowerScanResult(
            freq_pow_sweep=np.asarray(freq_pow_sweep, dtype=float),
            mag_pow_sweep=np.asarray(mag_pow_sweep, dtype=float),
            phase_pow_sweep=np.asarray(phase_pow_sweep, dtype=float),
            powers_dbm=np.asarray(powers_dbm, dtype=float),
            config=config,
        )
    raise TypeError(f"Unsupported power-scan payload: {type(payload)!r}")


def build_sweep_result(
    freq_parts: Sequence[np.ndarray],
    mag_parts: Sequence[np.ndarray],
    phase_parts: Sequence[np.ndarray],
    config: VNASweepConfig,
) -> SweepResult:
    freq = np.concatenate(freq_parts) if freq_parts else np.array([], dtype=float)
    mag = np.concatenate(mag_parts) if mag_parts else np.array([], dtype=float)
    phase = np.concatenate(phase_parts) if phase_parts else np.array([], dtype=float)
    return SweepResult(
        freq_ghz=freq,
        mag_db=mag,
        phase_raw=phase,
        phase_rad=detrended_phase(freq, phase, config=config),
        config=config,
    )


def db_phase_to_complex(mag_db: Sequence[float], phase_rad: Sequence[float]) -> np.ndarray:
    return 10.0 ** (np.asarray(mag_db, dtype=float) / 20.0) * np.exp(
        1j * np.asarray(phase_rad, dtype=float)
    )


def phase_to_rad(phase: Sequence[float], config: Optional[VNASweepConfig] = None) -> np.ndarray:
    phase = np.asarray(phase, dtype=float)
    units = "deg" if config is None else config.phase_units.lower()
    if units in {"deg", "degree", "degrees"}:
        return np.deg2rad(phase)
    if units in {"rad", "radian", "radians"}:
        return phase
    raise ValueError(f"Unsupported phase_units: {units!r}")


def detrended_phase(
    freq_ghz: Sequence[float],
    phase: Sequence[float],
    config: Optional[VNASweepConfig] = None,
) -> np.ndarray:
    freq = np.asarray(freq_ghz, dtype=float)
    phase_rad = np.unwrap(phase_to_rad(phase, config))
    if len(freq) < 3:
        return phase_rad

    edge_fraction = 0.1 if config is None else config.phase_fit_edge_fraction
    n_edge = max(2, min(len(freq), int(round(len(freq) * edge_fraction))))
    fit = np.polyfit(freq[:n_edge], phase_rad[:n_edge], 1)
    return phase_rad - np.polyval(fit, freq)


def estimate_resonance_from_phase(freq_ghz: Sequence[float], phase_rad: Sequence[float]) -> float:
    freq = np.asarray(freq_ghz, dtype=float)
    phase = np.asarray(phase_rad, dtype=float)
    if len(freq) < 3:
        raise ValueError("Need at least three frequency points to estimate a resonance.")
    slope = np.gradient(phase, freq)
    return float(freq[int(np.argmax(np.abs(slope)))])


def gain_from_baseline(sweep: SweepResult, baseline: SweepResult) -> np.ndarray:
    s21_on = sweep.s21(corrected_phase=False)
    baseline_s21 = baseline.s21(corrected_phase=False)
    baseline_interp = complex_interp(sweep.freq_ghz, baseline.freq_ghz, baseline_s21)
    return 20.0 * np.log10(np.abs(s21_on / baseline_interp))


def complex_interp(x: Sequence[float], xp: Sequence[float], fp: Sequence[complex]) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=complex)
    real = np.interp(x, xp, fp.real)
    imag = np.interp(x, xp, fp.imag)
    return real + 1j * imag


def plot_mag_phase(
    sweep: SweepResult,
    title: str = "",
    save_path: Optional[str | os.PathLike[str]] = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(sweep.freq_ghz, sweep.mag_db, "b-")
    ax1.set_xlabel("Frequency (GHz)")
    ax1.set_ylabel("Magnitude (dB)", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(sweep.freq_ghz, sweep.phase_rad, "r-")
    ax2.set_ylabel("Phase (rad)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax1.set_title(title)
    ax1.grid(True)
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")


def plot_gain_heatmap(result: PumpBiasScanResult, window_hz: float = 0.5e6) -> None:
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

    clear_output(wait=True)
    gain_at_res = result.gain_at_resonance(window_hz=window_hz)
    plt.figure(figsize=(6, 4))
    plt.pcolormesh(result.dc_biases_mv, result.pump_powers_dbm, gain_at_res, shading="auto")
    plt.xlabel("DC bias (mV)")
    plt.ylabel("Pump power (dBm)")
    plt.title(f"Gain at {result.waste_resonance_ghz:.6f} GHz")
    plt.colorbar(label="Gain (dB)")
    plt.tight_layout()
    plt.show()


def twilight_shifted_cream_black(n: int = 256):
    import matplotlib as mpl
    from matplotlib.colors import ListedColormap

    base = mpl.colormaps["twilight_shifted"](np.linspace(0, 1, n))
    x = np.linspace(0, 1, n)
    edge = np.clip((np.abs(x - 0.5) - 0.35) / 0.15, 0, 1)
    colors = base.copy()
    colors[:, :3] = (1 - edge)[:, None] * colors[:, :3]

    cream = np.array([1.00, 0.97, 0.90])
    white = np.array([1.00, 1.00, 1.00])
    t_white = np.clip(1 - np.abs(x - 0.5) / 0.06, 0, 1)
    center_color = (1 - t_white)[:, None] * cream + t_white[:, None] * white
    center_blend = np.exp(-((x - 0.5) / 0.10) ** 2)
    colors[:, :3] = (1 - center_blend)[:, None] * colors[:, :3] + center_blend[:, None] * center_color
    return ListedColormap(colors, name="twilight_shifted_cream_black")


def to_float_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        try:
            return value.astype(float)
        except ValueError:
            if value.size == 1:
                return to_float_array(value.reshape(-1)[0])
            if value.dtype.kind in {"O", "S", "U"}:
                return np.asarray([to_float_array(item).reshape(-1) for item in value], dtype=float).ravel()
            raise
    if isinstance(value, str):
        text = value.strip()
        try:
            return np.asarray(ast.literal_eval(text), dtype=float)
        except (SyntaxError, ValueError):
            pass
        try:
            return np.asarray(eval(text, {"__builtins__": {}}, {"array": np.array, "np": np}), dtype=float)
        except Exception:
            pass
        stripped = text.strip("[]")
        if "," in stripped:
            parsed = np.fromstring(stripped, sep=",")
            if parsed.size:
                return parsed.astype(float)
        parsed = np.fromstring(stripped, sep=" ")
        if parsed.size:
            return parsed.astype(float)
        raise ValueError(f"Could not parse numeric array from string: {value[:80]!r}")
    return np.asarray(value, dtype=float)


def scalar_frequency(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError("Initial resonance guess must be a scalar or a one-element sequence.")
    return float(arr.reshape(-1)[0])


def _load_sweep_h5(path: Path, config: Optional[VNASweepConfig] = None) -> SweepResult:
    with h5py.File(path, "r") as handle:
        freq = np.asarray(handle["freq_ghz"], dtype=float)
        mag = np.asarray(handle["mag_db"], dtype=float)
        phase_raw = np.asarray(handle["phase_raw"], dtype=float)
        loaded_config = config or _config_from_json_attr(handle.attrs.get("config_json"))
        if "phase_rad" in handle:
            phase_rad = np.asarray(handle["phase_rad"], dtype=float)
        else:
            phase_rad = detrended_phase(freq, phase_raw, config=loaded_config)
        metadata = _from_json_attr(handle.attrs.get("metadata_json"), default={})
    return SweepResult(
        freq_ghz=freq,
        mag_db=mag,
        phase_raw=phase_raw,
        phase_rad=phase_rad,
        config=loaded_config,
        metadata=metadata,
    )


def _load_pump_bias_scan_h5(path: Path) -> PumpBiasScanResult:
    with h5py.File(path, "r") as handle:
        baseline_path = str(handle.attrs.get("baseline_path", ""))
        if not baseline_path:
            baseline_path = None
        return PumpBiasScanResult(
            freq_ghz=np.asarray(handle["freq_ghz"], dtype=float),
            gain_db=np.asarray(handle["gain_db"], dtype=float),
            pump_powers_dbm=np.asarray(handle["pump_powers_dbm"], dtype=float),
            dc_biases_mv=np.asarray(handle["dc_biases_mv"], dtype=float),
            waste_resonance_ghz=float(handle.attrs["waste_resonance_ghz"]),
            baseline_path=baseline_path,
            metadata=_from_json_attr(handle.attrs.get("metadata_json"), default={}),
        )


def _load_power_scan_h5(
    path: Path,
    config: Optional[VNASweepConfig] = None,
) -> PowerScanResult:
    with h5py.File(path, "r") as handle:
        loaded_config = config or _config_from_json_attr(handle.attrs.get("config_json"))
        return PowerScanResult(
            freq_pow_sweep=np.asarray(handle["freq_pow_sweep"], dtype=float),
            mag_pow_sweep=np.asarray(handle["mag_pow_sweep"], dtype=float),
            phase_pow_sweep=np.asarray(handle["phase_pow_sweep"], dtype=float),
            powers_dbm=np.asarray(handle["powers_dbm"], dtype=float),
            config=loaded_config,
            metadata=_from_json_attr(handle.attrs.get("metadata_json"), default={}),
        )


def _config_from_json_attr(value: Any) -> Optional[VNASweepConfig]:
    data = _from_json_attr(value)
    if not data:
        return None
    return VNASweepConfig(**data)


def _to_json_attr(value: Any) -> str:
    return json.dumps(_json_safe(value))


def _from_json_attr(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _is_h5_path(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in {".h5", ".hdf5"}


def ensure_h5_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        path = path.with_suffix(".h5")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_pickle_path(path: str | os.PathLike[str]) -> Path:
    path = Path(path)
    if path.suffix != ".pkl":
        path = path.with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
