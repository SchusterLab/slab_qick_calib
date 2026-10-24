"""Shared setup and persistence helpers for the numbered SMPD scripts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import _pylab_helpers

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from qick import QickConfig

from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
from slab_qick_calib.experiments.jpa.jpa_tone import turn_off_jpa, turn_on_jpa
from slab_qick_calib.experiments.general.qick_experiment import (
    effective_final_delay,  # noqa: F401  (re-exported for the scripts)
)
from slab_qick_calib.experiments.general.qick_program import reset_mode_of
from slab_qick_calib.experiments.single_qubit.qubit_spec_shot import (
    band_frequency_from_fit,
    band_gain_from_fit,
)
from slab_qick_calib.helpers import config, rfboard


def _fmt_band_coeffs(coeffs, fit_type):
    """`a=..., c=...` for a stored band fit, naming each coefficient."""
    names = {
        "linear_quadratic": ["a", "b", "c"],
        "cubic": ["a", "b", "c", "d"],
        "quartic": ["a", "b", "c", "d", "e"],
    }.get(fit_type, ["a", "c"])
    return ", ".join(f"{n}={v:.4f}" for n, v in zip(names, coeffs))


_CONFIG_NAME = re.compile(r"_bfg(?P<run>[^_]+)_.*_(?P<device>[^_]+)_rfboard$")


def run_and_device_from_config(config_name):
    """
    Read the run number and device name out of the config filename.

    Configs are named ``YYYY_MM_DD_bfg<run>_..._<device>_rfboard.yml``, so
    ``..._bfg33_smpd_v2_yuvi_rfboard.yml`` gives ``("33", "Yuvi")`` and the data
    lands in ``Data/run33_Yuvi``. The device is chosen by the CONFIG constant in
    the script being run; this module deliberately holds no default config, so
    switching device never means editing the shared harness.

    Raises rather than falling back, since a misnamed config that quietly wrote
    into another device's run folder would be worse than a stopped script.
    """
    match = _CONFIG_NAME.search(Path(config_name).stem)
    if match is None:
        raise ValueError(
            f"Cannot read a run number and device name from config "
            f"{config_name!r}. Expected YYYY_MM_DD_bfg<run>_..._<device>_"
            "rfboard.yml, or pass --run-number and --device-name explicitly."
        )
    return match.group("run"), match.group("device").capitalize()


def add_common_arguments(parser, config):
    parser.add_argument("--config", default=config)
    parser.add_argument("--run-number", default=None)
    parser.add_argument("--device-name", default=None)
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--no-update-config", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def parse_args(description, add_arguments=None, *, config):
    """
    Parse the shared flags, with the device taken from `config`.

    `config` is the calling script's own CONFIG constant and has no default, so
    a script that forgets to pass it fails immediately instead of running
    against whichever device a fallback happened to name. Nothing has to be
    given on the command line; `--config` is only an override.
    """
    parser = argparse.ArgumentParser(description=description)
    add_common_arguments(parser, config)
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args()

    run_number, device_name = run_and_device_from_config(args.config)
    if args.run_number is None:
        args.run_number = run_number
    if args.device_name is None:
        args.device_name = device_name
    return args


def prepare_run(args, script_path, qi, experiment_name=None, parent=None):
    """
    `parent` names an existing experiment folder to nest this one inside, under
    both data/ and metadata_and_images/, numbered from 1 within it. For looping
    scripts, so a night of measurements is one entry in each rather than hundreds.
    """
    fridge_run_dir = SCRIPT_DIR / "Data" / f"run{args.run_number}_{args.device_name}"
    fridge_run_dir.mkdir(parents=True, exist_ok=True)
    data_root = fridge_run_dir / "data"
    metadata_root = fridge_run_dir / "metadata_and_images"
    if parent is not None:
        data_root = data_root / parent
        metadata_root = metadata_root / parent
    experiment_number = next_experiment_number(data_root)
    if experiment_name is None:
        experiment_name = experiment_name_for_script(script_path)
    timestamp = datetime.now().strftime("%H%M%S")
    experiment_folder = f"#{experiment_number}_{experiment_name}_q{qi}_{timestamp}"
    data_dir = data_root / experiment_folder
    metadata_dir = metadata_root / experiment_folder
    data_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)
    shutil.copy2(script_path, metadata_dir / Path(script_path).name)
    for name in analysis_scripts_for_script(script_path):
        source = ANALYSIS_DIR / name
        if source.exists():
            shutil.copy2(source, metadata_dir / name)
        else:
            print(f"[warn] analysis script {source} is missing, not copied")

    cfg_path = REPO_DIR / "configs" / args.config
    auto_cfg = config.load(str(cfg_path))
    im = InstrumentManager(ns_address=auto_cfg["aliases"]["ip"], port=args.port)
    soc = im[auto_cfg["aliases"]["soc"]]
    soccfg = QickConfig(soc.get_cfg())
    cfg_dict = {
        "soc": soccfg,
        "expt_path": str(fridge_run_dir),
        "data_path": str(data_dir),
        "metadata_path": str(metadata_dir),
        "cfg_file": str(cfg_path),
        "im": im,
    }
    return data_dir, cfg_path, auto_cfg, soc, cfg_dict


def next_experiment_number(data_dir):
    numbers = []
    if not data_dir.exists():
        return 1
    for path in data_dir.iterdir():
        if path.is_dir() and path.name.startswith("#"):
            try:
                numbers.append(int(path.name.split("_", 1)[0][1:]))
            except ValueError:
                pass
    return max(numbers, default=0) + 1


ANALYSIS_DIR = SCRIPT_DIR / "analysis"


def analysis_scripts_for_script(script_path):
    """
    Offline analysis scripts to drop into a run's metadata folder.

    These are re-analyses you run *after* the fact, on the h5 files the run
    just wrote. Copying them in at acquisition time rather than leaving them in
    `analysis/` means each run carries the version that matches it, and the
    copy sits next to its own data -- both scripts resolve `DATA_DIR` as
    `HERE.parents[1] / "data" / HERE.name`, so from the metadata folder they
    need no arguments at all.
    """
    names = {
        "17_stark_spectroscopy.py": ("refit_two_peak.py",),
        "23_fwm_efficiency_vs_power.py": ("fit_shared_means.py",),
        "26_fwm_efficiency_vs_power_checked.py": ("fit_shared_means.py",),
        "37_fwm_efficiency_vs_buffer_frequency.py": ("fit_shared_means.py",),
        "40_fwm_efficiency_vs_buffer_length.py": ("fit_shared_means.py",),
    }
    return names.get(Path(script_path).name, ())


def experiment_name_for_script(script_path):
    names = {
        "00_soc_map.py": "soc_map",
        "01_tof.py": "time_of_flight",
        "02_resonator_spectroscopy.py": "resonator_spectroscopy",
        "02b_resonator_power.py": "resonator_spectroscopy_vs_amplitude",
        "03_qubit_spectroscopy.py": "qubit_spectroscopy",
        "03b_qubit_power.py": "qubit_spectroscopy_vs_amplitude",
        "04_rabi.py": "power_rabi",
        "04b_length_rabi.py": "time_rabi",
        "05_rabi_chevron.py": "rabi_chevron_amplitude",
        "10_t1.py": "T1",
        "11_t2.py": "T2_ramsey",
        "11b_t2_echo.py": "T2_echo",
        "12_single_shot.py": "IQ_blobs",
        "12b_active_reset.py": "active_reset",
        "42_reset_while_loop.py": "reset_while_loop",
        "12c_ground_vs_readout_power.py": "ground_vs_readout_power",
        "12d_readout_length_constant_snr.py": "readout_length_constant_snr",
        "12e_readout_nep_optimization.py": "readout_nep_optimization",
        "13_ef_spectroscopy.py": "f_state_spec",
        "13b_ef_rabi.py": "ef_power_rabi",
        "13c_ef_ramsey.py": "T2_ramsey_ef",
        "14_jpa_fine_tuning.py": "JPA_fine_tuning",
        "14a_jpa_bias_scan.py": "JPA_bias_scan",
        "15_jpa_on.py": "JPA_on",
        "15b_jpa_off.py": "JPA_off",
        "16_temperature.py": "temperature",
        "17_stark_spectroscopy.py": "stark_spectroscopy",
        "18_fwm_power.py": "FWM_power",
        "19_fwm_buffer.py": "FWM_buffer",
        "20_fwm_buffer_gain_length.py": "FWM_buffer_gain_length",
        "21_fwm_histogram.py": "FWM_histogram",
        "22_readout_optimization.py": "readout_optimization",
        "23_fwm_efficiency_vs_power.py": "FWM_efficiency_vs_power",
        "24_fwm_histogram_vs_power.py": "FWM_histogram_vs_power",
        "25_chi.py": "chi",
        "26_fwm_efficiency_vs_power_checked.py": "FWM_efficiency_vs_power_checked",
        "27_f0g1_spectroscopy.py": "f0g1_spectroscopy",
        "28_f0g1_stark_shift.py": "f0g1_stark_shift",
        "29_ef_rabi_flat_top.py": "ef_time_rabi_flat_top",
        "30_ef_rabi_vs_power.py": "ef_rabi_vs_power",
        "31_ef_spec_under_f0g1.py": "ef_spec_under_f0g1",
        "32_ef_stark_shift_vs_f0g1_power.py": "ef_stark_shift_vs_f0g1_power",
        "33_f0g1_rabi.py": "f0g1_rabi",
        "34_f0g1_rabi_vs_power.py": "f0g1_rabi_vs_power",
        "35_reset_time.py": "f0g1_reset_time",
        "35b_reset_floor.py": "f0g1_reset_floor",
        "36_reset_rate_vs_ef_gain.py": "reset_rate_vs_ef_gain",
        "37_reset_rate_map.py": "reset_rate_map",
        "37_fwm_efficiency_vs_buffer_frequency.py": "FWM_efficiency_vs_buffer_frequency",
        "38_t2_photon_calibration.py": "T2_photon_calibration",
        "39_t1_pump.py": "T1_pump",
        "40_fwm_efficiency_vs_buffer_length.py": "FWM_efficiency_vs_buffer_length",
        "41_buffer_dephasing_spectroscopy.py": "buffer_dephasing_spectroscopy",
        "17b_stark_probe_saturation.py": "stark_probe_saturation",
        "17c_stark_shift_offresonant.py": "stark_shift_offresonant",
        "42_buffer_ringdown.py": "buffer_ringdown",
    }
    return names.get(Path(script_path).name, Path(script_path).stem)


def fwm_gain_for_frequency(auto_cfg, pump_frequency, verbose=True):
    """
    Invert the FWM band fit to get the pump gain at a given pump frequency.

    `18_fwm_power.py` fits the bright FWM band and stores the coefficients in
    `device.fwm.pump_power_fit` alongside the model name in
    `device.fwm.pump_power_fit_type`:

        quadratic        [a, c]           freq = a*gain**2 + c
        linear           [a, c]           freq = a*gain + c
        linear_quadratic [a, b, c]        freq = a*gain**2 + b*gain + c
        cubic            [a, b, c, d]     freq = a*gain**3 + b*gain**2 + c*gain + d
        quartic          [a, b, c, d, e]  freq = a*gain**4 + b*gain**3 + c*gain**2 + d*gain + e

    The inversion itself lives in `band_gain_from_fit` so there is one copy of
    the root-finding branch selection rather than one per caller. It raises
    for a frequency off the fitted band.

    The returned gain is printed but never clamped -- if it lands above
    `device.qubit.max_gain` the fit is probably stale, and that is worth
    seeing rather than silently working around.
    """
    coeffs = list(auto_cfg.device.fwm.pump_power_fit)
    fit_type = auto_cfg.device.fwm.pump_power_fit_type
    gain = band_gain_from_fit(coeffs, fit_type, pump_frequency)

    if verbose:
        max_gain = auto_cfg.device.qubit.max_gain
        note = "" if gain <= max_gain else f"  [above max_gain={max_gain}]"
        print(
            f"FWM {fit_type} fit {_fmt_band_coeffs(coeffs, fit_type)} -> pump "
            f"gain {gain:.4f} at {pump_frequency:.3f} MHz{note}"
        )
    return gain


def fwm_frequency_for_gain(auto_cfg, pump_gain, verbose=True):
    """
    Evaluate the FWM band fit to get the pump frequency at a given gain.

    The forward direction of `fwm_gain_for_frequency`; see there for the
    coefficient layout of each model. Unlike the inverse there is nothing to
    go wrong here -- every gain maps to a frequency -- so this only prints
    what it found.
    """
    coeffs = list(auto_cfg.device.fwm.pump_power_fit)
    fit_type = auto_cfg.device.fwm.pump_power_fit_type
    frequency = band_frequency_from_fit(coeffs, fit_type, pump_gain)

    if verbose:
        print(
            f"FWM {fit_type} fit {_fmt_band_coeffs(coeffs, fit_type)} -> pump "
            f"{frequency:.3f} MHz at gain {pump_gain:.4f}"
        )
    return frequency


def fwm_band_point(auto_cfg, pump_frequency=None, pump_gain=None, verbose=True):
    """
    Resolve a pump setting from whichever of frequency / gain is given.

    Pass one and leave the other None to have it read off the band fit in
    `device.fwm`; pass both to go deliberately off the band (the distance from
    it is reported so that is visible rather than silent).

    Args:
        auto_cfg: Loaded configuration
        pump_frequency: Pump frequency in MHz, or None to derive it
        pump_gain: Pump gain, or None to derive it
        verbose: Print what was derived

    Returns:
        (pump_frequency, pump_gain)

    Raises:
        ValueError: if both are None -- there is nothing to anchor the point to.
    """
    if pump_frequency is None and pump_gain is None:
        raise ValueError(
            "Give a pump frequency or a pump gain (or both); with neither "
            "there is nothing to place on the FWM band."
        )

    if pump_gain is None:
        pump_gain = fwm_gain_for_frequency(auto_cfg, pump_frequency, verbose=verbose)
    elif pump_frequency is None:
        pump_frequency = fwm_frequency_for_gain(auto_cfg, pump_gain, verbose=verbose)
    elif verbose:
        # Both specified: say how far off the fitted band this sits.
        on_band = fwm_frequency_for_gain(auto_cfg, pump_gain, verbose=False)
        print(
            f"Pump set explicitly: {pump_frequency:.3f} MHz at gain "
            f"{pump_gain:.4f}. The band fit puts gain {pump_gain:.4f} at "
            f"{on_band:.3f} MHz, i.e. {pump_frequency - on_band:+.3f} MHz off."
        )

    if verbose:
        max_gain = auto_cfg.device.qubit.max_gain
        if pump_gain > 1.0:
            # Not just above the soft limit: 1.0 is full DAC scale, so this
            # cannot be played at all and the fit is almost certainly stale.
            print(
                f"  [gain {pump_gain:.4f} exceeds full DAC scale (1.0) -- "
                "unplayable; re-run 18_fwm_power.py to refresh the band fit]"
            )
        elif pump_gain > max_gain:
            print(f"  [gain {pump_gain:.4f} is above max_gain={max_gain}]")
    return pump_frequency, pump_gain


def activate(qi, soc, auto_cfg, cfg_path):
    auto_cfg = rfboard.activate_qubit_rf(
        qi, soc, auto_cfg, cfg_file=str(cfg_path), verbose=False
    )
    return auto_cfg


def enable_jpa(qi, soc, soccfg, auto_cfg):
    jpa = auto_cfg.hw.soc.dacs.jpa
    return turn_on_jpa(
        soc,
        soccfg,
        auto_cfg,
        idx=qi,
        pump_gain=jpa.pump_gain[qi],
        dc_bias_v=jpa.jpa_dc_val[qi],
        pump_ch=jpa.ch[qi],
        pump_freq_mhz=jpa.pump_freq[qi],
        mixer_freq_mhz=jpa.mixing_freq[qi],
        pump_nqz=jpa.nyquist[qi],
        attn1_db=jpa.attn1[qi],
        attn2_db=jpa.attn2[qi],
        verbose=False,
    )


@contextmanager
def nonblocking_plots():
    """
    Make plt.show() return immediately for the duration of the block.

    Experiment display() methods call plt.show() themselves, which blocks on an
    interactive backend and would stall a loop at its first figure. Patched
    here rather than at import so wait_for_plots() keeps its blocking show.
    """
    original_show = plt.show

    def nonblocking_show(*show_args, **show_kwargs):
        show_kwargs["block"] = False
        original_show(*show_args, **show_kwargs)

    plt.show = nonblocking_show
    try:
        yield
    finally:
        plt.show = original_show


def run_experiment(experiment, args):
    if args.no_show:
        experiment.run(display=False, save=True, progress=not args.no_progress)
        return experiment

    with nonblocking_plots():
        experiment.run(display=True, save=True, progress=not args.no_progress)
    _raise_windows()
    return experiment


def _figure_canvases():
    """Live canvases, without going through plt.figure() (which re-raises)."""
    canvases = []
    for num in plt.get_fignums():
        manager = _pylab_helpers.Gcf.get_fig_manager(num)
        if manager is not None:
            canvases.append(manager.canvas)
    return canvases


def _flush_events():
    """
    Pump pending GUI events once. Returns True if there was a window to pump.

    Used instead of plt.pause(), which calls manager.show() on every tick and
    so re-raises every figure -- that is what makes the plots flash.
    """
    pumped = False
    for canvas in _figure_canvases():
        if getattr(canvas.manager, "window", None) is None:
            continue
        try:
            canvas.flush_events()
            pumped = True
        except (AttributeError, RuntimeError, NotImplementedError):
            pass
    return pumped


def _raise_windows():
    """Bring figure windows to the front without stealing keyboard focus."""
    for canvas in _figure_canvases():
        window = getattr(canvas.manager, "window", None)
        if window is None:
            continue
        try:
            window.deiconify()
            window.lift()
        except (AttributeError, RuntimeError):
            try:
                window.show()
                window.raise_()
            except (AttributeError, RuntimeError):
                pass


def show_all_plots(block=True):
    """
    Bring all open figures to the front, optionally waiting for them to close.

    Deliberately does not call plt.pause() or focus_force(): both cause the
    windows to flash, the former by re-showing every figure on each tick and
    the latter by grabbing keyboard focus away from the terminal.
    """
    _raise_windows()
    try:
        if block:
            wait_for_plots(raise_windows=False)
        else:
            plt.show(block=False)
            _flush_events()
    except (AttributeError, RuntimeError):
        pass


def wait_for_plots(raise_windows=True):
    """
    Keep script-created plot windows open until the user closes them.

    Blocks in the GUI's own event loop with a single plt.show(block=True)
    rather than polling, so the figures are not repeatedly re-shown.

    Args:
        raise_windows: Bring the windows forward first. Skipped when the
            caller has just done it, to avoid a second raise.
    """
    if not plt.get_fignums():
        return
    if raise_windows:
        _raise_windows()

    try:
        plt.show(block=True)
    except (AttributeError, RuntimeError):
        pass

    # A non-interactive backend returns from show() immediately. Only fall
    # back to polling while real windows remain, otherwise this would spin
    # forever under Agg.
    while plt.get_fignums():
        if not _flush_events():
            break
        time.sleep(0.1)


def prepare_experiment(args, script_path, qi, jpa_on=False, additional_readout_qis=(),
                       experiment_name=None, parent=None):
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_run(
        args, script_path, qi, experiment_name=experiment_name, parent=parent
    )
    auto_cfg = activate(qi, soc, auto_cfg, cfg_path)
    for additional_qi in additional_readout_qis:
        if additional_qi != qi:
            auto_cfg = rfboard.activate_qubit_rf(
                additional_qi,
                soc,
                auto_cfg,
                cfg_file=cfg_path,
                channels=("dac_readout",),
                verbose=False,
            )
    if jpa_on:
        enable_jpa(qi, soc, cfg_dict["soc"], auto_cfg)
    elif auto_cfg.hw.soc.dacs.jpa.pump_gain[qi] != 0:  # pump_gain 0 = qubit has no JPA
        # The pump tone and DC bias persist on the board, so off must be explicit.
        turn_off_jpa(soc, cfg_dict["soc"], auto_cfg, idx=qi, verbose=False)
    return run_dir, cfg_path, auto_cfg, soc, cfg_dict


def suppress_duplicate_config(experiment, metadata_path):
    """
    Make this experiment skip its config yml if the folder already has one.

    The looping scripts build one experiment per point, and save_config writes
    <prefix>_000NN.yml every time -- 421 near-identical 6.6 kB files for a full
    script-26 grid. Consecutive copies differ by the one swept line, and that
    value is already in each point's h5 (h5.load_config(), which is what
    fit_shared_means.py and fwm_qc.load_saved_run actually read) and in
    qc_log.csv. So one copy is kept as the run's device record and the rest are
    skipped.

    Patches the instance, not the class, so it covers both ways a script
    persists the config: an explicit save_config() call and the one inside
    display(). Call it right after constructing the experiment. Single-point
    scripts are unaffected -- their folder is empty, so the first write lands.
    """
    if any(Path(metadata_path).glob("*.yml")):
        experiment.save_config = lambda *args, **kwargs: None
    return experiment


def finish_experiment(experiment, args, wait=True):
    run_experiment(experiment, args)
    if wait and not args.no_show:
        # run_experiment already brought the windows forward.
        wait_for_plots(raise_windows=False)
    return experiment