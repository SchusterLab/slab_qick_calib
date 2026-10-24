"""
Repeat buffer spectroscopy, single shot (g/e/f), T1, T1 with the FWM pump on (on the band, then at
the same gain off it), T2 Ramsey, T2 echo and temperature until a stop time,
re-optimizing the JPA once an hour.

The overnight version of running 02, 12, 10, 39 (twice), 11, 11b and 16 by hand,
with 14 (quickverify mode) interleaved. Each measurement gets its own numbered folder,
named as its numbered script would name it, nested inside this script's own
#<n>_coherence_monitor_... folder (under both data/ and metadata_and_images/)
so a night of them is one entry in each. Each writes back to the config
the way that script does unless --no-update-config -- so f_ge follows the
Ramsey and the threshold follows the single shot through the night.

The JPA is re-optimized at the start and then at the first step boundary after
each --jpa-interval minutes. Every other step re-enables the JPA from
hw.soc.dacs.jpa, so with --no-update-config the re-optimization is only a
measurement: the operating point stays the config's.

The stop time is checked before every step, so the run ends at the first step
boundary after it. One row per measurement goes to coherence_monitor.csv in
this script's own metadata folder, stamped with the time the measurement
started, and coherence_monitor.png next to it is redrawn from the CSV after
each one: every quantity vs time, open markers where the fit failed (those
don't set the axis limits). A step that raises is printed and skipped, not
fatal; Ctrl-C stops the loop.

    python coherence_monitor.py --no-show
    python coherence_monitor.py --stop-at 07:30 --jpa-interval 30
"""

import csv
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.figure import Figure

import slab_qick_calib.experiments as meas
from slab_qick_calib.calib import measure_func
from slab_qick_calib.helpers import config

from common import fwm_band_point, nonblocking_plots, parse_args, prepare_experiment

CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
QI = 1
buffer_line_qi = 0  # buffer line (readout index 0), for buffer spectroscopy
JPA_ON = True
RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
STOP_AT = "09:00"
JPA_INTERVAL_MIN = 60
# T1 pump, as in 39_t1_pump.py. Give one and leave the other None to read it
# off the band fit; both = off the band.
PUMP_FREQUENCY = 5220
PUMP_GAIN = None
# Off-band control: the same pump gain, moved this far (MHz) off the band.
PUMP_OFF_BAND_DETUNING = 10.0

COLUMNS = ["time", "cycle", "step",
           "jpa_pump_gain", "jpa_dc_bias_mv", "jpa_gain_db",
           "f_buffer", "kappa_buffer",
           "fidelity", "threshold", "T1", "T1_ok", "T1_pump", "T1_pump_ok",
           "frequency_p", "gain_p", "T1_pump_off", "T1_pump_off_ok",
           "frequency_p_off", "T2r", "f_ge", "T2r_ok",
           "T2e", "T2e_ok", "temp_mk", "population"]

# One panel each: (y label, [(column, legend label, fit-ok column or None)]).
PANELS = [
    ("JPA pump gain", [("jpa_pump_gain", None, None)]),
    ("JPA bias (mV)", [("jpa_dc_bias_mv", None, None)]),
    ("JPA gain (dB)", [("jpa_gain_db", None, None)]),
    ("buffer frequency", [("f_buffer", None, None)]),
    ("buffer kappa (MHz)", [("kappa_buffer", None, None)]),
    ("fidelity", [("fidelity", None, None)]),
    ("threshold", [("threshold", None, None)]),
    ("T1 (us)", [("T1", "no pump", "T1_ok"),
                 ("T1_pump", "pump on band", "T1_pump_ok"),
                 ("T1_pump_off", "pump off band", "T1_pump_off_ok")]),
    ("T2 (us)", [("T2r", "Ramsey", "T2r_ok"), ("T2e", "echo", "T2e_ok")]),
    ("f_ge", [("f_ge", None, "T2r_ok")]),
    ("temperature (mK)", [("temp_mk", None, None)]),
    ("excited population", [("population", None, None)]),
]
# Plotted as the shift from the first point in kHz, rather than as 7-digit MHz.
RELATIVE_KHZ = ("f_buffer", "f_ge")


def add_arguments(parser):
    parser.add_argument("--stop-at", default=STOP_AT,
                        help="HH:MM local time; the next time it comes round")
    parser.add_argument("--jpa-interval", type=float, default=JPA_INTERVAL_MIN,
                        help="minutes between JPA re-optimizations")


def next_occurrence(hhmm):
    now = datetime.now()
    hour, minute = (int(part) for part in hhmm.split(":"))
    stop = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if stop <= now:
        stop += timedelta(days=1)
    return stop


def prepare(args, name, jpa_on=JPA_ON, qi=QI):
    # Nested inside this script's own folder (args.parent, set in __main__).
    return prepare_experiment(args, __file__, qi, jpa_on, experiment_name=name,
                              parent=args.parent)


def run(expt, args):
    # display=True even under --no-show (Agg), since display() is what saves
    # the PNGs.
    with nonblocking_plots():
        expt.run(display=True, save=True, progress=not args.no_progress)


def jpa(args):
    _, _, auto_cfg, _, cfg_dict = prepare(args, "JPA_fine_tuning", jpa_on=False)
    jpa_cfg = auto_cfg.hw.soc.dacs.jpa
    # Script 14's quickverify window around the current operating point.
    set_gain = jpa_cfg.pump_gain[QI]
    set_v = jpa_cfg.jpa_dc_val[QI]
    params = {
        "waste_freq_mhz": auto_cfg.device.readout.frequency[QI],
        "center": auto_cfg.device.readout.frequency[QI],
        "span": 20.0,
        "expts": 100,
        "gain": auto_cfg.device.readout.gain[QI],
        "length": 10,
        "reps": 100,
        "rounds": 1,
        "pump_ch": jpa_cfg.ch[QI],
        "pump_nqz": jpa_cfg.nyquist[QI],
        "pump_freq_mhz": jpa_cfg.pump_freq[QI],
        "pump_mixer_freq_mhz": jpa_cfg.mixing_freq[QI],
        "pump_attn1_db": jpa_cfg.attn1[QI],
        "pump_attn2_db": jpa_cfg.attn2[QI],
        "pump_filter_bw_mhz": 1000.0,
        "pump_filter_type": "bandpass",
        "pump_gains": np.linspace(set_gain - 0.1, set_gain + 0.1, 11),
        "dc_biases_v": np.linspace(set_v - 1e-3, set_v + 1e-3, 11),
        "bias_chan": jpa_cfg.jpa_dc_ch[QI],
        "off_bias": 0.0,
        "dc_settle": 0.25,
        "pump_on_after": True,
        "rspec_update": not args.no_update_config,
        "rspec_update_verbose": True,
    }
    expt = meas.JPAFineTuning(cfg_dict, qi=QI, go=False, params=params)
    run(expt, args)
    return {"jpa_pump_gain": expt.data["best_pump_gain"],
            "jpa_dc_bias_mv": expt.data["best_dc_bias_mv"],
            "jpa_gain_db": expt.data["best_gain_db"]}


def buffer_spec(args):
    # As in 02_resonator_spectroscopy.py. The next step re-activates the qubit
    # line's RF settings.
    _, _, _, _, cfg_dict = prepare(args, "resonator_spectroscopy", jpa_on=False,
                                   qi=buffer_line_qi)
    params = {"span": 2, "gain": 0.3, "reps": 1000, "expts": 100,
              "pulse_e": False, "pulse_f": False}
    expt = meas.ResSpec(cfg_dict, qi=buffer_line_qi, go=False, style="fine",
                        params=params)
    run(expt, args)
    if not args.no_update_config:
        expt.update()
    # Absent when the complex fit failed (update() says so).
    return {"f_buffer": expt.data.get("complex_fit", [np.nan])[0],
            "kappa_buffer": expt.data.get("complex_fit_kappa", np.nan)}


def single_shot(args):
    _, _, _, _, cfg_dict = prepare(args, "IQ_blobs")
    params = {"shots": 30000, "reset_mode": RESET_MODE}
    expt = meas.HistogramExperiment(cfg_dict, qi=QI, go=False, check_f=True,
                                    params=params)
    run(expt, args)
    if not args.no_update_config:
        expt.update()
    return {"fidelity": expt.data["fids"][0],
            "threshold": expt.data["thresholds"][0]}


def t1(args):
    _, _, _, _, cfg_dict = prepare(args, "T1")
    params = {"reps": 1000, "pulse_ef": False, "reset_mode": RESET_MODE}
    expt = meas.T1Experiment(cfg_dict, qi=QI, go=False, params=params)
    run(expt, args)
    if not args.no_update_config:
        expt.update()
    return {"T1": expt.data["new_t1_i"], "T1_ok": expt.status}


def t1_pump(args):
    _, _, _, _, cfg_dict = prepare(args, "T1_pump")
    params = {"reps": 1000, "pump": True, "pulse_ef": False,
              "frequency_p": PUMP_FREQUENCY, "gain_p": PUMP_GAIN,
              "reset_mode": RESET_MODE}
    expt = meas.T1Experiment(cfg_dict, qi=QI, go=False, params=params)
    print(f"Pump: {expt.cfg.expt.frequency_p:.3f} MHz, gain {expt.cfg.expt.gain_p:.4f}")
    run(expt, args)
    if not args.no_update_config:
        expt.update()   # -> device.fwm.T1_pump (+ the pump setting); device T1 untouched
    return {"T1_pump": expt.data["new_t1_i"], "T1_pump_ok": expt.status,
            "frequency_p": expt.cfg.expt.frequency_p, "gain_p": expt.cfg.expt.gain_p}


def t1_pump_off_band(args):
    _, _, auto_cfg, _, cfg_dict = prepare(args, "T1_pump_off_band")
    # The on-band step's gain, at a frequency PUMP_OFF_BAND_DETUNING away.
    frequency_p, gain_p = fwm_band_point(auto_cfg, PUMP_FREQUENCY, PUMP_GAIN,
                                         verbose=False)
    params = {"reps": 1000, "pump": True, "pulse_ef": False,
              "frequency_p": frequency_p + PUMP_OFF_BAND_DETUNING,
              "gain_p": gain_p, "reset_mode": RESET_MODE}
    expt = meas.T1Experiment(cfg_dict, qi=QI, go=False, params=params)
    print(f"Pump: {expt.cfg.expt.frequency_p:.3f} MHz, gain {expt.cfg.expt.gain_p:.4f} "
          f"({PUMP_OFF_BAND_DETUNING:+g} MHz off the band)")
    run(expt, args)
    # No update(): it writes device.fwm.T1_pump, which is the on-band value.
    return {"T1_pump_off": expt.data["new_t1_i"], "T1_pump_off_ok": expt.status,
            "frequency_p_off": expt.cfg.expt.frequency_p}


def t2(args):
    _, cfg_path, _, _, cfg_dict = prepare(args, "T2_ramsey")
    params = {"reps": 500, "reset_mode": RESET_MODE}
    expt = meas.T2Experiment(cfg_dict, qi=QI, go=False, max_err=10, params=params)
    run(expt, args)
    if not args.no_update_config and expt.status:
        config.update_qubit(str(cfg_path), "f_ge", expt.data["new_freq"], QI)
        config.update_qubit(str(cfg_path), "T2r", expt.data["best_fit"][3], QI)
    return {"T2r": expt.data["best_fit"][3], "f_ge": expt.data["new_freq"],
            "T2r_ok": expt.status}


def t2_echo(args):
    _, cfg_path, _, _, cfg_dict = prepare(args, "T2_echo")
    params = {"experiment_type": "echo", "reset_mode": RESET_MODE}
    expt = meas.T2Experiment(cfg_dict, qi=QI, go=False, params=params)
    run(expt, args)
    if not args.no_update_config and expt.status:
        config.update_qubit(str(cfg_path), "T2e", expt.data["best_fit"][3], QI)
    return {"T2e": expt.data["best_fit"][3], "T2e_ok": expt.status}


def temperature(args):
    _, _, _, _, cfg_dict = prepare(args, "temperature")
    # No reset mode on purpose, as in 16_temperature.py: residual excited
    # population is what this measures.
    with nonblocking_plots():
        temp_mk, population, _, _ = measure_func.measure_temp(
            cfg_dict, qi=QI, temp=60, expts=32, reps=200, measure_f=False)
    return {"temp_mk": temp_mk, "population": population}


STEPS = [("buffer spectroscopy", buffer_spec), ("single shot", single_shot),
         ("T1", t1), ("T1 pump", t1_pump),
         ("T1 pump off band", t1_pump_off_band), ("T2 Ramsey", t2),
         ("T2 echo", t2_echo), ("temperature", temperature)]


def run_step(name, step, args):
    print(f"\n[{datetime.now():%H:%M:%S}] {name}")
    # One step's figures at a time, so a night of them doesn't pile up.
    plt.close("all")
    try:
        return step(args)
    except Exception:
        traceback.print_exc()
        print(f"[{name} failed; continuing]")
        return {}


def append_row(csv_path, row):
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, restval="")
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def plot_log(csv_path, png_path):
    """
    Every logged quantity vs time, redrawn from the CSV.

    A bare Figure rather than pyplot, so run_step's plt.close("all") never
    touches it and it never opens a window; open the PNG to watch the run.
    """
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig = Figure(figsize=(14, 11), constrained_layout=True)
    axes = fig.subplots(4, 3, sharex=True)
    for ax, (ylabel, series) in zip(axes.flat, PANELS):
        ok_values = []
        for i, (column, label, ok_column) in enumerate(series):
            points = [r for r in rows if r[column] not in ("", "nan")]
            if not points:
                continue
            times = np.array([datetime.fromisoformat(r["time"]) for r in points])
            values = np.array([float(r[column]) for r in points])
            if column in RELATIVE_KHZ:
                ylabel = f"{column} - {values[0]:.3f} MHz (kHz)"
                values = 1e3 * (values - values[0])
            ok = np.array([ok_column is None or r[ok_column] == "True"
                           for r in points])
            color = f"C{i}"
            ax.plot(times[ok], values[ok], "o-", color=color, ms=4, lw=1,
                    label=label)
            ax.plot(times[~ok], values[~ok], "o", color=color, ms=4, mfc="none")
            ok_values.extend(values[ok])
        # Limits from the good fits only, so one runaway failed fit can't
        # flatten the night's trend; failed points outside them are clipped.
        if ok_values:
            lo, hi = min(ok_values), max(ok_values)
            pad = 0.08 * (hi - lo) if hi > lo else 0.05 * abs(hi) or 1.0
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if len(series) > 1:
            ax.legend(fontsize="small")
    for ax in axes[-1]:
        ax.set_xlabel("time")
        ax.xaxis.set_major_locator(AutoDateLocator(minticks=3, maxticks=6))
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M"))
    fig.savefig(png_path, dpi=120)


def record(csv_path, cycle, name, step, args):
    row = {"time": datetime.now().isoformat(timespec="seconds"),
           "cycle": cycle, "step": name}
    row.update(run_step(name, step, args))
    append_row(csv_path, row)
    try:
        plot_log(csv_path, csv_path.with_suffix(".png"))
    except Exception:
        traceback.print_exc()
        print("[plot failed; continuing]")


if __name__ == "__main__":
    args = parse_args(__doc__, add_arguments, config=CONFIG)
    if args.no_show:
        plt.switch_backend("Agg")
    stop = next_occurrence(args.stop_at)

    # This script's own folder, for the CSV and the script copy; every
    # measurement after this one is nested inside it.
    _, _, _, _, cfg_dict = prepare_experiment(
        args, __file__, QI, JPA_ON, experiment_name="coherence_monitor")
    args.parent = Path(cfg_dict["data_path"]).name
    csv_path = Path(cfg_dict["metadata_path"]) / "coherence_monitor.csv"
    print(f"Running until {stop:%Y-%m-%d %H:%M}, JPA every "
          f"{args.jpa_interval:g} min. Log: {csv_path}")

    last_jpa = None
    cycle = 0
    try:
        while datetime.now() < stop:
            cycle += 1
            for name, step in STEPS:
                if datetime.now() >= stop:
                    break
                if last_jpa is None or time.time() - last_jpa >= 60 * args.jpa_interval:
                    record(csv_path, cycle, "JPA fine tuning", jpa, args)
                    last_jpa = time.time()
                record(csv_path, cycle, name, step, args)
    except KeyboardInterrupt:
        print("\ninterrupted")

    print(f"\nStopped after {cycle} cycles at {datetime.now():%H:%M}. "
          f"Log: {csv_path}, plot: {csv_path.with_suffix('.png')}")
