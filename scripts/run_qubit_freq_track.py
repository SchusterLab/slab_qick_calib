"""
Track f_ge, T1, T2 Ramsey, T2 Echo and the readout resonator on all 6 qubits of
S127 Quad1 chip2.

All 6 qubits share one ADC and one drive DAC with per-qubit tunable filters, so
they cannot be measured at the same instant. Each round switches the RF board to
a qubit and runs resonator -> Ramsey -> T1 -> Echo before moving on, then starts
the next round immediately (CADENCE_S = 0).

The qubit frequency comes from Ramsey, not spectroscopy: it measures the
detuning from the drive directly, to sub-kHz. Ramsey runs first so its f_ge is
written to the config before T1 and Echo, putting their pi pulses on the line
just measured. Spectroscopy is the recovery path only -- when a Ramsey fails,
the qubit_tuning.find_spec ladder relocates it.

Every result is written back to a working *copy* of the config inside the run
directory, so spans self-tune round to round and the master config is untouched.

Output, matching calib.time_tracking's layout:
    <expt_path>/Tracking/<run_id>/        h5, working config, CSVs, npz
    <expt_path>/Tracking/<run_id>/images/ per-experiment figures + summary plots
    <expt_path>/Tracking/csv/             CSVs for analyze_tracking.ipynb

Run it detached so it survives the shell:
    python scripts/run_qubit_freq_track.py
"""
import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no blocking plt.show() in a 5-hour script

import numpy as np
from qick import QickConfig

from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
from slab_qick_calib.helpers import config
from slab_qick_calib.calib import freq_tracking

# ---------------------------------------------------------------- settings
CFG_FILE = "coherence_qubit_S127_Quad1_chip2_2nd_rfboard.yml"
EXPT_PATH = r"C:\_Data\coherence_qubit_S127_Quad1_chip2_2nd"

QUBIT_LIST = [0, 1, 2, 3, 4, 5]
TOTAL_TIME = 30.0    # hours (upper bound)
MAX_ROUNDS = None    # stop after N complete rounds; None = time-limited only
# 0 = run rounds back to back, no idling, like calib.time_tracking. Maximises
# points but the interval between them jitters as the config-driven spans
# retune; use analysis/irregular.py rather than assuming a uniform rate.
CADENCE_S = 0.0
SIGNAL = "amps"
NS_PORT = 9090       # nameserver port (bf1_soc registers here)

# The qubit frequency comes from Ramsey, which measures the detuning from the
# drive directly and is far more precise than a Lorentzian on a broad saturated
# peak. Spectroscopy is kept only as the recovery path: SPEC_EVERY = 0 means it
# never runs on schedule, and SPEC_ON_FAIL runs it for a qubit whose Ramsey
# failed, to relocate it before the next round.
SPEC_EVERY = 0
SPEC_ON_FAIL = True
# Relocation ladder (qubit_tuning.find_spec): huge->coarse->medium->fine.
# It doubles readout.reps and spec_gain when even the coarsest scan misses,
# and that escalation is permanent -- the cap stops repeated rescues from
# compounding into much slower rounds for the rest of the run.
RELOCATE_START = "coarse"
RELOCATE_REPS_CAP = 10
SPEC_STYLE = "coarse"
SPEC_PARAMS = {"gain": 0.15, "expts": 100, "reps": 2000, "sep_readout": True,
               "length": 1, "readout_length": 1, "final_delay": 300}

# Coherence schedule as (kind, every, phase): a kind runs on round i when
# i % every == phase. Phases are staggered so each round carries exactly one
# coherence sweep -- T1 every 10 min, Ramsey and Echo every 20 min, all at the
# frequency that round's spectroscopy just fitted.
# Ramsey runs first in every round: it is the frequency source, and writing
# f_ge before T1 and Echo puts their pi pulses on the freshly measured line.
COHERENCE = (("ramsey", 1, 0), ("t1", 1, 0), ("echo", 1, 0))
COH_EXPTS = 40
COH_REPS = 300
# Kinds run as a plain meas.XExperiment(cfg_dict, qi=qi): all parameters from
# the config, and the experiment saves its own h5 and figure. These use the
# config f_ge rather than the frequency the round just fitted.
#   meas.T1Experiment(cfg_dict, qi=qi)
#   meas.T2Experiment(cfg_dict, qi=qi, max_err=10)
#   meas.T2Experiment(cfg_dict, qi=qi, params={'experiment_type': 'echo'})
COH_FROM_CONFIG = ("t1", "ramsey", "echo")
COH_CONFIG_KWARGS = {"ramsey": {"max_err": 10}}

# Resonator sweep alongside every qubit sweep, on the same RF setting:
#   meas.ResSpec(cfg_dict, qi=qi, params={'span': 'kappa'})
# All parameters from the config; ResSpec resolves 'kappa' to 7 x config kappa
# and saves its own h5 and figure.
RES_EVERY = 1
RES_FROM_CONFIG = True
# Only used when RES_FROM_CONFIG is False
RES_EXPTS = 100
RES_REPS = 300
RES_SPAN_KAPPA = 7.0

# Write a PNG of every individual sweep with its fit overlaid, under
# images/sweeps/<kind>/. ~1300 files over a 5 hr run, ~0.1 s each.
SAVE_IMAGES = True

# Write each result back to the config as it is measured (readout frequency
# and kappa, f_ge, T1, T2r, T2e). Only good fits are written. The run works on
# its own copy of the yml inside the output directory, so the master config in
# samples/configs is never modified.
UPDATE_CONFIG = True
UPDATE_FINAL_DELAY = True   # let the T1 write-back also set final_delay = 6*T1

# Optional fridge client exposing get_mxc_temperature() and is_pulsetube_on().
# Read once per round into the mxc_temp / pulsetube_on columns, which is what
# collections.plot_vs_temperature needs. Leave None if you have no client.
BF_CLIENT = None

# ---------------------------------------------------------------- connect
cfg_path = os.path.join(os.path.dirname(__file__), "..", "samples", "configs", CFG_FILE)
cfg_path = os.path.abspath(cfg_path)
auto_cfg = config.load(cfg_path)

# Output directory up front, so the working config can live inside it. Same
# layout as calib.time_tracking: <expt_path>/Tracking/<run_id> holds the whole
# run, and track_qubit_freqs points the experiments' expt_path at it so their
# h5 files and figures land inside rather than in the shared data folder.
run_id = f'{datetime.now().strftime("%Y_%m_%d_%H_%M")}_{TOTAL_TIME:.1f}hrs'
save_dir = Path(EXPT_PATH) / "Tracking" / run_id
save_dir.mkdir(parents=True, exist_ok=True)

# Everything -- the calibration, the RF board state, and every write-back --
# operates on this copy. The master config in samples/configs is left alone.
work_cfg = save_dir / Path(cfg_path).name
shutil.copy2(cfg_path, work_cfg)

im = InstrumentManager(ns_address=auto_cfg["aliases"]["ip"], port=NS_PORT)
soc = im[auto_cfg["aliases"]["soc"]]
soccfg = QickConfig(soc.get_cfg())
cfg_dict = {"soc": soccfg, "expt_path": EXPT_PATH, "cfg_file": str(work_cfg),
            "im": im}

print(f"Connected to {auto_cfg['aliases']['soc']} at {auto_cfg['aliases']['ip']}")
print(f"Tracking qubits {QUBIT_LIST} for {TOTAL_TIME} hr")
print(f"Master config: {cfg_path}  (read once, never written)")
print(f"Working copy:  {work_cfg}\n")

# --------------------------------------------------------------- settings
# The probe gain and averaging are fixed by SPEC_PARAMS, so there is no gain
# calibration to run. Only the fallback drive frequency is needed, and that is
# the config f_ge, refreshed from the working copy every round.
print("=== Frequency source: Ramsey ===")
print("  f_ge comes from t2r.data['new_freq'] each round; spectroscopy runs "
      + (f"every {SPEC_EVERY} rounds" if SPEC_EVERY else "only to relocate a "
         "qubit whose Ramsey failed"))
print(f"  recovery sweep: meas.QubitSpec(cfg_dict, qi=qi, style='{SPEC_STYLE}', "
      "params=" + ", ".join(f"{k}={v!r}" for k, v in SPEC_PARAMS.items()) + ")")
settings = {qi: {"center": auto_cfg["device"]["qubit"]["f_ge"][qi],
                 "gain": SPEC_PARAMS["gain"], "span": np.nan,
                 "reps": SPEC_PARAMS["reps"], "rounds": 1, "ok": True}
            for qi in QUBIT_LIST}
for qi in QUBIT_LIST:
    print(f"  Q{qi}: window centred on config f_ge "
          f"{settings[qi]['center']:.4f} MHz")
print()
if CADENCE_S:
    n = int(TOTAL_TIME * 3600 / CADENCE_S)
    print(f"Cadence: one round every {CADENCE_S/60:.0f} min "
          f"(~{n} points per qubit over {TOTAL_TIME} hr)")
    for kind, every, _ in (("resonator", RES_EVERY, 0),) + COHERENCE:
        if every:
            print(f"         {kind:>9} every {every} round(s) = "
                  f"{every*CADENCE_S/60:.0f} min (~{n//every} points each)")
else:
    print("Cadence: rounds run back to back, no idling. Interval is set by how "
          "long each round takes,")
    print("         so sampling is uneven -- analyse with analysis/irregular.py, "
          "not a uniform rate.")
    kinds = ", ".join(k for k, _, _ in COHERENCE)
    print(f"         every round measures: "
          f"{'resonator, ' if RES_EVERY else ''}{kinds} on all "
          f"{len(QUBIT_LIST)} qubits")
print()

# --------------------------------------------------------------- the track
limit = (f"{MAX_ROUNDS} round(s)" if MAX_ROUNDS else f"{TOTAL_TIME} hr")
print(f"=== round-robin: {limit} ===")
tracking, save_dir = freq_tracking.track_qubit_freqs(
    QUBIT_LIST, cfg_dict, soc, settings,
    total_time=TOTAL_TIME, max_rounds=MAX_ROUNDS, cadence_s=CADENCE_S,
    signal=SIGNAL,
    spec_params=SPEC_PARAMS, spec_style=SPEC_STYLE,
    spec_every=SPEC_EVERY, spec_on_fail=SPEC_ON_FAIL,
    relocate_start=RELOCATE_START, relocate_reps_cap=RELOCATE_REPS_CAP,
    bf_client=BF_CLIENT,
    coherence=COHERENCE, coh_expts=COH_EXPTS, coh_reps=COH_REPS,
    coh_from_config=COH_FROM_CONFIG, coh_config_kwargs=COH_CONFIG_KWARGS,
    res_every=RES_EVERY, res_expts=RES_EXPTS, res_reps=RES_REPS,
    res_span_kappa=RES_SPAN_KAPPA, res_from_config=RES_FROM_CONFIG,
    save_images=SAVE_IMAGES, update_config=UPDATE_CONFIG,
    update_final_delay=UPDATE_FINAL_DELAY, save_dir=save_dir,
)

# Keep the calibration alongside the data
with open(os.path.join(save_dir, "settings.json"), "w") as f:
    json.dump({str(qi): {k: (v if np.isscalar(v) else None)
                         for k, v in s.items() if k != "trials"}
               for qi, s in settings.items()}, f, indent=2, default=float)

freq_tracking.plot_tracking(tracking, save_dir)

print("\n=== Summary ===")
for qi in QUBIT_LIST:
    d = tracking[qi]
    # f_track is the frequency actually adopted, from whichever source
    # produced it. Keying off the spectroscopy 'ok' flag would report
    # "no good fits" for every qubit whenever Ramsey is the source.
    f0 = np.asarray(d["f_track"], dtype=float)
    src = np.asarray(d["f_track_src"], dtype=float)
    ok = np.isfinite(f0) & (src > 0)
    n = len(d["time"])
    if not np.any(ok):
        print(f"Q{qi}: no frequency measured ({n} rounds)")
        continue
    nr = int(np.sum(src[ok] == 2))
    line = (f"Q{qi}: {int(ok.sum())}/{n} good ({nr} Ramsey), "
            f"mean {np.mean(f0[ok]):.4f} MHz, std {1e3*np.std(f0[ok]):.1f} kHz, "
            f"drift {1e3*np.ptp(f0[ok]):.1f} kHz")
    rf = np.asarray(d.get("res_f0", []), dtype=float)
    rm = (np.asarray(d.get("res_ok", []), dtype=float) > .5) & np.isfinite(rf)
    if np.any(rm):
        line += (f" | res {np.mean(rf[rm]):.4f} MHz "
                 f"drift {1e3*np.ptp(rf[rm]):.0f} kHz ({int(rm.sum())})")
    for kind, label in (("t1", "T1"), ("ramsey", "T2*"), ("echo", "T2E")):
        if kind not in d:
            continue
        v = np.asarray(d[kind], dtype=float)
        r2 = np.asarray(d[f"{kind}_r2"], dtype=float)
        m = np.isfinite(v) & (v > 0) & (v < 5000) & (r2 > 0.5)
        if np.any(m):
            line += (f" | {label} {np.mean(v[m]):.0f}+-{np.std(v[m]):.0f}us "
                     f"({int(m.sum())})")
    print(line)
print("Done!")
