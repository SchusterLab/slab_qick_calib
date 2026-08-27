"""
Track f_ge, T1, T2 Ramsey, T2 Echo and the readout resonator on all 6 qubits of
S140 Pikachu.

Set MAX_ROUNDS to a small number for a shakedown of the full round-robin
(every code path -- RF switch, resonator, Ramsey, T1, Echo, config write-back,
CSV/npz/figure output -- runs once per qubit, and the per-round wall time gets
printed so the long run can be sized). Set MAX_ROUNDS = None and TOTAL_TIME to
the run length for the real thing.

Measured on this chip: 6.55 min/round, 66 s/qubit, so ~110 points per qubit
over 12 hours.

Same structure as run_qubit_freq_track.py; see that file for the reasoning
behind the Ramsey frequency source, the relocation ladder and the output layout.

Run it detached so it survives the shell:
    python scripts/run_qubit_freq_track_S140.py
"""
import os
import sys
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # no blocking plt.show() in an unattended script

import numpy as np
from qick import QickConfig

from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
from slab_qick_calib.helpers import config
from slab_qick_calib.calib import freq_tracking

# ---------------------------------------------------------------- settings
CFG_FILE = "coherence_qubit_S140_Pikachu_rfboard.yml"
EXPT_PATH = r"C:\_Data\coherence_qubit_S140_Pikachu"

# Absolute path to start from instead of samples/configs/CFG_FILE. Set this to
# a previous run's working copy to begin where that run left off, rather than
# replaying the settling transient from a stale master config. Q1, Q2 and Q3
# each sat several hundred kHz stale in the master, and the first run spent its
# opening rounds creeping and relocating onto their true lines; starting from
# the converged values skips that. None = use the master config.
START_CFG = r"C:\_Data\coherence_qubit_S140_Pikachu\Tracking\start_config_S140.yml"

QUBIT_LIST = [0, 1, 2, 3, 4, 5]
# Test run: stop after MAX_ROUNDS complete passes. TOTAL_TIME is only the
# safety cap that stops it running away if a round takes far longer than
# expected. For the 12 hour run set MAX_ROUNDS = None, TOTAL_TIME = 12.0.
TOTAL_TIME = 12.0    # hours (upper bound)
MAX_ROUNDS = None    # complete rounds; None = time-limited only
CADENCE_S = 0.0      # rounds back to back, no idling
SIGNAL = "amps"
NS_PORT = 9090       # nameserver port, matching the S140 notebook

# f_ge comes from Ramsey, which measures the detuning from the drive directly
# and is far more precise than a Lorentzian on a broad saturated peak.
# Spectroscopy is the recovery path only: SPEC_EVERY = 0 means it never runs on
# schedule, and SPEC_ON_FAIL relocates a qubit whose Ramsey failed.
SPEC_EVERY = 0
SPEC_ON_FAIL = True
RELOCATE_START = "coarse"
RELOCATE_REPS_CAP = 10
SPEC_STYLE = "coarse"
SPEC_PARAMS = {"gain": 0.15, "expts": 100, "reps": 2000, "sep_readout": True,
               "length": 1, "readout_length": 1, "final_delay": 300}

# Every round measures all three, Ramsey first so its f_ge is written to the
# config before T1 and Echo put their pi pulses on the line just measured.
COHERENCE = (("ramsey", 1, 0), ("t1", 1, 0), ("echo", 1, 0))
COH_EXPTS = 40
COH_REPS = 300
# All three run as a plain meas.XExperiment(cfg_dict, qi=qi): every parameter
# from the config, and each saves its own h5 and figure. COH_EXPTS/COH_REPS
# therefore go unused -- they only apply to kinds absent from this tuple.
COH_FROM_CONFIG = ("t1", "ramsey", "echo")
COH_CONFIG_KWARGS = {"ramsey": {"max_err": 10}}

RES_EVERY = 1
RES_FROM_CONFIG = True
RES_EXPTS = 100
RES_REPS = 300
RES_SPAN_KAPPA = 7.0

SAVE_IMAGES = True
UPDATE_CONFIG = True
UPDATE_FINAL_DELAY = True

# --- Undo the notebook's clamped T2 values -------------------------------
# The notebook's setup cell leaves max_t1 = 1, and its Ramsey/Echo cells pass
# rng_vals=[0.025, max_t1] / [0.04, 2*max_t1] to config.update_qubit. Every
# fitted T2 above the ceiling was replaced by the ceiling on write (see
# config.in_rng), so the config carries T2r = 1 us, T2e = 2 us for every qubit
# but Q2 -- real measurements, squashed to the bound.
#
# That matters here because Ramsey is the frequency source and T2Experiment
# takes span = 3*T2r and ramsey_freq = 1.5/T2r straight from the config. Left
# alone, round 0 would sweep 3 us at 1.5 MHz detuning on a ~100 us coherence,
# every Ramsey would fail, and all six qubits would queue for the relocation
# ladder before a single frequency was recorded.
#
# Seed the prior the repo uses elsewhere (config.init_config, and
# T1Experiment.update(first_time=True)): T2r = T1, T2e = 2*T1. Round 0 then
# measures the real values and writes them back, so the spans self-tune from
# round 1 on. Written only to the run's working copy -- the master config in
# samples/configs is never touched.
SEED_T2_BELOW = 5.0      # us; a T2r under this is a clamped/placeholder value

BF_CLIENT = None

# ---------------------------------------------------------------- connect
if START_CFG:
    cfg_path = os.path.abspath(START_CFG)
else:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "samples", "configs",
                            CFG_FILE)
    cfg_path = os.path.abspath(cfg_path)
auto_cfg = config.load(cfg_path)

_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
run_id = (f"{_stamp}_{TOTAL_TIME:.1f}hrs" if MAX_ROUNDS is None
          else f"{_stamp}_test_{MAX_ROUNDS}rounds")
save_dir = Path(EXPT_PATH) / "Tracking" / run_id
save_dir.mkdir(parents=True, exist_ok=True)

# Everything -- the RF board state and every write-back -- operates on this
# copy. The master config in samples/configs is left alone.
work_cfg = save_dir / Path(cfg_path).name
shutil.copy2(cfg_path, work_cfg)

seeded = []
for qi in QUBIT_LIST:
    t2r = auto_cfg["device"]["qubit"]["T2r"][qi]
    if t2r >= SEED_T2_BELOW:
        continue
    t1 = float(auto_cfg["device"]["qubit"]["T1"][qi])
    config.update_qubit(str(work_cfg), "T2r", t1, qi, sig=2, verbose=False)
    config.update_qubit(str(work_cfg), "T2e", 2 * t1, qi, sig=2, verbose=False)
    seeded.append((qi, t2r, t1))

im = InstrumentManager(ns_address=auto_cfg["aliases"]["ip"], port=NS_PORT)
soc = im[auto_cfg["aliases"]["soc"]]
soccfg = QickConfig(soc.get_cfg())
cfg_dict = {"soc": soccfg, "expt_path": EXPT_PATH, "cfg_file": str(work_cfg),
            "im": im}

print(f"Connected to {auto_cfg['aliases']['soc']} at {auto_cfg['aliases']['ip']}")
print(f"Tracking qubits {QUBIT_LIST} for "
      + (f"{TOTAL_TIME} hr" if MAX_ROUNDS is None
         else f"{MAX_ROUNDS} round(s) (safety cap {TOTAL_TIME} hr)"))
print(f"Start config:  {cfg_path}  (read once, never written)"
      + ("  [converged carry-over]" if START_CFG else ""))
print(f"Working copy:  {work_cfg}\n")

if seeded:
    print("=== Seeded clamped T2 values in the working copy ===")
    for qi, old, t1 in seeded:
        print(f"  Q{qi}: T2r {old:g} -> {t1:.2f} us, T2e -> {2*t1:.2f} us "
              f"(T1, 2*T1); Ramsey span {3*t1:.0f} us, "
              f"ramsey_freq {1e3*1.5/t1:.1f} kHz")
    print("  Round 0 measures the real values and writes them back, so the "
          "spans self-tune from round 1 on.\n")

# --------------------------------------------------------------- settings
print("=== Frequency source: Ramsey ===")
print("  f_ge comes from t2r.data['new_freq'] each round; spectroscopy runs "
      "only to relocate a qubit whose Ramsey failed")
settings = {qi: {"center": auto_cfg["device"]["qubit"]["f_ge"][qi],
                 "gain": SPEC_PARAMS["gain"], "span": np.nan,
                 "reps": SPEC_PARAMS["reps"], "rounds": 1, "ok": True}
            for qi in QUBIT_LIST}
for qi in QUBIT_LIST:
    print(f"  Q{qi}: f_ge {settings[qi]['center']:.4f} MHz, readout "
          f"{auto_cfg['device']['readout']['frequency'][qi]:.4f} MHz, "
          f"T1 {auto_cfg['device']['qubit']['T1'][qi]:.1f} us")
print()
print("Cadence: rounds run back to back, no idling. Every round measures "
      "resonator, ramsey, t1, echo")
print(f"         on all {len(QUBIT_LIST)} qubits.\n")

# --------------------------------------------------------------- the track
print("=== round-robin: "
      + (f"{TOTAL_TIME} hr" if MAX_ROUNDS is None else f"{MAX_ROUNDS} round(s)")
      + " ===")
t_start = datetime.now()
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
wall_s = (datetime.now() - t_start).total_seconds()

with open(os.path.join(save_dir, "settings.json"), "w") as f:
    json.dump({str(qi): {k: (v if np.isscalar(v) else None)
                         for k, v in s.items() if k != "trials"}
               for qi, s in settings.items()}, f, indent=2, default=float)

freq_tracking.plot_tracking(tracking, save_dir)

print("\n=== Summary ===")
for qi in QUBIT_LIST:
    d = tracking[qi]
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

# --- does the Ramsey-fed f_ge converge? ----------------------------------
# The 2-round test left Q2 and Q3 stepping further in the same direction each
# round while Q1/Q4/Q5 settled to sub-kHz. A correction larger than the
# artificial detuning (ramsey_freq = 1.5/T2r) is ambiguous in sign, so the
# write-back can push f_ge away from resonance instead of onto it, and over a
# 12 hour run that compounds. Shrinking steps mean it is converging; steps that
# hold or grow mean it is walking and ramsey_freq needs pinning.
final = config.load(str(work_cfg))
print("\n=== f_ge round-to-round steps (kHz) ===")
for qi in QUBIT_LIST:
    d = tracking[qi]
    f = np.asarray(d["f_track"], dtype=float)
    src = np.asarray(d["f_track_src"], dtype=float)
    f = np.where(src > 0, f, np.nan)
    steps = 1e3 * np.diff(f)
    txt = ", ".join("fail" if not np.isfinite(s) else f"{s:+.1f}" for s in steps)
    tail = np.abs(steps[-2:])
    tail = tail[np.isfinite(tail)]
    if tail.size and np.all(tail < 5):
        verdict = "converged"
    elif tail.size:
        verdict = "WALKING" if np.all(tail > 20) else "unsettled"
    else:
        verdict = "no data"
    rf = 1e3 * 1.5 / final["device"]["qubit"]["T2r"][qi]   # ramsey_freq, kHz
    print(f"  Q{qi}: {txt}   [{verdict}; ramsey_freq {rf:.0f} kHz]")

# --- what the test learned, for sizing the 12 hour run -------------------
n_rounds = max(len(tracking[qi]["time"]) for qi in QUBIT_LIST)
if n_rounds:
    per_round = wall_s / n_rounds
    print(f"\n=== Timing ===")
    print(f"{n_rounds} round(s) in {wall_s/60:.1f} min "
          f"= {per_round/60:.2f} min/round ({per_round/len(QUBIT_LIST):.0f} s/qubit)")
    print(f"A {TOTAL_TIME:.1f} hr run gives "
          f"~{int(TOTAL_TIME*3600/per_round)} points per qubit at this rate.")

# The measured values next to what the master config still says -- the numbers
# to copy over before the long run.
print("\n=== Config drift over the test (master -> working copy) ===")
for qi in QUBIT_LIST:
    parts = []
    for sec, field, fmt in (("qubit", "f_ge", "{:.4f}"),
                            ("qubit", "T1", "{:.1f}"),
                            ("qubit", "T2r", "{:.1f}"),
                            ("qubit", "T2e", "{:.1f}"),
                            ("readout", "frequency", "{:.4f}")):
        a = auto_cfg["device"][sec][field][qi]
        b = final["device"][sec][field][qi]
        if a != b:
            parts.append(f"{field} {fmt.format(a)}->{fmt.format(b)}")
    print(f"Q{qi}: " + (", ".join(parts) if parts else "unchanged"))
print(f"\nWorking config: {work_cfg}")
print(f"Output:         {save_dir}")
print("Done!")