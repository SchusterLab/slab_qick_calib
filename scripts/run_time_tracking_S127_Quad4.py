"""
Time-track T1, T2 Ramsey and f_ge on all 6 qubits of S127 Quad4 chip7.

Wraps calib.time_tracking.time_tracking in fast mode: each round switches the
RF board to a qubit and runs a T1 sweep and a Ramsey sweep, then moves to the
next qubit. Rounds run back to back, so the sampling interval is set by how
long a round takes and is not uniform.

Sweep spans self-tune: a qubit's next T1/Ramsey span follows its own last good
fit, seeded from the config on the first round.

Usage:
    python scripts/run_time_tracking_S127_Quad4.py 0.01    # short test, 1 round
    python scripts/run_time_tracking_S127_Quad4.py 30      # the real run

Output (all under <EXPT_PATH>/Tracking/):
    <run_id>/            per-sweep h5 files
    <run_id>/images/     one PNG (and the config that produced it) per sweep
    csv/<run_id>_qubit_<qi>_tracking.csv    rewritten after every round
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")   # no blocking plt.show() in a 30-hour script

import numpy as np
from qick import QickConfig

from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
from slab_qick_calib.helpers import config
from slab_qick_calib.calib.time_tracking import time_tracking

# ---------------------------------------------------------------- settings
CFG_FILE = "coherence_qubit_S127_Quad4_chip7_rfboard.yml"
EXPT_PATH = r"C:\_Data\coherence_qubit_S127_Quad4_chip7"

QUBIT_LIST = [0, 1, 2, 3, 4, 5]
NS_PORT = 9090          # bf1_soc registers on 9090, not the default

# fast=True measures only T1 and T2 Ramsey per qubit per round.
FAST = True
# Ceilings on the recorded T1/T2. A runaway fit would otherwise become the
# next round's sweep span and stall the run; config T1 tops out near 470 us.
T1_MAX = 700
T2_MAX = 250
# Write a PNG of every individual T1/Ramsey sweep with its fit overlaid, into
# <run_id>/images/, alongside the config that produced it. Figures are closed
# as they are written, so this does not accumulate memory over a long run.
SAVE_IMAGES = True
BF_CLIENT = None        # set to a fridge client for mxc_temp / pulsetube_on

TOTAL_TIME = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

# ---------------------------------------------------------------- connect
cfg_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "samples", "configs", CFG_FILE)
)
auto_cfg = config.load(cfg_path)

im = InstrumentManager(ns_address=auto_cfg["aliases"]["ip"], port=NS_PORT)
soc = im[auto_cfg["aliases"]["soc"]]
soccfg = QickConfig(soc.get_cfg())
cfg_dict = {"soc": soccfg, "expt_path": EXPT_PATH, "cfg_file": cfg_path,
            "im": im}

print(f"Connected to {auto_cfg['aliases']['soc']} at {auto_cfg['aliases']['ip']}:{NS_PORT}")
print(f"Config: {cfg_path}")
print(f"Data:   {EXPT_PATH}")
print(f"Tracking qubits {QUBIT_LIST} for {TOTAL_TIME} hr "
      f"(fast={FAST}, save_images={SAVE_IMAGES})\n")
print("Seed values from config:")
for qi in QUBIT_LIST:
    print(f"  Q{qi}: f_ge {auto_cfg['device']['qubit']['f_ge'][qi]:9.4f} MHz  "
          f"T1 {auto_cfg['device']['qubit']['T1'][qi]:7.2f} us  "
          f"T2r {auto_cfg['device']['qubit']['T2r'][qi]:7.2f} us")
print()

# ---------------------------------------------------------------- the track
tt, tracking_path, tt_stats, tracking_id = time_tracking(
    QUBIT_LIST, cfg_dict,
    total_time=TOTAL_TIME,
    display=SAVE_IMAGES,
    fast=FAST,
    soc=soc,
    bf_client=BF_CLIENT,
    t1_max=T1_MAX,
    t2_max=T2_MAX,
)

# ---------------------------------------------------------------- summary
print(f"\n=== Summary: {tracking_id} ===")
print(f"Data:  {tracking_path}")
if SAVE_IMAGES:
    print(f"Plots: {os.path.join(str(tracking_path), 'images')}")
print(f"CSVs:  {os.path.join(EXPT_PATH, 'Tracking', 'csv')}")
for j, qi in enumerate(QUBIT_LIST):
    if j >= len(tt) or len(tt[j]["t1"]) == 0:
        print(f"Q{qi}: no successful measurements")
        continue
    d = tt[j]
    n = len(d["t1"])
    line = f"Q{qi}: {n} points"
    for key, label, unit in (("t1", "T1", "us"), ("t2", "T2*", "us"),
                             ("tphi", "Tphi", "us")):
        v = np.asarray(d[key], dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            line += f" | {label} {np.mean(v):.1f}+-{np.std(v):.1f} {unit}"
    f = np.asarray(d["f_ge"], dtype=float)
    f = f[np.isfinite(f)]
    if f.size:
        line += (f" | f_ge {np.mean(f):.4f} MHz, std {1e3*np.std(f):.1f} kHz, "
                 f"drift {1e3*np.ptp(f):.1f} kHz")
    print(line)
print("Done!")
