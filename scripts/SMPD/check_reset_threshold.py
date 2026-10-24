"""reset_threshold: absent -> falls back, NaN -> falls back, set -> used."""
import shutil, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")

SMPD = Path(r"C:\_Lib\python\slab_qick_calib\scripts\SMPD")
sys.path.insert(0, str(SMPD))
from verify_active_reset import StubSocCfg, StubIM, REPO
import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

TMP = Path(__file__).parent / "_cfgtest"
TMP.mkdir(exist_ok=True)
SRC = REPO / "configs" / "2026_09_21_bfg33_smpd_v2_waymar_rfboard.yml"
QI = 1

def build(cfg_path):
    d = {"soc": StubSocCfg(), "expt_path": str(TMP), "data_path": str(TMP),
         "metadata_path": str(TMP), "cfg_file": str(cfg_path), "im": StubIM()}
    return meas.HistogramExperiment(d, qi=QI, go=False, check_f=False,
                                    params={"shots": 10, "reset_mode": "active"})

base = config.load(str(SRC))
disc = base.device.readout.threshold[QI]
print(f"device.readout.threshold[{QI}] = {disc}\n")

def strip_reset_threshold(path):
    """Remove the key, so the test builds its own starting state."""
    cfg = config.load(str(path))
    cfg.device.readout.pop("reset_threshold", None)
    config.save(cfg, str(path))


def set_reset_threshold(path, values):
    cfg = config.load(str(path))
    cfg.device.readout["reset_threshold"] = list(values)
    config.save(cfg, str(path))


# 1. key absent. Constructed rather than assumed: 12b writes a real value
# into the live config, so the source may already carry one.
p1 = TMP / "absent.yml"; shutil.copy2(SRC, p1); strip_reset_threshold(p1)
c = config.load(str(p1))
print("1. key absent      ->", "absent" if "reset_threshold" not in c.device.readout else "present")
e = build(p1)
print(f"   threshold_v = {e.cfg.expt.threshold_v}   expect {disc}   "
      f"{'PASS' if e.cfg.expt.threshold_v == disc else 'FAIL'}")

# 2. present but NaN for this qubit (what update_readout leaves for the
# qubits you have not calibrated)
p2 = TMP / "nan.yml"; shutil.copy2(SRC, p2)
set_reset_threshold(p2, [4.39, float("nan")])
c = config.load(str(p2))
print(f"\n2. NaN for qi={QI}   -> reset_threshold = {c.device.readout.reset_threshold}")
e = build(p2)
print(f"   threshold_v = {e.cfg.expt.threshold_v}   expect {disc}   "
      f"{'PASS' if e.cfg.expt.threshold_v == disc else 'FAIL'}")

# 3. set for this qubit
p3 = TMP / "set.yml"; shutil.copy2(SRC, p3); strip_reset_threshold(p3)
config.update_readout(str(p3), "reset_threshold", 4.39, QI, verbose=False)
c = config.load(str(p3))
print(f"\n3. set for qi={QI}   -> reset_threshold = {c.device.readout.reset_threshold}")
e = build(p3)
print(f"   threshold_v = {e.cfg.expt.threshold_v}   expect 4.39   "
      f"{'PASS' if e.cfg.expt.threshold_v == 4.39 else 'FAIL'}")

# 4. an explicit params override still wins over both
e = meas.HistogramExperiment(
    {"soc": StubSocCfg(), "expt_path": str(TMP), "data_path": str(TMP),
     "metadata_path": str(TMP), "cfg_file": str(p3), "im": StubIM()},
    qi=QI, go=False, check_f=False,
    params={"shots": 10, "reset_mode": "active", "threshold_v": -1.0})
print(f"\n4. params override -> threshold_v = {e.cfg.expt.threshold_v}   expect -1.0   "
      f"{'PASS' if e.cfg.expt.threshold_v == -1.0 else 'FAIL'}")

# 5. The integer threshold tracks the readout_length actually played, not the
# config's. 12d/12e sweep readout_length, and the accumulated I scales with it,
# so a threshold pinned to the config length would branch on a value off by the
# ratio -- silently resetting everything at the long end and nothing at the
# short end. Ratio, not absolute value, so the stub f_fabric cancels.
L_cfg = config.load(str(p3)).device.readout.readout_length[QI]


def thr_at(length):
    return meas.HistogramExperiment(
        {"soc": StubSocCfg(), "expt_path": str(TMP), "data_path": str(TMP),
         "metadata_path": str(TMP), "cfg_file": str(p3), "im": StubIM()},
        qi=QI, go=False, check_f=False,
        params={"shots": 10, "reset_mode": "active",
                "readout_length": length}).cfg.expt.threshold


base_thr = thr_at(L_cfg)
print("")
print(f"5. threshold scales with readout_length (config {L_cfg} us -> {base_thr})")
ok = True
for mult in (0.5, 2.0, 3.0):
    got, want = thr_at(L_cfg * mult), base_thr * mult
    good = abs(got - want) <= max(2, abs(want) * 1e-3)
    ok &= good
    print(f"   x{mult:<4} -> {got:>9}   expect ~{want:>10.0f}   "
          f"{'PASS' if good else 'FAIL'}")
print(f"   {'PASS' if ok else 'FAIL'}")

shutil.rmtree(TMP, ignore_errors=True)
