"""Check the three reset modes offline: what they derive and what they cost.

`cfg.expt.reset_mode` selects between "passive" (the full T1 wait), "active"
(the measurement-based conditional reset) and "f0g1" (the unconditional
all-microwave reset). This asserts, with no hardware:

  1. each mode reaches cfg.expt and resolves the way it should
  2. the f0g1 mode derives a complete, sane drive from device.f0g1
  3. `common.effective_final_delay` reports the right inter-shot tail for each
     -- script 26's NEP is computed from exactly that number
  4. an uncalibrated device refuses f0g1 mode, naming the script to run
  5. the legacy path still works: `active_reset=True` with no `reset_mode`
     resolves to "active", so the legacy classes (flux/, two_qubit/, t1_cont,
     ...) are untouched
  6. `reset_mode` and `active_reset` disagreeing raises rather than silently
     turning the reset off
  7. every cfg.expt key reset_f0g1 and _initialize read is present -- the
     Programs cannot be built offline, and a missing key would be a
     runtime AttributeError
  8. reset parameters resolve as params > config > derived: the config's
     device.f0g1.reset_* / device.readout.reset_ringdown are used, a params
     value overrides them, and None/NaN means "derive it"
  9. pulse types come from the config (device.fwm.*_pulse_type,
     device.f0g1.pulse_type / pulse_type_ef), a missing key means flat_top,
     and a const f0g1 reset adds no ramp time

    python check_reset_modes.py
"""

import contextlib
import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib

matplotlib.use("Agg")

SMPD = Path(__file__).resolve().parent
sys.path.insert(0, str(SMPD))

import numpy as np  # noqa: E402

import slab_qick_calib.experiments as meas  # noqa: E402
from common import effective_final_delay  # noqa: E402
from slab_qick_calib.experiments.general.qick_program import (  # noqa: E402
    RESET_MODES,
    reset_mode_of,
)
from slab_qick_calib.helpers import config  # noqa: E402
from verify_active_reset import REPO, StubIM, StubSocCfg  # noqa: E402

WAYMAR = "2026_09_21_bfg33_smpd_v2_waymar_rfboard.yml"
YUVI = "2026_09_21_bfg33_smpd_v2_yuvi_rfboard.yml"
QI = 1
TMP = SMPD / "_scratch_data"
TMP.mkdir(exist_ok=True)

FWM = {"shots": 10, "frequency_p": 5210, "gain_p": 0.5, "check_f": True}


def cfg_dict(cfg_name):
    return {
        "soc": StubSocCfg(),
        "expt_path": str(TMP),
        "data_path": str(TMP),
        "metadata_path": str(TMP),
        "cfg_file": str(REPO / "configs" / cfg_name),
        "im": StubIM(),
    }


def build(cfg_name, params, quiet=True):
    d = cfg_dict(cfg_name)
    auto = config.load(d["cfg_file"])
    p = {**FWM, "frequency_b": auto.device.readout.frequency[0], **params}
    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet \
        else contextlib.nullcontext()
    with ctx:
        return meas.SMPDHistogramExperiment(
            cfg_dict(cfg_name), qi=QI, go=False, params=p
        )


failures = []


def check(label, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(label)


print("=== 1/6  the resolver ===")
for cfgd, want in (
    ({}, "passive"),
    ({"active_reset": False}, "passive"),
    ({"active_reset": True}, "active"),
    ({"reset_mode": "passive"}, "passive"),
    ({"reset_mode": "active"}, "active"),
    ({"reset_mode": "f0g1"}, "f0g1"),
    ({"reset_mode": "active", "active_reset": True}, "active"),
    ({"reset_mode": "passive", "active_reset": False}, "passive"),
):
    got = reset_mode_of(cfgd)
    check(f"{cfgd} -> {got}", got == want, "" if got == want else f"wanted {want}")
check("RESET_MODES",
      tuple(RESET_MODES) == ("passive", "active", "f0g1", "while"),
      str(RESET_MODES))

print("\n=== 2/6  each mode reaches cfg.expt (SMPDHistogramExperiment) ===")
built = {}
# 'while' needs adcs.readout.feedback_ch, which the bfg33 configs predate.
_ro = config.load(str(REPO / "configs" / WAYMAR)).hw.soc.adcs.readout
MODES_HERE = [m for m in RESET_MODES
              if m != "while" or "feedback_ch" in _ro]
for mode in MODES_HERE:
    e = build(WAYMAR, {"reset_mode": mode, "reset": 3})
    built[mode] = e
    got = reset_mode_of(e.cfg.expt)
    check(f"reset_mode={mode!r}", got == mode, f"resolved {got!r}")

print("\n=== 3/6  the f0g1 mode derives a complete drive ===")
e = built["f0g1"]
KEYS = ("f0g1_reset_gain", "f0g1_reset_freq", "f0g1_reset_length",
        "f0g1_reset_gain_ef", "f0g1_reset_freq_ef", "f0g1_reset_gap",
        "f0g1_reset_ramp_sigma", "f0g1_reset_ramp_sigma_inc",
        "f0g1_reset_gamma", "f0g1_reset_omega_ef")
for k in KEYS:
    v = e.cfg.expt.get(k, None)
    check(f"{k} = {v}", v is not None and np.isfinite(v) and v > 0)
# Both tones must cover the same window, or one drives on after the other stops.
check("ef tone shares the f0g1 length and ramps", True,
      f"length {e.cfg.expt.f0g1_reset_length:.3f} us, ramps "
      f"{e.cfg.expt.f0g1_reset_ramp_sigma}x{e.cfg.expt.f0g1_reset_ramp_sigma_inc}")
# No threshold: nothing is measured, so there is nothing to discriminate.
check("no threshold in f0g1 mode", "threshold" not in e.cfg.expt,
      "threshold present" if "threshold" in e.cfg.expt else "absent, as expected")
check("threshold present in active mode", "threshold" in built["active"].cfg.expt)

print("\n=== 4/6  inter-shot tail per mode (script 26's NEP input) ===")
tails = {}
for mode in MODES_HERE:
    tails[mode] = effective_final_delay(built[mode].cfg.expt)
    print(f"      {mode:8} {tails[mode]:8.2f} us   "
          f"(_get_final_delay {built[mode]._get_final_delay()})")
passive_cfg = built["passive"].cfg.expt
check("passive == device.readout.final_delay",
      abs(tails["passive"] - passive_cfg.final_delay) < 1e-9,
      f"{tails['passive']:.2f} us")
check("f0g1 < active < passive",
      tails["f0g1"] < tails["active"] < tails["passive"],
      f"{tails['f0g1']:.1f} < {tails['active']:.1f} < {tails['passive']:.1f}")
# Structural, not a ratio: the f0g1 tail must be exactly its own parts, with
# no readout term. That is the claim -- reset() adds `reset - 1` conditional
# readouts per shot at readout_length each, and this adds none.
f = built["f0g1"].cfg.expt
parts = {
    "base (1 us)": 1.0,
    "read_wait": f.read_wait,
    "extra_delay": f.extra_delay,
    "drive": f.f0g1_reset_length,
    "ramps": f.f0g1_reset_ramp_sigma * f.f0g1_reset_ramp_sigma_inc,
    "photon gap": f.f0g1_reset_gap,
}
for name, v in parts.items():
    print(f"        {name:14} {v:7.3f} us")
check("f0g1 tail is exactly its parts (no readouts)",
      abs(tails["f0g1"] - sum(parts.values())) < 1e-9,
      f"{tails['f0g1']:.3f} vs {sum(parts.values()):.3f}")
a = built["active"].cfg.expt
n_reads = a.reset if a.reset_verify else a.reset - 1
check("active tail includes its conditional readouts",
      abs(tails["active"] - (1.0 + a.reset * (a.read_wait + a.extra_delay)
                             + n_reads * a.readout_length)) < 1e-9,
      f"{n_reads} readouts x {a.readout_length:.2f} us = "
      f"{n_reads * a.readout_length:.1f} us of the {tails['active']:.1f}")
check("f0g1 beats active", tails["f0g1"] < tails["active"],
      f"{tails['active'] / tails['f0g1']:.2f}x faster")
# setup_reset keeps the long delay whatever the mode, so a reset that is not
# working cannot quietly corrupt the data.
staged = build(WAYMAR, {"reset_mode": "f0g1", "reset": 3, "setup_reset": True})
check("setup_reset keeps the passive delay",
      effective_final_delay(staged.cfg.expt) > passive_cfg.final_delay,
      f"{effective_final_delay(staged.cfg.expt):.1f} us")

print("\n=== 5/6  an uncalibrated device refuses f0g1 mode ===")
try:
    build(YUVI, {"reset_mode": "f0g1", "reset": 3})
    check("Yuvi (no hw.soc.dacs.f0g1) raises", False, "no error")
except ValueError as exc:
    check("Yuvi (no hw.soc.dacs.f0g1) raises", "34" in str(exc) or "f0g1" in str(exc),
          str(exc)[:70])
# A wired line but a missing fit must also raise, naming the script.
import shutil  # noqa: E402

stale = TMP / "no_gtilde.yml"
shutil.copy2(REPO / "configs" / WAYMAR, stale)
c = config.load(str(stale))
c.device.f0g1["g_tilde_fit"] = [float("nan"), float("nan")]
config.save(c, str(stale))
d = cfg_dict(WAYMAR)
d["cfg_file"] = str(stale)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        meas.SMPDHistogramExperiment(
            d, qi=QI, go=False,
            params={**FWM, "frequency_b": 6682.254, "reset_mode": "f0g1",
                    "reset": 3},
        )
    check("missing g_tilde_fit raises", False, "no error")
except ValueError as exc:
    check("missing g_tilde_fit raises, naming 34", "34" in str(exc), str(exc)[:70])

print("\n=== 6/6  the legacy path and the disagreement guard ===")
# An unmigrated class: active_reset with no reset_mode must behave as before.
with contextlib.redirect_stdout(io.StringIO()):
    legacy = meas.T1ContExperiment(cfg_dict(WAYMAR), qi=QI, go=False,
                                   params={"reps": 10, "active_reset": True,
                                           "reset": 3})
check("T1ContExperiment (unmigrated) active_reset=True -> active",
      reset_mode_of(legacy.cfg.expt) == "active")
check("  and it still gets a threshold", "threshold" in legacy.cfg.expt,
      str(legacy.cfg.expt.get("threshold")))
check("  and the short inter-shot delay", legacy._get_final_delay() == 1)

with contextlib.redirect_stdout(io.StringIO()):
    legacy_off = meas.T1ContExperiment(cfg_dict(WAYMAR), qi=QI, go=False,
                                       params={"reps": 10, "active_reset": False})
check("T1ContExperiment active_reset=False -> passive",
      reset_mode_of(legacy_off.cfg.expt) == "passive")

# A migrated class handed the legacy spelling: silently ignoring it would turn
# the reset off, so it has to raise.
try:
    build(WAYMAR, {"active_reset": True, "reset": 3})
    check("migrated class + active_reset=True raises", False, "no error")
except ValueError as exc:
    check("migrated class + active_reset=True raises", "disagree" in str(exc),
          str(exc)[:70])
for bad in ({"reset_mode": "nonsense"},):
    try:
        reset_mode_of(bad)
        check(f"{bad} raises", False, "no error")
    except ValueError as exc:
        check(f"{bad} raises", True, str(exc)[:60])

print()
print("=== 7/7  every cfg.expt key the Program reads is actually there ===")
# Building a Program needs real QICK objects, so reset_f0g1() and the f0g1
# branch of _initialize are never executed offline. This closes the gap that
# matters: the repo forbids cfg.expt.get() inside a Program, so a key the
# Program reads but params_def never sets is a runtime AttributeError. Parse
# the two functions and confirm every cfg.expt.X they touch exists.
import ast  # noqa: E402

PROG = REPO / "experiments" / "general" / "qick_program.py"
tree = ast.parse(io.open(PROG, encoding="utf-8").read())


def cfg_expt_keys(node):
    """Every X in cfg.expt.X reachable under `node`."""
    out = set()
    for n in ast.walk(node):
        if (isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Attribute)
                and n.value.attr == "expt"
                and isinstance(n.value.value, ast.Name)
                and n.value.value.id == "cfg"):
            out.add(n.attr)
    return out


# Scoped to QickProgram: there are three _initialize methods in that file
# (QickProgram, QickProgram2Q, QickSMPDProgram) and only the first chains into
# the reset path SMPDHistogramProgram uses.
qp = next(n for n in ast.walk(tree)
          if isinstance(n, ast.ClassDef) and n.name == "QickProgram")
fns = {n.name: n for n in qp.body if isinstance(n, ast.FunctionDef)}
reset_keys = cfg_expt_keys(fns["reset_f0g1"])
print(f"      reset_f0g1 reads: {sorted(reset_keys)}")
for k in sorted(reset_keys):
    check(f"reset_f0g1 needs cfg.expt.{k}", k in built["f0g1"].cfg.expt)

# The f0g1 branch of _initialize: find the `if reset_mode_of(...) == "f0g1"`.
init_keys = set()
for n in ast.walk(fns["_initialize"]):
    if isinstance(n, ast.If) and "f0g1" in ast.unparse(n.test):
        init_keys |= cfg_expt_keys(n)
print(f"      _initialize f0g1 branch reads: {sorted(init_keys)}")
check("the f0g1 branch was found in _initialize", bool(init_keys))
for k in sorted(init_keys):
    check(f"_initialize needs cfg.expt.{k}", k in built["f0g1"].cfg.expt)

print()
print("=== 8/8  params > config > derived ===")
# Every knob is checked against a value computed independently of the code
# under test. WAYMAR (bfg33) has no reset_* keys, so it exercises "missing".
GAMMA = built["f0g1"].cfg.expt.f0g1_reset_gamma
DEFAULT_GAP = built["f0g1"].cfg.expt.f0g1_reset_gap
DEFAULT_EXTRA = built["active"].cfg.expt.extra_delay


def config_with(name, f0g1=None, readout=None, fwm=None):
    """A temp copy of WAYMAR with some device.f0g1 / readout / fwm keys set."""
    path = TMP / name
    shutil.copy2(REPO / "configs" / WAYMAR, path)
    c = config.load(str(path))
    for section, keys in (("f0g1", f0g1), ("readout", readout), ("fwm", fwm)):
        for k, v in (keys or {}).items():
            c.device[section][k] = v
    config.save(c, str(path))
    return name


def tmp_cfg_dict(name):
    d = cfg_dict(WAYMAR)
    d["cfg_file"] = str(TMP / name)
    return d


def build_tmp(name, params):
    d = cfg_dict(WAYMAR)
    d["cfg_file"] = str(TMP / name)
    auto = config.load(d["cfg_file"])
    p = {**FWM, "frequency_b": auto.device.readout.frequency[0], **params}
    with contextlib.redirect_stdout(io.StringIO()):
        return meas.SMPDHistogramExperiment(d, qi=QI, go=False, params=p)


# Missing keys: everything derived (section 3 already checked the values).
for residual, label in ((0.001, "0.1%"), (1e-4, "0.01%")):
    name = config_with(f"residual_{label}.yml", f0g1={"reset_residual": residual})
    e8 = build_tmp(name, {"reset_mode": "f0g1"})
    want = np.log(1 / residual) / (2 * np.pi * GAMMA)
    got = e8.cfg.expt.f0g1_reset_length
    check(f"config reset_residual {label} -> {got:.3f} us drive",
          abs(got - want) < 1e-9, f"wanted {want:.3f}")

name = config_with("pinned.yml", f0g1={"reset_length": 25.0, "reset_gap": 1.0,
                                       "reset_gain": 0.74, "reset_gain_ef": 0.05,
                                       "reset_residual": 0.001})
e8 = build_tmp(name, {"reset_mode": "f0g1"})
check("config reset_length pins it, residual ignored",
      e8.cfg.expt.f0g1_reset_length == 25.0, f"{e8.cfg.expt.f0g1_reset_length}")
check("config reset_gap pins it", e8.cfg.expt.f0g1_reset_gap == 1.0,
      f"{e8.cfg.expt.f0g1_reset_gap}")
check("config reset_gain / reset_gain_ef are used",
      e8.cfg.expt.f0g1_reset_gain == 0.74 and e8.cfg.expt.f0g1_reset_gain_ef == 0.05,
      f"{e8.cfg.expt.f0g1_reset_gain}, {e8.cfg.expt.f0g1_reset_gain_ef}")
check("  and the gap shortens the tail accordingly",
      abs(effective_final_delay(e8.cfg.expt)
          - (tails["f0g1"] - DEFAULT_GAP + 1.0 - built["f0g1"].cfg.expt.f0g1_reset_length
             + 25.0)) < 1e-6,
      f"{effective_final_delay(e8.cfg.expt):.3f} us")

e8 = build_tmp(name, {"reset_mode": "f0g1", "f0g1_reset_length": 3.0,
                      "f0g1_reset_gap": 0.0, "f0g1_reset_gain": None})
check("params override the config",
      e8.cfg.expt.f0g1_reset_length == 3.0 and e8.cfg.expt.f0g1_reset_gap == 0.0,
      f"length {e8.cfg.expt.f0g1_reset_length}, gap {e8.cfg.expt.f0g1_reset_gap}")
check("  a None in params falls through to the config",
      e8.cfg.expt.f0g1_reset_gain == 0.74, f"{e8.cfg.expt.f0g1_reset_gain}")

# null / NaN in the config means "derive", not "override with nothing".
name = config_with("nulls.yml", f0g1={"reset_length": None, "reset_gap": float("nan"),
                                      "reset_gain": None, "reset_residual": None})
e8 = build_tmp(name, {"reset_mode": "f0g1"})
for k in ("f0g1_reset_length", "f0g1_reset_gap", "f0g1_reset_gain"):
    v = e8.cfg.expt.get(k, None)
    check(f"config {k}=null/NaN was derived", v is not None and np.isfinite(v) and v > 0,
          str(v))
check("  to the same values as a config without the keys",
      abs(e8.cfg.expt.f0g1_reset_length - built["f0g1"].cfg.expt.f0g1_reset_length) < 1e-9
      and abs(e8.cfg.expt.f0g1_reset_gap - DEFAULT_GAP) < 1e-9)

# Ring-down: reset_ringdown = read_wait + extra_delay, NaN = the old default.
name = config_with("ringdown.yml", readout={"reset_ringdown": [float("nan"), 1.0]})
for mode in ("active", "f0g1"):
    e8 = build_tmp(name, {"reset_mode": mode})
    got = e8.cfg.expt.read_wait + e8.cfg.expt.extra_delay
    check(f"{mode}: config reset_ringdown 1.0 -> {got:.3f} us", abs(got - 1.0) < 1e-9)
e8 = build_tmp(name, {"reset_mode": "active", "extra_delay": 0.4})
check("  params extra_delay overrides it", e8.cfg.expt.extra_delay == 0.4)
name = config_with("ringdown_nan.yml", readout={"reset_ringdown": [float("nan")] * 2})
e8 = build_tmp(name, {"reset_mode": "active"})
check("  NaN keeps the default", abs(e8.cfg.expt.extra_delay - DEFAULT_EXTRA) < 1e-9,
      f"{e8.cfg.expt.extra_delay:.3f} us")

print()
print("=== 9/9  pulse types come from the config ===")
b = built["f0g1"].cfg.expt
check("bfg33 (no keys) -> flat_top everywhere",
      (b.pulse_type_p, b.pulse_type_b, b.f0g1_reset_pulse_type,
       b.f0g1_reset_pulse_type_ef) == ("flat_top",) * 4)

name = config_with("const_shapes.yml",
                   fwm={"pump_pulse_type": "const", "buffer_pulse_type": "const"},
                   f0g1={"pulse_type": "const", "pulse_type_ef": "const"})
e9 = build_tmp(name, {"reset_mode": "f0g1"})
check("FWM pump / buffer from device.fwm",
      (e9.cfg.expt.pulse_type_p, e9.cfg.expt.pulse_type_b) == ("const", "const"))
check("reset tones from device.f0g1",
      (e9.cfg.expt.f0g1_reset_pulse_type, e9.cfg.expt.f0g1_reset_pulse_type_ef)
      == ("const", "const"))
x = e9.cfg.expt
want = 1.0 + x.read_wait + x.extra_delay + x.f0g1_reset_length + x.f0g1_reset_gap
check("const reset tail has no ramp term",
      abs(effective_final_delay(x) - want) < 1e-9,
      f"{effective_final_delay(x):.3f} vs {want:.3f} us")
e9 = build_tmp(name, {"pulse_type_b": "flat_top"})
check("params override the config", e9.cfg.expt.pulse_type_b == "flat_top")

with contextlib.redirect_stdout(io.StringIO()):
    d = tmp_cfg_dict(name)
    stark = meas.StarkSpec(d, qi=QI, go=False, params={
        "expts": 20, "stark_expts": 3, "stark_chan": 0, "stark_qubit": 0,
        "stark_freq": 6682.0})
    t2 = meas.T2Experiment(d, qi=QI, go=False, params={"reps": 10, "buffer": True})
    t1 = meas.T1Experiment(d, qi=QI, go=False,
                           params={"reps": 10, "pump": True, "frequency_p": 5210, "gain_p": 0.3})
    spec = meas.F0G1Spec(d, qi=QI, go=False, params={"span": 5, "expts": 20})
    efspec = meas.EFSpecF0G1(d, qi=QI, go=False, params={
        "span": 5, "expts": 20, "gain_f0g1": 0.5, "gain_ef": 0.05})
    efrabi = meas.EFRabiPower(d, qi=QI, go=False, params={})
check("Stark tone (17) follows buffer_pulse_type", stark.cfg.expt.stark_pulse_type == "const")
check("Ramsey buffer (38) follows buffer_pulse_type", t2.cfg.expt.pulse_type_b == "const")
check("T1 pump (39) follows pump_pulse_type", t1.cfg.expt.pulse_type_p == "const")
check("F0G1Spec drive follows pulse_type", spec.cfg.expt.pulse_type_f0g1 == "const")
check("EFRabiPower drive follows pulse_type_ef", efrabi.cfg.expt.pulse_type == "const")
x = efspec.cfg.expt
from slab_qick_calib.helpers.f0g1 import omega_ef_for_gain  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
    t_pi = 1 / (2 * omega_ef_for_gain(config.load(d["cfg_file"]), x.gain_ef))
check("EFSpecF0G1 const probe: the whole pi is flat", abs(x.length_ef - t_pi) < 1e-9,
      f"length_ef {x.length_ef:.4f} vs t_pi {t_pi:.4f} us")

name = config_with("mixed_f0g1.yml", f0g1={"pulse_type": "const", "pulse_type_ef": "flat_top"})
try:
    build_tmp(name, {"reset_mode": "f0g1"})
    check("mismatched reset tone types raise", False, "no error")
except ValueError as exc:
    check("mismatched reset tone types raise", "must match" in str(exc), str(exc)[:60])
name = config_with("bad_type.yml", fwm={"buffer_pulse_type": "gauss"})
try:
    build_tmp(name, {})
    check("an unsupported type raises", False, "no error")
except ValueError as exc:
    check("an unsupported type raises", "buffer_pulse_type" in str(exc), str(exc)[:60])

print()
if failures:
    print(f"{len(failures)} FAILURE(S): " + "; ".join(failures))
    sys.exit(1)
print("ALL OK")
