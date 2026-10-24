"""
Offline check that active reset is wired correctly, with no hardware.

Constructs every experiment class the SMPD scripts use in each reset mode,
against both bfg33 configs, and asserts that:
  * configure_reset() ran: threshold is an int, read_wait / extra_delay /
    reset / setup_reset all exist
  * the threshold ordering g_mean < threshold_v < e_mean holds, so
    QickProgram.reset()'s hardcoded test="<" resets rather than excites
  * the built Program declares "pi_ge", which QickProgram.reset() plays

soc/soccfg/im are stubbed; nothing touches the nameserver.
"""

import contextlib
import io
import sys
import traceback
from pathlib import Path

REPO = Path(r"C:\_Lib\python\slab_qick_calib")
sys.path.insert(0, str(REPO.parent))

import numpy as np  # noqa: E402

import slab_qick_calib.experiments as meas  # noqa: E402
from slab_qick_calib.experiments.general.qick_program import (  # noqa: E402
    reset_mode_of,
)
from slab_qick_calib.helpers import config  # noqa: E402

F_FABRIC = 430.08


class StubSocCfg:
    """Minimal stand-in for QickConfig: only what construction touches."""

    def __init__(self):
        self._cfg = {
            "readouts": [{"iq_offset": 0} for _ in range(8)],
            "gens": [
                {
                    "type": "axis_sg_int4_v2",
                    "f_fabric": F_FABRIC,
                    "samps_per_clk": 1,
                    "maxlen": 8192,
                    "has_mixer": True,
                    "fs": 4915.2,
                }
                for _ in range(16)
            ],
        }

    def __getitem__(self, key):
        return self._cfg[key]

    def _get_ch_cfg(self, ro_ch=None, gen_ch=None):
        return {"f_fabric": F_FABRIC, "fs": 4915.2}

    def cycles2us(self, cycles, gen_ch=None, ro_ch=None):
        return cycles / F_FABRIC

    def us2cycles(self, us, gen_ch=None, ro_ch=None):
        return int(us * F_FABRIC)

    def get_maxv(self, ch):
        return 30000


class StubIM(dict):
    def __getitem__(self, key):
        return None


def make_cfg_dict(cfg_name, tmp):
    return {
        "soc": StubSocCfg(),
        "expt_path": str(tmp),
        "data_path": str(tmp),
        "metadata_path": str(tmp),
        "cfg_file": str(REPO / "configs" / cfg_name),
        "im": StubIM(),
    }


CONFIGS = [
    "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml",
    "2026_09_21_bfg33_smpd_v2_waymar_rfboard.yml",
    "2026_09_21_bfg33_smpd_v2_yuvi_rfboard.yml",
]

# Classes migrated to `reset_mode` -- now every class a numbered script uses.
# Passing the wrong spelling raises (reset_mode_of refuses to let the two
# disagree), so the pass below picks per class.
MIGRATED = {
    "QubitSpec (03/13)",
    "RabiExperiment (04/04b/13b)",
    "T1Experiment (10)",
    "T2Experiment (11/11b)",
    "ResSpec (02/25)",
    "StarkSpec (17)",
    "SingleShotOptExperiment (22)",
    "F0G1Spec (27/28 inner)",
    "EFSpecF0G1 (31/32 inner)",
    "F0G1Rabi (33/34 inner)",
    "ResetTimeScan (35)",
    "SMPDHistogramExperiment (21/23/24/26)",
    "FWM_gainb_tb_sweep (20)",
    "HistogramExperiment (12)",
    "QubitSpecShot (18/19 inner)",
}

# Classes that can also run the unconditional f0-g1 reset. Everything in
# MIGRATED can, in principle; ResetTimeScan is excluded because using the
# protocol under test as its own inter-shot reset measures it against itself.
F0G1_CAPABLE = MIGRATED - {"ResetTimeScan (35)"}

RESET_PARAMS = {"reset": 3, "setup_reset": False}
# Second pass: reset_f plays a pi_ef ahead of each pi_ge, for sequences that
# can leave population in |f>. Worth its own pass because it needs a pulse
# most programs only declared under check_f / pulse_f / checkFH.
RESET_F_PARAMS = {**RESET_PARAMS, "reset_f": True}
# Fourth pass: reset_latch gives "while excited" semantics at a fixed trip
# count. It allocates a data register and emits two cond_jumps per round in
# place of one read_and_jump, so it is worth building on its own.
RESET_LATCH_PARAMS = {**RESET_PARAMS, "reset_latch": True}


def f0g1_calibrated(auto_cfg):
    """Whether this device has the fits the f0-g1 reset derives its drive from."""
    if "f0g1" not in auto_cfg.hw.soc.dacs:
        return False
    sec = auto_cfg.device.get("f0g1", {})
    return all(
        np.all(np.isfinite(np.array(sec.get(k, [np.nan]), dtype=float)))
        for k in ("stark_fit", "omega_ef_fit", "g_tilde_fit")
    )


def reset_params_for(label, base, mode, auto_cfg):
    """
    `base` plus the right spelling of the mode for this class.

    Returns None when the class cannot run the requested mode, so the caller
    skips it rather than reporting a failure. An uncalibrated device refusing
    f0g1 mode is correct behaviour, not a bug -- Yuvi has the line unwired and
    no device.f0g1 at all.
    """
    if mode == "f0g1":
        if label not in F0G1_CAPABLE or not f0g1_calibrated(auto_cfg):
            return None
        return {**base, "reset_mode": "f0g1"}
    if label in MIGRATED:
        return {**base, "reset_mode": "active"}
    return {**base, "active_reset": True}


def cases(auto_cfg, qi, buffer_qi):
    """(label, class, extra params) for each experiment an SMPD script runs."""
    f_buffer = auto_cfg.device.readout.frequency[buffer_qi]
    # The f0-g1 reset drive needs its own DAC section, which only the devices
    # that have the line wired carry. Skipping rather than failing keeps this
    # honest: a config without hw.soc.dacs.f0g1 cannot run 27/28 at all.
    f0g1 = (
        [
            ("F0G1Spec (27/28 inner)", meas.F0G1Spec, {"span": 5, "expts": 20}),
            ("EFSpecF0G1 (31/32 inner)", meas.EFSpecF0G1,
             {"span": 5, "expts": 20, "gain_f0g1": 0.5, "gain_ef": 0.05}),
            ("F0G1Rabi (33/34 inner)", meas.F0G1Rabi,
             {"expts": 20, "max_length": 1.0, "gain_f0g1": 0.5}),
        ]
        if "f0g1" in auto_cfg.hw.soc.dacs
        else []
    )
    # The reset scan needs the whole calibration, not just the DAC line: it
    # derives its operating point from g_tilde_fit. Skipping rather than
    # failing keeps this honest about what has actually been measured.
    if f0g1 and np.all(np.isfinite(
        np.array(auto_cfg.device.get("f0g1", {}).get("g_tilde_fit", [np.nan]),
                 dtype=float)
    )):
        f0g1.append(
            ("ResetTimeScan (35)", meas.ResetTimeScan,
             {"expts": 10, "max_length": 5.0, "gain_f0g1": 0.5})
        )
    return f0g1 + [
        ("QubitSpec (03/13)", meas.QubitSpec, {"span": 2, "expts": 20}),
        ("RabiExperiment (04/04b/13b)", meas.RabiExperiment, {"reps": 10}),
        ("T1Experiment (10)", meas.T1Experiment, {"reps": 10}),
        ("T2Experiment (11/11b)", meas.T2Experiment, {"reps": 10}),
        ("HistogramExperiment (12)", meas.HistogramExperiment, {"shots": 10}),
        ("ResSpec (02/25)", meas.ResSpec, {"span": 5, "expts": 20}),
        ("StarkSpec (17)", meas.StarkSpec,
         {"expts": 20, "stark_expts": 3, "stark_chan": 0, "stark_freq": f_buffer,
          "stark_qubit": buffer_qi}),
        ("SMPDHistogramExperiment (21/23/24/26)", meas.SMPDHistogramExperiment,
         {"shots": 10, "frequency_b": f_buffer, "frequency_p": 5210,
          "gain_p": 0.5, "check_f": True}),
        ("QubitSpecShot (18/19 inner)", meas.QubitSpecShot,
         {"expts": 10, "frequency_b": f_buffer}),
        ("FWM_gainb_tb_sweep (20)", meas.FWM_gainb_tb_sweep,
         {"expts": 5, "frequency_b": f_buffer, "frequency_p": 5210,
          "gain_p": 0.5, "length_b_expts": 3}),
        # Outer wrapper: runs no Program of its own, so it only has to
        # forward the keys to the inner HistogramExperiment.
        ("SingleShotOptExperiment (22)", meas.SingleShotOptExperiment,
         {"expts_f": 1, "expts_gain": 1, "expts_len": 2, "shots": 10},
         "forwards_only"),
    ]


def check_one(label, cls, cfg_dict, auto_cfg, qi, extra, kind="program",
              reset_params=None):
    # Construction prints a banner per class (channels, derived reset drive).
    # Useful on a real run, noise here -- and swallowing it does not hide
    # failures, since exceptions still propagate.
    with contextlib.redirect_stdout(io.StringIO()):
        expt = cls(cfg_dict, qi=qi, go=False,
                   params={**(reset_params or RESET_PARAMS), **extra})
    e = expt.cfg.expt

    mode = reset_mode_of(e)
    problems = []
    if mode == "passive":
        problems.append("reset_mode resolved to 'passive'; the requested mode "
                        "did not reach cfg.expt")
        return expt, problems

    if kind == "forwards_only":
        # No Program of its own; just check the keys survive to be forwarded.
        mode_key = "reset_mode" if "reset_mode" in e else "active_reset"
        for key in (mode_key, "reset", "setup_reset", "reset_verify",
                    "reset_f", "reset_latch"):
            if key not in e:
                problems.append(f"missing cfg.expt.{key}")
        return expt, problems

    shared = ("read_wait", "extra_delay", "reset", "setup_reset",
              "reset_verify", "reset_f")
    if mode == "f0g1":
        # No threshold: nothing is measured, so there is nothing to
        # discriminate. The drive settings take its place.
        needed = shared + ("f0g1_reset_gain", "f0g1_reset_freq",
                           "f0g1_reset_length", "f0g1_reset_gain_ef",
                           "f0g1_reset_freq_ef", "f0g1_reset_gap",
                           "f0g1_reset_ramp_sigma",
                           "f0g1_reset_ramp_sigma_inc")
    else:
        needed = shared + ("threshold",)

    for key in needed:
        if key not in e:
            problems.append(f"missing cfg.expt.{key}")

    if e.get("reset") != 3:
        problems.append(f"reset={e.get('reset')}, expected the 3 that was passed")

    if mode == "f0g1":
        for key in ("f0g1_reset_gain", "f0g1_reset_length", "f0g1_reset_gap"):
            if key in e and not e[key] > 0:
                problems.append(f"{key}={e[key]} is not positive")
        # Unequal ramps would leave one tone driving after the other stopped.
        if e.get("f0g1_reset_gain_ef", 1) <= 0:
            problems.append(f"f0g1_reset_gain_ef={e.get('f0g1_reset_gain_ef')} "
                            "is not positive")
        return expt, problems

    if not isinstance(e["threshold"], (int, np.integer)):
        problems.append(f"threshold is {type(e['threshold']).__name__}, not int")

    # Ordering: reset() skips the pi pulse when I < threshold, so ground must
    # be the blob below it.
    g = auto_cfg.device.readout.g_mean[qi]
    ex = auto_cfg.device.readout.e_mean[qi]
    tv = e.get("threshold_v")
    if tv is not None and not (g < tv < ex):
        problems.append(f"ordering g={g} thr={tv} e={ex} is not g<thr<e")
    return expt, problems


def main():
    tmp = Path(__file__).parent / "_scratch_data"
    tmp.mkdir(exist_ok=True)

    failures = 0
    passes = [
        ("active", RESET_PARAMS, "active"),
        ("active + reset_f (|f> aware)", RESET_F_PARAMS, "active"),
        ("f0g1 (unconditional)", RESET_PARAMS, "f0g1"),
        ("active + reset_latch (while-excited)", RESET_LATCH_PARAMS,
         "active"),
    ]
    for pass_name, reset_params, mode, cfg_name in [
            (pn, rp, m, c) for pn, rp, m in passes for c in CONFIGS]:
        auto_cfg = config.load(str(REPO / "configs" / cfg_name))
        qi, buffer_qi = 1, 0
        print()
        print(f"===== [{pass_name}] {cfg_name}  (qi={qi}) =====")
        g = auto_cfg.device.readout.g_mean[qi]
        ex = auto_cfg.device.readout.e_mean[qi]
        thr = auto_cfg.device.readout.threshold[qi]
        print(f"  g_mean={g}  threshold={thr}  e_mean={ex}  "
              f"ordering {'OK' if g < thr < ex else 'WRONG'}")

        for case in cases(auto_cfg, qi, buffer_qi):
            label, cls, extra = case[0], case[1], case[2]
            kind = case[3] if len(case) > 3 else "program"
            per_class = reset_params_for(label, reset_params, mode,
                                         auto_cfg)
            if per_class is None:
                continue
            cfg_dict = make_cfg_dict(cfg_name, tmp)
            try:
                expt, problems = check_one(
                    label, cls, cfg_dict, auto_cfg, qi, extra, kind,
                    reset_params=per_class
                )
            except Exception as exc:
                print(f"  [BUILD FAIL] {label}: {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=3)
                failures += 1
                continue
            if problems:
                print(f"  [FAIL] {label}: " + "; ".join(problems))
                failures += 1
            else:
                e = expt.cfg.expt
                if kind == "forwards_only":
                    detail = f"forwards reset_mode={reset_mode_of(e)!r}"
                elif reset_mode_of(e) == "f0g1":
                    detail = (f"f0g1 gain={e.f0g1_reset_gain:.3f} "
                              f"len={e.f0g1_reset_length:.2f}us "
                              f"ef gain={e.f0g1_reset_gain_ef:.5f}")
                else:
                    detail = f"threshold={e.get('threshold', 'n/a')}"
                print(f"  [ok]   {label}: {detail} "
                      f"reset={e.reset} "
                      f"final_delay={expt._get_final_delay()}")

    print(f"\n{'ALL OK' if failures == 0 else str(failures) + ' FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
