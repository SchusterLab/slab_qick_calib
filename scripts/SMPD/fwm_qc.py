"""
Per-point quality control for the FWM histogram grid, plus the NEP reduction.

`23_fwm_efficiency_vs_power.py` takes 130 `SMPDHistogramExperiment` points and
checks nothing. A fridge, JPA or qubit excursion partway through an overnight
run corrupts a block of points silently, and it is only found hours later when
`fit_shared_means.py` is run offline. This module is what
`26_fwm_efficiency_vs_power_checked.py` uses to catch that at the point it
happens, while the point can still be retaken.

Every one of the five scans is checked. The three reference scans are the
strongest canaries, because each is a *repeated identical measurement* over
part of the grid -- not just a repeated number, a repeated distribution:

    n  no drives     identical at every point in the grid
    p  no buffer     identical across the inner buffer-gain loop; this is the
                     dark count, which is what sets the NEP
    g  no pump       depends on buffer gain only, so identical at the same
                     buffer gain across pump powers

The `e` scan has no such twin, but it does have strong internal consistency
requirements: it cannot sit below its own backgrounds, and it must equal `p`
where the buffer gain is zero, since a zero-amplitude buffer pulse makes those
two sequences identical.

`f` is deliberately NOT gated. It reads the same conversion on the g/f axis, so
p_f/p_e looks like it should be a constant of the run -- and on run 33 #671 it
is not. It ran 1.34 at the lowest conversion down to 1.04 at the highest, the
small-signal limit being the ratio of the two scans' *backgrounds* rather than
of their signals; and at *matched* conversion it was 1.30-1.37 for pump gain
>= 0.8 against 1.04-1.08 below it, with the readout provably stable throughout
(width identical at every point, peak within one bin). Both are real structure,
not instability, so a trailing-median gate on it fired on physics. It is still
logged, and still worth watching: f beating e by more than the pi_ef and the
readout contrast allow means the *e* readout is losing population that f
retains, which would make the e-scan efficiency under-report.

Nothing here fits a model, and that is deliberate. Two model-based approaches
were tried against run 33 #138 and both failed on that data:

  * Pinning `vg`, `ve`, `sigma`, `t_ratio` from `device.readout` and
    projecting, as `scripts/SMPD/CLAUDE.md` describes. The stored calibration
    was a full sigma away from that run's own blobs, which biased the `e`-scan
    fractions by -0.048 against `fit_shared_means.csv` and left residuals large
    enough that a shape gate would have measured the staleness of the config
    rather than the quality of the data.
  * Fitting those four shared parameters on the run's leading points, the way
    `fit_shared_means.py` fits them over a whole run at once. The grid starts
    at its lowest pump and buffer gain, where every scan is nearly all ground,
    so `ve` and the decay tail are unconstrained: unbounded the fit walked
    `t_ratio` to 31 and called every fraction 1.0, and bounded it sat on the
    bound with `ve` swinging from 13 to 21 depending on how many leading points
    it saw.

What replaces it needs no levels of its own. The monitor statistic is the
thresholded excited fraction -- shots past `device.readout.threshold`, which is
how every other experiment in this tree computes `p_e`.

Every gate is an absolute bar with no history. Differential gates against a
trailing median were tried and removed: they missed a slow ramp (each step
inside the tolerance) while banking every bad point into the reference, and
then blocked recovered data against the poisoned baseline. See CLAUDE.md
"Why the gates are absolute" for the run-34 #621 evidence.

The fraction is biased low by the T1 decay the threshold splits, exactly as
`p_e` is everywhere else in this tree. That does not affect the gates -- the
bars are calibrated on fractions computed the same way -- and
the end-of-run NEP summary is labelled indicative for the same reason.
`fit_shared_means.py` remains the authority on the final numbers.

Run directly to check a saved run and calibrate the gate tolerances against
data already known to be good:

    python fwm_qc.py --data-dir Data/run33_Waymar/data/#138_FWM_efficiency_vs_power_q1_091826
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent
for _path in (str(REPO_DIR), str(REPO_DIR.parent)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from slab_qick_calib.exp_handling.datamanagement import SlabFile
from slab_qick_calib.experiments.single_qubit.t2 import ramsey_photons_per_rate
from slab_qick_calib.helpers import config

# Device config supplying the reference frame (g_mean / e_mean / sigma), named
# the same way the runner scripts name theirs. Override with --config.
CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"

# device.readout index of the buffer line in the SMPD scripts, and of the waste
# line the qubit is actually read out on.
BUFFER_LINE = 0
READOUT_LINE = 1

# 'n' and 'b' are listed so runs taken before no-drives moved into the 'g'
# slot still load; CLAUDE.md "The detuned-pump scan".
SCANS = ("g", "e", "p", "d", "b", "n", "f", "w")

# The no-drives scans, gated against an absolute floor. 'n' is the legacy key.
# 'w' is deliberately absent: the T1 wait only reaches thermal equilibrium,
# which f0g1 can cool below, so p_w may sit at or above the bar legitimately
# (CLAUDE.md "The passive-wait scan").
FLOOR_SCANS = ("g", "n")
# Pump-on, buffer-off: legitimately above the floor, so a looser bar.
PUMP_BACKGROUND_SCANS = ("p", "d")

# Histogram grid, computed once per run from the config levels and then never
# changed, so a peak offset or a profile distance means the same thing at every
# point. `fit_shared_means.load_run` can derive its grid from a whole run at
# once; a live sweep only ever has one point in hand.
QC_BINS = 120
GRID_PAD_SIGMA = 6.0

# Half-width of the window the ground blob is located in, in units of sigma.
# Wide enough to hold it after a drift worth flagging, narrow enough to exclude
# the excited blob, which is ~5.8 sigma away in the run-33 config.
GROUND_WINDOW_SIGMA = 2.5

# --- Gate tolerances -------------------------------------------------------
# Calibrated offline against the 130 points of run 33 #138. Rerun
# `python fwm_qc.py --data-dir ...` after any change to the readout to check
# they still hold.

# Ground-peak offset from the config's g_mean, in sigma, and ground-blob width
# as a fraction of the config's sigma. These are absolute, so they work from
# the first point of a run; the config is only a coarse reference frame here,
# which is all that is needed to measure a drift in units of sigma.
# Loose because argmax on a 0.2 sigma grid jitters +/-0.11 sigma and the floor
# gate is 4x tighter in the only direction that matters (CLAUDE.md "The peak
# gate is a gross-excursion catch").
PEAK_TOL = 1.25
WIDTH_LO, WIDTH_HI = 0.7, 1.5

# The pump-on background bar, 2x the floor bar. Pooled over run 34, p_p has
# median 0.031 and p99 0.069, so 0.10 rejects 0.4% of previously accepted
# points while catching #621's excursion at 0.15.
PUMP_MAX_PE = 0.10
K_SIGMA = 5.0

# Ordering and the zero-buffer anchor compare two different scans rather than
# one scan against itself, so they carry both scans' wobble and need their own,
# looser floor. At the bottom of the buffer range all five scans coincide to
# within the ~0.02 the backgrounds move by anyway.
ORDER_TOL = 0.03


# --- Ground-state recovery (CLAUDE.md "Ground-state recovery") -------------
# A failed point blocks the sweep until a ground-only readout gets back under
# GROUND_MAX_PE. 5000 shots give sigma = 0.0031 at p = 0.05, so the bar sits
# ~16 sigma off a healthy 0.03 floor and cannot be crossed by counting noise.
GROUND_MAX_PE = 0.05
GROUND_SHOTS = 5000
GROUND_SETTLE = 5.0
GROUND_WARN_EVERY = 10
# Consecutive passes required: one pass can be a fluctuation, or the low half
# of something oscillating.
GROUND_CONSECUTIVE = 3

# Copied from the sweep's own params so the check prepares the qubit the same
# way; config defaults would clear more completely and hide the problem.
GROUND_RESET_KEYS = ("reset_mode", "reset", "read_wait", "extra_delay",
                     "f0g1_reset_gap", "f0g1_reset_length", "f0g1_reset_gain",
                     "f0g1_reset_gain_ef", "f0g1_reset_residual",
                     "reset_f", "reset_latch")

# Which powers of buffer gain each buffer_power_fit_type tag stores, in the
# order the coefficients appear. Must match GAIN_FIT_POWERS in the #351 Stark
# folder's refit_two_peak.py, which writes them.
GAIN_FIT_POWERS = {
    "quadratic": (2,),
    "linear_quadratic": (1, 2),
    "constant_linear_quadratic": (0, 1, 2),
    "constant_quadratic": (0, 2),
}

# Efficiency is the *initial* slope of excited fraction against arriving
# photons: by the top of the buffer range the qubit is saturated -- many
# photons arrive but it can only report one excitation -- so a slope over the
# whole range would measure the saturation instead.
EFFICIENCY_SCANS = ("e", "f")
EFFICIENCY_POINTS = 5

# The dark-count scan: pump on, buffer off, so the detector is armed with no
# signal and every click is a false positive. Not the pump-off scan, which is
# the detector switched off and cannot produce a dark count at all.
DARK_SCAN = "p"

PLANCK = 6.62607015e-34  # J s


# --------------------------------------------------------------------------
# Reading one point
# --------------------------------------------------------------------------


class Reference:
    """
    The fixed reference frame a run's histograms are measured in.

    Not a fit and not claimed to be exact: `vg` and `sigma` only set the
    window the ground blob is looked for in and the units a drift is reported
    in, and `threshold` is the discriminator every other experiment in this
    tree already uses for `p_e`.
    """

    def __init__(self, vg, sigma, threshold, shots):
        self.vg = float(vg)
        self.sigma = float(sigma)
        self.threshold = float(threshold)
        self.shots = int(shots)

    def __repr__(self):
        return (f"Reference(vg={self.vg:.4f}, sigma={self.sigma:.4f}, "
                f"threshold={self.threshold:.4f}, shots={self.shots})")


def reference_from_config(auto_cfg, qi, shots):
    """Read the reference frame out of a loaded device config."""
    readout = auto_cfg.device.readout
    for key in ("g_mean", "sigma", "threshold"):
        if key not in readout:
            raise ValueError(
                f"device.readout.{key} is missing; run 12_single_shot.py with "
                "update() -- the QC checks are measured against it."
            )
    return Reference(
        vg=readout["g_mean"][qi],
        sigma=readout["sigma"][qi],
        threshold=readout["threshold"][qi],
        shots=shots,
    )


def bin_grid(reference, bins=QC_BINS):
    """
    Shared histogram grid, wide enough to hold both blobs after a drift.

    Spans the ground level to well past the excited one. Sized from `sigma`
    rather than from `e_mean` so that a stale excited level cannot clip the
    grid, which would break the density normalization.
    """
    lo = reference.vg - GRID_PAD_SIGMA * reference.sigma
    hi = reference.vg + (GRID_PAD_SIGMA + 12.0) * reference.sigma
    edges = np.linspace(lo, hi, bins + 1)
    return edges, (edges[:-1] + edges[1:]) / 2


def rotated_shots(data, scan):
    """
    One scan's shots on the discriminating axis.

    The stored rotation angle is reapplied, as `fit_shared_means` does; it is 0
    for this data because `SMPDHistogramExperiment` analyses with
    `rotate=False`, the readout phase having already been calibrated.
    """
    theta = np.deg2rad(float(np.atleast_1d(data.get("angle", 0.0))[0]))
    i = np.asarray(data[f"I{scan}"], dtype=float)
    q = np.asarray(data[f"Q{scan}"], dtype=float)
    return i * np.cos(theta) - q * np.sin(theta)


def ground_peak_and_width(centers, hist, reference):
    """
    Position and width of the ground blob, model-free.

    Restricted to a window around the reference `vg` rather than taken from the
    whole histogram, because at the top of the pump range the `e` scan's global
    maximum is the *excited* blob -- an unrestricted argmax would report that as
    a drifted ground level at exactly the points where conversion works best.

    Width is the half width at half maximum converted to a Gaussian sigma,
    which needs no fit and is unaffected by anything outside the window.
    """
    window = np.abs(centers - reference.vg) < GROUND_WINDOW_SIGMA * reference.sigma
    if not window.any():
        return float("nan"), float("nan")
    x, y = centers[window], hist[window]
    imax = int(np.argmax(y))
    peak, ymax = float(x[imax]), float(y[imax])
    if ymax <= 0:
        return peak, float("nan")
    # Walk out from the peak to the last bin still at or above half maximum on
    # each side. Contiguous by construction, so a second blob inside the window
    # cannot widen the answer.
    above = y >= ymax / 2.0
    left = imax
    while left > 0 and above[left - 1]:
        left -= 1
    right = imax
    while right < len(y) - 1 and above[right + 1]:
        right += 1
    fwhm = float(x[right] - x[left])
    if fwhm <= 0:
        return peak, float("nan")
    return peak, fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def analyze_point(data, edges, centers, reference):
    """
    Reduce one point's scans to a histogram and a few numbers each.

    Histogrammed on the shared grid from the raw I shots, not from the
    `hist<label>` arrays `fwm_shots.analyze()` stores -- those are binned over
    each scan's own range, which is not comparable between scans or between
    points.

    Returns {'centers', 'hist', 'metrics'}, the last holding
    {p, peak_offset, width_ratio} per scan label.
    """
    hists, metrics = {}, {}
    for scan in SCANS:
        if f"I{scan}" not in data:
            continue
        shots = rotated_shots(data, scan)
        hist = np.histogram(shots, bins=edges, density=True)[0]
        peak, width = ground_peak_and_width(centers, hist, reference)
        hists[scan] = hist
        metrics[scan] = {
            # The same discriminator every other experiment in this tree uses
            # for p_e: count what landed past the stored threshold.
            "p": float(np.mean(shots > reference.threshold)),
            "peak_offset": (peak - reference.vg) / reference.sigma,
            "width_ratio": width / reference.sigma,
        }
    return {"centers": centers, "hist": hists, "metrics": metrics}


# --------------------------------------------------------------------------
# Ground-state recovery
# --------------------------------------------------------------------------


def make_ground_check(cfg_dict, qi, params, shots=GROUND_SHOTS):
    """
    A |g>-only HistogramExperiment for `wait_for_ground`. Never saved.

    Built once and re-acquired; see CLAUDE.md "Ground-state recovery" for why
    the reset params are inherited from the sweep rather than re-derived.
    """
    import slab_qick_calib.experiments as meas

    reset = {k: v for k, v in params.items() if k in GROUND_RESET_KEYS}
    return meas.HistogramExperiment(
        cfg_dict, qi=qi, go=False, check_f=False,
        params={**reset, "shots": int(shots), "check_e": False},
    )


def ground_excited_fraction(check, reference):
    """
    One |g>-only acquisition, as the fraction past the readout threshold.

    `analyze()` is deliberately not called -- it needs Ie and raises without it
    (CLAUDE.md "Ground-state recovery").
    """
    from slab_qick_calib.experiments.general.qick_experiment import reset_mode_of

    data = check.acquire(progress=False)
    i = np.asarray(data["Ig"], dtype=float).flatten()
    q = np.asarray(data["Qg"], dtype=float).flatten()
    # HistogramProgram only puts device.readout.phase on the pulse when the
    # reset is not passive, so rotating always would double-rotate those.
    if reset_mode_of(check.cfg.expt) == "passive":
        theta = np.deg2rad(float(check.cfg.device.readout.phase[
            check.cfg.expt.qubit[0]]))
        i = i * np.cos(theta) - q * np.sin(theta)
    return float(np.mean(i > reference.threshold))


def wait_for_ground(check, reference, settle=GROUND_SETTLE,
                    consecutive=GROUND_CONSECUTIVE, verbose=True):
    """
    Block until a |g>-only readout is under GROUND_MAX_PE `consecutive` times.

    Unbounded by design. The streak resets on any failure, and the settle runs
    between every check so the streak spans time rather than being one burst of
    shots. Returns a dict of the `GROUND_WAIT_COLUMNS` fields.
    """
    import time

    start = time.time()
    checks = streak = failures = 0
    first = None
    while True:
        if checks:
            time.sleep(settle)
        p_e = ground_excited_fraction(check, reference)
        checks += 1
        if first is None:
            first = p_e
        if p_e < GROUND_MAX_PE:
            streak += 1
            if streak >= consecutive:
                if failures and verbose:
                    print(f"  ground state back under {GROUND_MAX_PE}: "
                          f"{streak} in a row after {failures} failed check(s), "
                          f"{time.time() - start:.0f} s")
                return {"checks": checks, "failures": failures,
                        "elapsed_s": round(time.time() - start, 1),
                        "p_e_first": round(first, 5),
                        "p_e_final": round(p_e, 5)}
        else:
            streak = 0
            failures += 1
            if verbose and failures % GROUND_WARN_EVERY == 0:
                print(f"  !! ground state {p_e:.4f} >= {GROUND_MAX_PE} -- still "
                      f"waiting after {failures} failed checks, "
                      f"{(time.time() - start) / 60:.1f} min")


GROUND_WAIT_COLUMNS = ["index", "attempt", "checks", "failures", "elapsed_s",
                       "p_e_first", "p_e_final"]


# --------------------------------------------------------------------------
# The gates
# --------------------------------------------------------------------------


def total_variation(hist, other, edges):
    """
    Total variation distance between two density histograms, in [0, 1].

    The fraction of the distribution that would have to be moved to turn one
    into the other, so it reads directly as "how different in shape" without
    reference to any model. Sensitive to a shifted blob, a broadened one, and a
    changed population alike -- which is the point, since any of the three
    means the point should be retaken.
    """
    width = float(edges[1] - edges[0])
    return 0.5 * float(np.sum(np.abs(np.asarray(hist) - np.asarray(other)))) * width


def _shot_sigma(p, shots):
    if shots <= 0:
        return 0.0
    return float(np.sqrt(max(p * (1.0 - p), 0.0) / shots))


def _tol(floor, p, shots):
    """A calibrated floor, or K standard errors if the shot count is low."""
    return max(floor, K_SIGMA * _shot_sigma(p, shots))


def _key(gain):
    """Grid gains come from np.linspace and repeat exactly, but round anyway."""
    return round(float(gain), 9)


class QCMonitor:
    """
    Decides whether a point is good enough to keep, given the grid so far.

    Only accepted points enter the history, so one bad point cannot widen the
    reference and let its neighbours through. Gates whose reference does not
    exist yet -- the first pump power for `g`, the first buffer gains of each
    pump power for `p` -- are skipped rather than failed.
    """

    def __init__(self, reference, edges):
        self.reference = reference
        self.edges = edges
        self.shots = reference.shots

    def check(self, metrics, hists, gain_p, gain_b):
        """
        Apply every gate. Returns (ok, failed) with failed a list of reasons.

        `metrics` and `hists` are the 'metrics' and 'hist' sub-dicts of
        `analyze_point`'s result.
        """
        failed = []

        # Shape, absolute: has the ground blob moved or broadened. Works from
        # the first point of the run, and applies to every scan -- all five
        # share a readout, so a readout excursion shows in all of them.
        for scan, m in metrics.items():
            if not np.isfinite(m["peak_offset"]) or abs(m["peak_offset"]) > PEAK_TOL:
                failed.append(f"peak:{scan} {m['peak_offset']:+.2f}sigma")
            if not WIDTH_LO < m["width_ratio"] < WIDTH_HI:
                failed.append(f"width:{scan} {m['width_ratio']:.2f}x")

        # Absolute floor: the no-drives scan should sit near the thermal
        # population whatever the sweep is doing. Absolute, not differential --
        # run 34 #621 ramped p_g 0.039 -> 0.160 in steps under the old
        # DRIFT_TOL and every one was banked (CLAUDE.md "Why the gates are
        # absolute").
        for scan in FLOOR_SCANS:
            if scan not in metrics:
                continue
            p_floor = metrics[scan]["p"]
            if p_floor >= GROUND_MAX_PE:
                failed.append(f"floor:{scan} {p_floor:.4f}>={GROUND_MAX_PE}")

        # The pump-on backgrounds sit legitimately above the floor, so they get
        # a looser bar of their own.
        for scan in PUMP_BACKGROUND_SCANS:
            if scan not in metrics:
                continue
            p_pump = metrics[scan]["p"]
            if p_pump >= PUMP_MAX_PE:
                failed.append(f"pump:{scan} {p_pump:.4f}>={PUMP_MAX_PE}")

        # Ordering and the zero-buffer anchor on the conversion scan.
        if "e" in metrics:
            p_e = metrics["e"]["p"]
            tol = _tol(ORDER_TOL, p_e, self.shots)
            backgrounds = {s: metrics[s]["p"]
                           for s in ("g", "p", "d", "b", "n") if s in metrics}
            if backgrounds:
                worst = max(backgrounds, key=backgrounds.get)
                if p_e < backgrounds[worst] - tol:
                    failed.append(
                        f"ordering:e {p_e:.4f} below {worst} "
                        f"{backgrounds[worst]:.4f} (tol {tol:.4f})")
            # With no buffer amplitude the pump+buffer sequence *is* the
            # pump-only one -- same pulses, same timing -- so this anchors the
            # whole inner loop against the dark-count scan.
            if abs(gain_b) < 1e-12 and "p" in metrics:
                if abs(p_e - metrics["p"]["p"]) > tol:
                    failed.append(
                        f"anchor:e {p_e:.4f} != p {metrics['p']['p']:.4f} "
                        f"at gain_b=0 (tol {tol:.4f})")

        return not failed, failed


# --------------------------------------------------------------------------
# Logging the attempts
# --------------------------------------------------------------------------

def f_over_e(metrics):
    """
    p_f/p_e, the two readouts of one conversion, or None if either is missing.

    Reported only -- it is not gated, for the reasons in the module docstring.
    No floor on p_e either: the old one existed to keep a noisy ratio out of a
    tolerance, and reading this column alongside p_e in the same row is a
    better filter than a constant baked in here.
    """
    if "f" not in metrics or "e" not in metrics:
        return None
    p_e = metrics["e"]["p"]
    if p_e <= 0:
        return None
    return metrics["f"]["p"] / p_e


LOG_COLUMNS = (["index", "attempt", "gain_p", "gain_b", "status", "failed_gates"]
               + [f"p_{s}" for s in SCANS]
               + ["peak_g", "width_g", "ratio_fe", "fid_ge"])


def log_row(index, attempt, gain_p, gain_b, status, failed, result, data):
    """One `qc_log.csv` row per *attempt*, so the reject rate is recoverable."""
    metrics = result["metrics"]
    fids = np.atleast_1d(data.get("fids", [np.nan]))
    ratio = f_over_e(metrics)
    row = {
        "index": index,
        "attempt": attempt,
        "gain_p": f"{gain_p:.6f}",
        "gain_b": f"{gain_b:.6f}",
        "status": status,
        "failed_gates": "; ".join(failed),
        "peak_g": f"{metrics['g']['peak_offset']:.4f}" if "g" in metrics else "",
        "width_g": f"{metrics['g']['width_ratio']:.4f}" if "g" in metrics else "",
        "ratio_fe": "" if ratio is None else f"{ratio:.4f}",
        "fid_ge": f"{float(fids[0]):.4f}",
    }
    for scan in SCANS:
        m = metrics.get(scan)
        row[f"p_{scan}"] = "" if m is None else f"{m['p']:.5f}"
    return row


def record_reject(path, index, attempt, gain_p, gain_b, failed, result,
                  frequency_b=None):
    """
    Append a rejected attempt's histograms to `qc_rejects.h5`.

    The histograms, not the shots: five 120-bin curves is enough to see what
    the failure looked like and to replot it later, against 150 k shots that are
    of no further use once the point has been retaken. `frequency_b` is only
    worth passing from a sweep that varies it.
    """
    import h5py

    with h5py.File(str(path), "a") as h5:
        name = f"reject_{index:05d}_attempt{attempt}"
        if name in h5:
            del h5[name]
        group = h5.create_group(name)
        group.attrs["index"] = index
        group.attrs["attempt"] = attempt
        group.attrs["gain_p"] = float(gain_p)
        group.attrs["gain_b"] = float(gain_b)
        if frequency_b is not None:
            group.attrs["frequency_b"] = float(frequency_b)
        group.attrs["failed_gates"] = "; ".join(failed)
        group.create_dataset("centers", data=result["centers"])
        for scan, hist in result["hist"].items():
            group.create_dataset(f"hist{scan}", data=hist)
        for scan, m in result["metrics"].items():
            for key, value in m.items():
                group.attrs[f"{scan}_{key}"] = float(value)


# --------------------------------------------------------------------------
# Buffer gain -> photons, and the NEP reduction
# --------------------------------------------------------------------------


# Photons at a const buffer pulse relative to flat_top at the same gain: QICK
# plays a flat_top plateau at half the const amplitude. Same table as
# analysis/fit_shared_means.py.
PULSE_PHOTON_FACTOR = {"flat_top": 1.0, "const": 4.0}


def buffer_attenuation_db(auto_cfg):
    """
    Attenuation the RF board actually applies to the buffer DAC, in dB.

    attn1 + attn2 on the buffer line's own DAC channel, read from
    hw.soc.rfboard_active -- what was applied, not what a default claims. The
    Ramsey photon calibration is stated for one attenuation, so this is what
    rescales it onto the run: 10 dB is a factor of 10 in photon number, so
    getting it wrong dwarfs every other error in the reduction.

    Same lookup analysis/fit_shared_means.py does per saved file, so the live
    summary and the offline analysis land on the same photon axis.
    """
    soc = auto_cfg.hw.soc
    channel = soc.dacs.readout.ch[BUFFER_LINE]
    board = soc.get("rfboard_active", {}).get("dac_readout", {}).get(str(channel))
    if board is None:
        raise ValueError(
            f"No hw.soc.rfboard_active.dac_readout[{channel}] in the config, so "
            "the buffer attenuation is unknown. prepare_experiment() records it "
            "when it activates the buffer line -- additional_readout_qis has to "
            f"include {BUFFER_LINE}."
        )
    return float(board["attn1"]) + float(board["attn2"])


def load_calibration(auto_cfg, source="ramsey_spec", attn_db=None, pulse=None):
    """
    Read what is needed to turn a buffer gain into a photon count.

    Two pieces: the gain -> n_bar fit from device.fwm (with its type tag, so
    `photons` can refuse coefficients in a form it does not know), and the
    buffer's output coupling from device.readout.kappa_c, which converts a
    standing occupation into a flux.

    `source="ramsey_spec"` (default) is device.fwm.buffer_photon_ramsey_spec,
    on-resonance photons per gain^2 straight from script 41's dephasing
    spectrum, which fits its own kappa and chi. `source="stark"` is
    device.fwm.buffer_power_fit (refit_two_peak.py's PNS fit, which reads ~1.3x
    high on run 34). `"ramsey"` is built from device.fwm.buffer_dephasing_ramsey, the
    latest T2PhotonCal run's dephasing slope, converted to on-resonance
    photons with device.readout kappa and chi (the same kappa the flux step
    uses) at the drive detuning it was taken at (`_detuning`; untagged runs
    drove at the dressed midpoint). Both are stated for one buffer attenuation
    and pulse shape (`_attn_db` / `_pulse` tags) and are rescaled to the run's
    own `attn_db` (attn1 + attn2 actually applied to the buffer DAC) and
    `pulse` (pulse_type_b) -- exactly the rescale fit_shared_means.py applies,
    so the two report the same photon axis. A Stark fit written before
    refit_two_peak.py tagged it is used as stored.
    """
    fwm = auto_cfg.device.get("fwm", {})
    key = {"stark": "buffer_power_fit", "ramsey": "buffer_dephasing_ramsey",
           "ramsey_spec": "buffer_photon_ramsey_spec"}[source]
    if key not in fwm:
        raise ValueError(
            f"No device.fwm.{key} in the config. "
            + ("Run refit_two_peak.py in a Stark spectroscopy folder first -- "
               "it fits buffer gain -> photon number and writes it there."
               if source == "stark" else
               "Run script 41 (BufferDephasingSpec) with config updates on; its "
               "update() writes it there."
               if source == "ramsey_spec" else
               "Run script 38 (T2PhotonCal) with config updates on; its "
               "update() writes it there.")
        )
    readout = auto_cfg.device.readout
    kappa = float(readout["kappa"][BUFFER_LINE])
    if source == "ramsey":
        chi = float(readout["chi"][BUFFER_LINE])
        detuning = float(fwm.get(f"{key}_detuning", chi / 2))
        _, on_resonance = ramsey_photons_per_rate(kappa, chi, detuning)
        coeffs, fit_type = [float(fwm[key][0]) * on_resonance], "quadratic"
    elif source == "ramsey_spec":
        coeffs, fit_type = [float(fwm[key][0])], "quadratic"
    else:
        coeffs, fit_type = list(fwm[key]), fwm[f"{key}_type"]
    tagged = f"{key}_attn_db" in fwm and f"{key}_pulse" in fwm
    if source != "stark" and not tagged:
        raise ValueError(f"device.fwm.{key} has no _attn_db / _pulse tags.")
    scale, reference = 1.0, None
    if tagged:
        if attn_db is None or pulse not in PULSE_PHOTON_FACTOR:
            raise ValueError(
                f"The {source} calibration has to be rescaled to the run's buffer "
                f"attenuation and pulse shape; got attn_db={attn_db}, "
                f"pulse={pulse!r} (known shapes {sorted(PULSE_PHOTON_FACTOR)}).")
        reference = (float(fwm[f"{key}_attn_db"]), str(fwm[f"{key}_pulse"]))
        scale = (10 ** ((reference[0] - float(attn_db)) / 10)
                 * PULSE_PHOTON_FACTOR[pulse] / PULSE_PHOTON_FACTOR[reference[1]])
        # photons() is linear in the coefficients, so the rescale is exact.
        coeffs = [c * scale for c in coeffs]
    kappa_c = float(readout.get("kappa_c", [float("nan")])[BUFFER_LINE])
    if not np.isfinite(kappa_c):
        raise ValueError(
            f"No device.readout.kappa_c[{BUFFER_LINE}] in the config. It comes "
            "from a resonator spectroscopy complex fit of the buffer "
            "(complex_fit_kappa_c)."
        )
    if kappa_c > kappa:
        raise ValueError(
            f"device.readout.kappa_c[{BUFFER_LINE}] = {kappa_c} exceeds kappa = "
            f"{kappa}; a coupling rate cannot beat the total linewidth, so one "
            "of the two is from a stale fit."
        )
    return {
        "coeffs": coeffs,
        "fit_type": fit_type,
        "kappa": kappa,
        "kappa_c": kappa_c,
        "source": source,
        "scale": scale,
        # (attn_db, pulse) the fit is stated for, or None if it carries no
        # tags. Not the same thing as scale == 1.0, which a tagged calibration
        # also gives when the run happens to match its reference.
        "reference": reference,
    }


def photons(gain, coeffs, fit_type):
    """
    Mean photon number in the buffer at a given buffer gain.

    Which powers of gain the stored coefficients belong to comes from the type
    tag, so the Stark refit can change the form of its fit without this being
    silently misread.
    """
    if fit_type not in GAIN_FIT_POWERS:
        raise ValueError(
            f"Unknown device.fwm.buffer_power_fit_type {fit_type!r}; known "
            f"forms are {sorted(GAIN_FIT_POWERS)}."
        )
    powers = GAIN_FIT_POWERS[fit_type]
    if len(coeffs) != len(powers):
        raise ValueError(
            f"device.fwm.buffer_power_fit has {len(coeffs)} coefficients but "
            f"buffer_power_fit_type {fit_type!r} needs {len(powers)}."
        )
    gain = np.asarray(gain, dtype=float)
    return sum(c * gain ** power for c, power in zip(coeffs, powers))


def photons_incident(gain, length_b, calib):
    """
    Photons arriving down the buffer line during one buffer pulse.

    The Stark measurement gives `n_bar`, the *standing* occupation, reached
    within 2/kappa and then held. Over a 40 us pulse the buffer is not holding
    a photon, it is passing a stream, so the count that matters is a flux times
    a time. In steady state arrivals balance departures, and the buffer is
    driven in reflection through a single port, which gives

        n_bar = 4 * kappa_c * Phi_in / kappa**2
        N     = Phi_in * length_b = kappa**2 * n_bar * length_b / (4 kappa_c)

    with kappa and kappa_c as rates in rad/us, i.e. 2*pi times the MHz
    linewidths the config stores -- dropping that 2*pi moves the answer 6.3x.
    """
    n_bar = photons(gain, calib["coeffs"], calib["fit_type"])
    rate, rate_c = 2.0 * np.pi * calib["kappa"], 2.0 * np.pi * calib["kappa_c"]
    flux = rate ** 2 * n_bar / (4.0 * rate_c)
    return flux * np.asarray(length_b, dtype=float)


def n_bar_from_arrivals(arrivals, length_b, calib):
    """
    Inverse of `photons_incident`.

    Used to quote a dark-count probability as an equivalent buffer occupation:
    the standing n_bar a real signal would need in order to produce the same
    number of clicks.
    """
    rate, rate_c = 2.0 * np.pi * calib["kappa"], 2.0 * np.pi * calib["kappa_c"]
    return (np.asarray(arrivals, dtype=float) * 4.0 * rate_c
            / (rate ** 2 * np.asarray(length_b, dtype=float)))


def qubit_efficiency(t1, t_d):
    """
    Fraction of conversions that survive to be read out.

        eta_qubit = (T1/T_d) * (1 - exp(-T_d/T1))

    A photon can convert at any point in the detection window T_d and the
    resulting excitation then has to survive to readout; averaging exp(-t/T1)
    over a uniform arrival time in [0, T_d] gives the expression above. It caps
    what any measurement of this kind can report, so it belongs on the ceiling
    rather than as a correction to the data.
    """
    return (t1 / t_d) * (1.0 - np.exp(-t_d / t1))


def noise_equivalent_power(p_dark, eta, frequency_b, length_b, cycle):
    """
    NEP in W/sqrt(Hz), including the duty cycle of the pulse sequence.

        NEP = (h nu / (eta * eta_D)) * sqrt(2 Gamma_dark)

    with `eta_D = length_b / cycle` -- photons arriving while the detector is
    not armed are simply missed -- and `Gamma = p_dark / cycle` the observed
    wall-clock dark rate.

    Not proportional to the noise-equivalent n_bar, which goes as p_dark/eta
    rather than sqrt(p_dark)/eta: the square root is why NEP keeps improving
    with pump power after the equivalent n_bar has turned back up, and why the
    two cannot share a rescaled axis.
    """
    h_nu = PLANCK * frequency_b * 1e6          # frequency_b is in MHz
    duty = length_b / cycle
    gamma_dark = p_dark / (cycle * 1e-6)       # per second, wall clock
    return (h_nu / (eta * duty)) * np.sqrt(2.0 * gamma_dark)


def initial_slope(xs, ys):
    """
    Straight line through the points; returns (slope, standard error, intercept).

    The intercept is left free rather than forced through zero. A pump-only
    background is present at every buffer gain including the lowest, so with a
    fixed intercept it would tilt the slope; free, it lands in the intercept and
    the slope measures only the response to photons.
    """
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    design = np.column_stack([np.ones_like(xs), xs])
    (intercept, slope), *_ = np.linalg.lstsq(design, ys, rcond=None)
    resid = design @ np.array([intercept, slope]) - ys
    dof = max(len(xs) - 2, 1)
    cov = np.linalg.inv(design.T @ design) * float(resid @ resid) / dof
    return float(slope), float(np.sqrt(cov[1, 1])), float(intercept)


def reduce_to_nep(points, calib, t1, cycle_us, n_points=EFFICIENCY_POINTS,
                  by="gain_p", dynamics=False):
    """
    Efficiency, dark count, noise-equivalent n_bar and NEP at each pump gain.

    `points` is a list of dicts with keys gain_p, gain_b, length_b,
    frequency_b and `p` (the per-scan fractions of an accepted point), and
    optionally `cycle_us` (the sequence period, which a buffer-length sweep
    changes point to point; the `cycle_us` argument is the fallback when the
    points do not carry it). One row per distinct value of `by` -- the pump
    gain by default, the buffer frequency for a frequency sweep, the buffer
    length for a length sweep -- ordered by that value.

    Each row carries its own `length_b`, `cycle_us`, `eta_qubit` and
    `eta_ceiling`, since all four move with the buffer length. The returned
    `info` holds the whole-run medians, which is what a single-length caller
    wants.

    Call it once per pump gain with `by="length_b"` to get efficiency against
    buffer length, which is the comparison a length sweep is for: measured
    efficiency should fall like `eta_qubit(length_b)` if decay before readout
    is what limits it, and faster or slower if something else does.

    `dynamics` (script 40, `by="length_b"`) takes each length's efficiency
    from one `fit_window_dynamics` over every point instead of the initial
    slope, and returns those fits as `info["dynamics"]`; CLAUDE.md
    "Buffer-length sweep".

    Indicative, not authoritative: the fractions are thresholded rather than
    fitted, so they are biased low by however much of the T1 decay continuum
    falls on the ground side of the threshold. Run `fit_shared_means.py` on the
    saved folder for the numbers to quote.
    """
    rows = []
    # Global values, kept for the `info` dict that single-length callers read.
    t_d_all = float(np.median([pt["length_b"] for pt in points]))
    eta_q_all = qubit_efficiency(t1, t_d_all)
    fits = {}
    if dynamics:
        # Pump-off amplitude ring-up 2/kappa seeds tau.
        tau_guess = 1 / (np.pi * calib["kappa"])
        for scan in EFFICIENCY_SCANS:
            groups = {}
            for value in {_key(pt[by]) for pt in points}:
                at = [pt for pt in points if _key(pt[by]) == value and scan in pt["p"]]
                if len({pt["gain_b"] for pt in at}) >= 2:
                    groups[value] = (np.array([photons_incident(pt["gain_b"], pt["length_b"], calib)
                                               for pt in at], dtype=float),
                                     np.array([pt["p"][scan] for pt in at]))
            fit = fit_window_dynamics(groups, t1, tau_guess)
            if fit is not None:
                fits[scan] = fit
    for value in sorted({_key(pt[by]) for pt in points}):
        at_gain = [pt for pt in points if _key(pt[by]) == value]
        # Per group, not global: a buffer-length sweep moves both the
        # detection window and the sequence period, and both feed n_equiv and
        # the NEP. A single median over the whole run would smear a 5 us point
        # and a 200 us one into one number.
        t_d = float(np.median([pt["length_b"] for pt in at_gain]))
        # A length sweep's ceiling also rings up, at the matched Gamma = kappa (linewidth 2 kappa).
        eta_q = (float(ringup_efficiency(t_d, 1.0, t1, 1 / (2 * np.pi * calib["kappa"])))
                 if dynamics else qubit_efficiency(t1, t_d))
        cycles = [pt["cycle_us"] for pt in at_gain if "cycle_us" in pt]
        cycle = float(np.median(cycles)) if cycles else cycle_us
        arrivals = {id(pt): float(photons_incident(pt["gain_b"], pt["length_b"],
                                                   calib))
                    for pt in at_gain}
        lowest = sorted(at_gain, key=lambda pt: arrivals[id(pt)])[:n_points]
        row = {by: value, "n_points": len(at_gain), "length_b": t_d,
               "cycle_us": cycle, "eta_qubit": eta_q,
               "eta_ceiling": calib["kappa_c"] / calib["kappa"] * eta_q}
        for scan in EFFICIENCY_SCANS:
            if dynamics:
                eta, err = (fits[scan]["eta"].get(value, (np.nan, np.nan))
                            if scan in fits else (np.nan, np.nan))
                row[f"eta_{scan}"], row[f"eta_err_{scan}"] = eta, err
                continue
            usable = [pt for pt in lowest if scan in pt["p"]]
            if len(usable) < 2:
                row[f"eta_{scan}"] = float("nan")
                row[f"eta_err_{scan}"] = float("nan")
                continue
            slope, err, _ = initial_slope([arrivals[id(pt)] for pt in usable],
                                          [pt["p"][scan] for pt in usable])
            row[f"eta_{scan}"] = slope
            row[f"eta_err_{scan}"] = err
        dark = [pt["p"][DARK_SCAN] for pt in at_gain if DARK_SCAN in pt["p"]]
        row["p_dark"] = float(np.mean(dark)) if dark else float("nan")
        eta = row.get("eta_e", float("nan"))
        frequency_b = float(np.median([pt["frequency_b"] for pt in at_gain]))
        if np.isfinite(eta) and eta > 0 and np.isfinite(row["p_dark"]):
            # The dark count expressed as the arriving-photon count that would
            # produce it, then back to the standing occupation that supplies it.
            row["n_equiv"] = float(n_bar_from_arrivals(row["p_dark"] / eta,
                                                       t_d, calib))
            row["nep"] = float(noise_equivalent_power(
                row["p_dark"], eta, frequency_b, t_d, cycle))
        else:
            row["n_equiv"] = float("nan")
            row["nep"] = float("nan")
        rows.append(row)
    return rows, {"eta_qubit": eta_q_all, "t_d": t_d_all, "t1": t1,
                  "eta_ceiling": (calib["kappa_c"] / calib["kappa"]
                                  * eta_q_all), "dynamics": fits}


def plot_point(centers, hists, title=None, failed=()):
    """
    One point's five scan histograms, linear and log.

    Drawn from the histograms rather than the shots, so the same figure comes
    out of a live rejected attempt and out of `qc_rejects.h5` afterwards --
    which is the whole reason the rejects file stores curves. Deliberately not
    `SMPDHistogramExperiment.display()`: that needs the raw shots for its IQ
    panels, and a rejected attempt's shots are not kept.

    The log panel is where the reference scans are readable; on a linear scale
    they are a flat line at the bottom.
    """
    import matplotlib.pyplot as plt

    # The legend names and colours the experiment class already defines, so a
    # trace means the same thing here as on a script-21 figure. Imported
    # lazily to keep the offline statistics path off the experiments package.
    from slab_qick_calib.experiments import SMPDHistogramExperiment as _Expt

    colors = {"g": "tab:blue", "e": "tab:red", "p": "green",
              "d": "darkorange", "b": "tab:cyan", "n": "0.35",
              "f": "magenta", "w": "tab:purple"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for scan in SCANS:
        if scan not in hists:
            continue
        label = _Expt._SCAN_LABELS[scan]
        for plot in (axes[0].plot, axes[1].semilogy):
            plot(centers, hists[scan], lw=1.4, color=colors[scan], label=label)
    axes[0].set_ylabel("Probability")
    axes[1].set_ylabel("Probability (log)")
    # The shared grid is deliberately wider than any one point needs, so a
    # drift has somewhere to go; left untrimmed most of the figure is empty
    # axis. Trimmed by *mass* rather than by a density threshold, because one
    # stray shot clears any threshold low enough to keep the log panel's tails
    # and would hold the axis open to the full grid.
    width = float(centers[1] - centers[0])
    lo_idx, hi_idx = len(centers) - 1, 0
    for hist in hists.values():
        mass = np.cumsum(hist) * width
        inside = np.flatnonzero((mass > 5e-4) & (mass < 1 - 5e-4))
        if inside.size:
            lo_idx, hi_idx = min(lo_idx, inside[0]), max(hi_idx, inside[-1])
    if hi_idx > lo_idx:
        span = centers[hi_idx] - centers[lo_idx]
        for axis in axes:
            axis.set_xlim(centers[lo_idx] - 0.05 * span,
                          centers[hi_idx] + 0.05 * span)
    for axis in axes:
        axis.set_xlabel("I")
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize="small", loc="upper right")

    caption = title or ""
    if failed:
        # Wrapped: a point can fail a dozen gates at once and a single line
        # would run off the figure.
        reasons = "; ".join(failed)
        caption = "\n".join(part for part in (caption, reasons) if part)
    if caption:
        fig.suptitle(caption, fontsize="small")
    return fig


def plot_nep_summary(rows, info, by="gain_p", xlabel="pump gain", series=None):
    """
    The one figure the checked sweep draws: the four reduced quantities.

    Same four panels as the bottom row of `fit_shared_means_grid.png`, which
    remains the authoritative version -- these fractions are thresholded rather
    than fitted, so this is a check that the run produced something sane, not
    the final number.

    `series` overlays one curve per distinct value of that key -- a pump gain
    per curve against buffer length, say -- so a 3D sweep still fits in one
    figure. The efficiency ceiling is drawn per row when the rows carry
    `eta_ceiling` (it moves with the buffer length), otherwise as one line.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)
    groups = ([(None, rows)] if series is None else
              [(v, [r for r in rows if _key(r[series]) == v])
               for v in sorted({_key(r[series]) for r in rows})])
    shades = plt.cm.viridis(np.linspace(0, 0.85, max(len(groups), 1)))

    for (value, group), shade in zip(groups, shades):
        x = np.array([r[by] for r in group])
        order = np.argsort(x)
        x = x[order]
        tag = "" if value is None else f"{series} {value:g}"

        # With one series the two readouts are distinguishable by colour; with
        # several, colour has to carry the series instead, so only the e
        # readout is drawn and f is left to the CSV.
        scans = (("e", "tab:red"), ("f", "magenta")) if series is None else (("e", shade),)
        for scan, color in scans:
            eta = np.array([r.get(f"eta_{scan}", np.nan) for r in group])[order]
            err = np.array([r.get(f"eta_err_{scan}", np.nan) for r in group])[order]
            label = tag or f"{scan} readout"
            axes[0].errorbar(x, eta, yerr=err, marker="o", ms=4, color=color,
                             label=label)
        for axis, key, color in ((axes[1], "p_dark", "green" if series is None else shade),
                                 (axes[2], "n_equiv", "0.35" if series is None else shade),
                                 (axes[3], "nep", "tab:blue" if series is None else shade)):
            axis.semilogy(x, np.array([r[key] for r in group])[order], "o-",
                          ms=4, color=color, label=tag or None)

    # Per row where available: eta_qubit falls with the detection window, so a
    # length sweep has a ceiling that slopes rather than a single level.
    ceilings = [r["eta_ceiling"] for r in rows if "eta_ceiling" in r]
    if ceilings and len(set(np.round(ceilings, 6))) > 1:
        ref = sorted(((r[by], r["eta_ceiling"]) for r in rows), key=lambda t: t[0])
        axes[0].plot([a for a, _ in ref], [b for _, b in ref], color="0.5",
                     ls="--", label="ceiling")
    else:
        axes[0].axhline(info["eta_ceiling"], color="0.5", ls="--",
                        label=f"ceiling {info['eta_ceiling']:.3f}")

    axes[0].set_ylabel("efficiency per photon")
    axes[1].set_ylabel("dark count P(e), pump only")
    axes[2].set_ylabel(r"noise-equivalent $\bar{n}$")
    axes[3].set_ylabel(r"NEP (W/$\sqrt{\rm Hz}$)")
    axes[0].legend(fontsize="x-small", ncol=1 if series is None else 2)
    for axis in axes:
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.3)
    return fig


def ringup_efficiency(lengths, a, t, tau):
    """A * window average of ring-up (1 - e^(-t/tau))^2 x survival e^(-(L-t)/T); tau = 0 is A * qubit_efficiency(T, L)."""
    u = np.linspace(0.0, 1.0, 2001)
    L = np.asarray(lengths, dtype=float)[..., None]
    ring = (1.0 - np.exp(-u * L / max(tau, 1e-9))) ** 2
    return a * np.trapz(ring * np.exp(-(1.0 - u) * L / t), u, axis=-1)


def window_population(length, arrivals, a, t, tau):
    """P(e) at the window's end: dP/dt = r(1 - P) - P/T, r = a (N/L)(1 - e^(-t/tau))^2, so decay and re-excitation are in."""
    s = np.linspace(0.0, length, 1501)
    rate = (a * np.asarray(arrivals, dtype=float)[:, None] / length
            * (1.0 - np.exp(-s / max(tau, 1e-9))) ** 2)
    k = rate + 1.0 / t
    loss = np.concatenate([np.zeros((rate.shape[0], 1)),
                           np.cumsum((k[:, 1:] + k[:, :-1]) / 2 * np.diff(s), axis=1)], axis=1)
    return np.trapz(rate * np.exp(loss - loss[:, -1:]), s, axis=1)


def fit_window_dynamics(groups, t_guess, tau_guess):
    """
    One fit of window_population to every point of a length sweep; CLAUDE.md "Buffer-length sweep (script 40)".

    `groups` is {length: (arrivals, P)}. A, T, tau are shared, the background
    c is per length. Also returns eta {length: (eta, err)}, the linear-regime
    efficiency per arriving photon there, from refitting only A at that length
    with T, tau and c held. None if there is too little to fit.
    """
    from scipy.optimize import least_squares

    lengths = sorted(groups)
    n = sum(len(groups[L][1]) for L in lengths)
    if len(lengths) < 3 or n <= 3 + len(lengths):
        return None

    def model(a, t, tau, c, L):
        return c + (1 - c) * window_population(L, groups[L][0], a, t, tau)

    def resid(x):
        return np.concatenate([model(*x[:3], x[3 + i], L) - groups[L][1]
                               for i, L in enumerate(lengths)])

    def errors(res, dof):
        return np.sqrt(np.diag(np.linalg.pinv(res.jac.T @ res.jac))
                       * (res.fun @ res.fun) / max(dof, 1))

    x0 = [1.0, t_guess, tau_guess] + [float(np.clip(np.min(groups[L][1]), 0, 0.5))
                                      for L in lengths]
    res = least_squares(resid, x0, bounds=([0, 1e-3, 0] + [0] * len(lengths),
                                           [np.inf, np.inf, max(lengths)] + [1] * len(lengths)))
    err = errors(res, n - len(x0))
    a, t, tau = map(float, res.x[:3])
    c = dict(zip(lengths, map(float, res.x[3:])))
    eta = {}
    for L in lengths:
        one = least_squares(lambda x: model(x[0], t, tau, c[L], L) - groups[L][1], [a],
                            bounds=([0], [np.inf]))
        unit = float(ringup_efficiency(L, 1.0, t, tau))
        eta[L] = (float(one.x[0]) * unit, float(errors(one, len(one.fun) - 1)[0]) * unit)
    return {"A": a, "A_err": float(err[0]), "T": t, "T_err": float(err[1]),
            "tau": tau, "tau_err": float(err[2]), "c": c, "eta": eta,
            "rms": float(np.sqrt(np.mean(res.fun ** 2)))}


def plot_efficiency_vs_length(rows, t1_pump, dynamics, tau_ceiling):
    """
    Efficiency and efficiency x duty vs buffer length for one pump gain's rows.

    `dynamics` is reduce_to_nep's info["dynamics"]; the ceiling rings up with
    `tau_ceiling`, the matched 1/(2 pi kappa). Returns (fig, fits):
    fits[scan] is that fit plus L_peak, where the fitted efficiency x duty
    peaks (NaN if at a sweep edge).
    """
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["length_b"])
    L = np.array([r["length_b"] for r in rows])
    duty = L / np.array([r["cycle_us"] for r in rows])
    dead = float(np.median(L / duty - L))
    # From L -> 0, so the ring-up roll-off below the shortest length shows.
    grid = np.linspace(0, L.max(), 801)[1:]
    grid_duty = grid / (grid + dead)
    kappa_ratio = rows[0]["eta_ceiling"] / rows[0]["eta_qubit"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    fits = {}
    for scan, color in (("e", "tab:red"), ("f", "magenta")):
        eta = np.array([r[f"eta_{scan}"] for r in rows])
        err = np.array([r[f"eta_err_{scan}"] for r in rows])
        fit = dynamics.get(scan)
        good = fit is not None
        model = ringup_efficiency(grid, fit["A"], fit["T"], fit["tau"]) if good else grid * np.nan
        peak = int(np.argmax(model * grid_duty)) if good else 0
        if good:
            fits[scan] = {**fit, "L_peak": grid[peak] if 0 < peak < len(grid) - 1 else np.nan}
        for axis, scale, model_scale in ((axes[0], 1, 1), (axes[1], duty, grid_duty)):
            axis.errorbar(L, eta * scale, yerr=err * scale, fmt="o", ms=4,
                          color=color, label=f"{scan} readout")
            if good:
                axis.plot(grid, model * model_scale, color=color, lw=1,
                          label=f"fit T = {fit['T']:.0f} ± {fit['T_err']:.0f} us, "
                                f"τ = {fit['tau']:.1f} ± {fit['tau_err']:.1f} us")
    ceiling = kappa_ratio * ringup_efficiency(grid, 1.0, t1_pump, tau_ceiling)
    for axis, model_scale in ((axes[0], 1), (axes[1], grid_duty)):
        axis.plot(grid, ceiling * model_scale, color="0.5", ls="--",
                  label=f"ceiling, T1_pump = {t1_pump:.0f} us, τ = {tau_ceiling:.2f} us")
    l_star = np.sqrt(2 * dead * t1_pump)
    axes[1].axvline(l_star, color="0.5", ls=":",
                    label=f"sqrt(2 dead T1_pump) = {l_star:.0f} us")

    axes[0].set_ylabel("efficiency per photon")
    axes[1].set_ylabel("efficiency x duty cycle")
    for axis in axes:
        axis.set_xlabel("buffer length (us)")
        axis.grid(alpha=0.3)
        axis.legend(fontsize="x-small")
    return fig, fits


# --------------------------------------------------------------------------
# Offline checking of a saved run
# --------------------------------------------------------------------------


def load_saved_run(data_dir):
    """Read every .h5 in a run folder, in acquisition (filename index) order."""
    files = sorted(Path(data_dir).glob("*.h5"))
    if not files:
        raise SystemExit(f"No .h5 files in {data_dir}")
    records = []
    for path in files:
        with SlabFile(str(path), "r") as h5:
            keys = set(h5.keys())
            data = {"angle": float(np.array(h5["angle"])) if "angle" in keys else 0.0}
            for scan in SCANS:
                if f"I{scan}" in keys:
                    data[f"I{scan}"] = np.array(h5[f"I{scan}"])
                    data[f"Q{scan}"] = np.array(h5[f"Q{scan}"])
            if "fids" in keys:
                data["fids"] = np.array(h5["fids"])
            cfg = h5.load_config()
        expt = cfg["expt"]
        records.append({
            "index": int(path.stem.rsplit("_", 1)[1]),
            "gain_p": float(expt["gain_p"]),
            "gain_b": float(expt["gain_b"]),
            "length_b": float(expt["length_b"]),
            "frequency_b": float(expt["frequency_b"]),
            "final_delay": float(expt["final_delay"]),
            "shots": int(expt["shots"]),
            "qubit": int(np.atleast_1d(expt["qubit"])[0]),
            "data": data,
        })
    return records


def _describe(name, values):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not values.size:
        print(f"    {name:<24}  (no finite values)")
        return
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    print(f"    {name:<24}  median {median:9.4f}  mad {mad:8.4f}  "
          f"p95 {float(np.percentile(values, 95)):9.4f}  "
          f"max {float(values.max()):9.4f}  "
          f"-> med+5mad {median + 5 * mad:8.4f}")


def replot_rejects(rejects_path):
    """
    Draw one PNG per rejected attempt stored in a `qc_rejects.h5`.

    The counterpart to running the sweep with `--plots none`: the histograms
    are kept either way, so the failure figures can be made afterwards without
    having paid for them during the run.
    """
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = rejects_path.parent
    with h5py.File(str(rejects_path), "r") as h5:
        names = sorted(h5.keys())
        for name in names:
            group = h5[name]
            centers = np.array(group["centers"])
            hists = {scan: np.array(group[f"hist{scan}"]) for scan in SCANS
                     if f"hist{scan}" in group}
            title = (f"#{group.attrs['index']:05d} attempt "
                     f"{group.attrs['attempt']} -- pump gain "
                     f"{group.attrs['gain_p']:.4f}, buffer gain "
                     f"{group.attrs['gain_b']:.4f}")
            if "frequency_b" in group.attrs:
                title += f", buffer {group.attrs['frequency_b']:.4f} MHz"
            failed = str(group.attrs.get("failed_gates", "")).split("; ")
            fig = plot_point(centers, hists, title=title,
                             failed=[f for f in failed if f])
            fig.savefig(out_dir / f"qc_{name}.png")
            plt.close(fig)
    print(f"Wrote {len(names)} PNGs next to {rejects_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir",
                        help="run folder holding the .h5 files")
    parser.add_argument("--config", default=CONFIG,
                        help="device config for the reference frame")
    parser.add_argument("--bins", type=int, default=QC_BINS)
    parser.add_argument("--replot-rejects", metavar="QC_REJECTS_H5",
                        help="draw one PNG per group in a qc_rejects.h5 and "
                             "exit; --data-dir is not read")
    args = parser.parse_args()

    if args.replot_rejects:
        replot_rejects(Path(args.replot_rejects))
        return

    cfg_name = args.config
    auto_cfg = config.load(str(REPO_DIR / "configs" / cfg_name))

    if not args.data_dir:
        raise SystemExit("--data-dir is required unless --replot-rejects is given.")
    records = load_saved_run(args.data_dir)
    qi = records[0]["qubit"]
    reference = reference_from_config(auto_cfg, qi, records[0]["shots"])
    edges, centers = bin_grid(reference, args.bins)
    print(f"Read {len(records)} files from {args.data_dir}")
    print(f"Reference frame from {cfg_name}, qubit {qi}")
    print(f"  {reference}")
    print(f"  grid {edges[0]:.2f} .. {edges[-1]:.2f} in {args.bins} bins "
          f"of {edges[1] - edges[0]:.3f}")

    results = [analyze_point(r["data"], edges, centers, reference)
               for r in records]

    print("\nAbsolute shape statistics (the numbers PEAK_TOL / WIDTH_* come from):")
    for scan in SCANS:
        present = [res["metrics"][scan] for res in results
                   if scan in res["metrics"]]
        if not present:
            continue
        print(f"  scan {scan!r} ({len(present)} points)")
        _describe("|peak_offset| / sigma", [abs(m["peak_offset"]) for m in present])
        _describe("width / sigma", [m["width_ratio"] for m in present])

    # Replay in acquisition order. The gates are stateless now, so this is
    # just "which points would a live run have retaken".
    monitor = QCMonitor(reference, edges)
    rejected, ratios = [], []
    for record, result in zip(records, results):
        metrics, hists = result["metrics"], result["hist"]
        ratio = f_over_e(metrics)
        if ratio is not None:
            ratios.append(ratio)
        ok, failed = monitor.check(metrics, hists, record["gain_p"],
                                   record["gain_b"])
        if not ok:
            rejected.append((record, failed))

    print("\nBackground levels (the numbers the absolute bars come from):")
    for scan in FLOOR_SCANS + PUMP_BACKGROUND_SCANS:
        present = [res["metrics"][scan]["p"] for res in results
                   if scan in res["metrics"]]
        if present:
            bar = (GROUND_MAX_PE if scan in FLOOR_SCANS else PUMP_MAX_PE)
            _describe(f"p_{scan} (bar {bar})", present)
    # Not a tolerance: reported because its structure is worth seeing. A
    # spread here is real physics, not instability -- see the module docstring.
    print("\nf/e readout ratio (diagnostic, not gated):")
    _describe("p_f/p_e", ratios)

    print(f"\nWould have retaken {len(rejected)}/{len(records)} points "
          f"({100 * len(rejected) / len(records):.0f}%):")
    for record, failed in rejected:
        print(f"  #{record['index']:05d} pump {record['gain_p']:.4f} "
              f"buffer {record['gain_b']:.4f}")
        for reason in failed:
            print(f"      {reason}")
    if not rejected:
        print("  (none -- the gates pass everything in this run)")

    counts = {}
    for _, failed in rejected:
        for reason in failed:
            counts[reason.split(":")[0]] = counts.get(reason.split(":")[0], 0) + 1
    if counts:
        print("\nFailures by gate:")
        for gate, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {gate:<12} {count}")


if __name__ == "__main__":
    main()
