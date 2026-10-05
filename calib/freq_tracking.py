"""
Multi-qubit qubit-frequency drift tracking.

Round-robins qubit spectroscopy across a list of qubits for a fixed wall-clock
duration, fitting each sweep to a Lorentzian so f_ge(t) can be followed for
hours. Structured like ``time_tracking.time_tracking``: the RF board is switched
to each qubit in turn with ``rfboard.activate_qubit_rf``, and CSVs are rewritten
after every round so an interrupted run keeps its data.

Two entry points:

- ``optimize_spec_params``: per-qubit probe-gain / reps calibration (below).
- ``track_qubit_freqs``: the round-robin monitor itself.

Why the RF board forces interleaving
------------------------------------
Every qubit shares one ADC and one qubit-drive DAC; the per-qubit entries in
``hw.soc.adcs/dacs`` are tunable bandpass filter and attenuator settings on that
shared chain. ``activate_qubit_rf`` reprograms them, so exactly one qubit is
addressable at a time and simultaneous acquisition is not possible. All qubits
still get traces spanning the same window, sampled once per round.

Choosing the probe gain
-----------------------
Maximizing raw SNR is the wrong objective for a drift monitor: SNR keeps rising
with probe gain long after the line has power-broadened and AC-Stark-shifted,
and both effects corrupt the frequency being tracked. What actually matters is
the statistical uncertainty on the fitted center, ``sigma_f``, which comes
straight out of the fit covariance. ``optimize_spec_params`` walks a gain ladder
and minimizes

    fom = sigma_f * sqrt(t_row)          # MHz per sqrt(second) of averaging

subject to the linewidth staying within ``broaden_factor`` of the narrowest one
seen. That rejects the over-driven settings a pure-SNR search would pick.

Choosing reps
-------------
``sigma_f`` falls as 1/sqrt(shots) with no interior optimum, so the averaging is
not independently optimizable -- more is always better per shot and worse per
second. It is instead set by the cadence budget: given ``total_time``, the qubit
count and a target number of rows per qubit, each sweep gets a fixed time slice,
and ``plan_averaging`` fills it using the per-shot time measured during the gain
scan, splitting the result across reps and software rounds.
"""

import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..helpers import config, rfboard
from ..analysis import fitting as fitter
import slab_qick_calib.experiments as meas

# Fit-quality floor. A qubit peak can wash out entirely (unlike a resonator dip),
# and curve_fit returns confident garbage rather than raising when it does.
MIN_R2 = 0.2
# Reject a gain once the line is this much wider than the narrowest one measured.
BROADEN_FACTOR = 1.5
# Span is kept at this multiple of the fitted HWHM: wide enough to see baseline
# on both sides, narrow enough to put many points across the line.
SPAN_KAPPA_MULT = 20.0
# Guard rails on the span in MHz, so a bad fit cannot collapse or explode it.
SPAN_MIN = 0.5
SPAN_MAX = 60.0
# Wall-clock seconds to allow for one activate_qubit_rf call.
RF_SWITCH_OVERHEAD = 2.0
# Largest reps to put in a single QICK program; excess averaging goes to rounds.
MAX_REPS = 2000
# Fraction of each cadence period spent acquiring; the rest is slack.
DUTY_FRAC = 0.5
# A fitted center is only acted on when its 1-sigma error is below this (MHz).
TRUST_SIGMA_F = 0.1
# Coherence fits below this R^2 are never written back to the config.
COH_MIN_R2 = 0.5
# Clamps for config write-back, matching the notebook's conventions.
MAX_T1 = 1100.0
T1_RNG = [0.5, MAX_T1]
T2R_RNG = [0.025, MAX_T1]
T2E_RNG = [0.04, 2 * MAX_T1]


def _fit_row(freq, sig):
    """
    Fit one spectroscopy trace to a Lorentzian.

    Returns a dict with the center, HWHM, the 1-sigma error on the center taken
    from the fit covariance, R^2, and an 'ok' flag. Never raises.
    """
    out = {"f0": np.nan, "kappa": np.nan, "sigma_f": np.nan, "r2": np.nan,
           "amp": np.nan, "ok": False, "pars": None}
    try:
        p, pcov, pinit = fitter.fitlor(freq, sig)
        out["pars"] = p          # kept so the sweep image can draw the fit
        out["f0"] = p[2]
        out["kappa"] = abs(p[3])
        out["amp"] = abs(p[1])
        out["r2"] = fitter.get_r2(freq, sig, fitter.lorfunc, p)
        if pcov is not None and np.all(np.isfinite(pcov)):
            out["sigma_f"] = float(np.sqrt(abs(pcov[2, 2])))
        # A center outside the swept window means the fit ran away
        in_window = freq.min() <= out["f0"] <= freq.max()
        out["ok"] = bool(out["r2"] > MIN_R2 and in_window and np.isfinite(out["sigma_f"]))
    except Exception:
        pass
    return out


def _run_spec(cfg_dict, qi, params, signal="amps", style="medium",
              from_config=False):
    """
    Run one QubitSpec sweep and fit it.

    ``from_config=True`` is the plain call::

        meas.QubitSpec(cfg_dict, qi=qi, style='coarse', params=params)

    with ``params`` used verbatim. The window centre then comes from the config
    ``f_ge`` via the style's span, so with config write-back enabled the window
    follows the qubit on its own and needs no explicit re-centering. go stays
    True, so the sweep saves its own h5 and figure and QubitSpec's own fit
    supplies f0.

    ``from_config=False`` drives the experiment manually with an injected
    centre, span and averaging, writing nothing to disk.

    Either way a Lorentzian is also fitted, for the ``sigma_f`` the trust gate
    needs. Returns (fit_dict, freq, sig, raw, elapsed_s).
    """
    t0 = time.time()
    if from_config:
        _figs = set(plt.get_fignums())
        expt = meas.QubitSpec(cfg_dict, qi=qi, style=style, params=dict(params),
                              progress=False)
        data = expt.data                      # go=True already ran and analyzed
        _close_new_figs(_figs)
    else:
        expt = meas.QubitSpec(cfg_dict, qi=qi, go=False, params=params,
                              style=style, check_params=False, progress=False)
        data = expt.acquire(progress=False)
    elapsed = time.time() - t0

    freq = np.asarray(data["xpts"], dtype=float)
    sig = np.asarray(data[signal], dtype=float)
    raw = {k: np.asarray(data[k], dtype=float)
           for k in ("avgi", "avgq", "amps", "phases") if k in data}

    fit = _fit_row(freq, sig)
    if from_config:
        # Prefer QubitSpec's own fit for the centre; keep the Lorentzian
        # sigma_f, which its analyze does not report.
        try:
            f0 = float(data["new_freq"])
            if freq.min() <= f0 <= freq.max():
                fit["f0"] = f0
                fit["r2"] = float(data.get("r2", fit["r2"]))
                fit["ok"] = bool(fit["r2"] > MIN_R2)
        except Exception:
            pass                              # fall back to the Lorentzian fit
    return fit, freq, sig, raw, elapsed


def _close_new_figs(before):
    """
    Close any pyplot figures opened since ``before`` was captured.

    The experiment classes' ``save_fig`` does ``savefig`` then ``plt.show()``
    and never closes; under Agg, show() is a no-op, so every figure stays in
    pyplot's registry holding its data. At ~4.6 MB each that is fatal over a
    long run -- 200 figures is already ~1 GB. Only figures this call created
    are closed, so the reusable sweep figure survives.
    """
    for n in set(plt.get_fignums()) - before:
        plt.close(n)


# One reusable figure for the per-sweep images. Creating and closing a figure
# for each of ~1000 sweeps is slow and leaks; clearing one axis is not.
_SWEEP_FIG = [None]


def _save_sweep_image(path, x, y, xlabel, ylabel, fitfunc=None, fitpars=None):
    """
    Write a small PNG of one sweep with its fit overlaid.

    No title, per project convention -- the filename carries the identity.
    Never raises: an imaging failure must not take down a multi-hour run.
    """
    try:
        if _SWEEP_FIG[0] is None:
            _SWEEP_FIG[0] = plt.subplots(figsize=(4, 3), dpi=90)
        fig, ax = _SWEEP_FIG[0]
        ax.clear()
        ax.plot(x, y, ".", ms=3, lw=0)
        if fitfunc is not None and fitpars is not None:
            xs = np.linspace(np.min(x), np.max(x), 400)
            ax.plot(xs, fitfunc(xs, *fitpars), "-", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    except Exception as e:
        print(f"    [image] {Path(path).name} failed: {e}")


def _relocate(cfg_dict, qi, start="coarse", reps_cap=None, verbose=True):
    """
    Re-find a qubit that Ramsey can no longer see, using the repo's own
    ``qubit_tuning.find_spec`` ladder: huge -> coarse -> medium -> fine,
    narrowing on success and widening on failure, up to 6 attempts. When even
    the coarsest scan fails it doubles ``readout.reps`` and ``spec_gain`` and
    starts over. find_spec writes the recovered f_ge to the config itself.

    That power escalation is permanent and compounds across a long run -- a
    qubit that needs rescuing three times ends up at 8x the averaging, slowing
    every later round. ``reps_cap`` bounds it.

    Returns (found, f_ge) with f_ge read back from the config.
    """
    from . import qubit_tuning

    before = config.load(cfg_dict["cfg_file"])
    reps0 = before["device"]["readout"]["reps"][qi]
    gain0 = before["device"]["qubit"]["spec_gain"][qi]

    figs = set(plt.get_fignums())
    try:
        found, ntries = qubit_tuning.find_spec(qi, cfg_dict, start=start)
    except Exception as e:
        print(f"    Q{qi} relocation failed: {e}")
        return False, np.nan
    finally:
        _close_new_figs(figs)          # find_spec's sweeps run with display on

    after = config.load(cfg_dict["cfg_file"])
    reps1 = after["device"]["readout"]["reps"][qi]
    gain1 = after["device"]["qubit"]["spec_gain"][qi]
    if reps_cap is not None and reps1 > reps_cap:
        config.update_readout(cfg_dict["cfg_file"], "reps", reps_cap, qi,
                              verbose=False)
        print(f"    Q{qi} relocation raised reps to {reps1}; capped at {reps_cap}")
        reps1 = reps_cap
    elif verbose and (reps1 != reps0 or gain1 != gain0):
        print(f"    Q{qi} relocation escalated power: reps {reps0}->{reps1}, "
              f"spec_gain {gain0}->{gain1}")

    f_ge = after["device"]["qubit"]["f_ge"][qi]
    if verbose:
        print(f"    Q{qi} relocation {'found' if found else 'FAILED'} after "
              f"{ntries} scan(s); f_ge = {f_ge:.4f} MHz")
    return bool(found), float(f_ge)


def _write_back(cfg_file, qi, kind, expt, value, r2, fadj=np.nan,
                update_final_delay=True, verbose=False):
    """
    Write one experiment's result into the config, following the notebook's
    conventions (same fields, same clamps).

    Only ever called for fits the caller has already judged good. That gate
    matters: these clamps do not reject a bad fit, they squash it to the
    boundary -- a failed T1 lands on 0.5 us, which then makes the next round's
    sweep span 1.9 us and the qubit unmeasurable from then on.

    Returns a short description of what was written, or None.
    """
    try:
        if kind == "resonator":
            expt.update(freq=True, fast=True, verbose=verbose)
            return "readout.frequency, kappa"
        if kind == "spectroscopy":
            config.update_qubit(cfg_file, "f_ge", value, qi, verbose=verbose)
            return f"f_ge={value:.4f}"
        if kind == "t1":
            expt.update(rng_vals=T1_RNG, no_final=not update_final_delay,
                        verbose=verbose)
            return f"T1={value:.1f}us" + ("" if update_final_delay
                                          else " (final_delay held)")
        if kind == "ramsey":
            config.update_qubit(cfg_file, "T2r", value, qi, sig=2,
                                rng_vals=T2R_RNG, verbose=verbose)
            wrote = f"T2r={value:.2f}us"
            if np.isfinite(fadj):
                config.update_qubit(cfg_file, "f_ge", fadj, qi, verbose=verbose)
                wrote += f", f_ge={fadj:.4f}"
            return wrote
        if kind == "echo":
            config.update_qubit(cfg_file, "T2e", value, qi, sig=2,
                                rng_vals=T2E_RNG, verbose=verbose)
            return f"T2e={value:.2f}us"
    except Exception as e:
        print(f"    Q{qi} {kind} config update failed: {e}")
    return None


def _run_res_spec(cfg_dict, qi, center, span, expts=100, reps=None, signal="amps",
                  from_config=False):
    """
    Run one resonator sweep and fit it.

    ``from_config=True`` is the plain call::

        meas.ResSpec(cfg_dict, qi=qi, params={'span': 'kappa'})

    ResSpec resolves 'kappa' to 7 x the config kappa and centers on the config
    readout frequency; gain, reps, rounds, length and expts all come from the
    config. go stays True so it saves its own h5 and figure, and its hanger fit
    supplies f0 and kappa. There is no window to inject a center, so the
    resonator window does not follow drift in this mode.

    ``from_config=False`` uses an explicit centre and span so the window can be
    re-centered between rounds.

    Either way a Lorentzian is also fitted to the amplitude -- it costs no
    hardware time and yields sigma_f from the covariance, which the trust gate
    and the error bars need. This config has no readout mixer or LO offset, so
    'xpts' is already absolute frequency.

    Returns (fit_dict, freq, sig, raw, elapsed_s).
    """
    t0 = time.time()
    if from_config:
        _figs = set(plt.get_fignums())
        expt = meas.ResSpec(cfg_dict, qi=qi, params={"span": "kappa"},
                            progress=False)
        data = expt.data                     # go=True already ran and analyzed
        _close_new_figs(_figs)
    else:
        params = {"center": center, "span": span, "expts": expts}
        if reps is not None:
            params["reps"] = reps
        expt = meas.ResSpec(cfg_dict, qi=qi, go=False, params=params,
                            check_params=False, progress=False)
        data = expt.acquire(progress=False)
    elapsed = time.time() - t0

    # 'freq' is the LO-corrected axis when analyze ran; otherwise xpts is it
    freq = np.asarray(data.get("freq", data["xpts"]), dtype=float)
    sig = np.asarray(data[signal], dtype=float)
    raw = {k: np.asarray(data[k], dtype=float)
           for k in ("avgi", "avgq", "amps", "phases") if k in data}

    fit = _fit_row(freq, sig)
    fit["expt"] = expt          # kept so the caller can call expt.update()
    if from_config:
        # Prefer ResSpec's own hanger fit for f0 and kappa; keep the Lorentzian
        # sigma_f, which the hanger fit does not report per-parameter.
        try:
            fit["f0"] = float(np.asarray(data["freq_fit"])[0])
            fit["kappa"] = float(data["kappa"])
            fit["r2"] = float(data.get("r2", fit["r2"]))
            fit["ok"] = bool(fit["r2"] > MIN_R2
                             and freq.min() <= fit["f0"] <= freq.max())
        except Exception:
            pass                              # fall back to the Lorentzian fit
    return fit, freq, sig, raw, elapsed


# Filename stems analyze_tracking.ipynb globs for, via collections.detect_t2_type
_H5_PREFIX = {"t1": "t1", "ramsey": "T2r", "echo": "T2e"}


def _run_coherence(cfg_dict, qi, f_ge, kind, expts=40, reps=None, span_mult=3.7,
                   from_config=False, config_kwargs=None, fname=None):
    """
    Run one coherence measurement. ``kind`` is 't1', 'ramsey' or 'echo'.

    Two modes:

    ``from_config=True`` -- the plain call, every parameter taken from the
    config through the experiment class's own defaults::

        meas.T1Experiment(cfg_dict, qi=qi)
        meas.T2Experiment(cfg_dict, qi=qi, max_err=10)
        meas.T2Experiment(cfg_dict, qi=qi, params={'experiment_type': 'echo'})

    ``config_kwargs`` supplies the per-kind extras (e.g. ``max_err``). go stays
    True so the experiment saves its own h5 and figure through the standard
    machinery. Because the sweep runs inside __init__ there is no opportunity
    to retune the drive, so this uses the config ``f_ge``, not the frequency
    this round measured.

    ``from_config=False`` -- ``expts``/``reps`` are overridden for the time
    budget, T1 spans ``span_mult`` x the config T1, and ``f_ge`` is written into
    the experiment's own config copy (never the file) so the pulses follow the
    frequency spectroscopy just fitted.

    Returns (value_us, r2, f_adjust_MHz, raw, elapsed_s). ``f_adjust`` is the
    Ramsey-corrected qubit frequency, NaN for T1. ``raw`` carries the sweep --
    'xpts', the quadratures, and the fit function and parameters so the sweep
    image can be drawn. Values are NaN on fit failure.
    """
    t0 = time.time()

    if from_config:
        kw = dict(config_kwargs or {})
        if fname is not None:
            # Absolute fname overrides the default data dir, putting the h5
            # where analyze_tracking.ipynb's raw-scan cell globs for it
            kw["fname"] = fname
        _figs = set(plt.get_fignums())
        if kind == "t1":
            expt = meas.T1Experiment(cfg_dict, qi=qi, progress=False, **kw)
        elif kind == "ramsey":
            # experiment_type already defaults to 'ramsey'
            expt = meas.T2Experiment(cfg_dict, qi=qi, progress=False, **kw)
        else:
            expt = meas.T2Experiment(cfg_dict, qi=qi, progress=False,
                                     params={"experiment_type": "echo"}, **kw)
        data = expt.data                      # go=True already ran and analyzed
        _close_new_figs(_figs)
        elapsed = time.time() - t0
    else:
        params = {"expts": expts}
        if reps is not None:
            params["reps"] = reps
        if kind == "t1":
            expt = meas.T1Experiment(cfg_dict, qi=qi, go=False, params=params,
                                     check_params=False, progress=False)
            expt.cfg.device.qubit.f_ge[qi] = f_ge
            expt.cfg.expt.span = span_mult * expt.cfg.device.qubit.T1[qi]
        else:
            params["experiment_type"] = kind          # 'ramsey' or 'echo'
            expt = meas.T2Experiment(cfg_dict, qi=qi, go=False, params=params,
                                     check_params=False, progress=False)
            expt.cfg.device.qubit.f_ge[qi] = f_ge
        data = expt.acquire(progress=False)
        elapsed = time.time() - t0

    raw = {"xpts": np.asarray(data["xpts"], dtype=float)}
    for k in ("avgi", "avgq", "amps", "phases"):
        if k in data:
            raw[k] = np.asarray(data[k], dtype=float)

    try:
        if not from_config:
            data = expt.analyze(data)
        raw["fitfunc"] = getattr(expt, "fitfunc", None)
        raw["fitpars"] = data.get("best_fit")
        bf = np.asarray(data["best_fit"], dtype=float)
        cov = np.asarray(data.get("fit_err_avgi", np.full((5, 5), np.nan)),
                         dtype=float)

        def _cov(k):
            try:
                return float(np.sqrt(np.abs(cov[k][k])))
            except Exception:
                return np.nan

        if kind == "t1":
            val, fadj = float(data["new_t1"]), np.nan
            # T1 fit is [y0, yscale, decay]
            raw["off"], raw["amp"], raw["err"] = bf[0], bf[1], _cov(2)
        else:
            # T2 fit is [yscale, freq, phase_deg, decay, y0]; decay is T2
            val = float(bf[3])
            fadj = float(data.get("new_freq", np.nan))
            raw["off"], raw["amp"], raw["err"] = bf[4], bf[0], _cov(3)
            # Index 1 is the oscillation frequency, so its covariance entry is
            # the error on the corrected f_ge -- the same quantity
            # time_tracking records as 'fge_err'.
            raw["f_err"] = _cov(1)
        raw["expt"] = expt      # kept so the caller can write back to the config
        return val, float(data.get("r2", np.nan)), fadj, raw, elapsed
    except Exception:
        return np.nan, np.nan, np.nan, raw, elapsed


def optimize_spec_params(
    qi,
    cfg_dict,
    soc=None,
    signal="amps",
    gain_ladder=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    scan_reps=150,
    scan_expts=80,
    locate_style="medium",
    length=10,
    broaden_factor=BROADEN_FACTOR,
    r2_reference=0.8,
    trust_sigma_f=TRUST_SIGMA_F,
    verbose=True,
):
    """
    Find the probe gain that minimizes frequency uncertainty per unit time.

    Runs in three stages: locate the line with a wide medium-style sweep, narrow
    the span to a fixed multiple of the fitted linewidth, then walk the gain
    ladder recording sigma_f and the elapsed time for each setting.

    Args:
        qi: Qubit index
        cfg_dict: Standard experiment config dictionary
        soc: QICK SoC object; if given, the RF board is switched to this qubit first
        signal: Which quadrature to fit ('amps', 'avgi', 'avgq')
        gain_ladder: Multipliers applied to the config's medium-style spec gain
        scan_reps: reps used for every point of the ladder (kept equal so the
            fom comparison is fair)
        scan_expts: Frequency points per scan sweep
        locate_style: QubitSpec style for the initial wide locate sweep
        length: Probe pulse length in us
        broaden_factor: Reject gains whose linewidth exceeds this multiple of the
            narrowest linewidth seen in the ladder
        r2_reference: Only trials fitting at least this well may define that
            narrowest linewidth, so a noisy low-gain sweep cannot set a
            spuriously narrow reference and veto every usable gain
        trust_sigma_f: Centers fitted less precisely than this (MHz) are not
            trusted; the config f_ge is used instead
        verbose: Print per-gain results

    Returns:
        dict with 'gain', 'span', 'center', 'kappa', 'sigma_f', 'time_per_shot',
        'trials' (the full ladder) and 'ok'. When no gain produces a usable fit,
        or the best one is not precise enough to trust, 'ok' is False and the
        center falls back to the config f_ge.
    """
    if soc is not None:
        auto_cfg = config.load(cfg_dict["cfg_file"])
        rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_dict["cfg_file"],
                                  verbose=False)

    auto_cfg = config.load(cfg_dict["cfg_file"])
    base_gain = 5 * auto_cfg["device"]["qubit"]["low_gain"] * \
        auto_cfg["device"]["qubit"]["spec_gain"][qi]
    max_gain = auto_cfg["device"]["qubit"]["max_gain"]
    f_ge = auto_cfg["device"]["qubit"]["f_ge"][qi]

    result = {"qubit": qi, "gain": base_gain, "span": SPAN_MAX, "center": f_ge,
              "kappa": np.nan, "sigma_f": np.nan, "time_per_shot": np.nan,
              "trials": [], "ok": False}

    # --- Stage 1: locate the line ---------------------------------------
    # Gain, span and expts come from locate_style; only the averaging is pinned.
    fit, freq, sig, elapsed = _run_spec(
        cfg_dict, qi,
        {"reps": scan_reps, "length": length, "rounds": 1},
        signal=signal, style=locate_style,
    )
    if not fit["ok"]:
        if verbose:
            print(f"  Q{qi}: locate sweep failed (r2={fit['r2']:.3f}); "
                  f"keeping config gain {base_gain:.4g}")
        return result

    center = fit["f0"]
    span = float(np.clip(SPAN_KAPPA_MULT * fit["kappa"], SPAN_MIN, SPAN_MAX))
    if verbose:
        print(f"  Q{qi}: located at {center:.4f} MHz "
              f"(kappa={fit['kappa']:.4f} MHz) -> span {span:.3f} MHz")

    # --- Stage 2: gain ladder -------------------------------------------
    trials = []
    for mult in gain_ladder:
        gain = min(base_gain * mult, max_gain)

        # Ascending gain broadens the line, so grow the span off the previous
        # linewidth and re-run once if the fit still fills the window.
        for _ in range(2):
            params = {
                "gain": gain,
                "span": span,
                "start": center - span / 2,
                "expts": scan_expts,
                "reps": scan_reps,
                "rounds": 1,
                "length": length,
            }
            fit, freq, sig, elapsed = _run_spec(cfg_dict, qi, params, signal=signal)
            if fit["ok"] and fit["kappa"] > span / 8:
                span = float(np.clip(2 * SPAN_KAPPA_MULT * fit["kappa"],
                                     SPAN_MIN, SPAN_MAX))
                continue
            break

        shots = scan_expts * scan_reps
        trial = {
            "gain": gain, "span": span, "elapsed": elapsed,
            "time_per_shot": elapsed / shots,
            "fom": fit["sigma_f"] * np.sqrt(elapsed) if fit["ok"] else np.nan,
            **{k: fit[k] for k in ("f0", "kappa", "sigma_f", "r2", "amp", "ok")},
        }
        trials.append(trial)

        if verbose:
            if trial["ok"]:
                print(f"  Q{qi}: gain {gain:.4g} -> kappa {fit['kappa']:.4f} MHz, "
                      f"sigma_f {1e3*fit['sigma_f']:.2f} kHz, r2 {fit['r2']:.3f}, "
                      f"fom {1e3*trial['fom']:.2f} kHz*sqrt(s)")
            else:
                print(f"  Q{qi}: gain {gain:.4g} -> no usable fit "
                      f"(r2 {fit['r2']:.3f})")

        # Track the line as it Stark-shifts so the window stays centered
        if fit["ok"]:
            center = fit["f0"]

    result["trials"] = trials
    good = [t for t in trials if t["ok"] and np.isfinite(t["fom"])]
    if not good:
        if verbose:
            print(f"  Q{qi}: no gain gave a usable fit; keeping {base_gain:.4g}")
        return result

    # --- Stage 3: reject power-broadened settings, then minimize the fom ---
    # The reference linewidth has to come from a trial that actually fit well.
    # A marginal low-gain sweep often returns a spuriously narrow kappa, and
    # using that as the reference would reject every usable gain above it.
    reference = [t for t in good if t["r2"] >= r2_reference] or good
    kappa_min = min(t["kappa"] for t in reference)
    narrow = [t for t in good if t["kappa"] <= broaden_factor * kappa_min]
    if not narrow:
        narrow = good
    best = min(narrow, key=lambda t: t["fom"])

    dropped = [t["gain"] for t in good if t not in narrow]
    if verbose and dropped:
        print(f"  Q{qi}: rejected for power broadening (>{broaden_factor}x "
              f"{kappa_min:.4f} MHz): {', '.join(f'{g:.4g}' for g in dropped)}")

    result.update({
        "gain": best["gain"],
        "span": best["span"],
        "center": best["f0"],
        "kappa": best["kappa"],
        "sigma_f": best["sigma_f"],
        "time_per_shot": best["time_per_shot"],
        "scan_shots": scan_expts * scan_reps,   # lets plan_reps project sigma_f
        "ok": True,
    })

    # The ladder always returns a best entry, even when every trial was fitting
    # noise. If the winner is not precise enough to act on, keep the gain it
    # found but fall back to the config prior for the window center: a noise
    # center becomes the fallback that the T1 drive frequency depends on, so a
    # bad one is worse than a stale one. Widen the span to keep a chance of
    # catching the real line.
    if best["sigma_f"] >= trust_sigma_f:
        result["center"] = f_ge
        result["span"] = SPAN_MAX
        result["ok"] = False
        if verbose:
            print(f"  Q{qi}: best sigma_f {1e3*best['sigma_f']:.0f} kHz exceeds "
                  f"{1e3*trust_sigma_f:.0f} kHz -- centering on config f_ge "
                  f"{f_ge:.4f} MHz instead of the fitted {best['f0']:.4f} MHz")
        return result
    if verbose:
        print(f"  Q{qi}: chose gain {best['gain']:.4g} "
              f"(sigma_f {1e3*best['sigma_f']:.2f} kHz in {best['elapsed']:.1f} s)")
    return result


def plan_averaging(opt, n_qubits, cadence_s, expts, duty_frac=DUTY_FRAC,
                   min_reps=20, max_reps=MAX_REPS, verbose=True):
    """
    Pick reps and rounds so one sweep fills its share of the cadence period.

    sigma_f falls monotonically as 1/sqrt(shots), so there is no interior
    optimum; the averaging is whatever fits the time budget. Uses the per-shot
    time measured during the gain scan rather than modelling QICK overheads.

    Only ``duty_frac`` of each period is spent acquiring. The slack absorbs
    sweeps that run long -- rounds carry program-rebuild overhead the per-shot
    estimate does not see -- so the cadence still holds when they do.

    The affordable shot count at these budgets runs to 10^4, which is far more
    than belongs in a single tProcessor loop, so anything past ``max_reps`` is
    moved into software rounds.

    Args:
        opt: Result dict from optimize_spec_params
        n_qubits: Number of qubits sharing the period
        cadence_s: Seconds between successive points on the same qubit
        expts: Frequency points per sweep in the tracking run
        duty_frac: Fraction of the period spent acquiring
        min_reps: Floor on reps when the budget is tiny
        max_reps: Largest reps to put in one program; the rest becomes rounds
        verbose: Print the budget

    Returns:
        (reps, rounds, row_budget_s)
    """
    row_budget = cadence_s * duty_frac / n_qubits - RF_SWITCH_OVERHEAD

    tps = opt.get("time_per_shot", np.nan)
    if not np.isfinite(tps) or tps <= 0 or row_budget <= 0:
        if verbose:
            print(f"  Q{opt['qubit']}: no usable timing budget; falling back to "
                  f"reps={min_reps}, rounds=1")
        return min_reps, 1, row_budget

    shots_per_sweep = row_budget / (tps * expts)   # = reps * rounds
    reps = int(np.clip(shots_per_sweep, min_reps, max_reps))
    rounds = max(1, int(shots_per_sweep / reps))

    if verbose:
        est = reps * rounds * expts * tps
        # sigma_f falls as 1/sqrt(total shots) relative to the scan that measured it
        scan_shots = opt.get("scan_shots")
        proj = (opt["sigma_f"] * np.sqrt(scan_shots / (reps * rounds * expts))
                if scan_shots else np.nan)
        msg = (f"  Q{opt['qubit']}: budget {row_budget:.1f} s/sweep -> "
               f"reps {reps} x rounds {rounds} (~{est:.1f} s/sweep)")
        if np.isfinite(proj):
            msg += f", projected sigma_f ~{1e3*proj:.1f} kHz"
        print(msg)
    return reps, rounds, row_budget


def track_qubit_freqs(
    qubit_list,
    cfg_dict,
    soc,
    settings,
    total_time=5.0,
    max_rounds=None,
    cadence_s=300.0,
    expts=80,
    length=10,
    spec_params=None,
    spec_style="medium",
    spec_every=0,
    spec_on_fail=True,
    relocate_start="coarse",
    relocate_reps_cap=None,
    bf_client=None,
    signal="amps",
    recenter=True,
    recenter_frac=0.3,
    trust_sigma_f=TRUST_SIGMA_F,
    coherence=(),
    coh_expts=40,
    coh_reps=None,
    coh_from_config=("t1",),
    coh_config_kwargs=None,
    res_every=0,
    res_expts=100,
    res_reps=None,
    res_span_kappa=7.0,
    res_from_config=True,
    update_config=False,
    update_final_delay=True,
    save_images=True,
    save_dir=None,
    verbose=True,
):
    """
    Round-robin qubit spectroscopy for a fixed wall-clock duration.

    Each round switches the RF board to a qubit, takes one frequency sweep, fits
    it, and moves on. CSVs (one per qubit) and a combined .npz holding the full
    spectra are rewritten after every round, so KeyboardInterrupt or a crash
    keeps everything measured so far.

    Args:
        qubit_list: Qubit indices to track
        cfg_dict: Standard experiment config dictionary
        soc: QICK SoC object, required for the per-qubit RF board switch
        settings: {qi: dict} from optimize_spec_params, each carrying at least
            'gain', 'span', 'center', 'reps' and 'rounds'
        total_time: Run length in hours
        max_rounds: Stop after this many complete rounds, whichever comes
            first with total_time (None = time-limited only). Useful for a
            short shakedown of a fixed number of rounds.
        cadence_s: Seconds between successive points on the same qubit. Each
            round measures every qubit once, then idles until the period is up.
            Set to 0 to run rounds back to back.
        expts: Frequency points per sweep (ignored when spec_params is given)
        length: Probe pulse length in us (ignored when spec_params is given)
        spec_params: When set, spectroscopy runs the plain
            ``meas.QubitSpec(cfg_dict, qi=qi, style=spec_style, params=...)``
            with these params verbatim -- fixed gain and averaging, no
            calibration-derived values, and the window centred on the config
            f_ge by the style's span. Saves its own h5 and figure.
        spec_style: QubitSpec style used with spec_params
        spec_every: Run spectroscopy on every Nth round. 0 means never on
            schedule -- the frequency then comes from Ramsey, which measures
            the detuning directly and is far more precise than a Lorentzian on
            a broad saturated peak.
        spec_on_fail: Run the relocation ladder for a qubit whose Ramsey
            failed. Without this a qubit that drifts beyond Ramsey's capture
            range is lost for the rest of the run, since nothing else can find
            it again.
        relocate_start: Ladder entry point -- 'huge', 'coarse', 'medium' or
            'fine'. It widens on failure, so 'coarse' is a reasonable start.
        relocate_reps_cap: Ceiling on the readout.reps that the ladder's power
            escalation may leave behind (None = uncapped). Escalation is
            permanent, so repeated rescues otherwise compound into much slower
            rounds.
        bf_client: Optional fridge client exposing get_mxc_temperature() and
            is_pulsetube_on(). Read once per round and stored on every row, so
            a frequency jump can be checked against the fridge rather than
            guessed at.
        signal: Which quadrature to fit
        recenter: Re-center a qubit's window on its last good fit when the peak
            drifts into the outer part of the span. Necessary here because the
            span is only ~20 linewidths wide.
        recenter_frac: Re-center once |f0 - center| exceeds this fraction of the span
        trust_sigma_f: A fit is only acted on -- for the T1 drive frequency and
            for re-centering -- when its sigma_f is below this, in MHz. The 'ok'
            flag alone is far too permissive: a Lorentzian fitted to noise
            passes it routinely, and acting on that frequency is worse than
            using a stale but stable one.
        coherence: Schedule of coherence measurements as (kind, every, phase)
            tuples, kind in {'t1', 'ramsey', 'echo'}. A kind runs on round i
            when ``i % every == phase``. Stagger the phases so at most one kind
            lands per round -- e.g. (('t1',2,0), ('ramsey',4,1), ('echo',4,3))
            gives T1 every 2 rounds and each T2 every 4, never colliding, so a
            round never has to fit two sweeps into one time slot. All of them
            use the frequency that round's spectroscopy just fitted.
        coh_expts: Points per coherence sweep
        coh_reps: reps per point (None = each experiment class's default)
        coh_from_config: Kinds run as a plain ``meas.XExperiment(cfg_dict,
            qi=qi)`` -- every parameter from the config, and the experiment
            saves its own h5 and figure. These use the config ``f_ge`` rather
            than the frequency the round just fitted, since the sweep runs
            inside __init__.
        coh_config_kwargs: {kind: dict} of extra constructor kwargs for the
            config-driven kinds, e.g. ``{'ramsey': {'max_err': 10}}``
        res_every: Sweep the readout resonator on every Nth round (0 = never).
            A resonator sweep costs well under a second, so 1 is affordable.
        res_expts: Frequency points per resonator sweep (manual mode only)
        res_reps: reps per point (manual mode only; None = the ResSpec default)
        res_span_kappa: Resonator span as a multiple of the config kappa
            (manual mode only)
        res_from_config: Run the plain ``meas.ResSpec(cfg_dict, qi=qi,
            params={'span': 'kappa'})`` -- all parameters from the config, saves
            its own h5 and figure. The window then stays on the config readout
            frequency rather than following drift.
        update_config: Write each experiment's result back to the config file
            as it completes -- readout frequency and kappa, f_ge, T1, T2r, T2e.
            Only good fits are written (see ``trust_sigma_f`` and COH_MIN_R2);
            a bad fit is skipped rather than clamped into the config. With this
            on, the config-driven experiments pick up the tracked frequency and
            spans on the following round, so the run self-tunes.
        update_final_delay: Let the T1 write-back also set
            readout.final_delay = 6 x T1. This changes the shot time and so the
            duration of every later round.
        save_images: Write a PNG of every individual sweep, with its fit
            overlaid, under <save_dir>/images/sweeps/<kind>/q<N>_r<round>.png.
            Costs roughly 0.1 s per sweep.
        save_dir: Output directory (default: <expt_path>/FreqTracking/<timestamp>)
        verbose: Print per-sweep results

    Returns:
        (tracking, save_dir) where tracking is {qi: dict of arrays}
    """
    # Layout follows calib.time_tracking: the run directory becomes the
    # experiment path, so every h5 and figure the experiments produce lands
    # inside it and the run is self-contained. Capture the original path first
    # -- the notebook-compatible CSVs go to <base>/Tracking/csv, not inside
    # the run.
    base_path = Path(cfg_dict["expt_path"])
    if save_dir is None:
        run_id = f'{datetime.now().strftime("%Y_%m_%d_%H_%M")}_{total_time:.1f}hrs'
        save_dir = base_path / "Tracking" / run_id
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "data").mkdir(exist_ok=True)         # experiments given no fname
    (base_path / "Tracking" / "images").mkdir(parents=True, exist_ok=True)

    cfg_dict = dict(cfg_dict)                        # shallow: soc/im are proxies
    cfg_dict["expt_path"] = str(save_dir)

    # Per-qubit accumulators; 'freq' and 'sig' keep the full spectra
    coh_keys = []
    for kind, _, _ in coherence:
        # value, fit quality, decay-fit offset/amplitude, and the 1-sigma error
        # on the decay constant -- the same set time_tracking records
        coh_keys += [kind, f"{kind}_r2", f"{kind}_off", f"{kind}_amp",
                     f"{kind}_err"]
        if kind != "t1":
            coh_keys.append(f"{kind}_fadj")   # Ramsey/Echo frequency correction
    res_keys = (["res_f0", "res_kappa", "res_sigma_f", "res_r2", "res_ok",
                 "res_center"] if res_every else [])
    # f_track is the frequency actually adopted each round, whichever
    # measurement produced it: 1 = spectroscopy, 2 = Ramsey, 0 = none.
    env_keys = ("mxc_temp", "pulsetube_on") if bf_client is not None else ()
    keys = (("time", "elapsed", "f0", "kappa", "sigma_f", "r2", "amp", "ok",
             "center", "gain", "reps", "rounds",
             "f_track", "f_track_err", "f_track_src") + tuple(coh_keys)
            + tuple(res_keys) + env_keys)
    tracking = {qi: {k: [] for k in keys} for qi in qubit_list}

    # Per-sweep raw arrays; these stack into (rounds, points) and are excluded
    # from the CSVs, which stay one row per round.
    raw_keys = ["freq", "sig", "spec_avgi", "spec_avgq", "spec_phases"]
    if res_every:
        raw_keys += ["res_freq", "res_sig", "res_avgi", "res_avgq", "res_phases"]
    for kind, _, _ in coherence:
        raw_keys += [f"{kind}_{k}" for k in ("xpts", "avgi", "avgq", "amps")]
    for qi in qubit_list:
        for k in raw_keys:
            tracking[qi][k] = []

    img_root = Path(save_dir) / "images" / "sweeps"

    # h5 files land directly in the run directory, which is what
    # analyze_tracking.ipynb's raw-scan cell globs and what
    # qubit_params.stats passes to detect_t2_type.
    h5_dir = save_dir

    spec_from_config = spec_params is not None
    centers = {qi: settings[qi]["center"] for qi in qubit_list}
    # Qubits whose Ramsey failed and that need a spectroscopy sweep to relocate
    relocate = set()

    # Resonator windows start on the config readout frequency and follow the
    # same trust gate as the qubit windows.
    _cfg0 = config.load(cfg_dict["cfg_file"])
    res_centers = {qi: _cfg0["device"]["readout"]["frequency"][qi]
                   for qi in qubit_list}
    res_spans = {qi: max(res_span_kappa * _cfg0["device"]["readout"]["kappa"][qi],
                         SPAN_MIN) for qi in qubit_list}
    start_time = time.time()
    elapsed = 0.0
    i = 0

    try:
        while elapsed < total_time and (max_rounds is None or i < max_rounds):
            round_start = time.time()

            # Fridge state once per round, shared by every qubit in it
            mxc_temp, pulsetube_on = np.nan, np.nan
            if bf_client is not None:
                try:
                    mxc_temp = float(bf_client.get_mxc_temperature())
                except Exception as e:
                    print(f"  warning: MXC temperature read failed: {e}")
                try:
                    pulsetube_on = float(bf_client.is_pulsetube_on())
                except Exception as e:
                    print(f"  warning: pulsetube read failed: {e}")
                if verbose:
                    print(f"  round {i}: MXC {1e3*mxc_temp:.1f} mK, "
                          f"pulsetube {'on' if pulsetube_on else 'off'}")

            for qi in qubit_list:
                tm = time.time()
                elapsed = (tm - start_time) / 3600
                if elapsed >= total_time:
                    break

                s = settings[qi]
                span = s["span"]

                # One bad qubit must not end an unattended multi-hour run, so
                # the RF switch and the sweep are guarded together.
                blank_spec = np.full(2, np.nan)
                fit = {"f0": np.nan, "kappa": np.nan, "sigma_f": np.nan,
                       "r2": np.nan, "amp": np.nan, "ok": False, "pars": None}
                freq, sig, spec_raw = blank_spec, blank_spec, {}
                run_spec, relocated = False, None
                try:
                    auto_cfg = config.load(cfg_dict["cfg_file"])
                    rfboard.activate_qubit_rf(qi, soc, auto_cfg,
                                              cfg_file=cfg_dict["cfg_file"],
                                              verbose=False)
                    centers[qi] = auto_cfg["device"]["qubit"]["f_ge"][qi]

                    # Spectroscopy is optional: when Ramsey is the frequency
                    # source it only runs on schedule, or to relocate a qubit
                    # whose Ramsey failed last round.
                    if spec_on_fail and qi in relocate:
                        relocate.discard(qi)
                        print(f"  round {i} Q{qi}: Ramsey lost the qubit, "
                              f"running the relocation ladder")
                        found, f_new = _relocate(
                            cfg_dict, qi, start=relocate_start,
                            reps_cap=relocate_reps_cap, verbose=verbose)
                        if found:
                            centers[qi] = f_new
                            relocated = f_new
                    run_spec = bool(spec_every and i % spec_every == 0)
                    if run_spec:
                        if spec_from_config:
                            params = spec_params
                        else:
                            params = {
                                "gain": s["gain"], "span": span,
                                "start": centers[qi] - span / 2,
                                "expts": expts, "reps": s["reps"],
                                "rounds": s["rounds"], "length": length,
                            }
                        fit, freq, sig, spec_raw, _ = _run_spec(
                            cfg_dict, qi, params, signal=signal,
                            style=spec_style, from_config=spec_from_config)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"  round {i} Q{qi}: sweep failed: {e}")
                    continue

                # Resonator sweep on the same RF-board setting, before the
                # qubit line so both describe the same instant.
                res_fit = None
                if res_every and (i % res_every == 0):
                    try:
                        res_fit, rfreq, rsig, res_raw, _ = _run_res_spec(
                            cfg_dict, qi, res_centers[qi], res_spans[qi],
                            expts=res_expts, reps=res_reps, signal=signal,
                            from_config=res_from_config)
                        # A config-driven run already wrote its own figure
                        if save_images and not res_from_config:
                            _save_sweep_image(
                                img_root / "resonator" / f"q{qi}_r{i:04d}.png",
                                rfreq, rsig, "Readout Frequency (MHz)", signal,
                                fitter.lorfunc, res_fit["pars"])
                    except KeyboardInterrupt:
                        raise
                    except Exception as e:
                        print(f"    Q{qi} resonator sweep failed: {e}")

                d = tracking[qi]
                d["time"].append(tm)
                d["elapsed"].append(elapsed)
                d["center"].append(centers[qi])
                d["gain"].append(s["gain"])
                d["reps"].append(s["reps"])
                d["rounds"].append(s["rounds"])
                d["freq"].append(freq)
                d["sig"].append(sig)
                for k in ("avgi", "avgq", "phases"):
                    if k in spec_raw:
                        d[f"spec_{k}"].append(spec_raw[k])
                # A config-driven sweep already wrote its own figure
                if save_images and not spec_from_config:
                    _save_sweep_image(
                        img_root / "spectroscopy" / f"q{qi}_r{i:04d}.png",
                        freq, sig, f"Q{qi} Drive Frequency (MHz)", signal,
                        fitter.lorfunc, fit["pars"])
                for k in ("f0", "kappa", "sigma_f", "r2", "amp"):
                    d[k].append(fit[k])
                d["ok"].append(float(fit["ok"]))
                # Placeholders; the coherence block below fills them in
                d["f_track"].append(np.nan)
                d["f_track_err"].append(np.nan)
                d["f_track_src"].append(0.0)
                if bf_client is not None:
                    d["mxc_temp"].append(mxc_temp)
                    d["pulsetube_on"].append(pulsetube_on)

                if res_every:
                    rf = res_fit or {"f0": np.nan, "kappa": np.nan,
                                     "sigma_f": np.nan, "r2": np.nan, "ok": False}
                    n_rf = len(d["res_freq"])
                    d["res_center"].append(res_centers[qi])
                    for k in ("f0", "kappa", "sigma_f", "r2"):
                        d[f"res_{k}"].append(rf[k])
                    d["res_ok"].append(float(rf["ok"]))
                    # Keep the arrays rectangular on skipped or failed rounds
                    d["res_freq"].append(rfreq if res_fit else
                                         (d["res_freq"][-1] * np.nan if n_rf
                                          else np.full(res_expts, np.nan)))
                    d["res_sig"].append(rsig if res_fit else
                                        (d["res_sig"][-1] * np.nan if n_rf
                                         else np.full(res_expts, np.nan)))
                    for k in ("avgi", "avgq", "phases"):
                        blank = np.full(res_expts, np.nan)
                        d[f"res_{k}"].append(
                            res_raw.get(k, blank) if res_fit else blank)
                    # In config mode the hanger fit is authoritative and
                    # rf["ok"] already encodes its R^2; the Lorentzian sigma_f
                    # is a bonus check, so a Lorentzian that failed to converge
                    # must not veto a good hanger fit.
                    res_trust = bool(rf["ok"] and (not np.isfinite(rf["sigma_f"])
                                                   or rf["sigma_f"] < trust_sigma_f))
                    if verbose and res_fit:
                        if res_trust:
                            print(f"    Q{qi} resonator: {rf['f0']:.4f} MHz "
                                  f"(+-{1e3*rf['sigma_f']:.1f} kHz), "
                                  f"kappa {rf['kappa']:.4f} MHz")
                        else:
                            print(f"    Q{qi} resonator: untrusted "
                                  f"(r2 {rf['r2']:.2f})")
                    # Config mode re-reads the centre from the config each
                    # sweep, so there is nothing for re-centering to act on
                    if recenter and res_trust and not res_from_config:
                        if abs(rf["f0"] - res_centers[qi]) > recenter_frac * res_spans[qi]:
                            res_centers[qi] = rf["f0"]
                    if update_config and res_trust and res_fit:
                        w = _write_back(cfg_dict["cfg_file"], qi, "resonator",
                                        res_fit.get("expt"), rf["f0"], rf["r2"])
                        if verbose and w:
                            print(f"      -> config: {w}")

                # 'ok' only means the fit converged somewhere inside the
                # window -- a Lorentzian fitted to noise clears it routinely.
                # Anything that acts on f0 (driving T1, moving the window)
                # needs the fit to be *precise*, not merely successful.
                trust = bool(fit["ok"] and fit["sigma_f"] < trust_sigma_f)

                # Adopted frequency for this round; Ramsey overwrites it below
                f_track, f_track_err, f_track_src = np.nan, np.nan, 0.0
                if trust:
                    f_track, f_track_err, f_track_src = (fit["f0"],
                                                         fit["sigma_f"], 1.0)
                elif relocated is not None:
                    f_track, f_track_src = relocated, 1.0

                if verbose and run_spec:
                    if trust:
                        print(f"  round {i} Q{qi} t={elapsed:.2f} hr: spec "
                              f"{fit['f0']:.4f} MHz "
                              f"(+-{1e3*fit['sigma_f']:.1f} kHz), r2 {fit['r2']:.2f}")
                    elif fit["ok"]:
                        print(f"  round {i} Q{qi} t={elapsed:.2f} hr: spec "
                              f"imprecise (+-{1e3*fit['sigma_f']:.0f} kHz), using "
                              f"center {centers[qi]:.4f} MHz")
                    else:
                        print(f"  round {i} Q{qi} t={elapsed:.2f} hr: spec failed "
                              f"(r2 {fit['r2']:.2f})")
                elif verbose:
                    print(f"  round {i} Q{qi} t={elapsed:.2f} hr: "
                          f"drive {centers[qi]:.4f} MHz (config)")

                # Coherence on the frequency this round just measured. Without
                # a trustworthy fit, fall back to the stable calibration
                # center: a pi pulse a few MHz off resonance barely excites the
                # qubit, so a wrong frequency is worse than a slightly stale one.
                # Write f_ge before the coherence sweeps, so the config-driven
                # ones this round already pick up the frequency just fitted
                if update_config and trust:
                    w = _write_back(cfg_dict["cfg_file"], qi, "spectroscopy",
                                    None, fit["f0"], fit["r2"])
                    if verbose and w:
                        print(f"      -> config: {w}")

                f_drive = fit["f0"] if trust else centers[qi]
                for kind, every, phase in coherence:
                    val, vr2, fadj, craw = np.nan, np.nan, np.nan, None
                    if i % every == phase:
                        cfg_driven = kind in coh_from_config
                        try:
                            h5 = (str(h5_dir / f"{_H5_PREFIX[kind]}_qubit{qi}"
                                              f"_{i:05d}") if h5_dir else None)
                            val, vr2, fadj, craw, _ = _run_coherence(
                                cfg_dict, qi, f_drive, kind,
                                expts=coh_expts, reps=coh_reps,
                                from_config=cfg_driven,
                                config_kwargs=(coh_config_kwargs or {}).get(kind),
                                fname=h5)
                            if verbose:
                                extra = ("" if kind == "t1" or not np.isfinite(fadj)
                                         else f", f_ramsey {fadj:.4f} MHz")
                                at = ("config f_ge" if cfg_driven
                                      else f"{f_drive:.4f} MHz")
                                print(f"    Q{qi} {kind}: {val:.1f} us "
                                      f"(r2 {vr2:.2f}) at {at}{extra}")
                            # A config-driven run already wrote its own figure
                            if save_images and not cfg_driven:
                                _save_sweep_image(
                                    img_root / kind / f"q{qi}_r{i:04d}.png",
                                    craw["xpts"], craw.get("avgi", craw["xpts"]),
                                    "Wait Time ($\\mu$s)", "avgi",
                                    craw.get("fitfunc"), craw.get("fitpars"))
                            good = (np.isfinite(val) and val > 0
                                    and np.isfinite(vr2) and vr2 > COH_MIN_R2)

                            # Ramsey measures the detuning from the drive, so
                            # its f_ge beats any spectroscopy fit. It supersedes
                            # whatever the spectroscopy gave, and its failure is
                            # what triggers a relocation sweep next round.
                            if kind == "ramsey":
                                if good and np.isfinite(fadj):
                                    f_track = fadj
                                    f_track_err = (craw or {}).get("f_err", np.nan)
                                    f_track_src = 2.0
                                    relocate.discard(qi)
                                elif spec_on_fail:
                                    relocate.add(qi)
                                    if verbose:
                                        print(f"      Q{qi} Ramsey unusable -- "
                                              f"queued for relocation")
                            if update_config and good:
                                w = _write_back(
                                    cfg_dict["cfg_file"], qi, kind,
                                    (craw or {}).get("expt"), val, vr2, fadj,
                                    update_final_delay=update_final_delay)
                                if verbose and w:
                                    print(f"      -> config: {w}")
                            elif update_config and verbose:
                                print(f"      -> config: {kind} not written "
                                      f"(r2 {vr2:.2f})")
                        except KeyboardInterrupt:
                            raise
                        except Exception as e:
                            print(f"    Q{qi} {kind} failed: {e}")
                    d[kind].append(val)
                    d[f"{kind}_r2"].append(vr2)
                    for sfx in ("off", "amp", "err"):
                        d[f"{kind}_{sfx}"].append((craw or {}).get(sfx, np.nan))
                    if kind != "t1":
                        d[f"{kind}_fadj"].append(fadj)
                    # Raw sweep, NaN-filled on rounds this kind did not run so
                    # the stacked array stays rectangular
                    blank = np.full(coh_expts, np.nan)
                    for k in ("xpts", "avgi", "avgq", "amps"):
                        d[f"{kind}_{k}"].append(
                            craw.get(k, blank) if craw else blank)

                # Record the frequency finally adopted, now that Ramsey has had
                # its chance to supersede the spectroscopy value
                d["f_track"][-1] = f_track
                d["f_track_err"][-1] = f_track_err
                d["f_track_src"][-1] = f_track_src

                # Keep the peak near the middle of the window as it drifts.
                # Gated on 'trust' too: re-centering on an imprecise fit walks
                # the window off into noise and destroys the very fallback the
                # T1 drive frequency depends on.
                # In config mode the window is centred on the config f_ge each
                # sweep, so there is nothing for re-centering to act on
                if recenter and trust and not spec_from_config:
                    if abs(fit["f0"] - centers[qi]) > recenter_frac * span:
                        if verbose:
                            print(f"    Q{qi}: re-centering "
                                  f"{centers[qi]:.4f} -> {fit['f0']:.4f} MHz")
                        centers[qi] = fit["f0"]

            i += 1
            _save_tracking(tracking, save_dir, expt_path=base_path)

            # Idle out the rest of the period so each qubit is sampled on a
            # fixed cadence rather than as fast as the board can go. Never
            # sleep past the end of the run.
            elapsed = (time.time() - start_time) / 3600
            if cadence_s and elapsed < total_time:
                wait = cadence_s - (time.time() - round_start)
                wait = min(wait, (total_time - elapsed) * 3600)
                if wait > 0:
                    print(f"  round {i} done at t={elapsed:.2f} hr; "
                          f"idling {wait:.0f} s to hold the "
                          f"{cadence_s/60:.0f} min cadence")
                    time.sleep(wait)
                elif wait < -1:
                    print(f"  round {i} overran the {cadence_s/60:.0f} min "
                          f"cadence by {-wait:.0f} s")
                elapsed = (time.time() - start_time) / 3600
    except KeyboardInterrupt:
        print(f"Interrupted after {i} rounds; keeping data collected so far.")

    _save_tracking(tracking, save_dir, expt_path=base_path)
    print(f"Saved to {save_dir}")
    return tracking, save_dir


# Column names calib.time_tracking.load_tracking and analyze_tracking.ipynb
# expect. Ramsey becomes t2r and Echo becomes t2, which is what
# collections.process_data uses to derive tsphi and tphi respectively.
_COMPAT_RENAME = {
    "f_track": "f_ge", "f_track_err": "fge_err",
    "ramsey": "t2r", "ramsey_r2": "t2r_r2", "ramsey_err": "t2r_err",
    "ramsey_off": "t2r_off", "ramsey_amp": "t2r_amp",
    "echo": "t2", "echo_r2": "t2_r2", "echo_err": "t2_err",
    "echo_off": "t2_off", "echo_amp": "t2_amp",
}


def _save_tracking_compat(tracking, expt_path, run_id):
    """
    Write a second copy of the per-round scalars where the analysis notebook
    looks for them: <expt_path>/Tracking/csv/<run_id>_qubit_<N>_tracking.csv,
    with columns renamed to the schema load_tracking expects.

    Kept separate from the primary CSVs so the tracking output stays readable
    on its own terms and the notebook needs no changes.
    """
    out = Path(expt_path) / "Tracking" / "csv"
    out.mkdir(parents=True, exist_ok=True)
    for qi, d in tracking.items():
        if not d.get("time"):
            continue
        n = len(d["time"])
        cols = {}
        for k, v in d.items():
            if len(v) != n or np.ndim(v[0]) != 0:
                continue                       # skip the 2D raw sweeps
            cols[_COMPAT_RENAME.get(k, k)] = np.asarray(v, dtype=float)

        # calc_stats indexes t1, t2 and f_ge unconditionally
        if "t1" not in cols or "f_ge" not in cols:
            continue
        if "t2" not in cols:                   # Echo not in the schedule
            cols["t2"] = cols.get("t2r", np.full(n, np.nan))

        names = list(cols)
        rows = np.vstack([cols[k] for k in names]).T
        np.savetxt(out / f"{run_id}_qubit_{qi}_tracking.csv", rows,
                   delimiter=",", header=",".join(names), comments="",
                   fmt="%.6f")


def _stack_ragged(rows):
    """
    Stack 1D sweeps into a 2D array, NaN-padding to the longest.

    Row lengths can differ: a config-driven experiment uses whatever ``expts``
    the config gives it, which need not match the placeholder width used on
    rounds where that measurement did not run.
    """
    arrs = [np.atleast_1d(np.asarray(r, dtype=float)) for r in rows]
    w = max(a.size for a in arrs)
    out = np.full((len(arrs), w), np.nan)
    for j, a in enumerate(arrs):
        out[j, :a.size] = a
    return out


def _save_tracking(tracking, save_dir, expt_path=None):
    """
    Write one CSV of scalars per qubit plus a combined npz with the spectra,
    and a notebook-compatible copy under <expt_path>/Tracking/csv/.
    """
    save_dir = Path(save_dir)
    npz = {}
    for qi, d in tracking.items():
        if not d["time"]:
            continue
        # One row per round in the CSV; anything per-sweep (2D) goes to the npz
        # only. Which raw keys exist depends on the schedule.
        n = len(d["time"])
        scalar_keys = [k for k, v in d.items()
                       if len(v) == n and np.ndim(v[0]) == 0]
        rows = np.vstack([np.asarray(d[k], dtype=float) for k in scalar_keys]).T
        np.savetxt(save_dir / f"qubit_{qi}_freq_tracking.csv", rows,
                   delimiter=",", header=",".join(scalar_keys), comments="",
                   fmt="%.6f")
        for k, v in d.items():
            if not v:
                continue
            npz[f"q{qi}_{k}"] = (np.asarray(v, dtype=float) if k in scalar_keys
                                 else _stack_ragged(v))
    if npz:
        np.savez_compressed(save_dir / "freq_tracking.npz", **npz)

    # Second copy where analyze_tracking.ipynb looks. Never let a schema
    # mismatch here take down the run -- the primary CSVs are already written.
    if expt_path:
        try:
            _save_tracking_compat(tracking, expt_path, save_dir.name)
        except Exception as e:
            print(f"  [compat csv] skipped: {e}")


def plot_tracking(tracking, save_dir, signal_label="|S21| (ADC units)",
                  t1_min_r2=0.5, t1_max_us=5000.0):
    """
    Two figures: f0(t) for every qubit, and a per-qubit spectroscopy map.

    The maps use 2D coordinates because re-centering lets each row have its own
    frequency axis. Saved to <save_dir>/images, never shown (script-safe).

    A T1 fit on a qubit with no visible line still returns a number -- negative,
    or thousands of microseconds -- so T1 points are kept only when the fit
    clears ``t1_min_r2`` and lands in (0, ``t1_max_us``).
    """
    save_dir = Path(save_dir)
    img_dir = save_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    qubits = [qi for qi, d in tracking.items() if len(d["time"]) > 0]
    if not qubits:
        print("No data to plot.")
        return

    # --- f0 vs time, one panel per qubit --------------------------------
    n = len(qubits)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, 3 * nrow), squeeze=False)
    for ax, qi in zip(axs.flat, qubits):
        d = tracking[qi]
        t = np.asarray(d["elapsed"], dtype=float)
        # Prefer the adopted frequency (Ramsey when available, else
        # spectroscopy); fall back to the raw spectroscopy fit for older data
        if "f_track" in d:
            f0 = np.asarray(d["f_track"], dtype=float)
            sf = np.asarray(d["f_track_err"], dtype=float)
            src = np.asarray(d["f_track_src"], dtype=float)
            ok = np.isfinite(f0) & (src > 0)
        else:
            f0 = np.asarray(d["f0"], dtype=float)
            sf = np.asarray(d["sigma_f"], dtype=float)
            src = np.zeros_like(f0)
            ok = np.asarray(d["ok"], dtype=float) > 0.5
        if np.any(ok):
            yerr = sf[ok] if np.all(np.isfinite(sf[ok])) else None
            ax.errorbar(t[ok], f0[ok], yerr=yerr, fmt=".-", ms=4, lw=1,
                        elinewidth=0.8, capsize=0)
            drift = 1e3 * (np.nanmax(f0[ok]) - np.nanmin(f0[ok]))
            std = 1e3 * np.nanstd(f0[ok])
            n_ram = int(np.sum(src[ok] == 2))
            ax.set_ylabel(f"Q{qi} $f_{{ge}}$ (MHz)")
            ax.text(0.02, 0.95, f"drift {drift:.0f} kHz, std {std:.0f} kHz\n"
                                f"{int(ok.sum())}/{len(ok)} good "
                                f"({n_ram} Ramsey)",
                    transform=ax.transAxes, va="top", fontsize=8)
        else:
            ax.text(0.5, 0.5, f"Q{qi}: no good fits", transform=ax.transAxes,
                    ha="center")
        ax.set_xlabel("Lab Time (hr)")
    for ax in axs.flat[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(img_dir / "freq_vs_time.png", dpi=150)
    plt.close(fig)

    # --- resonator frequency vs time, if it was tracked -----------------
    if any(np.any(np.asarray(tracking[qi].get("res_ok", []), dtype=float) > .5)
           for qi in qubits):
        fig, axs = plt.subplots(nrow, ncol, figsize=(12, 3 * nrow), squeeze=False)
        for ax, qi in zip(axs.flat, qubits):
            d = tracking[qi]
            t = np.asarray(d["elapsed"], dtype=float)
            rf = np.asarray(d.get("res_f0", []), dtype=float)
            rs = np.asarray(d.get("res_sigma_f", []), dtype=float)
            m = (np.asarray(d.get("res_ok", []), dtype=float) > .5) & np.isfinite(rf)
            if np.any(m):
                ax.errorbar(t[m], rf[m], yerr=rs[m], fmt=".-", ms=4, lw=1,
                            elinewidth=0.8, capsize=0)
                ax.text(0.02, 0.95,
                        f"drift {1e3*np.ptp(rf[m]):.0f} kHz, "
                        f"std {1e3*np.std(rf[m]):.0f} kHz\n"
                        f"{int(m.sum())}/{len(rf)} good",
                        transform=ax.transAxes, va="top", fontsize=8)
            else:
                ax.text(0.5, 0.5, f"Q{qi}: no usable resonator fits",
                        transform=ax.transAxes, ha="center")
            ax.set_xlabel("Lab Time (hr)")
            ax.set_ylabel(f"Q{qi} $f_r$ (MHz)")
        for ax in axs.flat[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(img_dir / "resonator_vs_time.png", dpi=150)
        plt.close(fig)

    # --- one figure per coherence kind present in the data --------------
    def _coh_mask(d, kind):
        v = np.asarray(d.get(kind, []), dtype=float)
        if v.size == 0:
            return v, np.zeros(0, dtype=bool)
        r2 = np.asarray(d.get(f"{kind}_r2", np.full(v.shape, np.nan)), dtype=float)
        return v, (np.isfinite(v) & (v > 0) & (v < t1_max_us) & (r2 > t1_min_r2))

    labels = {"t1": "$T_1$", "ramsey": "$T_2^*$", "echo": "$T_2^E$"}
    for kind, label in labels.items():
        if not any(np.any(_coh_mask(tracking[qi], kind)[1]) for qi in qubits):
            continue
        fig, axs = plt.subplots(nrow, ncol, figsize=(12, 3 * nrow), squeeze=False)
        for ax, qi in zip(axs.flat, qubits):
            d = tracking[qi]
            t = np.asarray(d["elapsed"], dtype=float)
            v, m = _coh_mask(d, kind)
            if np.any(m):
                ax.plot(t[m], v[m], ".-", ms=5, lw=1)
                ax.text(0.02, 0.95,
                        f"mean {np.mean(v[m]):.1f} us, "
                        f"std {np.std(v[m]):.1f} us\n"
                        f"{int(m.sum())}/{int(np.isfinite(v).sum())} kept",
                        transform=ax.transAxes, va="top", fontsize=8)
            else:
                ax.text(0.5, 0.5, f"Q{qi}: no usable {kind} fits",
                        transform=ax.transAxes, ha="center")
            ax.set_xlabel("Lab Time (hr)")
            ax.set_ylabel(f"Q{qi} {label} ($\\mu$s)")
        for ax in axs.flat[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(img_dir / f"{kind}_vs_time.png", dpi=150)
        plt.close(fig)

    # --- spectroscopy maps ----------------------------------------------
    # With Ramsey as the frequency source, spectroscopy only runs to relocate a
    # lost qubit, so most rows are NaN placeholders. pcolormesh rejects
    # non-finite coordinates outright, so keep only the rows that really ran --
    # and skip the figure entirely if none did.
    def _spec_rows(d):
        F = np.atleast_2d(np.asarray(d["freq"], dtype=float))
        Z = np.atleast_2d(np.asarray(d["sig"], dtype=float))
        keep = np.all(np.isfinite(F), axis=1) & np.any(np.isfinite(Z), axis=1)
        return F, Z, keep

    if any(np.any(_spec_rows(tracking[qi])[2]) for qi in qubits):
        fig, axs = plt.subplots(nrow, ncol, figsize=(12, 3.5 * nrow),
                                squeeze=False)
        for ax, qi in zip(axs.flat, qubits):
            d = tracking[qi]
            t = np.asarray(d["elapsed"], dtype=float)
            F, Z, keep = _spec_rows(d)
            if np.sum(keep) >= 1:
                Tk = np.repeat(t[keep][:, None], F.shape[1], axis=1)
                pcm = ax.pcolormesh(Tk, F[keep], np.nan_to_num(Z[keep]),
                                    shading="nearest", cmap="viridis",
                                    rasterized=True)
                plt.colorbar(pcm, ax=ax, label=signal_label)
                f0 = np.asarray(d["f0"], dtype=float)
                ok = (np.asarray(d["ok"], dtype=float) > 0.5) & np.isfinite(f0)
                if np.any(ok):
                    ax.plot(t[ok], f0[ok], "r.-", lw=0.8, ms=2, zorder=5)
            else:
                ax.text(0.5, 0.5, f"Q{qi}: no spectroscopy sweeps",
                        transform=ax.transAxes, ha="center")
            ax.set_xlabel("Lab Time (hr)")
            ax.set_ylabel(f"Q{qi} Drive Frequency (MHz)")
        for ax in axs.flat[n:]:
            ax.set_visible(False)
        fig.tight_layout()
        fig.savefig(img_dir / "spec_maps.png", dpi=150)
        plt.close(fig)

    print(f"Saved plots to {img_dir}")
