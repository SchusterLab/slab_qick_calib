"""
f0-g1 unconditional microwave reset.

Implements the calibration sequence of Magnard et al., arXiv:1801.07689, for an
all-microwave unconditional reset of a transmon qutrit. A drive at

    f_f0g1 = f_ge + f_ef - f_resonator

converts |f,0> -> |g,1>, and the resonator then emits the photon at rate kappa.
Driving e-f simultaneously funnels |e> into |f> so any excitation in the three
lowest levels is removed. Unlike the paper, the *readout* resonator (the waste
line, readout index 1) doubles as the reset resonator -- the paper's closing
paragraph argues that is preferable, since larger kappa and larger g both speed
the reset up.

The drive lives on its own DAC (``hw.soc.dacs.f0g1``), combined onto the qubit
line, because f_f0g1 is a few hundred MHz here -- far from anything the qubit
DAC's filter passes.

Module contents, covering all four calibration steps:

- declare_f0g1_gen: declare the reset-drive generator with the right nqz/mixer
- f0g1_design_frequency: f_ge + f_ef - f_waste from the config
- f0g1_frequency_for_gain / omega_ef_for_gain: invert the stored fits
- reset_rate / optimal_omega_ef: the paper's Eq. 1 reset rate and its optimum
  (the Fig. 3a red curve)
- F0G1Program: the pulse sequence, shared by steps 1, 3 and 4

| step | 1D | 2D vs power | writes |
|------|----|-------------|--------|
| 1 | F0G1Spec                  | F0G1SpecPower   | stark_fit          |
| 2 | RabiExperiment (existing) | EFRabiPower     | omega_ef_fit       |
| 3 | EFSpecF0G1                | EFSpecF0G1Power | ef_stark_fit       |
| 4 | F0G1Rabi                  | F0G1RabiPower   | g_tilde_fit, kappa |

Steps 3 and 4 read back the fits the earlier steps wrote, so the drives stay on
their Stark-shifted resonances; the helpers raise naming the script to run
rather than proceeding with a NaN.

Everything is in MHz, i.e. rates are quoted as nu = Omega / 2*pi. That matches
how the configs store frequencies and how ``fitter.fitsin`` reports a Rabi rate.
"""

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import QickSweep1D
from scipy.stats import f as f_dist

from ...analysis import fitting as fitter
from ...exp_handling.datamanagement import AttrDict
from ...helpers import config
from ..general.qick_experiment import QickExperiment, QickExperiment2DSimple
from ..general.qick_program import (
    QickProgram,
    declare_f0g1_gen,
    pulse_type_from_config,
    ramp_length,
    reset_mode_of,
    reset_plays_pi_pulses,
)
# The config math and the generator declaration moved down so the base
# classes can use them (this module imports from them, so they cannot
# import back). Re-exported here, unchanged, so every existing import
# and all of scripts 27-35 keep working.
from ...helpers.f0g1 import (  # noqa: F401
    ef_gain_for_omega,
    f0g1_design_frequency,
    f0g1_frequency_for_gain,
    f0g1_reset_params,
    fit_reset_trajectory,
    g_tilde_for_gain,
    omega_ef_for_gain,
    optimal_omega_ef,
    reset_rate,
    reset_trajectory,
)
# The FWM band tracker solves the same problem this module's fits have -- a
# sigma-clipped straight line through one value per row -- so it is shared
# rather than reimplemented. Every model here is `value = slope*design + offset`
# (design = gain**2 for a Stark shift, gain for a Rabi rate).
from ..single_qubit.qubit_spec_shot import _fit_band_model
from ..single_qubit.rabi import Rabi2D
# The f0-g1 drive dresses |f,0> with |g,1>, so the e-f line comes out as
# a doublet split by 2*g~. StarkSpec's photon-number-peak fitter is the
# same shape -- two Lorentzians of shared width, bounded separation --
# so it is reused rather than reimplemented.
from ..single_qubit.stark_spectroscopy import (
    fit_two_lorentzians,
    two_lorentzians,
)

FIT_FUNC = fitter.lorfunc
FITTER_FUNC = fitter.fitlor
XLABEL = "f0-g1 Drive Frequency (MHz)"
YLABEL = "f0-g1 Drive Gain (DAC level)"


def _per_row_r2(xpts, rows, fits, fitfunc):
    """
    Coefficient of determination of each row's fit.

    The rejection criterion both 2D classes here need. A failed ``curve_fit``
    returns [nan]*n, which is easy to spot, but the dangerous case is a row
    with no signal at all: the fit converges onto noise and comes back with
    perfectly plausible parameters. R2 ~ 0 is what identifies those.

    Returns an array of R2, NaN where the fit parameters were unusable.
    """
    r2 = np.full(len(rows), np.nan)
    for i, pars in enumerate(fits):
        pars = np.asarray(pars, dtype=float)
        if not len(pars) or not np.all(np.isfinite(pars)):
            continue
        row = np.asarray(rows[i], dtype=float)
        ss_tot = np.sum((row - row.mean()) ** 2)
        if ss_tot > 0:
            ss_res = np.sum((row - fitfunc(xpts, *pars)) ** 2)
            r2[i] = 1 - ss_res / ss_tot
    return r2


def _analyze_quadratic_stark(data, key, label, min_r2=0.1):
    """
    Fit one Lorentzian centre per row, then the centres to a*gain**2 + c.

    Shared by the two ac Stark measurements -- the f0-g1 transition pulled by
    its own drive (step 1) and the e-f transition pulled by the f0-g1 drive
    (step 3). Both are forced-vertex parabolas (no shift at zero drive), i.e.
    straight lines in gain**2, and both are read off a per-row Lorentzian, so
    the only differences are the data keys and the printed label.

    Assumes ``data["fit_avgi"]`` is already populated by the caller's per-row
    Lorentzian fit. Adds ``<key>_center``, ``<key>_gamma``, ``<key>_r2``,
    ``<key>_keep`` and, when at least 3 rows survive, ``<key>_fit_a``,
    ``<key>_fit_c`` and ``<key>_resid``.

    Two independent rejection stages, because a row can be bad in two
    different ways:

    1. ``min_r2`` drops rows where the Lorentzian did not describe the data.
       ``fitlor`` returns [nan]*4 only when curve_fit raises; a row with no
       peak at all converges happily onto a noise spike, lands its centre
       inside the window with a plausible linewidth, and is identifiable only
       by R2 ~ 0. Default 0.1 matches ``QickExperiment.run``'s own floor.
    2. Sigma-clipping drops centres that are real fits but off the quadratic
       -- a row that locked onto a neighbouring feature, say.
    """
    gains = np.asarray(data["ypts"], dtype=float)
    xpts = np.asarray(data["xpts"], dtype=float)
    n = len(gains)

    centers = np.full(n, np.nan)
    gammas = np.full(n, np.nan)
    for i, pars in enumerate(data["fit_avgi"]):
        pars = np.asarray(pars, dtype=float)
        if len(pars) < 4 or not np.all(np.isfinite(pars)):
            continue
        centers[i] = pars[2]
        gammas[i] = pars[3]
    r2 = _per_row_r2(xpts, data["avgi"], data["fit_avgi"], FIT_FUNC)

    data[f"{key}_center"] = centers
    data[f"{key}_gamma"] = gammas
    data[f"{key}_r2"] = r2

    xmin, xmax = float(np.min(xpts)), float(np.max(xpts))
    good = (
        np.isfinite(centers)
        & (centers > xmin)
        & (centers < xmax)
        & np.isfinite(r2)
        & (r2 > min_r2)
    )
    n_bad_fit = int(np.sum(~good))
    if n_bad_fit:
        print(
            f"{label} Stark fit: dropped {n_bad_fit} of {n} rows with no usable "
            f"Lorentzian (R2 <= {min_r2} or centre outside the scan window)"
        )

    if good.sum() < 3:
        print(f"{label} Stark fit: fewer than 3 usable rows, not fitting")
        data[f"{key}_keep"] = good
        return data

    return _fit_stark_quadratic(data, key, label, gains, centers, good)


def _fit_stark_quadratic(data, key, label, gains, centers, good):
    """
    Sigma-clipped fit of per-row centres to ``freq = a*gain**2 + c``.

    Shared by the single-peak path (step 1, and step 3 before the doublet
    resolves) and the doublet path, which differ only in how the centre of
    each row was obtained.
    """
    n = len(gains)
    a, c, keep_sub = _fit_band_model(
        gains[good] ** 2, centers[good], n_iter=4, sigma=3.0
    )
    keep = np.zeros(n, dtype=bool)
    keep[np.flatnonzero(good)[keep_sub]] = True
    data[f"{key}_keep"] = keep

    n_clipped = int(good.sum() - keep.sum())
    if n_clipped:
        print(
            f"{label} Stark fit: sigma-clipped {n_clipped} further rows off the "
            "quadratic"
        )

    resid = centers[keep] - (a * gains[keep] ** 2 + c)
    data[f"{key}_fit_a"] = float(a)
    data[f"{key}_fit_c"] = float(c)
    data[f"{key}_resid"] = float(np.median(np.abs(resid)))
    print(
        f"{label} ac Stark shift: freq = {a:.4f} * gain^2 + {c:.4f} MHz "
        f"(median |residual| {data[f'{key}_resid']:.4f} MHz, "
        f"{int(keep.sum())} of {n} rows)"
    )
    return data


def fit_ef_doublet(x, y, delta_guess, min_r2=0.1):
    """
    Fit the e-f line as a dressed-state doublet, falling back to one peak.

    The f0-g1 drive coherently couples |f,0> and |g,1>, so |f,0> hybridizes
    into two dressed states at +/- g~ and the e-f probe sees both. The pair is
    therefore split by 2*g~ and the *centre* is the bare (Stark-shifted) e-f
    frequency -- which is what the reset needs, since Eq. 1 assumes each drive
    resonant with its bare transition and produces the splitting as an output.
    Driving a peak instead is delta_ef = +/- g~ and costs most of Gamma.

    Two peaks are only preferred when the pair is believable. Six free
    parameters against four means the doublet can always fit at least as well
    as one Lorentzian, usually by parking a small satellite on noise, so a
    better R2 is not evidence. The doublet is accepted only when its two
    amplitudes have the same sign and are within 4x of each other (both
    dressed states carry half the |f,0> character, so a real pair is nominally
    1:1), the separation exceeds its own linewidth (below 2*g~ ~ kappa/2 the
    states merge and the fit is degenerate), and the improvement passes an
    F-test at 95% for the two extra parameters.

    Args:
        x: probe frequencies (MHz)
        y: measured signal for one row
        delta_guess: expected splitting (MHz), i.e. 2*g~ at this drive gain
        min_r2: floor below which a row counts as unfitted

    Returns:
        dict with 'center', 'splitting' (NaN when unresolved), 'gamma', 'r2',
        'n_peaks' (1 or 2) and 'pars' (the winning parameter vector).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    def r2_of(model):
        if model is None or ss_tot <= 0:
            return -np.inf
        return 1 - float(np.sum((y - model) ** 2)) / ss_tot

    one = None
    try:
        p1, _, _ = FITTER_FUNC(x, y)
        p1 = np.asarray(p1, dtype=float)
        if np.all(np.isfinite(p1)):
            one = p1
    except Exception:
        pass

    two = None
    if delta_guess and np.isfinite(delta_guess):
        try:
            p2, _ = fit_two_lorentzians(x, y, delta_guess=float(delta_guess))
            if p2 is not None and np.all(np.isfinite(p2)):
                two = np.asarray(p2, dtype=float)
        except Exception:
            pass

    r2_one = r2_of(FIT_FUNC(x, *one)) if one is not None else -np.inf
    r2_two = r2_of(two_lorentzians(x, *two)) if two is not None else -np.inf

    # Believing the doublet needs more than a better R2: two Lorentzians have
    # six free parameters against four, so they can always fit at least as
    # well, typically by parking a small satellite on noise. Three checks, the
    # first two from the physics of the dressed pair:
    #
    #   same sign, comparable amplitude -- both dressed states carry half the
    #     |f,0> character, so a genuine pair is nominally 1:1
    #   separated by more than its own linewidth -- below 2*g~ ~ kappa/2 the
    #     states merge and a two-peak fit of one line is degenerate
    #   an F-test at 95% on the two extra parameters
    #
    # The F-test rather than a fixed "R2 must improve by X%": the two models
    # are nested (a doublet with delta=0 *is* a Lorentzian), so this is the
    # textbook comparison, and it scales with how many points the row has --
    # which a fixed fraction ignores. On the real #364 scan (100 points) a
    # 20% rule rejected the gain-0.8 row at F = 8.9 against F_crit = 3.1.
    accepted = False
    f_stat = np.nan
    if two is not None:
        y0, a0, a1, x0, delta, gamma = two
        same_sign = a0 * a1 > 0
        ratio = min(abs(a0), abs(a1)) / max(abs(a0), abs(a1), 1e-30)
        resolved = abs(delta) > abs(gamma)
        # fit_two_lorentzians bounds delta to 0.3-3x the seed. A fit that
        # comes back sitting on a bound was constrained by the bound, not by
        # the data -- the usual cause is a broad shallow line with no real
        # structure, where the pair runs apart until it is stopped. Measured
        # on #366 at f0g1 gain 1.0: delta = 4.945 against a 4.944 ceiling,
        # a 3x overstatement of 2*g~ that flipped the whole Stark fit's sign
        # because gain^2 gives that row maximum leverage.
        lo_b, hi_b = sorted((0.3 * float(delta_guess), 3.0 * float(delta_guess)))
        pinned = (abs(abs(delta) - hi_b) / max(hi_b, 1e-30) < 0.02
                  or abs(abs(delta) - lo_b) / max(lo_b, 1e-30) < 0.02)
        dof = len(x) - 6
        if r2_one > -np.inf and dof > 0 and r2_two < 1:
            f_stat = ((r2_two - r2_one) / 2) / ((1 - r2_two) / dof)
            significant = f_stat > f_dist.ppf(0.95, 2, dof)
        else:
            significant = r2_two > r2_one
        accepted = (same_sign and ratio > 0.25 and resolved
                    and significant and not pinned)

    if accepted:
        # The *amplitude-weighted* centroid, not the midpoint of the two peaks.
        # Those agree for a 1:1 pair, but if the f0-g1 drive is detuned from
        # its own resonance by delta the dressed states move to
        # delta/2 +/- sqrt(g~^2 + delta^2/4) with unequal |f,0> character, and
        # the midpoint drifts by delta/2 while the weighted centroid stays put
        # -- exactly, by the sum rule sum_i |<f,0|i>|^2 E_i = <f,0|H|f,0>.
        # Verified: at a 1.6 MHz f0-g1 detuning the midpoint is 800 kHz out and
        # the weighted centroid 0 kHz.
        w = two[1] + two[2]
        center = two[3] + two[4] * (two[2] / w if w != 0 else 0.5)
        return {
            "center": float(center),
            "splitting": float(abs(two[4])),
            "gamma": float(abs(two[5])),
            "r2": float(r2_two),
            "f_stat": float(f_stat),
            "n_peaks": 2,
            "pars": two,
        }
    if one is not None:
        return {
            "center": float(one[2]),
            "splitting": np.nan,
            "gamma": float(abs(one[3])),
            "r2": float(r2_one),
            "f_stat": float(f_stat),
            "n_peaks": 1,
            "pars": one,
        }
    return {"center": np.nan, "splitting": np.nan, "gamma": np.nan,
            "r2": np.nan, "f_stat": np.nan, "n_peaks": 0, "pars": None}


def _analyze_ef_doublet(data, cfg, min_r2=0.1):
    """
    Per-row doublet fit of the e-f line, then the centres and the splitting.

    Adds to ``data``:
      ef_stark_center / _splitting / _gamma / _r2 / _npeaks  (per row)
      ef_stark_fit_a / _fit_c / _resid / _keep               the quadratic
      g_tilde_split_m / _b / _resid                          g~ from the pair

    The splitting is an independent measurement of g~ -- a frequency rather
    than a rate -- so it is reported alongside ``device.f0g1.g_tilde_fit``
    from the step-4 Rabi. It is not written to the config: the two should be
    reconciled deliberately, not silently overwritten.
    """
    gains = np.asarray(data["ypts"], dtype=float)
    xpts = np.asarray(data["xpts"], dtype=float)
    n = len(gains)

    # Seed each row's splitting from the step-4 fit at that gain. A seed is
    # required (fit_two_lorentzians bounds delta to 0.3-3x it), so fall back to
    # a fraction of the span when g_tilde_fit is absent.
    fit = cfg.device.get("f0g1", {}).get("g_tilde_fit", None)
    span = float(xpts.max() - xpts.min())
    if fit is not None and np.all(np.isfinite(np.array(fit, dtype=float))):
        gm, gb = (float(v) for v in fit)
        seeds = 2 * (gm * gains + gb)
        seed_src = f"2*g~ from g_tilde_fit ({gm:.3f}*gain + {gb:+.3f})"
    else:
        seeds = np.full(n, span / 10)
        seed_src = f"span/10 = {span / 10:.3f} MHz (no g_tilde_fit to seed from)"
    seeds = np.where(np.abs(seeds) > 1e-6, seeds, span / 10)
    print(f"e-f doublet: splitting seeded from {seed_src}")

    centers = np.full(n, np.nan)
    splits = np.full(n, np.nan)
    gammas = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    npk = np.zeros(n, dtype=int)
    for i in range(n):
        res = fit_ef_doublet(xpts, data["avgi"][i], seeds[i], min_r2=min_r2)
        centers[i], splits[i] = res["center"], res["splitting"]
        gammas[i], r2[i], npk[i] = res["gamma"], res["r2"], res["n_peaks"]

    data["ef_stark_center"] = centers
    data["ef_stark_splitting"] = splits
    data["ef_stark_gamma"] = gammas
    data["ef_stark_r2"] = r2
    data["ef_stark_npeaks"] = npk
    n_two = int(np.sum(npk == 2))
    print(f"  {n_two} of {n} rows resolved into two peaks, "
          f"{int(np.sum(npk == 1))} fit as one")

    xmin, xmax = float(np.min(xpts)), float(np.max(xpts))
    good = (
        np.isfinite(centers) & (centers > xmin) & (centers < xmax)
        & np.isfinite(r2) & (r2 > min_r2)
    )
    n_bad = int(np.sum(~good))
    if n_bad:
        print(f"  dropped {n_bad} of {n} rows (R2 <= {min_r2} or centre outside "
              "the scan window)")

    if good.sum() >= 3:
        _fit_stark_quadratic(data, "ef_stark", "e-f", gains, centers, good)
    else:
        print("  fewer than 3 usable rows, not fitting the Stark shift")
        data["ef_stark_keep"] = good

    # g~ from the splitting: splitting = 2*g~, so g~ = m*gain + b with
    # m = slope/2. Sigma-clipped like everything else here.
    res_ok = good & np.isfinite(splits) & (splits > 0)
    if res_ok.sum() >= 3:
        m2, b2, keep_sub = _fit_band_model(
            gains[res_ok], splits[res_ok], n_iter=4, sigma=3.0
        )
        keep = np.zeros(n, dtype=bool)
        keep[np.flatnonzero(res_ok)[keep_sub]] = True
        data["g_tilde_split_keep"] = keep
    else:
        keep = np.zeros(n, dtype=bool)

    # The clipping can leave too few rows to carry a line even when enough
    # were resolved: with 3 resolved rows, 4 passes at 3 sigma can discard all
    # of them, and _fit_band_model still returns the coefficients from its last
    # iteration. Reporting those would quote a slope no retained point
    # supports -- on the real #364 scan that read g~ = 1.088*gain - 0.404,
    # a negative intercept, which is unphysical (g~ must vanish with the
    # drive). Gate on what survived, not on what went in.
    if keep.sum() >= 3:
        data["g_tilde_split_m"] = float(m2 / 2)
        data["g_tilde_split_b"] = float(b2 / 2)
        resid = splits[keep] - (m2 * gains[keep] + b2)
        data["g_tilde_split_resid"] = float(np.median(np.abs(resid)) / 2)
        print(
            f"  splitting = {m2:.4f}*gain + {b2:+.4f} MHz over "
            f"{int(keep.sum())} rows -> g~/2pi = {m2 / 2:.4f}*gain "
            f"{b2 / 2:+.4f} MHz"
        )
        if fit is not None and np.all(np.isfinite(np.array(fit, dtype=float))):
            g_sp = m2 / 2 + b2 / 2
            g_rb = gm + gb
            print(
                f"  at gain 1: g~ = {g_sp:.4f} MHz from the splitting vs "
                f"{g_rb:.4f} from the step-4 Rabi ({g_sp / g_rb:.2f}x)"
            )
            if not 0.75 < g_sp / g_rb < 1.33:
                print(
                    "  NOTE: those disagree by more than 33%. The splitting is a "
                    "frequency and normally the more reliable of the two -- but "
                    "check it is sampled finely enough (several points per "
                    "linewidth) before preferring it over the Rabi fit."
                )
    else:
        print(
            f"  only {int(res_ok.sum())} rows resolved into a believable "
            "doublet and fewer than 3 survived clipping, so g~ is not fit "
            "from the splitting. Sample the e-f line more finely (several "
            "points per linewidth) to resolve the pair at more drive gains."
        )

    return data

def _display_quadratic_stark(expt, data, key, ylabel, suffix):
    """Fitted centre and linewidth vs drive gain, with the quadratic fit."""
    gains = np.asarray(data["ypts"], dtype=float)
    keep = data.get(f"{key}_keep", np.isfinite(data[f"{key}_center"]))

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    axes[0].plot(gains[keep], data[f"{key}_center"][keep], "o", label="used")
    if np.any(~keep):
        axes[0].plot(gains[~keep], data[f"{key}_center"][~keep], "x",
                     color="grey", label="rejected")
    if f"{key}_fit_a" in data:
        gfine = np.linspace(0, gains.max(), 200)
        axes[0].plot(
            gfine,
            data[f"{key}_fit_a"] * gfine**2 + data[f"{key}_fit_c"],
            "-",
            label=(
                f"{data[f'{key}_fit_a']:.3f}$g^2$ + "
                f"{data[f'{key}_fit_c']:.3f}"
            ),
        )
    axes[0].set_ylabel(ylabel)
    axes[0].legend()

    axes[1].plot(gains[keep], data[f"{key}_gamma"][keep], "o")
    axes[1].set_ylabel("Linewidth $\\kappa$ (MHz)")
    axes[1].set_xlabel(YLABEL)

    expt.save_fig(fig, suffix=suffix)


class F0G1Program(QickProgram):
    """
    Pulse sequence for the f0-g1 reset calibration.

    State prep -> f0-g1 drive on its own DAC (optionally with an e-f probe
    nested inside it) -> gap -> optional shelving pi_ge -> readout:

        steps 1/4   prep="f", ef_probe_on=False, shelve_ge=False
                    pi_ge -> pi_ef -> f0g1 -> gap -> measure

        step 3      prep="e", ef_probe_on=True,  shelve_ge=True
                    pi_ge -> f0g1 [... ef probe nested ...] -> gap
                          -> pi_ge -> measure

    Steps 1 and 4 need no pulse after the drive: the observable is "did the
    population come back to |g>", and |e> and |f> are not separable on this
    readout anyway (chi >> kappa puts those two blobs ~0.5 sigma apart), so
    both read as not-g -- which is all that is needed.

    Step 3's contrast *is* e-vs-f, which this readout cannot resolve, so it
    needs ``shelve_ge``. Tracing both outcomes with the f0-g1 drive on
    resonance:

        off resonance  |e> --(no ef rotation)--> |e> --pi_ge--> |g>    reads g
        on resonance   |e> --ef pi--> |f> --f0g1--> |g,1> -> |g,0>
                                            --pi_ge--> |e>            reads not-g

    so the two paths separate cleanly even though the on-resonance population
    ends up in |g> before the shelving pulse rather than in |f>.

    The e-f probe is nested so that both tones *end* together (the
    ``StarkSpec`` idiom): the f0-g1 drive starts first, and ``self.delay`` --
    which increments the reference time, unlike ``delay_auto`` which sets it
    -- pushes the probe out by the difference of the two total lengths.

    ``readout_gap`` is not cosmetic: the reset drive's whole point is to leave a
    photon in the readout resonator, and reading out before it has leaked away
    measures the photon instead of the qubit. A few 1/kappa is enough. The
    shelving pulse is placed *after* that gap so it is not detuned by the
    photon's own Stark shift.
    """

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        q = cfg.expt.qubit[0]

        self.frequency = cfg.device.readout.frequency[q]
        self.gain = cfg.device.readout.gain[q]
        self.readout_length = cfg.expt.readout_length
        self.phase = cfg.device.readout.phase[q]

        super()._initialize(cfg, readout="standard")

        self.f0g1_ch = declare_f0g1_gen(self, cfg)

        f0g1_pulse = {
            "chan": self.f0g1_ch,
            "freq": cfg.expt.frequency_f0g1,
            "gain": cfg.expt.gain_f0g1,
            "phase": 0,
            "type": cfg.expt.pulse_type_f0g1,
            "length": cfg.expt.length_f0g1,
            "ramp_sigma": cfg.expt.ramp_sigma_f0g1,
            "ramp_sigma_inc": cfg.expt.ramp_sigma_inc_f0g1,
        }
        super().make_pulse(f0g1_pulse, "f0g1_pulse")

        if cfg.expt.ef_probe_on:
            # On the qubit channel, so it shares the "ramp" envelope name with
            # pi_ge. That is safe: add_envelope allocates a fresh address per
            # call and add_pulse bakes it in immediately, so a later envelope
            # of the same name cannot alter an earlier pulse.
            ef_pulse = {
                "chan": self.qubit_ch,
                "freq": cfg.expt.frequency_ef,
                "gain": cfg.expt.gain_ef,
                "phase": 0,
                "type": cfg.expt.pulse_type_ef,
                "length": cfg.expt.length_ef,
                "ramp_sigma": cfg.expt.ramp_sigma_ef,
                "ramp_sigma_inc": cfg.expt.ramp_sigma_inc_ef,
            }
            super().make_pulse(ef_pulse, "ef_pulse")

        self.add_loop("sweep_loop", cfg.expt.expts)  # generic: freq for 1/3, length for 4

        # pi_ge is needed for prep into |e> or |f> and for the shelving pulse,
        # and is also what QickProgram.reset() plays, so a measurement-based
        # reset needs it declared too.
        if (cfg.expt.prep in ("e", "f") or cfg.expt.shelve_ge
                or reset_plays_pi_pulses(cfg.expt)):
            super().make_cfg_pulse(q, cfg.device.qubit.f_ge, "pi_ge")
        # pi_ef is needed for prep into |f>, and by reset_f, which walks
        # |f> -> |e> -> |g> ahead of each pi_ge.
        if cfg.expt.prep == "f" or cfg.expt.reset_f:
            super().make_cfg_pulse(q, cfg.device.qubit.f_ef, "pi_ef")

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)

        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        if cfg.expt.prep in ("e", "f"):
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait ge")
        if cfg.expt.prep == "f":
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
            self.delay_auto(t=0.01, tag="wait ef")

        self.pulse(ch=self.f0g1_ch, name="f0g1_pulse", t=0)

        if cfg.expt.ef_probe_on:
            if cfg.expt.ef_align == "end":
                # Step 3: a short e-f probe nested inside a longer f0-g1 tone
                # so both *end* together. Total lengths include any flat_top
                # ramps, which play either side of the flat segment.
                f0g1_total = cfg.expt.length_f0g1 + ramp_length(
                    cfg.expt.pulse_type_f0g1,
                    cfg.expt.ramp_sigma_f0g1, cfg.expt.ramp_sigma_inc_f0g1,
                )
                ef_total = cfg.expt.length_ef + ramp_length(
                    cfg.expt.pulse_type_ef,
                    cfg.expt.ramp_sigma_ef, cfg.expt.ramp_sigma_inc_ef,
                )
                # self.delay increments the reference time; delay_auto would
                # set it and discard the f0-g1 pulse's own start.
                self.delay(t=f0g1_total - ef_total)
            # Step 5 ("start"): both tones begin together and run for the same
            # reset time, which is the reset protocol itself rather than a
            # probe -- so no delay at all.
            self.pulse(ch=self.qubit_ch, name="ef_pulse", t=0)

        self.delay_auto(t=cfg.expt.readout_gap, tag="wait f0g1")

        if cfg.expt.shelve_ge:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait shelve")

        super().measure(cfg)


class F0G1Spec(QickExperiment):
    """
    Sweep the f0-g1 drive frequency with the qubit prepared in |f>.

    Step 1 of the paper's calibration. At the f0-g1 resonance the population
    is transferred to |g,1> and the photon leaves, so the readout sees |g>;
    off resonance it stays in |f>. The resulting peak in P(g) locates the
    transition, ac-Stark-shifted by whatever drive power is applied.

    Parameters:
    - 'span' / 'expts' / 'start': frequency window (MHz) and point count
    - 'gain_f0g1': drive amplitude (DAC level). The main unknown on a first
      run -- see the note in 27_f0g1_spectroscopy.py.
    - 'length_f0g1': flat duration of the drive (us). Long enough to saturate
      makes the line easier to *find*; keep it short for a clean centre.
    - 'ramp_sigma_f0g1' / 'ramp_sigma_inc_f0g1': gaussian edges of the flat-top
    - 'prep': 'g' | 'e' | 'f' -- state prep before the drive ('f' normally;
      'g' is the background check that the feature needs the f state)
    - 'readout_gap': wait between drive and readout, in us

    The style parameter sets 'span' and 'expts':
    - 'coarse': 200 MHz / 400 pts -- the design frequency is only good to a
      few MHz, so start here if nothing has been found yet
    - 'medium': 50 MHz / 200 pts
    - 'fine':   10 MHz / 100 pts
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        display=True,
        style="medium",
        min_r2=None,
        max_err=None,
        check_params=True,
    ):
        if prefix == "":
            prefix = f"f0g1_spectroscopy_{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress,
                         qi=qi, check_params=check_params)

        # Raise here rather than let the Program fail on a missing key: a
        # device with no reset drive line cannot run any of this, and the
        # useful message is which config section is absent.
        if "f0g1" not in self.cfg.hw.soc.dacs:
            raise ValueError(
                "hw.soc.dacs.f0g1 is missing from this config. The f0-g1 reset "
                "drive needs its own DAC section (ch, type, nyquist, "
                "mixer_freq, filter_*, attn*), mirroring hw.soc.dacs.pump."
            )

        if style == "coarse":
            params_def = {"span": 200, "expts": 400, "reps": self.reps}
        elif style == "fine":
            params_def = {"span": 10, "expts": 100, "reps": 2 * self.reps}
        else:
            params_def = {"span": 50, "expts": 200, "reps": self.reps}

        params_def2 = {
            "rounds": self.rounds,
            "final_delay": self.cfg.device.readout.final_delay[qi],
            "gain_f0g1": 0.5,
            "length_f0g1": 2.0,
            "ramp_sigma_f0g1": 0.02,
            "ramp_sigma_inc_f0g1": 3,
            "pulse_type_f0g1": pulse_type_from_config(self.cfg.device.get("f0g1"), "pulse_type"),
            "prep": "f",
            # Step 3's switches; off here. The observable is g-vs-not-g, and
            # |e>/|f> both read as not-g, so no shelving pulse is needed.
            "ef_probe_on": False,
            "shelve_ge": False,
            # The drive deliberately puts a photon in the readout resonator;
            # wait a few 1/kappa for it to leave before measuring.
            "readout_gap": 3 / self.cfg.device.readout.kappa[qi],
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            # Which reset runs between shots: "passive" (the full T1
            # wait), "active" (measurement-based conditional pi pulses)
            # or "f0g1" (the unconditional all-microwave reset, which
            # also clears |f> -- see QickProgram.reset_f0g1).
            "reset_mode": "passive",
            # "f0g1" only, and all optional -- everything else is
            # derived from device.f0g1. residual sets the drive length
            # as ln(1/residual)/(2*pi*Gamma), so it rescales on its own
            # as Gamma improves; f0g1_reset_length pins the length
            # instead. None means device.f0g1.reset_*, else derived.
            "f0g1_reset_residual": None,
            "f0g1_reset_gain": None,
            "f0g1_reset_gain_ef": None,
            "f0g1_reset_length": None,
            "f0g1_reset_gap": None,
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            "reset_verify": False,
            # Latched "while excited" reset; see CLAUDE.md "Reset modes".
            "reset_latch": False,
            # Mean rounds the while loop takes, for the cycle-time estimate
            # only; `reset` is its hard cap. See CLAUDE.md.
            "reset_while_typical": 1,
            # The |f> prep leaves population in |f> whenever the drive is off
            # resonance, which a pi_ge-only reset cannot clear.
            "reset_f": True,
        }
        params_def = {**params_def2, **params_def}
        params = {**params_def, **params}

        # Scan centre: the measured frequency if we have one, else the design
        # value from the bare transition frequencies.
        center = self.cfg.device.get("f0g1", {}).get("frequency", None)
        if center is None or not np.isfinite(center):
            center = f0g1_design_frequency(self.cfg, qi)
            center_source = f"design f_ge + f_ef - f_readout = {center:.4f} MHz"
        else:
            center_source = f"device.f0g1.frequency = {center:.4f} MHz"
        params_def["start"] = center - params["span"] / 2
        params = {**params_def, **params}

        # The program treats any unrecognised prep as "g", which would look
        # like a real measurement with no |f> population in it.
        if params["prep"] not in ("g", "e", "f"):
            raise ValueError(
                f"prep must be 'g', 'e' or 'f', got {params['prep']!r}"
            )

        self.cfg.expt = params
        super().check_params(params_def)

        if reset_mode_of(self.cfg.expt) != "passive":
            super().configure_reset()

        print(
            f"f0-g1 spectroscopy: DAC ch {self.cfg.hw.soc.dacs.f0g1.ch[qi]}, "
            f"scan centre from {center_source}, span {params['span']} MHz, "
            f"gain {params['gain_f0g1']}, length {params['length_f0g1']} us, "
            f"prep |{params['prep']}>, readout gap {params['readout_gap']:.2f} us"
        )

        if go:
            super().run(min_r2=min_r2, max_err=max_err, display=display,
                        progress=progress)

    def acquire(self, progress=False):
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay

        self.param = {"label": "f0g1_pulse", "param": "freq", "param_type": "pulse"}
        self.cfg.expt.frequency_f0g1 = QickSweep1D(
            "sweep_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        super().acquire(F0G1Program, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            self.fitterfunc = FITTER_FUNC
            self.fitfunc = FIT_FUNC
            super().analyze(use_i=False, **kwargs)
            # fitlor parameters are [y0, amp, center, gamma]; amp is signed, so
            # the same fit handles a peak (population returned to |g>) and a dip.
            if "best_fit" in data:
                data["f0g1_freq"] = data["best_fit"][2]
                data["f0g1_kappa"] = data["best_fit"][3]

        return data

    def display(self, data=None, fit=True, ax=None, plot_all=False, **kwargs):
        title = (
            f"f0-g1 spectroscopy Q{self.cfg.expt.qubit[0]}, prep |{self.cfg.expt.prep}>, "
            f"gain {self.cfg.expt.gain_f0g1}, length {self.cfg.expt.length_f0g1} us"
        )
        caption_params = [
            {"index": 2, "format": "$f_{{f0g1}}$: {val:.6} MHz"},
            {"index": 3, "format": "$\\kappa$: {val:.3} MHz"},
        ]
        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=XLABEL,
            fit=fit,
            show_hist=False,
            caption_params=caption_params,
            **kwargs,
        )

    def update(self, verbose=True):
        """Write the measured zero-power f0-g1 frequency to device.f0g1."""
        if "f0g1_freq" not in self.data:
            print("No f0g1_freq in data; run analyze() first")
            return
        config.update_f0g1(
            self.config_file, "frequency", self.data["f0g1_freq"], verbose=verbose
        )


class F0G1SpecPower(QickExperiment2DSimple):
    """
    f0-g1 spectroscopy versus drive power -- the ac Stark shift of the
    transition.

    Step 1b of the paper's calibration (their Fig. 2a/2b, light blue). The
    f0-g1 resonance pulls quadratically with drive amplitude, so the fitted
    centres are fit to

        freq(gain) = a * gain**2 + c

    a forced-vertex parabola (there is no shift at zero drive), which is the
    same two-parameter form ``_fit_band_model`` uses for the FWM band. The
    coefficients go to ``device.f0g1.stark_fit`` and are what step 4 reads back
    to keep the f0-g1 Rabi on resonance at every power.

    Parameters:
    - 'start_gain' / 'max_gain' / 'expts_gain': the power axis (linear spacing;
      a quadratic shift is best sampled evenly in gain, not logarithmically)
    - 'fixed_area': hold gain * length constant so every row gets a comparable
      Rabi angle, as the paper does. Off means a fixed pulse length, which
      saturates the line at high power and broadens it.
    - 'area_gain_length': the product to hold, in (DAC level) * us. Defaults to
      the inner experiment's gain_f0g1 * length_f0g1.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        style="medium",
        display=True,
        min_r2=None,
        max_err=None,
        live_plot=False,
    ):
        if prefix == "":
            prefix = f"f0g1_stark_shift_{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress,
                         qi=qi, live_plot=live_plot)
        self.nested_progress = False
        self.progress_desc = "f0-g1 gain sweep"

        params_def = {
            "start_gain": 0.05,
            "max_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 12,
            "fixed_area": True,
            "area_gain_length": None,
        }
        params = {**params_def, **params}

        self.expt = F0G1Spec(
            cfg_dict, qi=qi, go=False, params=params, style=style,
            check_params=False,
        )
        params = {**self.expt.cfg.expt, **params}

        if params["area_gain_length"] is None:
            params["area_gain_length"] = (
                params["gain_f0g1"] * params["length_f0g1"]
            )

        self.cfg.expt = params

        if go:
            self.run(progress=progress, display=display)

    def acquire(self, progress=False):
        gainpts = np.linspace(
            self.cfg.expt.start_gain,
            self.cfg.expt.max_gain,
            self.cfg.expt.expts_gain,
        )
        ysweep = [{"pts": gainpts, "var": "gain_f0g1"}]

        if self.cfg.expt.fixed_area:
            # Hold gain * length fixed so the Rabi angle is comparable at every
            # power. QickExperiment2DSimple applies every dict in the list at
            # the same index, so this rides along on the gain axis.
            lengths = self.cfg.expt.area_gain_length / gainpts
            max_len = self.cfg.expt.length_f0g1 * 5
            clipped = int(np.sum(lengths > max_len))
            if clipped:
                print(
                    f"fixed_area: {clipped} of {len(lengths)} lengths clipped to "
                    f"{max_len} us (the lowest gains cannot reach the requested "
                    "Rabi angle)"
                )
            lengths = np.minimum(lengths, max_len)
            ysweep.append({"pts": lengths, "var": "length_f0g1"})

        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {"label": "f0g1_pulse", "param": "freq", "param_type": "pulse"}
        self.xlabel = XLABEL
        self.ylabel = YLABEL

        super().acquire(ysweep, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, min_r2=0.1, **kwargs):
        """Lorentzian per drive power, then the centres to a*gain**2 + c."""
        if data is None:
            data = self.data

        if not fit:
            return data

        # Per-row Lorentzian on avgi, giving one centre per drive power.
        super().analyze(fitterfunc=FITTER_FUNC, fitfunc=FIT_FUNC)
        return _analyze_quadratic_stark(data, "stark", "f0-g1", min_r2=min_r2)

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):
        if data is None:
            data = self.data

        title = (
            f"f0-g1 Stark shift Q{self.cfg.expt.qubit[0]}, prep "
            f"|{self.cfg.expt.prep}>, "
            + (
                f"gain*length = {self.cfg.expt.area_gain_length:.3g}"
                if self.cfg.expt.fixed_area
                else f"length {self.cfg.expt.length_f0g1} us"
            )
        )
        super().display(
            data=data, ax=ax, plot_amps=plot_amps, title=title, xlabel=XLABEL,
            ylabel=YLABEL, fit=fit, **kwargs
        )
        if "stark_center" in data:
            self.display_stark_shift(data=data)

    def display_stark_shift(self, data=None):
        """Fitted centre and linewidth vs drive gain, with the quadratic fit."""
        if data is None:
            data = self.data
        _display_quadratic_stark(
            self, data, "stark", "f0-g1 frequency (MHz)", "_stark_shift"
        )

    def update(self, verbose=True):
        """
        Write the quadratic ac Stark model to device.f0g1.

        ``stark_fit`` = [a, c] for ``freq = a * gain**2 + c``, and ``frequency``
        = c, the zero-drive intercept. Step 4 inverts this to keep the f0-g1
        Rabi on resonance at each power.
        """
        if "stark_fit_a" not in self.data:
            print("No f0-g1 Stark fit available; run analyze() first")
            return
        config.update_f0g1(
            self.config_file,
            "stark_fit",
            [self.data["stark_fit_a"], self.data["stark_fit_c"]],
            verbose=verbose,
        )
        config.update_f0g1(
            self.config_file, "frequency", self.data["stark_fit_c"], verbose=verbose
        )


class EFSpecF0G1(QickExperiment):
    """
    e-f spectroscopy with the f0-g1 drive on -- the e-f transition's ac Stark
    shift.

    Step 3 of the paper's calibration (their Fig. 2e, purple). The f0-g1 drive
    pulls the e-f transition as well as its own, and the reset needs both
    drives on their shifted resonances, so this measures where e-f has moved
    to at a given f0-g1 power.

    Sequence: prep |e>, then a long f0-g1 tone with a short flat-top e-f pi
    pulse nested inside so both end together, then a shelving pi_ge, then
    readout (see ``F0G1Program``). The swept quantity is the e-f probe
    frequency.

    The f0-g1 tone is held on *its own* Stark-shifted resonance, read back from
    ``device.f0g1.stark_fit`` at whatever ``gain_f0g1`` is in use, unless
    ``frequency_f0g1`` is given explicitly. Driving it off resonance would
    under-report the shift.

    Parameters:
    - 'gain_f0g1' / 'length_f0g1': the drive whose shift is being measured
    - 'frequency_f0g1': override; defaults to the step-1 fit at that gain
    - 'gain_ef': e-f probe amplitude. Its only job is to put a dip in the
      spectrum -- the measurement reads the dip's *position*, not its depth,
      so an imperfect pi costs contrast and nothing else. A weaker probe gives
      a narrower line (the Fourier limit is ~2*Omega_ef) and a better centre.
    - 'length_ef': flat duration of the probe; defaults to the pi duration
      1/(2*Omega_ef) from ``device.f0g1.omega_ef_fit`` at ``gain_ef``
    - 'span' / 'expts': the probe frequency window

    The style parameter sets 'span' and 'expts':
    - 'coarse': 60 MHz / 300 pts
    - 'medium': 30 MHz / 150 pts
    - 'fine':   8 MHz / 100 pts
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        display=True,
        style="medium",
        min_r2=None,
        max_err=None,
        check_params=True,
    ):
        if prefix == "":
            prefix = f"ef_spec_under_f0g1_{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress,
                         qi=qi, check_params=check_params)

        if "f0g1" not in self.cfg.hw.soc.dacs:
            raise ValueError(
                "hw.soc.dacs.f0g1 is missing from this config. The f0-g1 reset "
                "drive needs its own DAC section (ch, type, nyquist, "
                "mixer_freq, filter_*, attn*), mirroring hw.soc.dacs.pump."
            )

        if style == "coarse":
            params_def = {"span": 60, "expts": 300, "reps": self.reps}
        elif style == "fine":
            params_def = {"span": 8, "expts": 100, "reps": 2 * self.reps}
        else:
            params_def = {"span": 30, "expts": 150, "reps": self.reps}

        params_def2 = {
            "rounds": self.rounds,
            "final_delay": self.cfg.device.readout.final_delay[qi],
            # The f0-g1 drive being characterised.
            "gain_f0g1": 0.5,
            "frequency_f0g1": None,
            "ramp_sigma_f0g1": 0.02,
            "ramp_sigma_inc_f0g1": 3,
            "pulse_type_f0g1": pulse_type_from_config(self.cfg.device.get("f0g1"), "pulse_type"),
            # The e-f probe.
            "gain_ef": 0.05,
            "length_ef": None,
            "ramp_sigma_ef": 0.02,
            "ramp_sigma_inc_ef": 3,
            "pulse_type_ef": pulse_type_from_config(self.cfg.device.get("f0g1"), "pulse_type_ef"),
            # Paper ratio: a 420 ns f0-g1 tone around a 140 ns e-f pi.
            "f0g1_length_ratio": 3.0,
            "length_f0g1": None,
            "prep": "e",
            "ef_probe_on": True,
            # Nest the probe inside the drive so both end together; step 5
            # uses "start" instead, running both for the same reset time.
            "ef_align": "end",
            "shelve_ge": True,
            "readout_gap": 3 / self.cfg.device.readout.kappa[qi],
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            # Which reset runs between shots: "passive" (the full T1
            # wait), "active" (measurement-based conditional pi pulses)
            # or "f0g1" (the unconditional all-microwave reset, which
            # also clears |f> -- see QickProgram.reset_f0g1).
            "reset_mode": "passive",
            # "f0g1" only, and all optional -- everything else is
            # derived from device.f0g1. residual sets the drive length
            # as ln(1/residual)/(2*pi*Gamma), so it rescales on its own
            # as Gamma improves; f0g1_reset_length pins the length
            # instead. None means device.f0g1.reset_*, else derived.
            "f0g1_reset_residual": None,
            "f0g1_reset_gain": None,
            "f0g1_reset_gain_ef": None,
            "f0g1_reset_length": None,
            "f0g1_reset_gap": None,
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            "reset_verify": False,
            # Latched "while excited" reset; see CLAUDE.md "Reset modes".
            "reset_latch": False,
            # Mean rounds the while loop takes, for the cycle-time estimate
            # only; `reset` is its hard cap. See CLAUDE.md.
            "reset_while_typical": 1,
            # The probe drives population into |f> off the f0-g1 resonance,
            # which a pi_ge-only reset cannot clear.
            "reset_f": True,
        }
        params_def = {**params_def2, **params_def}
        params = {**params_def, **params}

        # The e-f pi duration from step 2's rate calibration. Derived rather
        # than stored, so there is no second copy of the e-f pulse to keep in
        # sync with device.qubit.pulses.pi_ef (which stays the gaussian prep
        # pulse). t_pi is on the same total-length axis the rate was fit
        # against, so the flat part is t_pi minus the ramps.
        omega_ef = omega_ef_for_gain(self.cfg, params["gain_ef"])
        t_pi = 1 / (2 * omega_ef)
        ramp_ef = ramp_length(params["pulse_type_ef"], params["ramp_sigma_ef"],
                              params["ramp_sigma_inc_ef"])
        ramp_f0g1 = ramp_length(params["pulse_type_f0g1"], params["ramp_sigma_f0g1"],
                                params["ramp_sigma_inc_f0g1"])
        if params["length_ef"] is None:
            if t_pi <= ramp_ef:
                raise ValueError(
                    f"gain_ef={params['gain_ef']} gives Omega_ef/2pi="
                    f"{omega_ef:.4f} MHz, i.e. a pi in {t_pi * 1e3:.1f} ns, "
                    f"shorter than the {ramp_ef * 1e3:.1f} ns of flat-top ramp. "
                    "Lower gain_ef or shorten ramp_sigma_ef."
                )
            params["length_ef"] = t_pi - ramp_ef
        params_def["length_ef"] = params["length_ef"]

        if params["length_f0g1"] is None:
            params["length_f0g1"] = (
                params["f0g1_length_ratio"]
                * (params["length_ef"] + ramp_ef)
                - ramp_f0g1
            )
        params_def["length_f0g1"] = params["length_f0g1"]

        # The program nests the probe by delaying it by the difference of the
        # two total lengths, so a probe longer than the drive would emit a
        # negative delay and silently reorder the sequence.
        f0g1_total = params["length_f0g1"] + ramp_f0g1
        ef_total = params["length_ef"] + ramp_ef
        if f0g1_total < ef_total:
            raise ValueError(
                f"the f0-g1 tone ({f0g1_total:.4f} us total) must be at least as "
                f"long as the e-f probe ({ef_total:.4f} us total); raise "
                "length_f0g1 or f0g1_length_ratio."
            )

        # Keep the f0-g1 drive on its own Stark-shifted resonance.
        if params["frequency_f0g1"] is None:
            params["frequency_f0g1"] = f0g1_frequency_for_gain(
                self.cfg, params["gain_f0g1"]
            )
        params_def["frequency_f0g1"] = params["frequency_f0g1"]

        # Probe window centre: the measured e-f shift at this gain if we have
        # one, else the bare f_ef.
        ef_fit = self.cfg.device.get("f0g1", {}).get("ef_stark_fit", None)
        if ef_fit is not None and np.all(np.isfinite(np.array(ef_fit, dtype=float))):
            a, c = (float(v) for v in ef_fit)
            center = a * params["gain_f0g1"] ** 2 + c
            center_source = f"ef_stark_fit at gain {params['gain_f0g1']:.3f}"
        else:
            center = self.cfg.device.qubit.f_ef[qi]
            center_source = "device.qubit.f_ef (unshifted)"
        params_def["start"] = center - params["span"] / 2
        params = {**params_def, **params}

        if params["prep"] not in ("g", "e", "f"):
            raise ValueError(
                f"prep must be 'g', 'e' or 'f', got {params['prep']!r}"
            )

        self.cfg.expt = params
        super().check_params(params_def)

        if reset_mode_of(self.cfg.expt) != "passive":
            super().configure_reset()

        print(
            f"e-f spectroscopy under the f0-g1 drive: f0g1 gain "
            f"{params['gain_f0g1']} at {params['frequency_f0g1']:.4f} MHz for "
            f"{f0g1_total:.4f} us; e-f probe gain {params['gain_ef']} for "
            f"{ef_total:.4f} us (pi at Omega_ef/2pi = {omega_ef:.4f} MHz); "
            f"window centred on {center:.4f} MHz from {center_source}"
        )

        # The f0-g1 drive hybridizes |f,0> with |g,1>, so this line is a
        # doublet split by 2*g~ and the window has to contain both peaks --
        # the analysis reads the *centre*, which a truncated pair biases by
        # up to g~. At gain 0.1 the splitting is a fraction of a linewidth
        # and this never fires; it is the check you want when raising
        # gain_f0g1 to look at the pair directly.
        g_fit = self.cfg.device.get("f0g1", {}).get("g_tilde_fit", None)
        if g_fit is not None and np.all(np.isfinite(np.array(g_fit, dtype=float))):
            gm, gb = (float(v) for v in g_fit)
            split = 2 * (gm * params["gain_f0g1"] + gb)
            # The feature is not just the splitting: each dressed state is
            # half |g,1> and so carries ~kappa/2 of width, and the pair spans
            # roughly delta + 2*gamma. Needs ~2x that in span to leave any
            # baseline. On #366 at gain 1.0 the feature was 2.97 MHz wide in a
            # 4 MHz window and the fit ran its separation into the bound.
            kap = float(self.cfg.device.f0g1.get("kappa", np.nan))
            if not np.isfinite(kap):
                kap = float(self.cfg.device.readout.kappa[qi])
            # Each dressed state is half |g,1>, so carries ~kappa/2 of
            # width, and the probe adds its own ~2*Omega_ef of power
            # broadening. With a strong probe that second term dominates --
            # at the operating gain_ef it is 1.08 MHz against kappa/2 = 0.66
            # -- so the window has to allow for both, not just kappa.
            line = np.hypot(2 * omega_ef, kap / 2)
            width = split + 2 * line
            if width > 0.6 * params["span"]:
                print(
                    f"  NOTE: the feature should span ~{width:.3f} MHz "
                    f"(2*g~ = {split:.3f} plus 2 x {line:.3f} MHz of linewidth) "
                    f"in a {params['span']:.3f} MHz window. Widen span to >~ "
                    f"{2 * width:.0f} MHz, or it fills the scan and the fit has "
                    "no baseline to work against."
                )
        # Resolving the pair is limited by the *linewidth*, not the
        # splitting: two peaks 1.6 MHz apart are still one broad line if
        # neither is sampled. The probe's Fourier limit ~2*Omega_ef is the
        # narrowest the line can be, so it is the right yardstick and it is
        # known before the run. Measured against #364 (65 kHz probe linewidth,
        # 101 kHz steps): 1.3 points per linewidth, and 7 of its 10 rows
        # fitted as single lines.
        linewidth = 2 * omega_ef
        step = params["span"] / params["expts"]
        if linewidth > 0 and step > linewidth / 3:
            print(
                f"  NOTE: {step * 1e3:.0f} kHz steps against a probe linewidth "
                f"of ~{linewidth * 1e3:.0f} kHz (2*Omega_ef at gain_ef="
                f"{params['gain_ef']}), i.e. {linewidth / step:.1f} points per "
                f"linewidth. Raise expts to >~ "
                f"{int(np.ceil(3 * params['span'] / linewidth))} -- an "
                "under-sampled doublet fits as one broad line centred between "
                "the peaks."
            )

        if go:
            super().run(min_r2=min_r2, max_err=max_err, display=display,
                        progress=progress)

    def acquire(self, progress=False):
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay

        self.param = {"label": "ef_pulse", "param": "freq", "param_type": "pulse"}
        self.cfg.expt.frequency_ef = QickSweep1D(
            "sweep_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        super().acquire(F0G1Program, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            self.fitterfunc = FITTER_FUNC
            self.fitfunc = FIT_FUNC
            super().analyze(use_i=False, **kwargs)

            # The f0-g1 drive dresses |f,0> with |g,1>, so the line is a
            # doublet split by 2*g~ whenever the pair is resolved. The reset
            # needs the *centre*; a single Lorentzian lands on one peak or
            # somewhere between them.
            qi = self.cfg.expt.qubit[0]
            g_fit = self.cfg.device.get("f0g1", {}).get("g_tilde_fit", None)
            have_g = g_fit is not None and np.all(
                np.isfinite(np.array(g_fit, dtype=float))
            )
            if have_g:
                gm, gb = (float(v) for v in g_fit)
                seed = 2 * (gm * self.cfg.expt.gain_f0g1 + gb)
            else:
                seed = self.cfg.expt.span / 10
            res = fit_ef_doublet(data["xpts"], data["avgi"], seed)
            data["ef_doublet"] = res["pars"]
            data["ef_npeaks"] = res["n_peaks"]
            data["ef_splitting"] = res["splitting"]
            data["f_ef_shifted"] = res["center"]
            data["f_ef_kappa"] = res["gamma"]
            data["f_ef_shift"] = res["center"] - self.cfg.device.qubit.f_ef[qi]
            print(
                f"e-f at f0g1 gain {self.cfg.expt.gain_f0g1}: fit as "
                f"{res['n_peaks']} peak(s), R2 {res['r2']:.3f}"
            )
            print(
                f"  centre {res['center']:.4f} MHz "
                f"({data['f_ef_shift']:+.4f} from the bare f_ef), "
                f"linewidth {res['gamma']:.4f} MHz"
            )
            if res["n_peaks"] == 2:
                g_split = res["splitting"] / 2
                print(
                    f"  resolved doublet: splitting {res['splitting']:.4f} MHz "
                    f"-> g~/2pi = {g_split:.4f} MHz at this gain"
                )
                if have_g:
                    expect = gm * self.cfg.expt.gain_f0g1 + gb
                    print(f"  (the step-4 Rabi predicts {expect:.4f} MHz here, "
                          f"{g_split / expect:.2f}x)")
                print("  the reset wants the CENTRE -- driving a peak is "
                      "delta_ef = +/- g~ and costs most of Gamma")

        return data

    def display(self, data=None, fit=True, ax=None, plot_all=False, **kwargs):
        title = (
            f"e-f spectroscopy under f0-g1 Q{self.cfg.expt.qubit[0]}, f0g1 gain "
            f"{self.cfg.expt.gain_f0g1} at "
            f"{self.cfg.expt.frequency_f0g1:.3f} MHz"
        )
        caption_params = [
            {"index": 2, "format": "$f_{{ef}}$: {val:.6} MHz"},
            {"index": 3, "format": "$\gamma$: {val:.3} MHz"},
        ]
        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel="e-f Probe Frequency (MHz)",
            fit=fit,
            show_hist=False,
            caption_params=caption_params,
            **kwargs,
        )
        # A second figure only when the dressed pair actually resolved. At the
        # operating gain_ef the line is merged and the single-Lorentzian figure
        # above is the whole story, so this must not be the only plot drawn --
        # it was, briefly, which left a merged run with no figure at all.
        if data is None:
            data = self.data
        if data is not None and data.get("ef_npeaks", 0) == 2:
            self.display_ef_doublet(data=data)

    def display_ef_doublet(self, data=None):
        """
        The resolved doublet, its two-Lorentzian fit, and the centre.

        Drawn in addition to the standard figure, not instead of it. The
        vertical lines are what the reset cares about: the solid one is the
        centre to drive, the dotted ones the two dressed peaks, each of which
        costs most of Gamma (delta_ef = +/- g~).
        """
        if data is None:
            data = self.data
        pars = data.get("ef_doublet", None)
        if pars is None or data.get("ef_npeaks", 0) != 2:
            return
        x = np.asarray(data["xpts"], dtype=float)
        fine = np.linspace(x.min(), x.max(), 600)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(x, data["avgi"], "o", ms=3, color="C7", label="data")
        ax.plot(fine, two_lorentzians(fine, *pars), "-", color="C3", lw=2,
                label="two Lorentzians")
        ctr = data["f_ef_shifted"]
        ax.axvline(ctr, color="C0", lw=2,
                   label=f"centre {ctr:.4f} MHz (drive here)")
        split = data.get("ef_splitting", np.nan)
        if np.isfinite(split):
            for sign in (-1, 1):
                ax.axvline(ctr + sign * split / 2, ls=":", color="C3", lw=1.2)
        ax.set_xlabel("e-f Probe Frequency (MHz)")
        ax.set_ylabel("Signal (a.u.)")
        ax.legend(fontsize=8)
        self.save_fig(fig, suffix="_ef_doublet")


class EFSpecF0G1Power(QickExperiment2DSimple):
    """
    e-f spectroscopy versus f0-g1 drive power -- the e-f ac Stark shift.

    Step 3b of the paper's calibration (their Fig. 2b, purple). Repeats
    ``EFSpecF0G1`` over f0-g1 amplitude and fits the pulled e-f centres to

        freq(gain) = a * gain**2 + c

    written to ``device.f0g1.ef_stark_fit``. Step 5 needs both this and the
    step-1 ``stark_fit`` to put the two reset drives on their shifted
    resonances simultaneously.

    **Expect `a` to have the same sign as step 1's** -- both transitions are
    pushed the same way by the same drive -- which is the cross-check that the
    two runs saw the same drive amplitude. The magnitudes need not match.

    The f0-g1 frequency is co-varied with the gain, from
    ``device.f0g1.stark_fit``, so every row has its drive on resonance. That is
    the same trick scripts 23/24 use to walk the FWM band.

    Parameters:
    - 'start_gain' / 'max_gain' / 'expts_gain': the f0-g1 power axis
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        style="medium",
        display=True,
        min_r2=None,
        max_err=None,
        live_plot=False,
    ):
        if prefix == "":
            prefix = f"ef_stark_shift_vs_f0g1_{style}_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress,
                         qi=qi, live_plot=live_plot)
        self.nested_progress = False
        self.progress_desc = "f0-g1 gain sweep"

        params_def = {
            "start_gain": 0.05,
            "max_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 12,
        }
        params = {**params_def, **params}

        self.expt = EFSpecF0G1(
            cfg_dict, qi=qi, go=False, params=params, style=style,
            check_params=False,
        )
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        if go:
            self.run(progress=progress, display=display)

    def acquire(self, progress=False):
        gainpts = np.linspace(
            self.cfg.expt.start_gain,
            self.cfg.expt.max_gain,
            self.cfg.expt.expts_gain,
        )
        # Co-vary the f0-g1 frequency with its gain so every row drives the
        # Stark-shifted resonance rather than a fixed frequency that walks off
        # it. QickExperiment2DSimple applies every dict at the same index.
        freqpts = np.array(
            [f0g1_frequency_for_gain(self.cfg, g) for g in gainpts]
        )
        print(
            f"f0-g1 drive tracks its Stark shift: {freqpts[0]:.4f} MHz at gain "
            f"{gainpts[0]:.3f} -> {freqpts[-1]:.4f} MHz at gain {gainpts[-1]:.3f}"
        )
        ysweep = [
            {"pts": gainpts, "var": "gain_f0g1"},
            {"pts": freqpts, "var": "frequency_f0g1"},
        ]

        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {"label": "ef_pulse", "param": "freq", "param_type": "pulse"}
        self.xlabel = "e-f Probe Frequency (MHz)"
        self.ylabel = YLABEL

        super().acquire(ysweep, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, min_r2=0.1, **kwargs):
        """Lorentzian per f0-g1 power, then the e-f centres to a*gain**2 + c."""
        if data is None:
            data = self.data

        if not fit:
            return data

        # Per-row doublet fit. The f0-g1 drive dresses |f,0> with |g,1>, so
        # each row's e-f line is a pair split by 2*g~ and a single Lorentzian
        # lands between or on one of them -- at this device's g~ that is an
        # error of hundreds of kHz, comparable to Omega_ef itself.
        data = _analyze_ef_doublet(data, self.cfg, min_r2=min_r2)

        # Same-sign cross-check against step 1: one drive, both transitions.
        f0g1_fit = self.cfg.device.get("f0g1", {}).get("stark_fit", None)
        if "ef_stark_fit_a" in data and f0g1_fit is not None:
            a_f0g1 = float(f0g1_fit[0])
            a_ef = data["ef_stark_fit_a"]
            if np.isfinite(a_f0g1) and a_f0g1 * a_ef < 0:
                print(
                    f"WARNING: the e-f shift ({a_ef:+.4f} MHz/gain^2) and the "
                    f"f0-g1 shift ({a_f0g1:+.4f} MHz/gain^2) have opposite "
                    "signs. One drive pushes both transitions the same way, so "
                    "one of the two runs probably saw a different drive "
                    "amplitude -- check the attenuation was the same."
                )
        return data

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):
        if data is None:
            data = self.data

        title = (
            f"e-f Stark shift from the f0-g1 drive, Q{self.cfg.expt.qubit[0]}, "
            f"e-f probe gain {self.cfg.expt.gain_ef} for "
            f"{self.cfg.expt.length_ef:.3f} us"
        )
        super().display(
            data=data, ax=ax, plot_amps=plot_amps, title=title,
            xlabel=self.xlabel if hasattr(self, "xlabel") else "",
            ylabel=YLABEL, fit=fit, **kwargs
        )
        if "ef_stark_center" in data:
            self.display_ef_stark_shift(data=data)

    def display_ef_stark_shift(self, data=None):
        """
        Doublet centre and splitting vs f0-g1 gain.

        Top: the centre with the quadratic Stark fit, and the two dressed-state
        peaks either side of it where they resolved. Bottom: the splitting,
        which is 2*g~ -- so its slope is an independent measurement of the
        f0-g1 coupling, plotted against the step-4 Rabi fit for comparison.
        """
        if data is None:
            data = self.data
        if "ef_stark_center" not in data:
            return

        gains = np.asarray(data["ypts"], dtype=float)
        ctr = np.asarray(data["ef_stark_center"], dtype=float)
        spl = np.asarray(data["ef_stark_splitting"], dtype=float)
        keep = data.get("ef_stark_keep", np.isfinite(ctr))

        fig, axes = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)

        two = np.isfinite(spl)
        axes[0].plot(gains[keep], ctr[keep], "o", color="C0", label="centre")
        if np.any(~keep):
            axes[0].plot(gains[~keep], ctr[~keep], "x", color="grey",
                         label="rejected")
        if np.any(two):
            axes[0].plot(gains[two], ctr[two] - spl[two] / 2, ".", ms=5,
                         color="C3", label="dressed peaks")
            axes[0].plot(gains[two], ctr[two] + spl[two] / 2, ".", ms=5,
                         color="C3")
        if "ef_stark_fit_a" in data:
            gf = np.linspace(0, gains.max(), 200)
            axes[0].plot(
                gf, data["ef_stark_fit_a"] * gf**2 + data["ef_stark_fit_c"], "-",
                color="C0",
                label=(f"{data['ef_stark_fit_a']:.3f}$g^2$ + "
                       f"{data['ef_stark_fit_c']:.3f}"),
            )
        axes[0].set_ylabel("e-f frequency (MHz)")
        axes[0].legend(fontsize=8)

        axes[1].plot(gains[two], spl[two], "o", color="C3", label="measured")
        if "g_tilde_split_m" in data:
            gf = np.linspace(0, gains.max(), 200)
            m, b = data["g_tilde_split_m"], data["g_tilde_split_b"]
            axes[1].plot(gf, 2 * (m * gf + b), "-", color="C3",
                         label=f"$2\tilde{{g}}$ = 2({m:.3f}$g$ {b:+.3f})")
        rb = self.cfg.device.get("f0g1", {}).get("g_tilde_fit", None)
        if rb is not None and np.all(np.isfinite(np.array(rb, dtype=float))):
            gm, gb = (float(v) for v in rb)
            gf = np.linspace(0, gains.max(), 200)
            axes[1].plot(gf, 2 * (gm * gf + gb), "--", color="C7",
                         label=f"$2\tilde{{g}}$ from the step-4 Rabi")
        axes[1].set_ylabel("doublet splitting, $2\tilde{g}/2\pi$ (MHz)")
        axes[1].set_xlabel(YLABEL)
        axes[1].legend(fontsize=8)

        self.save_fig(fig, suffix="_ef_stark_shift")

    def update(self, verbose=True):
        """Write the quadratic e-f ac Stark model to device.f0g1.ef_stark_fit."""
        if "ef_stark_fit_a" not in self.data:
            print("No e-f Stark fit available; run analyze() first")
            return
        config.update_f0g1(
            self.config_file,
            "ef_stark_fit",
            [self.data["ef_stark_fit_a"], self.data["ef_stark_fit_c"]],
            verbose=verbose,
        )


class EFRabiPower(Rabi2D):
    """
    e-f Rabi rate versus e-f drive power -- Omega_ef(V_ef).

    Step 2b of the paper's calibration (their Fig. 2c/2d, purple). Each row is
    a flat-top e-f time Rabi at one amplitude; the oscillation frequency is
    Omega_ef/2pi, and it is linear in gain, so the rates are fit to

        Omega_ef/2pi = m * gain + b

    written to ``device.f0g1.omega_ef_fit``. Together with step 4's
    ``g_tilde_fit`` that is what lets an operating point on the Fig. 3a red
    curve be chosen in gains rather than in rates.

    This is ``Rabi2D`` with ``checkEF``, a flat-top pulse, and a gain y-axis;
    the 2D acquisition and the per-row sine fit are entirely inherited. Only
    the rate extraction, the linear fit and ``update()`` are added.

    Two things to know about the inherited sequence:

    - ``checkEF`` already gives ``pi_ge -> swept e-f pulse -> pi_ge ->
      measure``. The trailing pi_ge is the shelving pulse that turns an
      e-vs-f contrast (which this readout cannot resolve) into g-vs-f.
    - The **time axis is fixed across rows**: ``RabiExperiment`` computes
      ``max_length = 2*num_osc*length`` once at construction, and the y loop
      only rewrites ``gain``. Since Omega_ef is proportional to gain, the
      number of visible periods scales with gain across the scan, so keep the
      gain range to roughly 4-10x or the low rows show no full period and the
      high rows are badly undersampled.

    ``pulse_type`` and ``type`` are aliases in ``RabiExperiment`` and setting
    either sets the other, so only ``pulse_type`` is passed here -- matching
    scripts 04b and 05. (Passing ``type`` used to raise ``KeyError: 'type'``;
    fixed in ``_configure_pulse_params``.)

    Parameters (on top of RabiExperiment's):
    - 'start_y' / 'span_y' / 'expts_y': the gain axis (Rabi2D's own names)
    - 'length': flat duration at which the time window is set
    - 'gain': only sets the inner experiment's defaults; the y sweep overrides
      it row by row
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        style="",
        prefix=None,
        progress=False,
    ):
        # Read before super().__init__, which needs params already merged.
        device = config.load(cfg_dict["cfg_file"]).device
        params_def = {
            "checkEF": True,
            "sweep": "length",
            # The reset's e-f drive shape, device.f0g1.pulse_type_ef. `type` is
            # the alias and follows automatically.
            "pulse_type": pulse_type_from_config(device.get("f0g1"), "pulse_type_ef"),
            "ramp_sigma": 0.02,
            "ramp_sigma_inc": 3,
            "yval": "gain",
            "start_y": 0.02,
            "span_y": 0.18,
            "expts_y": 10,
        }
        params = {**params_def, **params}

        if params["yval"] != "gain":
            raise ValueError(
                f"EFRabiPower sweeps gain on the y axis, got yval={params['yval']!r}"
            )
        if params["sweep"] != "length":
            raise ValueError(
                "EFRabiPower needs sweep='length' (a time Rabi per amplitude), "
                f"got {params['sweep']!r}"
            )

        if prefix is None:
            prefix = f"ef_rabi_vs_power_qubit{qi}"

        super().__init__(cfg_dict, qi=qi, go=go, params=params, style=style,
                         prefix=prefix, progress=progress)

    def analyze(self, data=None, fit=True, min_r2=0.1, **kwargs):
        """
        Read Omega_ef off each row's sine fit, then fit the rates against gain.

        Same two rejection stages as the f0-g1 Stark fit: ``min_r2`` for rows
        the sine did not describe, then sigma-clipping for rates that are real
        fits but off the line.
        """
        if data is None:
            data = self.data

        if not fit:
            return data

        # Rabi2D.analyze -> per-row fitsin on avgi.
        super().analyze(data=data, fit=True, **kwargs)

        gains = np.asarray(data["ypts"], dtype=float)
        xpts = np.asarray(data["xpts"], dtype=float)
        n = len(gains)

        # sinfunc parameters are [yscale, freq, phase_deg, y0]; freq is in MHz
        # because xpts is the pulse length in us. That is Omega_ef/2pi.
        rates = np.full(n, np.nan)
        for i, pars in enumerate(data["fit_avgi"]):
            pars = np.asarray(pars, dtype=float)
            if len(pars) > 1 and np.all(np.isfinite(pars)):
                rates[i] = abs(pars[1])
        r2 = _per_row_r2(xpts, data["avgi"], data["fit_avgi"], fitter.sinfunc)

        data["omega_ef"] = rates
        data["omega_ef_r2"] = r2

        # A fitted period longer than the scanned window is an extrapolation,
        # not a measurement -- there is no full oscillation in the data to set
        # the rate. fitsin's lower bound is 1e-3 MHz, so this is reachable.
        window = float(xpts.max() - xpts.min())
        good = (
            np.isfinite(rates)
            & (rates > 0)
            & (rates * window > 1.0)
            & np.isfinite(r2)
            & (r2 > min_r2)
        )
        n_bad = int(np.sum(~good))
        if n_bad:
            print(
                f"Omega_ef fit: dropped {n_bad} of {n} rows with no usable Rabi "
                f"(R2 <= {min_r2}, or less than one full period in "
                f"{window:.3f} us)"
            )

        if good.sum() < 3:
            print("Omega_ef fit: fewer than 3 usable rows, not fitting")
            data["omega_ef_keep"] = good
            return data

        m, b, keep_sub = _fit_band_model(
            gains[good], rates[good], n_iter=4, sigma=3.0
        )
        keep = np.zeros(n, dtype=bool)
        keep[np.flatnonzero(good)[keep_sub]] = True
        data["omega_ef_keep"] = keep

        n_clipped = int(good.sum() - keep.sum())
        if n_clipped:
            print(f"Omega_ef fit: sigma-clipped {n_clipped} further rows off the line")

        resid = rates[keep] - (m * gains[keep] + b)
        data["omega_ef_fit_m"] = float(m)
        data["omega_ef_fit_b"] = float(b)
        data["omega_ef_resid"] = float(np.median(np.abs(resid)))
        print(
            f"e-f Rabi rate: Omega_ef/2pi = {m:.4f} * gain + {b:.4f} MHz "
            f"(median |residual| {data['omega_ef_resid']:.4f} MHz, "
            f"{int(keep.sum())} of {n} rows)"
        )
        print(
            f"  covered {rates[keep].min():.4f} - {rates[keep].max():.4f} MHz; "
            "step 3's e-f pi pulse is 1/(2*Omega_ef) long, i.e. "
            f"{1 / (2 * rates[keep].max()):.4f} - "
            f"{1 / (2 * rates[keep].min()):.4f} us"
        )
        kappa = self.cfg.device.readout.kappa[self.cfg.expt.qubit[0]]
        print(
            f"  the reset plateau needs Omega_ef of order kappa/4 = "
            f"{kappa / 4:.4f} MHz; check_f0g1_config.py turns that into a gain "
            "once g_tilde_fit exists"
        )
        return data

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data
        super().display(data=data, fit=fit, plot_both=plot_both, **kwargs)
        if "omega_ef" in data:
            self.display_omega_ef(data=data)

    def display_omega_ef(self, data=None):
        """Fitted Rabi rate vs gain, with the linear fit and the rejected rows."""
        if data is None:
            data = self.data

        gains = np.asarray(data["ypts"], dtype=float)
        keep = data.get("omega_ef_keep", np.isfinite(data["omega_ef"]))

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(gains[keep], data["omega_ef"][keep], "o", label="used")
        if np.any(~keep):
            ax.plot(gains[~keep], data["omega_ef"][~keep], "x", color="grey",
                    label="rejected")
        if "omega_ef_fit_m" in data:
            gfine = np.linspace(0, gains.max(), 200)
            ax.plot(
                gfine,
                data["omega_ef_fit_m"] * gfine + data["omega_ef_fit_b"],
                "-",
                label=(
                    f"{data['omega_ef_fit_m']:.3f}$g$ + "
                    f"{data['omega_ef_fit_b']:.3f}"
                ),
            )
        ax.set_xlabel("e-f Drive Gain (DAC level)")
        ax.set_ylabel("$\\Omega_{ef}/2\\pi$ (MHz)")
        ax.legend()

        self.save_fig(fig, suffix="_omega_ef")

    def update(self, verbose=True):
        """Write the linear Omega_ef(gain) model to device.f0g1.omega_ef_fit."""
        if "omega_ef_fit_m" not in self.data:
            print("No Omega_ef fit available; run analyze() first")
            return
        config.update_f0g1(
            self.config_file,
            "omega_ef_fit",
            [self.data["omega_ef_fit_m"], self.data["omega_ef_fit_b"]],
            verbose=verbose,
        )


class F0G1Rabi(QickExperiment):
    """
    f0-g1 Rabi: sweep the drive duration with the qubit prepared in |f>.

    Step 4 of the paper's calibration (their Fig. 2f, light blue). The |f,0>
    population oscillates into |g,1> at the drive-induced coupling g~ while
    |g,1> leaks out at kappa, so the trace gives both rates at once.

    Same sequence as step 1 (``prep="f"``, no probe, no shelving pulse); the
    swept quantity is the drive's flat duration instead of its frequency, and
    the frequency is held on the Stark-shifted resonance read back from
    ``device.f0g1.stark_fit``. Driving off resonance slows the transfer, which
    is indistinguishable from a smaller g~ -- hence the hard dependency on
    step 1 rather than a quiet fallback to the bare frequency.

    Fitted with ``fitter.fit_f0g1_rabi`` (the paper's Eq. C2), not a decaying
    sine: the reset's kappa/3 plateau sits in the *overdamped* regime
    (2*g~ < kappa/2), where there is no oscillation for a sine to measure. On
    synthetic data at g~ = 0.2 MHz, kappa = 1.18 MHz, a decaying sine reports
    0.075 MHz where 2*g~ = 0.4 MHz; this model returns 0.2000 MHz.

    Parameters:
    - 'gain_f0g1': drive amplitude
    - 'frequency_f0g1': override; defaults to the step-1 fit at that gain
    - 'start_length' / 'max_length' / 'expts': the time axis (us). Pick
      ``max_length`` from the slowest transfer expected -- at 2*g~ << kappa the
      transfer rate is the Purcell-like 4*g~^2/kappa, so a weak drive needs a
      long window, while too long a window costs time resolution on the fast
      rows.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        display=True,
        min_r2=None,
        max_err=None,
        check_params=True,
    ):
        if prefix == "":
            prefix = f"f0g1_rabi_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress,
                         qi=qi, check_params=check_params)

        if "f0g1" not in self.cfg.hw.soc.dacs:
            raise ValueError(
                "hw.soc.dacs.f0g1 is missing from this config. The f0-g1 reset "
                "drive needs its own DAC section (ch, type, nyquist, "
                "mixer_freq, filter_*, attn*), mirroring hw.soc.dacs.pump."
            )

        params_def = {
            "reps": self.reps,
            "rounds": self.rounds,
            "expts": 100,
            "final_delay": self.cfg.device.readout.final_delay[qi],
            "gain_f0g1": 0.5,
            "frequency_f0g1": None,
            "start_length": 3 * self.soccfg.cycles2us(1),
            "max_length": 3.0,
            "ramp_sigma_f0g1": 0.02,
            "ramp_sigma_inc_f0g1": 3,
            "pulse_type_f0g1": pulse_type_from_config(self.cfg.device.get("f0g1"), "pulse_type"),
            "prep": "f",
            # Keep the raw shots out of the h5 unless asked; p_e is derived
            # from them in acquire and is what the analysis uses.
            "save_shots": False,
            "ef_probe_on": False,
            "shelve_ge": False,
            "readout_gap": 3 / self.cfg.device.readout.kappa[qi],
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            # Which reset runs between shots: "passive" (the full T1
            # wait), "active" (measurement-based conditional pi pulses)
            # or "f0g1" (the unconditional all-microwave reset, which
            # also clears |f> -- see QickProgram.reset_f0g1).
            "reset_mode": "passive",
            # "f0g1" only, and all optional -- everything else is
            # derived from device.f0g1. residual sets the drive length
            # as ln(1/residual)/(2*pi*Gamma), so it rescales on its own
            # as Gamma improves; f0g1_reset_length pins the length
            # instead. None means device.f0g1.reset_*, else derived.
            "f0g1_reset_residual": None,
            "f0g1_reset_gain": None,
            "f0g1_reset_gain_ef": None,
            "f0g1_reset_length": None,
            "f0g1_reset_gap": None,
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            "reset_verify": False,
            # Latched "while excited" reset; see CLAUDE.md "Reset modes".
            "reset_latch": False,
            # Mean rounds the while loop takes, for the cycle-time estimate
            # only; `reset` is its hard cap. See CLAUDE.md.
            "reset_while_typical": 1,
            # Short drives leave population in |f>, which a pi_ge-only reset
            # cannot clear.
            "reset_f": True,
            # Replaced by a QickSweep1D in acquire(); the value here only has
            # to exist for the Program's direct attribute access.
            "length_f0g1": 3.0,
        }
        params = {**params_def, **params}

        if params["prep"] not in ("g", "e", "f"):
            raise ValueError(
                f"prep must be 'g', 'e' or 'f', got {params['prep']!r}"
            )

        if params["frequency_f0g1"] is None:
            params["frequency_f0g1"] = f0g1_frequency_for_gain(
                self.cfg, params["gain_f0g1"]
            )
        params_def["frequency_f0g1"] = params["frequency_f0g1"]
        params = {**params_def, **params}

        self.cfg.expt = params
        super().check_params(params_def)

        if reset_mode_of(self.cfg.expt) != "passive":
            super().configure_reset()

        print(
            f"f0-g1 Rabi: gain {params['gain_f0g1']} at "
            f"{params['frequency_f0g1']:.4f} MHz (on the step-1 Stark curve), "
            f"{params['start_length']:.4f} - {params['max_length']:.4f} us in "
            f"{params['expts']} points"
        )

        if go:
            super().run(min_r2=min_r2, max_err=max_err, display=display,
                        progress=progress)

    def acquire(self, progress=False):
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay

        # total_length, not length: for a flat_top pulse the swept parameter is
        # the flat segment while the reported axis includes the ramps. That is
        # a constant offset, which the model's t0 absorbs.
        self.param = {
            "label": "f0g1_pulse", "param": "total_length", "param_type": "pulse"
        }
        self.cfg.expt.length_f0g1 = QickSweep1D(
            "sweep_loop", self.cfg.expt.start_length, self.cfg.expt.max_length
        )

        # shots=True so the thresholded population is available alongside the
        # averaged signal. The Rabi model is fit to avgi as before, but p_e is
        # the quantity to compare against the reset-time scans and against
        # script 26's p_dark -- those are all thresholded "not g" fractions,
        # and an averaged I in ADC levels cannot be lined up with them.
        super().acquire(F0G1Program, progress=progress, shots=True)
        self.add_p_e(self.data)
        self._stash_histograms(self.data)
        if not self.cfg.expt.save_shots:
            self.data.pop("shots", None)
        return self.data

    def _stash_histograms(self, data, n_show=3):
        """
        Histogram the shots at a few points along the sweep, before they go.

        Kept as histograms rather than raw shots so the figure survives
        save_shots=False -- the same trade fwm_shots makes. One shared set of
        bins across all three, or the shapes could not be overlaid.

        p_e collapses each distribution to one number, which cannot tell a
        blob that has *moved* from one that has grown a second lobe or a tail.
        That distinction is the whole question for the drive-induced heating,
        so it is worth the few hundred floats.
        """
        if "shots" not in data:
            return data
        qi = self.cfg.expt.qubit[0]
        i_shots = np.asarray(data["shots"])[0, 0, ..., 0]   # (reps, expts)
        n = i_shots.shape[1]
        idx = sorted({0, n // 2, n - 1})[:n_show]
        edges = np.histogram_bin_edges(i_shots.ravel(), bins=60)
        data["hist_bins"] = (edges[:-1] + edges[1:]) / 2
        data["hist_idx"] = np.asarray(idx)
        data["hist_xpts"] = np.asarray(data["xpts"], dtype=float)[idx]
        data["hists"] = np.array([
            np.histogram(i_shots[:, k], bins=edges, density=True)[0]
            for k in idx
        ])
        data["hist_threshold"] = float(self.cfg.device.readout.threshold[qi])
        return data

    def display_histograms(self, data=None):
        """
        Shot histograms at the first, middle and last point of the sweep.

        One figure, overlaid on shared bins, with the readout threshold marked.
        Reading it: a blob that slides toward |e> is real population; one that
        stays put while a second lobe or a high-side tail grows is a mixture;
        one that simply broadens is a readout that has degraded. p_e alone
        gives the same number for all three.
        """
        if data is None:
            data = self.data
        if data is None or "hists" not in data:
            return
        centers = np.asarray(data["hist_bins"], dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        colors = ("C0", "C2", "C3")
        for row, t, color in zip(data["hists"], data["hist_xpts"], colors):
            for ax in axes:
                ax.plot(centers, row, lw=1.4, color=color,
                        label=f"t = {t:.2f} $\mu$s")
        for ax, logy in zip(axes, (False, True)):
            ax.axvline(data["hist_threshold"], ls=":", color="k", lw=1.2,
                       label="threshold")
            ax.set_xlabel("I (ADC levels)")
            if logy:
                ax.set_yscale("log")
        axes[0].set_ylabel("probability density")
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel("probability density (log)")
        fig.suptitle("")
        self.save_fig(fig, suffix="_hists")

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        if fit:
            self.fitterfunc = fitter.fit_f0g1_rabi
            self.fitfunc = fitter.f0g1_rabi_func
            super().analyze(use_i=False, **kwargs)
            if "best_fit" in data:
                data["g_tilde"] = abs(data["best_fit"][0])
                data["kappa_fit"] = abs(data["best_fit"][1])
                kappa = data["kappa_fit"]
                damping = "over" if 2 * data["g_tilde"] < kappa / 2 else "under"
                print(
                    f"f0-g1 Rabi at gain {self.cfg.expt.gain_f0g1}: "
                    f"g~/2pi = {data['g_tilde']:.4f} MHz, "
                    f"kappa/2pi = {kappa:.4f} MHz ({damping}damped)"
                )
                omega_opt, gamma_max = optimal_omega_ef(data["g_tilde"], kappa)
                print(
                    f"  at this g~ the best reset rate is Gamma/2pi = "
                    f"{gamma_max:.4f} MHz, {gamma_max / (kappa / 3) * 100:.0f}% of "
                    f"kappa/3 = {kappa / 3:.4f} MHz, needing Omega_ef/2pi = "
                    f"{omega_opt:.4f} MHz"
                )

        return data

    def display(self, data=None, fit=True, ax=None, plot_all=False, **kwargs):
        title = (
            f"f0-g1 Rabi Q{self.cfg.expt.qubit[0]}, gain "
            f"{self.cfg.expt.gain_f0g1} at "
            f"{self.cfg.expt.frequency_f0g1:.3f} MHz"
        )
        caption_params = [
            {"index": 0, "format": "$\\tilde{{g}}/2\\pi$: {val:.4} MHz"},
            {"index": 1, "format": "$\\kappa/2\\pi$: {val:.4} MHz"},
        ]
        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel="f0-g1 Drive Duration ($\\mu$s)",
            fit=fit,
            show_hist=False,
            caption_params=caption_params,
            **kwargs,
        )
        if data is None:
            data = self.data
        if data is not None and "p_e" in data:
            self.display_p_e(data=data)
        if data is not None and "hists" in data:
            self.display_histograms(data=data)

    def display_p_e(self, data=None):
        """
        The thresholded population against drive duration.

        A second figure rather than a panel on the first: the shared display
        plots avgi in ADC levels, and p_e is the only axis that can be lined
        up with the reset-time scans and with script 26's p_dark. The
        readout_gap is in the label because that is the variable this was
        built to scan -- with prep="g" the rise of this curve is the deposit
        accumulating with drive time, and repeating at several gaps maps how
        long it takes to become visible.
        """
        if data is None:
            data = self.data
        if "p_e" not in data:
            return
        t = np.asarray(data["xpts"], dtype=float)
        pe = np.asarray(data["p_e"], dtype=float)
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(t, pe, "o-", ms=3, color="C3")
        slope, icept = np.polyfit(t, pe, 1)
        ax.plot(t, slope * t + icept, "-", color="k", lw=1,
                label=f"{icept * 100:.2f}% {slope * 100:+.3f}%/$\mu$s")
        ax.set_xlabel("f0-g1 Drive Duration ($\mu$s)")
        ax.set_ylabel("P(not g)")
        ax.set_title(
            f"prep |{self.cfg.expt.prep}>, gain {self.cfg.expt.gain_f0g1}, "
            f"readout_gap {self.cfg.expt.readout_gap:.2f} $\mu$s",
            fontsize="medium")
        ax.legend(fontsize=8)
        self.save_fig(fig, suffix="_p_e")
        print(f"  p_e vs drive duration: {icept * 100:.2f}% at t=0, "
              f"slope {slope * 100:+.3f}%/us  "
              f"(readout_gap {self.cfg.expt.readout_gap:.2f} us)")


class F0G1RabiPower(QickExperiment2DSimple):
    """
    f0-g1 Rabi versus drive power -- g~(V_f0g1) and kappa.

    Step 4b of the paper's calibration (their Fig. 2d, light blue). One Eq.-C2
    fit per amplitude gives g~ and kappa at that power; g~ is linear in gain,
    so the rates are fit to

        g~/2pi = m * gain + b

    written to ``device.f0g1.g_tilde_fit``, and ``device.f0g1.kappa`` gets the
    median of the per-row kappa. With step 2's ``omega_ef_fit`` that completes
    the calibration: an operating point on the paper's Fig. 3a red curve can
    then be chosen in gains rather than in rates, which ``analyze()`` prints.

    The drive frequency is co-varied with the gain from
    ``device.f0g1.stark_fit``, so every row sits on the Stark-shifted
    resonance -- the same trick scripts 23/24 use to walk the FWM band. An
    uncorrected frequency would detune the drive further at every step and read
    as g~ saturating.

    kappa is fit per row rather than shared across rows, which the paper does
    constrain. Per-row first is deliberate: the scatter across rows is the
    honest error bar on kappa, and a kappa that drifts with power is a signal
    that something else is wrong -- the frequency tracking, most likely --
    rather than something to average away.

    Parameters:
    - 'start_gain' / 'max_gain' / 'expts_gain': the power axis
    - 'max_length' / 'expts': the time axis, fixed across rows
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        display=True,
        min_r2=None,
        max_err=None,
        live_plot=False,
    ):
        if prefix == "":
            prefix = f"f0g1_rabi_vs_power_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress,
                         qi=qi, live_plot=live_plot)
        self.nested_progress = False
        self.progress_desc = "f0-g1 gain sweep"

        params_def = {
            "start_gain": 0.1,
            "max_gain": self.cfg.device.qubit.max_gain,
            "expts_gain": 12,
        }
        params = {**params_def, **params}

        self.expt = F0G1Rabi(
            cfg_dict, qi=qi, go=False, params=params, check_params=False,
        )
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        if go:
            self.run(progress=progress, display=display)

    def acquire(self, progress=False):
        gainpts = np.linspace(
            self.cfg.expt.start_gain,
            self.cfg.expt.max_gain,
            self.cfg.expt.expts_gain,
        )
        freqpts = np.array(
            [f0g1_frequency_for_gain(self.cfg, g) for g in gainpts]
        )
        print(
            f"f0-g1 drive tracks its Stark shift: {freqpts[0]:.4f} MHz at gain "
            f"{gainpts[0]:.3f} -> {freqpts[-1]:.4f} MHz at gain {gainpts[-1]:.3f}"
        )
        ysweep = [
            {"pts": gainpts, "var": "gain_f0g1"},
            {"pts": freqpts, "var": "frequency_f0g1"},
        ]

        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {
            "label": "f0g1_pulse", "param": "total_length", "param_type": "pulse"
        }
        self.xlabel = "f0-g1 Drive Duration ($\\mu$s)"
        self.ylabel = YLABEL

        super().acquire(ysweep, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, min_r2=0.1, **kwargs):
        """
        Eq.-C2 fit per drive power, then g~ against gain.

        Same two rejection stages as the Stark fits: ``min_r2`` for rows the
        model did not describe, then sigma-clipping for values that are real
        fits but off the line.
        """
        if data is None:
            data = self.data

        if not fit:
            return data

        super().analyze(
            fitterfunc=fitter.fit_f0g1_rabi, fitfunc=fitter.f0g1_rabi_func
        )

        gains = np.asarray(data["ypts"], dtype=float)
        xpts = np.asarray(data["xpts"], dtype=float)
        n = len(gains)

        g_tildes = np.full(n, np.nan)
        kappas = np.full(n, np.nan)
        for i, pars in enumerate(data["fit_avgi"]):
            pars = np.asarray(pars, dtype=float)
            if len(pars) > 1 and np.all(np.isfinite(pars)):
                g_tildes[i] = abs(pars[0])
                kappas[i] = abs(pars[1])
        r2 = _per_row_r2(
            xpts, data["avgi"], data["fit_avgi"], fitter.f0g1_rabi_func
        )

        data["g_tilde"] = g_tildes
        data["kappa_row"] = kappas
        data["g_tilde_r2"] = r2

        good = (
            np.isfinite(g_tildes)
            & (g_tildes > 0)
            & np.isfinite(kappas)
            & (kappas > 0)
            & np.isfinite(r2)
            & (r2 > min_r2)
        )
        n_bad = int(np.sum(~good))
        if n_bad:
            print(
                f"g~ fit: dropped {n_bad} of {n} rows the Eq.-C2 model did not "
                f"describe (R2 <= {min_r2})"
            )

        if good.sum() < 3:
            print("g~ fit: fewer than 3 usable rows, not fitting")
            data["g_tilde_keep"] = good
            return data

        m, b, keep_sub = _fit_band_model(
            gains[good], g_tildes[good], n_iter=4, sigma=3.0
        )
        keep = np.zeros(n, dtype=bool)
        keep[np.flatnonzero(good)[keep_sub]] = True
        data["g_tilde_keep"] = keep

        n_clipped = int(good.sum() - keep.sum())
        if n_clipped:
            print(f"g~ fit: sigma-clipped {n_clipped} further rows off the line")

        resid = g_tildes[keep] - (m * gains[keep] + b)
        data["g_tilde_fit_m"] = float(m)
        data["g_tilde_fit_b"] = float(b)
        data["g_tilde_resid"] = float(np.median(np.abs(resid)))
        data["kappa_median"] = float(np.median(kappas[keep]))
        data["kappa_spread"] = float(
            np.median(np.abs(kappas[keep] - data["kappa_median"]))
        )
        print(
            f"f0-g1 coupling: g~/2pi = {m:.4f} * gain + {b:.4f} MHz "
            f"(median |residual| {data['g_tilde_resid']:.4f} MHz, "
            f"{int(keep.sum())} of {n} rows)"
        )
        kappa_cfg = self.cfg.device.readout.kappa[self.cfg.expt.qubit[0]]
        print(
            f"  kappa/2pi = {data['kappa_median']:.4f} +/- "
            f"{data['kappa_spread']:.4f} MHz (median +/- MAD across rows); "
            f"config has {kappa_cfg:.4f}"
        )

        # g~ must grow with drive amplitude. A non-positive slope is not a
        # measurement, it is an unconstrained fit -- most often because the
        # time window did not cover the transfer, which at 2*g~ << kappa takes
        # ~kappa/(4*g~^2) and gets slow fast as the drive weakens.
        data["g_tilde_fit_ok"] = bool(m > 0 and m * float(np.max(gains)) + b > 0)
        if not data["g_tilde_fit_ok"]:
            print(
                f"  WARNING: the fitted slope is {m:+.4f} MHz per unit gain. g~ "
                "cannot fall as the drive gets stronger, so this fit is not a "
                "measurement. Most likely max_length is too short to see the "
                "transfer at these gains (it takes about kappa/(4*g~^2), so "
                "~1 us at g~ = 0.2 MHz but ~100 us at g~ = 0.02 MHz), or the "
                "drive is too attenuated to reach |f> -> |g,1> at all. No "
                "operating point is reported and update() will refuse to write."
            )
            return data

        if not 1 / 3 < data["kappa_median"] / kappa_cfg < 3:
            print(
                f"  WARNING: the fitted kappa is {data['kappa_median'] / kappa_cfg:.2f}x "
                "the config value. Either the resonator linewidth in the config "
                "is stale, or the rows are not constraining the fit -- check the "
                "lower panel of the _g_tilde figure for a kappa that drifts "
                "with power."
            )

        self._report_operating_point(data)
        return data

    def _report_operating_point(self, data):
        """
        How far along the Fig. 3a red curve the measured g~ range reaches.

        Only called once ``g_tilde_fit_ok`` is set, so g_max is positive and
        the scan below cannot hand ``optimal_omega_ef`` a negative coupling.
        """
        kappa = data["kappa_median"]
        m, b = data["g_tilde_fit_m"], data["g_tilde_fit_b"]
        g_max = m * float(np.max(data["ypts"])) + b
        target = kappa / 3
        print(f"  best possible reset rate Gamma/2pi = kappa/3 = {target:.4f} MHz")

        plateau = None
        for g in np.linspace(max(g_max / 40, 1e-3), g_max, 80):
            omega, gamma = optimal_omega_ef(g, kappa)
            if gamma > 0.99 * target:
                plateau = (g, omega)
                break

        if plateau is None:
            omega, gamma = optimal_omega_ef(g_max, kappa)
            print(
                f"  g~ only reaches {g_max:.4f} MHz at full gain, giving "
                f"Gamma/2pi = {gamma:.4f} MHz, {gamma / target * 100:.0f}% of the "
                "plateau. More f0-g1 drive power is needed."
            )
            return

        g, omega = plateau
        gain_f0g1 = (g - b) / m if m else float("nan")
        print(
            f"  plateau reached at g~/2pi = {g:.4f} MHz, i.e. f0g1 gain "
            f"{gain_f0g1:.4f}, with Omega_ef/2pi = {omega:.4f} MHz"
        )
        try:
            om, ob = (float(v) for v in self.cfg.device.f0g1.omega_ef_fit)
            gain_ef = (omega - ob) / om
            print(
                f"  -> e-f gain {gain_ef:.4f} from omega_ef_fit. That pair is "
                "the operating point for the reset scan."
            )
        except Exception:
            print("  -> e-f gain unknown; run 30_ef_rabi_vs_power.py")

    def display(self, data=None, fit=True, plot_amps=True, ax=None, **kwargs):
        if data is None:
            data = self.data

        title = (
            f"f0-g1 Rabi vs power, Q{self.cfg.expt.qubit[0]}, drive tracking "
            "its own Stark shift"
        )
        super().display(
            data=data, ax=ax, plot_amps=plot_amps, title=title,
            xlabel="f0-g1 Drive Duration ($\\mu$s)", ylabel=YLABEL, fit=fit,
            **kwargs
        )
        if "g_tilde" in data:
            self.display_g_tilde(data=data)

    def display_g_tilde(self, data=None):
        """g~ and kappa vs drive gain, with the linear fit and the kappa/4 line."""
        if data is None:
            data = self.data

        gains = np.asarray(data["ypts"], dtype=float)
        keep = data.get("g_tilde_keep", np.isfinite(data["g_tilde"]))

        fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        axes[0].plot(gains[keep], data["g_tilde"][keep], "o", label="used")
        if np.any(~keep):
            axes[0].plot(gains[~keep], data["g_tilde"][~keep], "x", color="grey",
                         label="rejected")
        if "g_tilde_fit_m" in data:
            gfine = np.linspace(0, gains.max(), 200)
            axes[0].plot(
                gfine,
                data["g_tilde_fit_m"] * gfine + data["g_tilde_fit_b"],
                "-",
                label=(
                    f"{data['g_tilde_fit_m']:.3f}$g$ + "
                    f"{data['g_tilde_fit_b']:.3f}"
                ),
            )
        if "kappa_median" in data:
            # 2*g~ = kappa/2 separates a damped oscillation from a pure decay,
            # and is roughly where the reset rate plateaus.
            axes[0].axhline(
                data["kappa_median"] / 4, ls=":", color="k",
                label="$\\kappa/4$ (damping boundary)",
            )
        axes[0].set_ylabel("$\\tilde{g}/2\\pi$ (MHz)")
        axes[0].legend(fontsize=8)

        axes[1].plot(gains[keep], data["kappa_row"][keep], "o")
        if "kappa_median" in data:
            axes[1].axhline(data["kappa_median"], ls="--", color="C1",
                            label=f"median {data['kappa_median']:.3f}")
            axes[1].legend(fontsize=8)
        axes[1].set_ylabel("$\\kappa/2\\pi$ (MHz)")
        axes[1].set_xlabel(YLABEL)

        self.save_fig(fig, suffix="_g_tilde")

    def update(self, verbose=True):
        """Write g_tilde_fit and kappa to device.f0g1."""
        if "g_tilde_fit_m" not in self.data:
            print("No g~ fit available; run analyze() first")
            return
        if not self.data.get("g_tilde_fit_ok", False):
            # A falling g~ is unphysical; writing it would silently poison
            # every gain step 5 derives from it.
            print(
                "Refusing to write g_tilde_fit: the fitted slope is not "
                "positive (see the warning from analyze()). Fix the time "
                "window or the drive power and retake."
            )
            return
        config.update_f0g1(
            self.config_file,
            "g_tilde_fit",
            [self.data["g_tilde_fit_m"], self.data["g_tilde_fit_b"]],
            verbose=verbose,
        )
        config.update_f0g1(
            self.config_file, "kappa", self.data["kappa_median"], verbose=verbose
        )


class ResetTimeScan(QickExperiment):
    """
    The reset measurement: residual excitation versus reset time.

    Step 5, the paper's Fig. 4. Both drives are turned on together for a time
    t_r and the qutrit is read out, for each of three preparations:

    - ``g``: nothing to remove. This is the **floor** -- the |g> blob's tail
      above ``device.readout.threshold`` plus thermal population. On run 33
      Waymar the threshold sits 1.6 sigma above g_mean, so expect a few
      percent. Everything else has to be read against it.
    - ``e``: the headline trace. P(not g) here is Pexc = Pe + Pf.
    - ``f``: the harder case, since |f> has to come down two levels.

    The observable is the thresholded "not g" fraction, which *is* Pe + Pf --
    exactly the paper's Pexc. This device cannot resolve |e> from |f> (chi >>
    kappa puts those blobs ~0.5 sigma apart), so the paper's three-state
    assignment matrix is not available and is not attempted; the g/not-g split
    is both what the readout supports and what the quantity of interest needs.

    Both gains default to the calibrated optimum: ``gain_f0g1`` at full scale
    (more coupling is always better below the plateau) and ``gain_ef`` from
    ``optimal_omega_ef`` at the resulting g~, inverted through
    ``omega_ef_fit``. Both can be overridden.

    ``analyze()`` fits the ``e`` trace to a single exponential and reports the
    rate against the prediction from the paper's Eq. 1, the steady-state Pexc,
    and -- importantly -- how the reset time constant compares with T1. If the
    reset is not much faster than passive decay, the measurement is still
    valid but the protocol is not yet buying anything, and it says so.

    Parameters:
    - 'gain_f0g1' / 'gain_ef': the operating point; both derived if omitted
    - 'max_length' / 'expts': the reset-time axis (us). Print-out at
      construction gives the predicted 1% time, which is what to set this from.
    - 'preps': which preparations to run, default ('g', 'e', 'f')
    - 'shots': repetitions per time point
    """

    PREPS = ("g", "e", "f")
    _PREP_NAMES = {
        "g": "prepared |g> (floor)",
        "e": "prepared |e>",
        "f": "prepared |f>",
    }

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=True,
        display=True,
        min_r2=None,
        max_err=None,
        check_params=True,
    ):
        if prefix == "":
            prefix = f"f0g1_reset_time_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress,
                         qi=qi, check_params=check_params)

        if "f0g1" not in self.cfg.hw.soc.dacs:
            raise ValueError(
                "hw.soc.dacs.f0g1 is missing from this config. The f0-g1 reset "
                "drive needs its own DAC section (ch, type, nyquist, "
                "mixer_freq, filter_*, attn*), mirroring hw.soc.dacs.pump."
            )

        params_def = {
            "reps": self.reps,
            "rounds": self.rounds,
            "expts": 60,
            "final_delay": self.cfg.device.readout.final_delay[qi],
            "gain_f0g1": self.cfg.device.qubit.max_gain,
            "gain_ef": None,
            "frequency_f0g1": None,
            "frequency_ef": None,
            "start_length": 3 * self.soccfg.cycles2us(1),
            "max_length": 20.0,
            "ramp_sigma_f0g1": 0.02,
            "ramp_sigma_inc_f0g1": 3,
            "pulse_type_f0g1": pulse_type_from_config(self.cfg.device.get("f0g1"), "pulse_type"),
            "ramp_sigma_ef": 0.02,
            "ramp_sigma_inc_ef": 3,
            "pulse_type_ef": pulse_type_from_config(self.cfg.device.get("f0g1"), "pulse_type_ef"),
            "preps": list(self.PREPS),   # list, not tuple: save_config writes yaml
            # Preparations to ALSO run with both drive gains at zero, as
            # a same-timing baseline. See acquire() for why zero gain
            # rather than removing the pulses.
            "baseline_preps": [],
            # Both drives run for the same reset time, so they start together.
            "ef_probe_on": True,
            "ef_align": "start",
            # No shelving pulse: the observable is P(not g) = Pe + Pf, and both
            # excited states already read as not-g.
            "shelve_ge": False,
            "prep": "e",
            "readout_gap": 3 / self.cfg.device.readout.kappa[qi],
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            # Which reset runs between shots: "passive" (the full T1
            # wait), "active" (measurement-based conditional pi pulses)
            # or "f0g1" (the unconditional all-microwave reset, which
            # also clears |f> -- see QickProgram.reset_f0g1).
            "reset_mode": "passive",
            # "f0g1" only, and all optional -- everything else is
            # derived from device.f0g1. residual sets the drive length
            # as ln(1/residual)/(2*pi*Gamma), so it rescales on its own
            # as Gamma improves; f0g1_reset_length pins the length
            # instead. None means device.f0g1.reset_*, else derived.
            "f0g1_reset_residual": None,
            "f0g1_reset_gain": None,
            "f0g1_reset_gain_ef": None,
            "f0g1_reset_length": None,
            "f0g1_reset_gap": None,
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            "reset_verify": False,
            # Latched "while excited" reset; see CLAUDE.md "Reset modes".
            "reset_latch": False,
            # Mean rounds the while loop takes, for the cycle-time estimate
            # only; `reset` is its hard cap. See CLAUDE.md.
            "reset_while_typical": 1,
            # Short reset times leave population in |f>, which a pi_ge-only
            # reset between shots cannot clear.
            "reset_f": True,
            # Replaced by QickSweep1D objects in acquire(); these only have to
            # exist for the Program's direct attribute access.
            "length_f0g1": 1.0,
            "length_ef": 1.0,
            "save_shots": False,
        }
        params = {**params_def, **params}

        unknown = [p for p in params["baseline_preps"] if p not in self.PREPS]
        if unknown:
            raise ValueError(
                f"baseline_preps must be drawn from {self.PREPS}, got "
                f"{unknown}"
            )
        unknown = [p for p in params["preps"] if p not in self.PREPS]
        if unknown:
            raise ValueError(
                f"Unknown prep(s) {unknown}; expected any of {list(self.PREPS)}."
            )
        if "e" not in params["preps"] or "g" not in params["preps"]:
            raise ValueError(
                "preps must include 'g' and 'e': 'e' is the reset trace and 'g' "
                "is the readout floor it has to be read against."
            )

        # Both tones run for the same swept time, so their shapes and ramps
        # must match or one would end before the other.
        if params["pulse_type_f0g1"] != params["pulse_type_ef"]:
            raise ValueError(
                "the f0-g1 and e-f drives must have the same pulse type so both "
                f"cover the same reset time; got {params['pulse_type_f0g1']!r} "
                f"and {params['pulse_type_ef']!r} (device.f0g1.pulse_type / "
                "pulse_type_ef)."
            )
        if params["pulse_type_f0g1"] == "flat_top" and (
                params["ramp_sigma_f0g1"] != params["ramp_sigma_ef"]
                or params["ramp_sigma_inc_f0g1"] != params["ramp_sigma_inc_ef"]):
            raise ValueError(
                "the f0-g1 and e-f ramps must match so both drives cover the "
                "same reset time; got "
                f"{params['ramp_sigma_f0g1']}x{params['ramp_sigma_inc_f0g1']} "
                f"and {params['ramp_sigma_ef']}x{params['ramp_sigma_inc_ef']}."
            )

        # --- the operating point -------------------------------------------
        kappa = self.cfg.device.get("f0g1", {}).get("kappa", None)
        if kappa is None or not np.isfinite(kappa):
            kappa = float(self.cfg.device.readout.kappa[qi])
            kappa_src = f"device.readout.kappa[{qi}]"
        else:
            kappa = float(kappa)
            kappa_src = "device.f0g1.kappa"

        g_tilde = g_tilde_for_gain(self.cfg, params["gain_f0g1"])
        omega_opt, gamma_pred = optimal_omega_ef(g_tilde, kappa)
        print(
            f"f0-g1 reset scan: f0g1 gain {params['gain_f0g1']} gives "
            f"g~/2pi = {g_tilde:.4f} MHz; kappa/2pi = {kappa:.4f} MHz from "
            f"{kappa_src}"
        )
        print(
            f"  optimal Omega_ef/2pi = {omega_opt:.4f} MHz -> predicted "
            f"Gamma/2pi = {gamma_pred:.4f} MHz, "
            f"{gamma_pred / (kappa / 3) * 100:.0f}% of the kappa/3 = "
            f"{kappa / 3:.4f} MHz ceiling"
        )
        # omega_ef_target is the Omega_ef this gain is *meant* to produce, kept
        # so analyze() predicts Gamma from the same number the gain was chosen
        # for. Recomputing it forward through omega_ef_fit would re-apply the
        # intercept that ef_gain_for_omega drops, and quote a different rate
        # than the one being aimed at.
        if params["gain_ef"] is None:
            params["gain_ef"] = ef_gain_for_omega(self.cfg, omega_opt)
            params["omega_ef_target"] = omega_opt
        else:
            params["omega_ef_target"] = omega_ef_for_gain(
                self.cfg, params["gain_ef"]
            )
            print(
                f"  e-f gain {params['gain_ef']} given explicitly -> "
                f"Omega_ef/2pi = {params['omega_ef_target']:.4f} MHz from "
                "omega_ef_fit"
            )
        params_def["gain_ef"] = params["gain_ef"]
        params_def["omega_ef_target"] = params["omega_ef_target"]

        # At the tiny gains the optimum calls for, the fit's intercept can be
        # the whole of Omega_ef -- so the number above is an extrapolation, not
        # a measurement, and the operating point is worth bracketing.
        b_ef = float(self.cfg.device.f0g1.omega_ef_fit[1])
        if abs(b_ef) > params["omega_ef_target"] / 2:
            print(
                f"  NOTE: omega_ef_fit's intercept ({b_ef:+.4f} MHz) is "
                f"comparable to the target Omega_ef "
                f"({params['omega_ef_target']:.4f} MHz), so the e-f gain above "
                "rests on extrapolating step 2's fit well below the range it "
                "measured. Treat it as a starting point and scan gain_ef "
                "around it -- Gamma is not monotonic in Omega_ef (the paper's "
                "configuration C is slower than B precisely by overshooting)."
            )

        # Time to 1% residual at the predicted rate, which is what max_length
        # should be set from. Gamma is a rate/2pi, hence the 2*pi.
        if gamma_pred > 0:
            t_1pct = np.log(100) / (2 * np.pi * gamma_pred)
            print(
                f"  predicted time to 1% residual: {t_1pct:.2f} us "
                f"(max_length is {params['max_length']:.2f} us)"
            )
            t1 = float(self.cfg.device.qubit.T1[qi])
            if t_1pct > t1 / 3:
                print(
                    f"  WARNING: that is not much faster than passive decay -- "
                    f"T1 = {t1:.1f} us would reach 1% in {np.log(100) * t1:.0f} us. "
                    "The scan will still be valid, but at this drive strength "
                    "the protocol is barely beating doing nothing. More f0-g1 "
                    "power (less attenuation) is what buys speed."
                )

        # Both drive frequencies on their Stark-shifted resonances.
        if params["frequency_f0g1"] is None:
            params["frequency_f0g1"] = f0g1_frequency_for_gain(
                self.cfg, params["gain_f0g1"]
            )
        if params["frequency_ef"] is None:
            ef_fit = self.cfg.device.get("f0g1", {}).get("ef_stark_fit", None)
            if ef_fit is not None and np.all(
                np.isfinite(np.array(ef_fit, dtype=float))
            ):
                a, c = (float(v) for v in ef_fit)
                params["frequency_ef"] = a * params["gain_f0g1"] ** 2 + c
                ef_src = "ef_stark_fit"
            else:
                params["frequency_ef"] = self.cfg.device.qubit.f_ef[qi]
                ef_src = "device.qubit.f_ef (unshifted -- run 32)"
            print(
                f"  e-f drive at {params['frequency_ef']:.4f} MHz from {ef_src}"
            )
        params_def["frequency_f0g1"] = params["frequency_f0g1"]
        params_def["frequency_ef"] = params["frequency_ef"]
        params = {**params_def, **params}

        self.cfg.expt = params
        super().check_params(params_def)

        if reset_mode_of(self.cfg.expt) != "passive":
            super().configure_reset()

        self.data = {}
        if go:
            super().run(min_r2=min_r2, max_err=max_err, display=display,
                        progress=progress)

    def acquire(self, progress=False):
        """
        One hardware time sweep per preparation, thresholded into P(not g).

        The three preparations are a Python loop rather than a sweep axis
        because ``prep`` selects which pi pulses the Program *declares*, not a
        pulse parameter -- it cannot be a QickSweep1D. Each pass is a full
        hardware sweep over the reset time, so the loop costs three
        acquisitions, not 3*expts.
        """
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay

        self.param = {
            "label": "f0g1_pulse", "param": "total_length", "param_type": "pulse"
        }
        # Two sweeps on one loop: both drives step together, so the pair is on
        # for the same reset time at every point.
        self.cfg.expt.length_f0g1 = QickSweep1D(
            "sweep_loop", self.cfg.expt.start_length, self.cfg.expt.max_length
        )
        self.cfg.expt.length_ef = QickSweep1D(
            "sweep_loop", self.cfg.expt.start_length, self.cfg.expt.max_length
        )

        data = {}
        requested_prep = self.cfg.expt.prep
        for prep in self.cfg.expt.preps:
            self.cfg.expt.prep = prep
            # QickExperiment.acquire owns the shots plumbing and pulls xpts
            # from self.param, so each pass goes through the tested path and
            # the per-prep results are copied out afterwards.
            d = super().acquire(F0G1Program, progress=progress, shots=True)
            if "xpts" not in data:
                data["xpts"] = np.asarray(d["xpts"])
            for key in ("avgi", "avgq", "amps", "phases"):
                data[f"{key}_{prep}"] = np.asarray(d[key])
            # Reuse add_p_e's shot indexing rather than repeating it.
            tmp = {"shots": d["shots"]}
            self.add_p_e(tmp)
            data[f"p_e_{prep}"] = tmp["p_e"]
            if self.cfg.expt.save_shots:
                data[f"shots_{prep}"] = np.asarray(d["shots"])
        self.cfg.expt.prep = requested_prep

        # The same-timing baseline: identical sequence with both drive gains
        # at zero. Zeroing the gain rather than removing the pulses is the
        # point -- the pulses still occupy their slots, so the tproc timing,
        # the readout_gap and the inter-shot delay are bit-identical and the
        # only difference is that no power comes out. Any difference between
        # a trace and its baseline is therefore the drives and nothing else.
        #
        # This is what makes "is the reset working" answerable: the |e> trace
        # decays partly because the reset clears it and partly because T1
        # does, and only the baseline separates those. For |g> it measures
        # drive-induced excitation directly, which a flat |g> trace alone
        # cannot distinguish from readout misassignment.
        if self.cfg.expt.baseline_preps:
            gain_f0g1, gain_ef = self.cfg.expt.gain_f0g1, self.cfg.expt.gain_ef
            self.cfg.expt.gain_f0g1 = 0
            self.cfg.expt.gain_ef = 0
            try:
                for prep in self.cfg.expt.baseline_preps:
                    self.cfg.expt.prep = prep
                    d = super().acquire(F0G1Program, progress=progress, shots=True)
                    for key in ("avgi", "avgq", "amps", "phases"):
                        data[f"{key}_{prep}_off"] = np.asarray(d[key])
                    tmp = {"shots": d["shots"]}
                    self.add_p_e(tmp)
                    data[f"p_e_{prep}_off"] = tmp["p_e"]
            finally:
                # Restore even on failure, or a later save_config would record
                # zero gains and misdescribe the run.
                self.cfg.expt.gain_f0g1 = gain_f0g1
                self.cfg.expt.gain_ef = gain_ef
                self.cfg.expt.prep = requested_prep

        # The shared display/save path expects the headline trace under the
        # usual keys.
        for key in ("avgi", "avgq", "amps", "phases"):
            data[key] = data[f"{key}_e"]
        data["p_e"] = data["p_e_e"]

        self.data = data
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Fit the |e> trace's approach to steady state and report the rate.

        A single exponential, not the paper's master equation: it is the one
        number worth comparing against ``reset_rate``, and the oscillatory
        structure the paper resolves needs a rate an order of magnitude above
        what this drive strength reaches.
        """
        if data is None:
            data = self.data
        if not fit or "p_e_e" not in data:
            return data

        t = np.asarray(data["xpts"], dtype=float)
        n_tail = max(3, len(t) // 10)

        for prep in self.cfg.expt.preps:
            trace = np.asarray(data[f"p_e_{prep}"], dtype=float)
            data[f"p_sat_{prep}"] = float(np.mean(trace[-n_tail:]))

        floor = data["p_sat_g"]
        print(
            f"readout floor from the |g> preparation: P(not g) = "
            f"{floor:.4f} at long reset times"
        )

        # The drives-off baseline, where it was run. This is the only thing in
        # the experiment that attributes the floor, so report it plainly.
        for prep in self.cfg.expt.baseline_preps:
            key = f"p_e_{prep}_off"
            if key not in data:
                continue
            off = np.asarray(data[key], dtype=float)
            sat_off = float(np.mean(off[-n_tail:]))
            data[f"p_sat_{prep}_off"] = sat_off
            excess = data[f"p_sat_{prep}"] - sat_off
            data[f"p_drive_{prep}"] = excess
            print(
                f"  |{prep}> baseline (same timing, both gains 0): "
                f"{sat_off:.4f} -> the drives add {excess:+.4f}"
            )
            if prep == "g":
                if excess > 0.01:
                    print(
                        "     that is drive-induced excitation, not "
                        "misassignment -- 35b_reset_floor.py separates which "
                        "of the two tones does it"
                    )
                else:
                    print(
                        "     so the floor is NOT the drives; compare it "
                        "against the |g> blob of the single-shot calibration "
                        "to see how much is misassignment"
                    )

        # The |g> trace's *shape* separates three unrelated effects that all
        # raise P(not g), and which are easy to confuse:
        #
        #   flat     - EITHER pure readout misassignment (the |g> blob's tail
        #              above device.readout.threshold) OR excitation that
        #              saturates faster than the first time point. The shape
        #              alone cannot tell these apart, so this case is
        #              deliberately reported as ambiguous -- see below.
        #   decaying - the shot started with excitation the reset then cleared.
        #              Expected when the "active" reset mode prepared it (reset_e is ~9%
        #              here, against ~1.6% thermal) -- a preparation problem,
        #              not a reset problem.
        #   rising   - the drives are adding excitation on the scan's timescale.
        #
        # This used to call a flat trace "readout misassignment alone, not
        # excitation", which is wrong and was believed on run 33 #370: the
        # trace was flat at 11.35% while the single-shot calibration #377
        # measured only 3.88% misassignment, and both a threshold estimate
        # (9.2%) and the blob mean (11.8%, which never touches the threshold)
        # put the rest down to real excitation. Fast heating is flat too.
        # Only measuring the floor with the drives off settles it, which is
        # what scripts/SMPD/35b_reset_floor.py does.
        g_trace = np.asarray(data["p_e_g"], dtype=float)
        g_head = float(np.mean(g_trace[:n_tail]))
        drift = floor - g_head
        data["p_g_head"] = g_head
        data["p_g_drift"] = drift
        scale = max(abs(floor), 1e-6)
        if abs(drift) < 0.2 * scale:
            # A flat trace is ambiguous on its own, but the drives-off
            # baseline resolves it outright, so defer to that when it ran
            # rather than printing a contradiction.
            if "p_drive_g" in data:
                verdict = (
                    f"flat, and the baseline above already attributes it: "
                    f"{data['p_drive_g']:+.4f} of this floor is the drives, "
                    f"{data['p_sat_g_off']:.4f} is there without them"
                )
            else:
                verdict = (
                    "flat -> AMBIGUOUS. Either readout misassignment, or "
                    "excitation that saturates faster than the first point; a "
                    "flat trace does not distinguish them. Add "
                    "baseline_preps=['g'] to settle it, or run "
                    "35b_reset_floor.py to attribute it per tone"
                )
        elif drift < 0:
            verdict = (
                "DECAYING -> the shots began with excitation the reset then "
                "cleared. Check the reset mode: \"active\" leaves "
                f"{self.cfg.device.readout.reset_e[self.cfg.expt.qubit[0]]} in "
                "|e> against device.qubit.pop at equilibrium, so a passive run "
                "prepares |g> far better"
            )
        else:
            verdict = (
                "RISING -> the drives are adding excitation while they are on. "
                "Rerun with gain_ef=0, then gain_f0g1=0, to see which tone "
                "does it (gain 0 keeps the sequence timing identical)"
            )
        print(
            f"  |g> trace runs {g_head:.4f} -> {floor:.4f} "
            f"({drift:+.4f}): {verdict}"
        )

        for prep in self.cfg.expt.preps:
            if prep == "g":
                continue
            sat = data[f"p_sat_{prep}"]
            data[f"p_exc_sat_{prep}"] = sat - floor
            print(
                f"  prepared |{prep}>: steady-state P(not g) = {sat:.4f}, "
                f"i.e. {sat - floor:+.4f} above the floor"
            )

        # Rate from the |e> trace. Subtract the floor first so the exponential
        # decays toward ~0 and its offset is the genuine residual.
        y = np.asarray(data["p_e_e"], dtype=float) - floor
        pars, _, _ = fitter.fitexp(t, y)
        data["reset_fit"] = np.asarray(pars, dtype=float)
        tau = float(pars[2])
        if np.isfinite(tau) and tau > 0:
            gamma_meas = 1 / (2 * np.pi * tau)
            data["reset_tau"] = tau
            data["reset_gamma"] = gamma_meas
            print(
                f"  |e> trace: tau = {tau:.3f} us -> Gamma/2pi = "
                f"{gamma_meas:.4f} MHz"
            )
            try:
                kappa = float(self.cfg.device.f0g1.kappa)
                if not np.isfinite(kappa):
                    raise ValueError
            except Exception:
                kappa = float(
                    self.cfg.device.readout.kappa[self.cfg.expt.qubit[0]]
                )
            g_tilde = g_tilde_for_gain(self.cfg, self.cfg.expt.gain_f0g1)
            omega_ef = self.cfg.expt.omega_ef_target
            gamma_pred = reset_rate(g_tilde, omega_ef, kappa)
            data["reset_gamma_pred"] = gamma_pred
            msg = (
                f"  predicted from Eq. 1 at g~ = {g_tilde:.4f}, Omega_ef = "
                f"{omega_ef:.4f}, kappa = {kappa:.4f}: Gamma/2pi = "
                f"{gamma_pred:.4f} MHz"
            )
            if gamma_pred > 0:
                msg += f" (measured/predicted = {gamma_meas / gamma_pred:.2f})"
            print(msg)
            t1 = float(self.cfg.device.qubit.T1[self.cfg.expt.qubit[0]])
            print(
                f"  for scale, T1 = {t1:.1f} us, i.e. passive Gamma/2pi = "
                f"{1 / (2 * np.pi * t1):.4f} MHz -- the reset is "
                f"{t1 / tau:.1f}x faster than doing nothing"
            )
        else:
            print("  |e> trace: exponential fit failed, no rate reported")

        # Trajectory fit, alongside the exponential rather than instead of it:
        # scripts 36/37 key off reset_gamma. CLAUDE.md "Fitting the ripples".
        traj = fit_reset_trajectory(
            t, np.asarray(data["p_e_e"], dtype=float),
            g_tilde_for_gain(self.cfg, self.cfg.expt.gain_f0g1),
            self.cfg.expt.omega_ef_target,
            float(self.cfg.device.f0g1.kappa),
            reps=self.cfg.expt.reps,
        )
        for k, v in traj.items():
            data[f"traj_{k}"] = v
        if traj["ok"]:
            gated = "" if traj["ripples"] else "   NOT TRUSTED"
            print(
                f"  trajectory fit: g~ = {traj['g_tilde']:.4f} "
                f"+- {traj['g_tilde_err']:.4f}, Omega_ef = "
                f"{traj['omega_ef']:.4f} +- {traj['omega_ef_err']:.4f}, "
                f"kappa = {traj['kappa']:.4f} +- {traj['kappa_err']:.4f}, "
                f"rms = {traj['rms']:.4f}{gated}"
            )
            if not traj["ripples"]:
                print(
                    f"    the trace does not ring (rebound {traj['rebound']:.4f}"
                    " is within 3 sigma of shot noise), so g~, Omega_ef and "
                    "kappa trade off freely -- raise gain_ef until it does"
                )
        else:
            print("  trajectory fit: did not converge")

        return data

    def display(self, data=None, fit=True, ax=None, **kwargs):
        """
        Pexc vs reset time, linear and log, in the style of the paper's Fig. 4c.

        The log panel is where the residual is readable; the floor from the |g>
        preparation is drawn on both, since nothing below it is measurable.
        """
        if data is None:
            data = self.data
        if "p_e_e" not in data:
            return

        t = np.asarray(data["xpts"], dtype=float)
        fig, axes = plt.subplots(2, 1, figsize=(6.5, 7), sharex=True)
        colors = {"g": "C0", "e": "C3", "f": "C2"}

        for panel, logy in zip(axes, (False, True)):
            for prep in self.cfg.expt.preps:
                panel.plot(
                    t, data[f"p_e_{prep}"], "o-", ms=3, color=colors.get(prep),
                    label=self._PREP_NAMES.get(prep, prep),
                )
            for prep in self.cfg.expt.baseline_preps:
                key = f"p_e_{prep}_off"
                if key in data:
                    panel.plot(
                        t, data[key], "s--", ms=3, lw=1.2, alpha=0.75,
                        color=colors.get(prep),
                        label=f"|{prep}>, drives off (baseline)",
                    )
            if "p_sat_g" in data:
                panel.axhline(
                    data["p_sat_g"], ls=":", color="k",
                    label=f"floor {data['p_sat_g']:.3f}",
                )
            if logy:
                panel.set_yscale("log")
            panel.set_ylabel("P(not g) = $P_e + P_f$")

        if "reset_fit" in data and np.all(np.isfinite(data["reset_fit"])):
            fitted = fitter.expfunc(t, *data["reset_fit"]) + data["p_sat_g"]
            label = "exp fit"
            if "reset_gamma" in data:
                label = f"exp fit, $\\Gamma/2\\pi$ = {data['reset_gamma']:.4f} MHz"
            for panel in axes:
                panel.plot(t, fitted, "-", color="k", lw=1, label=label)

        if data.get("traj_ok"):
            # Drawn on a fine grid: under strong drive the ripples are the
            # point, and the acquired grid undersamples them visually.
            tf = np.linspace(t[0], t[-1], 800)
            curve = data["traj_floor"] + data["traj_amp"] * reset_trajectory(
                tf, data["traj_g_tilde"], data["traj_omega_ef"],
                data["traj_kappa"]
            )
            lbl = (f"Eq. 1: $\\tilde{{g}}$ = {data['traj_g_tilde']:.3f}, "
                   f"$\\Omega_{{ef}}$ = {data['traj_omega_ef']:.3f}, "
                   f"$\\kappa$ = {data['traj_kappa']:.3f}")
            if not data.get("traj_ripples"):
                lbl += " (no ripples: not trusted)"
            for panel in axes:
                # Violet: the preparations already use C0/C3/C2 and the
                # exponential is black, and the trajectory lies right on top of
                # the |e> trace by construction, so it needs a distinct hue.
                panel.plot(tf, curve, "-", color="#7048E8", lw=1.4,
                           alpha=0.55 if not data.get("traj_ripples") else 1.0,
                           label=lbl, zorder=4)

        axes[0].legend(fontsize=8)
        axes[1].set_xlabel("Reset Time $t_r$ ($\\mu$s)")

        self.save_fig(fig)
        # The base display() pairs save_fig with save_config; this one
        # builds its own figure and so has to say it explicitly. Without
        # it a run leaves no record of the readout threshold its
        # P(not g) was computed against -- which is what made #397's
        # 4.6% floor impossible to line up against the FWM runs on
        # either side of it, since 12_single_shot had been re-run in
        # between and moved the threshold.
        self.save_config()
