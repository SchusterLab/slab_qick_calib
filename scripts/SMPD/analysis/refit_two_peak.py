"""
Photon-number-splitting fit of this run's Stark data, after analysis.ipynb.

At weak probe this per-row fit (free amplitude per row) is underdetermined; the
global shared-area fit described in CLAUDE.md "Probe power and the PNS comb"
agrees with the Ramsey and line-shift calibrations on run 34 (~320 per gain^2).

The spectrum at each buffer gain is modelled as the full PNS comb from the
notebook: a Poisson-weighted sum of Lorentzians, one per photon number,

    P(e)(f) = y0 + amp * sum_n Poisson(n; n_bar) * L(f; center_n, gamma_n)
    center_n = f0 - (2 n chi + chi_prime n^2)
    gamma_n  = kappa/2 + n delta_gamma

which replaces the two independent peak amplitudes of the experiment's
`two_peak` analysis with a single photon number. That is the point: every peak
height is then tied to one physical quantity per gain instead of being free, so
what comes out is n_bar(gain) -- photons in the buffer -- rather than a pair of
amplitudes.

The comb geometry -- `f0`, `chi`, `chi_prime` and `delta_gamma` -- is *shared
across all buffer gains*. Those describe the qubit and its coupling, not the
drive, so fitting them once uses all ten rows to pin them down.

`kappa` is fit per row, as the notebook does. It has to be: the comb broadens
peaks only with photon *number*, through `n delta_gamma`, but the n=0 peak
itself broadens with the buffer *occupation* -- measurement-induced dephasing
goes as n_bar. Sharing it splits the difference and visibly under-fits the
sharp low-gain peak. So `n_bar`, `kappa`, the readout amplitude and the
baseline vary per row.

Differences from the notebook, both deliberate:

- The notebook normalises data and model to their maxima. Here `amp` and `y0`
  are fit instead, because P(e) has a meaningful scale and zero, and because
  normalising by `max()` inside the model puts a kink in the residual wherever
  the tallest peak changes from n=0 to n=1.
- The notebook fits each row independently and plots all six parameters against
  buffer gain. With the comb shared only `n_bar` and `kappa` vary, so those are
  what get plotted, and `n_bar` is additionally fit against buffer gain as
  `n_bar = c0 + c1 * gain + c2 * gain^2`. Those coefficients are written to the
  config as `device.fwm.buffer_power_fit`, tagged `buffer_power_fit_type`, the
  same convention `pump_power_fit` uses for the FWM band. Note `c2` is *not*
  `device.readout.nstark`: that one assumes photons go purely as gain^2, and
  this scan needs all three terms.

A per-row two-Lorentzian fit with the experiment's own fitter runs first, to
seed this one and to confirm the script reproduces what the experiment stored.

    python refit_two_peak.py
    python refit_two_peak.py --n-max 8 --no-show
    python refit_two_peak.py --gain-max 0.03      # calibrate the low-gain end
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.stats import poisson

HERE = Path(__file__).resolve().parent
# .../scripts/SMPD/Data/run<N>_<device>/metadata_and_images/#351_... -> repo root
REPO_DIR = HERE.parents[5]
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from slab_qick_calib.exp_handling.datamanagement import SlabFile
from slab_qick_calib.experiments.single_qubit.stark_spectroscopy import (
    fit_two_lorentzians,
)
from slab_qick_calib.helpers import config

DATA_DIR = HERE.parents[1] / "data" / HERE.name


def _config_for_run(run_dir_name):
    """
    The ``configs/`` entry matching a ``Data/run<N>_<Device>/`` folder name.

    Derived rather than hardcoded. With two SMPDs wired to one box a baked-in
    filename is actively dangerous: this script would read -- and in
    ``refit_two_peak`` *write* -- the other device's YAML. The run folder this
    copy was dropped into names both the run and the device, which is exactly
    what the config filename carries, so read it back off that.

    Returns None if nothing matches, in which case ``--config`` is required.
    """
    m = re.fullmatch(r"run([^_]+)_(.+)", run_dir_name)
    if m is None:
        return None
    run, device = m.group(1), m.group(2).lower()
    matches = sorted((REPO_DIR / "configs").glob(f"*bfg{run}_*_{device}_rfboard.yml"))
    return matches[-1].name if matches else None


DEFAULT_CONFIG = _config_for_run(HERE.parents[1].name)

# Candidate forms for n_bar(buffer gain), keyed by the tag stored in the config
# alongside the coefficients -- exactly as pump_power_fit_type does for the band
# fit. Each tag names its terms in ascending power order and the coefficient
# list follows that order, so a reader elsewhere can interpret
# device.fwm.buffer_power_fit from the tag alone.
GAIN_FIT_POWERS = {
    "quadratic": (2,),
    "linear_quadratic": (1, 2),
    "constant_linear_quadratic": (0, 1, 2),
    "constant_quadratic": (0, 2),
}

# Which form to fit. Override with --gain-fit.
BUFFER_FIT_TYPE = "constant_quadratic"

# A row whose residual rms exceeds this multiple of the median row rms is
# dropped from the n_bar(gain) fit. Override with --rms-factor.
RMS_FACTOR = 3.0

# Which line property carries the Poisson weight; override with --normalization.
# "area" is right for a weak (unsaturated) qubit probe.
NORMALIZATION = "area"

# Fit n_bar(gain) only up to this buffer gain; None uses the whole scan.
# Rows above it are still fit, plotted and printed -- they just do not pull on
# the coefficients, so the curve extrapolating through them is a check on the
# form rather than something the form was fit to. Override with --gain-max.
FIT_GAIN_MAX = 1

# Parameters common to every buffer gain, and those free at each one.
SHARED_NAMES = ("f0", "chi", "chi_prime", "delta_gamma")
ROW_NAMES = ("n_bar", "amp", "y0", "kappa")
STORED_NAMES = ("peak_y0", "peak_amp0", "peak_amp1",
                "peak_freq", "peak_delta", "peak_gamma")


def pns_spectrum(f, f0, n_bar, chi, kappa, chi_prime, delta_gamma, n_max=20,
                 normalization="area"):
    """
    Poisson-weighted comb of Lorentzians, one per photon number.

    Same model as `calculate_pns_spectrum` in analysis.ipynb, minus the
    normalisation by `max()` -- the caller supplies an amplitude instead.
    `normalization` says which property of each line carries its Poisson
    weight: "area" (weak probe, the n-th line's height scaled by
    gamma_0/gamma_n) or "height" (saturated probe, unit-height lines).
    CLAUDE.md "Probe power and the PNS comb".
    """
    df = np.asarray(f, dtype=float) - f0
    n = np.arange(n_max)
    weights = poisson.pmf(n, n_bar)
    centers = -(2 * n * chi + chi_prime * n ** 2)
    gammas = kappa / 2.0 + n * delta_gamma
    if normalization == "area":
        weights = weights * gammas[0] / gammas
    elif normalization != "height":
        raise ValueError(f"normalization must be 'area' or 'height', got {normalization!r}")

    total = np.zeros_like(df)
    for weight, center, gamma in zip(weights, centers, gammas):
        total += weight * gamma ** 2 / ((df - center) ** 2 + gamma ** 2)
    return total


def load(data_dir):
    """Read the single .h5 in this run, with its config."""
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise SystemExit(f"No .h5 files in {data_dir}")
    if len(files) > 1:
        raise SystemExit(f"Expected one file, found {[f.name for f in files]}")
    path = files[0]
    with SlabFile(str(path), "r") as h5:
        data = {key: h5[key][()] for key in h5.keys()}
        cfg = h5.load_config()
    return path, data, cfg


def refit_per_row(xpts, p_e, chi_buffer):
    """
    Fit each buffer-gain row with two independent Lorentzians.

    This is the experiment's `two_peak` analysis verbatim, kept because it
    seeds the comb fit well: the n=1/n=0 amplitude ratio is a Poisson estimate
    of n_bar, and the separation is an estimate of 2*chi. Rows whose fit fails
    come back as NaN rather than being dropped, so the arrays stay aligned with
    the gain axis.
    """
    out = {name: np.full(len(p_e), np.nan) for name in STORED_NAMES}
    for row in range(len(p_e)):
        popt, _ = fit_two_lorentzians(xpts, p_e[row], chi_buffer)
        if popt is None:
            continue
        for name, value in zip(STORED_NAMES, popt):
            out[name][row] = value
    return out


def _seed(xpts, p_e, per_row, chi_buffer):
    """
    Starting point for the comb fit, taken from the two-peak fit where possible.

    For a Poisson distribution P(1)/P(0) = n_bar, so the amplitude ratio of the
    two fitted peaks seeds the photon number directly; dividing the n=0
    amplitude by P(0) = exp(-n_bar) then seeds the overall scale.
    """
    freq = np.nanmean(per_row["peak_freq"])
    delta = np.nanmean(per_row["peak_delta"])
    gamma = np.nanmin(per_row["peak_gamma"])
    if not np.isfinite(freq):
        # Seed at the largest excursion from baseline (works for dips too).
        dev0 = p_e[0] - float(np.median(p_e[0]))
        freq = float(xpts[np.argmax(np.abs(dev0))])
    if not np.isfinite(delta):
        delta = chi_buffer
    if not np.isfinite(gamma):
        gamma = 0.05

    # Notebook convention: peaks are spaced by 2*chi below f0.
    shared = [freq, abs(delta) / 2.0, 0.0, 0.05]

    rows = []
    for row, trace in enumerate(p_e):
        amp0 = per_row["peak_amp0"][row]
        amp1 = per_row["peak_amp1"][row]
        y0 = per_row["peak_y0"][row]
        kappa = per_row["peak_gamma"][row]
        kappa = 2.0 * kappa if np.isfinite(kappa) else 2.0 * gamma
        if not np.isfinite(amp0) or abs(amp0) < 1e-6:
            # Fallback: seed amp from the signed largest excursion so this
            # branch also works when peaks are inverted (amp0 < 0).
            y0 = float(np.median(trace))
            dev = trace - y0
            amp_seed = float(dev[np.argmax(np.abs(dev))])
            rows.append([0.1, amp_seed, y0, kappa])
            continue
        n_bar = float(np.clip(amp1 / amp0, 1e-3, 5.0))
        rows.append([n_bar, float(amp0 / np.exp(-n_bar)), float(y0), kappa])

    return np.asarray(shared), np.asarray(rows, dtype=float)


def fit_pns(xpts, p_e, per_row, chi_buffer, n_max, normalization="area"):
    """
    Fit every row at once: comb geometry shared, n_bar/amp/y0/kappa per row.

    Putting all rows into one residual is what shares the comb -- f0, chi,
    chi_prime and delta_gamma have to suit the whole map at once.
    """
    rows = len(p_e)
    shared_seed, row_seed = _seed(xpts, p_e, per_row, chi_buffer)

    n_shared = len(SHARED_NAMES)

    def unpack(theta):
        return theta[:n_shared], theta[n_shared:].reshape(rows, len(ROW_NAMES))

    def residuals(theta):
        (f0, chi, chi_prime, delta_gamma), per = unpack(theta)
        return np.concatenate([
            y0 + amp * pns_spectrum(xpts, f0, n_bar, chi, kappa, chi_prime,
                                    delta_gamma, n_max=n_max,
                                    normalization=normalization) - p_e[row]
            for row, (n_bar, amp, y0, kappa) in enumerate(per)
        ])

    # chi, delta_gamma, n_bar, and kappa are positive; f0 and chi_prime are
    # free. amp is also free: negative for inverted spectra (qubit starts in
    # |e>, so spectroscopy creates dips rather than peaks in P(e)).
    lower = np.concatenate([[-np.inf, 1e-6, -np.inf, 0.0],
                            np.tile([0.0, -np.inf, -np.inf, 1e-6], rows)])
    upper = np.full(n_shared + len(ROW_NAMES) * rows, np.inf)

    result = least_squares(
        residuals,
        x0=np.concatenate([shared_seed, row_seed.ravel()]),
        bounds=(lower, upper),
        x_scale="jac",
        xtol=1e-12,
        ftol=1e-12,
    )

    shared, per = unpack(result.x)
    fit = dict(zip(SHARED_NAMES, (float(v) for v in shared)))
    for index, name in enumerate(ROW_NAMES):
        fit[name] = per[:, index]

    # One sigma on each shared parameter, from the linearised covariance.
    dof = max(result.fun.size - result.x.size, 1)
    scale = float(result.fun @ result.fun) / dof
    try:
        cov = np.linalg.inv(result.jac.T @ result.jac) * scale
        errors = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        errors = np.full(result.x.size, np.nan)
    fit["shared_err"] = dict(zip(SHARED_NAMES, errors[:n_shared]))
    row_err = errors[n_shared:].reshape(rows, len(ROW_NAMES))
    fit["n_bar_err"] = row_err[:, 0]
    fit["kappa_err"] = row_err[:, 3]

    fit["rms"] = np.sqrt(np.mean(result.fun.reshape(rows, -1) ** 2, axis=1))
    fit["result"] = result
    fit["normalization"] = normalization
    return fit


def flag_bad_rows(fit, n_max, rms_factor=RMS_FACTOR):
    """
    Mark rows whose comb fit cannot be believed, to keep them out of n_bar(gain).

    Apart from the shared comb the row fits are independent, and nothing stops
    one of them running away: `least_squares` is free to push `n_bar` past the
    `n_max` states the comb actually sums over, at which point every Poisson
    weight in the model is ~0, the spectrum it predicts is flat, and `amp`
    grows without bound to compensate. Such a row barely shows in the total
    residual but reports an n_bar wrong by orders of magnitude -- and since
    the gain fit is a plain least squares, one of them is enough to wreck it.

    Two tests, both on quantities the fit already reports:

    - any of `n_bar`, `kappa`, `amp` non-finite, or `n_bar >= n_max`: the row
      has left the range the model is defined over.
    - row rms above `rms_factor` times the median row rms: whatever the
      reason, the fit does not describe this trace.

    Returns:
        (good, reasons) -- a boolean mask over rows, and a per-row string that
        is empty wherever the row was kept.
    """
    n_bar, kappa, amp, rms = (np.asarray(fit[name], dtype=float)
                              for name in ("n_bar", "kappa", "amp", "rms"))
    finite = np.isfinite(n_bar) & np.isfinite(kappa) & np.isfinite(amp)
    in_range = finite & (n_bar < n_max)
    # Median over the in-range rows only: a runaway row's rms would otherwise
    # help set the very threshold meant to catch it.
    reference = float(np.median(rms[in_range])) if in_range.any() else np.nan
    good = in_range & (rms <= rms_factor * reference)

    reasons = []
    for row in range(len(n_bar)):
        if good[row]:
            reasons.append("")
        elif not finite[row]:
            reasons.append("fit not finite")
        elif n_bar[row] >= n_max:
            reasons.append(f"n_bar {n_bar[row]:.3g} >= n_max {n_max}")
        else:
            reasons.append(f"rms {rms[row] / reference:.1f}x median")
    return good, reasons


def fit_photons_per_gain(gains, n_bar, fit_type):
    """
    Fit n_bar against buffer gain in whichever form `fit_type` names.

    `quadratic` is photons proportional to drive power and nothing else.
    `linear_quadratic` adds a linear term, which this scan wants -- pure gain^2
    undershoots the middle of the range and only catches up at the top.
    `constant_linear_quadratic` also allows residual occupation at zero drive
    (thermal population, or leakage down the line), though with the n_bar = 0
    row pinned at its bound there is little here to constrain it.

    Every form is linear in its coefficients, so this is a least squares --
    bounded rather than plain, because the constant term is held at or above
    zero. It is the occupation at zero buffer drive: negative is unphysical,
    and unbounded the fit will take a small negative value to buy curvature
    elsewhere, which then makes 1 - P(0) negative at the bottom of the range.

    Returns:
        list of coefficients in ascending power order
    """
    powers = GAIN_FIT_POWERS[fit_type]
    design = np.column_stack([gains ** power for power in powers])
    lower = [0.0 if power == 0 else -np.inf for power in powers]
    result = lsq_linear(design, n_bar, bounds=(lower, np.inf))
    return [float(c) for c in result.x]


def photons_for_gain(gain, coeffs, fit_type):
    """Mean buffer photon number at a given buffer gain."""
    gain = np.asarray(gain, dtype=float)
    return sum(c * gain ** power
               for c, power in zip(coeffs, GAIN_FIT_POWERS[fit_type]))


def gain_fit_label(coeffs, fit_type, mathtext=False):
    """The fitted n_bar(gain) as an expression, for printing or a plot label."""
    squared = "gain$^2$" if mathtext else "gain^2"
    parts = []
    for c, power in zip(coeffs, GAIN_FIT_POWERS[fit_type]):
        if power == 0:
            parts.append(f"{c:.4f}")
        elif power == 1:
            parts.append(f"{c:.2f} gain")
        else:
            parts.append(f"{c:.2f} {squared}")
    return " + ".join(parts)


class _Tee:
    """Write to a stream and a file at once."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "w", encoding="utf-8")

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--chi-buffer", type=float,
                        help="override the expected peak separation (MHz)")
    parser.add_argument("--n-max", type=int, default=20,
                        help="photon states in the comb (notebook uses 20)")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="device config to write the gain fit into")
    parser.add_argument("--gain-fit", default=BUFFER_FIT_TYPE,
                        choices=sorted(GAIN_FIT_POWERS),
                        help="form to fit n_bar(buffer gain) with")
    parser.add_argument("--gain-max", type=float, default=FIT_GAIN_MAX,
                        help="fit n_bar(gain) only over buffer gains up to "
                             "this value (default: the whole scan)")
    parser.add_argument("--rms-factor", type=float, default=RMS_FACTOR,
                        help="drop a row from the n_bar(gain) fit when its "
                             "residual rms exceeds this multiple of the "
                             "median row rms (inf keeps every row)")
    parser.add_argument("--normalization", default=NORMALIZATION,
                        choices=("area", "height"),
                        help="line property carrying the Poisson weight: "
                             "area (weak probe) or height (saturated)")
    parser.add_argument("--no-update-config", action="store_true",
                        help="fit and plot without touching the config")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.config is None:
        parser.error(
            f"could not find a configs/*_rfboard.yml matching run folder "
            f"{HERE.parents[1].name!r}; pass --config explicitly")

    path, data, cfg = load(Path(args.data_dir))
    expt = cfg["expt"]
    # Keep the printout next to the figures
    sys.stdout = _Tee(sys.stdout, HERE / f"{path.stem}_refit.txt")
    print(f"Reading {path.name}")
    soc = cfg["hw"]["soc"]
    applied = soc["rfboard_active"]["dac_readout"][str(soc["dacs"]["readout"]["ch"][0])]
    attn_db = float(applied["attn1"]) + float(applied["attn2"])
    pulse = expt["stark_pulse_type"]
    print(f"  buffer {attn_db:g} dB, {pulse}; qubit probe gain {expt['gain']}")

    xpts = np.asarray(data["xpts"])
    gains = np.asarray(data["stark_gain"])
    p_e = np.asarray(data["p_e"])
    chi_buffer = (args.chi_buffer if args.chi_buffer is not None
                  else float(expt["chi_buffer"]))
    print(f"  {p_e.shape[0]} buffer gains x {p_e.shape[1]} frequencies, "
          f"{xpts.min():.3f}-{xpts.max():.3f} MHz")
    print(f"  fit_type={expt['fit_type']}, chi_buffer={chi_buffer}")

    # --- per-row two-peak fit: seed, and a check against the stored fit ---
    per_row = refit_per_row(xpts, p_e, chi_buffer)
    n_ok = int(np.isfinite(per_row["peak_amp0"]).sum())
    worst = max(
        float(np.max(np.abs(np.asarray(data[name], dtype=float) - per_row[name])))
        for name in STORED_NAMES if name in data
    )
    print(f"\nPer-row two-peak fit (the experiment's analysis): "
          f"{n_ok}/{len(gains)} rows, max |diff| vs stored {worst:.3e}")

    # --- the PNS comb fit -------------------------------------------------
    fit = fit_pns(xpts, p_e, per_row, chi_buffer, args.n_max,
                  normalization=args.normalization)
    err = fit["shared_err"]
    n_params = len(SHARED_NAMES) + len(ROW_NAMES) * len(gains)
    print(f"\nPNS comb fit: {n_params} parameters, {p_e.size} points, "
          f"n_max={args.n_max}, normalization={args.normalization}")
    print("  shared across all buffer gains:")
    for name in SHARED_NAMES:
        print(f"    {name:12s} {fit[name]:10.4f} +/- {err[name]:.4f} MHz")
    print(f"  n=0 -> n=1 spacing 2*chi = {2 * fit['chi']:.4f} MHz "
          f"(config chi_buffer = {abs(chi_buffer)} MHz)")
    print(f"  residual rms {np.sqrt(np.mean(fit['result'].fun ** 2)):.3e}")

    good, reasons = flag_bad_rows(fit, args.n_max, args.rms_factor)
    dropped = np.flatnonzero(~good)
    if dropped.size:
        print(f"\n  dropped from the n_bar(gain) fit: "
              f"{dropped.size}/{len(gains)} rows")
        for row in dropped:
            print(f"    gain {gains[row]:.4f}  {reasons[row]}")
    # The cutoff is a separate thing from the quality mask: those rows are
    # believed, they are just outside the range being calibrated.
    in_range = (good if args.gain_max is None
                else good & (gains <= args.gain_max))
    if args.gain_max is not None:
        print(f"  fitting gains <= {args.gain_max:g}: "
              f"{int(in_range.sum())} of {int(good.sum())} kept rows")

    n_terms = len(GAIN_FIT_POWERS[args.gain_fit])
    if int(in_range.sum()) < n_terms:
        raise SystemExit(f"Only {int(in_range.sum())} rows are usable, too "
                         f"few for a {n_terms}-term {args.gain_fit} fit")

    coeffs = fit_photons_per_gain(gains[in_range], fit["n_bar"][in_range],
                                  args.gain_fit)
    units = {0: "photons at zero buffer drive",
             1: "photons per unit gain",
             2: "photons per unit gain^2"}
    terms = args.gain_fit.split("_")
    print(f"\n  n_bar = {gain_fit_label(coeffs, args.gain_fit)} "
          f"({args.gain_fit})")
    for term, c, power in zip(terms, coeffs, GAIN_FIT_POWERS[args.gain_fit]):
        print(f"    {term:9s} {c:10.4f} {units[power]}")
    # The top of the *fitted* range, not of the scan: past a cutoff the model
    # is extrapolating and that is not the number to quote.
    top = float(gains[in_range].max())
    n_top = photons_for_gain(top, coeffs, args.gain_fit)
    print(f"    at gain {top:.3f}: n_bar {n_top:.4f}, "
          f"1-P(0) {1 - np.exp(-n_top):.4f}")

    # Stored in the config rather than quoted in other scripts, so that
    # revising this fit cannot leave a stale copy of the coefficients
    # somewhere else. #447's fit_shared_means.py reads it back to put its
    # histograms on a 1 - P(0) axis.
    if args.no_update_config:
        print("    (config not updated)")
    else:
        cfg_path = REPO_DIR / "configs" / args.config
        config.update_fwm(str(cfg_path), "buffer_power_fit", coeffs)
        config.update_fwm(str(cfg_path), "buffer_power_fit_type",
                          args.gain_fit)
        # The buffer setting it was taken at, so consumers can rescale it
        config.update_fwm(str(cfg_path), "buffer_power_fit_attn_db", attn_db)
        config.update_fwm(str(cfg_path), "buffer_power_fit_pulse", pulse)

    print("\n  gain     n_bar      err    kappa      err      amp       y0"
          "       rms")
    for row, gain in enumerate(gains):
        flag = "" if good[row] else f"   DROPPED ({reasons[row]})"
        print(f"  {gain:.4f}  {fit['n_bar'][row]:7.4f}  "
              f"{fit['n_bar_err'][row]:7.4f}  {fit['kappa'][row]:7.4f}  "
              f"{fit['kappa_err'][row]:7.4f}  {fit['amp'][row]:7.4f}  "
              f"{fit['y0'][row]:7.4f}  {fit['rms'][row]:.3e}{flag}")

    # --- figure 1: map, photon number, example cuts -----------------------
    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(4, 2, width_ratios=[1.15, 1], hspace=0.3,
                            wspace=0.28)
    ax_map = fig.add_subplot(grid[:, 0])
    ax_nbar = fig.add_subplot(grid[0, 1])
    ax_kappa = fig.add_subplot(grid[1, 1], sharex=ax_nbar)
    ax_amp = fig.add_subplot(grid[2, 1], sharex=ax_nbar)
    ax_cut = fig.add_subplot(grid[3, 1])
    ax_nbar.tick_params(labelbottom=False)
    ax_kappa.tick_params(labelbottom=False)

    mesh = ax_map.pcolormesh(xpts, gains, p_e, cmap="viridis", shading="auto",
                             rasterized=True)
    plt.colorbar(mesh, ax=ax_map, label="P(e)")
    # The comb is shared, so each photon number is a straight line.
    visible = 0
    for n in range(args.n_max):
        center = fit["f0"] - (2 * n * fit["chi"] + fit["chi_prime"] * n ** 2)
        if not xpts.min() <= center <= xpts.max():
            continue
        ax_map.axvline(center, color="red", lw=0.8, ls="--", alpha=0.8)
        ax_map.text(center, gains.max(), f" n={n}", color="red", va="bottom",
                    ha="center", fontsize=9)
        visible += 1
    ax_map.set_xlabel("Qubit Frequency (MHz)")
    ax_map.set_ylabel("Buffer Gain")

    dense_gain = np.linspace(0, gains.max(), 100)
    # Dropped rows are drawn as grey rules rather than left out silently: a
    # runaway row would otherwise set the y scale of all three panels, and a
    # gap in the points with no visible cause reads as missing data.
    for index, row in enumerate(dropped):
        ax_nbar.axvline(gains[row], color="0.85", lw=1, zorder=0,
                        label="dropped" if index == 0 else None)
        ax_kappa.axvline(gains[row], color="0.85", lw=1, zorder=0)
        ax_amp.axvline(gains[row], color="0.85", lw=1, zorder=0)

    # Above the cutoff: same points, hollow, and not fit to.
    above = good & ~in_range
    ax_nbar.errorbar(gains[in_range], fit["n_bar"][in_range],
                     yerr=fit["n_bar_err"][in_range], fmt="o", ms=4,
                     capsize=2, label="fit")
    if above.any():
        ax_nbar.errorbar(gains[above], fit["n_bar"][above],
                         yerr=fit["n_bar_err"][above], fmt="o", ms=4,
                         mfc="none", capsize=2, color="C0",
                         label="above cutoff")
    ax_nbar.plot(dense_gain, photons_for_gain(dense_gain, coeffs, args.gain_fit),
                 "-", lw=1, color="red",
                 label=gain_fit_label(coeffs, args.gain_fit, mathtext=True))
    ax_nbar.set_ylabel(r"$\bar{n}$ (photons)")
    if args.gain_max is not None:
        ax_nbar.axvline(args.gain_max, color="0.4", ls="--", lw=1,
                        label="fit cutoff")
    ax_nbar.legend(fontsize="small")

    # The n=0 linewidth against occupation: measurement-induced dephasing,
    # which the comb's n*delta_gamma broadening cannot express.
    ax_kappa.errorbar(gains[good], fit["kappa"][good],
                      yerr=fit["kappa_err"][good], fmt="o", ms=4, capsize=2)
    ax_kappa.set_ylabel(r"$\kappa$ (MHz)")

    # amp should be roughly flat: it is the readout contrast, and the drop in
    # the n=0 peak with gain ought to be the Poisson weight, not lost contrast.
    ax_amp.plot(gains[good], fit["amp"][good], "o-", ms=4, lw=1)
    ax_amp.set_xlabel("Buffer Gain")
    ax_amp.set_ylabel("Readout amplitude")

    picks = sorted({0, len(gains) // 2, len(gains) - 1})
    dense = np.linspace(xpts.min(), xpts.max(), 1500)
    for row in picks:
        line, = ax_cut.plot(xpts, p_e[row], "o", ms=3,
                            label=f"gain = {gains[row]:.4g}")
        ax_cut.plot(dense, fit["y0"][row] + fit["amp"][row] * pns_spectrum(
            dense, fit["f0"], fit["n_bar"][row], fit["chi"],
            fit["kappa"][row], fit["chi_prime"], fit["delta_gamma"],
            n_max=args.n_max, normalization=fit["normalization"],
        ), "-", lw=1.2, color=line.get_color())
        ax_nbar.axvline(gains[row], color=line.get_color(), ls=":", lw=1,
                        alpha=0.6)
    ax_cut.set_xlabel("Qubit Frequency (MHz)")
    ax_cut.set_ylabel("P(e)")
    ax_cut.legend(fontsize="small")

    fig.suptitle(
        f"{HERE.name}   PNS comb, shared 2*chi = {2 * fit['chi']:.4f} MHz, "
        f"{visible} peaks in span"
    )
    out = HERE / f"{path.stem}_pns_fit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")

    # --- figure 2: every row with its fit, as the notebook does ----------
    nrows = (len(gains) + 1) // 2
    fig2, axs = plt.subplots(nrows, 2, figsize=(11, 1.7 * nrows),
                             sharex=True, sharey=True)
    axs = np.atleast_2d(axs)
    for row, gain in enumerate(gains):
        axis = axs[row // 2, row % 2]
        axis.plot(xpts, p_e[row], color="k", lw=0.6)
        axis.plot(dense, fit["y0"][row] + fit["amp"][row] * pns_spectrum(
            dense, fit["f0"], fit["n_bar"][row], fit["chi"],
            fit["kappa"][row], fit["chi_prime"], fit["delta_gamma"],
            n_max=args.n_max, normalization=fit["normalization"],
        ), color="red", ls="--", lw=1.2)
        label = f"gain {gain:.4f},  $\\bar{{n}}$ = {fit['n_bar'][row]:.3g}"
        axis.text(0.02, 0.9, label if good[row] else f"{label}  (dropped)",
                  transform=axis.transAxes, fontsize=9, va="top",
                  color="k" if good[row] else "red")
    for axis in axs[-1]:
        axis.set_xlabel("Qubit Frequency (MHz)")
    for axis in axs[:, 0]:
        axis.set_ylabel("P(e)")
    # An odd number of gains leaves the last cell empty.
    if len(gains) % 2:
        axs[-1, -1].axis("off")
    out2 = HERE / f"{path.stem}_pns_rows.png"
    fig2.tight_layout()
    fig2.savefig(out2, dpi=150, bbox_inches="tight")

    # --- figure 3: the calibration #447 consumes, in three steps ----------
    # Kept as separate panels rather than one set of axes: n_bar and 1 - P(0)
    # are different quantities and overlaying them invites reading the gap
    # between them as data rather than as the Poisson conversion.
    fig3, (ax_n, ax_p, ax_conv) = plt.subplots(1, 3, figsize=(15, 4.5))
    n_dense = photons_for_gain(dense_gain, coeffs, args.gain_fit)

    # 1. what the comb fit measures.
    ax_n.errorbar(gains[in_range], fit["n_bar"][in_range],
                  yerr=fit["n_bar_err"][in_range], fmt="o", ms=5, capsize=2,
                  color="C0", label="comb fit")
    if above.any():
        ax_n.errorbar(gains[above], fit["n_bar"][above],
                      yerr=fit["n_bar_err"][above], fmt="o", ms=5, mfc="none",
                      capsize=2, color="C0", label="above cutoff")
    ax_n.plot(dense_gain, n_dense, "-", lw=1.5, color="red",
              label=gain_fit_label(coeffs, args.gain_fit, mathtext=True))
    if args.gain_max is not None:
        ax_n.axvline(args.gain_max, color="0.4", ls="--", lw=1,
                     label="fit cutoff")
        ax_p.axvline(args.gain_max, color="0.4", ls="--", lw=1)
    ax_n.set_xlabel("Buffer Gain")
    ax_n.set_ylabel(r"$\bar{n}$ (photons)")
    ax_n.set_xlim(0, gains.max())
    ax_n.set_ylim(0,1)
    ax_n.legend(fontsize="small", loc="upper left")

    # 2. the same thing after conversion -- the axis #447 plots against.
    # Errors carry through as d(1 - e^-n)/dn = e^-n.
    ax_p.errorbar(gains[in_range], 1.0 - np.exp(-fit["n_bar"][in_range]),
                  yerr=np.exp(-fit["n_bar"][in_range]) * fit["n_bar_err"][in_range],
                  fmt="o", ms=5, capsize=2, color="C0")
    if above.any():
        ax_p.errorbar(gains[above], 1.0 - np.exp(-fit["n_bar"][above]),
                      yerr=np.exp(-fit["n_bar"][above]) * fit["n_bar_err"][above],
                      fmt="o", ms=5, mfc="none", capsize=2, color="C0")
    ax_p.plot(dense_gain, 1.0 - np.exp(-n_dense), "-", lw=1.5, color="red")
    ax_p.set_xlabel("Buffer Gain")
    ax_p.set_ylabel(r"$1 - P(0)$")
    ax_p.set_xlim(0, gains.max())
    ax_p.set_ylim(0,1)

    # 3. the conversion alone: no gain axis and nothing fitted, just Poisson.
    # The gap from the dashed line is what the conversion actually does --
    # photons landing in shots that already had one, so they add no detection.
    n_axis = np.linspace(0, max(1.0, float(np.max(fit["n_bar"][good])) * 1.25),
                         200)
    ax_conv.plot(n_axis, 1.0 - np.exp(-n_axis), "-", lw=1.5, color="red",
                 label=r"$1 - P(0) = 1 - e^{-\bar{n}}$")
    ax_conv.plot(n_axis, n_axis, "--", lw=1, color="0.5",
                 label=r"$\bar{n}$ (small-$\bar{n}$ limit)")
    ax_conv.plot(fit["n_bar"][good], 1.0 - np.exp(-fit["n_bar"][good]), "o",
                 ms=5, color="C0", label="this scan")
    ax_conv.set_xlabel(r"$\bar{n}$ (photons)")
    ax_conv.set_ylabel(r"$1 - P(0)$")
    ax_conv.set_xlim(0, n_axis.max())
    ax_conv.set_ylim(0, 1)
    ax_conv.legend(fontsize="small", loc="lower right")
    out3 = HERE / f"{path.stem}_gain_to_occupation.png"
    fig3.tight_layout()
    fig3.savefig(out3, dpi=150, bbox_inches="tight")

    print(f"\nSaved {out.name}, {out2.name}, {out3.name}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
