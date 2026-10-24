"""
Fit every histogram in this run to two Gaussians with globally shared means.

Each measured distribution is modelled as

    hist(I) = (1 - p) * N(I; vg, sigma) + p * N(I; ve, sigma)

where `vg`, `ve` and `sigma` are common to *all* distributions in the run and
only the excited fraction `p` varies. Pinning the component positions once is
what makes the fractions comparable across the grid: a per-histogram fit can
trade position against weight and produce numbers that cannot be put side by
side.

The fit is done by variable projection. For any trial (vg, ve, sigma) the model
is linear in `p`, so the best fraction for each histogram has a closed form and
only the three shared parameters are searched numerically.

The fractions are then plotted against the number of photons that passed
through the buffer during one buffer pulse, `kappa**2*n_bar*length_b/(4*kappa_c)`,
rather than against raw buffer gain. That needs two things from the device
config: a gain -> n_bar fit and the buffer's output coupling
`device.readout.kappa_c`. The gain -> n_bar fit is, by default
(`--calibration ramsey_spec`), `device.fwm.buffer_photon_ramsey_spec` --
on-resonance photons per gain^2 from script 41's dephasing spectrum -- rescaled
to the buffer attenuation and pulse shape recorded in this run's files.
`--calibration ramsey` is built from `device.fwm.buffer_dephasing_ramsey` --
the most recent T2PhotonCal (script 38) run's d(1/T2)/d(gain^2) -- with the
config's buffer kappa and chi, the same ones the flux step uses, then rescaled
to the buffer attenuation and pulse shape recorded in this run's files.
`--calibration stark` uses `device.fwm.buffer_power_fit` from
`refit_two_peak.py` as stored, as before.

An earlier version used `1 - P(0) = 1 - exp(-n_bar)`, the chance the buffer is
occupied at an instant. That is the wrong denominator for a detector that
integrates for 40 us: it implied efficiencies above 1, because the buffer holds
0.7 photons while passing tens of them through the window. `1 - P(0)` is still
written to the CSV, as a statement about occupancy rather than about supply.

The run's sweep axis is read off the files: pump gain for scripts 23/26, buffer
frequency for script 37 (one pump power, buffer frequency x buffer gain),
buffer length for script 40. Efficiency, dark count and NEP are reduced per
value of whichever one varies. A length sweep also gets the efficiency and
efficiency x duty vs length summary; CLAUDE.md "Buffer-length sweep (script 40)".
The photon axis needs no change for a frequency sweep: the calibration turns
gain into the flux *arriving* down the line, which the drive sets regardless of
where it sits relative to the buffer, so a detuned point is still efficiency
per arriving photon.

    python fit_shared_means.py
    python fit_shared_means.py --bins 150 --no-show
    python fit_shared_means.py --pump-gain 0.6
    python fit_shared_means.py --frequency-b 6230.25   # script-37 runs
    python fit_shared_means.py --length-b 55           # script-40 runs
"""

from __future__ import annotations

import argparse
import re
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
# .../scripts/SMPD/Data/run<N>_<device>/metadata_and_images/#447_... -> repo root
REPO_DIR = HERE.parents[5]
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))

from slab_qick_calib.calib.readout_helpers import t1_decay_distribution
from slab_qick_calib.exp_handling.datamanagement import AttrDict, SlabFile
from slab_qick_calib.experiments.general.qick_experiment import (
    effective_final_delay,
)
from slab_qick_calib.experiments.single_qubit.t2 import ramsey_photons_per_rate
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

# device.readout index of the buffer line in the SMPD scripts, and of the
# waste line the qubit is actually read out on.
BUFFER_LINE = 0
READOUT_LINE = 1

# Which powers of buffer gain each buffer_power_fit_type tag stores, in the
# order the coefficients appear. Must match GAIN_FIT_POWERS in the #351
# folder's refit_two_peak.py, which writes them; `photons` checks the
# coefficient count against the tag so a mismatch cannot pass silently.
GAIN_FIT_POWERS = {
    "quadratic": (2,),
    "linear_quadratic": (1, 2),
    "constant_linear_quadratic": (0, 1, 2),
    "constant_quadratic": (0, 2)
}

# 'n' and 'b' are listed so runs taken before no-drives moved into the 'g'
# slot still load; CLAUDE.md "The detuned-pump scan".
SCANS = ("g", "e", "p", "d", "b", "n", "f", "w")
SCAN_LABELS = {
    "g": "no drives",
    "e": "buffer + pump",
    "p": "no buffer",
    "d": "no buffer, off band",
    "b": "no pump",
    "n": "no drives",
    "f": "buffer + pump, f readout",
    "w": "no drives, T1 wait",
}
SCAN_COLORS = {"g": "tab:blue", "e": "tab:red", "p": "green",
               "d": "darkorange", "b": "tab:cyan", "n": "0.35",
               "f": "magenta", "w": "tab:purple"}

# Panel order and y scale for the grid figure. The scans carrying real signal
# come first on a shared linear scale so their sizes compare directly; the two
# background scans follow on log axes, where the structure worth seeing is a
# few parts in a thousand and would be a flat line on a linear scale.
GRID_ORDER = (("p", "linear"), ("e", "linear"), ("f", "linear"),
              ("d", "log"), ("g", "log"), ("w", "log"), ("b", "log"),
              ("n", "log"))

# Pump gain for the per-histogram figure: the buffer+pump and f-readout
# distributions with their fits, one row per buffer gain. Override with
# --pump-gain.
EXAMPLE_PUMP_GAIN = 0.6
EXAMPLE_SCANS = ("e", "f")

# Efficiency is the linear-regime (initial) slope of excited fraction against
# arriving photons. The response bends over -- the qubit reports at most one
# excitation per shot -- so a straight line through the low points reads low
# (~20% at a 41 us window). Every sweep therefore fits the window dynamics
# (`window_population`) to all its buffer gains and reports eta from that; a
# length sweep fits T and tau too, a single-length run holds them. Only the two
# scans that carry conversion signal get one. CLAUDE.md "Buffer-length sweep (script 40)".
EFFICIENCY_SCANS = ("e", "f")

# Which scans get a photon-count x axis rather than raw buffer gain. Narrower
# than "played the buffer pulse", which would also include `g`: with the pump
# off nothing converts those arrivals, so counting them suggests a detection
# relationship that is not there. Only the two scans that actually detect get
# the photon axis; everything else shows the buffer gain that was set.
PHOTON_AXIS_SCANS = ("e", "f")

# The dark-count scan: pump on, buffer off -- the detector armed with no
# signal, so every click is a false positive. Not the pump-off scan, which is
# the detector switched off and cannot produce a dark count at all. Its panel
# reports P(e) averaged over the buffer-gain settings, which are repeats here
# since the buffer pulse was never played.
DARK_SCAN = "p"

# The same pump power detuned off the band, so the part of the dark count that
# does not need the FWM resonance. Overlaid on the dark-count panels; the gap
# to DARK_SCAN is the resonant contribution. Absent from sweeps that do not
# request it (37/40), where the overlay is simply skipped.
OFFBAND_SCAN = "d"

# The no-drives floor both of the above sit on. Plotted with them so the
# decomposition is visible rather than something to work out afterwards:
# FLOOR_SCAN is the readout floor, OFFBAND - FLOOR the non-resonant pump
# background, DARK - OFFBAND the resonant dark count. Not the same use of 'g'
# as NEP_SCAN, which is only a layout slot.
FLOOR_SCAN = "g"


def _thermal_label(pop, temp_mk):
    """Legend text for the thermal-P(e) line on the dark-count panels."""
    if temp_mk:
        return f"thermal {100 * pop:.2f}%  ({temp_mk:.0f} mK)"
    return f"thermal {100 * pop:.2f}%"

# Which column's second row holds the noise-equivalent photon number. This is
# a layout slot, not a data source: the NEP is the dark count divided by the
# efficiency, so it comes from the `p` and `e` scans, not from this one. It
# lives here only because this column's second row would otherwise be blank.
NEP_SCAN = "g"

# Also a layout slot, not a data source: the last otherwise-blank cell, which
# carries the NEP in W/sqrt(Hz). Kept separate from the equivalent-n_bar panel
# because the two are not proportional -- see `noise_equivalent_power`.
# Points at 'd' since no-drives moved into the 'g' slot. A four-scan run
# (37/40, no 'd') has no spare column and so loses this cell from the grid
# figure; fwm_summary.png still draws the NEP.
NEP_WATT_SCAN = "d"


def gaussian(x, mu, sigma):
    """Unit-area Gaussian, to match density=True histograms."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def load_calibration(cfg_path, source="ramsey_spec"):
    """
    Read what is needed to turn a buffer gain into a photon count.

    Two pieces: the gain -> n_bar fit from device.fwm (with its type tag, so
    `photons` can refuse coefficients in a form it does not know -- the same
    reason `pump_power_fit_type` exists), and the buffer's output coupling
    rate from device.readout.kappa_c, which converts the standing occupation
    into a flux.

    `source` picks the gain -> n_bar fit. "ramsey_spec" is
    device.fwm.buffer_photon_ramsey_spec, photons per gain^2 at f_g from script
    41 directly. "ramsey" is built from
    device.fwm.buffer_dephasing_ramsey, the latest T2PhotonCal run's dephasing
    slope, converted to on-resonance photons with device.readout kappa and chi
    -- so kappa comes from one place for the whole chain -- at the drive
    detuning it was taken at (_detuning). "stark" is
    device.fwm.buffer_power_fit from refit_two_peak.py. Each is stated for one
    buffer attenuation and pulse shape (its _attn_db / _pulse keys), returned
    as `reference` so the caller can rescale to the data's own settings; a
    Stark fit from before refit_two_peak.py wrote tags has reference None and
    is used as stored.
    """
    auto_cfg = config.load(str(cfg_path))
    fwm = auto_cfg.device.get("fwm", {})
    key = {"ramsey": "buffer_dephasing_ramsey", "stark": "buffer_power_fit",
           "ramsey_spec": "buffer_photon_ramsey_spec"}[source]
    if key not in fwm:
        raise SystemExit(
            f"No device.fwm.{key} in {cfg_path}.\n"
            + ("Run script 38 (T2PhotonCal) with config updates on; its "
               "update() writes it there."
               if source == "ramsey" else
               "Run script 41 (BufferDephasingSpec) with config updates on; its "
               "update() writes it there."
               if source == "ramsey_spec" else
               "Run refit_two_peak.py in the #351 Stark folder first -- it "
               "fits buffer gain -> photon number and writes it there.")
        )
    readout = auto_cfg.device.readout
    kappa = float(readout["kappa"][BUFFER_LINE])
    reference = None
    if f"{key}_attn_db" in fwm and f"{key}_pulse" in fwm:
        reference = (float(fwm[f"{key}_attn_db"]), str(fwm[f"{key}_pulse"]))
    if source == "ramsey":
        chi = float(readout["chi"][BUFFER_LINE])
        # Drive offset from f_g the slope was taken at; untagged runs drove at the midpoint
        detuning = float(fwm.get(f"{key}_detuning", chi / 2))
        _, on_resonance = ramsey_photons_per_rate(kappa, chi, detuning)
        coeffs, fit_type = [float(fwm[key][0]) * on_resonance], "quadratic"
    elif source == "ramsey_spec":
        coeffs, fit_type = [float(fwm[key][0])], "quadratic"
    else:
        coeffs, fit_type = list(fwm[key]), fwm[f"{key}_type"]
    kappa_c = float(readout.get("kappa_c", [float("nan")])[BUFFER_LINE])
    if not np.isfinite(kappa_c):
        raise SystemExit(
            f"No device.readout.kappa_c[{BUFFER_LINE}] in {cfg_path}.\n"
            "It comes from a resonator spectroscopy complex fit of the buffer "
            "(complex_fit_kappa_c); run #201's was 0.2063 MHz."
        )
    if kappa_c > kappa:
        raise SystemExit(
            f"device.readout.kappa_c[{BUFFER_LINE}] = {kappa_c} exceeds "
            f"kappa = {kappa}; a coupling rate cannot beat the total "
            "linewidth, so one of the two is from a stale fit."
        )
    return (coeffs, fit_type, reference,
            kappa, kappa_c, list(auto_cfg.device.qubit["T1"]),
            list(auto_cfg.device.qubit.get("T1_ef", [])),
            float(readout["readout_length"][READOUT_LINE]),
            # Thermal P(e) and the temperature it was inferred from, written by
            # 16_temperature.py. Drawn on the dark-count panels as the floor
            # the readout cannot beat.
            list(auto_cfg.device.qubit.get("pop", [])),
            list(auto_cfg.device.qubit.get("temp", [])))


# Photons at a const buffer pulse relative to flat_top at the same gain: QICK
# plays a flat_top plateau at half the const amplitude (flat_scale in asm_v2).
PULSE_PHOTON_FACTOR = {"flat_top": 1.0, "const": 4.0}


def calibration_scale(records, reference):
    """
    Factor taking the calibration's reference settings to this run's.

    Photons at a given gain go as 10**(-attn/10) with the buffer attenuation
    actually applied, and 4x for const over flat_top. Every file in the run
    must share one setting, since the factor is folded into the coefficients
    once.
    """
    settings = {(r["buffer_attn_db"], r["pulse_type_b"]) for r in records}
    if len(settings) != 1:
        raise SystemExit(f"Files in this run differ in buffer attenuation / "
                         f"pulse shape: {sorted(settings)}")
    (attn, pulse), = settings
    if pulse not in PULSE_PHOTON_FACTOR:
        raise SystemExit(f"Buffer pulse shape {pulse!r} not recorded or unknown; "
                         "cannot rescale the photon calibration to it.")
    ref_attn, ref_pulse = reference
    scale = (10 ** ((ref_attn - attn) / 10)
             * PULSE_PHOTON_FACTOR[pulse] / PULSE_PHOTON_FACTOR[ref_pulse])
    return scale, attn, pulse


def photons(gain, coeffs, fit_type):
    """
    Mean photon number in the buffer at a given buffer gain.

    Which powers of gain the stored coefficients belong to comes from the type
    tag, so refit_two_peak.py can change the form of its fit without this
    script silently misreading the numbers.
    """
    if fit_type not in GAIN_FIT_POWERS:
        raise ValueError(
            f"Unknown device.fwm.buffer_power_fit_type {fit_type!r}; "
            f"known forms are {sorted(GAIN_FIT_POWERS)}."
        )
    powers = GAIN_FIT_POWERS[fit_type]
    if len(coeffs) != len(powers):
        raise ValueError(
            f"device.fwm.buffer_power_fit has {len(coeffs)} coefficients but "
            f"buffer_power_fit_type {fit_type!r} needs {len(powers)}. Re-run "
            "refit_two_peak.py in the #351 Stark folder to rewrite both."
        )
    gain = np.asarray(gain, dtype=float)
    return sum(c * gain ** power for c, power in zip(coeffs, powers))


def one_minus_p0(gain, coeffs, fit_type):
    """
    Probability the buffer holds at least one photon at any instant.

    For a coherent state of mean n_bar the vacuum probability is
    P(0) = exp(-n_bar), so this is 1 - exp(-n_bar). Kept in the CSV because it
    is a meaningful statement about buffer occupancy, but it is *not* the right
    axis for a detection efficiency -- see `photons_incident`.
    """
    return 1.0 - np.exp(-photons(gain, coeffs, fit_type))


def photons_incident(gain, length_b, coeffs, fit_type, kappa, kappa_c):
    """
    Photons arriving down the buffer line during one buffer pulse.

    The Stark measurement gives `n_bar`, the *standing* occupation, reached
    within 2/kappa = 1.4 us and then held. Over a 40 us pulse the buffer is not
    holding a photon, it is passing a stream, and the detector has the whole
    window to convert any of it -- so the count that matters is a flux times a
    time, not an occupation.

    The flux is the *arrival* rate. `kappa_c * n_bar` is a tempting answer but
    wrong: that is the rate the cavity field radiates back into the line, which
    interferes with the prompt reflection and is ~3x the arrival rate here. In
    steady state arrivals must balance departures, so the arrival rate follows
    from inverting the steady-state occupation. The buffer is measured and
    driven in reflection -- a single port, drive coupling in as sqrt(kappa_c) --
    which gives

        n_bar = 4 * kappa_c * Phi_in / kappa**2
        N     = Phi_in * length_b = kappa**2 * n_bar * length_b / (4 kappa_c)

    with kappa and kappa_c as rates in rad/us, i.e. 2*pi times the MHz
    linewidths the config stores. Dropping that 2*pi moves the answer by 6.3x.

    No further geometry correction belongs here: the resonator fit uses Gao's
    notch S21, but `complex_fit_kappa_c` is already f0/(2*Qc), and that factor
    of two *is* the notch-to-reflection conversion. Applying it twice would
    halve the arrival count.

    One loose end: `kappa` and `kappa_c` in the config come from different
    fits (0.2294 vs #201's 0.2205 MHz), and kappa enters squared, so that is
    ~8% on N.
    """
    n_bar = photons(gain, coeffs, fit_type)
    rate, rate_c = 2.0 * np.pi * kappa, 2.0 * np.pi * kappa_c
    flux = rate ** 2 * n_bar / (4.0 * rate_c)
    return flux * np.asarray(length_b, dtype=float)


PLANCK = 6.62607015e-34  # J s


def noise_equivalent_power(p_dark, eta, frequency_b, length_b, cycle):
    """
    NEP in W/sqrt(Hz), including the duty cycle eta_D of the pulse sequence.

        NEP = (h nu / (eta * eta_D)) * sqrt(2 Gamma_dark)

    with `eta_D = length_b / cycle` -- photons arriving while the detector is
    not armed are simply missed -- and `Gamma = p_dark / cycle` the observed
    wall-clock dark rate, so

        NEP ~ sqrt(Gamma) / (eta * eta_D)

    Careful with which rate Gamma means. Against the *armed-window* rate
    `p_dark/length_b` the same expression carries sqrt(eta_D) instead, because
    that rate does not already include the dead time. Either way the answer is
    the same; at 14% duty it costs 2.7x relative to continuous operation,
    since the dark rate falls with duty as well as the efficiency.

    Note this is *not* proportional to the noise-equivalent n_bar, which goes
    as p_dark/eta rather than sqrt(p_dark)/eta -- the square root is why NEP
    keeps improving with pump power after the equivalent n_bar has turned back
    up. The two cannot share a rescaled axis.

    The factor 2 is the usual tau = 1/(2 df) bandwidth convention; drop it if
    you are comparing against a reference that does.
    """
    h_nu = PLANCK * frequency_b * 1e6          # frequency_b is in MHz
    duty = length_b / cycle
    gamma_dark = p_dark / (cycle * 1e-6)       # per second, wall clock
    return (h_nu / (eta * duty)) * np.sqrt(2.0 * gamma_dark)


def ringup_efficiency(lengths, a, t, tau):
    """
    A * window average of ring-up (1 - e^(-t/tau))^2 x survival e^(-(L-t)/T).

    tau = 0 gives A (T/L)(1 - e^(-L/T)), a uniform arrival time surviving to readout.
    T is the g-e lifetime (under the pump), never T1_ef: a conversion leaves the
    qubit in |e>; on run 33 T1_ef would have inflated the ceiling 0.606 -> 0.721.
    """
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


def plot_efficiency_vs_length(efficiencies, cycle, t1, kappa_ratio, dynamics, tau_ceiling):
    """
    Efficiency and efficiency x duty vs buffer length, with the window-dynamics fit and ceiling.

    `dynamics` is {scan: fit_window_dynamics result}; the ceiling rings up
    with `tau_ceiling`, the matched 1/(2 pi kappa). Returns (fig, fits):
    fits[scan] is that result plus L_peak, where the fitted efficiency x duty
    peaks (NaN if at the edge of the sweep).
    """
    markers = {"e": "o", "f": "s"}
    lengths = np.array(sorted(cycle))
    duty = lengths / np.array([cycle[v] for v in lengths])
    dead = float(np.median(lengths / duty - lengths))
    # From L -> 0, so the ring-up roll-off below the shortest length shows.
    grid = np.linspace(0, lengths.max(), 801)[1:]
    grid_duty = grid / (grid + dead)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    fits = {}
    for scan, table in efficiencies.items():
        eta = np.array([row[1] for row in table])
        err = np.array([row[2] for row in table])
        fit = dynamics.get(scan)
        good = fit is not None
        model = ringup_efficiency(grid, fit["A"], fit["T"], fit["tau"]) if good else grid * np.nan
        peak = int(np.argmax(model * grid_duty)) if good else 0
        if good:
            fits[scan] = {**fit, "L_peak": grid[peak] if 0 < peak < len(grid) - 1 else np.nan}
        for axis, scale, model_scale in ((axes[0], 1, 1), (axes[1], duty, grid_duty)):
            axis.errorbar(lengths, eta * scale, yerr=err * scale, fmt=markers[scan],
                          ms=6, capsize=3, color=SCAN_COLORS[scan],
                          label=f"{scan} readout")
            if good:
                axis.plot(grid, model * model_scale, color=SCAN_COLORS[scan], lw=1,
                          label=f"fit T = {fit['T']:.0f} ± {fit['T_err']:.0f} us, "
                                f"τ = {fit['tau']:.1f} ± {fit['tau_err']:.1f} us")
    ceiling = kappa_ratio * ringup_efficiency(grid, 1.0, t1, tau_ceiling)
    for axis, model_scale in ((axes[0], 1), (axes[1], grid_duty)):
        axis.plot(grid, ceiling * model_scale, color="0.5", ls="--",
                  label=f"ceiling, T1 = {t1:.0f} us, τ = {tau_ceiling:.2f} us")
    l_star = np.sqrt(2 * dead * t1)
    axes[1].axvline(l_star, color="0.5", ls=":",
                    label=f"sqrt(2 dead T1) = {l_star:.0f} us")

    axes[0].set_ylabel("efficiency per arriving photon")
    axes[1].set_ylabel("efficiency x duty cycle")
    for axis in axes:
        axis.set_xlabel("buffer length (us)")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.3)
        axis.legend(fontsize="x-small")
    return fig, fits


def n_bar_from_arrivals(arrivals, length_b, kappa, kappa_c):
    """
    Inverse of `photons_incident`: the n_bar that would deliver this count.

    Used to quote a dark-count probability as an equivalent buffer occupation
    -- the standing n_bar a real signal would need to produce the same number
    of clicks.
    """
    rate, rate_c = 2.0 * np.pi * kappa, 2.0 * np.pi * kappa_c
    return (np.asarray(arrivals, dtype=float) * 4.0 * rate_c
            / (rate ** 2 * np.asarray(length_b, dtype=float)))


def efficiency_vs_pump(records, by_key, scan, arrivals_of, t_window, tau,
                       sweep="gain_p"):
    """
    Per-photon efficiency at each value of `sweep`, from the window dynamics.

    At each value -- pump gain, or buffer frequency for a script-37 run -- fits
    c + (1 - c) window_population(L, arrivals, A, T, tau) to every buffer gain
    with T = `t_window` and `tau` held (one window length cannot separate them
    from A), and reports eta = A ringup_efficiency(L, 1, T, tau), the
    linear-regime slope -- the same quantity fit_window_dynamics gives per
    length. Returns a list of (value, eta, eta_err, c).
    """
    from scipy.optimize import least_squares

    out = []
    for value in sorted({r[sweep] for r in records}):
        rows = [r for r in records if r[sweep] == value]
        xs = np.array([arrivals_of(r) for r in rows])
        ys = np.array([by_key[(r["index"], scan)] for r in rows])
        if len(np.unique(xs)) < 3:
            out.append((value, float("nan"), float("nan"), float("nan")))
            continue
        length = float(np.median([r["length_b"] for r in rows]))
        res = least_squares(
            lambda x: x[1] + (1 - x[1]) * window_population(length, xs, x[0], t_window, tau) - ys,
            [1.0, float(np.clip(ys.min(), 0, 0.5))], bounds=([0, 0], [np.inf, 1]))
        cov = np.linalg.pinv(res.jac.T @ res.jac) * (res.fun @ res.fun) / max(len(xs) - 2, 1)
        unit = float(ringup_efficiency(length, 1.0, t_window, tau))
        out.append((value, float(res.x[0]) * unit, float(np.sqrt(cov[0, 0])) * unit,
                    float(res.x[1])))
    return out


# The axis a run steps along, with its label and how its values print.
SWEEP_AXES = {
    "gain_p": ("pump gain", "{:.2f}"),
    "frequency_b": ("buffer frequency (MHz)", "{:.4f}"),
    "length_b": ("buffer length (us)", "{:.1f}"),
}
# Row header in the printed tables.
SWEEP_SHORT = {"gain_p": "pump", "frequency_b": "buffer MHz",
               "length_b": "length us"}


def sweep_axis(records):
    """
    Which of pump gain / buffer frequency / buffer length this run sweeps.

    Script 23/26 grids vary pump gain, script 37 buffer frequency, script 40
    buffer length. More than one at once has no single efficiency curve to
    reduce to, so it is refused rather than guessed at.
    """
    varying = [key for key in SWEEP_AXES if len({r[key] for r in records}) > 1]
    if len(varying) > 1:
        raise SystemExit(f"Run varies {', '.join(varying)}; expected one of them.")
    return varying[0] if varying else "gain_p"


def load_run(data_dir, bins):
    """
    Read every .h5 in the run and histogram all scans on one common grid.

    Histogrammed from the raw I shots rather than the stored `hist*` arrays,
    because each file bins over its own range and a shared fit needs a shared
    grid. The stored rotation angle is reapplied; it is 0 for these files.
    """
    files = sorted(data_dir.glob("*.h5"))
    if not files:
        raise SystemExit(f"No .h5 files in {data_dir}")

    records = []
    for path in files:
        with SlabFile(str(path), "r") as h5:
            keys = set(h5.keys())
            angle = float(np.array(h5["angle"])) if "angle" in keys else 0.0
            theta = np.deg2rad(angle)
            shots = {}
            for scan in SCANS:
                if f"I{scan}" not in keys:
                    continue
                i = np.array(h5[f"I{scan}"])
                q = np.array(h5[f"Q{scan}"])
                shots[scan] = i * np.cos(theta) - q * np.sin(theta)
            cfg = h5.load_config()
        expt = cfg["expt"]
        # What the RF board actually applied to the buffer DAC
        soc = cfg["hw"]["soc"]
        board = soc["rfboard_active"]["dac_readout"][str(soc["dacs"]["readout"]["ch"][BUFFER_LINE])]
        records.append({
            "buffer_attn_db": float(board["attn1"]) + float(board["attn2"]),
            "pulse_type_b": expt.get("pulse_type_b"),
            "path": path,
            "index": int(path.stem.rsplit("_", 1)[1]),
            "gain_p": float(expt["gain_p"]),
            "frequency_p": float(expt["frequency_p"]),
            "gain_b": float(expt["gain_b"]),
            "length_b": float(expt["length_b"]),
            "frequency_b": float(expt["frequency_b"]),
            "final_delay": float(expt["final_delay"]),
            # The tail actually spent between shots, which is NOT
            # final_delay once a reset runs: effective_final_delay
            # resolves reset_mode for us.
            # reset_latch / reset_verify postdate the earlier active-reset
            # runs; absent from a file means the feature was off.
            "tail": float(effective_final_delay(AttrDict(
                {"reset_latch": False, "reset_verify": False, **expt}))),
            "reset_mode": str(expt.get("reset_mode",
                                       "active" if expt.get("active_reset")
                                       else "passive")),
            "sep_readout": bool(expt["sep_readout"]),
            "qubit": int(np.atleast_1d(expt["qubit"])[0]),
            "shots": shots,
        })

    everything = np.concatenate([s for r in records for s in r["shots"].values()])
    edges = np.histogram_bin_edges(everything, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    for record in records:
        record["hist"] = {
            scan: np.histogram(values, bins=edges, density=True)[0]
            for scan, values in record["shots"].items()
        }
    return records, centers


def state_shapes(centers, vg, ve, sigma, t_ratio):
    """
    Unit-area distributions for a shot prepared in |g> and one in |e>.

    The ground shape is a plain Gaussian. The excited shape is not: over a
    readout of length t_meas a fraction 1 - exp(-t_ratio) of excited shots
    relax part-way through, and those land *between* the two levels rather
    than in either. With t_ratio = t_meas/T1 = 0.19 here that is 17% of
    excited shots, and a symmetric two-Gaussian model has nowhere to put them
    -- it apportions them partly to ground and under-reports P(e).

    The decay continuum is `t1_decay_distribution` from the codebase's own
    single-shot analysis, so this is the same model `combined_state_model`
    uses, split into its two shapes so the fraction stays a linear parameter.
    """
    ground = gaussian(centers, vg, sigma)
    excited = (np.exp(-t_ratio) * gaussian(centers, ve, sigma)
               + t1_decay_distribution(centers, vg, ve, sigma, t_ratio))
    return ground, excited


def fractions_for(centers, hists, vg, ve, sigma, t_ratio,
                  t_ratio_f=None, is_f=None):
    """
    Best excited fraction for each histogram at fixed component shapes.

    Still linear in the fraction even with the decay tail -- the tail belongs
    to the excited shape and does not depend on how much of it there is -- so
    this remains a one-dimensional projection: hist - G = p * (E - G). Clipped
    to [0, 1] so an unphysical fraction cannot drag the shared parameters
    around.
    """
    if is_f is None:
        is_f = np.zeros(len(hists), dtype=bool)
    if t_ratio_f is None:
        t_ratio_f = t_ratio
    # Only two distinct shape pairs, so build them once rather than per row.
    shapes = {False: state_shapes(centers, vg, ve, sigma, t_ratio),
              True: state_shapes(centers, vg, ve, sigma, t_ratio_f)}
    out = []
    for hist, flag in zip(hists, is_f):
        ground, excited = shapes[bool(flag)]
        delta = excited - ground
        denom = float(delta @ delta)
        out.append(0.0 if denom <= 0
                   else float(np.clip((hist - ground) @ delta / denom, 0.0, 1.0)))
    return np.array(out)


def fit_shared(centers, hists, seed, is_f=None, t_ratio_bounds=(0.01, 3.0)):
    """
    Search the four shared parameters; the fractions follow analytically.

    sigma and t_ratio are searched as logarithms to keep them positive.
    t_ratio is shared across all 150 histograms, so it is far better
    constrained here than the per-histogram T1 fits that this codebase's notes
    record as unreliable.
    """

    if is_f is None:
        is_f = np.zeros(len(hists), dtype=bool)
    is_f = np.asarray(is_f, dtype=bool)

    def residuals(theta):
        vg, ve, log_sigma, log_t_ratio, log_t_ratio_f = theta
        sigma = np.exp(log_sigma)
        t_ratio, t_ratio_f = np.exp(log_t_ratio), np.exp(log_t_ratio_f)
        shapes = {False: state_shapes(centers, vg, ve, sigma, t_ratio),
                  True: state_shapes(centers, vg, ve, sigma, t_ratio_f)}
        fracs = fractions_for(centers, hists, vg, ve, sigma, t_ratio,
                              t_ratio_f, is_f)
        out = []
        for p, hist, flag in zip(fracs, hists, is_f):
            ground, excited = shapes[bool(flag)]
            out.append((1 - p) * ground + p * excited - hist)
        return np.concatenate(out)

    # Bound the two decay ratios. t_ratio = t_meas/T1_eff, so the bounds are
    # really a statement about T1: the default (0.01, 3) spans T1_eff from
    # ~500x the readout down to a third of it. Unbounded, run 33 #461 walked
    # t_ratio to 184.9 -- "100% of excited shots decay during readout" -- which
    # is not a fit but a degenerate solution where the excited shape collapses
    # onto the decay continuum and every fraction is meaningless. The f ratio
    # in that same run stayed sensible, so only the shared e-state parameter
    # diverged and nothing flagged it.
    lo, hi = (np.log(t_ratio_bounds[0]), np.log(t_ratio_bounds[1]))
    x0 = [seed[0], seed[1], np.log(seed[2]),
          float(np.clip(np.log(seed[3]), lo, hi)),
          float(np.clip(np.log(seed[4]), lo, hi))]
    result = least_squares(
        residuals,
        x0=x0,
        bounds=([-np.inf, -np.inf, -np.inf, lo, lo],
                [np.inf, np.inf, np.inf, hi, hi]),
        method="trf",
        xtol=1e-12,
        ftol=1e-12,
    )
    for val, name in ((result.x[3], "t_ratio"), (result.x[4], "t_ratio_f")):
        if abs(val - lo) < 1e-6 or abs(val - hi) < 1e-6:
            print(f"  WARNING: {name} hit its bound at {np.exp(val):.4f}. The "
                  "fit is being held there, not measured -- check the readout "
                  "calibration and this run's histograms before using any "
                  "fraction from it.")
    vg, ve, log_sigma, log_t_ratio, log_t_ratio_f = result.x
    return (float(vg), float(ve), float(np.exp(log_sigma)),
            float(np.exp(log_t_ratio)), float(np.exp(log_t_ratio_f)), result)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="device config holding device.fwm.buffer_power_fit")
    parser.add_argument("--calibration", choices=("ramsey_spec", "ramsey", "stark"),
                        default="ramsey_spec",
                        help="gain -> photon fit, rescaled to this run's attenuation "
                             "and pulse shape: script 41's spectrum (default), the "
                             "latest script-38 slope, or the Stark/PNS fit")
    parser.add_argument("--pump-gain", type=float, default=EXAMPLE_PUMP_GAIN,
                        help="pump gain for the per-histogram figure")
    parser.add_argument("--frequency-b", type=float, default=None,
                        help="buffer frequency (MHz) for the per-histogram "
                             "figure of a frequency sweep; default the one with "
                             "the highest buffer+pump efficiency")
    parser.add_argument("--length-b", type=float, default=None,
                        help="buffer length (us) for the per-histogram figure "
                             "of a length sweep; default as --frequency-b")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.config is None:
        parser.error(
            f"could not find a configs/*_rfboard.yml matching run folder "
            f"{HERE.parents[1].name!r}; pass --config explicitly")

    (coeffs, fit_type, reference, kappa, kappa_c, t1_list, t1_ef_list,
     readout_length, pop_list, temp_list) = load_calibration(
        REPO_DIR / "configs" / args.config, args.calibration)
    print(f"Photon calibration from {args.config} ({args.calibration})")
    print(f"  gain -> n_bar = {coeffs}, type = {fit_type}")
    print(f"  buffer kappa {kappa:.4f} MHz, kappa_c {kappa_c:.4f} MHz "
          f"(kappa_c/kappa = {kappa_c / kappa:.3f})")
    print(f"  arrival flux = kappa^2 n_bar / (4 kappa_c) = "
          f"{(2 * np.pi * kappa) ** 2 / (4 * 2 * np.pi * kappa_c):.4f}"
          f" * n_bar photons/us  (reflection)")
    # Detection is absorption into the pump-induced port to the waste, at some
    # rate Gamma, competing with internal loss:
    #     eta(Gamma) = 4 kappa_c Gamma / (kappa_c + kappa_i + Gamma)**2
    # which peaks at Gamma = kappa_c + kappa_i = kappa (the pump-off
    # linewidth), so the ceiling is set purely by internal loss. Note this is
    # *not* the pump-off absorbed fraction 4 kappa_c kappa_i / kappa**2 --
    # photons taken by internal loss are lost, not detected.
    print(f"  eta ceiling = kappa_c/kappa = {kappa_c / kappa:.3f}, reached "
          f"when the pump opens Gamma = {kappa:.4f} MHz")

    data_dir = Path(args.data_dir)
    print(f"Reading {data_dir}")
    records, centers = load_run(data_dir, args.bins)
    calib_note = f"{args.calibration}, gain -> n_bar {coeffs} ({fit_type})"
    if reference is not None:
        # photons() is linear in the coefficients, so the rescale is exact
        scale, attn, pulse = calibration_scale(records, reference)
        coeffs = [c * scale for c in coeffs]
        print(f"  run taken at {attn:g} dB, {pulse}; calibration stated at "
              f"{reference[0]:g} dB, {reference[1]}: x{scale:.4g} -> {coeffs}")
        calib_note += (f" at {reference[0]:g} dB {reference[1]}, x{scale:.4g} "
                       f"for this run's {attn:g} dB {pulse} -> {coeffs}")
    scans = [s for s in SCANS if s in records[0]["hist"]]
    # A run carrying 'n' predates no-drives moving into the 'g' slot, so its
    # 'g' is buffer-only and the new label would be wrong for it. 'n' is the
    # discriminator: new runs never have it.
    if "n" in scans:
        SCAN_LABELS["g"] = "no pump"
        print("  (pre-'d' run: 'g' is buffer-only here, labelled accordingly)")
    print(f"  {len(records)} files x {len(scans)} scans = "
          f"{len(records) * len(scans)} histograms, {args.bins} shared bins")
    sweep = sweep_axis(records)
    sweep_label, sweep_fmt = SWEEP_AXES[sweep]
    print(f"  sweep axis: {sweep_label}")

    def fmt(value):
        """A sweep value as the printed tables show it."""
        return (f"{value:4.1f}" if sweep == "gain_p"
                else sweep_fmt.format(value))

    keys = [(r["index"], s) for r in records for s in scans]
    hists = [r["hist"][s] for r in records for s in scans]

    # Seeds: the no-pump scans are nearly all ground, so their pooled peak
    # locates vg; the excited side of the strongest pump+buffer scan gives ve.
    pooled_g = np.mean([r["hist"]["g"] for r in records], axis=0)
    vg0 = float(centers[np.argmax(pooled_g)])
    sigma0 = float(np.std(records[0]["shots"]["g"]))
    away = np.abs(centers - vg0) > 3 * sigma0 / 2
    if sweep == "gain_p":
        strongest = max(records, key=lambda r: r["gain_p"] * r["gain_b"])
    else:
        # One pump power, so the largest buffer gain is a tie across every
        # frequency and most of those sit off resonance with little excited
        # blob to seed from. Take the point that actually shows the most.
        strongest = max(records, key=lambda r: float(r["hist"]["e"][away].sum()))
    ve0 = float(centers[away][np.argmax(strongest["hist"]["e"][away])])
    # t_ratio = t_meas/T1 is seeded from the config and then fit, so the
    # fitted value is a check on the readout length and T1 rather than an
    # assumption taken from them.
    qubit_i = records[0]["qubit"]
    t1 = float(t1_list[qubit_i])
    # Thermal P(e) from the last 16_temperature.py run; None when absent or 0,
    # which is how the config marks a line that was never measured.
    thermal_pe = (float(pop_list[qubit_i])
                  if qubit_i < len(pop_list) and pop_list[qubit_i] else None)
    thermal_mk = (float(temp_list[qubit_i])
                  if qubit_i < len(temp_list) and temp_list[qubit_i] else None)
    t_ratio0 = readout_length / t1
    # The f scan gets its own Tm/T1. A shot prepared in |f> relaxes at the
    # f-state rate, not the e-state one, so forcing both through a single
    # ratio makes the fit split the difference and biases every scan's
    # fraction. Seeded from device.qubit.T1_ef when 10_t1 has been run with
    # pulse_ef (which now writes there rather than overwriting T1).
    t1_ef = float(t1_ef_list[qubit_i]) if qubit_i < len(t1_ef_list) else float("nan")
    if np.isfinite(t1_ef) and t1_ef > 0:
        t_ratio_f0 = readout_length / t1_ef
        ef_src = f"{readout_length:.2f} us / T1_ef {t1_ef:.2f} us"
    else:
        t_ratio_f0 = t_ratio0
        ef_src = "no device.qubit.T1_ef, seeded from the e-state value"
    is_f = np.array([s_ == "f" for r in records for s_ in scans])
    print(f"  seeds: vg={vg0:.3f}  ve={ve0:.3f}  sigma={sigma0:.3f}  "
          f"t_ratio={t_ratio0:.4f} ({readout_length:.2f} us / T1 {t1:.2f} us)")
    print(f"         f-scan t_ratio={t_ratio_f0:.4f} ({ef_src})")

    # T1_eff from 500x the readout length down to a third of it -- generous,
    # but enough to stop the degenerate runaway.
    t_ratio_bounds = (readout_length / 500.0, readout_length / 2.0)
    vg, ve, sigma, t_ratio, t_ratio_f, result = fit_shared(
        centers, hists, (vg0, ve0, sigma0, t_ratio0, t_ratio_f0), is_f,
        t_ratio_bounds)
    fracs = fractions_for(centers, hists, vg, ve, sigma, t_ratio,
                          t_ratio_f, is_f)
    rms = float(np.sqrt(np.mean(result.fun ** 2)))
    print(f"\nShared components:  vg={vg:.4f}  ve={ve:.4f}  sigma={sigma:.4f}")
    print(f"  t_ratio = {t_ratio:.4f}  (seeded {t_ratio0:.4f}) "
          f"-> {100 * (1 - np.exp(-t_ratio)):.1f}% of excited shots decay "
          "during readout")
    if is_f.any():
        print(f"  t_ratio_f = {t_ratio_f:.4f}  (seeded {t_ratio_f0:.4f}) "
              f"-> {100 * (1 - np.exp(-t_ratio_f)):.1f}% of |f> shots decay "
              f"during readout, i.e. T1_ef = {readout_length / t_ratio_f:.1f} us")
    print(f"  separation {abs(ve - vg) / sigma:.2f} sigma,  residual rms {rms:.3e}")

    by_key = dict(zip(keys, fracs))

    out_csv = HERE / "fit_shared_means.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "pump_frequency_MHz", "pump_gain",
                         "buffer_frequency_MHz",
                         "buffer_gain", "buffer_length_us", "n_bar",
                         "one_minus_p0", "photons_incident", "scan",
                         "scan_label", "excited_fraction"])
        for record in records:
            arrivals = photons_incident(record["gain_b"], record["length_b"],
                                        coeffs, fit_type, kappa, kappa_c)
            for scan in scans:
                writer.writerow([
                    record["index"], f"{record['frequency_p']:.4f}",
                    record["gain_p"], f"{record['frequency_b']:.4f}",
                    record["gain_b"], record["length_b"],
                    f"{photons(record['gain_b'], coeffs, fit_type):.6f}",
                    f"{one_minus_p0(record['gain_b'], coeffs, fit_type):.6f}",
                    f"{arrivals:.4f}",
                    scan, SCAN_LABELS[scan],
                    f"{by_key[(record['index'], scan)]:.6f}",
                ])
    print(f"  wrote {out_csv.name}")

    # Pump gains for a script-23/26 grid, buffer frequencies for script 37.
    sweep_values = sorted({r[sweep] for r in records})
    buffer_gains = sorted({r["gain_b"] for r in records})

    # Per sweep value: one value for 23/26/37, and per length for script 40.
    t_d = {v: float(np.median([r["length_b"] for r in records if r[sweep] == v]))
           for v in sweep_values}
    length_b = t_d[sweep_values[0]]
    by_length = sweep == "length_b"
    # The pump is on for the whole window, so every sweep uses T1 under the pump.
    t1_window, t1_src = t1, "device.qubit.T1"
    t1_pump = config.load(str(REPO_DIR / "configs" / args.config)).device.get(
        "fwm", {}).get("T1_pump")
    if t1_pump is not None and np.isfinite(t1_pump) and t1_pump > 0:
        t1_window, t1_src = float(t1_pump), "device.fwm.T1_pump"
    else:
        print("  no device.fwm.T1_pump; the efficiency fit and ceiling use device.qubit.T1")

    # Two factors nothing about the pump can beat: the buffer reflects
    # 1 - kappa_c/kappa of what arrives, and a conversion has to survive to
    # readout. Their product is the most any measured slope could report.
    # Distinct from the decay *during* readout, which the histogram fit now
    # models and so no longer biases the points.
    # t1, not t1_ef -- a converted photon leaves the qubit in |e>, so the
    # g-e lifetime is what limits survival to the readout. See the note
    # on ringup_efficiency.
    # The ceiling rings up at the matched Gamma = kappa (linewidth 2 kappa); every sweep alike.
    tau_match = 1 / (2 * np.pi * kappa)
    eta_qubit = {v: float(ringup_efficiency(t_d[v], 1.0, t1_window, tau_match))
                 for v in sweep_values}
    ceiling = {v: kappa_c / kappa * eta_qubit[v] for v in sweep_values}
    print(f"\nqubit T1 = {t1_window:.2f} us ({t1_src}), ceiling ring-up at "
          f"matching 1/(2 pi kappa) = {tau_match:.2f} us; per window:")
    for v in (sweep_values if by_length else sweep_values[:1]):
        print(f"  T_d {t_d[v]:6.1f} us  eta_qubit {eta_qubit[v]:.4f}  "
              f"ceiling {ceiling[v]:.3f}")
    # Arrivals per us of window for a length sweep, per pulse otherwise.
    arrival_window = 1.0 if by_length else length_b
    print("\nbuffer gain -> photons ("
          + ("arrivals per us of window" if by_length else f"length_b = {length_b:g} us")
          + "):")
    print("  gain      n_bar   1-P(0)   arrivals")
    for gain in buffer_gains:
        print(f"  {gain:.3f}  {photons(gain, coeffs, fit_type):8.4f}  "
              f"{one_minus_p0(gain, coeffs, fit_type):7.4f}  "
              f"{photons_incident(gain, arrival_window, coeffs, fit_type, kappa, kappa_c):9.2f}")
    lookup = {(r[sweep], r["gain_b"]): r["index"] for r in records}
    for scan in scans:
        print(f"\nexcited fraction, {SCAN_LABELS[scan]} (rows = {sweep_label}):")
        print("  " + SWEEP_SHORT[sweep].rjust(9)
              + "\\buf " + "".join(f"{b:>9.3f}" for b in buffer_gains))
        for pg in sweep_values:
            row = f"  {sweep_fmt.format(pg):>9}  "
            for bg in buffer_gains:
                index = lookup.get((pg, bg))
                row += ("       - " if index is None
                        else f"{by_key[(index, scan)]:>9.4f}")
            print(row)

    # --- figure 1: the fit on a few histograms ---------------------------
    ground, excited = state_shapes(centers, vg, ve, sigma, t_ratio)
    ground_f, excited_f = state_shapes(centers, vg, ve, sigma, t_ratio_f)
    examples = [records[0], records[len(records) // 2], records[-1]]
    fig1, axes = plt.subplots(1, len(examples), figsize=(5 * len(examples), 4.2),
                              sharey=True)
    axes = np.atleast_1d(axes)
    for axis, record in zip(axes, examples):
        for scan in ("g", "e"):
            p = by_key[(record["index"], scan)]
            axis.semilogy(centers, record["hist"][scan], lw=1,
                          color=SCAN_COLORS[scan], label=SCAN_LABELS[scan])
            axis.semilogy(centers, (1 - p) * ground + p * excited, "k--", lw=1)
            axis.semilogy(centers, (1 - p) * ground, ":", color="0.6", lw=0.8)
            axis.semilogy(centers, p * excited, ":", color="0.6", lw=0.8)
        axis.set_xlabel("I (ADC levels)")
        axis.set_title(
            {"gain_p": f"pump {record['gain_p']:.2f}",
             "frequency_b": f"{record['frequency_b']:.4f} MHz",
             "length_b": f"{record['length_b']:g} us"}[sweep]
            + f", buffer {record['gain_b']:.3f}",
            fontsize="medium")
        axis.set_ylim(1e-5, 1)
        axis.tick_params(labelsize=11)
    axes[0].set_ylabel("Probability density")
    axes[0].legend(fontsize="small")
    fig1.suptitle(
        f"{HERE.name}    shared vg={vg:.3f}, ve={ve:.3f}, sigma={sigma:.3f}"
        "    (dashed = fit, dotted = components)"
    )
    fig1.tight_layout()
    fig1.savefig(HERE / "fit_shared_means_examples.png", dpi=150)

    # --- figure 2: fractions across the grid -----------------------------
    order = [(scan, scale) for scan, scale in GRID_ORDER if scan in scans]
    fig2 = plt.figure(figsize=(3.6 * len(order), 7.6))
    grid2 = fig2.add_gridspec(2, len(order), hspace=0.32, wspace=0.58)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(sweep_values), vmax=max(sweep_values))

    def arrivals_of(record):
        return float(photons_incident(record["gain_b"], record["length_b"],
                                      coeffs, fit_type, kappa, kappa_c))

    # y is shared within each scale group but not across them -- one shared
    # axis for all five would drag the linear panels onto a log scale too.
    lead = {}
    axes2 = []
    dropped = []
    for position, (scan, scale) in enumerate(order):
        axis = fig2.add_subplot(grid2[0, position],
                                sharey=lead.get(scale))
        if scale in lead:
            axis.tick_params(labelleft=False)
        else:
            lead[scale] = axis
            axis.set_ylabel("excited fraction")
        axes2.append(axis)

        for pg in sweep_values:
            rows = sorted((r for r in records if r[sweep] == pg),
                          key=lambda r: r["gain_b"])
            if scan in PHOTON_AXIS_SCANS:
                xs = [arrivals_of(r) for r in rows]
            else:
                xs = [r["gain_b"] for r in rows]
            ys = np.array([by_key[(r["index"], scan)] for r in rows])
            if scale == "log":
                # NaN rather than left as zero: a log axis drops the point but
                # still draws the line toward it, which reads as a plunge to
                # some tiny value instead of a gap.
                dropped += [(scan, pg, record["gain_b"])
                            for record, y in zip(rows, ys) if y <= 0]
                ys = np.where(ys > 0, ys, np.nan)
            axis.plot(xs, ys, "o-", ms=4, lw=1, color=cmap(norm(pg)),
                      label=f"{pg:.1f}" if sweep == "gain_p"
                      else sweep_fmt.format(pg))
        if scale == "log":
            axis.set_yscale("log")
        else:
            axis.set_ylim(top =1)
            axis.set_xlim(left=0)
        axis.set_title(SCAN_LABELS[scan], fontsize="medium")
        axis.set_xlabel("photons arriving at buffer"
                        if scan in PHOTON_AXIS_SCANS else "buffer gain")
        if scan == 'e' or scan == 'f':
            axis.grid(True)
        axis.tick_params(labelsize=11)

    axes2[-1].legend(fontsize="small", title=sweep_label)

    # Bottom row: a summary under the column it came from -- efficiency for the
    # conversion scans, dark count for the pump-off scan. The two pure
    # background columns stay blank rather than carrying a slope through noise.
    efficiencies = {scan: efficiency_vs_pump(records, by_key, scan,
                                             arrivals_of, t1_window, tau_match, sweep)
                    for scan in EFFICIENCY_SCANS if scan in scans}
    # A length sweep can also fit T and tau, shared across lengths, so its
    # efficiencies come from that one fit (CLAUDE.md script 40).
    dynamics = {}
    # Pump-off amplitude ring-up 2/kappa; the fit's tau should be at or below it.
    tau_ring = 1 / (np.pi * kappa)
    if by_length:
        for scan in efficiencies:
            groups = {}
            for v in sweep_values:
                at = [r for r in records if r[sweep] == v]
                if len({r["gain_b"] for r in at}) >= 2:
                    groups[v] = (np.array([arrivals_of(r) for r in at]),
                                 np.array([by_key[(r["index"], scan)] for r in at]))
            fit = fit_window_dynamics(groups, t1_window, tau_ring)
            if fit is None:
                continue
            dynamics[scan] = fit
            nan = (float("nan"), float("nan"))
            efficiencies[scan] = [(v, *fit["eta"].get(v, nan), fit["c"].get(v, float("nan")))
                                  for v in sweep_values]
    # Bottom-row x axis. Pump gain as is; a buffer frequency as the detuning
    # from the middle of the sweep, since 6230.xx MHz ticks are unreadable.
    if sweep != "frequency_b":
        x_origin, x_label = 0.0, sweep_label
    else:
        x_origin = float(np.median(sweep_values))
        x_label = f"$f_b$ $-$ {x_origin:.3f} (MHz)"
    sweep_x = np.array(sweep_values) - x_origin
    eff_lead = None
    dark_rows = []
    # Dark counts and the noise-equivalent photon number they imply, computed
    # before the panel loop because they are drawn in different columns. One
    # NEP per readout: the dark count is the same either way, but dividing by
    # the two efficiencies gives two different equivalent signals.
    eta_by_pump = {scan: {row[0]: row[1] for row in table}
                   for scan, table in efficiencies.items()}
    # The measurement cycle, straight from the pulse sequence: armed window,
    # the gap before readout, the readout, and the passive relaxation wait.
    gap = 0.01 if records[0]["sep_readout"] else 0.0
    # Dead time is the tail actually spent, not the passive final_delay:
    # under a reset, _get_final_delay() drops that to 1 us and the reset
    # sequence replaces it. Reading final_delay raw put run 33 #383 at
    # 8.06% duty when the f0g1 reset made it 72.23% -- a 9x cycle error,
    # which overstates the NEP by sqrt(9) = 3x.
    mode = records[0]["reset_mode"]
    tail = records[0]["tail"]
    cycle = {v: t_d[v] + gap + readout_length + tail for v in sweep_values}
    duty = {v: t_d[v] / cycle[v] for v in sweep_values}
    dead = gap + readout_length + tail
    # The single-window values, for everything below that is not per length.
    first = sweep_values[0]
    cycle0, duty0 = cycle[first], duty[first]
    if by_length:
        print(f"\nduty cycle per window, dead {dead:.2f} us ({gap:g} gap + "
              f"{readout_length:.2f} readout + {tail:.2f} tail, reset_mode {mode!r}):")
        for v in sweep_values:
            print(f"  T_d {t_d[v]:6.1f} us  cycle {cycle[v]:6.1f} us  "
                  f"duty {100 * duty[v]:5.1f}%")
    else:
        print(f"\nduty cycle {100 * duty0:.1f}%: {length_b:g} us armed of a "
              f"{cycle0:.1f} us cycle\n  ({gap:g} gap + {readout_length:.2f} readout"
              f" + {tail:.2f} inter-shot tail dead, reset_mode {mode!r})")
    if mode != "passive" and not by_length:
        passive_cycle = length_b + gap + readout_length + records[0]["final_delay"]
        print(f"  (the passive final_delay would have been "
              f"{records[0]['final_delay']:.2f} us, i.e. a "
              f"{passive_cycle / cycle0:.1f}x longer cycle and "
              f"{(passive_cycle / cycle0) ** 0.5:.1f}x the NEP)")

    dark_rows = []
    offband_rows = []
    floor_rows = []
    nep = {scan: [] for scan in efficiencies}
    nep_watt = {scan: [] for scan in efficiencies}
    for pump_gain in sweep_values:
        group = [r for r in records if r[sweep] == pump_gain]
        values = np.array([by_key[(r["index"], DARK_SCAN)] for r in group])
        mean, spread = float(values.mean()), float(values.std())
        dark_rows.append((pump_gain, mean, spread))
        if OFFBAND_SCAN in scans:
            off = np.array([by_key[(r["index"], OFFBAND_SCAN)] for r in group])
            offband_rows.append((pump_gain, float(off.mean()), float(off.std())))
        if FLOOR_SCAN in scans:
            flo = np.array([by_key[(r["index"], FLOOR_SCAN)] for r in group])
            floor_rows.append((pump_gain, float(flo.mean()), float(flo.std())))
        # The same click rate expressed as a real signal: divide by the
        # efficiency at this pump gain to get equivalent arrivals, then back
        # out the buffer occupation that would supply them.
        for scan_name in nep:
            eta = eta_by_pump[scan_name].get(pump_gain)
            nep[scan_name].append(
                float(n_bar_from_arrivals(mean / eta, t_d[pump_gain], kappa, kappa_c))
                if eta else float("nan")
            )
            nep_watt[scan_name].append(
                float(noise_equivalent_power(mean, eta,
                                             group[0]["frequency_b"],
                                             t_d[pump_gain], cycle[pump_gain]))
                if eta else float("nan")
            )

    for position, (scan, _) in enumerate(order):
        if scan not in EFFICIENCY_SCANS + (DARK_SCAN, NEP_SCAN, NEP_WATT_SCAN):
            fig2.add_subplot(grid2[1, position]).axis("off")
            continue

        axis = fig2.add_subplot(
            grid2[1, position],
            sharey=eff_lead if scan in EFFICIENCY_SCANS else None,
        )
        axis.set_xlabel(x_label)
        if sweep == "gain_p":
            axis.set_xlim(left=0)
        axis.grid(True)
        axis.tick_params(labelsize=11)

        if scan in EFFICIENCY_SCANS:
            if eff_lead is None:
                eff_lead = axis
                axis.set_ylabel("efficiency per arriving photon")
            else:
                axis.tick_params(labelleft=False)
            table = efficiencies[scan]
            axis.errorbar([row[0] - x_origin for row in table],
                          [row[1] for row in table],
                          yerr=[row[2] for row in table], fmt="o-", ms=5, lw=1,
                          capsize=3, color=SCAN_COLORS[scan])
            axis.set_ylim(0, 1)
            if by_length:
                axis.plot(sweep_x, [ceiling[v] for v in sweep_values],
                          color="0.4", ls="--", lw=1)
                continue
            axis.axhline(ceiling[first], color="0.4", ls="--", lw=1)
            axis.text(0.04, 0.94,
                      f"ceiling $\\kappa_c/\\kappa \\times \\eta_{{qubit}}$ = "
                      f"{kappa_c / kappa:.2f} $\\times$ {eta_qubit[first]:.2f} = "
                      f"{ceiling[first]:.2f}",
                      transform=axis.transAxes, fontsize=8, color="0.35",
                      va="top")
            continue

        if scan == DARK_SCAN:
            # The pump is on but no buffer pulse was played, so any excitation
            # is a false positive. Averaged over the buffer-gain settings,
            # which are repeats of the same measurement here, with their
            # spread as the error bar. Expected to *rise* with pump gain -- a
            # stronger pump excites the qubit directly.
            axis.errorbar(sweep_x, 100 * np.array([r[1] for r in dark_rows]),
                          yerr=100 * np.array([r[2] for r in dark_rows]),
                          fmt="o-", ms=5, lw=1, capsize=3,
                          color=SCAN_COLORS[scan],
                          label=SCAN_LABELS[scan] if offband_rows else None)
            if offband_rows:
                # Same power off the band, so the gap is the resonant part.
                axis.errorbar(
                    sweep_x, 100 * np.array([r[1] for r in offband_rows]),
                    yerr=100 * np.array([r[2] for r in offband_rows]),
                    fmt="s--", ms=4, lw=1, capsize=3,
                    color=SCAN_COLORS[OFFBAND_SCAN],
                    label=SCAN_LABELS[OFFBAND_SCAN])
            if floor_rows and offband_rows:
                # The floor the other two stand on; flat in pump gain by
                # construction, so it shows how much of the dark count is
                # readout rather than pump.
                axis.errorbar(
                    sweep_x, 100 * np.array([r[1] for r in floor_rows]),
                    yerr=100 * np.array([r[2] for r in floor_rows]),
                    fmt="^:", ms=4, lw=1, capsize=3,
                    color=SCAN_COLORS[FLOOR_SCAN],
                    label=SCAN_LABELS[FLOOR_SCAN])
            if thermal_pe is not None:
                axis.axhline(100 * thermal_pe, color="0.45", ls=(0, (6, 3)),
                             lw=1, label=_thermal_label(thermal_pe, thermal_mk))
            if offband_rows:
                axis.legend(fontsize=7, loc="upper left")
            axis.set_ylabel("dark count P(e)  (%)")
            axis.set_ylim(bottom=0)
            # The rate axis is a rescaling only when every point shares one cycle.
            if by_length:
                continue
            # A linked secondary axis is right here, unlike on the NEP panel:
            # Gamma = P(e)/cycle is a pure rescaling of the same data, so one
            # set of points carries both readings. Divided by the whole cycle,
            # not the armed window, so this is the count rate actually
            # observed -- dead time included. That choice is what makes eta_D
            # appear linearly in the NEP callout rather than as a square root:
            # one duty factor is already inside Gamma.
            seconds = cycle0 * 1e-6
            rate_axis = axis.secondary_yaxis(
                "right",
                functions=(lambda pct: pct / 100.0 / seconds,
                           lambda rate: rate * seconds * 100.0),
            )
            rate_axis.set_ylabel(r"$\Gamma$ (s$^{-1}$)")
            continue

        # Noise-equivalent photon number: the buffer occupation a real signal
        # would need to match the dark count. Falls where the efficiency is
        # rising faster than the dark count and turns back up once it is not,
        # so its minimum is the best operating point for sensitivity.
        # The two sensitivity metrics get separate panels because they are not
        # proportional: one goes as p_dark/eta, the other as sqrt(p_dark)/eta,
        # so the square root makes NEP more forgiving of dark counts and they
        # minimise at different pump gains. The callouts say which is which.
        if scan == NEP_SCAN:
            series, ylabel, scale = nep, r"noise-equivalent $\bar{n}$", 1.0
            # P_e, not Gamma: this is a per-shot occupation, with no rate and
            # no duty cycle in it. Gamma/eta would be a rate over an
            # efficiency, which is not an occupation at all.
            title, callout = ("dark count / efficiency",
                              r"$\propto P_e\,/\,\eta$")
        else:
            series, scale = nep_watt, 1e21
            ylabel = r"NEP (zW/$\sqrt{\mathrm{Hz}}$)"
            title = "NEP" if by_length else f"NEP,  $\\eta_D$ = {duty0:.2f}"
            # Gamma is the observed wall-clock dark rate, p_dark/cycle, as on
            # the secondary axis of the first panel in this row. It already
            # carries one factor of eta_D, which is why eta_D appears linearly
            # here; against the armed-window rate it would be a square root.
            callout = r"$\propto \sqrt{\Gamma}\,/\,(\eta\,\eta_D)$"

        for scan_name, values in series.items():
            axis.plot(sweep_x, np.array(values) * scale, "s--", ms=5, lw=1,
                      color=SCAN_COLORS[scan_name],
                      label=f"{scan_name} readout")
        axis.set_ylabel(ylabel)
        # Log y: both metrics run over more than a decade across the pump
        # scan, and on a linear axis with bottom=0 the low-NEP end -- the
        # part worth reading -- is compressed into the bottom few pixels.
        # Non-positive points are masked by matplotlib rather than plotted.
        axis.set_yscale("log")
        axis.set_title(title, fontsize="medium")
        # Bottom-centre put this on top of the curve once the axis went log
        # (the minimum sits low by construction); a boxed corner label stays
        # clear of the data at any scale.
        axis.text(0.5, 0.6, callout, ha="left", va="bottom", fontsize=12,
                  color="0.25", transform=axis.transAxes,
                  bbox=dict(boxstyle="round,pad=0.25", fc="white",
                            ec="0.85", alpha=0.9))
        axis.legend(fontsize=7, loc="upper center")

    fig2.savefig(HERE / "fit_shared_means_grid.png", dpi=150,
                 bbox_inches="tight")

    if by_length:
        print("\nefficiency = linear-regime eta from the window-dynamics fit over "
              "every buffer gain (intercept column = background c)")
    else:
        print(f"\nefficiency = linear-regime eta from the window-dynamics fit over every "
              f"buffer gain, T = {t1_window:.1f} us and tau = {tau_match:.2f} us held "
              "(intercept column = background c)")
    for scan, table in efficiencies.items():
        print(f"  {SCAN_LABELS[scan]}:")
        print(("    " + SWEEP_SHORT[sweep])
              + "   efficiency         intercept")
        for pump_gain, slope, err, intercept in table:
            print(f"    {fmt(pump_gain)}   {100 * slope:6.2f} +/- "
                  f"{100 * err:4.2f} %    {intercept:+.4f}")

    if dark_rows:
        print(f"\ndark count ({SCAN_LABELS[DARK_SCAN]}), averaged over the "
              f"{len(buffer_gains)} buffer gains, and the noise-equivalent "
              "n_bar it implies")
        nep_scans = list(nep)
        print(("    " + SWEEP_SHORT[sweep])
              + "      P(e)      spread   Gamma /s"
              + "".join(f"   n_bar[{s}]" for s in nep_scans)
              + "".join(f"  NEP[{s}] zW/rtHz" for s in nep_scans))
        for index, (pump_gain, mean, spread) in enumerate(dark_rows):
            # Wall-clock dark rate, the Gamma the panel callouts refer to.
            gamma = mean / (cycle[pump_gain] * 1e-6)
            print(f"    {fmt(pump_gain)}   {mean:.5f}   {spread:.5f}"
                  f"   {gamma:8.1f}"
                  + "".join(f"     {nep[s][index]:.5f}" for s in nep_scans)
                  + "".join(f"           {nep_watt[s][index] * 1e21:6.2f}"
                            for s in nep_scans))
        if offband_rows:
            # The split the detuned scan exists for: off-band is the pump
            # background that does not need the resonance, so the remainder is
            # the band-specific dark count.
            print(f"\n  resonant split, {SCAN_LABELS[DARK_SCAN]} vs "
                  f"{SCAN_LABELS[OFFBAND_SCAN]}")
            # Against the floor, not against the raw on-band number: both
            # scans stand on the same readout floor, and dividing by the total
            # buries the pump-dependent part under it.
            print("    " + SWEEP_SHORT[sweep]
                  + "     floor   off band   on band   nonres   resonant"
                    "   resonant share")
            for index, (value, mean, _) in enumerate(dark_rows):
                off = offband_rows[index][1]
                flo = floor_rows[index][1] if floor_rows else float("nan")
                excess = mean - flo
                share = (mean - off) / excess if excess else float("nan")
                print(f"    {fmt(value)}   {flo:.5f}    {off:.5f}   "
                      f"{mean:.5f}  {off - flo:+.5f}   {mean - off:+.5f}"
                      f"   {100 * share:11.0f} %")
        for scan_name in nep_scans:
            best_n = min(zip(sweep_values, nep[scan_name]), key=lambda p: p[1])
            best_w = min(zip(sweep_values, nep_watt[scan_name]),
                         key=lambda p: p[1])
            where = SWEEP_SHORT[sweep]
            print(f"    {SCAN_LABELS[scan_name]}: best n_bar "
                  f"{best_n[1]:.5f} at {where} {fmt(best_n[0]).strip()}; "
                  f"best NEP {best_w[1] * 1e21:.2f} zW/rtHz at {where} "
                  f"{fmt(best_w[0]).strip()}")

    # A log axis drops non-positive points without saying so, so say so here.
    if dropped:
        print("\nnot drawn on the log panels (fraction fit to zero):")
        for scan, pg, gain_b in dropped:
            print(f"  {SCAN_LABELS[scan]}, {sweep_label} {fmt(pg)}, "
                  f"buffer gain {gain_b:.3f}")

    # --- summary/: the reduced numbers in one place ----------------------
    # The grid figure's bottom row has these beside the raw fractions; this is
    # the short version to look at first. Same numbers as printed above, not a
    # second reduction. Written before figure 3, which can exit on a missing
    # pump gain.
    summary_dir = HERE / "summary"
    summary_dir.mkdir(exist_ok=True)
    eta_err = {scan: {row[0]: row[2] for row in table}
               for scan, table in efficiencies.items()}
    summary_rows = []
    for index, (value, mean, spread) in enumerate(dark_rows):
        row = {sweep: value,
               # Buffer gains in the window-dynamics fit at this value.
               "slope_points": sum(r[sweep] == value for r in records)}
        for scan in efficiencies:
            row[f"eta_{scan}"] = eta_by_pump[scan].get(value, float("nan"))
            row[f"eta_err_{scan}"] = eta_err[scan].get(value, float("nan"))
        row["p_dark"] = mean
        row["p_dark_spread"] = spread
        row["gamma_dark_per_s"] = mean / (cycle[value] * 1e-6)
        if offband_rows:
            row["p_offband"] = offband_rows[index][1]
            row["p_offband_spread"] = offband_rows[index][2]
            row["p_dark_resonant"] = mean - offband_rows[index][1]
        if floor_rows:
            row["p_floor"] = floor_rows[index][1]
            row["p_floor_spread"] = floor_rows[index][2]
            if offband_rows:
                row["p_pump_nonresonant"] = offband_rows[index][1] - floor_rows[index][1]
        if by_length:
            row.update(cycle_us=cycle[value], duty=duty[value],
                       eta_qubit=eta_qubit[value], ceiling=ceiling[value])
        for scan in nep:
            row[f"n_equiv_{scan}"] = nep[scan][index]
            row[f"nep_{scan}_zW_rtHz"] = nep_watt[scan][index] * 1e21
        summary_rows.append(row)
    with (summary_dir / "fwm_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    # A length sweep: efficiency and efficiency x duty vs length, with the A, T fit.
    length_fits = {}
    if by_length:
        fig4, length_fits = plot_efficiency_vs_length(
            efficiencies, cycle, t1_window, kappa_c / kappa, dynamics, tau_match)
        fig4.savefig(summary_dir / "fwm_summary.png", dpi=150)
        print(f"\nefficiency vs length, one fit of dP/dt = r(1 - P) - P/T, "
              f"r = A (N/L)(1 - e^(-t/tau))^2, to every point; "
              f"T1 = {t1_window:.1f} us ({t1_src}), "
              f"sqrt(2 dead T1) = {np.sqrt(2 * dead * t1_window):.1f} us, "
              f"pump-off ring-up 1/(pi kappa) = {tau_ring:.2f} us")
        for scan, fit in length_fits.items():
            print(f"  {SCAN_LABELS[scan]}: A = {fit['A']:.4f} +/- {fit['A_err']:.4f}, "
                  f"T = {fit['T']:.1f} +/- {fit['T_err']:.1f} us, "
                  f"tau = {fit['tau']:.2f} +/- {fit['tau_err']:.2f} us, "
                  f"rms {fit['rms']:.4f}, eta x duty peaks at {fit['L_peak']:.1f} us")
        print(("    " + SWEEP_SHORT[sweep]) + "    duty  "
              + "".join(f"   eta_{s} x duty" for s in efficiencies))
        for i, v in enumerate(sweep_values):
            print(f"    {fmt(v):>9}  {duty[v]:6.3f}"
                  + "".join(f"   {efficiencies[s][i][1] * duty[v]:12.4f}"
                            for s in efficiencies))
    # Efficiency, dark count, noise-equivalent n_bar, NEP. One y scale per
    # panel; e and f keep their colours from every other figure here, with a
    # marker each so they do not rely on colour alone.
    else:
        markers = {"e": "o", "f": "s"}
        fig4, axes4 = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
        ceiling0 = ceiling[first]
        eta_top = ceiling0
        for scan, table in efficiencies.items():
            eta = np.array([row[1] for row in table])
            err = np.array([row[2] for row in table])
            eta_top = max(eta_top, float(np.nanmax(eta + err)))
            axes4[0].errorbar(sweep_x, eta, yerr=err, marker=markers[scan], ms=6,
                              lw=1.5, capsize=3, color=SCAN_COLORS[scan],
                              label=f"{scan} readout")
        axes4[0].axhline(ceiling0, color="0.5", ls="--", lw=1)
        axes4[0].annotate(f"ceiling {ceiling0:.2f}", (1, ceiling0),
                          xycoords=("axes fraction", "data"), xytext=(-4, 4),
                          textcoords="offset points", ha="right", va="bottom",
                          fontsize=9, color="0.35")
        axes4[0].set_ylim(0, 1.15 * eta_top)
        axes4[0].set_ylabel("efficiency per arriving photon")

        axes4[1].errorbar(sweep_x, 100 * np.array([r[1] for r in dark_rows]),
                          yerr=100 * np.array([r[2] for r in dark_rows]),
                          marker="o", ms=6, lw=1.5, capsize=3,
                          color=SCAN_COLORS[DARK_SCAN],
                          label=SCAN_LABELS[DARK_SCAN] if offband_rows else None)
        if offband_rows:
            axes4[1].errorbar(
                sweep_x, 100 * np.array([r[1] for r in offband_rows]),
                yerr=100 * np.array([r[2] for r in offband_rows]),
                marker="s", ms=5, lw=1.5, ls="--", capsize=3,
                color=SCAN_COLORS[OFFBAND_SCAN],
                label=SCAN_LABELS[OFFBAND_SCAN])
            if floor_rows:
                axes4[1].errorbar(
                    sweep_x, 100 * np.array([r[1] for r in floor_rows]),
                    yerr=100 * np.array([r[2] for r in floor_rows]),
                    marker="^", ms=5, lw=1.5, ls=":", capsize=3,
                    color=SCAN_COLORS[FLOOR_SCAN],
                    label=SCAN_LABELS[FLOOR_SCAN])
            if thermal_pe is not None:
                axes4[1].axhline(
                    100 * thermal_pe, color="0.45", ls=(0, (6, 3)), lw=1.2,
                    label=_thermal_label(thermal_pe, thermal_mk))
            axes4[1].legend(fontsize=8, loc="upper left")
        axes4[1].set_ylim(bottom=0)
        axes4[1].set_ylabel("dark count P(e), pump only (%)")

        for axis, series, factor, ylabel in (
                (axes4[2], nep, 1.0, r"noise-equivalent $\bar{n}$"),
                (axes4[3], nep_watt, 1e21, r"NEP (zW/$\sqrt{\mathrm{Hz}}$)")):
            for scan, values in series.items():
                axis.plot(sweep_x, np.array(values) * factor, marker=markers[scan],
                          ms=6, lw=1.5, color=SCAN_COLORS[scan],
                          label=f"{scan} readout")
            axis.set_yscale("log")
            axis.set_ylabel(ylabel)
        # One direct label: the best NEP, which is the number this run is for.
        best = None
        for scan, values in nep_watt.items():
            for x, v in zip(sweep_x, values):
                if np.isfinite(v) and v > 0 and (best is None or v < best[2]):
                    best = (scan, x, v)
        if best is not None:
            axes4[3].annotate(f"{best[2] * 1e21:.2f} ({best[0]})",
                              (best[1], best[2] * 1e21), xytext=(0, -14),
                              textcoords="offset points", ha="center", va="top",
                              fontsize=9, color="0.2")
        for axis in axes4:
            axis.set_xlabel(x_label)
            if sweep == "gain_p":
                axis.set_xlim(left=0)
            axis.grid(alpha=0.3)
        for axis in (axes4[0], axes4[2], axes4[3]):
            axis.legend(fontsize="small")
        fig4.savefig(summary_dir / "fwm_summary.png", dpi=150)

    # The calibration this run's photon axis rests on: standing occupation and
    # arrival flux against buffer gain, with the gains actually played marked.
    rate, rate_c = 2.0 * np.pi * kappa, 2.0 * np.pi * kappa_c
    flux_per_n = rate ** 2 / (4.0 * rate_c)
    gains_played = np.array(buffer_gains)
    dense = np.linspace(0, gains_played.max(), 200)
    fig5, axes5 = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for axis, factor, ylabel in (
            (axes5[0], 1.0, r"$\bar{n}$ in buffer"),
            (axes5[1], flux_per_n, r"arrival flux $|b_\mathrm{in}|^2$ (photons/$\mu$s)")):
        axis.plot(dense, factor * photons(dense, coeffs, fit_type), color="#2a78d6",
                  lw=2, label=f"{args.calibration} calibration")
        axis.plot(gains_played, factor * photons(gains_played, coeffs, fit_type),
                  "o", ms=7, color="#2a78d6", mec="white", mew=1,
                  label="buffer gains in this run")
        axis.set_xlabel("buffer gain")
        axis.set_ylabel(ylabel)
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.3)
    axes5[0].legend(fontsize="small")
    longest = max(t_d.values())
    axes5[1].annotate(f"x {longest:g} us = "
                      f"{flux_per_n * photons(gains_played.max(), coeffs, fit_type) * longest:.2f}"
                      " arrivals at the top gain"
                      + (", longest window" if by_length else ""),
                      (0.03, 0.95), xycoords="axes fraction", ha="left", va="top",
                      fontsize=9, color="0.35")
    fig5.savefig(summary_dir / "photon_calibration.png", dpi=150)

    # The headline numbers, and what they depend on.
    def at_bound(value):
        return any(np.isclose(value, b) for b in t_ratio_bounds)

    lines = [
        f"FWM efficiency / NEP summary -- {HERE.name}",
        "",
        f"photon calibration   {calib_note}",
        f"buffer               kappa {kappa:.4f} MHz, kappa_c {kappa_c:.4f} MHz",
    ]
    if by_length:
        lines.append(f"ceiling              kappa_c/kappa * eta_qubit(T_d), T1 "
                     f"{t1_window:.2f} us ({t1_src}), ring-up at matching "
                     f"{tau_match:.2f} us; dead {dead:.2f} us, reset_mode {mode!r}")
        lines += [f"  T_d {t_d[v]:6.1f} us   ceiling {ceiling[v]:.3f}   "
                  f"duty {duty[v]:.3f}" for v in sweep_values]
        lines += [f"fit ({scan})              A {fit['A']:.4f} +/- {fit['A_err']:.4f}, "
                  f"T {fit['T']:.1f} +/- {fit['T_err']:.1f} us, "
                  f"tau {fit['tau']:.2f} +/- {fit['tau_err']:.2f} us, eta x duty peaks "
                  f"at {fit['L_peak']:.1f} us" for scan, fit in length_fits.items()]
    else:
        lines += [
            f"ceiling              kappa_c/kappa * eta_qubit = {kappa_c / kappa:.3f}"
            f" * {eta_qubit[first]:.3f} = {ceiling[first]:.3f}  (T1 {t1_window:.2f} us "
            f"({t1_src}), ring-up {tau_match:.2f} us, T_d {length_b:g} us)",
            f"efficiency           window-dynamics fit per {SWEEP_SHORT[sweep]}, T and tau held as above",
            f"duty cycle           eta_D = {duty0:.3f}  ({length_b:g} us armed of "
            f"{cycle0:.1f} us, reset_mode {mode!r})",
        ]
    lines += [
        f"shared fit           vg {vg:.3f}, ve {ve:.3f}, sigma {sigma:.3f}, "
        f"separation {abs(ve - vg) / sigma:.2f} sigma",
        f"                     t_ratio {t_ratio:.4f}"
        + ("  ** AT ITS BOUND -- fractions not trustworthy **"
           if at_bound(t_ratio) else ""),
    ]
    if is_f.any():
        lines.append(f"                     t_ratio_f {t_ratio_f:.4f}"
                     + ("  ** AT ITS BOUND -- f-readout fractions not "
                        "trustworthy **" if at_bound(t_ratio_f) else ""))
    lines.append("")
    where = "pump gain" if sweep == "gain_p" else SWEEP_SHORT[sweep]
    for scan, table in efficiencies.items():
        measured = [row for row in table if np.isfinite(row[1])]
        if not measured:
            continue
        value, slope, err, _ = max(measured, key=lambda row: row[1])
        lines.append(f"best efficiency ({scan})  {100 * slope:.2f} +/- "
                     f"{100 * err:.2f} %  at {where} {fmt(value).strip()}  "
                     f"({slope / ceiling[value]:.0%} of ceiling)")
    for scan in nep_watt:
        if not np.isfinite(nep_watt[scan]).any():
            continue
        index = int(np.nanargmin(nep_watt[scan]))
        lines.append(f"best NEP ({scan})         "
                     f"{nep_watt[scan][index] * 1e21:.2f} zW/rtHz  at {where} "
                     f"{fmt(sweep_values[index]).strip()}  (dark P(e) "
                     f"{100 * dark_rows[index][1]:.2f} %, n_equiv "
                     f"{nep[scan][index]:.4f})")
    thin = [fmt(row[sweep]).strip() for row in summary_rows
            if row["slope_points"] <= 2]
    if thin:
        lines.append(f"\nonly 2 points in the slope at {where} {', '.join(thin)}: "
                     "the line fits them exactly, so their error bar is 0 and "
                     "says nothing about the uncertainty.")
    lines += [
        "",
        ("fwm_summary.png   efficiency | efficiency x duty, vs buffer length" if by_length
         else "fwm_summary.png   efficiency | dark count | noise-equivalent n_bar | NEP"),
        "fwm_summary.csv   the same, one row per " + where,
        "full detail       ../fit_shared_means_grid.png, ../fit_shared_means.csv",
        "../qc_nep_summary.png, if present, is the acquisition script's live",
        "reduction from thresholded fractions -- indicative only; quote these.",
    ]
    (summary_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n  wrote summary/ (fwm_summary.png, fwm_summary.csv, summary.txt)")

    # --- figure 3: every histogram at one pump power ----------------------
    # Or, for a frequency sweep, at one buffer frequency: --frequency-b, else
    # the one where the buffer+pump efficiency peaked.
    if sweep == "gain_p":
        example = args.pump_gain
        example_text = f"pump gain {example}"
        out_name = f"fit_shared_means_pump{example:g}.png"
    else:
        example = args.frequency_b if sweep == "frequency_b" else args.length_b
        if example is None and "e" in efficiencies:
            example = max(efficiencies["e"], key=lambda row: row[1])[0]
        elif example is None:
            example = float(np.median(sweep_values))
        if sweep == "frequency_b":
            example_text = f"buffer {example:.4f} MHz"
            out_name = f"fit_shared_means_freq{example:.4f}.png"
        else:
            example_text = f"buffer length {example:g} us"
            out_name = f"fit_shared_means_length{example:g}.png"
    at_power = sorted((r for r in records if np.isclose(r[sweep], example)),
                      key=lambda r: r["gain_b"])
    if not at_power:
        raise SystemExit(
            f"No records at {sweep_label} {example}. "
            f"Available: {sweep_values}"
        )
    columns = [scan for scan in EXAMPLE_SCANS if scan in scans]
    print(f"\nper-histogram figure at {example_text}: "
          f"{len(at_power)} buffer gains x {len(columns)} scans")

    # One row per scan, one column per buffer gain: reading along a row shows
    # the excited blob filling in as photons arrive, and the two rows sit above
    # each other for a direct e-vs-f comparison at the same buffer gain.
    fig3, axs3 = plt.subplots(
        len(columns), len(at_power),
        figsize=(2.7 * len(at_power), 3.0 * len(columns)),
        sharex=True, sharey=True, squeeze=False,
    )
    for row, scan in enumerate(columns):
        for col, record in enumerate(at_power):
            axis = axs3[row][col]
            p = by_key[(record["index"], scan)]
            # The f row is fit with its own decay ratio, so draw it with the
            # matching shapes or the overlay would not be the curve that was
            # actually fit.
            gnd, exc = (ground_f, excited_f) if scan == "f" else (ground, excited)
            axis.semilogy(centers, record["hist"][scan], lw=1,
                          color=SCAN_COLORS[scan])
            axis.semilogy(centers, (1 - p) * gnd + p * exc, "k--", lw=1)
            axis.semilogy(centers, (1 - p) * gnd, ":", color="0.6", lw=0.8)
            axis.semilogy(centers, p * exc, ":", color="0.6", lw=0.8)
            axis.set_ylim(1e-5, 1)
            axis.tick_params(labelsize=10)
            axis.text(0.04, 0.06, f"p = {p:.4f}", transform=axis.transAxes,
                      fontsize=9)
            if row == 0:
                arrivals = photons_incident(record["gain_b"],
                                            record["length_b"], coeffs,
                                            fit_type, kappa, kappa_c)
                axis.set_title(f"buffer {record['gain_b']:.3f}\n"
                               f"{arrivals:.1f} photons",
                               fontsize="medium")
    for axis in axs3[-1]:
        axis.set_xlabel("I (ADC levels)")
    for row, scan in enumerate(columns):
        axs3[row][0].set_ylabel(f"{SCAN_LABELS[scan]}\nProb. density")
    fig3.suptitle(
        f"{example_text}    "
        f"shared vg={vg:.3f}, ve={ve:.3f}, sigma={sigma:.3f}\n"
        "dashed = fit, dotted = components",
        fontsize="medium",
    )
    fig3.tight_layout()
    out3 = HERE / out_name
    fig3.savefig(out3, dpi=150)
    print(f"  wrote {out3.name}")
    print("  wrote fit_shared_means_examples.png, fit_shared_means_grid.png")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
