"""
Stark/PNS and Ramsey buffer photon calibrations, side by side on one footing.

Every number is on-resonance photons per gain^2: photons for a buffer drive at
f_g = device.readout.frequency[0], the quantity the PNS comb measures directly
and the FWM analyses consume. Each run is reported at its own buffer
attenuation and pulse shape, printed alongside; compare runs that match.

Ramsey (T2PhotonCal) runs give two independent slopes vs gain^2: the
dephasing rate 1/T2 and the Ramsey frequency shift. Each converts to photons
at f_g with `buffer_dephasing_per_photon` (t2.py), which needs kappa and chi.
Their ratio depends only on kappa (given chi), so the pair also yields a
self-consistent kappa. Stark runs are refit with the PNS comb from
refit_two_peak.py, with both line normalizations.

Run from this folder:

    python compare_photon_cal.py --stark 497 --ramsey 333 371 500 --exclude 500:3-7

--exclude RUN:i,j-k drops Ramsey points by index (e.g. a qubit-frequency
excursion). Writes compare_photon_cal.png/.txt under
Data/<run>/metadata_and_images/photon_cal_comparison/.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SMPD_DIR = HERE.parent
REPO_DIR = SMPD_DIR.parent.parent
if str(REPO_DIR.parent) not in sys.path:
    sys.path.insert(0, str(REPO_DIR.parent))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from slab_qick_calib.exp_handling.datamanagement import SlabFile  # noqa: E402
from slab_qick_calib.experiments.single_qubit.t2 import (  # noqa: E402
    buffer_dephasing_per_photon,
    ramsey_kappa_from_ratio,
)
import refit_two_peak as pns  # noqa: E402

DEFAULT_RUN = "run34_Waymar"


def find_h5(run_dir, number):
    """The single h5 of experiment #number in a Data/run<N>_<Device> folder."""
    folders = sorted((run_dir / "data").glob(f"#{number}_*"))
    if len(folders) != 1:
        raise SystemExit(f"Expected one data folder for #{number}, found {folders}")
    files = sorted(folders[0].glob("*.h5"))
    if len(files) != 1:
        raise SystemExit(f"Expected one h5 in {folders[0]}, found {files}")
    return files[0]


def load(path):
    with SlabFile(str(path), "r") as h5:
        data = {key: h5[key][()] for key in h5.keys()}
        cfg = h5.load_config()
    return data, cfg


def buffer_setting(cfg, pulse):
    """(attenuation dB, pulse shape) actually applied to the buffer DAC."""
    soc = cfg["hw"]["soc"]
    applied = soc["rfboard_active"]["dac_readout"][str(soc["dacs"]["readout"]["ch"][0])]
    return float(applied["attn1"]) + float(applied["attn2"]), pulse


def parse_excludes(items):
    """['500:3-7', '371:2'] -> {500: {3,4,5,6,7}, 371: {2}}"""
    out = {}
    for item in items or []:
        run, _, spec = item.partition(":")
        idx = set()
        for part in spec.split(","):
            lo, _, hi = part.partition("-")
            idx.update(range(int(lo), int(hi or lo) + 1))
        out.setdefault(int(run), set()).update(idx)
    return out


def line_fit(x, y, yerr=None):
    """Weighted y = slope*x + offset; returns (slope, slope_err, offset)."""
    w = None if yerr is None else 1 / np.asarray(yerr)
    popt, pcov = np.polyfit(x, y, 1, w=w, cov="unscaled" if w is not None else True)
    if w is not None:
        # Scale by reduced chi^2 only when it is > 1, so a good fit keeps its
        # statistical error and a scattered one is not over-trusted.
        resid = (y - np.polyval(popt, x)) * w
        pcov = pcov * max(1.0, float(resid @ resid) / max(len(x) - 2, 1))
    return float(popt[0]), float(np.sqrt(pcov[0, 0])), float(popt[1])


def programmed_ramsey_freq(expt, t2):
    """
    T2PhotonCal's Ramsey frequency at each point, for runs that predate it
    being stored (data["ramsey_freq"]). It starts from the inner T2Experiment's span/ramsey_freq and after each
    point shrinks span to min(3 T2, span) with ramsey_freq = 4.5/span.
    """
    span, freq = float(expt["span"]), float(expt["ramsey_freq"])
    out = []
    for value in t2:
        out.append(freq)
        if np.isfinite(value) and value > 0:
            span = min(3 * value, span)
            freq = 4.5 / span
    return np.array(out)


def ramsey_run(run_dir, number, exclude):
    path = find_h5(run_dir, number)
    data, cfg = load(path)
    expt, readout = cfg["expt"], cfg["device"]["readout"]
    gains = np.asarray(data["gain_b_pts"], dtype=float)
    t2 = np.asarray(data["t2"], dtype=float)
    fit_freq = np.asarray(data["fit_pars"], dtype=float)[:, 1]
    rf = (np.asarray(data["ramsey_freq"]) if "ramsey_freq" in data
          else programmed_ramsey_freq(expt, t2))
    keep = np.isfinite(t2) & (t2 > 0)
    if "reference" in data:
        keep &= ~np.asarray(data["reference"]).astype(bool)
    keep[list(exclude)] = False
    run = {
        "number": number,
        "gains": gains,
        "t2": t2,
        "gamma": 1 / t2,
        "gamma_err": np.asarray(data["t2_err"]) / t2 ** 2,
        "shift": fit_freq - rf,  # MHz; sign convention-dependent
        "keep": keep,
        "f_g": float(readout["frequency"][0]),
        "detuning": float(expt["frequency_b"]) - float(readout["frequency"][0]),
        "chi_cfg": float(readout["chi"][0]),
        "kappa_cfg": float(readout["kappa"][0]),
        "setting": buffer_setting(cfg, expt["pulse_type_b"]),
    }
    g2 = gains[keep] ** 2
    run["slope_gamma"], run["slope_gamma_err"], run["gamma0"] = line_fit(
        g2, run["gamma"][keep], run["gamma_err"][keep])
    run["slope_shift"], run["slope_shift_err"], run["shift0"] = line_fit(
        g2, run["shift"][keep])
    return run


def ramsey_photons(run, kappa, chi):
    """On-resonance photons/gain^2 from the dephasing and from the shift."""
    gamma_pp, shift_pp = buffer_dephasing_per_photon(kappa, chi, run["detuning"])
    shift_ang = 2 * np.pi * abs(run["slope_shift"])
    shift_ang_err = 2 * np.pi * run["slope_shift_err"]
    return (run["slope_gamma"] / gamma_pp, run["slope_gamma_err"] / gamma_pp,
            shift_ang / abs(shift_pp), shift_ang_err / abs(shift_pp))


def stark_run(run_dir, number, normalization, n_max=20):
    path = find_h5(run_dir, number)
    data, cfg = load(path)
    expt = cfg["expt"]
    xpts = np.asarray(data["xpts"])
    gains = np.asarray(data["stark_gain"])
    p_e = np.asarray(data["p_e"])
    chi_buffer = float(expt["chi_buffer"])
    with contextlib.redirect_stdout(io.StringIO()):
        per_row = pns.refit_per_row(xpts, p_e, chi_buffer)
        fit = pns.fit_pns(xpts, p_e, per_row, chi_buffer, n_max,
                          normalization=normalization)
    good, _ = pns.flag_bad_rows(fit, n_max)
    n_bar, n_err = fit["n_bar"][good], fit["n_bar_err"][good]
    g2 = gains[good] ** 2
    ok = np.isfinite(n_err) & (n_err > 0)
    c2, c2_err, c0 = line_fit(g2[ok], n_bar[ok], n_err[ok])
    return {
        "number": number,
        "normalization": normalization,
        "gains": gains[good],
        "n_bar": n_bar,
        "n_err": n_err,
        "c2": c2,
        "c2_err": c2_err,
        "c0": c0,
        "chi_full": -2 * fit["chi"],  # config sign convention: chi[0] < 0
        "chi_full_err": 2 * fit["shared_err"]["chi"],
        "chi_prime": fit["chi_prime"],
        "kappa_pns": 2 * fit["delta_gamma"],  # n-th line HWHM grows by kappa/2
        "kappa_pns_err": 2 * fit["shared_err"]["delta_gamma"],
        "setting": buffer_setting(cfg, expt["stark_pulse_type"]),
        "probe_gain": float(expt["gain"]),
    }


def saturation_run(run_dir, number):
    """
    Probe saturation from a 17b QubitSpecPower map at zero buffer drive.

    A driven line's width goes as w^2 = w0^2 (1 + s) with s proportional to
    probe power, so a fit of HWHM^2 against gain^2 gives w0 and s(gain).
    Uses whichever quadrature's Lorentzian fits have the smallest median
    relative width error.
    """
    data, cfg = load(find_h5(run_dir, number))
    gains = np.asarray(data["ypts"], dtype=float)
    best = None
    for key in ("avgi", "avgq", "amps"):
        if f"fit_{key}" not in data:
            continue
        pars, errs = np.asarray(data[f"fit_{key}"]), np.asarray(data[f"fit_err_{key}"])
        hwhm = np.abs(pars[:, 3])
        hwhm_err = np.sqrt(np.abs(errs[:, 3, 3])) if errs.ndim == 3 else np.abs(errs[:, 3])
        score = np.nanmedian(hwhm_err / hwhm)
        if best is None or score < best[0]:
            best = (score, key, hwhm, hwhm_err, pars[:, 1])
    _, key, hwhm, hwhm_err, height = best
    # A width past half the scan is a failed fit, not a line
    span = np.ptp(np.asarray(data["xpts"]))
    ok = np.isfinite(hwhm) & np.isfinite(hwhm_err) & (hwhm_err > 0) & (hwhm < span / 2)
    k, _, w0_sq = line_fit(gains[ok] ** 2, hwhm[ok] ** 2, 2 * hwhm[ok] * hwhm_err[ok])
    return {"number": number, "key": key, "gains": gains, "hwhm": hwhm,
            "hwhm_err": hwhm_err, "height": height, "w0": np.sqrt(max(w0_sq, 0)),
            "s": k * gains ** 2 / w0_sq, "k_over_w0sq": k / w0_sq,
            "attn": cfg["hw"]["soc"]["rfboard_active"]["dac_qubit"][
                str(cfg["hw"]["soc"]["dacs"]["qubit"]["ch"][cfg["expt"]["qubit"][0]])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=DEFAULT_RUN, help="Data/<run> folder name")
    parser.add_argument("--stark", type=int, nargs="*", default=[])
    parser.add_argument("--ramsey", type=int, nargs="*", default=[])
    parser.add_argument("--exclude", nargs="*", help="RUN:i,j-k Ramsey points to drop")
    parser.add_argument("--saturation", type=int, nargs="*", default=[],
                        help="17b probe-saturation run(s) to report s vs probe gain")
    parser.add_argument("--chi", type=float, help="full g->e buffer shift (MHz), "
                        "default the first Stark run's PNS value")
    parser.add_argument("--kappa", type=float, help="buffer FWHM (MHz), default config")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    run_dir = SMPD_DIR / "Data" / args.run
    out_dir = run_dir / "metadata_and_images" / "photon_cal_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    excludes = parse_excludes(args.exclude)

    starks = [stark_run(run_dir, n, norm) for n in args.stark for norm in ("height", "area")]
    ramseys = [ramsey_run(run_dir, n, excludes.get(n, set())) for n in args.ramsey]

    log = io.StringIO()

    def out(line=""):
        print(line)
        log.write(line + "\n")

    out("On-resonance photons per gain^2 (drive at f_g), each at its own buffer setting\n")
    for s in starks:
        out(f"Stark #{s['number']} [{s['normalization']:6s}] probe gain {s['probe_gain']:.3g}, "
            f"buffer {s['setting'][0]:.0f} dB {s['setting'][1]}:  "
            f"c2 = {s['c2']:.1f} +/- {s['c2_err']:.1f}   (c0 = {s['c0']:.3f})")
        out(f"    PNS: chi_full = {s['chi_full']:.4f} +/- {s['chi_full_err']:.4f} MHz, "
            f"chi' = {s['chi_prime']:.4f}, kappa from line broadening = "
            f"{s['kappa_pns']:.3f} +/- {s['kappa_pns_err']:.3f} MHz")

    chi_pns = starks[0]["chi_full"] if starks else None
    for r in ramseys:
        chis = {"config": r["chi_cfg"]}
        if args.chi is not None:
            chis["--chi"] = args.chi
        elif chi_pns is not None:
            chis["PNS"] = chi_pns
        kappa0 = args.kappa if args.kappa is not None else r["kappa_cfg"]
        dropped = np.flatnonzero(~r["keep"])
        out(f"\nRamsey #{r['number']}  drive f_g{r['detuning']:+.4f} MHz, buffer "
            f"{r['setting'][0]:.0f} dB {r['setting'][1]}, dropped points {list(dropped)}")
        out(f"    d(1/T2)/dg^2 = {r['slope_gamma']:.2f} +/- {r['slope_gamma_err']:.2f} /us, "
            f"d(shift)/dg^2 = {r['slope_shift'] * 1e3:.0f} +/- {r['slope_shift_err'] * 1e3:.0f} kHz")
        ratio = r["slope_gamma"] / (2 * np.pi * abs(r["slope_shift"]))
        ratio_err = ratio * np.hypot(r["slope_gamma_err"] / r["slope_gamma"],
                                     r["slope_shift_err"] / abs(r["slope_shift"]))
        for chi_name, chi in chis.items():
            k_self = ramsey_kappa_from_ratio(ratio, chi, r["detuning"], kappa0)
            k_band = [ramsey_kappa_from_ratio(ratio + s * ratio_err, chi, r["detuning"], kappa0)
                      for s in (-1, 1)]
            out(f"    chi {chi_name:6s} = {chi:.4f} MHz  self-consistent kappa = {k_self:.3f} "
                f"[{min(k_band):.3f}, {max(k_band):.3f}] MHz")
            for k_name, kappa in (("config", kappa0), ("self", k_self)):
                if not np.isfinite(kappa):
                    continue
                n_g, n_g_err, n_s, n_s_err = ramsey_photons(r, kappa, chi)
                out(f"        kappa {k_name:6s} {kappa:.3f}:  from 1/T2 {n_g:7.1f} +/- {n_g_err:5.1f}"
                    f"   from shift {n_s:7.1f} +/- {n_s_err:5.1f}")

    for number in args.saturation:
        sat = saturation_run(run_dir, number)
        out(f"\nProbe saturation #{number} ({sat['key']}, qubit DAC attn "
            f"{sat['attn']['attn1']}/{sat['attn']['attn2']}): unsaturated HWHM w0 = "
            f"{1e3 * sat['w0']:.1f} kHz")
        out("    gain     HWHM (kHz)    s = (w/w0)^2 - 1    height")
        for g, w, we, sv, h in zip(sat["gains"], sat["hwhm"], sat["hwhm_err"],
                                   sat["s"], sat["height"]):
            out(f"    {g:.4f}   {1e3 * w:6.1f} +/- {1e3 * we:4.1f}    {sv:8.3f}          {h:.3g}")
        for target in (1.0, 0.3, 0.1):
            out(f"    s = {target:g} at probe gain {np.sqrt(target / sat['k_over_w0sq']):.4f}")

    (out_dir / "compare_photon_cal.txt").write_text(log.getvalue(), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax_gamma, ax_shift, ax_stark = axes
    for i, r in enumerate(ramseys):
        color = f"C{i}"
        g2 = r["gains"] ** 2
        dense = np.linspace(0, g2.max(), 50)
        k = r["keep"]
        ax_gamma.errorbar(g2[k], r["gamma"][k], yerr=r["gamma_err"][k], fmt="o", ms=4,
                          color=color, label=f"#{r['number']}")
        ax_gamma.plot(g2[~k], r["gamma"][~k], "o", ms=4, mfc="none", color=color)
        ax_gamma.plot(dense, r["slope_gamma"] * dense + r["gamma0"], "-", lw=1, color=color)
        shift = (r["shift"] - r["shift0"]) * 1e3
        ax_shift.plot(g2[k], shift[k], "o", ms=4, color=color, label=f"#{r['number']}")
        ax_shift.plot(g2[~k], shift[~k], "o", ms=4, mfc="none", color=color)
        ax_shift.plot(dense, r["slope_shift"] * dense * 1e3, "-", lw=1, color=color)
    ax_gamma.set_xlabel("Buffer gain$^2$")
    ax_gamma.set_ylabel("$1/T_2$ ($\\mu$s$^{-1}$)")
    ax_shift.set_xlabel("Buffer gain$^2$")
    ax_shift.set_ylabel("Ramsey frequency shift (kHz)")
    for s in starks:
        marker = "o" if s["normalization"] == "area" else "s"
        line = ax_stark.errorbar(s["gains"] ** 2, s["n_bar"], yerr=s["n_err"], fmt=marker,
                                 ms=4, mfc="none" if marker == "s" else None,
                                 label=f"#{s['number']} {s['normalization']}: "
                                       f"{s['c2']:.0f}/gain$^2$")
        dense = np.linspace(0, (s["gains"] ** 2).max(), 50)
        ax_stark.plot(dense, s["c2"] * dense + s["c0"], "-", lw=1, color=line[0].get_color())
    ax_stark.set_xlabel("Buffer gain$^2$")
    ax_stark.set_ylabel("PNS $\\bar{n}$")
    for axis in axes:
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(out_dir / "compare_photon_cal.png", dpi=150)
    print(f"\nWrote {out_dir}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
