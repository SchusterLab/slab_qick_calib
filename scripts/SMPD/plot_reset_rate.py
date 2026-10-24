"""Figure 3 of Magnard et al. (arXiv:1801.07689), for this device's calibration.

Panel (a) is their reset-rate map Gamma(g~, Omega_ef) from the Eq. 1
non-Hermitian Hamiltonian, with the red dashed line marking the Omega_ef that
maximises Gamma at each g~. Panel (b) follows that line: Gamma rises, then
abruptly saturates at the kappa/3 ceiling.

Everything comes from device.f0g1, so re-run it after any calibration change:

    kappa          the waste linewidth              (34_f0g1_rabi_vs_power)
    g_tilde_fit    g~/2pi   = m*gain + b            (34_f0g1_rabi_vs_power)
    omega_ef_fit   Omega_ef = m*gain + b            (30_ef_rabi_vs_power)

The shaded band is the g~ this device can actually reach as the f0-g1 gain runs
0 -> max_gain. Whether that band overlaps the plateau is the whole question.
It did not on run 33 until the f0-g1 line was de-attenuated; since then
g~ = 0.824 MHz at full gain and Gamma sits at 99.9% of the kappa/3 ceiling.

Two figures are written:

  reset_rate_figure3.png   the paper's Fig. 3 for this calibration
  reset_rate_measured.png  our measured g~ and Omega_ef vs their own gains,
                           on a shared y axis, plus the operating point on
                           the Eq.-1 map

    python plot_reset_rate.py [--config <name>.yml] [--qubit 1]
"""

import argparse
import re
import sys
from pathlib import Path

import h5py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from slab_qick_calib.helpers import config  # noqa: E402
from slab_qick_calib.helpers.f0g1 import (  # noqa: E402
    optimal_omega_ef,
    reset_rate,
)

# One hue, light -> dark, for a magnitude field; matches the paper's greyscale.
CMAP = "Greys"
ACCENT = "#C1440E"      # the optimum curve, readable on grey
BAND = "#4C6EF5"        # this device's reachable range
INK = "#33322E"


# The e-f drive is ~15x stronger per unit gain than the f0-g1 drive, so the two
# rate panels below deliberately keep separate x axes (their gain ranges differ
# by 20x) but share y, which is the comparison that matters.
EF = ACCENT      # e-f / Omega_ef, throughout
F0G1 = BAND      # f0-g1 / g~, throughout


def _latest_data(run_dir, pattern):
    """Highest-numbered ``#<n>_<pattern>*`` folder's h5, or None."""
    hits = []
    for d in (run_dir / "data").glob("#*_" + pattern + "*"):
        m = re.match(r"#(\d+)_", d.name)
        h5 = sorted(d.glob("*.h5"))
        if m and h5:
            hits.append((int(m.group(1)), h5[0]))
    return max(hits)[1] if hits else None


def _read(h5, keys):
    if h5 is None:
        return None
    with h5py.File(h5, "r") as f:
        if not all(k in f for k in keys):
            return None
        return {k: np.array(f[k]).astype(float) for k in keys}


def figure_rates(cfg, run_dir, kappa, out):
    """
    The measured drive rates, and where they put us on the reset-rate map.

    (a) g~ vs f0-g1 gain and (b) Omega_ef vs e-f gain, on a shared y axis so
    the ~15x difference in drive strength per unit gain is the first thing
    visible -- it is why the operating e-f gain is a few percent while the
    f0-g1 gain sits at full scale. (c) the Eq.-1 rate map with the operating
    point, so the two panels above connect to an actual reset rate.
    """
    f0 = cfg.device.get("f0g1", {})
    gm, gb = (float(v) for v in f0["g_tilde_fit"])
    om_m, om_b = (float(v) for v in f0["omega_ef_fit"])
    max_gain = float(cfg.device.qubit.max_gain)

    g_dat = _read(_latest_data(run_dir, "f0g1_rabi_vs_power"),
                  ("gain_f0g1_pts", "g_tilde", "g_tilde_keep", "g_tilde_r2"))
    o_dat = _read(_latest_data(run_dir, "ef_rabi_vs_power"),
                  ("gain_pts", "omega_ef", "omega_ef_keep", "omega_ef_r2"))

    g_hi = gm * max_gain + gb
    om_at_hi, gam_at_hi = optimal_omega_ef(g_hi, kappa)
    ef_gain_op = (om_at_hi - om_b) / om_m

    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(14, 4.2),
        gridspec_kw={"width_ratios": [1, 1, 1.35], "wspace": 0.30},
    )
    axb.sharey(axa)

    # ---- (a) g~ vs f0-g1 gain ---------------------------------------------
    x = np.linspace(0, max_gain, 100)
    axa.plot(x, gm * x + gb, "-", color=F0G1, lw=2, zorder=2)
    if g_dat is not None:
        keep = g_dat["g_tilde_keep"] > 0
        axa.plot(g_dat["gain_f0g1_pts"][keep], g_dat["g_tilde"][keep], "o",
                 ms=7, color=F0G1, mec="white", mew=1.2, zorder=3)
        if (~keep).any():
            axa.plot(g_dat["gain_f0g1_pts"][~keep], g_dat["g_tilde"][~keep],
                     "x", ms=7, color="#999690", zorder=3)
    axa.plot([max_gain], [g_hi], "*", ms=17, color=F0G1, mec="white", mew=1.4,
             zorder=4)
    axa.annotate("operating\n$\\tilde{g}$ = %.3f MHz" % g_hi,
                 xy=(max_gain, g_hi), xytext=(-8, -42),
                 textcoords="offset points", ha="right", fontsize=8.5,
                 color=INK, arrowprops=dict(arrowstyle="-", color=F0G1, lw=1))
    axa.set_xlabel("f0-g1 drive gain (DAC level)", color=INK)
    axa.set_ylabel("Rabi rate / $2\\pi$ (MHz)", color=INK)
    axa.set_xlim(0, max_gain * 1.02)
    axa.text(0.03, 0.95,
             "$\\tilde{g}/2\\pi = %.3f\\,$gain $%+.3f$" % (gm, gb),
             transform=axa.transAxes, fontsize=9, color=F0G1, va="top",
             fontweight="bold")

    # ---- (b) Omega_ef vs e-f gain -----------------------------------------
    if o_dat is not None:
        ef_max = max(o_dat["gain_pts"].max(), ef_gain_op) * 1.18
    else:
        ef_max = ef_gain_op * 2
    xe = np.linspace(0, ef_max, 100)
    axb.plot(xe, om_m * xe + om_b, "-", color=EF, lw=2, zorder=2)
    if o_dat is not None:
        keep = o_dat["omega_ef_keep"] > 0
        axb.plot(o_dat["gain_pts"][keep], o_dat["omega_ef"][keep], "o", ms=7,
                 color=EF, mec="white", mew=1.2, zorder=3)
        if (~keep).any():
            axb.plot(o_dat["gain_pts"][~keep], o_dat["omega_ef"][~keep], "x",
                     ms=7, color="#999690", zorder=3)
    axb.plot([ef_gain_op], [om_at_hi], "*", ms=17, color=EF, mec="white",
             mew=1.4, zorder=4)
    axb.annotate("operating\n$\\Omega_{ef}$ = %.3f MHz\nat gain %.4f"
                 % (om_at_hi, ef_gain_op),
                 xy=(ef_gain_op, om_at_hi), xytext=(-12, -52),
                 textcoords="offset points", ha="right", fontsize=8.5,
                 color=INK,
                 arrowprops=dict(arrowstyle="-", color=EF, lw=1))
    axb.set_xlabel("e-f drive gain (DAC level)", color=INK)
    axb.set_xlim(0, ef_max)
    axb.text(0.03, 0.95,
             "$\\Omega_{ef}/2\\pi = %.2f\\,$gain $%+.4f$" % (om_m, om_b),
             transform=axb.transAxes, fontsize=9, color=EF, va="top",
             fontweight="bold")
    # The headline comparison: same y axis, ~20x different x axis.
    axb.text(0.97, 0.06,
             "%.0f$\\times$ stronger per unit gain\nthan $\\tilde{g}$ (panel a)"
             % (om_m / gm), transform=axb.transAxes, ha="right", fontsize=8.5,
             color=INK, style="italic")

    # ---- (c) Gamma vs both rates ------------------------------------------
    ceiling = kappa / 3
    g_ax = np.linspace(1e-4, max(2.2 * g_hi, 2.5 * ceiling), 200)
    o_ax = np.linspace(1e-4, max(2.2 * om_at_hi, 1.2 * ceiling), 190)
    gamma = np.array([[reset_rate(gg, oo, kappa) for gg in g_ax]
                      for oo in o_ax])
    mesh = axc.pcolormesh(g_ax, o_ax, gamma, cmap=CMAP, vmin=0, vmax=ceiling,
                          shading="auto", rasterized=True)
    cb = fig.colorbar(mesh, ax=axc, pad=0.02)
    cb.set_label("$\\Gamma/2\\pi$ (MHz)", color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK, length=3)
    opt = np.array([optimal_omega_ef(gg, kappa) for gg in g_ax])
    axc.plot(g_ax, opt[:, 0], "--", color=EF, lw=2,
             label="$\\Omega_{ef}^{\\mathrm{opt}}(\\tilde{g})$")
    if o_dat is not None:
        axc.axhspan(o_dat["omega_ef"].min(), o_dat["omega_ef"].max(),
                    color=EF, alpha=0.10, lw=0,
                    label="$\\Omega_{ef}$ measured (30)")
    axc.axvspan(gb, g_hi, color=F0G1, alpha=0.13, lw=0,
                label="$\\tilde{g}$ measured (34)")
    axc.plot([g_hi], [om_at_hi], "*", ms=18, color="#F5C518", mec=INK,
             mew=1.2, zorder=5,
             label="operating: $\\Gamma$ = %.3f MHz (%.0f%% of $\\kappa/3$)"
                   % (gam_at_hi, gam_at_hi / ceiling * 100))
    axc.set_xlabel("f0-g1 Rabi rate, $\\tilde{g}/2\\pi$ (MHz)", color=INK)
    axc.set_ylabel("e-f Rabi rate, $\\Omega_{ef}/2\\pi$ (MHz)", color=INK)
    axc.set_xlim(0, g_ax[-1])
    axc.set_ylim(0, o_ax[-1])
    axc.legend(loc="upper left", fontsize=7.5, framealpha=0.93)

    for ax in (axa, axb, axc):
        ax.tick_params(colors=INK, length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#BFBDB5")
    for ax in (axa, axb):
        ax.grid(axis="y", color="#E8E6E1", lw=0.8)
        ax.set_axisbelow(True)

    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")

    def _prov(dat, keep_key, r2_key):
        if dat is None:
            return "  (no data found, fit line only)"
        return "  (%d points, min R2 %.2f)" % (
            int((dat[keep_key] > 0).sum()), dat[r2_key].min())

    print("  g~/2pi       = %.4f*gain %+.4f MHz%s"
          % (gm, gb, _prov(g_dat, "g_tilde_keep", "g_tilde_r2")))
    print("  Omega_ef/2pi = %.4f*gain %+.4f MHz%s"
          % (om_m, om_b, _prov(o_dat, "omega_ef_keep", "omega_ef_r2")))
    print("  the e-f drive is %.1fx stronger per unit gain" % (om_m / gm))
    if o_dat is not None:
        lo, hi = o_dat["omega_ef"].min(), o_dat["omega_ef"].max()
        verdict = ("INSIDE the calibrated range" if lo <= om_at_hi <= hi
                   else "OUTSIDE it -- extrapolating, so scan the e-f gain")
        print("  operating Omega_ef = %.4f MHz, measured %.4f-%.4f -> %s"
              % (om_at_hi, lo, hi, verdict))
    print("  wrote %s" % out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config",
                    default="2026_09_21_bfg33_smpd_v2_waymar_rfboard.yml")
    ap.add_argument("--qubit", type=int, default=1)
    ap.add_argument("--out", default="reset_rate_figure3.png")
    ap.add_argument("--out-measured", default="reset_rate_measured.png")
    ap.add_argument("--data-dir", default=None,
                    help="run folder holding data/; defaults to the one the "
                         "config name implies (Data/run<N>_<Device>)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = REPO / "configs" / args.config
    cfg = config.load(str(cfg_path))
    qi = args.qubit
    f0 = cfg.device.get("f0g1", {})

    kappa = f0.get("kappa", None)
    if kappa is None or not np.isfinite(kappa):
        kappa = float(cfg.device.readout.kappa[qi])
        kappa_src = f"device.readout.kappa[{qi}]"
    else:
        kappa = float(kappa)
        kappa_src = "device.f0g1.kappa"
    ceiling = kappa / 3

    gm, gb = (float(v) for v in f0["g_tilde_fit"])
    om_m, om_b = (float(v) for v in f0["omega_ef_fit"])
    max_gain = float(cfg.device.qubit.max_gain)

    # What this device can reach as the f0-g1 gain runs 0 -> max_gain.
    g_lo, g_hi = gb, gm * max_gain + gb
    om_at_hi, gam_at_hi = optimal_omega_ef(g_hi, kappa)

    # Plot out to where the plateau is clearly established, so the gap between
    # what is reachable and what is needed is visible rather than cropped out.
    g_max_plot = max(4 * g_hi, 3 * ceiling)
    om_max_plot = max(3 * om_at_hi, 1.2 * ceiling)

    g_axis = np.linspace(1e-4, g_max_plot, 260)
    om_axis = np.linspace(1e-4, om_max_plot, 240)
    gamma = np.array([[reset_rate(g, om, kappa) for g in g_axis]
                      for om in om_axis])
    opt = np.array([optimal_omega_ef(g, kappa) for g in g_axis])
    om_opt, gam_opt = opt[:, 0], opt[:, 1]

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11, 4.3),
        gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.32},
    )

    # ---- (a) the Gamma map -------------------------------------------------
    mesh = axa.pcolormesh(g_axis, om_axis, gamma, cmap=CMAP,
                          vmin=0, vmax=ceiling, shading="auto", rasterized=True)
    cb = fig.colorbar(mesh, ax=axa, pad=0.02)
    cb.set_label("$\\Gamma/2\\pi$ (MHz)", color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=INK, length=3)

    axa.plot(g_axis, om_opt, "--", color=ACCENT, lw=2,
             label="$\\Omega_{ef}^{\\mathrm{opt}}(\\tilde{g})$")
    axa.axvspan(g_lo, g_hi, color=BAND, alpha=0.16, lw=0)
    axa.axvline(g_hi, color=BAND, lw=1.5)
    axa.plot([g_hi], [om_at_hi], "o", ms=9, color=BAND, mec="white", mew=1.5,
             label=f"this device at full gain\n"
                   f"$\\tilde{{g}}$={g_hi:.3f}, $\\Omega_{{ef}}$={om_at_hi:.3f}")
    axa.set_xlabel("f0-g1 Rabi rate, $\\tilde{g}/2\\pi$ (MHz)", color=INK)
    axa.set_ylabel("e-f Rabi rate, $\\Omega_{ef}/2\\pi$ (MHz)", color=INK)
    axa.set_xlim(0, g_max_plot)
    axa.set_ylim(0, om_max_plot)
    axa.legend(loc="upper left", fontsize=8, framealpha=0.92)

    # Gain is what actually gets set, so give it its own axis. Affine in g~,
    # so the mapping is exact rather than approximate.
    top = axa.secondary_xaxis(
        "top", functions=(lambda g: (g - gb) / gm, lambda x: gm * x + gb))
    top.set_xlabel("f0-g1 drive gain (DAC level)", color=INK, fontsize=9)
    top.tick_params(colors=INK, labelsize=8, length=3)

    # ---- (b) Gamma along the optimum --------------------------------------
    axb.axhline(ceiling, ls=":", color=INK, lw=1.5,
                label=f"$\\kappa/3$ = {ceiling:.3f} MHz")
    axb.plot(g_axis, gam_opt, "-", color=ACCENT, lw=2,
             label="$\\Gamma$ at $\\Omega_{ef}^{\\mathrm{opt}}$")
    axb.axvspan(g_lo, g_hi, color=BAND, alpha=0.16, lw=0,
                label="reachable $\\tilde{g}$ (gain 0 - %g)" % max_gain)
    axb.axvline(g_hi, color=BAND, lw=1.5)
    axb.plot([g_hi], [gam_at_hi], "o", ms=9, color=BAND, mec="white", mew=1.5)
    axb.annotate(
        f"{gam_at_hi:.3f} MHz\n{gam_at_hi / ceiling * 100:.0f}% of $\\kappa/3$",
        xy=(g_hi, gam_at_hi), xytext=(g_hi + 0.06 * g_max_plot, gam_at_hi),
        fontsize=9, color=INK, va="center",
        arrowprops=dict(arrowstyle="-", color=BAND, lw=1.2),
    )
    axb.set_xlabel("f0-g1 Rabi rate, $\\tilde{g}/2\\pi$ (MHz)", color=INK)
    axb.set_ylabel("reset rate, $\\Gamma/2\\pi$ (MHz)", color=INK)
    axb.set_xlim(0, g_max_plot)
    axb.set_ylim(0, ceiling * 1.15)
    axb.legend(loc="lower right", fontsize=8, framealpha=0.92)

    for ax in (axa, axb):
        ax.tick_params(colors=INK, length=3)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#BFBDB5")
    axb.grid(axis="y", color="#E8E6E1", lw=0.8)
    axb.set_axisbelow(True)

    out = Path(args.out)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")

    print(f"{cfg_path.name}, qubit {qi}")
    print(f"  kappa/2pi          = {kappa:.4f} MHz  (from {kappa_src})")
    print(f"  ceiling kappa/3    = {ceiling:.4f} MHz")
    print(f"  g~ reachable       = {g_lo:.4f} - {g_hi:.4f} MHz "
          f"(gain 0 - {max_gain:g})")
    print(f"  at full gain       : Omega_ef_opt = {om_at_hi:.4f} MHz "
          f"-> e-f gain {(om_at_hi - om_b) / om_m:.5f} with the fit, "
          f"{om_at_hi / om_m:.5f} through the origin")
    print(f"                       Gamma = {gam_at_hi:.4f} MHz, "
          f"{gam_at_hi / ceiling * 100:.0f}% of the ceiling")
    # Where the plateau actually starts, i.e. what g~ would be needed.
    reach = np.flatnonzero(gam_opt > 0.99 * ceiling)
    if len(reach):
        g_need = g_axis[reach[0]]
        print(f"  plateau needs g~  >= {g_need:.4f} MHz, i.e. {g_need / g_hi:.1f}x "
              f"more f0-g1 coupling than full gain gives now")
    else:
        print("  plateau not reached anywhere in the plotted range")
    print(f"  wrote {out}")

    # ---- the measured-rates figure ----------------------------------------
    if args.data_dir:
        run_dir = Path(args.data_dir)
    else:
        from common import run_and_device_from_config
        run, device = run_and_device_from_config(cfg_path.name)
        run_dir = Path(__file__).resolve().parent / "Data" / f"run{run}_{device}"
    print("")
    print(f"measured rates (from {run_dir.name})")
    if not (run_dir / "data").is_dir():
        print(f"  no data/ under {run_dir}, skipping the measured figure")
    else:
        figure_rates(cfg, run_dir, kappa, Path(args.out_measured))


if __name__ == "__main__":
    main()
