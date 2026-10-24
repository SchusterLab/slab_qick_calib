"""Measured reset rate at chosen points in (g~, Omega_ef), against Eq. 1.

One ResetTimeScan per point. Draws the Fig. 3a map with the optimal-Omega_ef
line, the measured points on it, and measured-vs-predicted Gamma.

Points are given as (f0-g1 gain, multiple of the optimal Omega_ef at that g~),
so "1.0" sits on the red line and anything else is a deliberate step off it.

    python 37_reset_rate_map.py --no-show
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config as cfglib
from slab_qick_calib.helpers.f0g1 import (
    ef_gain_for_omega,
    fitted_reset_rate,
    g_tilde_for_gain,
    omega_ef_for_gain,
    optimal_omega_ef,
    reset_rate,
)

from common import (
    REPO_DIR,
    finish_experiment,
    nonblocking_plots,
    parse_args,
    prepare_experiment,
    wait_for_plots,
)


def reanalyze(data_dir, out_dir, kappa, omega_ef_scale, mults):
    """
    Re-run analyze/display on a finished run's h5 files, remaking every PNG.

    Each point is reloaded with `ResetTimeScan.from_h5file`, which bypasses
    __init__ so no hardware is needed, then re-analysed and re-displayed -- so
    the per-point figures pick up whatever `analyze` now computes (the
    trajectory fit) rather than only the map being redrawn.

    `metadata_path` has to be set by hand: `from_h5file` does not set it, and
    `save_fig` would otherwise fall back to a metadata_and_images folder
    *inside* data/.
    """
    rows = []
    files = sorted(Path(data_dir).glob("*.h5"))
    if not files:
        raise SystemExit(f"no h5 files under {data_dir}")
    for i, f in enumerate(files):
        expt = meas.ResetTimeScan.from_h5file(f)
        expt.metadata_path = str(out_dir)
        print(f"\n===== {f.name}")
        expt.analyze(data=expt.data)
        with nonblocking_plots():
            expt.display(data=expt.data)
        for n in plt.get_fignums():
            plt.close(n)
        e = expt.cfg.expt
        g_t = g_tilde_for_gain(expt.cfg, e.gain_f0g1)
        om = omega_ef_scale * float(e.omega_ef_target)
        rows.append((
            g_t, om,
            fitted_reset_rate(g_t, om, kappa,
                              np.asarray(expt.data["xpts"], dtype=float)),
            float(expt.data.get("reset_gamma", np.nan)),
            float(e.gain_f0g1), float(e.gain_ef),
            mults[i] if i < len(mults) else np.nan,
        ))
    return np.array(rows, dtype=float)


def summarize(rows):
    """Per-point table and the measured/predicted spread."""
    pred, meas_g = rows[:, 2], rows[:, 3]
    ok = np.isfinite(meas_g) & (meas_g > 0)

    print("\n" + "=" * 70)
    print(f"  {'f0g1':>6}{'g~':>8}{'gain_ef':>9}{'Om_ef':>8}{'xopt':>6}"
          f"{'pred':>8}{'meas':>8}{'ratio':>7}")
    for r in rows:
        print(f"  {r[4]:>6.2f}{r[0]:>8.3f}{r[5]:>9.4f}{r[1]:>8.3f}{r[6]:>6.1f}"
              f"{r[2]:>8.4f}{r[3]:>8.4f}{r[3] / r[2] if r[2] > 0 else np.nan:>7.2f}")
    if ok.sum():
        print(f"\n  measured/predicted: mean {np.mean(meas_g[ok] / pred[ok]):.2f}, "
              f"range {np.min(meas_g[ok] / pred[ok]):.2f}-"
              f"{np.max(meas_g[ok] / pred[ok]):.2f}")


def draw_map(rows, kappa):
    """Fig. 3a map with the measured points, and measured vs predicted."""
    g_t, om, pred, meas_g = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    ok = np.isfinite(meas_g) & (meas_g > 0)
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(12.5, 4.8),
                                   gridspec_kw={"wspace": 0.28})

    # --- (a) the Eq. 1 map, the optimum line, the measured points ---------
    gx = np.linspace(1e-4, max(2.0 * g_t.max(), 2.5 * kappa / 3), 220)
    oy = np.linspace(1e-4, max(1.6 * om.max(), 1.2 * kappa / 3), 200)
    Z = np.array([[reset_rate(a, b, kappa) for a in gx] for b in oy])
    mesh = axa.pcolormesh(gx, oy, Z, cmap="Greys", vmin=0, vmax=kappa / 3,
                          shading="auto", rasterized=True)
    cb = fig.colorbar(mesh, ax=axa, pad=0.02)
    cb.set_label("asymptotic $\\Gamma/2\\pi$ (MHz)")
    opt = np.array([optimal_omega_ef(a, kappa)[0] for a in gx])
    axa.plot(gx, opt, "--", color="#C1440E", lw=2,
             label="$\\Omega_{ef}^{\\mathrm{opt}}(\\tilde{g})$")
    # The fitted rate can exceed kappa/3 -- that bounds the eigenvalue, not an
    # exponential fit of the early transient -- so the scale has to follow it.
    vtop = max(kappa / 3, float(np.nanmax(meas_g[ok])) if ok.sum() else 0.0)
    sc = axa.scatter(g_t[ok], om[ok], c=meas_g[ok], s=190, cmap="viridis",
                     vmin=0, vmax=vtop, edgecolor="white", lw=1.6,
                     zorder=5, label="measured $\\Gamma$")
    cb2 = fig.colorbar(sc, ax=axa, pad=0.02)
    cb2.set_label("measured $\\Gamma/2\\pi$ (MHz)")
    for r in rows[ok]:
        # Flip below the marker near the top edge, and sit the text on a
        # scrim: the map is near-black exactly where the points cluster.
        high = r[1] > 0.75 * oy[-1]
        axa.annotate(f"{r[3]:.2f}", xy=(r[0], r[1]),
                     xytext=(0, -22 if high else 14),
                     textcoords="offset points", ha="center", fontsize=8,
                     fontweight="bold", color="#33322E",
                     bbox=dict(boxstyle="round,pad=0.15", fc="white",
                               ec="none", alpha=0.8))
    axa.set_xlabel("f0-g1 Rabi rate, $\\tilde{g}/2\\pi$ (MHz)")
    axa.set_ylabel("e-f Rabi rate, $\\Omega_{ef}/2\\pi$ (MHz)")
    axa.set_xlim(0, gx[-1])
    axa.set_ylim(0, oy[-1])
    axa.legend(fontsize=8, loc="upper left")

    # --- (b) measured vs predicted ---------------------------------------
    lim = max(pred.max(), np.nanmax(meas_g[ok]) if ok.sum() else 0) * 1.15
    axb.plot([0, lim], [0, lim], "-", color="0.6", lw=1.4, label="y = x")
    on = rows[:, 6] == 1.0
    for sel, mk, lbl in ((ok & on, "o", "on the optimum"),
                         (ok & ~on, "s", "off the optimum")):
        if sel.sum():
            axb.plot(pred[sel], meas_g[sel], mk, ms=9, mec="white", mew=1.4,
                     color="#4C6EF5" if mk == "o" else "#2F9E44", label=lbl)
    axb.axhline(kappa / 3, ls=":", color="#33322E", lw=1.3,
                label=f"$\\kappa/3$ = {kappa / 3:.3f} (asymptotic only)")
    axb.set_xlabel("predicted $\\Gamma/2\\pi$ (MHz), Eq. 1 fit the same way")
    axb.set_ylabel("measured $\\Gamma/2\\pi$ (MHz)")
    axb.set_xlim(0, lim)
    axb.set_ylim(0, lim)
    axb.legend(fontsize=8)
    for ax in (axa, axb):
        for sd in ("top", "right"):
            ax.spines[sd].set_visible(False)
    return fig


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    RESET_MODE = "passive"      # never the protocol under test

    # Measured in #293: the e-f drive runs at this fraction of omega_ef_fit.
    # Applied here rather than written to the config, which scripts 31/32 also
    # read for their pi length. CLAUDE.md "Runner notes", 36.
    OMEGA_EF_SCALE = 0.51

    # (f0-g1 gain, Omega_ef as a multiple of the optimum at that g~).
    # With OMEGA_EF_SCALE applied, mult is a true multiple again -- before it
    # was a multiple of an optimum computed from a 2x-high fit, which is why
    # #292's "mult = 1.0" points all sat at about half the real optimum.
    #
    # Walking UP in f0-g1 amplitude. Pooling #446 and #473, measured Gamma
    # rises monotonically with f0-g1 gain at every mult over 0.15-0.55 with no
    # turnover -- at mult 1.3 it goes 0.10, 0.18, 0.25, 0.47, 0.82 and the
    # steepest step is the last. Gain 1.00 has never been measured near the
    # optimum: #287/#288 ran there at mult 0.51, about half the right Omega_ef.
    # mult 1.0-1.6 brackets the e-f optimum, which was flat-topped between 1.3
    # and 1.6 at f0g1 0.40 (0.488 and 0.483).
    POINTS = [
    # Filling the two missing mult rungs at f0-g1 0.74 -- the best-measured
    # gain, but sampled only at 1.0/1.3/1.6 where 0.80 and 0.88 both have
    # 1.15/1.3/1.45 -- plus the upper rung at 0.68.
    #
    # The 1.45 points map the CLIFF, which is the operational question: Gamma
    # goes 0.86 at mult 1.3 to 0.35 at 1.6 (f0g1 0.74), and since Omega_ef
    # drifts ~9% between runs, the margin from a setpoint to that edge decides
    # how safe the setpoint is.
    #
    # (0.74, 1.3) is the in-run anchor. Its three existing takes span 10%
    # (0.798 / 0.860 / 0.778), which is the same size as the mult 1.15-vs-1.30
    # difference being tested, so that comparison has to be internal.
    (0.74, 1.15), (0.74, 1.3), (0.74, 1.45),
    (0.68, 1.45),
    ]
    # Time window per point, bracketed over SCALE_RANGE rather than taken from
    # one prediction. CLAUDE.md "Runner notes", 37.
    # Residual uncertainty on Omega_ef now that OMEGA_EF_SCALE is applied, so
    # these bracket 1.0. #292's per-point implied scale was 0.42-0.59 against
    # the 0.51 used here, i.e. 0.82-1.16; +/-40% leaves margin on that.
    SCALE_RANGE = (0.7, 1.0, 1.4)
    N_TAU = 4.0                     # span, in tau of the slowest case
    PTS_PER_TAU = 4.0               # sampling, for the fastest case
    # LENGTH_MIN is 4.0, not 1.0: sizing the span at 4 exponential tau is not
    # enough to watch a ringing trace SETTLE, because the ripples outlast the
    # exponential. In #446 the mult-1.3 points got 1.72 us and never reached
    # floor + 1% at all, so their residual read 0.03 purely from truncation and
    # their settling time had to be extrapolated -- which is what produced the
    # wrong "time to 1% is flat down to gain 0.40" conclusion. Every point that
    # did get >= 2.4 us reached floor + 1% at 2.3-3.5 us, so 4.0 covers it.
    LENGTH_MIN, LENGTH_MAX = 4.0, 40.0
    # EXPTS_MIN rises with it. The binding timescale is now the ripple period,
    # ~1/(2*g~) = 0.29 us at f0g1 1.00, not tau -- the trajectory fit needs the
    # oscillations resolved, and that is what reads out g~, Omega_ef and kappa.
    # 130 points over 4 us is a 0.031 us step, ~9 per period. PTS_PER_TAU is
    # left as the (weaker) upper-bound criterion; the floor is what binds here.
    EXPTS_MIN, EXPTS_MAX = 130, 220
    REPS = 1500
    READOUT_GAP = 2.0           # short: a long gap lets drive heating develop

    def add_arguments(parser):
        parser.add_argument(
            "--reanalyze", metavar="DATA_DIR",
            help="re-fit a saved run and remake every PNG; no hardware",
        )

    args = parse_args(__doc__, add_arguments, config=CONFIG)

    if args.reanalyze:
        auto_cfg = cfglib.load(str(REPO_DIR / "configs" / args.config))
        data_dir = Path(args.reanalyze)
        parts = list(data_dir.parts)
        out_dir = data_dir
        if "data" in parts:
            parts[len(parts) - 1 - parts[::-1].index("data")] = "metadata_and_images"
            out_dir = Path(*parts)
        if not out_dir.is_dir():
            out_dir = data_dir
    else:
        run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
            args, __file__, qi, JPA_ON
        )
        out_dir = Path(cfg_dict["metadata_path"])
    kappa = float(auto_cfg.device.f0g1.kappa)
    print(f"\nkappa = {kappa:.4f} MHz, ceiling = {kappa / 3:.4f}\n")

    def window_for(g_t, om):
        """(span, expts) bracketing Gamma over SCALE_RANGE."""
        gams = []
        for s in SCALE_RANGE:
            gam = reset_rate(g_t, s * om, kappa)        # eigenvalue seed
            if not (np.isfinite(gam) and gam > 0):
                continue
            probe = np.linspace(0, 6 / (2 * np.pi * gam), 200)[1:]
            gam = fitted_reset_rate(g_t, s * om, kappa, probe)
            if np.isfinite(gam) and gam > 0:
                gams.append(gam)
        if not gams:
            return LENGTH_MAX, EXPTS_MAX
        span = float(np.clip(N_TAU / (2 * np.pi * min(gams)),
                             LENGTH_MIN, LENGTH_MAX))
        expts = int(np.clip(round(span * PTS_PER_TAU * 2 * np.pi * max(gams)),
                            EXPTS_MIN, EXPTS_MAX))
        return span, expts

    if args.reanalyze:
        rows = reanalyze(data_dir, out_dir, kappa, OMEGA_EF_SCALE,
                         [m for _, m in POINTS])
    else:
        rows = []
        for gain_f0g1, mult in POINTS:
            g_t = g_tilde_for_gain(auto_cfg, gain_f0g1)
            om_opt, _ = optimal_omega_ef(g_t, kappa)
            # Ask omega_ef_fit for the gain that DELIVERS mult*om_opt, then track
            # what is delivered rather than what the fit nominally claims.
            gain_ef = ef_gain_for_omega(auto_cfg, mult * om_opt / OMEGA_EF_SCALE,
                                        verbose=False)
            om = OMEGA_EF_SCALE * omega_ef_for_gain(auto_cfg, gain_ef)
            span, expts = window_for(g_t, om)
            pred = fitted_reset_rate(g_t, om, kappa,
                                     np.linspace(span / expts, span, expts))
            print(f"===== f0g1 {gain_f0g1:.2f} (g~ {g_t:.3f}), "
                  f"Omega_ef {mult:.1f}x opt -> gain_ef {gain_ef:.4f} "
                  f"(Om {om:.3f}), predicted Gamma {pred:.4f} "
                  f"(eigval {reset_rate(g_t, om, kappa):.4f}), "
                  f"window {span:.2f} us / {expts} pts")
            expt = meas.ResetTimeScan(
                cfg_dict, qi=qi, go=False,
                params={"gain_f0g1": gain_f0g1, "gain_ef": gain_ef,
                        "max_length": span, "expts": expts, "reps": REPS,
                        "readout_gap": READOUT_GAP, "preps": ["g", "e"],
                        "baseline_preps": ["g"], "reset_mode": RESET_MODE},
            )
            finish_experiment(expt, args, wait=False)
            # Re-predict on the grid actually acquired, so prediction and
            # measurement are the same quantity.
            pred = fitted_reset_rate(g_t, om, kappa,
                                     np.asarray(expt.data["xpts"], dtype=float))
            rows.append((g_t, om, pred, float(expt.data.get("reset_gamma", np.nan)),
                         gain_f0g1, gain_ef, mult))
            for n in plt.get_fignums():
                plt.close(n)

    rows = np.asarray(rows, dtype=float)
    summarize(rows)
    fig = draw_map(rows, kappa)
    # out_dir is metadata_and_images, never the data folder. Same rule as 35b.
    out = out_dir / "reset_rate_map.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")
    if not args.no_show:
        wait_for_plots(raise_windows=False)
