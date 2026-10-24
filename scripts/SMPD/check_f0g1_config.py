"""Check the f0-g1 reset config and operating point, with no hardware.

Reads the config and reports:
  1. the hw.soc.dacs.f0g1 section shape, against the other DAC sections
  2. the design frequency f_ge + f_ef - f_waste next to device.f0g1.frequency
  3. every stored fit, evaluated over the gain range, so a stale or nonsense
     one is visible before a run rather than after
  4. the reset rate Gamma over the measured g~ range, and where it reaches
     kappa/3 -- i.e. whether the operating point we are aiming for is
     reachable with the drive powers we actually measured

Run it after each of the eight calibration scripts writes to device.f0g1.

    python check_f0g1_config.py [--config <name>.yml] [--qubit 1]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from slab_qick_calib.experiments.smpd.f0g1_reset import (  # noqa: E402
    f0g1_design_frequency,
    optimal_omega_ef,
    reset_rate,
)
from slab_qick_calib.helpers import config  # noqa: E402

DAC_KEYS = ("ch", "type", "nyquist", "filter_type", "filter_fc", "filter_bw",
            "attn1", "attn2")
FIT_FIELDS = {
    "stark_fit": ("f0-g1 frequency", "quadratic", "MHz"),
    "ef_stark_fit": ("e-f frequency", "quadratic", "MHz"),
    "omega_ef_fit": ("Omega_ef / 2pi", "linear", "MHz"),
    "g_tilde_fit": ("g~ / 2pi", "linear", "MHz"),
}


def _finite_pair(value):
    """A stored [a, c] pair, or None if it is absent or not yet fitted."""
    if value is None or len(value) != 2:
        return None
    pair = [float(v) for v in value]
    return pair if all(np.isfinite(pair)) else None


def check_dac_section(cfg, qi):
    print("== hw.soc.dacs.f0g1 ==")
    dacs = cfg.hw.soc.dacs
    if "f0g1" not in dacs:
        print("  MISSING. The reset drive has no DAC section; add one "
              "mirroring hw.soc.dacs.pump.")
        return False
    sec = dacs.f0g1
    missing = [k for k in DAC_KEYS if k not in sec]
    if missing:
        print(f"  missing keys: {missing}")
    n_expected = len(dacs.qubit.ch)
    for k in DAC_KEYS:
        if k in sec and len(sec[k]) != n_expected:
            print(f"  {k} has {len(sec[k])} entries, expected {n_expected}")
    ch = sec.ch[qi]
    used = {name: dacs[name].ch[qi] for name in dacs if name != "f0g1"}
    clash = [n for n, c in used.items() if c == ch]
    print(f"  qubit {qi}: DAC ch {ch}, type {sec.type[qi]}, "
          f"nyquist {sec.nyquist[qi]}, mixer_freq {sec.get('mixer_freq', ['-'] * (qi + 1))[qi]}")
    print(f"  filter {sec.filter_type[qi]} (fc {sec.filter_fc[qi]}, "
          f"bw {sec.filter_bw[qi]}), attn {sec.attn1[qi]}/{sec.attn2[qi]} dB")
    if clash:
        print(f"  CLASH: ch {ch} is also {', '.join(clash)}")
    else:
        print(f"  ch {ch} is not shared with {sorted(used)}")
    return not (missing or clash)


def check_frequency(cfg, qi, readout_line_qi):
    print("\n== f0-g1 frequency ==")
    design = f0g1_design_frequency(cfg, qi, readout_line_qi)
    print(f"  f_ge {cfg.device.qubit.f_ge[qi]:.4f} + "
          f"f_ef {cfg.device.qubit.f_ef[qi]:.4f} - "
          f"f_readout[{readout_line_qi}] "
          f"{cfg.device.readout.frequency[readout_line_qi]:.4f}")
    print(f"  design          = {design:.4f} MHz")
    stored = cfg.device.get("f0g1", {}).get("frequency", None)
    if stored is None or not np.isfinite(stored):
        print("  measured        = not yet measured (run 27_f0g1_spectroscopy)")
    else:
        print(f"  measured        = {float(stored):.4f} MHz "
              f"({float(stored) - design:+.4f} from design)")
    # A sub-GHz carrier is outside the ADMV8818's 2-18 GHz range, so the RF
    # board filter has to be bypassed or the tone never reaches the sample.
    ftype = cfg.hw.soc.dacs.get("f0g1", {}).get("filter_type", [None])[qi]
    if design < 2000 and ftype != "bypass":
        print(f"  WARNING: carrier is {design:.0f} MHz but filter_type is "
              f"'{ftype}'. The RF board filter spans 2-18 GHz; it must be "
              "'bypass' at this frequency.")
    return design


def check_fits(cfg, gain_max):
    print("\n== stored fits ==")
    section = cfg.device.get("f0g1", {})
    gains = np.linspace(0, gain_max, 5)
    out = {}
    for field, (label, form, unit) in FIT_FIELDS.items():
        pair = _finite_pair(section.get(field, None))
        if pair is None:
            print(f"  {field:<14} not yet fitted")
            continue
        a, c = pair
        design = gains**2 if form == "quadratic" else gains
        vals = a * design + c
        out[field] = pair
        arrow = "gain^2" if form == "quadratic" else "gain"
        print(f"  {field:<14} {label} = {a:.4f}*{arrow} + {c:.4f} {unit}")
        print(f"                 at gain {np.array2string(gains, precision=2)}"
              f" -> {np.array2string(vals, precision=3)}")
    return out


def check_operating_point(cfg, fits, gain_max, readout_line_qi):
    print("\n== reset rate ==")
    kappa = cfg.device.get("f0g1", {}).get("kappa", None)
    source = "device.f0g1.kappa"
    if kappa is None or not np.isfinite(kappa):
        kappa = cfg.device.readout.kappa[readout_line_qi]
        source = f"device.readout.kappa[{readout_line_qi}]"
    kappa = float(kappa)
    print(f"  kappa = {kappa:.4f} MHz (from {source}) -> best Gamma = "
          f"kappa/3 = {kappa / 3:.4f} MHz")

    g_fit = fits.get("g_tilde_fit")
    if g_fit is None:
        print("  g_tilde_fit not yet fitted; run 34_f0g1_rabi_vs_power to see "
              "which drive powers reach the plateau.")
        return
    m, b = g_fit
    g_at_max = m * gain_max + b
    print(f"  g~ reaches {g_at_max:.4f} MHz at gain {gain_max}")
    print(f"  {'g~':>8} {'Omega_ef opt':>13} {'Gamma':>9} {'Gamma/(kappa/3)':>17}")
    plateau_g = None
    for g in np.linspace(max(g_at_max / 8, 1e-3), g_at_max, 8):
        omega, gamma = optimal_omega_ef(g, kappa)
        print(f"  {g:8.4f} {omega:13.4f} {gamma:9.4f} {gamma / (kappa / 3):17.3f}")
        if plateau_g is None and gamma > 0.99 * kappa / 3:
            plateau_g = (g, omega)
    if plateau_g is None:
        print("  Gamma does not reach kappa/3 over this g~ range -- more f0-g1 "
              "drive power is needed.")
    else:
        g, omega = plateau_g
        gain_f0g1 = (g - b) / m if m else float("nan")
        print(f"  plateau from g~ = {g:.4f} MHz (f0g1 gain {gain_f0g1:.4f}), "
              f"Omega_ef = {omega:.4f} MHz")
        o_fit = fits.get("omega_ef_fit")
        if o_fit is not None:
            om, ob = o_fit
            gain_ef = (omega - ob) / om if om else float("nan")
            print(f"  -> e-f gain {gain_ef:.4f} from omega_ef_fit")
        else:
            print("  -> e-f gain unknown; run 30_ef_rabi_vs_power")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="2026_09_21_bfg33_smpd_v2_waymar_rfboard.yml",
                        help="name in configs/, or a path to any yml "
                             "(e.g. a saved per-experiment copy)")
    parser.add_argument("--qubit", type=int, default=1)
    parser.add_argument("--readout-line", type=int, default=None,
                        help="readout index of the reset resonator (default: --qubit)")
    parser.add_argument("--gain-max", type=float, default=1.0)
    args = parser.parse_args()

    QI = args.qubit
    READOUT_LINE_QI = args.qubit if args.readout_line is None else args.readout_line

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = REPO / "configs" / args.config
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {args.config}")
    cfg = config.load(str(cfg_path))
    print(f"{cfg_path.name}, qubit {QI}, reset resonator = readout {READOUT_LINE_QI}\n")

    check_dac_section(cfg, QI)
    check_frequency(cfg, QI, READOUT_LINE_QI)
    fits = check_fits(cfg, args.gain_max)
    check_operating_point(cfg, fits, args.gain_max, READOUT_LINE_QI)
