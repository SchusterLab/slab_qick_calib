"""Measure the dispersive shift chi from resonator spectroscopy in |g> and |e>."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

from common import finish_experiment, parse_args, prepare_experiment, wait_for_plots


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    # The buffer (line 0) has no qubit of its own; its qubit entries are stale, so the
    # pi pulse comes from the real qubit's line (f_ge, pi_ge, T1)
    QUBIT_QI = 1
    JPA_ON = False
    RESET_MODE = "passive"  # "passive" | "active" | "f0g1"; settings in config, CLAUDE.md "Reset modes"
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON
    )

    # Needs a tuned-up pi pulse. The window is centred between the two expected resonances
    # using the config's chi, so either sign is in range (measure_func.check_chi only covers chi < 0).
    span = 15
    df = 0  # extra nudge, if the window still needs shifting
    expts = 301
    frequency = auto_cfg.device.readout.frequency[qi]
    chi_expected = auto_cfg.device.readout.chi[qi]
    center = frequency + chi_expected / 2 + df
    final_delay = 5 * auto_cfg.device.qubit.T1[QUBIT_QI]

    # Short readout so |e> barely decays during it (T1 ~60 us); ring-up is ~1/kappa
    scan = {"span": span, "center": center, "expts": expts, "rounds": 2,
            "length": 5, "reset_mode": RESET_MODE}
    low, high = center - span / 2, center + span / 2
    print(
        f"Chi scan: {low:.3f} to {high:.3f} MHz ({expts} points), centred on the "
        f"|g> resonance at {frequency:.3f} MHz and the |e> one expected at "
        f"{frequency + chi_expected:.3f} MHz (chi={chi_expected:.4f} in config)"
    )
    for state, f_res in (("|g>", frequency), ("|e> expected", frequency + chi_expected)):
        if not low <= f_res <= high:
            print(
                f"Warning: the {state} resonance at {f_res:.3f} MHz lies outside "
                f"the window, so the measured chi will be wrong. Widen span or "
                f"nudge df."
            )

    # Excited state first: a pi pulse, then the resonator sweep. final_delay is
    # long so the qubit fully relaxes between shots.
    chi_expt = meas.ResSpec(
        cfg_dict,
        qi=qi,
        go=False,
        params={**scan, "final_delay": final_delay, "pulse_e": True},
    )
    qubit = chi_expt.cfg.device.qubit
    qubit.f_ge[qi] = qubit.f_ge[QUBIT_QI]
    for key in qubit.pulses.pi_ge:
        qubit.pulses.pi_ge[key][qi] = qubit.pulses.pi_ge[key][QUBIT_QI]
    finish_experiment(chi_expt, args, wait=False)

    # Ground-state reference, same window, no pulse.
    rspec = meas.ResSpec(
        cfg_dict, qi=qi, go=False, params={**scan, "final_delay": 15}
    )
    finish_experiment(rspec, args, wait=False)

    # f_g from the |g> scan's complex fit. The pi-pulsed scan holds both resonances
    # (decay, imperfect pi), so f_e is the strongest line > 0.5 MHz (~2 kappa) from f_g.
    readout_dacs = auto_cfg.hw.soc.dacs.readout
    offset = readout_dacs.mixer_freq[qi] if "mixer_freq" in readout_dacs else 0.0
    xpts_e = chi_expt.data["xpts"] + offset
    xpts_g = rspec.data["xpts"] + offset

    def line_center(xpts, amps, mask):
        """Lorentzian centre of the largest |S21| excursion (dip or peak) inside mask."""
        dev = np.where(mask, np.abs(amps - np.median(amps)), -np.inf)
        i = int(np.argmax(dev))
        near = mask & (np.abs(xpts - xpts[i]) < 0.8)
        lor = lambda f, f0, w, a, b: b + a / (1 + ((f - f0) / (w / 2)) ** 2)
        try:
            popt, _ = curve_fit(lor, xpts[near], amps[near],
                                p0=[xpts[i], 0.25, amps[i] - np.median(amps), np.median(amps)])
            return float(popt[0])
        except RuntimeError:
            return float(xpts[i])

    # The |g> scan's amplitude can show no clear line, so fall back to the config, not to it
    if "complex_fit" in rspec.data:
        f_g = float(rspec.data["complex_fit"][0])
    else:
        f_g = float(frequency)
    f_e = line_center(xpts_e, chi_expt.data["amps"], np.abs(xpts_e - f_g) > 0.5)
    f_g_check = line_center(xpts_e, chi_expt.data["amps"], np.abs(xpts_e - f_e) > 0.5)
    chi_val = f_e - f_g
    print(f"Resonator: |g> {f_g:.4f} MHz, |e> {f_e:.4f} MHz -> chi = {chi_val:.4f} MHz "
          f"(residual |g> line in the pi-pulsed scan at {f_g_check:.4f})")

    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax[0].plot(xpts_g, rspec.data["amps"], label="no pulse (|g>)")
    ax[0].plot(xpts_e, chi_expt.data["amps"], label="e pulse (|e>)")
    for axis in ax:
        axis.axvline(f_g, color="k", ls="--", lw=1)
        axis.axvline(f_e, color="k", ls="--", lw=1)
    ax[0].text(
        0.04,
        0.35,
        rf"$\chi=${chi_val:0.3f} MHz",
        transform=ax[0].transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax[0].set_ylabel("Amplitude (ADC level)")
    ax[0].legend(fontsize="small")

    ax[1].plot(xpts_g, rspec.data["amps"] - chi_expt.data["amps"])
    ax[1].set_ylabel("Difference")

    # phase_fix is two points shorter than xpts (it comes from a difference).
    ax[2].plot(xpts_g[1:-1], rspec.data["phase_fix"], label="no pulse (|g>)")
    ax[2].plot(xpts_e[1:-1], chi_expt.data["phase_fix"], label="e pulse (|e>)")
    ax[2].set_ylabel("Phase (rad)")
    ax[2].set_xlabel("Frequency (MHz)")
    ax[2].legend(fontsize="small")

    fig.tight_layout()
    chi_fig = Path(cfg_dict["metadata_path"]) / "chi.png"
    fig.savefig(chi_fig)
    print(f"Saved {chi_fig}")

    if not args.no_update_config:
        config.update_readout(str(cfg_path), "chi", chi_val, qi)

    if not args.no_show:
        wait_for_plots()
