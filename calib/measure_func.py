"""
Measurement functions for calibration.

Standalone routines built on the experiment classes: dispersive shift
(chi) measurement via excited-state resonator spectroscopy and qubit
temperature measurement via EF Rabi contrast.
"""

import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
import seaborn as sns
from pathlib import Path


from ..helpers import config
from ..experiments.single_qubit.resonator_spectroscopy import ResSpec
from ..experiments.single_qubit.rabi import RabiExperiment

colors = ["#0869c8", "#b51d14"]


def check_chi(cfg_dict, qi=0, span=7, df=-0.5, npts=301, plot=False, check_f=False):
    """
    Measures the chi shift of a qubit.
    This is done by measuring the resonator frequency with and without a pi pulse on the qubit.
    The difference between these two frequencies is the chi shift.

    Parameters
    ----------
    cfg_dict : dict
        The configuration dictionary.
    qi : int
        The qubit index.
    span : float
        The frequency span of the resonator spectroscopy.
    npts : int
        The number of points in the resonator spectroscopy.
    plot : bool
        Whether to plot the results.
    check_f : bool
        Whether to also measure the resonator frequency with a pi pulse on the f state.

    Returns
    -------
    tuple
        A tuple containing the experiment objects and the chi value.
    """
    auto_cfg = config.load(cfg_dict["cfg_file"])
    freq = auto_cfg["device"]["readout"]["frequency"][qi]
    kappa = auto_cfg["device"]["readout"]["kappa"][qi]
    t1 = auto_cfg["device"]["qubit"]["T1"][qi]
    final_delay = t1*5
    start = freq - span/2 + df
    center = start + span / 2 + df/2
    chi = ResSpec(
        cfg_dict,
        qi=qi,
        params={
            "span": span,
            "center": center,
            "npts": npts,
            "rounds": 2,
            "final_delay": final_delay,
            "pulse_e": True,
        },
        go=False,
    )
    chi.go(analyze=True, display=False, progress=True, save=True)

    if check_f:
        chif =ResSpec(
            cfg_dict,
            qi=qi,
            params={
                "span": span,
                "center": center,
                "npts": npts,
                "rounds": 3,
                "final_delay": final_delay,
                "pulse_e": True,
                "pulse_f": True,
            },
            go=False,
        )
        chif.go(analyze=True, display=False, progress=True, save=True)

    rspec = ResSpec(
        cfg_dict,
        qi=qi,
        params={
            "span": span,
            "center": center,
            "npts": npts,
            "rounds": 2,
            "final_delay": 15,
        },
        go=False,
    )
    rspec.go(analyze=True, display=False, progress=True, save=True)

    if "mixer_freq" in chi.cfg.hw.soc.dacs.readout:
        xpts_chi = chi.cfg.hw.soc.dacs.readout.mixer_freq + chi.data["xpts"]
        xpts_res = rspec.cfg.hw.soc.dacs.readout.mixer_freq + chi.data["xpts"]
        rspec.data["fit"][0] + chi.cfg.hw.soc.dacs.readout.mixer_freq
        chi.data["fit"][0] + chi.cfg.hw.soc.dacs.readout.mixer_freq
    else:
        xpts_chi = chi.data["xpts"]
        xpts_res = rspec.data["xpts"]

    arg = np.argmin(chi.data["amps"])
    #arg = np.argmin(np.abs(np.abs(chi.data["amps"])) - np.abs(rspec.data["amps"]))
    arg2 = np.argmin(np.abs(rspec.data["amps"]))
    chi.data["rval"] = rspec.data["xpts"][arg2]
    chi.data["cval"] = chi.data["xpts"][arg]
    chi_val = xpts_chi[arg] - xpts_res[arg2]
    chi.data["chi_val"] = chi_val
    chi.data["freq_opt"] = rspec.data["xpts"][arg]

    fig, ax = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    ax[0].set_title(f"Chi Measurement Q{qi}")
    ax[0].plot(xpts_res, rspec.data["amps"], label="No Pulse")
    ax[0].plot(xpts_chi, chi.data["amps"], label=f"e Pulse")

    cap = f"$\chi=${chi_val:0.2f} MHz"
    ax[0].text(
        0.04,
        0.35,
        cap,
        transform=ax[0].transAxes,
        fontsize=12,
        verticalalignment="bottom",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.8),
    )
    ax[0].legend()
    ax[0].axvline(
        x=xpts_chi[arg], color="k", linestyle="--"
    )  # Add vertical line at selected point
    ax[0].axvline(x=xpts_res[arg2], color="k", linestyle="--")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Frequency (MHz)")

    ax[1].plot(xpts_res, rspec.data["amps"] - chi.data["amps"])
    ax[1].axvline(x=xpts_chi[arg], color="k", linestyle="--")
    ax[1].axvline(x=xpts_res[arg2], color="k", linestyle="--")
    ax[1].set_ylabel("Difference")
    ax[1].set_xlabel("Frequency (MHz)")

    ax[2].plot(xpts_res[1:-1], rspec.data["phase_fix"])
    ax[2].plot(xpts_chi[1:-1], chi.data["phase_fix"], label="e Pulse")

    if check_f:
        ax[0].plot(chif.data["xpts"], chif.data["amps"], label="f pulse")
        ax[2].plot(chif.data["xpts"][1:-1], chif.data["phase_fix"], label="f pulse")

    for a in ax:
        a.set_xlabel("Frequency (MHz)")

    ax[2].set_ylabel("Phase")
    plt.show()

    file_path = Path(rspec.fname)
    fig.savefig(file_path.parent.parent / "images" / (file_path.stem + "_chi.png"))
    return (
        [chi, rspec],
        chi_val,
    )


def measure_temp(cfg_dict, qi, temp=40, expts=32, reps=None, chan=None, bf_client=None, no_fit=False, measure_f=False):
    """
    Measures the residual population of an excited state via Rabi contrast.

    With measure_f=False (default), measures the thermal e population: compares
    an e-f Rabi oscillation preceded by a g-e pi pulse (full contrast, population
    starts entirely in e) against the same e-f Rabi with no prep pulse at all
    (oscillation amplitude then reflects only the thermal e population already
    present).

    With measure_f=True, measures the (much smaller) thermal f population the same
    way, one level up: an f-h Rabi oscillation preceded by a g-e-f prep chain (full
    contrast) against the same f-h Rabi with no prep at all (thermal f population
    only).

    Parameters
    ----------
    cfg_dict : dict
        The configuration dictionary.
    qi : int
        The qubit index.
    temp : float
        Guess for temperature, used to set the number of rounds.
    expts : int
        The number of experiments to run. Fewer experiments will yield a faster result.
    reps : int, optional
        Base number of repetitions (averages) per point. The full-contrast prep
        run uses reps; the no-prep (thermal) run uses 40*reps, preserving the
        style="temp" boost since the thermal signal is small. If None (default),
        each experiment uses its own default reps.
    chan : int
        The channel to use for the measurement.
    bf_client : object, optional
        Bluefors client for reading mixing chamber temperature.
    measure_f : bool, optional
        If True, measure the f-state population (via f-h Rabi + g-e-f prep/shelving)
        instead of the e-state population (via e-f Rabi + g-e prep/shelving).
        Default False.

    Returns
    -------
    tuple
        A tuple containing the qubit temperature and the population of the excited state.
    """
    # Read MXC temperature at the beginning if bf_client is provided
    auto_cfg = config.load(cfg_dict["cfg_file"])
    max_gain = auto_cfg["device"]["qubit"]["pulses"]["pi_ef"]["gain"][qi]*5.2
    mxc_temp = np.nan
    if bf_client is not None:
        try:
            mxc_temp = bf_client.get_mxc_temperature() * 1000  # Convert K to mK
        except Exception as e:
            print(f"Warning: failed to read MXC temperature: {e}")

    transition_flag = "checkFH" if measure_f else "checkEF"
    prep_label = "gef Pulse" if measure_f else "ge Pulse"
    noprep_label = "No gef Pulse" if measure_f else "No ge Pulse"

    prep_params = {"pulse_ge": True, "max_gain": max_gain, "expts": expts, transition_flag: True}
    if reps is not None:
        prep_params["reps"] = reps
    rabief = RabiExperiment(cfg_dict, qi=qi, params=prep_params)

    nopulse_params = {
        "expts": expts,
        "pulse_ge": False,
        "temp": temp,
        "max_gain": max_gain,
        transition_flag: True,
    }
    if reps is not None:
        # Preserve the style="temp" 40x boost (rabi.py) relative to the chosen base,
        # since the no-prep thermal signal is small and needs the extra averaging.
        nopulse_params["reps"] = 40 * reps
    rabief_nopulse = RabiExperiment(cfg_dict, qi=qi, params=nopulse_params, style="temp")

    # To measure temperature, use fewer points to get more signal more quickly
    if measure_f:
        transition_freq = 1e6 * rabief.cfg.device.qubit.f_fh[qi]
    else:
        transition_freq = 1e6 * rabief.cfg.device.qubit.f_ef[qi]
    if chan is None:
        chan = str(rabief_nopulse.data["i_best"])[2:-1]
    if no_fit:
        # Raw max-min range ratio -- biased by noise (the small no-prep oscillation's
        # range is inflated by noise), kept only as a fallback.
        rng_ge = np.max(rabief.data[chan]) - np.min(rabief.data[chan])
        rng_noge = np.max(rabief_nopulse.data[chan]) - np.min(rabief_nopulse.data[chan])
        population = rng_noge / (rng_ge + rng_noge)
    else:
        # Excited fraction from the fitted Rabi amplitudes (sinfunc yscale = best_fit[0]),
        # robust to noise. Sum-normalized so population = P_e/(P_e+P_g), which is what the
        # temperature formula below expects (population/(1-population) = P_e/P_g).
        amp_ge = abs(rabief.data["best_fit"][0])            # e-f Rabi amplitude WITH g-e prep
        amp_noge = abs(rabief_nopulse.data["best_fit"][0])  # e-f Rabi amplitude, no prep (thermal)
        population = amp_noge / (amp_ge + amp_noge)

    qubit_temp = -1e3 * cs.h * transition_freq / (cs.k * np.log(population/(1-population)))

    fig, ax = plt.subplots(1, 1)
    ax.plot(
        rabief.data["xpts"],
        rabief.data[chan] - np.min(rabief.data[chan]),
        "o-",
        label=prep_label,
    )
    ax.set_ylabel(prep_label)
    axt = plt.twinx()
    axt.plot(
        rabief_nopulse.data["xpts"],
        rabief_nopulse.data[chan] - np.min(rabief_nopulse.data[chan]),
        "o-",
        label=noprep_label,
        color=colors[1],
    )
    axt.tick_params(axis="y", colors=colors[1])
    ax.tick_params(axis="y", colors=colors[0])
    ax.yaxis.label.set_color(colors[0])
    axt.yaxis.label.set_color(colors[1])
    axt.set_xlabel("Gain (DAC units)")
    axt.set_ylabel(noprep_label)

    pop_label = "F Population" if measure_f else "Population"
    # Add MXC temperature to title if available
    if np.isnan(mxc_temp):
        title = f"Qubit {qi} Temperature: {qubit_temp:0.2f} mK, {pop_label}: {population:0.2g}"
    else:
        title = f"Qubit {qi} Temperature: {qubit_temp:0.2f} mK, {pop_label}: {population:0.2g}, MXC: {mxc_temp:.1f} mK"
    ax.set_title(title)

    file_path = Path(rabief.fname)
    metadata_path = Path(cfg_dict.get("metadata_path", file_path.parent.parent / "images"))
    metadata_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(metadata_path / (file_path.stem + "_temp.png"))
    plt.show()

    return qubit_temp, population, rabief_nopulse, rabief
