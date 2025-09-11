import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as cs
import seaborn as sns


from ..helpers import config
from ..experiments.single_qubit.resonator_spectroscopy import ResSpec
from ..experiments.single_qubit.rabi import RabiExperiment

colors = ["#0869c8", "#b51d14"]


def check_chi(cfg_dict, qi=0, span=7, offset=0.5, npts=301, plot=False, check_f=False):
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
    start = freq - span/2 - offset
    center = start + span / 2 + offset/2
    chi = ResSpec(
        cfg_dict,
        qi=qi,
        params={
            "span": span,
            "center": center,
            "npts": npts,
            "rounds": 5,
            "final_delay": 15,
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
                "rounds": 5,
                "final_delay": 15,
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

    arg = np.argmin(np.abs(np.abs(chi.data["amps"])) - np.abs(rspec.data["amps"]))
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

    imname = rspec.fname.split("\\")[-1]
    fig.savefig(rspec.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "_chi.png")
    return (
        [chi, rspec],
        chi_val,
    )


def measure_temp(cfg_dict, qi, temp=40, expts=20, chan=None):
    """
    Measures the temperature of a qubit.
    This is done by measuring the population of the excited state with and without a pi pulse.
    The ratio of these two populations is used to calculate the temperature.

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
    rounds : int
        The number of rounds to run.
    chan : int
        The channel to use for the measurement.

    Returns
    -------
    tuple
        A tuple containing the qubit temperature and the population of the excited state.
    """
    rabief = RabiExperiment(
        cfg_dict, qi=qi, params={"pulse_ge": True, "checkEF": True}
    )
    rabief_nopulse = RabiExperiment(
        cfg_dict,
        qi=qi,
        params={
            "expts": expts,
            "pulse_ge": False,
            "checkEF": True,
            "temp": temp,
        },
        style="temp",
    )

    # To measure temperature, use fewer points to get more signal more quickly
    fge = 1e6 * rabief.cfg.device.qubit.f_ge[qi]
    if chan is None:
        population = rabief_nopulse.data["best_fit"][0] / rabief.data["best_fit"][0]
    else:
        population = rabief_nopulse.data[chan][0] / rabief.data[chan][0]

    qubit_temp = -1e3 * cs.h * fge / (cs.k * np.log(population))

    fig, ax = plt.subplots(1, 1)
    i_best = str(rabief.data["i_best"])[2:-1]
    ax.plot(
        rabief.data["xpts"],
        rabief.data[i_best] - np.min(rabief.data[i_best]),
        label="ge Pulse",
    )
    ax.set_ylabel("ge Pulse")
    axt = plt.twinx()
    axt.plot(
        rabief_nopulse.data["xpts"],
        rabief_nopulse.data[i_best] - np.min(rabief_nopulse.data[i_best]),
        label="No ge Pulse",
        color=colors[1],
    )
    axt.tick_params(axis="y", colors=colors[1])
    ax.tick_params(axis="y", colors=colors[0])
    ax.yaxis.label.set_color(colors[0])
    axt.yaxis.label.set_color(colors[1])
    axt.set_xlabel("Gain (DAC units)")
    axt.set_ylabel("No ge Pulse")
    ax.set_title(
        f"Qubit {qi} Temperature: {qubit_temp:0.2f} mK, Population: {population:0.2g}"
    )

    imname = rabief.fname.split("\\")[-1]
    fig.savefig(
        rabief.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "_temp.png"
    )
    plt.show()

    return qubit_temp, population
