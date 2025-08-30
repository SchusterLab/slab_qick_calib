"""
Single-Shot Readout Experiment Module
=====================================

This module implements single-shot readout experiments for quantum processors.
It allows for discrimination between quantum states (g, e, and optionally f)
by collecting statistics on readout signals and analyzing their distributions.

The module contains two main classes:
- HistogramProgram: Defines the quantum pulse sequence for the experiment
- HistogramExperiment: Manages experiment execution, data acquisition, and analysis

Key features:
- Configurable readout parameters (frequency, gain, length)
- Support for ground, excited, and second excited state measurements
- Automatic data analysis with Gaussian fitting
- Visualization of readout histograms and IQ distributions
- Calculation of readout fidelity and optimal discrimination thresholds
- Support for active qubit reset and verification
"""

import matplotlib.pyplot as plt
import numpy as np
from qick import *
import copy
import seaborn as sns

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment
from ..general.qick_program import QickProgram

from ...helpers import config
from ...calib import readout_helpers as helpers

# Standard colors for plotting
BLUE = "#4053d3"  # Color for ground state
RED = "#b51d14"  # Color for excited state
GREEN = "#2ca02c"  # Color for second excited (f) state

# ====================================================== #


class HistogramProgram(QickProgram):
    """
    Quantum pulse sequence program for single-shot readout experiments.

    This class defines the pulse sequence for measuring the qubit state
    in a single shot. It can be configured to prepare the qubit in the
    ground (g), excited (e), or second excited (f) state before readout.

    Parameters
    ----------
    soccfg : dict
        SOC configuration dictionary
    final_delay : float
        Final delay time after readout
    cfg : dict
        Configuration dictionary containing experiment parameters.
        This dictionary is expected to have the following keys:
        - `expt.shots`: Number of shots in the experiment
        - `expt.frequency`: Readout frequency
        - `expt.gain`: Readout gain
        - `expt.readout_length`: Length of the readout pulse
        - `expt.qubit`: List of qubit indices (e.g., `[0]`)
        - `device.readout.phase`: Readout phase from device config
        - `device.qubit.f_ge`: Qubit g-e transition frequency
        - `expt.pulse_f` (optional): Boolean to enable pulsing to the f-state
        - `device.qubit.f_ef` (optional): Qubit e-f transition frequency
        - `expt.active_reset` (optional): Boolean to enable active reset
    """

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """
        Initialize the program with the given configuration.

        Sets up the experiment loop, readout parameters, and pulse definitions.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary
        """
        cfg = AttrDict(self.cfg)
        # Set up experiment loop for the specified number of shots
        self.add_loop("shotloop", cfg.expt.shots)

        # Configure readout parameters
        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        # Set phase based on whether active reset is enabled
        if cfg.expt.active_reset or cfg.expt.remeas:
            self.phase = cfg.device.readout.phase[cfg.expt.qubit[0]]
        else:
            self.phase = 0
        self.readout_length = cfg.expt.readout_length

        # Initialize the base program with readout configuration
        super()._initialize(cfg, readout="")

        # Define pi pulses for state preparation
        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
        if cfg.expt.pulse_f:
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

        # Add initial delay for tProc setup
        self.delay(0.5)

    def _body(self, cfg):
        """
        Define the main body of the pulse sequence.

        This includes state preparation pulses, readout pulse, and triggers.

        Parameters
        ----------
        cfg : dict
            Configuration dictionary
        """
        cfg = AttrDict(self.cfg)
        # Configure readout
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # Apply pi pulse to prepare excited state if requested
        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        # Apply second pi pulse to prepare f state if requested
        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)

        # Add small delay before readout
        self.delay_auto(t=0.01, tag="wait")

        # Apply readout pulse and trigger data acquisition
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.adc_ch], ddr4=True, pins=[0], t=self.trig_offset)

        # Perform active reset if enabled
        if cfg.expt.active_reset:
            self.reset(cfg.expt.reset)
        if cfg.expt.remeas:
            self.repeated_measurement(5)

    def reset(self, i):
        """
        Reset the qubit to ground state.

        Parameters
        ----------
        i : int
            Reset index
        """
        super().reset(i)

    def repeated_measurement(self, i):
        """
        Perform repeated measurement.

        Parameters
        ----------
        i : int
            Number of repetitions
        """
        cfg = AttrDict(self.cfg)
        for n in range(i):
            self.wait_auto(0.1)
            self.delay_auto(0.3)
            
            self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
            self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
            if self.lo_ch is not None:
                self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
            self.delay_auto(0.01)

    def collect_shots(self, offset=0):
        """
        Collect and process the raw I/Q data from the experiment.

        Parameters
        ----------
        offset : float, optional
            Offset to subtract from the raw data

        Returns
        -------
        tuple
            (i_shots, q_shots) arrays containing I and Q values for each shot
        """
        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            # Get raw IQ data
            iq_raw = self.get_raw()
            # Extract I values and flatten
            i_shots = iq_raw[i][:, :, 0, 0]
            i_shots = i_shots.flatten()
            # Extract Q values and flatten
            q_shots = iq_raw[i][:, :, 0, 1]
            q_shots = q_shots.flatten()

        return i_shots, q_shots


class HistogramExperiment(QickExperiment):
    """
    Single-shot readout experiment for quantum state discrimination.

    This class manages the execution of single-shot readout experiments,
    including data acquisition, analysis, and visualization. It can measure
    the ground (g), excited (e), and optionally second excited (f) states.

    Parameters
    ----------
    cfg_dict : dict
        Configuration dictionary
    prefix : str, optional
        Prefix for experiment name and saved files
    progress : bool, optional
        Whether to show progress during acquisition
    qi : int, optional
        Qubit index
    go : bool, optional
        Whether to run the experiment immediately
    check_f : bool, optional
        Whether to measure the second excited state
    params : dict, optional
        A dictionary of parameters to override the default values.
        If not provided, the following defaults are used:
        - `shots`: 10000
        - `reps`: 1
        - `rounds`: 1
        - `readout_length`: from device config for the specified qubit
        - `frequency`: from device config for the specified qubit
        - `gain`: from device config for the specified qubit
        - `active_reset`: `False`
        - `check_e`: `True`
        - `check_f`: as passed to `__init__`
        - `qubit`: `[qi]`
        - `qubit_chan`: from hardware config for the specified qubit
        - `ddr4`: `False`
    display : bool, optional
        Whether to display the results after acquisition
    """

    def __init__(
        self,
        cfg_dict,
        prefix=None,
        progress=True,
        qi=0,
        go=True,
        check_f=False,
        params={},
        display=True,
        print=False,
    ):
        # Set default prefix if not provided
        if prefix is None:
            prefix = f"single_shot_qubit{qi}"

        # Initialize base experiment
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        # Define default parameters
        params_def = dict(
            shots=10000,  # Number of shots per experiment
            reps=1,  # Number of repetitions
            rounds=1,  # Number of software averages
            readout_length=self.cfg.device.readout.readout_length[
                qi
            ],  # Readout pulse length
            frequency=self.cfg.device.readout.frequency[qi],  # Readout frequency
            gain=self.cfg.device.readout.gain[qi],  # Readout gain
            active_reset=False,  # Whether to use active reset
            reset=7,  # Reset index
            check_e=True,  # Whether to measure excited state
            check_f=check_f,  # Whether to measure second excited state
            qubit=[qi],  # Qubit index list
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[qi],  # Readout channel
            ddr4=False,  # Whether to use DDR4 memory
            remeas=False,  # Whether to do repeated measurement
        )

        # Merge default and user-provided parameters
        self.cfg.expt = {**params_def, **params}

        # Configure reset if active reset is enabled
        if self.cfg.expt.active_reset:
            super().configure_reset()

        if print:
            super().print()
            go = False
        # Run the experiment if requested
        if go:
            self.go(analyze=True, display=display, progress=progress, save=True)

    def acquire(self, progress=False, debug=False):
        """
        Acquire data for the single-shot experiment.

        This method collects single-shot data for the ground state and,
        if configured, the excited and second excited states.

        Parameters
        ----------
        progress : bool, optional
            Whether to show progress during acquisition
        debug : bool, optional
            Whether to print debug information

        Returns
        -------
        dict
            Dictionary containing the acquired data
        """
        data = dict()

        # Determine final delay based on configuration
        if "setup_reset" in self.cfg.expt and self.cfg.expt.setup_reset:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        elif self.cfg.expt.active_reset:
            final_delay = self.cfg.expt.readout_length
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # ----------------------------------------------------------------------
        # Ground state measurements
        # ----------------------------------------------------------------------
        # Create configuration for ground state measurement
        cfg2 = copy.deepcopy(dict(self.cfg))
        cfg = AttrDict(cfg2)
        cfg.expt.pulse_e = False
        cfg.expt.pulse_f = False

        # Create and configure histogram program
        histpro = HistogramProgram(soccfg=self.soccfg, final_delay=final_delay, cfg=cfg)

        # Configure DDR4 if enabled
        if self.cfg.expt.ddr4:
            # Each transfer (burst) is 256 decimated samples
            n_transfers = 1500000
            nt = n_transfers
            # Arm the DDR4 buffer
            self.im[self.cfg.aliases.soc].arm_ddr4(
                ch=self.cfg.expt.qubit_chan, nt=n_transfers
            )

        # Acquire ground state data
        iq_list = histpro.acquire(
            self.im[self.cfg.aliases.soc],
            threshold=None,
            progress=progress,
        )

        # Store ground state I/Q data
        data["Ig"] = iq_list[0][0][:, 0]
        data["Qg"] = iq_list[0][0][:, 1]

        # Store reset data if active reset is enabled
        if self.cfg.expt.active_reset or self.cfg.expt.remeas:
            data["Igr"] = iq_list[0][1:, :, 0]
            data["Qgr"] = iq_list[0][1:, :, 1]

        # Get DDR4 data if enabled
        if self.cfg.expt.ddr4:
            iq_ddr4 = self.im[self.cfg.aliases.soc].get_ddr4(nt)
            t = histpro.get_time_axis_ddr4(self.cfg.expt.qubit_chan, iq_ddr4)
            data["t_g"] = t
            data["iq_ddr4_g"] = iq_ddr4

        # Collect raw shots
        irawg, qrawg = histpro.collect_shots()

        # ----------------------------------------------------------------------
        # Excited state measurements
        # ----------------------------------------------------------------------
        if self.cfg.expt.check_e:
            # Create configuration for excited state measurement
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = False

            # Create and configure histogram program
            histpro = HistogramProgram(
                soccfg=self.soccfg, final_delay=final_delay, cfg=cfg
            )

            # Configure DDR4 if enabled
            if self.cfg.expt.ddr4:
                self.im[self.cfg.aliases.soc].arm_ddr4(
                    ch=self.cfg.expt.qubit_chan, nt=n_transfers
                )

            # Acquire excited state data
            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                progress=progress,
            )

            # Get DDR4 data if enabled
            if self.cfg.expt.ddr4:
                iq_ddr4 = self.im[self.cfg.aliases.soc].get_ddr4(nt)
                t = histpro.get_time_axis_ddr4(self.cfg.expt.qubit_chan, iq_ddr4)
                data["t_e"] = t
                data["iq_ddr4_e"] = iq_ddr4

            # Store excited state I/Q data
            data["Ie"] = iq_list[0][0][:, 0]
            data["Qe"] = iq_list[0][0][:, 1]

            # Collect raw shots
            irawe, qraw = histpro.collect_shots()

            # Store reset data if active reset is enabled
            if self.cfg.expt.active_reset or self.cfg.expt.remeas:
                data["Ier"] = iq_list[0][1:, :, 0]
                data["Qer"] = iq_list[0][1:, :, 1]

        # ----------------------------------------------------------------------
        # Second excited state measurements
        # ----------------------------------------------------------------------
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            # Create configuration for second excited state measurement
            cfg = AttrDict(self.cfg.copy())
            cfg.expt.pulse_e = True
            cfg.expt.pulse_f = True

            # Create and configure histogram program
            histpro = HistogramProgram(soccfg=self.soccfg, cfg=cfg)

            # Acquire second excited state data
            avgi, avgq = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                threshold=None,
                load_pulses=True,
                progress=progress,
            )

            # Store second excited state I/Q data
            data["If"], data["Qf"] = histpro.collect_shots()

        # Store data and return
        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=False, **kwargs):
        """
        Analyze the acquired single-shot data.

        This method processes the data to calculate readout fidelity,
        optimal thresholds, and fit parameters for state discrimination.

        Parameters
        ----------
        data : dict, optional
            Data dictionary to analyze (uses self.data if None)
        span : float, optional
            Span for histogram analysis
        verbose : bool, optional
            Whether to print detailed analysis information
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        dict
            Dictionary containing the analyzed data and results
        """
        if data is None:
            data = self.data

        # Perform initial histogram analysis
        params, _ = helpers.hist(data=data, plot=False, span=span, verbose=verbose)
        data.update(params)

        # Perform detailed single-shot analysis with fitting
        try:
            # Fit single-shot data
            data2, p, paramsg, paramse2 = helpers.fit_single_shot(data, plot=False)

            # Update data with fit results
            data.update(p)
            data["vhg"] = data2["vhg"]
            data["histg"] = data2["histg"]
            data["vhe"] = data2["vhe"]
            data["histe"] = data2["histe"]
            data["paramsg"] = paramsg
            data["shots"] = self.cfg.expt.shots
        except Exception as e:
            print(f"Fits failed: {str(e)}")

        return data

    def display(
        self,
        data=None,
        span=None,
        verbose=False,
        plot_e=True,
        plot_f=False,
        ax=None,
        plot=True,
        **kwargs,
    ):
        """
        Display the results of the single-shot experiment.

        This method creates visualizations of the single-shot data,
        including histograms and IQ distributions.

        Parameters
        ----------
        data : dict, optional
            Data dictionary to display (uses self.data if None)
        span : float, optional
            Span for histogram analysis
        verbose : bool, optional
            Whether to print detailed information
        plot_e : bool, optional
            Whether to plot excited state data
        plot_f : bool, optional
            Whether to plot second excited state data
        ax : list of matplotlib.axes, optional
            Axes for plotting
        plot : bool, optional
            Whether to create plots
        **kwargs : dict
            Additional keyword arguments

        Returns
        -------
        None
        """
        if data is None:
            data = self.data

        # Determine whether to save the figure
        if ax is not None:
            savefig = False
        else:
            savefig = True

        # Create histogram plots
        params, fig = helpers.hist(
            data=data,
            plot=plot,
            verbose=verbose,
            span=span,
            ax=ax,
            qubit=self.cfg.expt.qubit[0],
        )

        # Extract parameters
        fids = params["fids"]
        thresholds = params["thresholds"]
        angle = params["angle"]

        # Set experiment parameters if not already set
        if "expt" not in self.cfg:
            self.cfg.expt.check_e = plot_e
            self.cfg.expt.check_f = plot_f

        # Print detailed information if requested
        if verbose:
            print(f"ge Fidelity (%): {100*fids[0]:.3f}")

            if self.cfg.expt.check_f:
                print(f"gf Fidelity (%): {100*fids[1]:.3f}")
                print(f"ef Fidelity (%): {100*fids[2]:.3f}")
            print(f"Rotation angle (deg): {angle:.3f}")
            print(f"Threshold ge: {thresholds[0]:.3f}")
            if self.cfg.expt.check_f:
                print(f"Threshold gf: {thresholds[1]:.3f}")
                print(f"Threshold ef: {thresholds[2]:.3f}")

        # Extract image name for saving
        imname = self.fname.split("\\")[-1]

        # Show and save figure if requested
        if savefig:
            plt.show()
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )

    def update(self, freq=True, fast=False, verbose=True):
        """
        Update configuration file with the results of the experiment.

        This method updates the readout parameters in the configuration file
        based on the analysis results.

        Parameters
        ----------
        cfg_file : str
            Path to the configuration file
        freq : bool, optional
            Whether to update frequency
        fast : bool, optional
            Whether to perform a fast update (skip some parameters)
        verbose : bool, optional
            Whether to print update information

        Returns
        -------
        None
        """
        qi = self.cfg.expt.qubit[0]
        cfg_file=self.config_file

        # Update readout parameters
        config.update_readout(
            cfg_file, "phase", self.data["angle"], qi, verbose=verbose
        )
        config.update_readout(
            cfg_file, "threshold", self.data["thresholds"][0], qi, verbose=verbose
        )
        config.update_readout(
            cfg_file, "fidelity", self.data["fids"][0], qi, verbose=verbose
        )

        # Update additional parameters if not in fast mode
        if not fast:
            config.update_readout(
                cfg_file, "sigma", self.data["sigma"], qi, verbose=verbose
            )
            config.update_readout(cfg_file, "tm", self.data["tm"], qi, verbose=verbose)

            # Update qubit tuned_up status based on fidelity
            if self.data["fids"][0] > 0.07:
                config.update_qubit(cfg_file, "tuned_up", True, qi, verbose=verbose)
            else:
                config.update_qubit(cfg_file, "tuned_up", False, qi, verbose=verbose)
                print("Readout not tuned up")

    def check_reset(self):
        """
        Check the performance of active reset.

        This method analyzes and visualizes the effectiveness of active reset
        by comparing the distributions before and after reset.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # Create histograms with specified number of bins
        nbins = 75
        fig, ax = plt.subplots(2, 1, figsize=(6, 7))
        fig.suptitle(f"Q{self.cfg.expt.qubit[0]}")

        # Create ground state histogram
        vg, histg = helpers.make_hist(self.data["Ig"], nbins=nbins)
        max_g = np.max(histg)
        ax[0].semilogy(vg, histg/max_g, color=BLUE, linewidth=2)
        ax[1].semilogy(vg, histg/max_g, color=BLUE, linewidth=2)

        # Create color palette for reset histograms
        b = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(self.data["Igr"]))

        # Create excited state histogram
        ve, histe = helpers.make_hist(self.data["Ie"], nbins=nbins)
        max_e = np.max(histe)
        ax[1].semilogy(ve, histe/max_e, color=RED, linewidth=2)
        fig2, ax2 = plt.subplots(2, len(self.data["Igr"])+1, figsize=(17, 7), sharex=True, sharey=True)
        # Plot reset histograms for ground state
        ax2[0,0].plot(self.data['Ig'], self.data['Qg'], '.', markersize=1)
        ax2[1,0].plot(self.data['Ie'], self.data['Qe'], '.', markersize=1)
        for i in range(len(self.data["Igr"])):
            ax2[0,i+1].plot(self.data['Igr'][i], self.data['Qgr'][i], '.', markersize=1)
            ax2[1,i+1].plot(self.data['Ier'][i], self.data['Qer'][i], '.', markersize=1)
            v, hist = helpers.make_hist(self.data["Igr"][i], nbins=nbins)
            ax[0].semilogy(v, hist/max_g, color=b[i], linewidth=1, label=f"{i+1}")

            # Plot reset histograms for excited state
            v, hist = helpers.make_hist(self.data["Ier"][i], nbins=nbins)
            ax[1].semilogy(v, hist/max_e, color=b[i], linewidth=1, label=f"{i+1}")

        ax[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax[1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
        # Helper function to find bin index closest to a value
        def find_bin_closest_to_value(bins, value):
            return np.argmin(np.abs(bins - value))

        # Find indices for excited state level in different histograms
        ind = find_bin_closest_to_value(v, self.data["ie"])
        ind_e = find_bin_closest_to_value(ve, self.data["ie"])
        ind_g = find_bin_closest_to_value(vg, self.data["ie"])

        # Calculate reset performance metrics
        reset_level = hist[ind]
        e_level = histe[ind_e]
        g_level = histg[ind_g]

        # Print reset performance
        print(
            f"Reset is {reset_level/e_level:3g} of e and {reset_level/g_level:3g} of g"
        )

        # Store reset performance metrics
        self.data["reset_e"] = reset_level / e_level
        self.data["reset_g"] = reset_level / g_level

        # Add legend and titles
        ax[0].legend()
        ax[0].set_title("Ground state")
        ax[1].set_title("Excited state")
        plt.show()
        self.save_fig(fig, "_reset_performance")


# ====================================================== #
