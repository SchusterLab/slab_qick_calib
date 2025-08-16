import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import time
from pathlib import Path
from visdom import Visdom
from scipy.optimize import curve_fit

from qick import *

from ...exp_handling.experiment import Experiment
from ...analysis import fitting as fitter
from ...calib import readout_helpers as helpers

"""
QICK Experiment Module

This module provides classes for quantum experiments using the QICK (Quantum Instrumentation Control Kit) framework.
It extends the base Experiment class with specialized functionality for:
- Running quantum experiments on QICK hardware
- Acquiring and analyzing measurement data
- Fitting experimental results to theoretical models
- Visualizing and saving experiment data

The module contains five main classes:
- QickExperiment: Base class for single-shot quantum experiments
- QickExperimentLoop: Extension for loop-based experiments (parameter sweeps)
- QickExperiment2D: Extension for 2D parameter sweeps (e.g., parameter vs. time)
- QickExperiment2DSimple: Simplified version of 2D experiments, where you don't remake experiment each time.
- QickExperiment2DSweep: Variation of 2D sweeps where entire experiment run as one program on QICK.

These classes work with the QickProgram classes to implement complete quantum experiments.
"""


class QickExperiment(Experiment):
    """
    Base class for quantum experiments using the QICK framework.

    This class extends the Experiment base class to provide specialized functionality
    for quantum experiments on QICK hardware. It handles experiment configuration,
    data acquisition, analysis, visualization, and data storage.

    The class is designed to be extended by specific experiment implementations
    that override methods like acquire(), analyze(), and display() to implement
    specific experiment types (e.g., T1, T2, Rabi oscillations).
    """

    def __init__(
        self,
        cfg_dict=None,
        qi=0,
        prefix="QickExp",
        fname=None,
        progress=None,
        check_params=True,
    ):
        """
        Initialize the QickExperiment with hardware configuration and experiment parameters.

        Args:
            cfg_dict: Dictionary containing configuration parameters including:
                - soc: System-on-chip configuration
                - expt_path: Path for saving experiment data
                - cfg_file: Configuration file path
                - im: Instrument manager instance
            prefix: Prefix for saved data files
            progress: Whether to show progress bars during execution
            qi: Qubit index to use for the experiment
            check_params: Whether to check for unexpected parameters (default True)
        """
        soccfg = cfg_dict["soc"]
        path = cfg_dict["expt_path"]
        config_file = cfg_dict["cfg_file"]
        im = cfg_dict["im"]
        super().__init__(
            soccfg=soccfg,
            path=path,
            prefix=prefix,
            fname=fname,
            config_file=config_file,
            progress=progress,
            im=im,
        )
        # Store the check_params parameter for use in child classes
        self._check_params = check_params

        # Calculate repetitions and averages based on qubit-specific settings
        self.reps = int(
            self.cfg.device.readout.reps[qi] * self.cfg.device.readout.reps_base
        )
        self.rounds = int(
            self.cfg.device.readout.rounds[qi]
            * self.cfg.device.readout.rounds_base
        )

    def acquire(
        self, prog_name, progress=True, get_hist=True, single=True, compact=False
    ):
        """
        Acquire measurement data by running the specified quantum program.

        This method:
        1. Creates an instance of the specified program
        2. Runs the program on the QICK hardware
        3. Processes the raw measurement data
        4. Optionally generates histograms of measurement results

        Args:
            prog_name: Class reference to the QickProgram to run
            progress: Whether to show progress bar during acquisition
            get_hist: Whether to generate histogram of measurement results

        Returns:
            Dictionary containing measurement data including:
            - xpts: Swept parameter values
            - avgi/avgq: I and Q quadrature data
            - amps/phases: Amplitude and phase data
            - bin_centers/hist: Histogram data (if get_hist=True)
        """
        # Set appropriate final delay based on whether active reset is enabled
        if "active_reset" in self.cfg.expt and self.cfg.expt.active_reset:
            final_delay = self.cfg.device.readout.readout_length[
                self.cfg.expt.qubit[0]
            ]  # Not sure if this is about needed "wait" time for last readout, but seems necessary
            # final_delay = 10
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Create program instance
        prog = prog_name(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )
        # print(prog)

        # Record start time
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        # Run the program and acquire data
        iq_list = prog.acquire(
            self.im[self.cfg.aliases.soc],
            rounds=self.cfg.expt.rounds,
            threshold=None,
            #load_pulses=True,
            progress=progress,
        )

        # Get swept parameter values
        xpts = self.get_params(prog)

        # Process I/Q data to get amplitude and phase
        # Shape: Readout channels / Readouts in Program / Loops / I and Q
        iq = iq_list[0][0]
        amps = np.abs(iq.dot([1, 1j]))
        phases = np.angle(iq.dot([1, 1j]))
        avgi = np.squeeze(iq[..., 0])
        avgq = np.squeeze(iq[..., 1])

        # Generate histogram if requested
        if get_hist:
            v, hist = self.make_hist(prog, single=single)

        # Compile data dictionary
        if compact:
            data = {
                "xpts": xpts,
                "avgi": avgi,
                "avgq": avgq,
                "start_time": current_time,
            }
        else:
            data = {
                "xpts": xpts,
                "avgi": avgi,
                "avgq": avgq,
                "amps": amps,
                "phases": phases,
                "start_time": current_time,
            }
        if get_hist:
            data["bin_centers"] = v
            data["hist"] = hist

        # Convert all data to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
        self.data = data
        return data

    def analyze(
        self,
        fitfunc=None,
        fitterfunc=None,
        data=None,
        fit=True,
        use_i=None,
        get_hist=True,
        verbose=True,
        inds=None,
        **kwargs,
    ):
        """
        Analyze measurement data by fitting to theoretical models.

        This method:
        1. Fits the data to the specified model function
        2. Determines the best fit parameters and error estimates
        3. Calculates goodness-of-fit metrics (R²)
        4. Optionally scales data based on histogram analysis

        Args:
            fitfunc: Function to fit data to (e.g., exponential decay)
            fitterfunc: Function that performs the fitting
            data: Data dictionary to analyze (uses self.data if None)
            fit: Whether to perform fitting
            use_i: Whether to use I quadrature for fitting (auto-determined if None)
            get_hist: Whether to generate histogram and scale data
            **kwargs: Additional arguments passed to the fitter

        Returns:
            Data dictionary with added fit results
        """
        if data is None:
            data = self.data
        # Remove the first and last points from fit to avoid edge effects

        # Determine which data sets to fit
        ydata_lab = ["amps", "avgi", "avgq"]

        # Scale data based on histogram if requested
        if get_hist:
            self.scale_ge()
            ydata_lab.append("scale_data")

        # Perform fits on each data set (amplitude, I, Q)
        for i, ydata in enumerate(ydata_lab):
            # Use standard curve_fit via fitterfunc
            (
                data["fit_" + ydata],
                data["fit_err_" + ydata],
                data["fit_init_" + ydata],
            ) = fitterfunc(data["xpts"][1:-1], data[ydata][1:-1], **kwargs)

        # Determine which fit is best (I, Q, or amplitude)
        if use_i is None:
            use_i = self.cfg.device.qubit.tuned_up[self.cfg.expt.qubit[0]]
        if use_i:
            # For tuned-up qubits, use I quadrature by default
            i_best = "avgi"
            fit_pars = data["fit_avgi"]
            fit_err = data["fit_err_avgi"]
        else:
            # Otherwise, determine best fit automatically
            fit_pars, fit_err, i_best = fitter.get_best_fit(data, fitfunc)

        # Calculate goodness-of-fit (R²)
        r2 = fitter.get_r2(data["xpts"][1:-1], data[i_best][1:-1], fitfunc, fit_pars)
        data["r2"] = r2
        data["best_fit"] = fit_pars
        i_best = i_best.encode("ascii", "ignore")
        data["i_best"] = i_best

        if inds is None:
            inds = np.arange(len(fit_err))

        fit_err = fit_err[inds]
        # Calculate relative parameter errors
        fit_pars = np.array(fit_pars)
        data["fit_err_par"] = np.sqrt(np.diag(fit_err)) / fit_pars[inds]
        fit_err = np.mean(np.abs(data["fit_err_par"]))
        data["fit_err"] = fit_err

        # Print fit quality metrics
        if verbose:
            print(f"R2:{r2:.3f}\tFit par error:{fit_err:.3f}\t Best fit:{i_best}")

        self.get_status()

        return data

    def display(
        self,
        data=None,
        ax=None,
        plot_all=False,
        title="",
        xlabel="",
        fit=True,
        show_hist=False,
        rescale=False,
        fitfunc=None,
        caption_params=[],
        debug=False,
        **kwargs,
    ):
        """
        Display measurement results with optional fit curves.

        This method creates plots showing the measurement data and optional fit curves.
        It can display:
        - Single quadrature (I) or all quadratures (I, Q, amplitude)
        - Fit curves with parameter values in the legend
        - Histograms of single-shot measurements
        - Rescaled data based on histogram analysis

        Args:
            data: Data dictionary to display (uses self.data if None)
            ax: Matplotlib axis to plot on (creates new figure if None)
            plot_all: Whether to plot all quadratures (I, Q, amplitude)
            title: Plot title
            xlabel: X-axis label
            fit: Whether to show fit curves
            show_hist: Whether to show histogram plot
            rescale: Whether to show rescaled data (0-1 probability)
            fitfunc: Function used for fitting
            caption_params: List of parameters to display in the legend
            debug: Whether to show debug information (initial guess)
            **kwargs: Additional arguments for plotting
        """
        if data is None:
            data = self.data

        # Determine whether to save the figure
        if ax is None:
            save_fig = True
        else:
            save_fig = False

        # Configure plot layout based on what to display
        if plot_all:
            # Create 3-panel figure for amplitude, I, and Q
            fig, ax = plt.subplots(3, 1, figsize=(7, 9.5))
            fig.suptitle(title)
            ylabels = ["Amplitude (ADC units)", "I (ADC units)", "Q (ADC units)"]
            ydata_lab = ["amps", "avgi", "avgq"]
        else:
            # Create single panel figure
            if ax is None:
                fig, a = plt.subplots(1, 1, figsize=(7, 4))
                ax = [a]
            if rescale:
                # Show rescaled data (0-1 probability)
                ylabels = ["Excited State Probability"]
                ydata_lab = ["scale_data"]
            else:
                # Show raw I quadrature
                ylabels = ["I (ADC units)"]
                ydata_lab = ["avgi"]
            ax[0].set_title(title)

        # Plot each data set
        for i, ydata in enumerate(ydata_lab):
            # Plot data points (excluding first and last points)
            ax[i].plot(data["xpts"][1:-1], data[ydata][1:-1], "o-")

            # Add fit curve if requested
            if fit:
                p = data["fit_" + ydata]  # Fit parameters
                pCov = data["fit_err_" + ydata]  # Covariance matrix

                # Create caption with fit parameters
                caption = ""
                for j in range(len(caption_params)):
                    if j > 0:
                        caption += "\n"
                    if isinstance(caption_params[j]["index"], int):
                        # Display parameter value and error
                        ind = caption_params[j]["index"]
                        caption += caption_params[j]["format"].format(
                            val=(p[ind]), err=np.sqrt(pCov[ind, ind])
                        )
                    else:
                        # Display derived parameter
                        var = caption_params[j]["index"]
                        caption += caption_params[j]["format"].format(
                            val=data[var + "_" + ydata]
                        )

                # Plot fit curve
                ax[i].plot(
                    data["xpts"][1:-1], fitfunc(data["xpts"][1:-1], *p), label=caption
                )
                ax[i].legend()

            # Set axis labels
            ax[i].set_ylabel(ylabels[i])
            ax[i].set_xlabel(xlabel)

            # Show initial guess if in debug mode
            if debug:
                pinit = data["fit_init_" + ydata]
                print(pinit)
                ax[i].plot(
                    data["xpts"], fitfunc(data["xpts"], *pinit), label="Initial Guess"
                )

        # Show histogram if requested
        if show_hist:
            fig2, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.plot(data["bin_centers"], data["hist"], "o-")
            # Try to plot histogram fit if available
            try:
                ax.plot(
                    data["bin_centers"],
                    helpers.two_gaussians_decay(data["bin_centers"], *data["hist_fit"]),
                    label="Fit",
                )
            except:
                pass
            ax.set_xlabel("I [ADC units]")
            ax.set_ylabel("Probability")

        # Save figure if created in this method
        if save_fig:
            imname = self.fname.split("\\")[-1]
            fig.tight_layout()
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )
            plt.show()

    def make_hist(self, prog, single=True):
        """
        Generate histogram of single-shot measurement results.

        This method collects individual measurement shots and creates a histogram
        of the I quadrature values, which can be used for state discrimination
        and readout fidelity analysis.

        Args:
            prog: QickProgram instance to collect shots from
            single: Whether to collect shots for the entire experiment together, or separately for each point in the sweep

        Returns:
            Tuple of (bin_centers, hist) containing histogram data
        """
        # Get I/Q offset from configuration
        offset = self.soccfg._cfg["readouts"][self.cfg.expt.qubit_chan]["iq_offset"]

        # Collect individual measurement shots
        shots_i, shots_q = prog.collect_shots(offset=offset, single=single)

        # Create histogram with 60 bins
        # sturges_bins = int(np.ceil(np.log2(len(shots_i)) + 1))
        if single:
            hist, bin_edges = np.histogram(shots_i, bins=60, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            hist = []
            bin_centers = []
            for i in range(len(shots_i)):
                hist0, bin_edges0 = np.histogram(shots_i[i], bins=60, density=True)
                hist.append(hist0)
                bin_centers.append((bin_edges0[:-1] + bin_edges0[1:]) / 2)
        return bin_centers, hist

    def qubit_run(
        self,
        qi=0,
        progress=True,
        analyze=True,
        display=True,
        save=True,
        print=False,
        min_r2=0.1,
        max_err=1,
        disp_kwargs=None,
        **kwargs,
    ):
        # Configure active reset if enabled
        if self.cfg.expt.active_reset:
            self.configure_reset()

        # For untuned qubits, show all data points by default
        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {"plot_all": True}
            # For untuned qubits, show all data points by default
        if (
            self.cfg.device.readout.rescale[qi]
            or disp_kwargs is not None
            and "rescale" in disp_kwargs
        ):
            disp_kwargs = {"rescale": True}

        # Run the experiment if go=True
        if print:
            self.print()
        else:
            self.run(
                analyze=analyze,
                display=display,
                save=save,
                progress=progress,
                min_r2=min_r2,
                max_err=max_err,
                disp_kwargs=disp_kwargs,
            )

    def run(
        self,
        progress=True,
        analyze=True,
        display=True,
        save=True,
        min_r2=0.1,
        max_err=1,
        disp_kwargs=None,
        **kwargs,
    ):
        """
        Run the complete experiment workflow.

        This method executes the full experiment sequence:
        1. Acquire data
        2. Analyze results
        3. Display plots
        4. Save data to disk
        5. Determine if the experiment was successful

        Args:
            progress: Whether to show progress bar during acquisition
            analyze: Whether to perform data analysis
            display: Whether to display results
            save: Whether to save data to disk
            min_r2: Minimum R² value for acceptable fit
            max_err: Maximum error for acceptable fit
            disp_kwargs: Display options dictionary
            **kwargs: Additional arguments passed to the analyze method
        """

        # Set default values for fit quality thresholds
        if min_r2 is None:
            min_r2 = 0.1
        if max_err is None:
            max_err = 1
        if disp_kwargs is None:
            disp_kwargs = {}
            # These might be rescale, show_hist, plot_all. Eventually, want to put plot_all into the config.

        # Execute experiment workflow
        data = self.acquire(progress)
        if analyze:
            data = self.analyze(data, **kwargs)
        if save:
            self.save_data(data)
        if display:
            self.display(data, **disp_kwargs)

    def save_data(self, data=None, verbose=False):
        """
        Save experiment data to disk.

        Args:
            data: Data dictionary to save (uses self.data if None)
            verbose: Whether to print save confirmation

        Returns:
            Filename where data was saved
        """
        if verbose:
            print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname

    def print(self):
        """
        Print out the experimental config
        """
        for key, value in self.cfg.expt.items():
            print(f"{key}: {value}")

    def get_status(self, max_err=1, min_r2=0.1):
        # Determine if experiment was successful based on fit quality
        if (
            "fit_err" in self.data
            and "r2" in self.data
            and self.data["fit_err"] < max_err
            and self.data["r2"] > min_r2
        ):
            self.status = True
        elif "fit_err" not in self.data or "r2" not in self.data:
            # No fit performed, can't determine status
            pass
        else:
            # print("Fit failed")
            self.status = False

    def get_params(self, prog):
        """
        Get swept parameter values from the program.

        This method extracts the values of the parameter being swept in the experiment,
        either a pulse parameter (e.g., amplitude, frequency) or a time parameter
        (e.g., delay, pulse length).
        self.param needs to have fields set:
        - param_type: "pulse" or "time"
        - label: Label of the parameter to extract [listed in the program]
        - param: Name of the parameter to extract (freq, gain, total_length, t)

        Args:
            prog: QickProgram instance to get parameters from

        Returns:
            Array of parameter values
        """
        if self.param["param_type"] == "pulse":
            # Extract pulse parameter (amplitude, frequency, etc.)
            prog.list_pulse_params(self, self.param['label'])
            xpts = prog.get_pulse_param(
                self.param["label"], self.param["param"], as_array=True
            )
        else:
            # Extract time parameter (delay, pulse length, etc.)
            xpts = prog.get_time_param(
                self.param["label"], self.param["param"], as_array=True
            )
        return xpts

    def check_params(self, params_def):
        if self._check_params:
            unexpected_params = set(self.cfg.expt.keys()) - set(params_def.keys())
            if unexpected_params:
                print(f"Unexpected parameters found in params: {unexpected_params}")

    def configure_reset(self):
        qi = self.cfg.expt.qubit[0]
        # we may want to put these params in the config.
        params_def = dict(
            threshold_v=self.cfg.device.readout.threshold[qi],
            read_wait=0.1,
            extra_delay=0.2,
        )
        self.cfg.expt = {**params_def, **self.cfg.expt}
        # this number should be changed to be grabbed from soc
        self.cfg.expt["threshold"] = int(
            self.cfg.expt["threshold_v"]
            * self.cfg.device.readout.readout_length[qi]
            / 0.0032552083333333335
        )

    def get_freq(self, fit=True):
        """
        Provide correct frequency if mixers are in use, for LO coming from QICK or external source
        """
        freq_offset = 0
        q = self.cfg.expt.qubit[0]
        if "mixer_freq" in self.cfg.hw.soc.dacs.readout:
            freq_offset += self.cfg.hw.soc.dacs.readout.mixer_freq[q]
        # lo_freq is in readout; used for signal core.
        if "lo_freq" in self.cfg.hw.soc.dacs.readout:
            freq_offset += self.cfg.hw.soc.dacs.readout.lo_freq[q]
        if "lo" in self.cfg.hw.soc and "mixer_freq" in self.cfg.hw.soc.lo:
            freq_offset += self.cfg.hw.soc.lo.mixer_freq[q]

        self.data["freq"] = freq_offset + self.data["xpts"]
        self.data["freq_offset"] = freq_offset
        # if fit:
        #     self.data["freq_fit"] = self.data["fit"]
        #     self.data["freq_init"] = self.data["init"]
        #     self.data["freq_fit"][0] = freq_offset + self.data["fit"][0]
        #     self.data["freq_init"][0] = freq_offset + self.data["init"][0]

    def scale_ge(self):
        """
        Scale g->0 and e->1 based on histogram data"""

        hist = self.data["hist"]
        bin_centers = self.data["bin_centers"]
        v_rng = np.max(bin_centers) - np.min(bin_centers)

        p0 = [
            0.5,
            np.min(bin_centers) + v_rng / 3,
            0.5,
            v_rng / 10,
            np.max(bin_centers) - v_rng / 3,
        ]
        try:
            popt, pcov = curve_fit(helpers.two_gaussians, bin_centers, hist, p0=p0)
            vg = popt[1]
            ve = popt[4]
            dv = ve - vg
            # if (
            #     "tm" in self.cfg.device.readout
            #     and self.cfg.device.readout.tm[self.cfg.expt.qubit[0]] != 0
            # ):
            #     tm = self.cfg.device.readout.tm[self.cfg.expt.qubit[0]]
            #     sigma = self.cfg.device.readout.sigma[self.cfg.expt.qubit[0]]
            #     p0 = [popt[0], vg, ve]
            #     popt, pcov = curve_fit( #@IgnoreException
            #         lambda x, mag_g, vg, ve: helpers.two_gaussians_decay(
            #             x, mag_g, vg, ve, sigma, tm
            #         ),
            #         bin_centers,
            #         hist,
            #         p0=p0,
            #     )
            #     popt = np.concatenate((popt, [sigma, tm]))

            # dv = popt[2] - popt[1]
            self.data["scale_data"] = (self.data["avgi"] - popt[1]) / dv
            self.data["hist_fit"] = popt
        except:
            self.data["scale_data"] = self.data["avgi"]


class QickExperimentLoop(QickExperiment):
    """
    Extension of QickExperiment for loop-based parameter sweeps.

    This class implements experiments where a parameter is swept through a range of values.
    It handles the loop iteration, data collection for each parameter value, and
    aggregation of results into a complete dataset.

    """

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=False, qi=0):
        """
        Initialize the QickExperimentLoop.

        Args:
            cfg_dict: Configuration dictionary
            prefix: Prefix for saved data files
            progress: Whether to show progress bars
            qi: Qubit index to use for the experiment
        """
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

    def acquire(self, prog_name, x_sweep, progress=True, hist=False):
        """
        Acquire data by running the program for each point in the parameter sweep.

        This method:
        1. Iterates through each point in the parameter sweep
        2. Updates the configuration with the current parameter value
        3. Runs the program and collects data for that parameter value
        4. Aggregates results into a complete dataset

        Args:
            prog_name: Class reference to the QickProgram to run
            x_sweep: List of dictionaries defining the parameter sweep
                     Each dict contains 'var' (parameter name) and 'pts' (values)
            progress: Whether to show progress bar
            hist: Whether to collect histogram data

        Returns:
            Dictionary containing measurement data for all sweep points
        """
        # Set appropriate final delay based on whether active reset is enabled
        if "active_reset" in self.cfg.expt and self.cfg.expt.active_reset:
            final_delay = self.cfg.device.readout.readout_length[self.cfg.expt.qubit[0]]
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Initialize data dictionary
        data = {"xpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}
        shots_i = []

        # Record start time
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        # Iterate through each point in the parameter sweep
        xvals = np.arange(len(x_sweep[0]["pts"]))
        for i in tqdm(xvals, disable=not progress):
            # Update configuration with current parameter values
            for j in range(len(x_sweep)):
                self.cfg.expt[x_sweep[j]["var"]] = x_sweep[j]["pts"][i]

            # Create and run program for this parameter value
            prog = prog_name(soccfg=self.soccfg, final_delay=final_delay, cfg=self.cfg)

            iq_list = prog.acquire(
                self.im[self.cfg.aliases.soc],
                rounds=self.cfg.expt.rounds,
                threshold=None,
                progress=False,
            )

            # Store measurement data for this parameter value
            data = self.stow_data(iq_list, data)

            # Collect individual shots for histogram
            offset = self.soccfg._cfg["readouts"][self.cfg.expt.qubit_chan]["iq_offset"]
            shots_i_new, shots_q = prog.collect_shots(offset=offset)
            shots_i.append(shots_i_new)

            # Store parameter value
            xpt = self.get_params(prog)
            data["xpts"].append(xpt)

        # Generate histogram from all collected shots
        bin_centers, hist = self.make_hist(shots_i)
        data["bin_centers"] = bin_centers
        data["hist"] = hist

        # Store parameter sweep values
        for j in range(len(x_sweep)):
            data[x_sweep[j]["var"] + "_pts"] = x_sweep[j]["pts"]

        # Convert all data to numpy arrays
        for k, a in data.items():
            data[k] = np.array(a).flatten()

        # Add metadata and store data
        data["start_time"] = current_time
        self.data = data

        return data

    def stow_data(self, iq_list, data):
        """
        Process and store I/Q data from a measurement.

        This method extracts I and Q quadrature data from the measurement results,
        calculates amplitude and phase, and adds them to the data dictionary.

        Args:
            iq_list: List of I/Q data from program.acquire()
            data: Data dictionary to update

        Returns:
            Updated data dictionary
        """
        # Calculate amplitude and phase from I/Q data
        amps = np.abs(iq_list[0][0].dot([1, 1j]))
        phases = np.angle(iq_list[0][0].dot([1, 1j]))
        avgi = iq_list[0][0][:, 0]  # I quadrature
        avgq = iq_list[0][0][:, 1]  # Q quadrature

        # Append to data arrays
        data["avgi"].append(avgi)
        data["avgq"].append(avgq)
        data["amps"].append(amps)
        data["phases"].append(phases)
        return data

    def make_hist(self, shots_i):
        """
        Generate histogram from collected shots.

        This method creates a histogram of the I quadrature values from
        all collected shots across the parameter sweep.

        Args:
            shots_i: List of I quadrature shots

        Returns:
            Tuple of (bin_centers, hist) containing histogram data
        """
        hist, bin_edges = np.histogram(shots_i, bins=60)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist


class QickExperiment2D(QickExperimentLoop):
    """
    Extension of QickExperimentLoop for 2D parameter sweeps.

    This class implements experiments where two parameters are swept:
    - The x-axis parameter is typically swept by the program (e.g., pulse frequency)
    - The y-axis parameter is swept by this class (e.g., time, power, etc.)

    It creates a 2D map of measurement results, useful for:
    - Stability measurements (parameter vs. time)
    - Power dependence studies (parameter vs. amplitude)
    - Frequency vs. flux measurements
    - Any experiment requiring a 2D parameter space exploration
    """

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=0):
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)


    def acquire(self, prog_name, y_sweep, progress=True):
        """
        Acquire data for a 2D parameter sweep.

        This method:
        1. Iterates through each point in the y-axis parameter sweep
        2. For each y value, runs the program which sweeps the x-axis parameter
        3. Collects and organizes the 2D data

        Args:
            prog_name: Class reference to the QickProgram to run
            y_sweep: List of dictionaries defining the y-axis parameter sweep
                     Each dict contains 'var' (parameter name) and 'pts' (values)
            progress: Whether to show progress bar

        Returns:
            Dictionary containing 2D measurement data
        """
        # Initialize data dictionary
        data = {"avgi": [], "avgq": [], "amps": [], "phases": []}
        yvals = np.arange(len(y_sweep[0]["pts"]))
        data["time"] = []

        # Record start time
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        # Iterate through each point in the y-axis parameter sweep
        for i in tqdm(yvals):
            # Update configuration with current y-axis parameter value
            for j in range(len(y_sweep)):
                self.cfg.expt[y_sweep[j]["var"]] = y_sweep[j]["pts"][i]

            # Create and run program for this y value
            # (The program internally sweeps the x-axis parameter)
            prog = prog_name(
                soccfg=self.soccfg,
                final_delay=self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]],
                cfg=self.cfg,
            )
            iq_list = prog.acquire(
                self.im[self.cfg.aliases.soc],
                rounds=self.cfg.expt.rounds,
                threshold=None,
                progress=progress,
            )

            # Store measurement data for this y value
            data = self.stow_data(iq_list, data)
            data["time"].append(time.time())

        # Get x-axis parameter values from the program
        data["xpts"] = self.get_params(prog)

        # Set y-axis values (either time in hours or the swept parameter)
        if "count" in [y_sweep[j]["var"] for j in range(len(y_sweep))]:
            # Convert time to hours for time-based sweeps
            data["ypts"] = (data["time"] - np.min(data["time"])) / 3600
        else:
            # Use the swept parameter values
            data["ypts"] = y_sweep[0]["pts"]

        # Store y-axis parameter sweep values
        for j in range(len(y_sweep)):
            data[y_sweep[j]["var"] + "_pts"] = y_sweep[j]["pts"]

        # Convert all data to numpy arrays
        for k, a in data.items():
            data[k] = np.array(a)

        # Add metadata and store data
        data["start_time"] = current_time
        self.data = data
        return data

    def analyze(self, fitfunc=None, fitterfunc=None, data=None, fit=False, **kwargs):
        """
        Analyze 2D experiment data.

        This method fits each row of the 2D data (each y value) to the
        specified model function, creating a set of fit parameters for each row.

        Args:
            fitfunc: Function to fit data to
            fitterfunc: Function that performs the fitting
            data: Data dictionary to analyze
            fit: Whether to perform fitting
            **kwargs: Additional arguments passed to the fitter

        Returns:
            Data dictionary with added fit results
        """
        if data is None:
            data = self.data

        # Define which data sets to fit (focus on I quadrature)
        ydata_lab = ["amps", "avgi", "avgq"]
        ydata_lab = ["avgi"]  # Typically only fit I quadrature for speed

        # Fit each row (y value) separately
        for i, ydata in enumerate(ydata_lab):
            data["fit_" + ydata] = []
            data["fit_err_" + ydata] = []

            # Iterate through each y value
            for j in range(len(data["ypts"])):
                # Fit this row to the model function
                fit_pars, fit_err, init = fitterfunc(
                    data["xpts"], data[ydata][j], fitparams=None
                )
                # Store fit parameters and errors
                data["fit_" + ydata].append(fit_pars)
                data["fit_err_" + ydata].append(fit_err)

        return data

    def display(
        self,
        data=None,
        ax=None,
        plot_both=False,
        plot_amps=False,
        title="",
        xlabel="",
        ylabel="",
        **kwargs,
    ):
        """
        Display 2D experiment results.

        This method creates 2D color plots (heatmaps) showing the measurement
        results as a function of both swept parameters. It can display:
        - Single quadrature (I)
        - Both quadratures (I and Q)
        - Amplitude and phase

        Args:
            data: Data dictionary to display
            ax: Matplotlib axis to plot on
            plot_both: Whether to plot both I and Q quadratures
            plot_amps: Whether to plot amplitude and phase instead of I/Q
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            **kwargs: Additional arguments for plotting
        """
        if data is None:
            data = self.data

        # Get x and y sweep values for the 2D plot
        x_sweep = data["xpts"]
        y_sweep = data["ypts"]

        # Determine whether to save the figure
        if ax is None:
            savefig = True
        else:
            savefig = False

        # Configure plot layout based on what to display
        if plot_both:
            # Create 2-panel figure for I and Q
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            ydata_lab = ["avgi", "avgq"]
            ydata_labs = ["I (ADC level)", "Q (ADC level)"]
            fig.suptitle(title)
        elif plot_amps:
            # Create 2-panel figure for amplitude and phase
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            ydata_lab = ["amps", "phases"]
            ydata_labs = ["Amplitude (ADC level)", "Phase (radians)"]
            fig.suptitle(title)
        else:
            # Create single panel figure for I quadrature
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.set_title(title)
            ydata_lab = ["avgi"]
            ax = [ax]
            ydata_labs = ["I (ADC level)"]

        # Create 2D color plot for each data set
        for i, ydata in enumerate(ydata_lab):
            # Create heatmap using pcolormesh
            ax[i].pcolormesh(
                x_sweep, y_sweep, data[ydata], cmap="viridis", shading="auto"
            )
            # Add colorbar with label
            plt.colorbar(ax[i].collections[0], ax=ax[i], label=ydata_labs[i])
            # Set axis labels
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)

            # Use log scale for y-axis if specified in configuration
            if "log" in self.cfg.expt and self.cfg.expt.log:
                ax[i].set_yscale("log")

        # Save figure if created in this method
        if savefig:
            fig.tight_layout()

            file_path = Path(self.fname)

            # Get the parent directory
            parent_dir = file_path.parent

            # Get the filename and change its extension to .png
            new_filename = file_path.name.rsplit(".", 1)[0] + ".png"
            # Create the full output path and save the figure
            output_path = parent_dir / "images" / new_filename

            fig.savefig(output_path)
            plt.show()


class QickExperiment2DSimple(QickExperiment2D):
    """
    Simplified version of QickExperiment2D for nested experiments.

    This class provides a simpler interface for 2D experiments where the
    x-axis parameter is swept by a separate experiment instance rather than
    by the program directly. This is useful for:
    - Combining multiple experiment types
    - Reusing existing experiment implementations
    - Creating complex nested parameter sweeps
    """

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=0, live_plot=False):
        """
        Initialize the QickExperiment2DSimple.

        Args:
            cfg_dict: Configuration dictionary
            prefix: Prefix for saved data files
            progress: Whether to show progress bars
            qi: Qubit index to use for the experiment
        """
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        self.live_plot = live_plot
        self.viz = None
        self.viz_window = None

        if self.live_plot:
            try:
                self.viz = Visdom()
                assert self.viz.check_connection(), "Visdom server not running."
                self.viz_window = self.viz.text("Starting 2D Scan...", opts={"title": "Qick 2D Scan"})
            except Exception as e:
                print(f"[Visdom] Could not connect: {e}")
                self.live_plot = False  # disable to avoid crashes later

    def acquire(self, y_sweep, progress=False):
        """
        Acquire data for a 2D parameter sweep using a nested experiment.

        This method:
        1. Iterates through each point in the y-axis parameter sweep
        2. For each y value, runs a separate experiment instance (self.expt)
        3. Collects and organizes the 2D data from the nested experiment

        Args:
            y_sweep: List of dictionaries defining the y-axis parameter sweep
                     Each dict contains 'var' (parameter name) and 'pts' (values)
                     If 'var' is 'count', the y-axis is time in hours, it will repeat same experiment
            progress: Whether to show progress bar

        Returns:
            Dictionary containing 2D measurement data
        """
        # Initialize data dictionary with all expected fields
        data = {}

        # Prepare for y-axis sweep
        yvals = np.arange(len(y_sweep[0]["pts"]))
        data["time"] = []
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        # Iterate through each point in the y-axis parameter sweep
        for i in tqdm(yvals):
            # Update nested experiment configuration with current y-axis parameter value
            for j in range(len(y_sweep)):
                self.expt.cfg.expt[y_sweep[j]["var"]] = y_sweep[j]["pts"][i]

            # Run the nested experiment (which handles the x-axis sweep)
            data_new = self.expt.acquire(progress=progress)

            # Store all data from the nested experiment

            for key in data_new:
                if i == 0:
                    data[key] = []
                data[key].append(data_new[key])
            
            # 👉 Live update heatmap plot using Visdom
            if self.live_plot:
                try:
                    import matplotlib.pyplot as plt
                    from io import BytesIO
                    import PIL.Image

                    # Step 1: Safely extract data arrays
                    amps_so_far = np.array(data.get("amps", []))
                    phases_so_far = np.array(data.get("phases", []))
                    avgi_so_far = np.array(data.get("avgi", []))
                    avgq_so_far = np.array(data.get("avgq", []))

                    if amps_so_far.size == 0:
                        continue  # wait for at least one sweep to populate

                    # Step 2: Get axis values after we know amps_so_far is valid
                    xvals = data_new["xpts"]  # frequency values
                    yvals = np.array(y_sweep[0]["pts"][:amps_so_far.shape[0]])  # gain or time values
                    ylabel = "Time [hr]" if any(s["var"] == "count" for s in y_sweep) else "Gain [DAC level]"

                    # Step 3: Plot 4 subplots
                    fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)

                    im1 = axs[0, 0].imshow(amps_so_far, aspect='auto', cmap='viridis',
                                        extent=[xvals[0], xvals[-1], yvals[-1], yvals[0]])
                    axs[0, 0].set_title("Amps")
                    plt.colorbar(im1, ax=axs[0, 0])

                    im2 = axs[0, 1].imshow(phases_so_far, aspect='auto', cmap='twilight',
                                        extent=[xvals[0], xvals[-1], yvals[-1], yvals[0]])
                    axs[0, 1].set_title("Phases")
                    plt.colorbar(im2, ax=axs[0, 1])

                    im3 = axs[1, 0].imshow(avgi_so_far, aspect='auto', cmap='plasma',
                                        extent=[xvals[0], xvals[-1], yvals[-1], yvals[0]])
                    axs[1, 0].set_title("AvgI")
                    plt.colorbar(im3, ax=axs[1, 0])

                    im4 = axs[1, 1].imshow(avgq_so_far, aspect='auto', cmap='cividis',
                                        extent=[xvals[0], xvals[-1], yvals[-1], yvals[0]])
                    axs[1, 1].set_title("AvgQ")
                    plt.colorbar(im4, ax=axs[1, 1])

                    for ax in axs.flat:
                        ax.set_xlabel("Readout Frequency [MHz]")
                        ax.set_ylabel(ylabel)

                    plt.tight_layout()

                    buf = BytesIO()
                    plt.savefig(buf, format='png')
                    plt.close(fig)
                    buf.seek(0)
                    img = PIL.Image.open(buf).convert("RGB")
                    img = np.array(img).transpose(2, 0, 1)  # Visdom expects CHW

                    self.viz.image(img, win=self.viz_window, opts={"title": "Live Resonator Channels"})

                except Exception as e:
                    print(f"[Visdom] Quad plot failed at step {i}: {e}")


        # Set y-axis values (either time in hours or the swept parameter)
        if "count" in [y_sweep[j]["var"] for j in range(len(y_sweep))]:
            # Convert time to hours for time-based sweeps
            data["ypts"] = (data["time"] - np.min(data["time"])) / 3600
        else:
            # Use the swept parameter values
            data["ypts"] = y_sweep[0]["pts"]

        # Store y-axis parameter sweep values
        for j in range(len(y_sweep)):
            data[y_sweep[j]["var"] + "_pts"] = y_sweep[j]["pts"]

        # Use the x-axis values from the first nested experiment run
        data["xpts"] = data["xpts"][0]
        for k, a in data.items():
            data[k] = np.array(a)
        self.data = data

        return data


class QickExperiment2DSweep(QickExperiment):
    """
    Extension of QickExperiment for 2D parameter sweeps with a different analysis method.

    This class implements experiments where two parameters are swept, similar to QickExperiment2D,
    but it uses a different analysis method for fitting the 2D data.
    """

    def analyze(self, fitfunc=None, fitterfunc=None, data=None, fit=False, **kwargs):
        """
        Analyze 2D experiment data.

        This method fits each row of the 2D data (each y value) to the
        specified model function, creating a set of fit parameters for each row.

        Args:
            fitfunc: Function to fit data to
            fitterfunc: Function that performs the fitting
            data: Data dictionary to analyze
            fit: Whether to perform fitting
            **kwargs: Additional arguments passed to the fitter

        Returns:
            Data dictionary with added fit results
        """
        if data is None:
            data = self.data

        # Define which data sets to fit (focus on I quadrature)
        ydata_lab = ["amps", "avgi", "avgq"]
        ydata_lab = ["avgi"]  # Typically only fit I quadrature for speed

        # Fit each row (y value) separately
        for i, ydata in enumerate(ydata_lab):
            data["fit_" + ydata] = []
            data["fit_err_" + ydata] = []

            # Iterate through each y value
            for j in range(len(data["ypts"])):
                # Fit this row to the model function
                fit_pars, fit_err, init = fitterfunc(
                    data["xpts"], data[ydata][j], fitparams=None
                )
                # Store fit parameters and errors
                data["fit_" + ydata].append(fit_pars)
                data["fit_err_" + ydata].append(fit_err)

        return data

    def display(
        self,
        data=None,
        ax=None,
        plot_both=False,
        plot_amps=False,
        title="",
        xlabel="",
        ylabel="",
        **kwargs,
    ):
        """
        Display 2D experiment results.

        This method creates 2D color plots (heatmaps) showing the measurement
        results as a function of both swept parameters. It can display:
        - Single quadrature (I)
        - Both quadratures (I and Q)
        - Amplitude and phase

        Args:
            data: Data dictionary to display
            ax: Matplotlib axis to plot on
            plot_both: Whether to plot both I and Q quadratures
            plot_amps: Whether to plot amplitude and phase instead of I/Q
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            **kwargs: Additional arguments for plotting
        """
        if data is None:
            data = self.data

        # Get x and y sweep values for the 2D plot
        x_sweep = data["xpts"]
        y_sweep = data["ypts"]

        # Determine whether to save the figure
        if ax is None:
            savefig = True
        else:
            savefig = False

        # Configure plot layout based on what to display
        if plot_both:
            # Create 2-panel figure for I and Q
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            ydata_lab = ["avgi", "avgq"]
            ydata_labs = ["I (ADC level)", "Q (ADC level)"]
            fig.suptitle(title)
        elif plot_amps:
            # Create 2-panel figure for amplitude and phase
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            ydata_lab = ["amps", "phases"]
            ydata_labs = ["Amplitude (ADC level)", "Phase (radians)"]
            fig.suptitle(title)
        else:
            # Create single panel figure for I quadrature
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.set_title(title)
            ydata_lab = ["avgi"]
            ax = [ax]
            ydata_labs = ["I (ADC level)"]

        # Create 2D color plot for each data set
        for i, ydata in enumerate(ydata_lab):
            # Create heatmap using pcolormesh
            ax[i].pcolormesh(
                x_sweep, y_sweep, data[ydata], cmap="viridis", shading="auto"
            )
            # Add colorbar with label
            plt.colorbar(ax[i].collections[0], ax=ax[i], label=ydata_labs[i])
            # Set axis labels
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)

            # Use log scale for y-axis if specified in configuration
            if "log" in self.cfg.expt and self.cfg.expt.log:
                ax[i].set_yscale("log")

        # Save figure if created in this method
        if savefig:
            fig.tight_layout()

            file_path = Path(self.fname)

            # Get the parent directory
            parent_dir = file_path.parent

            # Get the filename and change its extension to .png
            new_filename = file_path.name.rsplit(".", 1)[0] + ".png"
            # Create the full output path and save the figure
            output_path = parent_dir / "images" / new_filename

            fig.savefig(output_path)
            plt.show()
