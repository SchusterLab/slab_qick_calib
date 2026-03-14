"""
QICK Two-Qubit Experiment Module

This module provides classes for two-qubit quantum experiments using the QICK framework.
It extends the base Experiment class with specialized functionality for:
- Running quantum experiments on two qubits simultaneously
- Acquiring and analyzing measurement data from multiple qubits
- Fitting experimental results to theoretical models
- Visualizing and saving two-qubit experiment data

The module contains one main class:
- QickExperiment2Q: Base class for two-qubit quantum experiments

This class works with two-qubit QickProgram classes to implement complete two-qubit experiments.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings

from qick import *

from ...exp_handling.experiment import Experiment
from ...analysis import fitting as fitter

from pathlib import Path



class QickExperiment2Q(Experiment):
    """
    Base class for two-qubit quantum experiments using the QICK framework.

    This class extends the base Experiment class to support simultaneous measurements
    on two qubits. It handles configuration, data acquisition for multiple qubits,
    analysis, visualization, and data storage.

    The class is designed to be extended by specific two-qubit experiment implementations
    that override methods like acquire(), analyze(), and display() to implement
    specific two-qubit experiment types (e.g., two-qubit Rabi, two-qubit T1).

    Attributes:
        reps: Number of repetitions for each measurement (maximum of the two qubits)
        rounds: Number of software averages (maximum of the two qubits)
    """

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=[0]):
        """
        Initialize the QickExperiment2Q with hardware configuration.

        Args:
            cfg_dict: Dictionary containing configuration parameters including:
                - soc: System-on-chip configuration
                - expt_path: Path for saving experiment data
                - cfg_file: Configuration file path
                - im: Instrument manager instance
            prefix: Prefix for saved data files (default: "QickExp")
            progress: Whether to show progress bars during execution
            qi: List of qubit indices to use for the experiment (default: [0])
        """
        soccfg = cfg_dict["soc"]
        path = cfg_dict["expt_path"]
        config_file = cfg_dict["cfg_file"]
        im = cfg_dict["im"]
        super().__init__(
            soccfg=soccfg,
            path=path,
            prefix=prefix,
            config_file=config_file,
            progress=progress,
            im=im,
        )
        # Use maximum reps and rounds from both qubits
        reps = np.max([self.cfg.device.readout.reps[q] for q in qi])
        rounds = np.max([self.cfg.device.readout.rounds[q] for q in qi])
        self.reps = int(reps * self.cfg.device.readout.reps_base)
        self.rounds = int(rounds * self.cfg.device.readout.rounds_base)

    def acquire(self, prog_name, progress=True, get_hist=True):
        """
        Acquire measurement data by running the specified quantum program on two qubits.

        This method:
        1. Creates an instance of the specified two-qubit program
        2. Runs the program on the QICK hardware
        3. Processes the raw measurement data for both qubits
        4. Optionally generates histograms of measurement results

        Args:
            prog_name: Class reference to the QickProgram to run
            progress: Whether to show progress bar during acquisition (default: True)
            get_hist: Whether to generate histograms of measurement results (default: True)

        Returns:
            Dictionary containing measurement data for both qubits including:
            - xpts: List of swept parameter values for each qubit
            - avgi: List of I quadrature data for each qubit
            - avgq: List of Q quadrature data for each qubit
            - amps: List of amplitude data for each qubit
            - phases: List of phase data for each qubit
            - bin_centers: Histogram bin centers (if get_hist=True)
            - hist: Histogram data (if get_hist=True)
        """
        # Set appropriate final delay based on whether active reset is enabled
        if "active_reset" in self.cfg.expt and self.cfg.expt.active_reset:
            final_delay = self.cfg.device.readout.readout_length[self.cfg.expt.qubit[0]]
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
            
        # Create program instance
        prog = prog_name(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )
        
        # Initialize data storage lists
        amps, phases, avgi, avgq = [], [], [], []

        # Record start time
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        # Run the program and acquire data
        iq_list = prog.acquire(
            self.im[self.cfg.aliases.soc],
            rounds=self.cfg.expt.rounds,
            threshold=None,
            load_pulses=True,
            progress=progress,
        )
        
        # Get swept parameter values for each qubit
        xpts = self.get_params(prog)

        # Process I/Q data for each qubit
        for i in range(len(iq_list)):
            amps.append(np.abs(iq_list[i][0].dot([1, 1j])))
            phases.append(np.angle(iq_list[i][0].dot([1, 1j])))
            avgi.append(iq_list[i][0][:, 0])
            avgq.append(iq_list[i][0][:, 1])

        # Compile data dictionary
        data = {
            "xpts": xpts,
            "avgi": avgi,
            "avgq": avgq,
            "amps": amps,
            "phases": phases,
            "start_time": current_time,
        }
        
        # Generate histograms if requested
        if get_hist:
            v, hist = self.make_hist(prog)
            data["bin_centers"] = v
            data["hist"] = hist

        # Convert all data to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
            
        self.data = data
        return data

    def analyze(
        self, fitfunc=None, fitterfunc=None, data=None, fit=False, use_i=True, **kwargs
    ):
        """
        Analyze two-qubit measurement data by fitting to theoretical models.

        This method:
        1. Fits the data for each qubit to the specified model function
        2. Determines the best fit parameters and error estimates for each qubit
        3. Calculates goodness-of-fit metrics (R²) for each qubit

        Args:
            fitfunc: Function to fit data to (e.g., exponential decay)
            fitterfunc: Function that performs the fitting operation
            data: Data dictionary to analyze (uses self.data if None)
            fit: Whether to perform fitting (default: False)
            use_i: Whether to use I quadrature for fitting (default: True)
            **kwargs: Additional arguments passed to the fitter

        Returns:
            Data dictionary with added fit results for each qubit
        """
        if data is None:
            data = self.data

        # Determine which data sets to fit
        ydata_lab = ["amps", "avgi", "avgq"]
        
        # Perform fits on each quadrature for each qubit
        for i, ydata in enumerate(ydata_lab):
            for j in range(len(data["amps"])):
                (
                    data["fit_" + ydata + "_" + str(j)],
                    data["fit_err_" + ydata + "_" + str(j)],
                    data["fit_init_" + ydata + "_" + str(j)],
                ) = fitterfunc(
                    data["xpts"][j][1:-1], data[ydata][j][1:-1], fitparams=None
                )

        # Determine best fit and calculate quality metrics for each qubit
        for j in range(len(data["amps"])):
            i_best = "avgi"
            fit_pars = data["fit_avgi_" + str(j)]
            fit_err = data["fit_err_avgi_" + str(j)]

            # Calculate goodness-of-fit (R²)
            r2 = fitter.get_r2(
                data["xpts"][j][1:-1], data[i_best][j][1:-1], fitfunc, fit_pars
            )
            data["r2_" + str(j)] = r2
            data["best_fit_" + str(j)] = fit_pars
            i_best = i_best.encode("ascii", "ignore")
            data["i_best_" + str(j)] = i_best
            
            # Calculate fit error
            fit_err = np.mean(np.abs(fit_err / fit_pars))
            data["fit_err_" + str(j)] = fit_err
            print(f"R2:{r2:.3f}\tFit par error:{fit_err:.3f}\t Best fit:{i_best}")

        return data

    def display(
        self,
        data=None,
        ax=None,
        plot_all=False,
        title="",
        xlabel="",
        fit=True,
        show_hist=True,
        fitfunc=None,
        caption_params=[],
        debug=False,
        **kwargs,
    ):
        """
        Display two-qubit measurement results with optional fit curves.

        This method creates plots showing the measurement data and optional fit curves
        for both qubits. It can display:
        - Single quadrature (I) or all quadratures (I, Q, amplitude) for each qubit
        - Fit curves with parameter values in the legend
        - Histograms of single-shot measurements for both qubits

        Args:
            data: Data dictionary to display (uses self.data if None)
            ax: Matplotlib axis to plot on (creates new figure if None)
            plot_all: Whether to plot all quadratures (I, Q, amplitude) (default: False)
            title: Plot title (can be a list for multiple qubits)
            xlabel: X-axis label
            fit: Whether to show fit curves (default: True)
            show_hist: Whether to show histogram plots (default: True)
            fitfunc: Function used for fitting
            caption_params: List of parameters to display in the legend
            debug: Whether to show debug information (initial guess) (default: False)
            **kwargs: Additional arguments for plotting
        """
        if data is None:
            data = self.data

        # Determine whether to save the figure
        if ax is None:
            save_fig = True
        else:
            save_fig = False

        nq = len(self.cfg.expt.qubit)
        
        # Configure plot layout based on what to display
        if plot_all:
            # Create 3-panel figure for amplitude, I, and Q
            fig, ax = plt.subplots(3, 1, figsize=(7.5, 9.5))
            fig.suptitle(title)
            ylabels = ["Amplitude [ADC units]", "I [ADC units]", "Q [ADC units]"]
            ydata_lab = ["amps", "avgi", "avgq"]
        else:
            # Create figure with one panel per qubit for I quadrature
            if ax is None:
                fig, ax = plt.subplots(nq, 1, figsize=(7, 7))
            ylabels = ["I [ADC units]"]
            ydata_lab = ["avgi"]
            # Set titles for each qubit
            for i in range(nq):
                ax[i].set_title(title[i])

        # Plot data for each quadrature and qubit
        for i, ydata in enumerate(ydata_lab):
            for k in range(nq):
                # Plot data points (excluding first and last points)
                ax[k].plot(data["xpts"][k][1:-1], data[ydata][k][1:-1], "o-")

                if fit:
                    # Get fit parameters and covariance
                    p = data["fit_" + ydata + "_" + str(k)]
                    pCov = data["fit_err_" + ydata + "_" + str(k)]
                    
                    # Create caption with fit parameters
                    caption = ""
                    for j in range(len(caption_params)):
                        if j > 0:
                            caption += "\n"
                        if isinstance(caption_params[j]["index"], int):
                            ind = caption_params[j]["index"]
                            caption += caption_params[j]["format"].format(
                                val=(p[ind]), err=np.sqrt(pCov[ind, ind])
                            )
                        else:
                            var = caption_params[j]["index"]
                            caption += caption_params[j]["format"].format(
                                val=data[var + "_" + ydata]
                            )
                    
                    # Plot fit curve
                    ax[k].plot(
                        data["xpts"][k][1:-1],
                        fitfunc(data["xpts"][k][1:-1], *p),
                        label=caption,
                    )
                    
                # Set axis labels
                ax[k].set_ylabel(ylabels[i])
                ax[k].set_xlabel(xlabel)
                ax[k].legend()
                
                # Show initial guess if in debug mode
                if debug:
                    pinit = data["init_guess_" + ydata]
                    print(pinit)
                    ax[i].plot(
                        data["xpts"],
                        fitfunc(data["xpts"], *pinit),
                        label="Initial Guess",
                    )

        # Show histograms if requested
        if show_hist:
            fig2, ax = plt.subplots(1, nq, figsize=(3 * nq, 3))
            for i in range(nq):
                ax[i].plot(
                    data["bin_centers"][i],
                    data["hist"][i] / np.sum(data["hist"][i]),
                    "o-",
                )
                ax[i].set_xlabel("I (ADC units)")
            ax[0].set_ylabel("Probability")
            fig2.tight_layout()

        # Save figure if created in this method
        if save_fig:
            self.save_fig(fig)

    def make_hist(self, prog):
        """
        Generate histograms of single-shot measurement results for both qubits.

        Args:
            prog: QickProgram instance to collect shots from

        Returns:
            Tuple of (bin_centers, hist) containing histogram data for both qubits
        """
        offset = []
        shots_i, shots_q = [], []
        
        # Get DC offset for each qubit channel
        for q in self.cfg.expt.qubit_chan:
            offset.append(self.soccfg._cfg["readouts"][q]["iq_offset"])
            
        # Collect shots from the program
        shots = prog.collect_shots(offset=offset)
        shots_i = shots[0]
        shots_q = shots[1]
        
        # Generate histograms for each qubit
        hist, bin_centers = [], []
        for q in range(len(self.cfg.expt.qubit_chan)):
            h, bin_edges = np.histogram(shots_i[q], bins=60)
            bin_centers.append((bin_edges[:-1] + bin_edges[1:]) / 2)
            hist.append(h)
            
        return bin_centers, hist

    def run(
        self,
        progress=True,
        analyze=True,
        display=True,
        save=True,
        min_r2=0.9,
        max_err=0.1,
    ):
        """
        Run the complete two-qubit experiment workflow.

        This method executes the full experiment sequence:
        1. Acquire data for both qubits
        2. Analyze results
        3. Display plots
        4. Save data to disk
        5. Determine if the experiment was successful

        Args:
            progress: Whether to show progress bar during acquisition (default: True)
            analyze: Whether to perform data analysis (default: True)
            display: Whether to display results (default: True)
            save: Whether to save data to disk (default: True)
            min_r2: Minimum R² value for acceptable fit (default: 0.9)
            max_err: Maximum error for acceptable fit (default: 0.1)
        """
        if min_r2 is None:
            min_r2 = 0.1
        if max_err is None:
            max_err = 0.5

        # Execute experiment workflow
        self.go(progress=progress, analyze=analyze, display=display, save=save)
        
        # Determine if experiment was successful
        if (
            "fit_err" in self.data
            and "r2" in self.data
            and self.data["fit_err"] < max_err
            and self.data["r2"] > min_r2
        ):
            self.status = True
        elif "fit_err" not in self.data or "r2" not in self.data:
            pass
        else:
            print("Fit failed")
            self.status = False

    def save_data(self, data=None):
        """
        Save two-qubit experiment data to disk.

        Args:
            data: Data dictionary to save (uses self.data if None)

        Returns:
            Filename where data was saved
        """
        print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname

    def save_fig(self, fig, suffix=''):
        """Save figure to disk."""
        fig.tight_layout()

        file_path = Path(self.fname)
        new_filename = file_path.name.rsplit(".", 1)[0] + suffix + ".png"
        output_path = Path(self.path) / "images" / new_filename

        fig.savefig(output_path)
        plt.show()

    def get_params(self, prog):
        """
        Get swept parameter values from the program for each qubit.

        Args:
            prog: QickProgram instance to get parameters from

        Returns:
            List of arrays of parameter values, one for each qubit
        """
        xpts = []
        
        # Handle both single parameter dict and list of parameter dicts
        if isinstance(self.param, dict):
            param = [self.param]
        else:
            param = self.param
            
        # Extract parameter values for each qubit
        for p in param:
            if p["param_type"] == "pulse":
                xpts.append(prog.get_pulse_param(p["label"], p["param"], as_array=True))
            else:
                xpts.append(prog.get_time_param(p["label"], p["param"], as_array=True))

        return xpts

    def check_params(self, params_def):
        """
        Check for unexpected parameters in the experiment configuration.

        Args:
            params_def: Dictionary defining expected parameters

        Warns:
            If unexpected parameters are found in self.cfg.expt
        """
        unexpected_params = set(self.cfg.expt.keys()) - set(params_def.keys())
        if unexpected_params:
            warnings.warn(f"Unexpected parameters found in params: {unexpected_params}")

    def configure_reset(self):
        """
        Configure active reset parameters for both qubits.

        This method sets up the threshold and timing parameters needed for
        active reset of both qubits based on their individual configurations.
        """
        qi = self.cfg.expt.qubit
        
        # Define reset parameters
        params_def = dict(
            threshold_v=[self.cfg.device.readout.threshold[q] for q in qi],
            read_wait=0.1,
            extra_delay=0.2,
        )
        self.cfg.expt = {**params_def, **self.cfg.expt}
        
        # Calculate readout length for each qubit
        readout_length = [self.cfg.device.readout.readout_length[q] for q in qi]

        # Convert threshold voltage to ADC units for each qubit
        self.cfg.expt["threshold"] = [
            int(
                self.cfg.expt["threshold_v"][q]
                * readout_length[q]
                / 0.0032552083333333335
            )
            for q in range(len(qi))
        ]

    def get_freq(self, fit=True):
        """
        Calculate the correct frequency accounting for mixer and LO offsets.

        This method adjusts the swept frequency values by adding the mixer
        and local oscillator (LO) frequency offsets for the first qubit.

        Args:
            fit: Whether to also adjust fit frequencies (default: True)
        """
        freq_offset = 0
        q = self.cfg.expt.qubit[0]
        
        # Add mixer frequency offset if present
        if "mixer_freq" in self.cfg.hw.soc.dacs.readout:
            freq_offset += self.cfg.hw.soc.dacs.readout.mixer_freq[q]
            
        # Add LO frequency offset if present (used for signal core)
        if "lo_freq" in self.cfg.hw.soc.dacs.readout:
            freq_offset += self.cfg.hw.soc.dacs.readout.lo_freq[q]
            
        # Add mixer frequency from LO section if present
        if "lo" in self.cfg.hw.soc and "mixer_freq" in self.cfg.hw.soc.lo:
            freq_offset += self.cfg.hw.soc.lo.mixer_freq[q]

        # Store adjusted frequencies
        self.data["freq"] = freq_offset + self.data["xpts"]
        self.data["freq_offset"] = freq_offset
