import matplotlib.pyplot as plt
import copy
import numpy as np
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import time
from pathlib import Path
from visdom import Visdom
from scipy.optimize import curve_fit
import yaml
from qick import *
import json

from ...exp_handling.experiment import Experiment
from ...exp_handling.experiment import NpEncoder, YamlNpEncoder
from ...analysis import fitting as fitter
from ...calib import readout_helpers as helpers


def get_current_time_string():
    """
    Get the current time as a formatted string.
    
    Returns:
        bytes: Current time formatted as "YYYY-MM-DD HH:MM:SS" and encoded as ASCII bytes
    """
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    current_time = current_time.encode("ascii", "replace")
    return current_time


class DataProcessor:
    """
    Utility class for common data processing operations.
    Centralizes data processing logic to reduce duplication.
    """
    
    @staticmethod
    def process_iq_data(iq_list):
        """
        Process I/Q data to extract amplitude, phase, and quadratures.
        
        Args:
            iq_list: Raw I/Q data from program.acquire()
            
        Returns:
            dict: Processed data containing amps, phases, avgi, avgq
        """
        iq = iq_list[0][0]
        amps = np.abs(iq.dot([1, 1j]))
        phases = np.angle(iq.dot([1, 1j]))
        avgi = np.squeeze(iq[..., 0])
        avgq = np.squeeze(iq[..., 1])
        
        return {
            'amps': amps,
            'phases': phases,
            'avgi': avgi,
            'avgq': avgq
        }
    
    @staticmethod
    def create_data_dict(xpts, processed_data, current_time, compact=False, hist_data=None, shots_data=None):
        """
        Create standardized data dictionary.
        
        Args:
            xpts: X-axis parameter values
            processed_data: Processed I/Q data
            current_time: Experiment start time
            compact: Whether to create compact version
            hist_data: Optional histogram data
            shots_data: Optional shots data
            
        Returns:
            dict: Standardized data dictionary
        """
        if compact:
            data = {
                "xpts": xpts,
                "avgi": processed_data['avgi'],
                "avgq": processed_data['avgq'],
                "start_time": current_time,
            }
        else:
            data = {
                "xpts": xpts,
                "avgi": processed_data['avgi'],
                "avgq": processed_data['avgq'],
                "amps": processed_data['amps'],
                "phases": processed_data['phases'],
                "start_time": current_time,
            }
        
        if hist_data:
            data["bin_centers"] = hist_data[0]
            data["hist"] = hist_data[1]
            
        if shots_data:
            data["shots"] = shots_data
            
        # Convert all data to numpy arrays
        for key in data:
            data[key] = np.array(data[key])
            
        return data


class HistogramProcessor:
    """
    Utility class for histogram generation and processing.
    Centralizes histogram logic to reduce duplication.
    """
    
    @staticmethod
    def generate_histogram(i_shots, bins=60, single=True):
        """
        Generate histogram from I quadrature shots.
        
        Args:
            i_shots: I quadrature measurement data
            bins: Number of histogram bins
            single: Whether to treat as single dataset or multiple
            Returns:
                tuple: (bin_centers, hist) data
            """
        if single:
            hist, bin_edges = np.histogram(i_shots, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            hist = []
            bin_centers = []
            for i in range(len(i_shots)):
                hist0, bin_edges0 = np.histogram(i_shots[i], bins=bins, density=True)
                hist.append(hist0)
                bin_centers.append((bin_edges0[:-1] + bin_edges0[1:]) / 2)
        
        return bin_centers, hist
    
    @staticmethod
    def collect_shots_from_prog(prog, offset=0):
        """
        Collect shots from program with offset correction.
        
        Args:
            prog: QickProgram instance
            offset: DC offset correction
            
        Returns:
            tuple: (i_shots, q_shots) data
        """
        i_shots_vec, q_shots_vec = prog.collect_shots(offset=offset)
        # i_shots = i_shots_vec[0].flatten()
        # q_shots = q_shots_vec[0].flatten()
        return i_shots_vec, q_shots_vec


class DisplayManager:
    """
    Utility class for display operations.
    Centralizes plotting logic to reduce duplication.
    """
    
    @staticmethod
    def setup_plot_layout(plot_all=False, plot_amps=False, rescale=False, ax=None):
        """
        Set up plot layout and return configuration.
        
        Args:
            plot_all: Whether to plot all quadratures
            plot_amps: Whether to plot amplitude and phase
            rescale: Whether to show rescaled data
            ax: Existing axis to use
            
        Returns:
            tuple: (fig, ax, ydata_lab, ylabels, save_fig)
        """
        save_fig = ax is None
        
        if plot_all:
            fig, ax = plt.subplots(3, 1, figsize=(7, 9.5))
            ylabels = ["Amplitude (ADC units)", "I (ADC units)", "Q (ADC units)"]
            ydata_lab = ["amps", "avgi", "avgq"]
        elif plot_amps:
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            ydata_lab = ["amps", "phases"]
            ylabels = ["Amplitude (ADC level)", "Phase (radians)"]
        else:
            if ax is None:
                fig, a = plt.subplots(1, 1, figsize=(7, 4))
                ax = [a]
            else:
                fig = None
                
            if rescale:
                ylabels = ["Excited State Probability"]
                ydata_lab = ["scale_data"]
            else:
                ylabels = ["I (ADC units)"]
                ydata_lab = ["avgi"]
        
        return fig, ax, ydata_lab, ylabels, save_fig
    
    @staticmethod
    def plot_data_with_fit(ax, data, ydata, fitfunc=None, fit_params=None, 
                          caption_params=None, xlabel="", ylabel="", debug=False):
        """
        Plot data with optional fit curve.
        
        Args:
            ax: Matplotlib axis
            data: Data dictionary
            ydata: Y-data key to plot
            fitfunc: Fit function
            fit_params: Fit parameters
            caption_params: Parameters for caption
            xlabel: X-axis label
            ylabel: Y-axis label
            debug: Whether to show debug info
        """
        # Plot data points (excluding first and last points)
        ax.plot(data["xpts"][1:-1], data[ydata][1:-1], "o-")
        
        if fit_params is not None and fitfunc is not None:
            p = fit_params
            pCov = data.get("fit_err_" + ydata, None)
            
            # Create caption with fit parameters
            caption = ""
            if caption_params:
                for j, param in enumerate(caption_params):
                    if j > 0:
                        caption += "\n"
                    if isinstance(param["index"], int):
                        ind = param["index"]
                        if pCov is not None:
                            caption += param["format"].format(
                                val=p[ind], err=np.sqrt(pCov[ind, ind])
                            )
                        else:
                            caption += param["format"].format(val=p[ind], err=0)
                    else:
                        var = param["index"]
                        caption += param["format"].format(
                            val=data[var + "_" + ydata]
                        )
            
            # Plot fit curve
            ax.plot(data["xpts"][1:-1], fitfunc(data["xpts"][1:-1], *p), label=caption)
            ax.legend()
        
        # Set axis labels
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        
        # Show initial guess if in debug mode
        if debug and "fit_init_" + ydata in data:
            pinit = data["fit_init_" + ydata]
            ax.plot(data["xpts"], fitfunc(data["xpts"], *pinit), label="Initial Guess")


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
        self, prog_name, progress=True, get_hist=True, single=True, compact=False, shots=False
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
            single: Whether to collect shots for the entire experiment together, or separately for each point in the sweep
            compact: Whether to return a compact data dictionary with fewer fields
            shots: Whether to collect individual shots data

        Returns:
            Dictionary containing measurement data including:
            - xpts: Swept parameter values
            - avgi/avgq: I and Q quadrature data
            - amps/phases: Amplitude and phase data
            - bin_centers/hist: Histogram data (if get_hist=True)
        """
        # Set appropriate final delay based on whether active reset is enabled
        final_delay = self._get_final_delay()
        #print(f"Final delay: {final_delay}")

        # Create program instance
        kwargs = {'final_delay':final_delay, 'cfg':self.cfg}
        # if self.cfg.expt.active_reset: 
        #     kwargs['final_wait']=None
        prog = prog_name(soccfg=self.soccfg,**kwargs)

        # Record start time
        current_time = get_current_time_string()

        # Run the program and acquire data
        iq_list = prog.acquire(
            self.im[self.cfg.aliases.soc],
            rounds=self.cfg.expt.rounds,
            threshold=None,
            progress=progress,
        )

        # Get swept parameter values
        xpts = self.get_params(prog)

        # Process I/Q data
        processed_data = DataProcessor.process_iq_data(iq_list)

        # Generate histogram if requested
        hist_data = None
        if get_hist:
            hist_data = self.make_hist(prog, single=single)

        # Collect shots if requested
        shots_data = None
        if shots:
            shots_data = prog.collect_shots()

        # Compile data dictionary
        data = DataProcessor.create_data_dict(
            xpts, processed_data, current_time, compact, hist_data, shots_data
        )

        self.data = data
        return data

    def _get_final_delay(self):
        """Get appropriate final delay based on active reset configuration."""
        if "active_reset" in self.cfg.expt and self.cfg.expt.active_reset:
            #return self.cfg.device.readout.readout_length[self.cfg.expt.qubit[0]]+4
            #return 10
            return 1
        else:
            return self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

    def analyze(
        self,
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
            data: Data dictionary to analyze (uses self.data if None)
            fit: Whether to perform fitting
            use_i: Whether to use I quadrature for fitting (auto-determined if None)
            get_hist: Whether to generate histogram and scale data
            verbose: Whether to print fit quality metrics
            inds: Indices of fit parameters to include in error calculation (uses all if None)
            **kwargs: Additional arguments passed to the fitter

        Returns:
            Data dictionary with added fit results
        """
        if data is None:
            data = self.data

        # Determine which data sets to fit
        ydata_lab = ["amps", "avgi", "avgq"]

        # Scale data based on histogram if requested
        if get_hist:
            self.scale_ge()
            ydata_lab.append("scale_data")

        # Perform fits on each data set
        for ydata in ydata_lab:
            self._fit_data_series(data, ydata, **kwargs)

        # Determine best fit and calculate quality metrics
        self._determine_best_fit(data, use_i, inds, verbose)

        self.get_status()
        return data

    def _fit_data_series(self, data, ydata, **kwargs):
        """Fit a single data series."""
        (
            data["fit_" + ydata],
            data["fit_err_" + ydata],
            data["fit_init_" + ydata],
        ) = self.fitterfunc(data["xpts"][1:-1], data[ydata][1:-1], **kwargs)

    def _determine_best_fit(self, data, use_i, inds, verbose):
        """Determine best fit and calculate quality metrics."""
        if use_i is None:
            use_i = self.cfg.device.qubit.tuned_up[self.cfg.expt.qubit[0]]
        
        if use_i:
            i_best = "avgi"
            fit_pars = data["fit_avgi"]
            fit_err = data["fit_err_avgi"]
        else:
            fit_pars, fit_err, i_best = fitter.get_best_fit(data, self.fitfunc)

        # Calculate goodness-of-fit (R²)
        r2 = fitter.get_r2(data["xpts"][1:-1], data[i_best][1:-1], self.fitfunc, fit_pars)
        data["r2"] = r2
        data["best_fit"] = fit_pars
        i_best = i_best.encode("ascii", "ignore")
        data["i_best"] = i_best

        if inds is None:
            inds = np.arange(len(fit_err))

        fit_err = fit_err[inds]
        fit_pars = np.array(fit_pars)
        data["fit_err_par"] = np.sqrt(np.diag(fit_err)) / fit_pars[inds]
        fit_err = np.mean(np.abs(data["fit_err_par"]))
        data["fit_err"] = fit_err

        if verbose:
            print(f"R2:{r2:.3f}\tFit par error:{fit_err:.3f}\t Best fit:{i_best}")

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

        # Set up plot layout
        fig, ax, ydata_lab, ylabels, save_fig = DisplayManager.setup_plot_layout(
            plot_all, False, rescale, ax
        )

        if fig and title:
            if plot_all:
                fig.suptitle(title)
            else:
                ax[0].set_title(title)

        # Plot each data set
        for i, ydata in enumerate(ydata_lab):
            fit_params = data.get("fit_" + ydata) if fit else None
            
            DisplayManager.plot_data_with_fit(
                ax[i], data, ydata, fitfunc, fit_params, 
                caption_params, xlabel, ylabels[i], debug
            )

        # Show histogram if requested
        if show_hist:
            self._show_histogram(data)

        # Save figure if created in this method
        if save_fig and fig:
            self.save_fig(fig)
            self.save_config()
            plt.show()

    def _show_histogram(self, data):
        """Display histogram plot."""
        fig2, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.plot(data["bin_centers"], data["hist"], "o-")
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

    def save_fig(self, fig, suffix=''):
        """Save figure to disk."""
        fig.tight_layout()

        file_path = Path(self.fname)
        parent_dir = file_path.parent
        new_filename = file_path.name.rsplit(".", 1)[0] + suffix + ".png"
        output_path = parent_dir / "images" / new_filename

        fig.savefig(output_path)
        plt.show()

    def save_config(self, suffix=''):
        file_path = Path(self.fname)
        parent_dir = file_path.parent
        new_filename = file_path.name.rsplit(".", 1)[0] + suffix + ".yml"
        output_path = parent_dir / "images" / new_filename
        
        with open(output_path, "w") as f:
            YamlNpEncoder.dump(self.cfg, f, default_flow_style=None)



        

    def make_hist(self, prog, single=True):
        """
        Generate histogram of single-shot measurement results.

        Args:
            prog: QickProgram instance to collect shots from
            single: Whether to collect shots for the entire experiment together

        Returns:
            Tuple of (bin_centers, hist) containing histogram data
        """
        offset = self.soccfg._cfg["readouts"][self.cfg.expt.qubit_chan]["iq_offset"]
        i_shots_vec, q_shots_vec = prog.collect_shots(offset=offset)
        
        # q_shots = q_shots_vec[0].flatten()
        if single:
            i_shots = i_shots_vec[0][:,:,0].flatten()
        else:
            # Handle multiple sweep points separately
            i_shots_vec, q_shots_vec = prog.collect_shots(offset=offset)
            i_shots = []
            for j in range(i_shots_vec.shape[1]):
                i_shots.append(i_shots_vec[0][0, j, :])
        
        return HistogramProcessor.generate_histogram(i_shots, single=single)

    def qubit_run(
        self,
        qi=0,
        go=True,
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
        """
        Run the quantum experiment with qubit-specific configuration.

        This method configures experiment settings based on the qubit index,
        including active reset setup and display options, then executes
        the complete experiment workflow.

        Args:
            qi: Qubit index to use for the experiment
            progress: Whether to show progress bar during acquisition
            analyze: Whether to perform data analysis
            display: Whether to display results
            save: Whether to save data to disk
            print: Whether to print experiment configuration instead of running
            min_r2: Minimum R² value for acceptable fit
            max_err: Maximum error for acceptable fit
            disp_kwargs: Display options dictionary (e.g., plot_all, rescale)
            **kwargs: Additional arguments passed to the run method
        """
        # Configure active reset if enabled
        if self.cfg.expt.active_reset:
            self.configure_reset()

        # Configure display options based on qubit state
        disp_kwargs = self._configure_display_options(qi, disp_kwargs)

        # Run the experiment
        if print:
            self.print()
        elif go:
            self.run(
                analyze=analyze,
                display=display,
                save=save,
                progress=progress,
                min_r2=min_r2,
                max_err=max_err,
                disp_kwargs=disp_kwargs,
            )

    def _configure_display_options(self, qi, disp_kwargs):
        """Configure display options based on qubit state."""
        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {"plot_all": True}
        
        if (self.cfg.device.readout.rescale[qi] or 
            (disp_kwargs is not None and "rescale" in disp_kwargs)):
            disp_kwargs = {"rescale": True}
        
        return disp_kwargs or {}

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
        # Set default values
        if min_r2 is None:
            min_r2 = 0.1
        if max_err is None:
            max_err = 1
        if disp_kwargs is None:
            disp_kwargs = {}

        # Execute experiment workflow
        data = self.acquire(progress)
        if analyze:
            data = self.analyze(data, **kwargs)
        if save:
            self.save_data(data)
        if display:
            if not analyze:
                disp_kwargs["fit"] = False
            self.display(data, **disp_kwargs)

    def save_data(self, data=None, verbose=False):
        """Save experiment data to disk."""
        if verbose:
            print(f"Saving {self.fname}")
        super().save_data(data=data)
        return self.fname

    def print(self):
        """Print out the experimental config"""
        for key, value in self.cfg.expt.items():
            print(f"{key}: {value}")

    def get_status(self, max_err=1, min_r2=0.1):
        """
        Determine if experiment was successful based on fit quality metrics.

        Args:
            max_err: Maximum acceptable fit error threshold
            min_r2: Minimum acceptable R² value threshold
        """
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
            self.status = False

    def get_params(self, prog):
        """
        Get swept parameter values from the program.

        Args:
            prog: QickProgram instance to get parameters from

        Returns:
            Array of parameter values
        """
        if self.param["param_type"] == "pulse":
            xpts = prog.get_pulse_param(
                self.param["label"], self.param["param"], as_array=True
            )
        else:
            xpts = prog.get_time_param(
                self.param["label"], self.param["param"], as_array=True
            )
        return xpts

    def check_params(self, params_def):
        """Check for unexpected parameters in the experiment configuration."""
        if self._check_params:
            unexpected_params = set(self.cfg.expt.keys()) - set(params_def.keys())
            if unexpected_params:
                print(f"Unexpected parameters found in params: {unexpected_params}")

    def configure_reset(self):
        """Configure active reset parameters for the experiment."""
        qi = self.cfg.expt.qubit[0]
        params_def = dict(
            threshold_v=self.cfg.device.readout.threshold[qi],
            read_wait=0.1,
            extra_delay=0.2 + 1.55/self.cfg.device.readout.kappa[qi],
            reset=self.cfg.device.readout.reset[qi],
        )
        self.cfg.expt = {**params_def, **self.cfg.expt}

        adc = self.cfg.hw.soc.adcs.readout.ch[qi]
        self.cfg.expt["threshold"] = int(
            self.cfg.expt["threshold_v"]
            * self.cfg.device.readout.readout_length[qi]
            * self.soccfg._get_ch_cfg(ro_ch=adc)['f_fabric']
        )

    def run_loop(self, prog, x_sweep, progress=True):
        """
        Run a loop-based acquisition with custom parameter sweep points.

        Args:
            prog: QickProgram class to run for each sweep point
            x_sweep: List of dictionaries defining the parameter sweep
            progress: Whether to show progress bar during acquisition

        Returns:
            Dictionary containing measurement data for all sweep points
        """
        cfg_dict = {
            "soc": self.soccfg,
            "cfg_file": self.config_file,
            "im": self.im,
            "expt_path": "dummy",
        }
        exp = QickExperimentLoop(
            cfg_dict=cfg_dict,
            prefix="dummy",
            progress=progress,
            qi=self.cfg.expt.qubit[0],
        )
        exp.cfg.expt = copy.deepcopy(self.cfg.expt)
        exp.param = self.param
        exp.cfg.expt.expts = 1
        data = exp.acquire(prog, x_sweep, progress=progress)
        return data

    def get_freq(self, fit=True):
        """Calculate the correct frequency accounting for mixer and LO offsets."""
        freq_offset = 0
        q = self.cfg.expt.qubit[0]
        if "mixer_freq" in self.cfg.hw.soc.dacs.readout:
            freq_offset += self.cfg.hw.soc.dacs.readout.mixer_freq[q]
        if "lo_freq" in self.cfg.hw.soc.dacs.readout:
            freq_offset += self.cfg.hw.soc.dacs.readout.lo_freq[q]
        if "lo" in self.cfg.hw.soc and "mixer_freq" in self.cfg.hw.soc.lo:
            freq_offset += self.cfg.hw.soc.lo.mixer_freq[q]

        self.data["freq"] = freq_offset + self.data["xpts"]
        self.data["freq_offset"] = freq_offset

    def scale_ge(self):
        """Scale g->0 and e->1 based on histogram data"""
        hist = self.data["hist"]
        bin_centers = self.data["bin_centers"]
        scale_data, hist_fit = helpers.fit_hist(bin_centers, hist, self.data["avgi"])
        self.data["scale_data"] = scale_data
        self.data["hist_fit"] = hist_fit


class QickExperimentLoop(QickExperiment):
    """
    Extension of QickExperiment for loop-based parameter sweeps.

    This class implements experiments where a parameter is swept through a range of values.
    It handles the loop iteration, data collection for each parameter value, and
    aggregation of results into a complete dataset.
    """

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=False, qi=0, check_params=True):
        """
        Initialize the QickExperimentLoop.

        Args:
            cfg_dict: Configuration dictionary
            prefix: Prefix for saved data files
            progress: Whether to show progress bars
            qi: Qubit index to use for the experiment
            check_params: Whether to check for unexpected parameters
        """
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, check_params=check_params)

    def acquire(self, prog_name, x_sweep, progress=True, hist=False):
        """
        Acquire data by running the program for each point in the parameter sweep.

        Args:
            prog_name: Class reference to the QickProgram to run
            x_sweep: List of dictionaries defining the parameter sweep
            progress: Whether to show progress bar
            hist: Whether to collect histogram data

        Returns:
            Dictionary containing measurement data for all sweep points
        """
        final_delay = self._get_final_delay()

        # Initialize data dictionary
        data = {"xpts": [], "avgi": [], "avgq": [], "amps": [], "phases": []}
        shots_i = []

        # Record start time
        current_time = get_current_time_string()

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
            data = self._stow_data(iq_list, data)

            # Collect individual shots for histogram
            offset = self.soccfg._cfg["readouts"][self.cfg.expt.qubit_chan]["iq_offset"]
            shots_i_new, shots_q = prog.collect_shots(offset=offset)
            shots_i.append(shots_i_new)

            # Store parameter value
            xpt = self.get_params(prog)
            data["xpts"].append(xpt)

        # Generate histogram from all collected shots
        bin_centers, hist = self._make_hist_from_shots(shots_i)
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

    def _stow_data(self, iq_list, data):
        """Process and store I/Q data from a measurement."""
        processed_data = DataProcessor.process_iq_data(iq_list)
        
        # Append to data arrays
        data["avgi"].append(processed_data['avgi'])
        data["avgq"].append(processed_data['avgq'])
        data["amps"].append(processed_data['amps'])
        data["phases"].append(processed_data['phases'])
        return data

    def _make_hist_from_shots(self, shots_i):
        """Generate histogram from collected shots."""
        hist, bin_edges = np.histogram(shots_i, bins=60)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, hist


class QickExperiment2DBase(QickExperimentLoop):
    """
    Base class for 2D experiments with common functionality.
    Reduces duplication between 2D experiment classes.
    """

    def _setup_2d_data_structure(self):
        """Initialize data structure for 2D experiments."""
        return {"avgi": [], "avgq": [], "amps": [], "phases": [], "time": []}

    def _determine_y_axis_values(self, data, y_sweep):
        """Determine y-axis values (time or parameter)."""
        if "count" in [y_sweep[j]["var"] for j in range(len(y_sweep))]:
            # Convert time to hours for time-based sweeps
            data["ypts"] = (data["time"] - np.min(data["time"])) / 3600
        else:
            # Use the swept parameter values
            data["ypts"] = y_sweep[0]["pts"]

    def _store_sweep_parameters(self, data, y_sweep):
        """Store y-axis parameter sweep values."""
        for j in range(len(y_sweep)):
            data[y_sweep[j]["var"] + "_pts"] = y_sweep[j]["pts"]

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
        Display 2D experiment results using centralized display logic.

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
        if np.ndim(x_sweep) > 1:
            x_sweep = x_sweep[0]
        y_sweep = data["ypts"]

        # Determine whether to save the figure
        save_fig = ax is None

        # Configure plot layout based on what to display
        if plot_both:
            # Create 2-panel figure for I and Q
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            ydata_lab = ["avgi", "avgq"]
            ylabels = ["I (ADC level)", "Q (ADC level)"]
            fig.suptitle(title)
        elif plot_amps:
            # Create 2-panel figure for amplitude and phase
            fig, ax = plt.subplots(2, 1, figsize=(8, 10))
            ydata_lab = ["amps", "phases"]
            ylabels = ["Amplitude (ADC level)", "Phase (radians)"]
            fig.suptitle(title)
        else:
            # Create single panel figure for I quadrature
            if ax is None:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.set_title(title)
            ydata_lab = ["avgi"]
            ax = [ax]
            ylabels = ["I (ADC level)"]

        # Create 2D color plot for each data set
        for i, ydata in enumerate(ydata_lab):
            # Create heatmap using pcolormesh
            ax[i].pcolormesh(
                x_sweep, y_sweep, data[ydata], cmap="viridis", shading="auto", rasterized=True
            )
            # Add colorbar with label
            plt.colorbar(ax[i].collections[0], ax=ax[i], label=ylabels[i])
            # Set axis labels
            ax[i].set_xlabel(xlabel)
            ax[i].set_ylabel(ylabel)

            # Use log scale for y-axis if specified in configuration
            if "log" in self.cfg.expt and self.cfg.expt.log:
                ax[i].set_yscale("log")

            #ax[i].grid(False)

        # Save figure if created in this method
        if save_fig and fig:
            self.save_fig(fig)
            plt.show()

    def analyze(self, fitfunc=None, fitterfunc=None, data=None, fit=True, rescale=False, **kwargs):
        """
        Analyze 2D experiment data by fitting each row.

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
        ydata_lab = ["avgi"]  # Typically only fit I quadrature for speed
        if rescale: 
            self.scale_ge()

        # Fit each row (y value) separately
        if fit: 
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
    
    def scale_ge(self):
        """Scale g->0 and e->1 based on histogram data"""
        self.data["scale_data"] = []
        self.data["hist_fit"] = []
        for j in range(len(self.data["ypts"])):
            hist = self.data["hist"][j]
            bin_centers = self.data["bin_centers"][j]
            scale_data, hist_fit = helpers.fit_hist(bin_centers, hist, self.data["avgi"][j,:])
            self.data["scale_data"].append(scale_data)
            self.data["hist_fit"].append(hist_fit)


class QickExperiment2D(QickExperiment2DBase):
    """
    Extension of QickExperimentLoop for 2D parameter sweeps.

    This class implements experiments where two parameters are swept:
    - The x-axis parameter is typically swept by the program (e.g., pulse frequency)
    - The y-axis parameter is swept by this class (e.g., time, power, etc.)
    """

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=0):
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

    def acquire(self, prog_name, y_sweep, progress=True):
        """
        Acquire data for a 2D parameter sweep.

        Args:
            prog_name: Class reference to the QickProgram to run
            y_sweep: List of dictionaries defining the y-axis parameter sweep
            progress: Whether to show progress bar

        Returns:
            Dictionary containing 2D measurement data
        """
        # Initialize data dictionary
        data = self._setup_2d_data_structure()
        yvals = np.arange(len(y_sweep[0]["pts"]))

        # Record start time
        current_time = get_current_time_string()

        # Iterate through each point in the y-axis parameter sweep
        for i in tqdm(yvals):
            # Update configuration with current y-axis parameter value
            for j in range(len(y_sweep)):
                self.cfg.expt[y_sweep[j]["var"]] = y_sweep[j]["pts"][i]

            # Create and run program for this y value
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
            data = self._stow_data(iq_list, data)
            data["time"].append(time.time())

        # Get x-axis parameter values from the program
        data["xpts"] = self.get_params(prog)

        # Set y-axis values
        self._determine_y_axis_values(data, y_sweep)
        self._store_sweep_parameters(data, y_sweep)

        # Convert all data to numpy arrays
        for k, a in data.items():
            data[k] = np.array(a)

        # Add metadata and store data
        data["start_time"] = current_time
        self.data = data
        return data


class QickExperiment2DSimple(QickExperiment2DBase):
    """
    Simplified version of QickExperiment2D for nested experiments.

    This class provides a simpler interface for 2D experiments where the
    x-axis parameter is swept by a separate experiment instance.
    """

    def __init__(self, cfg_dict=None, prefix="QickExp", progress=None, qi=0, live_plot=False):
        """
        Initialize the QickExperiment2DSimple.

        Args:
            cfg_dict: Configuration dictionary
            prefix: Prefix for saved data files
            progress: Whether to show progress bars
            qi: Qubit index to use for the experiment
            live_plot: Whether to enable live plotting with Visdom during acquisition
        """
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        self.live_plot = live_plot
        self.save_interim=True
        self.viz = None
        self.viz_window = None

        if self.live_plot:
            try:
                self.viz = Visdom()
                assert self.viz.check_connection(), "Visdom server not running."
                self.viz_window = self.viz.text("Starting 2D Scan...", opts={"title": "Qick 2D Scan"})
            except Exception as e:
                print(f"[Visdom] Could not connect: {e}")
                self.live_plot = False

    def acquire(self, y_sweep, progress=False):
        """
        Acquire data for a 2D parameter sweep using a nested experiment.

        Args:
            y_sweep: List of dictionaries defining the y-axis parameter sweep
            progress: Whether to show progress bar

        Returns:
            Dictionary containing 2D measurement data
        """
        # Initialize data dictionary
        data = {}
        yvals = np.arange(len(y_sweep[0]["pts"]))
        data['time'] = []

        # Iterate through each point in the y-axis parameter sweep
        for i in tqdm(yvals):
            # Update nested experiment configuration
            for j in range(len(y_sweep)):
                self.expt.cfg.expt[y_sweep[j]["var"]] = y_sweep[j]["pts"][i]

            # Run the nested experiment
            data_new = self.expt.acquire(progress=progress)

            # Store all data from the nested experiment
            for key in data_new:
                if i == 0:
                    data[key] = []
                data[key].append(data_new[key])
            
            data["time"].append(time.time())
            
            # Live update heatmap plot using Visdom
            if self.live_plot and i>0:
                self._plot_live_update(data, y_sweep)
            if self.save_interim: 
                super().save_data(data=data)

        # Set y-axis values
        self._determine_y_axis_values(data, y_sweep)
        self._store_sweep_parameters(data, y_sweep)

        # Use the x-axis values from the first nested experiment run
        data["xpts"] = data["xpts"][0]
        for k, a in data.items():
            data[k] = np.array(a)
        self.data = data

        return data
    
    def _plot_live_update(self, data, y_sweep):
        """Update the live plot with the latest data."""
        try:
            from io import BytesIO
            import PIL.Image

            # Extract data arrays
            amps_so_far = np.array(data.get("amps", []))
            phases_so_far = np.array(data.get("phases", []))
            avgi_so_far = np.array(data.get("avgi", []))
            avgq_so_far = np.array(data.get("avgq", []))

            if amps_so_far.size > 0:
                # Get axis values
                xvals = data["xpts"][0]
                yvals = np.array(y_sweep[0]["pts"][:amps_so_far.shape[0]])
                
                # Create 4 subplots
                fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=100)

                # Plot each data type
                im1 = axs[0,0].pcolormesh(xvals, yvals, amps_so_far, cmap="viridis", shading="auto")
                axs[0, 0].set_title("Amps")
                plt.colorbar(im1, ax=axs[0, 0])

                im2 = axs[0, 1].pcolormesh(xvals, yvals, phases_so_far, cmap="viridis", shading="auto")
                axs[0, 1].set_title("Phases")
                plt.colorbar(im2, ax=axs[0, 1])

                im3 = axs[1, 0].pcolormesh(xvals, yvals, avgi_so_far, cmap="viridis", shading="auto")
                axs[1, 0].set_title("AvgI")
                plt.colorbar(im3, ax=axs[1, 0])

                im4 = axs[1, 1].pcolormesh(xvals, yvals, avgq_so_far, cmap="viridis", shading="auto")
                axs[1, 1].set_title("AvgQ")
                plt.colorbar(im4, ax=axs[1, 1])

                # Set labels if available
                for ax in axs.flat:
                    if hasattr(self, "ylabel"):
                        ax.set_ylabel(self.ylabel)
                    if hasattr(self,'xlabel'):
                        ax.set_xlabel(self.xlabel)

                plt.tight_layout()

                # Convert to image for Visdom
                buf = BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img = PIL.Image.open(buf).convert("RGB")
                img = np.array(img).transpose(2, 0, 1)

                self.viz.image(img, win=self.viz_window, opts={"title": "Live Channels"})

        except Exception as e:
            print(f"[Visdom] Live plot failed: {e}")


class QickExperiment2DSweep(QickExperiment2DBase):
    """
    Extension of QickExperiment for 2D parameter sweeps with different analysis method.
    
    This class implements experiments where two parameters are swept, similar to QickExperiment2D,
    but uses a different analysis method for fitting the 2D data.
    """
    def acquire(
        self, prog_name, progress=True, get_hist=True, single=True, compact=False, shots=False
    ):
                # Set appropriate final delay based on whether active reset is enabled
        final_delay = self._get_final_delay()

        # Create program instance
        prog = prog_name(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )

        # Record start time
        current_time = get_current_time_string()

        # Run the program and acquire data
        iq_list = prog.acquire(
            self.im[self.cfg.aliases.soc],
            rounds=self.cfg.expt.rounds,
            threshold=None,
            progress=progress,
        )

        # Get swept parameter values
        xpts = self.get_params(prog)

        # Process I/Q data
        processed_data = DataProcessor.process_iq_data(iq_list)

        # Generate histogram if requested
        hist_data = None
        if get_hist:
            hist_data = self.make_hist(prog, single=single)

        # Collect shots if requested
        shots_data = None
        if shots:
            shots_data = prog.collect_shots()

        # Compile data dictionary
        data = DataProcessor.create_data_dict(
            xpts, processed_data, current_time, compact, hist_data, shots_data
        )

        self.data = data
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
        """Display 2D experiment results."""
        # Use the base class display method but ensure proper figure saving
        super().display(data, ax, plot_both, plot_amps, title, xlabel, ylabel, **kwargs)
        
        # Additional display logic specific to this class can be added here if needed
