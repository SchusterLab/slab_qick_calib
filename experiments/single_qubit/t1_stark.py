"""
T1 Stark Effect Measurement Module

This module implements T1 relaxation time measurements under the influence of AC Stark shifts.
The AC Stark effect occurs when a strong off-resonant drive creates a dynamic energy shift
in the qubit levels, effectively changing the qubit frequency and potentially affecting
the relaxation rate T1.

The module provides several experiment types:
1. T1StarkExperiment: Basic T1 measurement with AC Stark drive at fixed amplitude
2. T1StarkPowerExperiment: 2D sweep of T1 vs wait time and Stark drive amplitude
3. T1StarkFreqExperiment: 2D sweep of T1 vs wait time and Stark drive frequency
4. T1StarkPowerSingle: Single-shot T1 measurement with Stark amplitude sweep
5. T1StarkPowerQuadSingle: Quadratic calibrated Stark measurements
6. T1StarkPowerQuad2D: 2D version of quadratic Stark measurements

These experiments are useful for:
- Characterizing qubit frequency stability under drive conditions
- Understanding the impact of crosstalk from neighboring qubits
- Calibrating Stark shift parameters for advanced control protocols
"""

from operator import neg, pos
import matplotlib.pyplot as plt
import numpy as np
from qick import *
import seaborn as sns
from copy import deepcopy

from qick.asm_v2 import QickSweep1D

from ...analysis import fitting as fitter
from ..general.qick_experiment import (
    QickExperiment,
    QickExperiment2DSimple,
    QickExperimentLoop,
)
FIT_FUNC = fitter.expfunc
FITTER_FUNC = fitter.fitexp
XLABEL = "Wait Time ($\mu$s)"

from .t1 import T1Program


class T1StarkExperiment(QickExperiment):
    """
    Basic T1 relaxation time measurement under AC Stark effect.
    
    This experiment measures the T1 decay time while applying an off-resonant
    drive (Stark tone) to the qubit. The Stark tone creates a dynamic frequency
    shift that can affect the relaxation process.
    
    Experimental Configuration:
        start (float): Wait time sweep start [μs]
        span (float): Total span of wait time sweep [μs] 
        expts (int): Number of wait time points in sweep
        reps (int): Number of averages per experiment
        rounds (int): Number of rounds to repeat experiment sweep
        stark_gain (float): Amplitude of the Stark drive [DAC units]
        df (float): Frequency offset of Stark drive from qubit [MHz]
        acStark (bool): Enable AC Stark drive during wait time
        active_reset (bool): Use active reset between measurements
        end_wait (float): Additional wait time after measurement [μs]
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=None,
        display=True,
        analyze=True,
        disp_kwargs={},
        print=False,
        style="",
        min_r2=None,
        max_err=None,
        check_params=True
    ):
        """
        Initialize T1 Stark experiment.
        
        Args:
            cfg_dict: Configuration dictionary containing device and experiment parameters
            qi: Qubit index to measure (default: 0)
            go: Whether to run the experiment immediately (default: True)
            params: Dictionary to override default experiment parameters
            prefix: Prefix for data file names (default: auto-generated)
            progress: Show progress bar during acquisition
            style: Experiment style ("fine" for higher precision, "fast" for quick measurement)
            min_r2: Minimum R² value for acceptable fit quality
            max_err: Maximum error for acceptable fit quality
            check_params: Whether to validate parameter consistency
        """
        # Set default filename prefix if not provided
        if prefix is None:
            prefix = f"t1_stark_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, check_params=check_params)

        # Define default experimental parameters
        params_def = {
            "reps": 3 * self.reps,  # More repetitions for better statistics
            "rounds": self.rounds,  # Number of experiment repetitions
            "expts": 60,  # Number of wait time points
            "start": 0.05,  # Start wait time [μs]
            "span": 3.7 * self.cfg.device.qubit.T1[qi],  # Span ~3.7*T1 for good decay curve
            "acStark": True,  # Enable AC Stark effect
            "active_reset": self.cfg.device.readout.active_reset[qi],  # Use device default
            "qubit": [qi],  # Qubit index as list for compatibility
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],  # ADC readout channel
            "stark_gain": 1,  # Stark drive amplitude [DAC units]
            "end_wait": 0.5,  # Additional wait after measurement [μs]
            "df": 70,  # Stark frequency offset from qubit [MHz]
        }
        
        # Merge user parameters with defaults
        params = {**params_def, **params}
        
        # Adjust parameters based on measurement style
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2  # More rounds for precision
        elif style == "fast":
            params_def["expts"] = 30  # Fewer points for speed

        # Calculate actual Stark drive frequency
        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]

        # Finalize experiment configuration
        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
            
        # Run experiment immediately if requested
        super().qubit_run(
            qi=qi,
            go=go,
            display=display,
            progress=progress,
            analyze=analyze,
            min_r2=min_r2,
            max_err=max_err,
            print=print,
            disp_kwargs=disp_kwargs,
        )

    def acquire(self, progress=False):
        """
        Acquire T1 decay data with AC Stark drive.
        
        Sets up a 1D sweep over wait times and runs the T1 program
        with Stark drive enabled during the wait period.
        
        Args:
            progress: Show progress bar during data acquisition
            
        Returns:
            Dictionary containing acquired IQ data and metadata
        """
        # Set parameter metadata for plotting
        self.param = {"label": "wait", "param": "t", "param_type": "time"}
        
        # Configure wait time sweep from start to start+span
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        
        # Run T1 program with configured sweep
        super().acquire(T1Program, progress=progress)

        return self.data

    def analyze(self, data=None, **kwargs):
        """
        Analyze T1 decay data using exponential fitting.
        
        Fits the decay curve to extract T1 time constant and
        other decay parameters (offset, amplitude).
        
        Args:
            data: Data dictionary to analyze (uses self.data if None)
            **kwargs: Additional arguments passed to fitting routine
            
        Returns:
            Data dictionary with added fit parameters
        """
        if data is None:
            data = self.data

        # Use exponential decay function for fitting
        self.fitfunc = FIT_FUNC
        self.fitterfunc = FITTER_FUNC

        # Perform fitting analysis
        super().analyze(data)

        return self.data

    def display(self, data=None, fit=True, plot_all=False, ax=None, show_hist=False):
        """
        Display T1 Stark measurement results with exponential fit.
        
        Creates a plot showing the T1 decay curve under Stark drive
        with fitted exponential decay and extracted T1 value.
        
        Args:
            data: Data dictionary to plot (uses self.data if None)
            fit: Whether to show fitted curve and parameters
            plot_all: Whether to plot all data points or summary
            ax: Matplotlib axis to plot on (creates new figure if None)
            show_hist: Whether to show histogram of measurement results
        """
        # Get experiment parameters for plot labels
        q = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[q]
        
        # Configure plot labels and title
        title = f"$T_1$ Stark Q{q} Freq: {df}, Amp: {self.cfg.expt.stark_gain}"
        
        # Configure fit display parameters
        caption_params = [
            {"index": 2, "format": "$T_1$ fit: {val:.3} $\pm$ {err:.2} $\mu$s"},
        ]

        # Use parent class display method with configured parameters
        super().display(
            data=data,
            ax=ax,
            plot_all=plot_all,
            title=title,
            xlabel=XLABEL,
            fit=fit,
            show_hist=show_hist,
            fitfunc=self.fitfunc,
            caption_params=caption_params,
        )


class T1StarkPowerExperiment(QickExperiment2DSimple):
    """
    2D T1 Stark experiment sweeping both wait time and Stark drive amplitude.
    
    This experiment measures T1 decay curves at different Stark drive powers
    to characterize how the drive amplitude affects the relaxation rate.
    Creates a 2D map of T1 vs wait time and Stark power.
    
    Experimental Configuration:
        start_gain (float): Starting Stark drive amplitude [DAC units]
        end_gain (float): Ending Stark drive amplitude [DAC units]
        expts_gain (int): Number of gain points in sweep
        end_wait (float): Additional wait time after measurement [μs]
        
    The experiment inherits T1 sweep parameters from T1StarkExperiment.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=False,
        style="",
        min_r2=None,
        max_err=None,
        live_plot=False,
    ):
        """
        Initialize 2D T1 Stark power sweep experiment.
        
        Args:
            cfg_dict: Configuration dictionary containing device parameters
            qi: Qubit index to measure (default: 0)
            go: Whether to run the experiment immediately (default: True)
            params: Dictionary to override default experiment parameters
            prefix: Prefix for data file names (default: auto-generated)
            progress: Show progress bar during acquisition
            style: Experiment style ("fine"/"fast") passed to inner T1 experiment
            min_r2: Minimum R² value for acceptable fit quality
            max_err: Maximum error for acceptable fit quality
        """
        # Set default filename prefix if not provided
        if prefix == "":
            prefix = f"t1_stark_amp_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, qi=qi, prefix=prefix, progress=progress, live_plot=live_plot)

        # Define default parameters for power sweep
        params_def = {
            "end_gain": self.cfg.device.qubit.max_gain,  # Maximum safe gain for device
            "expts_gain": 20,  # Number of gain points to measure
            "start_gain": 0.15,  # Starting gain (15% of max)
        }
        
        # Create inner T1StarkExperiment without running it
        self.expt = T1StarkExperiment(
            cfg_dict, qi=qi, go=False, params=params, style=style, 
            check_params=False
        )
        
        # Merge all parameter configurations
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = {**params_def, **params}
        
        # Run experiment if requested
        if go:
            super().run(min_r2=min_r2, max_err=max_err, progress=progress)

    def acquire(self, progress=False):
        """
        Acquire 2D T1 data sweeping wait time and Stark amplitude.
        
        For each Stark amplitude, runs a full T1 decay measurement,
        then fits each curve to extract T1 parameters.
        
        Args:
            progress: Show progress bar during data acquisition
            
        Returns:
            Dictionary containing 2D measurement data and fit parameters
        """
        # Ensure end_gain doesn't exceed device maximum
        self.cfg.expt["end_gain"] = np.min(
            [self.cfg.device.qubit.max_gain, self.cfg.expt["end_gain"]]
        )
        
        # Create linear array of gain points from start to end
        gainpts = np.linspace(
            self.cfg.expt["start_gain"],
            self.cfg.expt["end_gain"],
            self.cfg.expt["expts_gain"],
        )

        # Configure 2D sweep with gain as the outer loop variable
        y_sweep = [{"var": "stark_gain", "pts": gainpts}]
        self.ylabel = "Gain (DAC units)"
        self.xlabel = XLABEL
        super().acquire(y_sweep=y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze 2D T1 Stark data by fitting each power curve.
        
        Fits exponential decay to each gain point to extract:
        - Offset: Asymptotic decay value
        - Amplitude: Initial signal amplitude  
        - T1: Characteristic decay time
        
        Args:
            data: Data dictionary to analyze (uses self.data if None)
            fit: Whether to perform exponential fitting
            **kwargs: Additional arguments passed to fitting routine
        """
        if data is None:
            data = self.data

        # Use exponential decay fitting for each gain point
        super().analyze(fitfunc=FIT_FUNC, fitterfunc=FITTER_FUNC, data=data)

        # Extract fit parameters for each gain point
        num_gains = len(data["stark_gain_pts"])
        
        # Offset parameter (baseline decay level)
        data["offset"] = [
            data["fit_avgi"][i][0] for i in range(num_gains)
        ]
        
        # Amplitude parameter (initial signal strength)
        data["amp"] = [
            data["fit_avgi"][i][1] for i in range(num_gains)
        ]
        
        # T1 parameter (decay time constant)
        data["t1"] = [
            data["fit_avgi"][i][2] for i in range(num_gains)
        ]

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        """
        Display 2D T1 Stark power results with fit parameter analysis.
        
        Creates multiple plots:
        1. 2D heatmap of T1 vs wait time and power
        2. Fit parameter trends vs power (offset, amplitude, T1)
        3. Individual decay curves for each power setting
        
        Args:
            data: Data dictionary to plot (uses self.data if None)
            fit: Whether to show fit results and parameter plots
            plot_both: Whether to show both I and Q data
            **kwargs: Additional plotting arguments
        """
        if data is None:
            data = self.data
            
        # Get experiment parameters for plot labels
        qubit = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[qubit]

        # Configure plot titles and labels
        title = f"T1 Stark Power Q{qubit} Freq: {df}"
        
        
        # Display main 2D plot using parent class method
        super().display(plot_both=plot_both, title=title, xlabel=self.xlabel, ylabel=self.ylabel)

        # Plot fit parameter trends if fitting was performed
        xval = data["stark_gain_pts"]
        if fit:
            fig, ax = plt.subplots(3, 1, figsize=(6, 8))
            
            # Plot each fit parameter vs gain
            ax[0].plot(xval, data["offset"], 'o-')
            ax[1].plot(xval, data["amp"], 'o-')
            ax[2].plot(xval, data["t1"], 'o-')

            # Configure axis labels and titles
            ax[0].set_ylabel("Offset")
            ax[1].set_ylabel("Amplitude")
            ax[2].set_ylabel("T1 [μs]")
            ax[0].set_title(f"T1 Stark Power Q{qubit} Freq: {df}")
            ax[2].set_xlabel("Gain (DAC units)")

        # Plot individual decay curves for each gain setting
        sns.set_palette("coolwarm", len(data["stark_gain_pts"]))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Plot each gain's decay curve with different colors
        for i in range(len(data["stark_gain_pts"])):
            ax.plot(data["xpts"], data["avgi"][i], linewidth=0.5)
            
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel("Signal (ADC units)")
        ax.set_title(f"T1 Decay Curves - {title}")
        fig.tight_layout()

        # Save additional figure and display
        super().save_fig(fig, '_individual_curves')
        plt.show()


class T1StarkFreqExperiment(QickExperiment2DSimple):
    """
    Stark Power Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz),
        step_f: frequency step (MHz),
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=False,
        style="",
        min_r2=None,
        max_err=None,
        live_plot=False,
    ):

        if prefix == "":
            prefix = f"t1_stark_freq_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, qi=qi, prefix=prefix, progress=progress, live_plot=live_plot)

        params_def = {
            "span_f": 200,
            "expts_f": 30,
            "start_df": 10,
            "end_wait": 0.5,
        }
        params = {**params_def, **params}
        params["start_f"] = self.cfg.device.qubit.f_ge[qi] + params["start_df"]
        self.expt = T1StarkExperiment(
            cfg_dict, qi=qi, go=False, params=params, style=style, check_params=False
        )
        params = {**params_def, **params}
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params

        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):

        freqpts = np.linspace(
            self.cfg.expt["start_f"],
            self.cfg.expt["start_f"] + self.cfg.expt["span_f"],
            self.cfg.expt["expts_f"],
        )

        y_sweep = [{"var": "stark_freq", "pts": freqpts}]
        super().acquire(y_sweep=y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        if data is None:
            data = self.data

        super().analyze(fitfunc=FIT_FUNC, fitterfunc=FITTER_FUNC, data=data)

        data["offset"] = [
            data["fit_avgi"][i][0] for i in range(len(data["stark_freq_pts"]))
        ]
        data["amp"] = [
            data["fit_avgi"][i][1] for i in range(len(data["stark_freq_pts"]))
        ]
        data["t1"] = [
            data["fit_avgi"][i][2] for i in range(len(data["stark_freq_pts"]))
        ]

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        if data is None:
            data = self.data
        qubit = self.cfg.expt.qubit[0]
        gain = self.cfg.expt.stark_gain

        title = f"T1 Stark Freq Q{qubit} Gain: {gain}"
        ylabel = "Frequency [MHz]"
        super().display(plot_both=False, title=title, xlabel=XLABEL, ylabel=ylabel)

        fig, ax = plt.subplots(3, 1, figsize=(6, 8))

        if fit:
            ax[0].plot(data["stark_freq_pts"], data["offset"])
            ax[1].plot(data["stark_freq_pts"], data["amp"])
            ax[2].plot(data["stark_freq_pts"], data["t1"])

            ax[2].set_xlabel("Gain [DAC units]")
            ax[0].set_ylabel("Offset")
            ax[1].set_ylabel("Amplitude")
            ax[2].set_ylabel("T1")
            ax[0].set_title(f"T1 Stark Power Q{qubit} Gain: {gain}")
            # print(f'Quadratic Fit: {data['quad_fit'][0]:.3g}x^2 + {data['quad_fit'][1]:.3g}x + {data['quad_fit'][2]:.3g}')
        sns.set_palette("coolwarm", len(data["stark_freq_pts"]))
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for i in range(len(data["stark_freq_pts"])):
            ax.plot(
                data["xpts"], data["avgi"][i], linewidth=0.5
            )  # , label=f'Gain {data['stark_gain_pts'][i]}')

        imname = self.fname.split("\\")[-1]
        fig.savefig(
            self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + "quad_fit.png"
        )
        plt.show()


class T1StarkPowerSingle(QickExperiment):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=None,
        style="",
        min_r2=None,
        max_err=None,
    ):

        if prefix is None:
            prefix = f"t1_stark_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "reps": 10 * self.reps,
            "rounds": self.rounds,
            "expts": 200,
            "start": 1,
            "wait_time": self.cfg.device.qubit.T1[qi],
            "acStark": True,
            "active_reset": False,
            "qubit": [qi],
            "max_gain": 1,
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "df": 70,
            "end_wait": 0.5,
        }
        params = {**params_def, **params}
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 30

        params["stark_freq"] = self.cfg.device.qubit.f_ge[qi] + params["df"]

        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        if self.cfg.expt.active_reset:
            super().configure_reset()
        if go:
            super().run(min_r2=min_r2, max_err=max_err)

    def acquire(self, progress=False):
        qi = self.cfg.expt.qubit[0]
        self.param = {"label": "stark_pulse", "param": "gain", "param_type": "pulse"}
        self.cfg.expt.stark_gain = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.max_gain
        )
        super().acquire(T1Program, progress=progress)
        data_t1 = deepcopy(self.data)
        self.cfg.expt.wait_time = 3.3 * self.cfg.device.qubit.T1[qi]
        self.cfg.expt.reps = int(4 * self.reps)
        data_g = super().acquire(T1Program, progress=progress)

        self.cfg.expt.wait_time = 0.025
        self.cfg.expt.reps = int(2.5 * self.reps)
        data_e = super().acquire(T1Program, progress=progress)

        data_types = ["avgi", "avgq", "amps", "phases"]
        for item in data_types:
            self.data[item + "_t1"] = data_t1[item]
            self.data[item + "_e"] = data_e[item]
            self.data[item + "_g"] = data_g[item]

        dv = self.data["avgi_e"] - self.data["avgi_g"]
        norm_data = (self.data["avgi_t1"] - self.data["avgi_g"]) / dv
        t1 = -1 / np.log(norm_data)
        self.data["t1"] = t1
        self.data["dv"] = dv

        return self.data

    def analyze(self, data=None, **kwargs):
        pass

    def display(self, data=None, fit=True, plot_all=False, ax=None, show_hist=True):
        if data is None:
            data = self.data

        q = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[q]
        xlabel = "Gain / Max Gain"
        title = (
            f"$T_1$ Stark Q{q} Freq: {df}, Delay Time: {self.cfg.expt.wait_time} $\mu$s"
        )

        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax = ax.flatten()
        ax[0].plot(data["xpts"], data["avgi_t1"])
        ax[1].plot(data["xpts"], data["avgi_e"])
        ax[2].plot(data["xpts"], data["avgi_g"])
        ax[3].set_xlabel(xlabel)
        ax[0].set_ylabel("I (ADC Units)")

        ax[3].plot(data["xpts"], data["t1"])
        ax[3].set_ylabel("$T_1$ / $T_{1,ave}$")
        fig.tight_layout()


class T1StarkPowerQuadSingle(QickExperimentLoop):
    """
    T1 Experiment
    Experimental Config:
    expt = dict(
        start: wait time sweep start [us]
        step: wait time sweep step
        expts: number steps in sweep
        reps: number averages per experiment
        rounds: number rounds to repeat experiment sweep
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=True,
        display=True,
        analyze=True,
        disp_kwargs={},
        print=False,
        style="",
        min_r2=None,
        max_err=None,
        check_params=True,
    ):

        if prefix is None:
            prefix = f"t1_stark_quad_power_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, check_params=check_params)

        params_def = {
            "reps": 10 * self.reps,
            "rounds": self.rounds,
            "expts": 200,
            "start": 1,
            "wait_time": self.cfg.device.qubit.T1[qi],
            "acStark": True,
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "df_pos": self.cfg.stark.f[qi],
            "df_neg": self.cfg.stark.fneg[qi],
            "stop_f": 20,
            'end_wait': 0.5,
            'pos': True,
            'neg': True,
        }

        conf = self.cfg.stark
        params = {**params_def, **params}
        if pos:
            params_def["quad_fit_pos"] = [conf.q[qi], conf.l[qi], conf.o[qi]]
            params_def["stark_freq_pos"] = self.cfg.device.qubit.f_ge[qi] + params["df_pos"]
        if neg:
            params_def["quad_fit_neg"] = [conf.qneg[qi], conf.lneg[qi], conf.oneg[qi]]
            params_def["stark_freq_neg"] = self.cfg.device.qubit.f_ge[qi] + params["df_neg"]
        
        params = {**params_def, **params}
        if style == "fine":
            params_def["rounds"] = params_def["rounds"] * 2
        elif style == "fast":
            params_def["expts"] = 30

        params['expts2']=params["expts"]
        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        
        # Run experiment immediately if requested
        super().qubit_run(
                qi=qi,
                go=go,
                display=display,
                progress=progress,
                analyze=analyze,
                min_r2=min_r2,
                max_err=max_err,
                print=print,
                disp_kwargs=disp_kwargs,
            )

    def get_gain_pts(self, pos=True, neg=True): 
        if pos:
            f_pts_pos = np.linspace(0, self.cfg.expt.stop_f, int(self.cfg.expt.expts2 / 2))
            gain_pos = find_inverse_quad_fit(f_pts_pos, *self.cfg.expt.quad_fit_pos)
        if neg:
            f_pts_neg = np.linspace(-self.cfg.expt.stop_f, 0, int(self.cfg.expt.expts2 / 2))
            gain_neg = find_inverse_quad_fit(-f_pts_neg, *self.cfg.expt.quad_fit_neg)
        if pos and neg:
            gain_pts = np.concatenate((gain_neg[0:-1], gain_pos))
            f_pts = np.concatenate((f_pts_neg[0:-1], f_pts_pos))
            # Remove indices where gain_pts is None
            valid_inds = [i for i, val in enumerate(gain_pts) if val is not None]
            gain_pts = np.array(gain_pts)[valid_inds]
            f_pts = np.array(f_pts)[valid_inds]
            gain_pts = np.array(gain_pts, dtype=float)
            m = len(f_pts_pos)  # Replace with the desired value of n
            n = len(f_pts_neg) - 1  # Replace with the desired value of m
            stark_freq = np.concatenate(
                (
                    np.full(n, self.cfg.expt.stark_freq_neg),
                    np.full(m, self.cfg.expt.stark_freq_pos),
                )
            )
        elif pos:
            gain_pts = gain_pos
            f_pts = f_pts_pos
            stark_freq = np.full(len(f_pts_pos), self.cfg.expt.stark_freq_pos)    
        elif neg:
            gain_pts = gain_neg
            f_pts = f_pts_neg
            stark_freq = np.full(len(f_pts_neg), self.cfg.expt.stark_freq_neg)

        return gain_pts, stark_freq, f_pts

    def acquire(self, progress=False):
        self.param = {"label": "stark_pulse", "param": "gain", "param_type": "pulse"}
        gain_pts, stark_freq, f_pts = self.get_gain_pts(pos=self.cfg.expt.pos, neg=self.cfg.expt.neg)

        x_sweep = [
            {"var": "stark_gain", "pts": gain_pts},
            {"var": "stark_freq", "pts": stark_freq},
        ]
        self.cfg.expt.expts = 1
        super().acquire(T1Program, x_sweep, progress=progress)
        self.data["f_pts"] = f_pts

        return self.data

    def analyze(self, data=None, **kwargs):
        pass

    def display(self, data=None, fit=True, plot_all=False, ax=None, show_hist=False):
        
        if data is None:
            data = self.data

        q = self.cfg.expt.qubit[0]
        df = self.cfg.expt.stark_freq - self.cfg.device.qubit.f_ge[q]
        xlabel = "Frequency (MHz)"
        title = (
            f"$T_1$ Stark Q{q} Freq: {df}, Delay Time: {self.cfg.expt.wait_time} $\mu$s"
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.set_title(title)
        ax.plot(data["f_pts"], data["avgi"])
        ax.set_xlabel(xlabel)
        ax.set_ylabel("I (ADC Units)")

        fig.tight_layout()
        plt.show()

        if show_hist:  # Plot histogram of shots if show_hist is True
            fig2, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.plot(data["bin_centers"], data["hist"] / np.sum(data["hist"]), "o-")
            ax.set_xlabel("I (ADC units)")
            ax.set_ylabel("Probability")


class T1StarkPowerQuad2D(QickExperiment2DSimple):
    """
    Stark Power Rabi Experiment
    Experimental Config:
    expt = dict(
        start_f: start qubit frequency (MHz),
        step_f: frequency step (MHz),
        expts_f: number of experiments in frequency,
        start_gain: qubit gain [dac level]
        step_gain: gain step [dac level]
        expts_gain: number steps
        reps: number averages per expt
        rounds: number repetitions of experiment sweep
        sigma_test: gaussian sigma for pulse length [us] (default: from pi_ge in config)
        pulse_type: 'gauss' or 'const'
    )
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix="",
        progress=False,
        style="",
        min_r2=None,
        max_err=None,
        live_plot=False,
    ):

        if prefix == "":
            prefix = f"t1_stark_power_quad_2d{qi}"

        super().__init__(cfg_dict=cfg_dict, qi=qi, prefix=prefix, progress=progress, live_plot=live_plot)

        params_def = {
            "sweep_pts":200
        }

        self.expt = T1StarkPowerQuadSingle(
            cfg_dict, qi=qi, go=False, params=params, style=style, check_params=False
        )
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = {**params_def, **params}
        
        if go:
            super().run(min_r2=min_r2, max_err=max_err, progress=progress)

    def acquire(self, progress=False):

        sweep_pts = np.arange(self.cfg.expt["sweep_pts"])
        y_sweep = [{"pts": sweep_pts, "var": "count"}]

        # Run the T1Program for each point in the 2D sweep
        self.xlabel = "Frequency (MHz)"
        super().acquire(y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=False, **kwargs):
        super().analyze(rescale=True, fit=fit)
        q = self.cfg.expt.qubit[0]
        if 'g_norm' in self.cfg.expt:
            self.data['t1_norm'] = -1/np.log((self.data['scale_data']-self.cfg.expt['g_norm'])/self.cfg.expt['e_norm'])*self.cfg.expt.wait_time/self.cfg.device.qubit.T1[q]

    def display(self, data=None, fit=False, plot_both=False, **kwargs):
        
        if data is None:
            data = self.data
        qubit = self.cfg.expt.qubit[0]
        df1 = self.cfg.expt.stark_freq_pos - self.cfg.device.qubit.f_ge[qubit]
        df2 = self.cfg.expt.stark_freq_neg - self.cfg.device.qubit.f_ge[qubit]

        self.data['xpts']=self.data['f_pts']
        title = f"T1 Stark Power Q{qubit} Freqs: {df1}, {df2}"
        ylabel = "Time (hr)"

        super().display(plot_both=plot_both, title=title, xlabel=self.xlabel, ylabel=ylabel)

        if 't1_norm' in data:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            im = ax.pcolor(data['xpts'], np.arange(self.cfg.expt["sweep_pts"]), data['t1_norm'], shading='auto')
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel('Sweep Index')
            ax.set_title(f'Normalized T1 - {title}')
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('T1 / T1_avg')
            fig.tight_layout()
            plt.show()




def find_inverse_quad_fit(y, a, b, c):
    """
    Find the inverse of a quadratic function for given y values.
    
    For a quadratic function f(x) = ax² + bx + c, this function finds
    the x values that produce the given y values by solving:
    ax² + bx + (c - y) = 0
    
    This is useful for Stark shift calibration where we know the desired
    frequency shift (y) and need to find the required drive amplitude (x).
    
    Args:
        y: Array-like of y values to find inverse for
        a: Quadratic coefficient of the fit
        b: Linear coefficient of the fit  
        c: Constant coefficient of the fit
        
    Returns:
        List of x values corresponding to input y values.
        Returns None if no real solution exists for any y value.
        
    Notes:
        - Uses the quadratic formula to solve for x
        - Takes the positive root when two solutions exist
        - Returns 0.0 for y=0 to avoid numerical issues
        - Prints warning and returns None if discriminant < 0
    """
    results = []
    
    for yt in y:
        # Handle zero case explicitly to avoid numerical issues
        if yt == 0:
            results.append(0.0)
            continue
            
        # Solve quadratic equation: ax² + bx + (c - yt) = 0
        # Using quadratic formula: x = (-b ± √(b² - 4a(c-yt))) / (2a)
        discriminant = b**2 - 4 * a * (c - yt)
        
        # Check for complex solutions
        if discriminant < 0:
            # print(f"Warning: No real roots for y={yt}")
            # print(f"Coefficients: a={a}, b={b}, c={c}")
            results.append(None)
            continue
            
        elif discriminant == 0:
            # Single solution case
            x_solution = -b / (2 * a)
            
        else:
            # Two solutions case - take the positive root
            sqrt_discriminant = np.sqrt(discriminant)
            root1 = (-b + sqrt_discriminant) / (2 * a)
            root2 = (-b - sqrt_discriminant) / (2 * a)
            
            # Choose the positive root (typically the physical solution)
            x_solution = root1 if root1 > 0 else root2
            
        results.append(x_solution)
        
    return results
