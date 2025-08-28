"""
Continuous T1 Measurement Module

This module implements continuous T1 relaxation time measurements for superconducting qubits.
Unlike standard T1 measurements that take discrete points, this experiment continuously
monitors the T1 time over an extended period to track temporal fluctuations.

The measurement sequence consists of:
1. Ground state measurements to establish baseline
2. Excited state measurements (π pulse followed by immediate measurement)
3. T1 measurements (π pulse, variable wait time, then measurement)

The module provides:
- T1ContProgram: Low-level pulse sequence implementation
- T1ContExperiment: Continuous T1 measurement experiment

This is particularly useful for studying T1 fluctuations and environmental effects on qubit coherence.
"""

import numpy as np
from qick import *
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment, QickExperiment2D
from ..general.qick_program import QickProgram

from ...exp_handling.datamanagement import AttrDict
from ...analysis import time_series


@dataclass
class PlotConfig:
    """Configuration parameters for plotting continuous T1 data."""
    navg: int = 10  # Number of points to average, in addition to those within a single experiment. 
    marker_size: float = 0.2  # Size of plot markers
    figure_size_raw: Tuple[float, float] = (15, 6)  # Raw data plot size
    figure_size_smoothed: Tuple[float, float] = (15, 12)  # Smoothed data plot size
    figure_size_t1: Tuple[float, float] = (15, 4)  # T1 estimate plot size
    figure_size_combined: Tuple[float, float] = (15, 4)  # Combined plot size
    figure_size_hist: Tuple[float, float] = (4, 4)  # Histogram plot size
    
    @property
    def nred(self) -> int:
        """Reduction factor for plotting."""
        return int(np.floor(self.navg ))


class T1ContProgram(QickProgram):
    """
    Quantum program for continuous T1 measurements.

    This class defines the pulse sequence for continuous T1 monitoring:
    1. Ground state measurements (no pulses, just readout)
    2. Excited state measurements (π pulse followed by immediate measurement)
    3. T1 measurements (π pulse, wait time, then measurement)

    The program can use active reset to speed up data collection.
    """

    def __init__(self, soccfg, final_delay, cfg):
        """
        Initialize the continuous T1 program.

        Args:
            soccfg: SOC configuration
            final_delay: Delay after measurement before next experiment
            cfg: Configuration dictionary containing experiment parameters
        """
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """
        Initialize the program by setting up the pulse sequence.

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)

        # Create a loop for the number of shots
        self.add_loop("shot_loop", cfg.expt.shots)

        # Initialize standard readout
        super()._initialize(cfg, readout="standard")

        # Create a π pulse to excite the qubit from |0⟩ to |1⟩
        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        """
        Define the main body of the pulse sequence.

        This method implements the actual continuous T1 measurement sequence:
        1. Ground state measurements
        2. Excited state measurements
        3. T1 measurements with variable wait times

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)

        # Configure readout
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # First, perform n_g ground state measurements
        self.delay_auto(t=0.01, tag=f"readout0_delay_1")
        for i in range(cfg.expt.n_g):
            self.measure(cfg)
        self.delay_auto(t=cfg.expt["readout"] + 0.01, tag=f"readout_delay_1_{i}")

        # Then, perform n_e excited state measurements
        for i in range(cfg.expt.n_e):
            # Apply π pulse to excite qubit
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag=f"wait0_{i}")
            # Measure excited state
            self.measure(cfg)

            # Handle active reset or standard delay
            if cfg.expt.active_reset:
                self.reset(3, 0, i)  # Perform active reset
                self.delay_auto(t=cfg.expt["readout"] + 0.01, tag=f"final_delay_0_{i}")
            else:
                # Standard delay between measurements
                self.delay_auto(
                    t=cfg.expt["final_delay"] + 0.01, tag=f"final_delay_1_{i}"
                )

        # Finally, perform n_t1 T1 measurements
        for i in range(cfg.expt.n_t1):
            # Apply π pulse to excite qubit
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            # Wait for T1 decay
            self.delay_auto(t=cfg.expt["wait_time"] + 0.01, tag=f"wait_{i}")
            # Measure qubit state
            self.measure(cfg)

            # Handle active reset or standard delay
            if cfg.expt.active_reset:
                self.reset(3, 1, i)  # Perform active reset
                self.delay_auto(t=cfg.expt["readout"] + 0.01, tag=f"final_delay_{i}")
            else:
                # Standard delay between measurements
                self.delay_auto(
                    t=cfg.expt["final_delay"] + 0.01, tag=f"final_delay_{i}"
                )

    def measure(self, cfg):
        """
        Perform a single measurement.

        This method sends a readout pulse and triggers the ADC.

        Args:
            cfg: Configuration dictionary
        """
        # Send readout pulse
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)

        # Apply LO pulse if needed
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)

        # Trigger ADC for measurement
        self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)

    def reset(self, i, j, k):
        """
        Perform active qubit reset.

        This method reads the qubit state and applies a π pulse if needed to return to ground state.

        Args:
            i: Number of reset attempts
            j, k: Indices for unique label generation
        """
        # Perform active reset i times
        cfg = AttrDict(self.cfg)

        for n in range(i):
            # Wait for readout result
            self.wait_auto(cfg.expt.read_wait)
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)

            # Read the input, test a threshold, and jump if it is met
            # If I < threshold (qubit in |0⟩), skip the π pulse
            self.read_and_jump(
                ro_ch=self.adc_ch,
                component="I",
                threshold=cfg.expt.threshold,
                test="<",
                label=f"NOPULSE{n}{j}{k}",
            )

            # Apply π pulse to return to |0⟩ if qubit was in |1⟩
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(0.01)

            # Label for jump target
            self.label(f"NOPULSE{n}{j}{k}")

            # For all but the last reset attempt, perform another measurement
            if n < i - 1:
                self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
                self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
                if self.lo_ch is not None:
                    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
                self.delay_auto(0.01)


class T1ContExperiment(QickExperiment):
    """
    Continuous T1 measurement experiment.

    This class implements a continuous T1 measurement that tracks T1 over time.
    Unlike standard T1 measurements that take discrete points at different wait times,
    this experiment uses a fixed wait time and continuously monitors the qubit state
    to track temporal fluctuations in T1.

    Configuration parameters (self.cfg.expt: dict):
        - shots (int): Number of measurement shots to take
        - reps (int): Number of repetitions for each experiment (inner loop)
        - rounds (int): Number of software averages (outer loop)
        - wait_time (float): Fixed wait time for T1 measurement in microseconds
        - active_reset (bool): Whether to use active qubit reset
        - final_delay (float): Delay between measurements in microseconds
        - readout (float): Readout pulse length in microseconds
        - n_g (int): Number of ground state measurements per sequence
        - n_e (int): Number of excited state measurements per sequence
        - n_t1 (int): Number of T1 measurements per sequence
        - qubit (list): Index of the qubit being measured
        - qubit_chan (int): Channel of the qubit being read out
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=True,
        style="",
        disp_kwargs=None,
        min_r2=None,
        max_err=None,
        display=True,
        analyze=True,
    ):
        """
        Initialize the continuous T1 experiment.

        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index to measure
            go: Whether to immediately run the experiment
            params: Additional parameters to override defaults
            prefix: Filename prefix for saved data
            progress: Whether to show progress bar
            style: Measurement style (not used in this experiment)
            disp_kwargs: Display options
            min_r2: Minimum R² value for acceptable fit
            max_err: Maximum error for acceptable fit
            display: Whether to display results
        """
        # Set default prefix based on qubit index if not provided
        if prefix is None:
            prefix = f"t1_cont_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        # Define default parameters
        params_def = {
            "shots": 50000,  # Number of measurement shots
            "reps": 1,  # Number of repetitions
            "rounds": self.rounds,  # Number of software averages
            "wait_time": self.cfg.device.qubit.T1[qi],  # Wait time set to current T1
            "active_reset": self.cfg.device.readout.active_reset[
                qi
            ],  # Use active reset if configured
            "final_delay": self.cfg.device.qubit.T1[qi] * 6,  # Final delay set to 6*T1
            "readout": self.cfg.device.readout.readout_length[
                qi
            ],  # Readout pulse length
            "n_g": 1,  # Number of ground state measurements
            "n_e": 2,  # Number of excited state measurements
            "n_t1": 7,  # Number of T1 measurements
            "qubit": [qi],  # Qubit index as a list
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],  # Readout channel
        }

        # Merge default and user-provided parameters
        self.cfg.expt = {**params_def, **params}

        # Check for unexpected parameters
        super().check_params(params_def)

        # Configure active reset if enabled
        if self.cfg.expt.active_reset:
            super().configure_reset()

        # For untuned qubits, show all data points by default
        if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
            disp_kwargs = {"plot_all": True}

        # Run the experiment if go=True
        if go:
            super().run(
                display=display,
                progress=progress,
                analyze=analyze,
                min_r2=min_r2,
                max_err=max_err,
                disp_kwargs=disp_kwargs,
            )

    def acquire(self, progress=False, get_hist=True):
        """
        Acquire continuous T1 measurement data.

        This method:
        1. Creates and runs the T1ContProgram
        2. Collects and processes the measurement results
        3. Organizes data into ground state, excited state, and T1 measurements

        Args:
            progress: Whether to show progress bar
            get_hist: Whether to generate histogram data

        Returns:
            Measurement data dictionary
        """
        # Define parameter metadata for plotting
        self.param = {"label": "wait_0", "param": "t", "param_type": "time"}

        # Set appropriate final delay based on active reset setting
        if "active_reset" in self.cfg.expt and self.cfg.expt.active_reset:
            final_delay = self.cfg.device.readout.readout_length[self.cfg.expt.qubit[0]]
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Create and initialize the T1ContProgram
        prog = T1ContProgram(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )

        # Record start time for the experiment
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        current_time = current_time.encode("ascii", "replace")

        # Run the program to acquire data
        iq_list = prog.acquire(
            self.im[self.cfg.aliases.soc],
            rounds=self.cfg.expt.rounds,
            threshold=None,
            progress=progress,
        )

        # Get parameter values
        xpts = self.get_params(prog)

        # Generate histogram data if requested
        if get_hist:
            v, hist = self.make_hist(prog)

        # Initialize data dictionary
        data = {
            "xpts": xpts,
            "start_time": current_time,
        }

        # Process data for ground state, excited state, and T1 measurements
        nms = ["g", "e", "t1"]  # Names for the three measurement types

        # Calculate start indices for each measurement type
        if self.cfg.expt.active_reset:
            # With active reset, indices are different due to reset measurements
            start_ind = [
                0,  # Start of ground state measurements
                self.cfg.expt.n_g,  # Start of excited state measurements
                self.cfg.expt.n_g + self.cfg.expt.n_e * 3,  # Start of T1 measurements
                len(iq_list[0]),  # End of all measurements
            ]
        else:
            # Without active reset, indices are sequential
            start_ind = [
                0,  # Start of ground state measurements
                self.cfg.expt.n_g,  # Start of excited state measurements
                self.cfg.expt.n_g + self.cfg.expt.n_e,  # Start of T1 measurements
                len(iq_list[0]),  # End of all measurements
            ]

        # Convert IQ list to numpy array for easier processing
        iq_array = np.array(iq_list)

        # Extract I and Q data for each measurement type
        for i in range(3):
            nm = nms[i]  # Current measurement type name

            # Get indices for this measurement type
            if self.cfg.expt.active_reset:
                # With active reset, skip reset measurements (every 3rd point)
                inds = np.arange(start_ind[i], start_ind[i + 1], 3)
            else:
                # Without active reset, use all points
                inds = np.arange(start_ind[i], start_ind[i + 1])

            # Extract I and Q data
            # Commented out amplitude and phase calculations
            # data['amps_'+nm] = np.abs(iq_array[0,inds,0].dot([1, 1j]))
            # data['phases_'+nm] = np.angle(iq_array[0,inds,:].dot([1, 1j]))
            data["avgi_" + nm] = iq_array[0, inds, :, 0]  # I quadrature
            data["avgq_" + nm] = iq_array[0, inds, :, 1]  # Q quadrature

        # Add histogram data if generated
        if get_hist:
            data["bin_centers"] = v
            data["hist"] = hist

        # Convert all data to numpy arrays
        for key in data:
            data[key] = np.array(data[key])

        # Store data in the object
        self.data = data

        return self.data

    def analyze(self, data=None, **kwargs):
        """
        Analyze continuous T1 measurement data.

        This method is a placeholder for more complex analysis.
        The actual T1 calculation is done in the display method.

        Args:
            data: Data dictionary to analyze (uses self.data if None)
            **kwargs: Additional arguments passed to the analyzer

        Returns:
            Data dictionary (unchanged)
        """
        if data is None:
            data = self.data

        nexp = self.cfg.expt.n_g + self.cfg.expt.n_e + self.cfg.expt.n_t1
        n_t1 = self.cfg.expt.n_t1
        pi_time = 0.4  # Approximate π pulse time in μs
        # Calculate total pulse sequence length
        if self.cfg.expt.active_reset:
            # With active reset
            n_reset = 3
            pulse_length = (
                (self.cfg.expt.readout+0.3)
                * (self.cfg.expt.n_g+1 + (n_reset+1) * (self.cfg.expt.n_e + n_t1))
                + self.cfg.expt.wait_time * n_t1
                + nexp * pi_time
            )
        else:
            # Without active reset
            pulse_length = (
                self.cfg.expt.readout * nexp
                + self.cfg.expt.wait_time * n_t1
                + self.cfg.expt.final_delay * (self.cfg.expt.n_e + n_t1)
                + nexp * pi_time
            )

        # Convert to seconds
        self.pulse_length = pulse_length / 1e6

        return data

    def _prepare_data_for_plotting(self, data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Prepare and flatten data arrays for time series plotting.
        
        Args:
            data: Raw measurement data dictionary
            
        Returns:
            Dictionary containing flattened data arrays
        """
        return {
            't1_data': data["avgi_t1"].transpose().flatten(),
            'g_data': data["avgi_g"].transpose().flatten(),
            'e_data': data["avgi_e"].transpose().flatten()
        }
    
    def _smooth_data(self, flattened_data: Dict[str, np.ndarray], 
                     plot_config: PlotConfig) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Apply uniform filtering to smooth the time series data.
        
        Args:
            flattened_data: Dictionary of flattened data arrays
            plot_config: Configuration for plotting parameters
            
        Returns:
            Tuple of (smoothed_data_dict, time_array)
        """
        smoothed = {}
        
        # Smooth each data type with appropriate filter size
        smoothed['t1'] = uniform_filter1d(
            flattened_data['t1_data'], 
            size=plot_config.navg * self.cfg.expt.n_t1
        )[::plot_config.nred * self.cfg.expt.n_t1]
        
        smoothed['g'] = uniform_filter1d(
            flattened_data['g_data'], 
            size=plot_config.navg * self.cfg.expt.n_g
        )[::plot_config.nred * self.cfg.expt.n_g]
        
        smoothed['e'] = uniform_filter1d(
            flattened_data['e_data'], 
            size=plot_config.navg * self.cfg.expt.n_e
        )[::plot_config.nred * self.cfg.expt.n_e]
        
        # Generate time array
        npts = len(smoothed['t1'])
        times = np.arange(npts) * self.pulse_length * plot_config.nred
        
        return smoothed, times
    
    def _calculate_t1(self, smoothed_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate T1 decay signal.
        
        Args:
            smoothed_data: Dictionary of smoothed data arrays
            
        Returns:
            Normalized T1 signal array
        """
        signal_difference = smoothed_data['e'] - smoothed_data['g']
        norm_data = (smoothed_data['t1'] - smoothed_data['g']) / signal_difference
        # Calculate T1 from normalized decay with error handling
        with np.errstate(divide='ignore', invalid='ignore'):
            t1_estimates = -1 / np.log(norm_data) * self.cfg.expt.wait_time

        return norm_data, t1_estimates
    
    def _plot_histogram(self, data: Dict[str, Any], qubit: int) -> None:
        """
        Plot histogram of ground and excited state measurements.
        
        Args:
            data: Measurement data dictionary
            qubit: Qubit index for labeling
        """
        fig, ax = plt.subplots(1, 1, figsize=PlotConfig.figure_size_hist)
        
        hist_params = {'bins': 50, 'alpha': 0.6, 'density': True}
        
        ax.hist(data["avgi_g"].flatten(), label="Ground State", **hist_params)
        ax.hist(data["avgi_e"].flatten(), label="Excited State", **hist_params)
        
        ax.set_xlabel("I [ADC units]")
        ax.set_ylabel("Probability")
        ax.legend()
        ax.set_title(f"Qubit {qubit} State Histogram")
        fig.tight_layout()
        super().save_fig(fig, "_hist")
    
    def _plot_raw_data(self, data: Dict[str, Any], qubit: int, 
                       plot_config: PlotConfig) -> None:
        """
        Plot raw I/Q data for all measurement types.
        
        Args:
            data: Measurement data dictionary
            qubit: Qubit index for labeling
            plot_config: Configuration for plotting parameters
        """
        fig, axes = plt.subplots(2, 1, figsize=plot_config.figure_size_raw, sharex=True)
        fig.suptitle(f"Qubit {qubit} Continuous T1 Measurement Raw Data")
        
        # Define data types and their plot properties
        data_types = [
            ('avgi_e', 'avgq_e', '.', 'Excited State'),
            ('avgi_g', 'avgq_g', 'k.', 'Ground State'),
            ('avgi_t1', 'avgq_t1', '.', 'T1 Measurement')
        ]
        
        for i_key, q_key, marker, label in data_types:
            for i in range(len(data[i_key])):
                axes[0].plot(data[i_key][i], marker, markersize=plot_config.marker_size)
                axes[1].plot(data[q_key][i], marker, markersize=plot_config.marker_size)
        
        axes[0].set_ylabel("I (ADC units)")
        axes[1].set_ylabel("Q (ADC units)")
        axes[1].set_xlabel("Shot number")
        fig.tight_layout()
        super().save_fig(fig, "_raw_data")
    
    def _plot_smoothed_data(self, smoothed_data: Dict[str, np.ndarray], times: np.ndarray,
                           normalized_t1: np.ndarray, qubit: int, 
                           plot_config: PlotConfig) -> plt.Figure:
        """
        Plot smoothed data and normalized T1 decay in a 4-panel figure.
        
        Args:
            smoothed_data: Dictionary of smoothed data arrays
            times: Time array for x-axis
            normalized_t1: Normalized T1 signal
            qubit: Qubit index for labeling
            plot_config: Configuration for plotting parameters
            
        Returns:
            Figure object for potential saving
        """
        fig, axes = plt.subplots(4, 1, figsize=plot_config.figure_size_smoothed, 
                                sharex=True)
        fig.suptitle(f"Qubit {qubit} Continuous T1 Measurement Analysis")
        
        # Plot parameters
        plot_params = {
            'linewidth': 0.1, 
            'markersize': plot_config.marker_size,
            'marker': '.',
            'color': 'black'
        }
        
        # Plot data for each measurement type
        data_labels = [
            ("T1 Data", "I (ADC), $T =T_1$"),
            ("Ground State", "I (ADC), $g$ state"),
            ("Excited State", "I (ADC), $e$ state"),
            ("Normalized T1", "$(v_{t1}-v_g)/(v_e-v_g)$")
        ]
        
        plot_data = [smoothed_data['t1'], smoothed_data['g'], 
                     smoothed_data['e'], normalized_t1]
        
        for i, (data_array, (label, ylabel)) in enumerate(zip(plot_data, data_labels)):
            axes[i].plot(times, data_array, label=f"Smoothed {label}", **plot_params)
            axes[i].set_ylabel(ylabel)
        
        # Add reference line for e^-1 on normalized plot
        axes[3].axhline(np.exp(-1), linestyle="--", color='red', 
                       alpha=0.7, label="$e^{-1}$")
        axes[3].legend()
        axes[3].set_xlabel("Time (s)")
        fig.tight_layout()
        super().save_fig(fig, "_smoothed_data")
        
        return fig
    
    def _plot_t1_estimates(self, t1_estimates: np.ndarray, times: np.ndarray,
                          qubit: int, plot_config: PlotConfig) -> None:
        """
        Plot calculated T1 values over time.
        
        Args:
            normalized_t1: Normalized T1 signal
            times: Time array for x-axis
            qubit: Qubit index for labeling
            plot_config: Configuration for plotting parameters
        """
        fig, ax = plt.subplots(1, 1, figsize=plot_config.figure_size_t1)
        fig.suptitle(f"Qubit {qubit} Continuous T1 Estimate")

        ax.plot(times, t1_estimates, "k.-", linewidth=0.1, 
                markersize=plot_config.marker_size, label="T1 Data")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$T_1$ (μs)")
        super().save_fig(fig, "_t1")
    
    def _plot_combined_data(self, smoothed_data: Dict[str, np.ndarray], times: np.ndarray,
                           qubit: int, plot_config: PlotConfig) -> None:
        """
        Create combined plot with all data types for jump detection.
        
        Args:
            smoothed_data: Dictionary of smoothed data arrays
            times: Time array for x-axis
            qubit: Qubit index for labeling
            plot_config: Configuration for plotting parameters
        """
        fig, ax = plt.subplots(1, 1, figsize=plot_config.figure_size_combined)
        fig.suptitle(f"Qubit {qubit} Continuous T1 Measurement Combined Data")
        
        plot_params = {'linewidth': 0.1, 'markersize': plot_config.marker_size, 'marker': '.'}
        
        # Primary y-axis: T1 data
        ax.plot(times, smoothed_data['t1'], color='black', 
                label="Smoothed T1 Data", **plot_params)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("T1 Signal (ADC units)", color='black')
        
        # Secondary y-axes for excited and ground state data
        ax2 = ax.twinx()
        ax2.plot(times, smoothed_data['e'], color='blue',
                 label="Smoothed e Data", **plot_params)
        ax2.set_ylabel("Excited State (ADC units)", color='blue')
        
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ax3.plot(times, smoothed_data['g'], color='red',
                 label="Smoothed g Data", **plot_params)
        ax3.set_ylabel("Ground State (ADC units)", color='red')
        fig.tight_layout()
        super().save_fig(fig, "_combo")
    
    def _get_save_path(self) -> Optional[Path]:
        """
        Generate the save path for figures using modern path handling.
        
        Returns:
            Path object for saving figures, or None if path cannot be determined
        """
        try:
            fname_path = Path(self.fname)
            images_dir = fname_path.parent / "images"
            images_dir.mkdir(exist_ok=True)
            return images_dir / f"{fname_path.stem}.png"
        except (AttributeError, TypeError):
            return None
    
    def _save_figure(self, fig: plt.Figure) -> None:
        """
        Save figure to disk with proper error handling.
        
        Args:
            fig: Matplotlib figure to save
        """
        save_path = self._get_save_path()
        if save_path:
            try:
                fig.tight_layout()
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Figure saved to: {save_path}")
            except Exception as e:
                print(f"Warning: Could not save figure to {save_path}: {e}")
        else:
            print("Warning: Could not determine save path for figure")

    def display(
        self,
        data: Optional[Dict[str, Any]] = None,
        fit: bool = True,
        plot_all: bool = False,
        ax = None,
        show_hist: bool = True,
        rescale: bool = False,
        savefig: bool = True,
        **kwargs,
    ) -> None:
        """
        Display continuous T1 measurement results with improved organization.

        Creates several plots:
        1. Histogram of ground and excited state measurements
        2. Raw I/Q data for all measurement types
        3. Smoothed data for T1, ground state, and excited state measurements
        4. Normalized T1 decay and calculated T1 values over time

        Args:
            data: Data dictionary to display (uses self.data if None)
            fit: Whether to show fit (legacy parameter, not used)
            plot_all: Whether to plot all data types (legacy parameter, not used)
            ax: Matplotlib axis to plot on (legacy parameter, not used)
            show_hist: Whether to show histogram of ground and excited states
            rescale: Whether to rescale data (legacy parameter, not used)
            savefig: Whether to save the figure to disk
            **kwargs: Additional arguments (for compatibility)
        """
        data = data or self.data
        qubit = self.cfg.expt.qubit[0]
        plot_config = PlotConfig()
        
        # Prepare and process data
        flattened_data = self._prepare_data_for_plotting(data)
        self.data['smoothed_data'], self.data['times'] = self._smooth_data(flattened_data, plot_config)
        self.data['normalized_t1'], self.data['t1_estimates'] = self._calculate_t1(self.data['smoothed_data'])

        
        
        # Generate plots
        if show_hist:
            self._plot_histogram(data, qubit)
            
        
        self._plot_raw_data(data, qubit, plot_config)
        
        main_fig = self._plot_smoothed_data(
            self.data['smoothed_data'], self.data['times'], self.data['normalized_t1'], qubit, plot_config
        )

        self._plot_t1_estimates(self.data['t1_estimates'], self.data['times'], qubit, plot_config)
        self._plot_combined_data(self.data['smoothed_data'], self.data['times'], qubit, plot_config)

        # Save main figure if requested
        # if savefig:
        #     self._save_figure(main_fig)
        
        plt.show()
        
        # Perform PSD analysis
        self._perform_psd_analysis(flattened_data['t1_data'])

    def _perform_psd_analysis(self, t1_data: np.ndarray) -> None:
        """
        Perform power spectral density analysis on T1 data.
        
        Args:
            t1_data: Flattened T1 measurement data
        """
        sampling_rate = 1 / self.pulse_length * self.cfg.expt.n_t1
        nperseg = min(2048, int(2 ** np.floor(np.log2(len(t1_data) / 4))))
        
        time_series.analyze_qubit_psd(
            t1_data, fs=sampling_rate, nperseg=nperseg
        )


