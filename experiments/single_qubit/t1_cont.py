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

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment, QickExperiment2D
from ..general.qick_program import QickProgram

from ...exp_handling.datamanagement import AttrDict
from ...analysis import time_series


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
                self.cfg.expt.readout
                * (self.cfg.expt.n_g + n_reset * (self.cfg.expt.n_e + n_t1))
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

    def display(
        self,
        data=None,
        fit=True,
        plot_all=False,
        ax=None,
        show_hist=True,
        rescale=False,
        savefig=True,
        **kwargs,
    ):
        """
        Display continuous T1 measurement results.

        This method creates several plots:
        1. Histogram of ground and excited state measurements
        2. Raw I/Q data for all measurement types
        3. Smoothed data for T1, ground state, and excited state measurements
        4. Normalized T1 decay and calculated T1 values over time

        Args:
            data: Data dictionary to display (uses self.data if None)
            fit: Whether to show fit (not used)
            plot_all: Whether to plot all data types (not used)
            ax: Matplotlib axis to plot on (not used)
            show_hist: Whether to show histogram of ground and excited states
            rescale: Whether to rescale data (not used)
            savefig: Whether to save the figure to disk
            **kwargs: Additional arguments passed to the display function
        """
        if data is None:
            data = self.data

        # Get qubit index and calculate sequence parameters
        qubit = self.cfg.expt.qubit[0]

        # Set smoothing parameters
        navg = 100  # Number of points to average
        nred = int(np.floor(navg / 10))  # Reduction factor for plotting

        # Plot histogram of ground and excited states if requested
        if show_hist:
            fig2, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.hist(
                data["avgi_e"].flatten(),
                bins=50,
                alpha=0.6,
                label="Excited State",
                density=True,
            )
            ax.hist(
                data["avgi_g"].flatten(),
                bins=50,
                alpha=0.6,
                label="Ground State",
                density=True,
            )
            # Commented out fit plot
            # try:
            #     ax.plot(data['bin_centers'], two_gaussians_decay(data['bin_centers'], *data['hist_fit']), label='Fit')
            # except:
            #     pass
            ax.set_xlabel("I [ADC units]")
            ax.set_ylabel("Probability")

        # Set marker size for plots
        m = 0.2

        # Plot raw I/Q data for all measurement types
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        fig.suptitle(f"Qubit {qubit} Continuous T1 Measurement Raw Data")
        # Plot excited state data
        for i in range(len(data["avgi_e"])):
            ax[0].plot(data["avgi_e"][i], ".", markersize=m)
            ax[1].plot(data["avgq_e"][i], ".", markersize=m)

        # Plot ground state data
        for i in range(len(data["avgi_g"])):
            ax[0].plot(data["avgi_g"][i], "k.", markersize=m)
            ax[1].plot(data["avgq_g"][i], "k.", markersize=m)

        # Plot T1 measurement data
        for i in range(len(data["avgi_t1"])):
            ax[0].plot(data["avgi_t1"][i], ".", markersize=m)
            ax[1].plot(data["avgq_t1"][i], ".", markersize=m)

        # Flatten and transpose data for time series analysis
        t1_data = data["avgi_t1"].transpose().flatten()
        g_data = data["avgi_g"].transpose().flatten()
        e_data = data["avgi_e"].transpose().flatten()

        # Create 4-panel plot for smoothed data and T1 calculation
        fig, ax = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

        # Smooth T1 data and plot
        smoothed_t1_data = uniform_filter1d(t1_data, size=navg * self.cfg.expt.n_t1)
        smoothed_t1_data = smoothed_t1_data[:: nred * self.cfg.expt.n_t1]
        npts = len(smoothed_t1_data)
        times = np.arange(npts) * self.pulse_length * nred
        ax[0].plot(
            times,
            smoothed_t1_data,
            "k.-",
            linewidth=0.1,
            markersize=m,
            label="Smoothed T1 Data",
        )

        # Smooth ground state data and plot
        smoothed_g_data = uniform_filter1d(g_data, size=navg * self.cfg.expt.n_g)
        smoothed_g_data = smoothed_g_data[:: nred * self.cfg.expt.n_g]
        ax[1].plot(
            times,
            smoothed_g_data,
            "k.-",
            linewidth=0.1,
            markersize=m,
            label="Smoothed g Data",
        )

        # Smooth excited state data and plot
        smoothed_e_data = uniform_filter1d(e_data, size=navg * self.cfg.expt.n_e)
        smoothed_e_data = smoothed_e_data[:: nred * self.cfg.expt.n_e]
        ax[2].plot(
            times,
            smoothed_e_data,
            "k.-",
            linewidth=0.1,
            markersize=m,
            label="Smoothed e Data",
        )

        # Calculate normalized T1 decay
        dv = smoothed_e_data - smoothed_g_data  # Signal difference between |e⟩ and |g⟩
        pt1 = (smoothed_t1_data - smoothed_g_data) / dv  # Normalized T1 signal
        ax[3].plot(
            times, pt1, "k.-", linewidth=0.1, markersize=m, label="Normalized T1 Data"
        )
        ax[3].axhline(
            np.exp(-1), linestyle="--", label="$e^{-1}$"
        )  # e^-1 line for T1 reference

        # Set y-axis labels
        ax[0].set_ylabel("I (ADC), $T =T_1$")
        ax[1].set_ylabel("I (ADC), $g$ state")
        ax[2].set_ylabel("I (ADC), $e$ state")
        ax[3].set_ylabel("$(v_{t1}-v_g)/(v_e-v_g)$")

        # Plot calculated T1 values over time
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        fig.suptitle(f"Qubit {qubit} Continuous T1 Estimate")
        t1m = (
            -1 / np.log(pt1) * self.cfg.expt.wait_time
        )  # Calculate T1 from normalized decay
        ax.plot(times, t1m, "k.-", linewidth=0.1, markersize=m, label="T1 Data")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("$T_1$")

        # Create combined plot with all data types, mostly for checking jumps
        fig2, ax = plt.subplots(1, 1, figsize=(14, 4))
        fig2.suptitle(f"Qubit {qubit} Continuous T1 Measurement Combined Data")
        ax.plot(
            times,
            smoothed_t1_data,
            "k.-",
            linewidth=0.1,
            markersize=m,
            label="Smoothed T1 Data",
        )
        ax2 = ax.twinx()
        ax2.plot(
            times,
            smoothed_e_data,
            ".-",
            linewidth=0.1,
            markersize=m,
            label="Smoothed e Data",
        )
        ax3 = ax.twinx()
        ax3.plot(
            times,
            smoothed_g_data,
            ".-",
            linewidth=0.1,
            markersize=m,
            label="Smoothed g Data",
        )

        # Save figure if requested
        if savefig:
            fig.tight_layout()
            imname = self.fname.split("\\")[-1]
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )
            plt.show()

        # Commented out legend code
        # ax[3].legend()
        # ax.legend()
        time_series.analyze_qubit_psd(
            t1_data,
            fs=1 / self.pulse_length * self.cfg.expt.n_t1,
            nperseg=int(2 ** np.floor(np.log2(7e6)) / 4),
        )

    def psd(self):
        t1_data = self.data["avgi_t1"].transpose().flatten()

        time_series.analyze_qubit_psd(
            t1_data, fs=1 / self.pulse_length * self.cfg.expt.n_t1, nperseg=2048
        )
