import numpy as np
import matplotlib.pyplot as plt

from qick import *

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment
from ..general.qick_program import QickProgram

from ..general.qick_experiment import QickExperiment2DSimple

"""
Time of Flight (ToF) Calibration Module

Purpose:
    Run this calibration when the wiring of the setup is changed.
    
    This calibration measures the time of flight of measurement pulse so we only start 
    capturing data from this point in time onwards. Time of flight (tof) is stored in 
    parameter cfg.device.readout.trig_offset.
    
    By calibrating this delay, we ensure that data acquisition starts at the optimal time
    when the signal actually arrives at the detector, avoiding dead time or missed signals.
"""


class LoopbackProgram(QickProgram):
    """
    A program that sends a readout pulse and captures the response to measure time of flight.

    This class extends QickProgram to implement the specific pulse sequence needed for
    time of flight calibration. It can optionally apply a pi pulse to excite the qubit
    to the |1⟩ state before measurement.
    """

    def __init__(self, soccfg, final_delay, cfg):
        """
        Initialize the LoopbackProgram.

        Args:
            soccfg: SoC configuration
            final_delay: Final delay time after the pulse sequence
            cfg: Configuration dictionary containing experiment parameters
        """
        super().__init__(soccfg, final_delay, cfg)

    def _initialize(self, cfg):
        """
        Initialize program parameters from configuration.

        Sets up the frequency, gain, readout length, and phase parameters from the
        experiment configuration. Optionally creates a pi pulse if checking the excited state.

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)
        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        self.readout_length = cfg.expt.readout_length
        self.phase = cfg.expt.phase
        super()._initialize(cfg, readout="custom")

        # Create a π pulse to excite the qubit from |0⟩ to |1⟩
        if cfg.expt.check_e:
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        """
        Define the main body of the pulse sequence.

        This method implements the actual pulse sequence:
        1. Configure readout
        2. Optionally apply a pi pulse if checking excited state
        3. Apply a readout pulse
        4. Trigger data acquisition

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(cfg)
        # if self.type=='full':
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        if cfg.expt.check_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01, tag="wait")
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=0,
        )


# ====================================================== #


class ToFCalibrationExperiment(QickExperiment):
    """
    Time of Flight Calibration Experiment

    This class implements the experiment to calibrate the time of flight between
    sending a readout pulse and receiving the response. It measures the delay
    that should be applied to the ADC trigger to capture the signal at the right time.

    Experimental Config Parameters:
        rounds: Number of software averages for the measurement
        readout_length [us]: Length of the readout pulse
        trig_offset [us]: Current trigger offset for the ADC
        gain [DAC units]: Amplitude of the readout pulse
        frequency [MHz]: Frequency of the readout pulse
        reps: Number of averages per point
        qubit: List containing the qubit index to calibrate
        phase: Phase of the readout pulse
        final_delay [us]: Final delay after the pulse sequence
        check_e: Whether to excite qubit to |1⟩ state before measurement
        use_readout: Whether to use existing readout parameters (gain and phase)
    """

    def __init__(
        self,
        cfg_dict={},
        progress=None,
        prefix=None,
        qi=0,
        params={},
        go=True,
        print=False,
    ):
        """
        Initialize the ToF calibration experiment.

        Args:
            cfg_dict: Configuration dictionary
            progress: Progress tracking object
            prefix: Prefix for experiment name (default: "adc_trig_offset_calibration_qubit{qi}")
            qi: Qubit index
            params: Additional parameters to override defaults
            go: Whether to run the experiment immediately after initialization
        """
        if prefix is None:
            prefix = f"adc_trig_offset_calibration_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, qi=qi, prefix=prefix, progress=progress)

        # Define default parameters for the experiment
        params_def = {
            "rounds": 1000,  # Number of software averages
            "readout_length": 1,  # Readout pulse length [us]
            "trig_offset": self.cfg.device.readout.trig_offset[
                qi
            ],  # Current trigger offset [us]
            "gain": self.cfg.device.readout.max_gain,  # Readout pulse amplitude
            "frequency": self.cfg.device.readout.frequency[
                qi
            ],  # Readout frequency [MHz]
            "reps": 1,  # Number of averages per point
            "qubit": [qi],  # Qubit index to calibrate
            "phase": 0,  # Phase of the readout pulse
            "final_delay": 0.1,  # Final delay after sequence
            "check_e": False,  # Whether to excite qubit before measurement
            "use_readout": False,  # Whether to use existing readout parameters
        }

        # If use_readout is True, use the existing readout gain and phase
        if "use_readout" in params and params["use_readout"]:
            params_def["gain"] = self.cfg.device.readout.gain[qi]
            params_def["phase"] = self.cfg.device.readout.phase[qi]

        # Merge default parameters with provided parameters
        self.cfg.expt = {**params_def, **params}
        if print:
            super().print()
            go = False
        # Run the experiment if go is True
        if go:
            self.go(analyze=False, display=False, progress=True, save=True)
            self.display(adc_trig_offset=self.cfg.expt.trig_offset)

    def acquire(self, progress=False):
        """
        Acquire data for the ToF calibration.

        This method runs the LoopbackProgram to send a readout pulse and capture
        the response. It calculates the amplitude and phase of the response signal.

        Args:
            progress: Whether to show progress during acquisition

        Returns:
            Dictionary containing the acquired data (time axis, I/Q values, amplitude, phase)
        """
        final_delay = 10

        # Create and run the LoopbackProgram
        prog = LoopbackProgram(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )

        # Acquire decimated I/Q data
        iq_list = prog.acquire_decimated(
            self.im[self.cfg.aliases.soc],
            rounds=self.cfg.expt.rounds,
            progress=progress,
        )

        # Extract time axis and I/Q values
        t = prog.get_time_axis(ro_index=0)
        i = iq_list[0][:, 0]
        q = iq_list[0][:, 1]
        plt.show()

        # Calculate amplitude and phase from I/Q data
        amp = np.abs(i + 1j * q)  # Calculating the magnitude
        phase = np.angle(i + 1j * q)  # Calculating the phase

        # Organize data into a dictionary
        data = {"xpts": t, "i": i, "q": q, "amps": amp, "phases": phase}

        # Convert all data to numpy arrays
        for k, a in data.items():
            data[k] = np.array(a)

        self.data = data
        return data

    def analyze(self, data=None, fit=False, findpeaks=False, **kwargs):
        """
        Analyze the acquired data.

        This method is a placeholder for data analysis. In the current implementation,
        it simply returns the data without performing any analysis.

        Args:
            data: Data to analyze (default: self.data)
            fit: Whether to fit the data
            findpeaks: Whether to find peaks in the data
            **kwargs: Additional keyword arguments

        Returns:
            The data dictionary
        """
        if data is None:
            data = self.data
        return data

    def display(self, data=None, adc_trig_offset=0, save_fig=True, **kwargs):
        """
        Display the results of the ToF calibration.

        This method plots the I and Q values against time and marks the current
        trigger offset with a vertical line.

        Args:
            data: Data to display (default: self.data)
            adc_trig_offset: Current ADC trigger offset to mark on the plot
            save_fig: Whether to save the figure
            **kwargs: Additional keyword arguments
        """
        if data is None:
            data = self.data

        # Get qubit index, ADC channel, and DAC channel
        q_ind = self.cfg.expt.qubit[0]
        adc_ch = self.cfg.hw.soc.adcs.readout.ch[q_ind]
        dac_ch = self.cfg.hw.soc.dacs.readout.ch[q_ind]

        # Create figure and plot I/Q data
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.set_title(
            f"Time of Flight: DAC Ch. {dac_ch} to ADC Ch. {adc_ch}, f: {self.cfg.expt.frequency} MHz"
        )
        ax.set_xlabel("Time ($\mu$s)")
        ax.set_ylabel("Transmission (ADC units)")

        plt.plot(data["xpts"], data["i"], label="I")
        plt.plot(data["xpts"], data["q"], label="Q")
        plt.axvline(adc_trig_offset, c="k", ls="--")
        plt.legend()
        plt.show()

        # Save figure if requested
        if save_fig:
            imname = self.fname.split("\\")[-1]
            fig.tight_layout()
            fig.savefig(
                self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
            )


class ToF2D(QickExperiment2DSimple):
    """
    2D Time of Flight Calibration Experiment

    This class extends the basic ToF calibration to perform multiple measurements
    over time, which can be useful for monitoring the stability of the time of flight
    or for more complex calibration procedures.
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
    ):
        """
        Initialize the 2D ToF calibration experiment.

        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index
            go: Whether to run the experiment immediately after initialization
            params: Additional parameters to override defaults
            prefix: Prefix for experiment name (default: "tof_2d_{qi}")
            progress: Whether to show progress during acquisition
            style: Plot style
        """
        if prefix == "":
            prefix = f"tof_2d_{qi}"

        super().__init__(cfg_dict=cfg_dict, qi=qi, prefix=prefix, progress=progress)

        # Create a ToFCalibrationExperiment instance
        exp_name = ToFCalibrationExperiment
        exp_name(cfg_dict, qi, go=False, params=params)

        # Define default parameters for the 2D experiment
        params_def = {
            "expts_count": 1000,  # Number of experiments to run
            "rounds": 1,  # Number of software averages per experiment
            "qubit": [qi],  # Qubit index to calibrate
        }
        params = {**params_def, **params}

        # Initialize the experiment
        self.expt = exp_name(cfg_dict, qi=qi, go=False, params=params)

        # Merge experiment parameters with provided parameters
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params

        # Run the experiment if go is True
        if go:
            super().run(progress=progress)

    def acquire(self, progress=False):
        """
        Acquire data for the 2D ToF calibration.

        This method sets up a sweep over multiple experiments and acquires data
        for each point in the sweep.

        Args:
            progress: Whether to show progress during acquisition

        Returns:
            Dictionary containing the acquired data
        """
        # Create a sweep over the number of experiments
        pts = np.arange(self.cfg.expt.expts_count)
        y_sweep = [{"var": "npts", "pts": pts}]

        # Acquire data for the sweep
        super().acquire(y_sweep=y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired 2D data.

        This method is a placeholder for data analysis. In the current implementation,
        it does not perform any analysis.

        Args:
            data: Data to analyze
            fit: Whether to fit the data
            **kwargs: Additional keyword arguments
        """
        pass

    def display(self, data=None, fit=True, plot_both=False, **kwargs):
        """
        Display the results of the 2D ToF calibration.

        This method is a placeholder for data visualization. In the current implementation,
        it does not display any data.

        Args:
            data: Data to display
            fit: Whether to show fit results
            plot_both: Whether to plot both raw and processed data
            **kwargs: Additional keyword arguments
        """
        pass
