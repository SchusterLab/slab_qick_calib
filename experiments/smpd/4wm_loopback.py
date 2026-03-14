import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from qick import *

from ...exp_handling.datamanagement import AttrDict
from ...helpers import config
from ..general.qick_experiment import QickExperiment
from ..general.qick_program import QickProgram

from ..general.qick_experiment import QickExperiment2DSimple

"""
4wm loopback for pulse setup. in progress...

Purpose:

"""


class SMPD4WMLoopbackProgram(QickProgram):
    """
    A program that sends pulses and captures the response to visualize the pulse sequence. 
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
        #self.f_w = cfg.expt.f_b # waste frequency
        #self.gain_w = cfg.expt.gain_w # waste gain
        #self.t_m = cfg.expt.readout_length # measurement length through the waste channel
        self.phase = cfg.expt.phase #??
        super()._initialize(cfg, readout="custom")

        # envelope for readout pulses and pump pulses
        self.add_gaussian(
            ch = cfg.hw.soc.dacs.readout.ch[0], 
            name = 'guass_buffer',
            sigma = cfg.expt.sigma,
            length = 10*cfg.expt.sigma,
            maxv = cfg.expt.gain_b,
            even_length = True)
        '''
        self.add_gaussian(
            ch = cfg.hw.soc.dacs.readout.ch[1], 
            name = 'guass_waste',
            sigma = cfg.expt.sigma,
            length = 10*cfg.expt.sigma,
            maxv = cfg.device.readout.gain[1],
            even_length = True)

        self.add_gaussian(
            ch = cfg.hw.soc.dacs.qubit.ch[0], 
            name = 'guass_pump',
            sigma = cfg.expt.sigma,
            length = 10*cfg.expt.sigma,
            maxv = cfg.expt.gain_p,
            even_length = True)
        '''
        # add pulses
        # buffer
        self.add_pulse(
            ch = cfg.hw.soc.dacs.readout.ch[0],
            name = 'pulse_buffer',
            ro_ch = cfg.hw.soc.adcs.readout.ch[0],
            style = 'flat_top',
            freq = cfg.expt.f_b,
            phase = cfg.device.readout.phase[0],
            gain = cfg.expt.gain_b,
            length = cfg.expt.t_b,
            envelope = 'guass_buffer',
        )
        
        # config readout
        self.add_readoutconfig(ch = cfg.hw.soc.adcs.readout.ch[0], name = 'buffer_ro', freq = cfg.expt.f_b, gen_ch = cfg.hw.soc.dacs.readout.ch[0])
        self.send_readoutconfig(ch=cfg.hw.soc.adcs.readout.ch[0], name="buffer_ro", t=0)

    def _body(self, cfg):
        """
        Define the main body of the pulse sequence.

        This method implements the actual pulse sequence:
        1. trigger the ADC to start acquisition 
        3. Apply a readout pulse

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(cfg)
        # if self.type=='full':

        self.trigger(ros = cfg.hw.soc.adcs.readout.ch[0], pins = [0], t=0)

        #if self.adc_type == "dyn":
        #    self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        self.pulse(ch=cfg.hw.soc.dacs.readout.ch[0], name="pulse_buffer", t=0)


# ====================================================== #


class SMPD4WMLoopbackExperiment(QickExperiment):
    """
    Time of Flight Calibration Experiment

    This class implements the experiment to see all the pulse sequence for 4WM experiments.

    Experimental Config Parameters:
        rounds: Number of software averages for the measurement
        t_m [us]: Length of the readout pulse
        t_b [us]: Length of the buffer pulse
        gain_b [DAC units]: Amplitude of the buffer pulse
        f_b [MHz]: Frequency of the buffer pulse

        trig_offset [us]: Current trigger offset for the ADC
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
            "t_m": 1,  # Readout pulse length [us]
            "t_b": 1,  # Buffer pulse length [us]
            "gain_b": self.cfg.device.readout.gain[0],  # Buffer pulse amplitude
            "f_b": self.cfg.device.readout.frequency[0],  # Buffer pulse frequency [MHz]
            "trig_offset": self.cfg.device.readout.trig_offset[0],  # Current trigger offset [us]
            "reps": 1,  # Number of averages per point
            "phase": 0,  # Phase of the readout pulse
            "final_delay": 0.1,  # Final delay after sequence
        }

        # Merge default parameters with provided parameters
        self.cfg.expt = {**params_def, **params}
        if print:
            super().print()
            go = False
        # Run the experiment if go is True
        if go:
            #self.go(analyze=False, display=False, progress=True, save=True)
            self.acquire(progress=True)
            self.analyze(fit=self.cfg.expt.use_readout)
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
        prog = SMPD4WMLoopbackProgram(
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

        When fit=True, performs a two-pass exponential fit to the amplitude data
        to extract the resonator ring-up time constant.

        Args:
            data: Data to analyze (default: self.data)
            fit: Whether to fit exponential to amplitude (resonator ring up)
            findpeaks: Whether to find peaks in the data
            **kwargs: Additional keyword arguments

        Returns:
            The data dictionary (with 'ring_up_fit' key added when fit=True)
        """
        if data is None:
            data = self.data

        if not fit:
            # Compute trig_offset: find half-max crossing point + 10 ns
            half_max = np.max(data['amps']) / 2
            crossing_idx = np.argmin(np.abs(data['amps'] - half_max))
            data['trig_offset'] = data['xpts'][crossing_idx] + 0.02  # +20 ns in µs

        if fit:
            qi = self.cfg.expt.qubit[0]
            kappa = self.cfg.device.readout.kappa[qi]
            xdata = data['xpts']
            t_min = self.cfg.device.readout.trig_offset[qi] + 0.06
            max_time = t_min + 2 / kappa
            mask = (xdata > t_min) & (xdata < max_time)
            xdata = xdata - t_min
            amps_masked = data['amps'][mask]

            # Initial guesses: c=steady state, a=start-end (negative for ring-up), b~kappa
            c0 = np.mean(amps_masked[-20:-1])
            a0 = amps_masked[0] - c0
            b0 = kappa * 3
            p0 = [a0, b0, c0]

            popt, pcov = curve_fit(exp_func, xdata[mask], amps_masked, p0=p0)
            self.data['t_min'] = t_min
            self.data['ring_up_fit'] = {'popt': popt, 'pcov': pcov, 'mask': mask}
            self.data['ring_up_amplitude'] = popt[0]
            self.data['ring_up_rate'] = popt[1]
            self.data['ring_up_offset'] = popt[2]

        return data

    def display(self, data=None, adc_trig_offset=0, save_fig=True, **kwargs):
        """
        Display the results of the ToF calibration.

        This method plots the I and Q values against time and marks the current
        trigger offset with a vertical line. If ring-up fit data is available,
        an additional plot shows the exponential fit.

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
        plt.plot(data["xpts"], data["amps"], label="Amplitude")
        #plt.axvline(adc_trig_offset, c="k", ls="--", label=f"Old trig_offset: {adc_trig_offset:.4f}")
        if 'trig_offset' in data:
            plt.axvline(data['trig_offset'], c="k", ls="--", label=f"New trig_offset: {data['trig_offset']:.4f}")
        plt.legend()
        plt.show()

        # Save figure if requested
        if save_fig:
            super().save_fig(fig)

        # Display ring-up fit if available
        if 'ring_up_fit' in self.data:
            popt = self.data['ring_up_fit']['popt']
            mask = self.data['ring_up_fit']['mask']
            t_min = self.data['t_min']
            rate = self.data['ring_up_rate']
            xfit = self.data['xpts'][mask] - t_min
            yfit = self.data['amps'][mask]

            kappa = self.cfg.device.readout.kappa[q_ind]
            fit_label = f'Fit ($\\kappa_{{fit}}$ = {rate/2/np.pi:.2f} MHz, $\\kappa_{{fit}}/\\kappa/$ = {rate / kappa/np.pi/2:.2f})'

            fig_ring, ax_ring = plt.subplots(1, 1, figsize=(8, 3))
            ax_ring.plot(xfit, yfit, 'o', label='Data')
            ax_ring.plot(xfit, exp_func(xfit, *popt), '-', label=fit_label)
            ax_ring.set_xlabel('Time ($\\mu$s)')
            ax_ring.set_ylabel('Amplitude')
            ax_ring.legend()
            ax_ring.set_title(f'Resonator ring up for Q{q_ind}')
            plt.show()

            if save_fig:
                super().save_fig(fig_ring, suffix="_ringup_fit")


    def update(self, verbose=True):
        qi = self.cfg.expt.qubit[0]
        cfg_file = self.config_file
        if 'trig_offset' in self.data:
            config.update_readout(cfg_file, 'trig_offset', self.data['trig_offset'], qi, verbose=verbose)
