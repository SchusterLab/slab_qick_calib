"""
Resonator Spectroscopy Experiment

This module implements resonator spectroscopy experiments for characterizing readout resonators.
It measures the resonant frequency of the readout resonator by sweeping the readout pulse 
frequency and looking for the frequency with the maximum measured amplitude.

The resonator frequency is stored in the parameter cfg.device.readout.frequency.

The module includes:
- ResSpecProgram: Defines the pulse sequence for the resonator spectroscopy experiment
- ResSpec: Main experiment class for resonator spectroscopy
- ResSpecPower: 2D version that sweeps both frequency and power
- ResSpec2D: 2D version for repeated measurements

Note that harmonics of the clock frequency (6144 MHz) will show up as "infinitely" narrow peaks!
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
from datetime import datetime

from qick import *
from qick.asm_v2 import QickSweep1D
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


XLABEL = "Readout Frequency (MHz)"
from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import (
    QickExperiment,
    QickExperiment2DSimple
)
from ..general.qick_program import QickProgram

from ...analysis import fitting as fitter
from ...helpers import config


def generate_filename(experiment_type, qubit_idx, style=None, state=None, extra=None):
    """
    Generate a standardized filename for resonator spectroscopy experiments.
    
    Args:
        experiment_type (str): Type of experiment ('spec', 'power', '2d')
        qubit_idx (int): Qubit index
        style (str, optional): Experiment style ('coarse', 'fine')
        state (str, optional): Qubit state ('g', 'e', 'f')
        extra (str, optional): Additional identifier
        
    Returns:
        str: Standardized filename
    """
    # Base filename
    filename = f"resonator_spectroscopy"
    
    # Add experiment type
    if experiment_type == 'power':
        filename += "_power_sweep"
    elif experiment_type == '2d':
        filename += "_2d"
    
    # Add qubit state if provided
    if state:
        if state == 'e':
            filename += "_chi"
        elif state == 'f':
            filename += "_f"
    
    # Add style if provided
    if style:
        filename += f"_{style}"
    
    # Add extra identifier if provided
    if extra:
        filename += f"_{extra}"
    
    # Add qubit index
    filename += f"_qubit{qubit_idx}"
    
    return filename


class ResSpecProgram(QickProgram):
    """
    Defines the pulse sequence for a resonator spectroscopy experiment.
    
    The sequence consists of:
    1. Optional π pulse on |g>-|e> transition (if pulse_e is True)
    2. Optional π pulse on |e>-|f> transition (if pulse_f is True)
    3. Readout pulse with variable frequency
    """
    def __init__(self, soccfg, final_delay, cfg):
        """
        Initialize the resonator spectroscopy program.
        
        Args:
            soccfg: SOC configuration
            final_delay: Delay time after measurement
            cfg: Configuration dictionary
        """
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """
        Initialize the program with the necessary pulses and loops.
        
        Args:
            cfg: Configuration dictionary containing experiment parameters
        """
        cfg = AttrDict(self.cfg)
        q = cfg.expt.qubit[0]
        
        # Store experiment parameters
        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        self.readout_length = cfg.expt.length
        self.phase = cfg.device.readout.phase[q]
        
        # Choose readout type based on pulse length
        readout = 'long' if cfg.expt.long_pulse else 'custom'
            
        # Initialize with appropriate readout type
        super()._initialize(cfg, readout=readout)
        
        # Add frequency sweep loop
        self.add_loop("freq_loop", cfg.expt.expts)

        # Create pi pulse if needed for excited state measurement
        if cfg.expt.pulse_e:
            super().make_cfg_pulse(q, cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        """
        Define the main body of the experiment sequence.
        
        Args:
            cfg: Configuration dictionary containing experiment parameters
        """
        cfg = AttrDict(self.cfg)
        
        # Configure readout
        if self.adc_type == 'dyn':
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        
        # Apply pi pulses if measuring in excited state
        if cfg.expt.pulse_e:
            # Apply pi pulse to go from |g> to |e>
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            
            # Apply additional pi pulse to go from |e> to |f> if requested
            if cfg.expt.pulse_f:
                self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
                
            # Wait for qubit to settle
            self.delay_auto(t=0.02, tag="waiting")
        
        # Perform measurement
        super().measure(cfg)


class ResSpec(QickExperiment):
    """
    Main experiment class for resonator spectroscopy.
    
    This class implements resonator spectroscopy by sweeping the frequency of the readout pulse
    and measuring the response. It can be used to characterize the resonator in different qubit
    states by optionally applying pi pulses before measurement.
    
    Parameters:
    - 'start': Start frequency (MHz)
    - 'span': Frequency span (MHz)
    - 'center': Center frequency (MHz) - alternative to start
    - 'expts': Number of frequency points
    - 'gain': Gain of the readout resonator
    - 'length': Length of the readout pulse
    - 'final_delay': Delay time between repetitions in μs
    - 'pulse_e': Boolean to add e pulse prior to measurement (excite qubit)
    - 'pulse_f': Boolean to add f pulse prior to measurement (excite to 2nd level)
    - 'reps': Number of repetitions
    - 'soft_avgs': Number of software averages
    - 'long_pulse': Whether to use a long readout pulse
    - 'loop': Whether to use loop mode for acquisition
    - 'phase_const': Whether to use constant phase spacing
    - 'active_reset': Whether to use active reset
    - 'kappa': Resonator linewidth
    
    The style parameter can be:
    - 'coarse': Wide frequency span for initial search
    - 'fine': Narrow span for precise measurement
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=True,
        display=True,
        save=True,
        analyze=True,
        qi=0,
        go=True,
        params={},
        style="fine", 
        disp_kwargs=None,
        print=False,
        check_params=True,
    ):
        """
        Initialize the resonator spectroscopy experiment.
        
        Args:
            cfg_dict: Configuration dictionary
            prefix: Prefix for data files
            progress: Whether to show progress bar
            display: Whether to display results
            save: Whether to save data
            analyze: Whether to analyze data
            qi: Qubit index
            go: Whether to run the experiment immediately
            params: Additional parameters to override defaults
            style: Style of experiment ('coarse' or 'fine')
        """
        # Determine qubit state for filename
        state = None
        if 'pulse_e' in params and params['pulse_e']:
            state = 'e'
        elif 'pulse_f' in params and params['pulse_f']:
            state = 'f'
            
        # Generate standardized filename
        if not prefix:
            prefix = generate_filename('spec', qi, style=style, state=state)
        
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, check_params=check_params)

        # Default parameters
        params_def = {
            "gain": self.cfg.device.readout.gain[qi],
            "reps": self.reps,
            "rounds": self.rounds,
            "length": self.cfg.device.readout.readout_length[qi],
            "final_delay": 5,
            "pulse_e": False,
            "pulse_f": False,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            'long_pulse': False,
            'loop': False,
            'phase_const': False,
            "active_reset": False,
            'kappa': self.cfg.device.readout.kappa[qi],
        }
        
        # Set style-specific parameters
        if style == "coarse":
            # Coarse scan uses wide frequency range
            params_def["start"] = 6000
            params_def["expts"] = 5000
            params_def["span"] = 1000
        else:
            # Fine scan centers around current resonator frequency
            params_def["center"] = self.cfg.device.readout.frequency[qi]
            params_def["expts"] = 220
            params_def["span"] = 5

        # Merge default and user-provided parameters
        params = {**params_def, **params}
        
        # Set long_pulse flag based on pulse length
        if params["length"] > 100:  # Can't be set directly to be greater than 100 on dynamic generator; may need to adapt for other generators. 
            params['long_pulse'] = True

        # Handle special case for span, using 7 linewidths as span (once have good value of kappa)
        if params["span"] == "kappa":
            params["span"] = float(7 * self.cfg.device.readout.kappa[qi])
        
        # Calculate start frequency from center if provided
        if "center" in params:
            params["start"] = params["center"] - params["span"] / 2
            
        # Set experiment configuration
        self.cfg.expt = params

        if print: 
            super().print()
            go=False
        # Run the experiment if requested
        if go:
            if style == "coarse":
                # For coarse scans, run without fitting first, then analyze with peak finding
                self.go(analyze=False, display=False, progress=True, save=True)
                self.analyze(fit=False, peaks=True)
                self.display(fit=False, peaks=True)
            else:
                # For fine scans, run with standard analysis
                super().run(display=display, progress=progress, save=save, analyze=analyze, disp_kwargs=disp_kwargs)

    def acquire(self, progress=False):
        """
        Acquire data for the resonator spectroscopy experiment.
        
        Args:
            progress: Whether to show progress bar
            
        Returns:
            Acquired data
        """
        # Get qubit index and set final delay
        q = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[q] = self.cfg.expt.final_delay
        
        # Set parameter to sweep
        self.param = {"label": "readout_pulse", "param": "freq", "param_type": "pulse"}
        
        # Choose acquisition method based on loop flag
        if not self.cfg.expt.loop:
            # Standard acquisition with frequency sweep
            self.cfg.expt.frequency = QickSweep1D(
                "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
            )
            super().acquire(ResSpecProgram, progress=progress)
        else:
            # Loop acquisition with custom frequency points            
            # Generate frequency points
            if self.cfg.expt.phase_const:
                # Use homophase frequency points for constant phase spacing
                freq_pts = get_homophase(self.cfg.expt)
            else:
                # Use linear frequency spacing
                df = 2.231597954960307e-05  # Frequency step size
                self.cfg.expt.span = np.round(self.cfg.expt.span/df/(self.cfg.expt.expts-1))*df*(self.cfg.expt.expts-1)
                freq_pts = np.linspace(self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span, self.cfg.expt.expts)
                
            x_sweep = [{"pts": freq_pts, "var": 'frequency'}]
            
            self.data = super().run_loop(ResSpecProgram, x_sweep, progress=progress)
            

        return self.data

    def analyze(
        self,
        data=None,
        fit=True,
        peaks=False,
        verbose=False,
        hanger=True,
        prom=20,
        debug=False,
        **kwargs,
    ):
        """
        Analyze the acquired data to extract resonator parameters.
        
        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data to a hanger or Lorentzian model
            peaks: Whether to find peaks in the data
            verbose: Whether to print detailed information
            hanger: Whether to use hanger model (True) or Lorentzian model (False)
            prom: Prominence parameter for peak finding
            debug: Whether to show debug plots
            **kwargs: Additional arguments for the fit
            
        Returns:
            Analyzed data with fit parameters
        """
        # Get frequency, correcting for LO
        super().get_freq(fit)
        
        if data is None:
            data = self.data
        
        if fit:
            # Extract amplitude data for fitting
            ydata = data["amps"][1:-1]
            xdata = data["freq"][1:-1]
            
            if hanger:
                self.fitfunc = fitter.hangerS21func_sloped
                self.fitterfunc = fitter.fithanger
                # Fit to hanger model (for transmission resonators)
                data["fit"], data["fit_err"], data["init"] = fitter.fithanger(
                    xdata, ydata
                )
                
                # Calculate goodness of fit
                r2 = fitter.get_r2(
                    xdata, ydata, fitter.hangerS21func_sloped, data["fit"]
                )
                data["r2"] = r2
                data["fit_err"] = np.mean(
                    np.sqrt(np.diag(data["fit_err"])) / np.abs(data["fit"])
                )
                
                # Extract fit parameters
                if isinstance(data["fit"], (list, np.ndarray)):
                    f0, Qi, Qe, phi, scale, slope = data["fit"]
               
                # Calculate resonator linewidth (kappa)
                data["kappa"] = f0 * (1 / Qi + 1 / Qe) * 1e-4
                    
                # Store fit results
                data["freq_fit"] = copy.deepcopy(data["fit"])
                data["freq_init"] = copy.deepcopy(data["init"])
                data["fit"][0] = data["fit"][0] - data["freq_offset"]
                data["init"][0] = data["init"][0] - data["freq_offset"]
                data['freq_min'] = xdata[np.argmin(ydata)] - data["freq_offset"]
            else:
                self.fitfunc = fitter.lorfunc
                self.fitterfunc = fitter.fitlor
                # Fit to Lorentzian model
                fitparams = [
                    max(ydata),
                    -(max(ydata) - min(ydata)),
                    xdata[np.argmin(ydata)],
                    0.1,
                ]
                data['freq_init'] = copy.deepcopy(fitparams)
                data["fit"] = fitter.fitlor(xdata, ydata, fitparams=fitparams)
                
            # Update experiment status
            self.get_status()

        # Process phase data - remove linear phase slope
        phs_data = np.unwrap(data["phases"][1:-1])
        slope, intercept = np.polyfit(data["xpts"][1:-1], phs_data, 1)
        phs_fix = phs_data - slope * data["xpts"][1:-1] - intercept
        data["phase_fix"] = np.unwrap(phs_fix)
        
        if peaks:
            data=self.find_resonator(data, prom=prom, debug=debug)
            
        return data


    def find_resonator(self, data, prom=0.1, debug=False): 
        # Find peaks in the data (useful for coarse scans)
            xdata = data["xpts"][1:-1]
            ydata = data["amps"][1:-1]
            
            # Peak finding parameters
            min_dist = 15  # minimum distance between peaks, may need to be edited if things are really close
            max_width = 12  # maximum width of peaks in MHz, may need to be edited if peaks are off
            freq_sigma = 2  # sigma for gaussian filter
            
            # Convert parameters to indices
            df = xdata[1] - xdata[0]
            min_dist_inds = int(min_dist / df)
            max_width_inds = int(max_width / df)
            filt_sigma = int(np.ceil(freq_sigma / df))
            
            # Apply Gaussian filter to smooth data
            ydata_smooth = gaussian_filter1d(ydata, sigma=filt_sigma)
            ydata = ydata / ydata_smooth
            
            # Show debug plots if requested
            if debug:
                fig, ax = plt.subplots(2, 1, figsize=(8, 7))
                ax[0].plot(data['freq'][1:-1], data["amps"][1:-1])
                ax[0].plot(data['freq'][1:-1], ydata_smooth)
                ax[1].plot(data['freq'][1:-1], ydata)
                
            # Find peaks in the data
            coarse_peaks, props = find_peaks(
                -ydata,
                distance=min_dist_inds,
                prominence=prom,
                width=[0, max_width_inds],
            )

            # Store peak information
            data["coarse_peaks_index"] = coarse_peaks
            data["coarse_peaks"] = xdata[coarse_peaks]
            data["coarse_props"] = props

            for i in range(len(coarse_peaks)):
                peak = coarse_peaks[i]
                ax[0].axvline(data["freq"][peak], linestyle="--", color="0.2", linewidth=0.5)
                ax[1].axvline(data["freq"][peak], linestyle="--", color="0.2", linewidth=0.5)

            return data


    def display(
        self,
        data=None,
        fit=True,
        peaks=False,
        hanger=True,
        debug=False,
        ax=None,
        plot_res=True,
        **kwargs,
    ):
        """
        Display the results of the resonator spectroscopy experiment.
        
        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit curve
            peaks: Whether to show identified peaks
            hanger: Whether to use hanger model (True) or Lorentzian model (False)
            debug: Whether to show debug information
            ax: Matplotlib axes to plot on
            plot_res: Whether to plot the current resonator frequency
            **kwargs: Additional arguments for the display
        """
        if data is None:
            data = self.data

        # Determine whether to save the figure
        savefig = ax is None

        # Set up plot title
        qubit = self.cfg.expt.qubit[0]
        title = f"Resonator Spectroscopy Q{qubit}, Gain {self.cfg.expt.gain}"

        # Create figure if needed
        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(8, 7))
            fig.suptitle(title)
        else:
            ax[0].set_title(title)

        # Plot amplitude data
        ax[0].set_ylabel("Amps (ADC units)")
        ax[0].plot(data["freq"][1:-1], data["amps"][1:-1], ".-")
        ax[0].set_xlim(data["freq"][1], data["freq"][-1])
        
        # Plot fit if requested
        if fit:
            if hanger:
                # Create label with fit parameters
                label = f"$\kappa$: {data['kappa']:.2f} MHz"
                label += f" \n$f$: {data['fit'][0]:.2f} MHz"
                    
            # Plot fit curve
            ax[0].plot(
                data["freq"],
                self.fitfunc(data["freq"], *data["freq_fit"]),
                label=label,
            )
            
            # Show initial fit for debugging
            if debug:
                ax[0].plot(
                    data["freq"],
                    self.fitfunc(data["freq"], *data["freq_init"]),
                    label="Initial fit",
                )
                
            ax[0].legend()
            
            # Show current resonator frequency
            if plot_res: 
                ax[0].axvline(
                    self.cfg.device.readout.frequency[qubit] + self.data['freq_offset'], 
                    color='k', 
                    linewidth=1
                )

        # Show peaks if requested
        if peaks:
            num_peaks = len(data["coarse_peaks_index"])
            print("Number of peaks:", num_peaks)
            peak_indices = data["coarse_peaks_index"]
            for i in range(num_peaks):
                peak = peak_indices[i]
                ax[0].axvline(data["freq"][peak], linestyle="--", color="0.2", linewidth=1)
                ax[1].axvline(data["freq"][peak], linestyle="--", color="0.2", linewidth=1)

        # Complete the figure and save if needed
        if savefig:
            # Plot phase data
            ax[1].set_xlabel(XLABEL)
            ax[1].set_ylabel("Phase (radians)")
            ax[1].plot(data["freq"][1:-1], data["phase_fix"], ".-")
            
            # Finalize and save
            fig.tight_layout()
            plt.show()
            super().save_fig(fig)
            super().save_config()
        
    def update(self, freq=True, fast=False, verbose=True):
        """
        Update the configuration file with the measured resonator parameters.
        
        Args:
            cfg_file: Configuration file to update
            freq: Whether to update the resonator frequency
            fast: Whether to do a fast update (skip Q factors)
            verbose: Whether to print update information
        """
        qi = self.cfg.expt.qubit[0]
        cfg_file=self.config_file
        
        # Only update if experiment was successful
        if self.status: 
            # Update resonator frequency if requested
            if freq: 
                config.update_readout(cfg_file, 'frequency', self.data['freq_min'], qi, verbose=verbose)
                
            # Update resonator linewidth if valid
            if self.data['kappa'] > 0:
                config.update_readout(cfg_file, 'kappa', self.data['kappa'], qi, verbose=verbose)
                
            # Update Q factors if not in fast mode
            if not fast:
                config.update_readout(cfg_file, 'qi', self.data['fit'][1], qi, verbose=verbose)
                config.update_readout(cfg_file, 'qe', self.data['fit'][2], qi, verbose=verbose)



class ResSpecPower(QickExperiment2DSimple):
    """
    2D resonator spectroscopy experiment that sweeps both frequency and power.
    
    This experiment performs a 2D sweep of both readout frequency and power (gain)
    to map out how the resonator response changes with power. This allows measurement
    of the Lamb shift and other power-dependent effects.
    If using logarithmic gain spacing, the start_gain and step_gain parameters are ignored.
    For log spacing, reps are increased to maintain SNR, with min_reps the minimum number used.

    
    Parameters:
    - 'rng' (int): Range for the gain sweep, going from max_gain to max_gain/rng
    - 'max_gain' (int): Maximum gain value.
    - 'expts_gain' (int): Number of gain points in the sweep.
    - 'start_gain' (int): Starting gain value.
    - 'step_gain' (int): Step size for the gain sweep.
    - 'f_off' (float): Frequency offset from resonant frequency in MHz (usually negative)
    - 'min_reps' (int): Minimum number of repetitions.
    - 'log' (bool): Whether to use logarithmic scaling for the gain sweep; in this case, ignore start_gain/step_gain, using max_gain and rng
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=None,
        qi=0,
        go=True,
        live_plot=False,
        params={},
    ):
        """
        Initialize the power sweep resonator spectroscopy experiment.
        
        Args:
            cfg_dict: Configuration dictionary
            prefix: Prefix for data files
            progress: Whether to show progress bar
            qi: Qubit index
            go: Whether to run the experiment immediately
            params: Additional parameters to override defaults
        """
        # Determine qubit state for filename
        state = 'e' if "pulse_e" in params and params['pulse_e'] else None
        
        # Generate standardized filename
        if not prefix:
            prefix = generate_filename('power', qi, state=state)
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, live_plot=live_plot)

        # Default parameters
        params_def = {
            "reps": self.reps / 3000,  # Reduce repetitions for efficiency
            "rng": 100,                # Dynamic range for logarithmic gain sweep
            "max_gain": self.cfg.device.qubit.max_gain, # Maximum gain used in log sweep
            "span": 15,                # Frequency span in MHz
            "expts": 300,              # Number of frequency points
            "start_gain": 0.3,       # Minimum gain value for linear sweep
            "step_gain": 0.05,         # Gain step for linear sweep
            "expts_gain": 20,          # Number of gain points
            "f_off": 4,                # Frequency offset from resonator frequency
            "min_reps": 100,           # Minimum repetitions for any gain value
            "pulse_e": False,        # Whether to apply pi pulse on |g>-|e> transition
            "log": True,               # Use logarithmic gain spacing
        }
        
        # Merge default and user-provided parameters
        params = {**params_def, **params}
        
        # Calculate start frequency
        params["start"] = (
            self.cfg.device.readout.frequency[qi]
            - params["span"] / 2
            - params["f_off"]
        )

        # Create a ResSpec experiment but don't run it
        exp_name = ResSpec 
        self.expt = exp_name(cfg_dict, qi=qi, go=False, style='coarse', params=params, check_params=False)
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params

        # Run the experiment if requested
        if go:
            self.run(analyze=False, display=False, progress=False, save=True)
            self.analyze(fit=True, lowgain=None, highgain=None)
            self.display(fit=True)
        


    def acquire(self, progress=False):
        """
        Acquire data for the power sweep experiment.
        
        Args:
            progress: Whether to show progress bar
            
        Returns:
            Acquired data
        """
        # Generate gain points and repetition counts
        if self.cfg.expt.get('log', False):
            # Use logarithmic gain spacing for better dynamic range
            rng = self.cfg.expt.rng
            rat = rng ** (-1 / (self.cfg.expt.expts_gain - 1))

            # Calculate gain points with logarithmic spacing
            gain_pts = self.cfg.expt.max_gain * rat ** np.arange(self.cfg.expt.expts_gain)

            # Scale repetitions inversely with gain squared for constant SNR
            rep_list = np.round(self.cfg.expt.reps * (1 / rat ** np.arange(self.cfg.expt.expts_gain)) ** 2)
            rep_list = np.maximum(rep_list, self.cfg.expt.min_reps)
            rep_list = [int(r) for r in rep_list]  # Ensure integer repetitions
        else:
            # Use linear gain spacing
            gain_pts = self.cfg.expt.start_gain + self.cfg.expt.step_gain * np.arange(self.cfg.expt.expts_gain)
            rep_list = self.cfg.expt.reps * np.ones(self.cfg.expt.expts_gain, dtype=int)
            
        # Set up the y-sweep (gain sweep)
        y_sweep = [{"var": "gain", "pts": gain_pts}, {"var": "reps", "pts": rep_list}]

        # Configure experiment parameters
        self.qubit = self.cfg.expt.qubit[0]
        self.cfg.device.readout.final_delay[self.qubit] = self.cfg.expt.final_delay
        self.param = {"label": "readout_pulse", "param": "freq", "param_type": "pulse"}
        
        # Set up frequency sweep
        self.cfg.expt.frequency = QickSweep1D(
            "freq_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )

        # Acquire data
        self.xlabel= XLABEL
        self.ylabel = "Resonator Gain (DAC level)"
        super().acquire(y_sweep, progress=progress)

        return self.data

    def analyze(self, data=None, fit=True, highgain=None, lowgain=None, **kwargs):
        """
        Analyze the acquired data to extract power-dependent resonator parameters.
        
        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data
            highgain: High gain value to use for fitting (if None, use maximum)
            lowgain: Low gain value to use for fitting (if None, use minimum)
            **kwargs: Additional arguments for the fit
            
        Returns:
            Analyzed data with fit parameters and Lamb shift
        """
        if data is None:
            data = self.data

        # Fit the data at high and low gain to measure Lamb shift
        if fit:
            # Use maximum and minimum gain if not specified
            if highgain == None:
                highgain = np.max(data["gain_pts"])
            if lowgain == None:
                lowgain = np.min(data["gain_pts"])
                
            # Find indices of closest gain values
            i_highgain = np.argmin(np.abs(data["gain_pts"] - highgain))
            i_lowgain = np.argmin(np.abs(data["gain_pts"] - lowgain))
            
            
            # Fit to hanger model (for transmission resonators)
            fit_highpow, fit_err, init = fitter.fithanger(data["xpts"], data["amps"][i_highgain])
            fhi, Qi, Qe, phi, scale, slope = fit_highpow
            kappa_hi = fhi * (1 / Qi + 1 / Qe) * 1e-4

            fit_lowpow, fit_err, init = fitter.fithanger(data["xpts"], data["amps"][i_lowgain])
            flo, Qi, Qe, phi, scale, slope = fit_lowpow
            kappa_lo = flo * (1 / Qi + 1 / Qe) * 1e-4
            
               
            # Fit high and low gain data to Lorentzian model
            # fit_highpow, err, pinit = fitter.fitlor(
            #
            # )
            # fit_lowpow, err, pinitlow = fitter.fitlor(
            #     data["xpts"], data["amps"][i_lowgain]
            # )
            
            # Store fit results
            data["fit"] = [fit_highpow, fit_lowpow]
            data["fit_gains"] = [highgain, lowgain]
            
            # Calculate Lamb shift (difference in resonator frequency)
            data["lamb_shift"] = fhi-flo
            data['freq']= [fhi, flo]
            data['kappa']= [kappa_hi, kappa_lo]

        return data

    def display(self, data=None, fit=True, **kwargs):
        """
        Display the results of the power sweep experiment.
        
        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit results
            **kwargs: Additional arguments for the display
        """
        data = self.data if data is None else data
        qubit = self.cfg.expt.qubit[0]
        
        # Get sweep parameters
        x_sweep = data["xpts"]      # Frequency sweep
        y_sweep = data["gain_pts"]  # Gain sweep

        # Normalize amplitude data by median for each gain
        amps = copy.deepcopy(data["amps"])
        for i in range(len(amps)):
            amps[i, :] = amps[i, :] / np.median(amps[i, :])

        # Create figure and plot 2D data
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        plt.pcolormesh(x_sweep, y_sweep, amps, cmap="viridis", shading="auto", rasterized=True)
        
        # Use logarithmic y-axis if requested
        if self.cfg.expt.get('log', False):
            plt.yscale("log")
            
        # Show fit results if requested
        if fit and "fit" in data:
            fhi, flo = data["freq"]
            highgain, lowgain = data["fit_gains"]
            
            # Show vertical lines at fitted resonator frequencies
            plt.axvline(fhi, linewidth=1, color="0.2")
            plt.axvline(flo, linewidth=1, color="0.2")
            
            # Show horizontal lines at high and low gain values
            plt.plot(x_sweep, [highgain] * len(x_sweep), linewidth=1, color="0.2")
            plt.plot(x_sweep, [lowgain] * len(x_sweep), linewidth=1, color="0.2")
            
            # Print fit results
            print(f"High power peak [MHz]: {fhi:.4f}")
            print(f"Low power peak [MHz]: {flo:.4f}")
            print(f'Lamb shift [MHz]: {data["lamb_shift"]:.4f}')

        # Set plot labels and title
        plt.title(f"Resonator Spectroscopy Power Sweep Q{qubit}")
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        plt.colorbar(label="Normalized Amplitude")
        
        # Configure tick parameters and show plot
        ax.tick_params(top=True, bottom=True, right=True)
        plt.tight_layout()
        plt.show()
        
        # Save figure
        super().save_fig(fig)

class ResSpec2D(QickExperiment2DSimple):
    """
    2D resonator spectroscopy experiment for repeated measurements.
    
    This experiment performs multiple resonator spectroscopy measurements and
    averages the results to improve signal-to-noise ratio. It's useful for
    characterizing resonators with low signal levels or high noise.
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        save=True,
        display=True,
        analyze=True,
        params={},
        prefix="",
        progress=False,
        style="",
    ):
        """
        Initialize the 2D resonator spectroscopy experiment.
        
        Args:
            cfg_dict: Configuration dictionary
            qi: Qubit index
            go: Whether to run the experiment immediately
            save: Whether to save data
            display: Whether to display results
            analyze: Whether to analyze data
            params: Additional parameters to override defaults
            prefix: Prefix for data files
            progress: Whether to show progress bar
            style: Style of experiment
        """
        # Generate standardized filename if not provided
        if not prefix:
            prefix = generate_filename('2d', qi, style=style)

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        # Default parameters
        params_def = {
            "expts_count": 50,  # Number of repetitions
        }
        params = {**params_def, **params}
        
        # Create a ResSpec experiment but don't run it
        self.expt = ResSpec(cfg_dict, prefix=prefix, qi=qi, go=False, params=params)
        
        # Set experiment configuration
        self.cfg.expt = {**self.expt.cfg.expt, **params}

        # Run the experiment if requested
        if go:
            super().run(progress=progress, save=save, display=display, analyze=analyze)

    def acquire(self, progress=False):
        """
        Acquire data for the 2D resonator spectroscopy experiment.
        
        Args:
            progress: Whether to show progress bar
            
        Returns:
            Acquired data
        """
        # Create points for the y-sweep (repetition count)
        pts = np.arange(self.cfg.expt.expts_count)
        y_sweep = [{"var": "npts", "pts": pts}]
        
        # Acquire data
        self.xlabel = XLABEL
        self.ylabel = "Repetition"
        super().acquire(y_sweep=y_sweep, progress=progress)

        # Store full data before averaging
        self.data['avgi_full'] = self.data['avgi']
        self.data['avgq_full'] = self.data['avgq']
        self.data['amps_full'] = self.data['amps']
        self.data['phases_full'] = self.data['phases']

        # Average data across repetitions
        self.data['avgi'] = np.mean(self.data['avgi'], axis=0)
        self.data['avgq'] = np.mean(self.data['avgq'], axis=0)
        self.data['amps'] = np.mean(self.data['amps'], axis=0)
        
        # Process phase data
        self.data['phase_raw']=self.data['phases']
        phs = self.data['phases'] - self.cfg.device.readout.phase_inc*self.data['xpts']
        phs = np.unwrap(phs)
        phs = np.mean(phs, axis=0)
        phs = phs - np.mean(phs)
        self.data['phases'] = phs

        # Remove linear phase slope
        phs_data = np.unwrap(self.data["phases"][1:-1])
        slope, intercept = np.polyfit(self.data["xpts"][1:-1], phs_data, 1)
        phs_fix = phs_data - slope * self.data["xpts"][1:-1] - intercept
        self.data["phase_fix"] = np.unwrap(phs_fix)

        return self.data
        
    def analyze(self, data=None, fit=True, **kwargs):
        """
        Analyze the acquired data.
        
        Args:
            data: Data to analyze (if None, use self.data)
            fit: Whether to fit the data
            **kwargs: Additional arguments for the fit
        """
        # No specific analysis for ResSpec2D
        pass
        
    def display(self, data=None, fit=True, plot_both=True, **kwargs):
        """
        Display the results of the 2D resonator spectroscopy experiment.
        
        Args:
            data: Data to display (if None, use self.data)
            fit: Whether to show the fit
            plot_both: Whether to plot both amplitude and phase
            **kwargs: Additional arguments for the display
        """
        data = self.data if data is None else data
        qubit = self.cfg.expt.qubit[0]
        
        # Create figure with subplots (amplitude and optionally phase)
        if plot_both:
            fig, ax = plt.subplots(2, 1, figsize=(10, 8))
            axes = ax
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            axes = [ax]
        
        # Set title and labels
        fig.suptitle(f"Resonator Spectroscopy 2D Q{qubit}")
        axes[0].set_xlabel(self.xlabel)
        axes[0].set_ylabel("Amplitude (ADC units)")
        
        # Plot amplitude data
        axes[0].plot(data["xpts"], data["amps"], ".-", label="Averaged Data")
        axes[0].legend()
        
        # Plot phase data if requested
        if plot_both:
            axes[1].set_xlabel(self.xlabel)
            axes[1].set_ylabel("Phase (radians)")
            axes[1].plot(data["xpts"][1:-1], data["phases"][1:-1], ".-")
        
        # Finalize and show plot
        plt.tight_layout()
        plt.show()
        
        # Save figure
        imname = self.fname.split("\\")[-1]
        fig.savefig(self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png")


def get_homophase(params):
    """
    Calculate a list of frequencies that gives equal phase spacing around a resonance.
    
    This function generates a non-linear frequency spacing that results in equal phase
    steps when measuring a resonator. This is useful for phase-sensitive measurements
    where you want uniform phase coverage.
    
    Parameters:
        params (dict): A dictionary containing the following keys:
            - "expts" (int): Number of points in the frequency list.
            - "span" (float): Frequency span in MHz.
            - "kappa" (float): Resonator linewidth in MHz.
            - "center" (float): Center frequency in MHz.
            
    Returns:
        numpy.ndarray: An array containing the calculated frequency list with equal phase spacing.
    """
    nlin = 3
    kappa_inc = 1.1

    N = params["expts"] - nlin * 2
    df = params["span"]
    w = df / params["kappa"] * kappa_inc
    at = np.arctan(2 * w / (1 - w**2)) + np.pi
    R = w / np.tan(at / 2)
    fr = params["center"]
    n = np.arange(N) - N / 2 + 1 / 2
    flist = fr + R * df / (2 * w) * np.tan(n / (N - 1) * at)
    flist_lin = (
        -np.arange(nlin, 0, -1) * df / N * 3
        + params["center"]
        - params["span"] / 2
    )
    flist_linp = (
        np.arange(1, nlin + 1) * df / N * 3 + params["center"] + params["span"] / 2
    )
    flist = np.concatenate([flist_lin, flist, flist_linp])
    return flist

class ResSpecFlux(QickExperiment2DSimple):
    """
    2D resonator spectroscopy: readout frequency (X) × DC bias (Y).

    For each DC bias setpoint (set on the SoC DC output), run a 1D resonator
    spectroscopic sweep (ResSpec) and stack the rows into a 2D map.

    Params (merge with ResSpec defaults):
      # Inner 1D (ResSpec) keys: span, expts, reps, gain, length, etc.
      # Outer DC sweep:
      - 'dc_start' (V): start of DC sweep
      - 'dc_stop'  (V): end of DC sweep
      - 'expts_dc' (# rows)
      - 'bias_chan' (int): SoC DC channel index
      - 'dc_settle' (s): time.sleep after setting bias (optional)
      - 'dc_verify_print' (bool): print set/readback line per row (optional)
    """

    def __init__(
        self,
        cfg_dict,
        prefix="",
        progress=None,
        qi=0,
        go=True,
        live_plot=False,
        params={},
        style="fine",
        display=True,
        check_params=True,
    ):
        # File prefix
        if not prefix:
            prefix = generate_filename('spec', qi, style=style, extra='flux')

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi, live_plot=live_plot)

        # ---- defaults for inner 1D (ResSpec) and outer DC sweep ----
        inner_defaults = {
            "qubit": [qi],
            "length": self.cfg.device.readout.readout_length[qi],
            "gain": self.cfg.device.readout.gain[qi],
            "reps": self.reps,            # inherit your experiment's default
            "final_delay": 5,
            "pulse_e": False,
            "pulse_f": False,
        }
        # pick a reasonable freq window around current resonator freq
        if style == "coarse":
            inner_defaults.update({"center": self.cfg.device.readout.frequency[qi], "span": 50, "expts": 1000})
        elif style == "fine":
            inner_defaults.update({"center": self.cfg.device.readout.frequency[qi], "span": 5, "expts": 220})
        else:
            inner_defaults.update({"center": self.cfg.device.readout.frequency[qi], "span": 15, "expts": 300})

        dc_defaults = {
            "dc_start": 0.0,
            "dc_stop":  0.40,
            "expts_dc":  41,
            "bias_chan": 0,
            "dc_settle": 0.0,
            "dc_verify_print": False,
        }

        # Merge user params
        p = {**inner_defaults, **dc_defaults, **params}

        # calculate 'start' from 'center' for inner sweep
        if "center" in p:
            p["start"] = p["center"] - p["span"] / 2.0

        # Build inner 1D experiment but don't run yet
        self.expt = ResSpec(cfg_dict, qi=qi, go=False, style=style, params=p, check_params=check_params)

        # keep outer config synced with inner
        self.cfg.expt = {**self.expt.cfg.expt, **p}

        # axis labels
        self.xlabel = XLABEL
        self.ylabel = "DC Bias (V)"

        if go:
            # no fitting in the sweep (you can call analyze later on a slice)
            self.run(progress=progress, analyze=False, display=display, save=True)

    def acquire(self, progress=False):
        import time
        try:
            from tqdm import tqdm
        except Exception:
            tqdm = lambda x, disable=False: x  # fallback no-op

        # Build DC vector
        dcpts = np.linspace(self.cfg.expt.dc_start, self.cfg.expt.dc_stop, int(self.cfg.expt.expts_dc))

        # Containers: we’ll append rows from the inner ResSpec.acquire()
        data = {}
        data["time"] = []
        yvals = np.arange(len(dcpts))

        # Instrument alias (SoC) used to set the bias
        soc = self.im[self.cfg.aliases.soc]
        chan = int(self.cfg.expt["bias_chan"])

        for i in tqdm(yvals, disable=not progress):
            dc_val = float(dcpts[i])

            # 1) set DC bias
            soc.rfb_set_bias(chan, dc_val)
            if self.cfg.expt.get("dc_settle", 0) > 0:
                time.sleep(self.cfg.expt.dc_settle)
            rb = soc.rfb_get_bias(chan)
            if self.cfg.expt.get("dc_verify_print", False):
                print(f"[DC Bias] ch {chan}: set {dc_val:.6f} V, read {rb:.6f} V")

            # 2) update inner expt’s cfg (record the dc in metadata)
            self.expt.cfg.expt["dc"] = dc_val

            # 3) run the inner 1D ResSpec and append rows
            row = self.expt.acquire(progress=False)
            for k, v in row.items():
                if i == 0:
                    data[k] = []
                data[k].append(v)
            data["time"].append(time.time())

            # optional autosave from base if enabled elsewhere
            if getattr(self, "save_interim", False):
                super().save_data(data=data)

        # y-axis bookkeeping: use physical DC values
        data["ypts"] = dcpts
        data["dc_pts"] = dcpts

        # x-axis: take the first row’s frequency axis
        data["xpts"] = data["xpts"][0]  # ResSpec writes 'xpts' already
        if "freq" in data:                     
            data["freq"] = data["freq"][0]     

        # convert lists to numpy arrays
        for k in list(data.keys()):
            data[k] = np.array(data[k])
        # stack 2D arrays properly for standard fields
        # (most fields already come as arrays per-row from ResSpec)
        self.data = data

         # Automatically analyze after collecting all rows
        if self.cfg.expt.get("auto_analyze", True):   # optional toggle
            self.analyze(fit=True, hanger=True, verbose=False)

        return data

    def analyze(self, data=None, fit=True, hanger=True, prom=20, verbose=False, **kwargs):
        """
        Row-by-row fit of each frequency sweep vs DC bias.
        Stores f0, Qi, Qe, kappa, r2, success flags, and a simple argmin fmin.
        """
        if data is None:
            data = self.data

        # frequency axis for the inner sweep (already MHz)
        x = data["xpts"]
        # use I/Q amplitude by default
        Y = data["amps"]  # shape: (N_bias, N_freq)

        # containers
        N = len(data["ypts"])
        f0     = np.full(N, np.nan)
        Qi     = np.full(N, np.nan)
        Qe     = np.full(N, np.nan)
        kappa  = np.full(N, np.nan)
        r2     = np.full(N, np.nan)
        ok     = np.zeros(N, dtype=bool)
        fmin   = np.full(N, np.nan)

        # 1D helpers
        fitfunc   = fitter.hangerS21func_sloped if hanger else fitter.lorfunc
        do_fit    = fitter.fithanger            if hanger else fitter.fitlor

        # fit each bias row
        for j in range(N):
            y = Y[j]
            # avoid the first/last guard points if you used them
            xs = x[1:-1]
            ys = y[1:-1]

            # simple estimate from minimum
            try:
                fmin[j] = xs[np.argmin(ys)]
            except Exception:
                fmin[j] = np.nan

            if not fit:
                continue

            try:
                if hanger:
                    p, pcov, pinit = do_fit(xs, ys)
                    f0[j], Qi[j], Qe[j], phi, scale, slope = p
                    # linewidth (MHz)
                    kappa[j] = f0[j] * (1.0/Qi[j] + 1.0/Qe[j]) * 1e-4
                    r2[j] = fitter.get_r2(xs, ys, fitfunc, p)
                    ok[j] = True
                else:
                    # Lorentzian: [A0, A1, f0, gamma]
                    p = do_fit(xs, ys, fitparams=None)
                    A0, A1, f0j, gam = p
                    f0[j] = f0j
                    # approximate kappa from width parameter if desired
                    kappa[j] = gam
                    r2[j] = fitter.get_r2(xs, ys, fitfunc, p)
                    ok[j] = True
            except Exception as e:
                if verbose:
                    print(f"[fit row {j}] failed: {e}")
                ok[j] = False

        # store results back into data
        data["f0"]     = f0
        data["Qi"]     = Qi
        data["Qe"]     = Qe
        data["kappa"]  = kappa
        data["r2"]     = r2
        data["fit_ok"] = ok
        data["fmin"]   = fmin

        self.data = data
        return data


    def display(self, data=None, fit=True, plot_both=False, ax=None, **kwargs):
        """
        Show 2D amplitude map (freq vs DC bias) and overlay fitted f0(bias) if available.
        """

        d = self.data if data is None else data

        # Basic sanity for shapes
        x = np.asarray(d["xpts"])           # freq axis, shape (Nx,)
        y = np.asarray(d["ypts"])           # bias axis, shape (Ny,)
        A = np.asarray(d["amps"])           # shape (Ny, Nx)

        # Build figure/axes
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            created_fig = True
        ax.set_title(f"Resonator Spectroscopy vs DC Bias  (Q{self.cfg.expt.qubit[0]})")

        # 2D colormap
        pcm = ax.pcolormesh(x, y, A, shading="auto", cmap="viridis", rasterized=True)
        cb = plt.colorbar(pcm, ax=ax, label="Amplitude (ADC units)")
        ax.set_xlabel("Readout Frequency (MHz)")
        ax.set_ylabel("DC Bias (V)")

        # ---- Overlay fitted f0(bias) if present ----
        if fit and ("f0" in d):
            f0 = np.asarray(d["f0"])  # expected shape (Ny,)
            # Optional boolean mask from analyze(); otherwise treat all as good
            fit_ok = np.asarray(d.get("fit_ok", np.isfinite(f0)))
            mask = np.isfinite(f0) & fit_ok

            if np.any(mask):
                ax.plot(f0[mask], y[mask], "r.-", lw=2, ms=5, zorder=5, label="Fitted $f_0$")
                ax.legend()
            else:
                # Nothing valid to draw
                pass

        if created_fig:
            plt.tight_layout()
            plt.show()
            try:
                super().save_fig(fig)
            except Exception:
                pass
