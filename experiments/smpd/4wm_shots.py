"""
SMPD 4wm Single-Shot Readout Experiment Module
=====================================

This module inherits from the single-shot experiment, but with modified pulse sequences to 
enable 4wm instead of pi pulse for qubit excitation. The analysis code is also modified to fit
two gaussians to both the g and e state data, to extract the dark count and the Pe. The Pe here
is threshold independent.

expt.seq_mode: "4wm" (default) or "pi". If "pi", same pulse sequence as the single-shot experiment is used, 
with a pi pulse for excitation. If "4wm", the pulse sequence is modified to use 4-wave mixing for excitation, 
and the analysis is modified accordingly.

4wm parameters:
    expt.f_b: buffer frequency. Default expt.device.readout.frequency[0]
    expt.t_b: buffer length. Default expt.device.readout.readout_length[0]
    expt.gain_b: buffer gain. Default expt.device.readout.gain[0]
    expt.delay_b: buffer delay. This is the delay between the start of the 4wm cycle and the start of the buffer pulse. Default is 0 us.
    expt.f_p: pump frequency. if expt.seq_mode == "4wm", need to be specified. if expt.seq_mode == "pi", default is expt.device.qubit.f_ge[0]
    expt.t_p: pump length. if expt.seq_mode == "4wm", need to be specified. if expt.seq_mode == "pi", default is expt.device.qubit.pulses.pi_ge.sigma[0]
    expt.gain_p: pump gain. if expt.seq_mode == "4wm", need to be specified. if expt.seq_mode == "pi", default is expt.device.qubit.pulses.pi_ge.gain[0]
    expt.f_r: readout/waste frequency. Default expt.device.readout.frequency[1]
    expt.t_r: readout/waste length. Default expt.device.readout.readout_length[1]
    expt.gain_r: readout/waste gain. Default expt.device.readout.gain[1]

The module contains two main classes:
- SMPDHistogramProgram: Defines the pulse sequence for the experiment
- SMPDHistogramExperiment: Manages experiment execution, data acquisition, and analysis

Key features (to be updated):
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
from ..general.qick_program import QickProgram, QickSMPDProgram

from ...helpers import config
from ...calib import readout_helpers as helpers

# Standard colors for plotting
BLUE = "#4053d3"  # Color for ground state
RED = "#b51d14"  # Color for excited state
GREEN = "#2ca02c"  # Color for second excited (f) state

# ====================================================== #


class SMPDHistogramProgram(QickSMPDProgram):
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

    def __init__(self, soccfg, final_delay, cfg, final_wait=0):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg, final_wait=final_wait)

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

        qi = cfg.expt.qubit[0]

        # 4WM sequence mode selection
        self.seq_mode = cfg.expt.get("seq_mode", "4wm")

        # Extract 4wm custom parameters falling back to configs appropriately
        self.f_b = cfg.expt.get("f_b", cfg.device.readout.frequency[0])
        self.t_b = cfg.expt.get("t_b", cfg.device.readout.readout_length[0])
        self.gain_b = cfg.expt.get("gain_b", cfg.device.readout.gain[0])
        self.delay_b = cfg.expt.get("delay_b", 0.0)

        self.f_r = cfg.expt.get("f_r", cfg.device.readout.frequency[1])
        self.t_r = cfg.expt.get("t_r", cfg.device.readout.readout_length[1])
        self.gain_r = cfg.expt.get("gain_r", cfg.device.readout.gain[1])

        self.t_reset = cfg.expt.get("t_reset", 0.2)
        self.gain_reset = cfg.expt.get("gain_reset", 0.05)

        if self.seq_mode == "4wm":
            self.f_p = cfg.expt.f_p
            self.t_p = cfg.expt.t_p
            self.gain_p = cfg.expt.gain_p
        else:
            self.f_p = cfg.expt.get("f_p", cfg.device.qubit.f_ge[qi])
            self.t_p = cfg.expt.get("t_p", cfg.device.qubit.pulses.pi_ge.sigma[qi])
            self.gain_p = cfg.expt.get("gain_p", cfg.device.qubit.pulses.pi_ge.gain[qi])

        self.frequency = self.f_r
        self.gain = self.gain_r
        self.readout_length = self.t_r

        if cfg.expt.active_reset or cfg.expt.remeas:
            self.phase = cfg.device.readout.phase[1]
        else:
            self.phase = 0

        super()._initialize(cfg, readout="")

        # self.add_gauss(
        #     ch = cfg.hw.soc.dacs.readout.ch[0], 
        #     name = 'gauss_buffer',
        #     sigma = cfg.expt.sigma,
        #     length = 6*cfg.expt.sigma,
        #     maxv = cfg.expt.gain_b,
        #     even_length = True)
        
        # self.add_pulse(
        #     ch = cfg.hw.soc.dacs.readout.ch[0],
        #     name = 'pulse_buffer',
        #     #ro_ch = cfg.hw.soc.adcs.readout.ch[1],
        #     style = 'flat_top',
        #     freq = cfg.expt.f_b,
        #     phase = cfg.device.readout.phase[0],
        #     gain = cfg.expt.gain_b,
        #     length = cfg.expt.t_b,
        #     envelope = 'gauss_buffer',
        # )

        # Buffer pulse (on resonator 0)
        self.add_gauss(
            ch=self.all_res_chs[0], name="gauss_buffer", sigma=0.02, length=0.02*6, maxv=self.gain_b, even_length=True
        )
        self.add_pulse(
            ch=self.all_res_chs[0], name="buffer_pulse", style="const", ro_ch=self.all_adc_chs[0],
            freq=self.f_b, phase=0, gain=self.gain_b, length=self.t_b#, envelope="gauss_buffer"
        )

        # Readout / Waste pulse (on resonator 1)
        self.add_pulse(
            ch=self.all_res_chs[1], name="waste_pulse", style="const", ro_ch=self.all_adc_chs[1],
            freq=self.f_r, phase=self.phase, gain=self.gain_r, length=self.t_r
        )

        # Pump pulse (on qubit)
        self.add_gauss(
            ch=self.qubit_ch, name="gauss_pump", sigma=0.02, length=0.02*6, maxv=self.gain_p, even_length=True
        )
        self.add_pulse(
            ch=self.qubit_ch, name="pump_pulse", style="const", ro_ch=self.all_adc_chs[1],
            freq=self.f_p, phase=0, gain=self.gain_p, length=self.t_p#, envelope="gauss_pump"
        )

        # 4wm reset pulses
        self.add_pulse(
            ch=self.qubit_ch, name="pump_reset", style="const", ro_ch=self.all_adc_chs[1],
            freq=self.f_p, phase=0, gain=self.gain_p, length=self.t_reset #, envelope="gauss_pump"
            #freq=self.f_p, phase=0, gain=0, length=self.t_reset #, envelope="gauss_pump"
        )
        self.add_pulse(
            ch=self.all_res_chs[1], name="waste_reset", style="const", ro_ch=self.all_adc_chs[1],
            freq=self.f_r, phase=self.phase, gain=self.gain_reset, length=self.t_reset #, envelope="gauss_pump"
        )

        super().make_pi_pulse(qi, cfg.device.qubit.f_ge, "pi_ge")
        if cfg.expt.pulse_f:
            super().make_pi_pulse(qi, cfg.device.qubit.f_ef, "pi_ef")

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
            self.send_readoutconfig(ch=self.all_adc_chs[1], name="readout", t=0)

        if cfg.expt.pulse_e:
            if self.seq_mode == "4wm":
                self.pulse(ch=self.qubit_ch, name="pump_pulse", t=0)
                self.pulse(ch=self.all_res_chs[0], name="buffer_pulse", t=self.delay_b)
                wait_time = max(self.t_p, self.delay_b + self.t_b)
                self.delay_auto(t=wait_time + 0.01) # TODO: decide if extra delay is needed after buffer pulse for stability
            else:
                self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        # Apply second pi pulse to prepare f state if requested
        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)

        # Add small delay before readout
        self.delay_auto(t=0.01, tag="wait")

        # Readout step on waste channel
        #self.pulse(ch=self.all_res_chs[1], name="waste_pulse", t=0)
        self.pulse(ch=self.all_res_chs[1], name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.all_adc_chs[1]], ddr4=True, pins=[0], t=self.all_trig_offsets[1])

        if cfg.expt.active_reset:
            #self.reset2(cfg.expt.reset)
            #super().reset(6) # works
            self.reset_4wm(10)
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
        cfg = AttrDict(self.cfg)

        # Perform reset sequence i times
        for n in range(i):
            #self.pulse(ch=self.all_res_chs[1], name="readout_pulse", t=0)
            #self.trigger(ros=[self.all_adc_chs[1]], ddr4=True, pins=[0], t=self.all_trig_offsets[1])
            self.wait_auto(cfg.expt.read_wait)
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)
            

            # Read qubit state and conditionally apply π pulse
            # If I < threshold (qubit in |1⟩), apply π pulse to return to |0⟩
            # If I >= threshold (qubit in |0⟩), skip the π pulse
            self.read_and_jump(
                ro_ch=self.all_adc_chs[1],
                component="I",  # Use I quadrature for state discrimination
                threshold=cfg.expt.threshold,  # Threshold for state discrimination
                test="<",  # Skip to end of no pulse if I < threshold
                label=f"NOPULSE{n}",  # Jump to this label if I >= threshold
            )

            # Apply π pulse to flip qubit from |1⟩ to |0⟩
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            # Small delay for pulse completion
            self.delay_auto(0.01)

            # Label for conditional jump target
            self.label(f"NOPULSE{n}")

            # For all but the last iteration, perform another measurement
            if n < i - 1:     
                # Apply readout pulse
                self.pulse(ch=self.all_res_chs[1], name="readout_pulse", t=0)
                # Trigger readout
                self.trigger(ros=[self.all_adc_chs[1]], ddr4=True, pins=[0], t=self.all_trig_offsets[1])
                # Apply LO pulse if available
                if self.lo_ch is not None:
                    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)

    def reset2(self, i):
        """
        Reset the qubit to ground state.

        Parameters
        ----------
        i : int
            Reset index
        """
        cfg = AttrDict(self.cfg)

        # Perform reset sequence i times
        for n in range(i):
            if i>0:
                self.pulse(ch=self.all_res_chs[1], name="waste_pulse", t=0)
                self.trigger(ros=[self.all_adc_chs[1]], ddr4=True, pins=[0], t=self.all_trig_offsets[1])
                self.wait_auto(cfg.expt.read_wait)
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)
            

            # Read qubit state and conditionally apply π pulse
            # If I < threshold (qubit in |1⟩), apply π pulse to return to |0⟩
            # If I >= threshold (qubit in |0⟩), skip the π pulse
            self.read_and_jump(
                ro_ch=self.all_adc_chs[1],
                component="I",  # Use I quadrature for state discrimination
                threshold=cfg.expt.threshold,  # Threshold for state discrimination
                test="<",  # Skip to end of no pulse if I < threshold
                label=f"NOPULSE{n}",  # Jump to this label if I >= threshold
            )

            # Apply π pulse to flip qubit from |1⟩ to |0⟩
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            # Small delay for pulse completion
            self.delay_auto(0.01)

            # Label for conditional jump target
            self.label(f"NOPULSE{n}")


    def reset_4wm(self, i):
        """
        Reset the qubit to ground state using 4-wave mixing sequence.

        Parameters
        ----------
        i : int
            Reset index
        """
        cfg = AttrDict(self.cfg)

        # Perform reset sequence i times
        for n in range(i):
            #self.pulse(ch=self.all_res_chs[1], name="waste_pulse", t=0)
            #self.trigger(ros=[self.all_adc_chs[1]], ddr4=True, pins=[0], t=self.all_trig_offsets[1])
            self.wait_auto(cfg.expt.read_wait)
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay + 10)
            

            # Read qubit state and conditionally apply π pulse
            # If I < threshold (qubit in |1⟩), apply π pulse to return to |0⟩
            # If I >= threshold (qubit in |0⟩), skip the π pulse
            self.read_and_jump(
                ro_ch=self.all_adc_chs[1],
                component="I",  # Use I quadrature for state discrimination
                threshold=cfg.expt.threshold,  # Threshold for state discrimination
                test="<",  # Skip to end of no pulse if I < threshold
                label=f"NOPULSE{n}",  # Jump to this label if I >= threshold
            )

            # Apply simultaneous waste and pump pulses for reversed 4-wave mixing reset
            self.pulse(ch=self.qubit_ch, name="pump_reset", t=0)
            self.pulse(ch=self.all_res_chs[1], name="waste_reset", t=0)
            # Small delay for pulse completion
            self.delay_auto(0.01)

            # Label for conditional jump target
            self.label(f"NOPULSE{n}")

            # For all but the last iteration, perform another measurement
            if n < i - 1:     
                # Apply readout pulse
                self.pulse(ch=self.all_res_chs[1], name="readout_pulse", t=0)
                # Trigger readout
                self.trigger(ros=[self.all_adc_chs[1]], ddr4=True, pins=[0], t=self.all_trig_offsets[1])
                # Apply LO pulse if available
                if self.lo_ch is not None:
                    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)

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
            
            self.trigger(ros=[self.all_adc_chs[1]], pins=[0], t=self.all_trig_offsets[1])
            self.pulse(ch=self.all_res_chs[1], name="waste_pulse", t=0)
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


class SMPDHistogramExperiment(QickExperiment):
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
            seq_mode="4wm",
            delay_b=0,
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
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[1],  # Waste is index 1
            ddr4=False,  # Whether to use DDR4 memory
            remeas=False,  # Whether to do repeated measurement
            final_delay=self.cfg.device.readout.readout_length[qi],  # Final delay
        )

        try:
            params_def["f_b"] = self.cfg.device.readout.frequency[0]
            params_def["t_b"] = self.cfg.device.readout.readout_length[0]
            params_def["gain_b"] = self.cfg.device.readout.gain[0]
            params_def["f_r"] = self.cfg.device.readout.frequency[1]
            params_def["t_r"] = self.cfg.device.readout.readout_length[1]
            params_def["gain_r"] = self.cfg.device.readout.gain[1]
            params_def["f_p"] = self.cfg.device.qubit.f_ge[qi]
            params_def["t_p"] = self.cfg.device.qubit.pulses.pi_ge.sigma[qi]
            params_def["gain_p"] = self.cfg.device.qubit.pulses.pi_ge.gain[qi]
        except Exception:
            pass

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
            #final_delay = self.cfg.expt.final_delay
            final_delay=1
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        # Configure DDR4 parameters if enabled
        if self.cfg.expt.ddr4:
            # Each transfer (burst) is 256 decimated samples
            n_transfers = 1500000
            nt = n_transfers

        # Define scan configurations for different quantum states
        scan_configs = [
            {
                'name': 'ground',
                'pulse_e': False,
                'pulse_f': False,
                'data_keys': ('Ig', 'Qg', 'Igr', 'Qgr', 't_g', 'iq_ddr4_g'),
                'enabled': True,
                'acquire_kwargs': {}
            }
        ]
        
        # Add excited state scan if enabled
        if self.cfg.expt.check_e:
            scan_configs.append({
                'name': 'excited',
                'pulse_e': True,
                'pulse_f': False,
                'data_keys': ('Ie', 'Qe', 'Ier', 'Qer', 't_e', 'iq_ddr4_e'),
                'enabled': True,
                'acquire_kwargs': {}
            })

        # Add second excited state scan if enabled
        self.check_f = self.cfg.expt.check_f
        if self.check_f:
            scan_configs.append({
                'name': 'second_excited',
                'pulse_e': True,
                'pulse_f': True,
                'data_keys': ('If', 'Qf', 'Ifr', 'Qfr', 't_f', 'iq_ddr4_f'),
                'enabled': True,
                'acquire_kwargs': {}
            })

        # Loop through each scan configuration
        for scan_config in scan_configs:
                
            if debug:
                print(f"Acquiring {scan_config['name']} state data...")

            # Create configuration for this scan
            cfg = AttrDict(copy.deepcopy(dict(self.cfg)))
            cfg.expt.pulse_e = scan_config['pulse_e']
            cfg.expt.pulse_f = scan_config['pulse_f']

            # Create and configure histogram program
            kwargs = {"soccfg": self.soccfg, "final_delay": final_delay, "cfg": cfg}
            # if self.cfg.expt.active_reset: 
            #     kwargs["final_wait"] = None
            histpro = SMPDHistogramProgram(**kwargs)

            # Configure DDR4 if enabled
            if self.cfg.expt.ddr4:
                self.im[self.cfg.aliases.soc].arm_ddr4(
                    ch=self.cfg.expt.qubit_chan, nt=n_transfers
                )

            # Acquire data for this scan
            acquire_kwargs = {
                'threshold': None,
                'progress': progress,
                **scan_config['acquire_kwargs']
            }
            
            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                **acquire_kwargs
            )

            # Extract data keys for this scan
            i_key, q_key, ir_key, qr_key, t_key, ddr4_key = scan_config['data_keys']

            # Use for active reset first 
            # # Store I/Q data
            # data[i_key] = iq_list[0][-1][:, 0]
            # data[q_key] = iq_list[0][-1][:, 1]

            # # Store reset data if active reset is enabled
            # if self.cfg.expt.active_reset or self.cfg.expt.remeas:
            #     data[ir_key] = iq_list[0][0:-1, :, 0]
            #     data[qr_key] = iq_list[0][0:-1, :, 1]

            # Use for active reset at the end
            # Store I/Q data
            data[i_key] = iq_list[0][0][:, 0]
            data[q_key] = iq_list[0][0][:, 1]

            # Store reset data if active reset is enabled
            if self.cfg.expt.active_reset or self.cfg.expt.remeas:
                data[ir_key] = iq_list[0][1:, :, 0]
                data[qr_key] = iq_list[0][1:, :, 1]

            # Get DDR4 data if enabled
            if self.cfg.expt.ddr4:
                iq_ddr4 = self.im[self.cfg.aliases.soc].get_ddr4(nt)
                t = histpro.get_time_axis_ddr4(self.cfg.expt.qubit_chan, iq_ddr4)
                data[t_key] = t
                data[ddr4_key] = iq_ddr4

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
        params, _ = helpers.analyze_single_shot_histograms(data=data, plot=False, span=span, verbose=verbose)
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
            data['e_mean'] = p['e_mean']
            data['g_mean'] = p['g_mean']
            dv = self.data['ve'] - self.data['vg']
            data['e_norm'] = (self.data['e_mean']-self.data['vg'])/dv

            data['g_norm'] = (self.data['g_mean']-self.data['vg'])/dv

            if self.cfg.expt.seq_mode == "4wm":
                # Specific 4WM analysis extracting threshold-independent Pe and Dark count
                from scipy.optimize import curve_fit

                def two_gauss(x, a1, mu1, sig1, a2, mu2, sig2):
                    return (a1 * np.exp(-0.5 * ((x - mu1) / sig1)**2) + 
                            a2 * np.exp(-0.5 * ((x - mu2) / sig2)**2))

                try:
                    # Fit for G state to determine Dark count
                    data = helpers.rotate_iq_data(data, theta=np.pi*self.cfg.device.readout.phase[1]/180)  # Example rotation, adjust as needed
                    histg, binsg = np.histogram(data["Ig"], bins=100, density=True)
                    vg = (binsg[:-1] + binsg[1:]) / 2
                    
                    histe, binse = np.histogram(data["Ie"], bins=100, density=True)
                    ve = (binse[:-1] + binse[1:]) / 2
                    
                    mu_g = vg[np.argmax(histg)]
                    mu_e = ve[np.argmax(histe)]
                    sig_guess = np.std(data["Ig"]) / 2

                    p0_g = [np.max(histg), mu_g, sig_guess, np.max(histg)/10, mu_e, sig_guess]
                    popt_g, _ = curve_fit(two_gauss, vg, histg, p0=p0_g)
            
                    data["mag1_g"] = popt_g[0]
                    data["mu1_g"] = popt_g[1]
                    data["sig1_g"] = popt_g[2]
                    data["mag2_g"] = popt_g[3]
                    data["mu2_g"] = popt_g[4]
                    data["sig2_g"] = popt_g[5]

                    area_g_g = abs(popt_g[0] * popt_g[2])
                    area_g_e = abs(popt_g[3] * popt_g[5])
                    data["dark_count"] = area_g_e / (area_g_g + area_g_e)
                    data["popt_g"] = popt_g

                    # Fit for E state to determine Pe
                    p0_e = [np.max(histe)/10, mu_g, sig_guess, np.max(histe), mu_e, sig_guess]
                    popt_e, _ = curve_fit(two_gauss, ve, histe, p0=p0_e)

                    data["mag1_e"] = popt_e[0]
                    data["mu1_e"] = popt_e[1]
                    data["sig1_e"] = popt_e[2]
                    data["mag2_e"] = popt_e[3]
                    data["mu2_e"] = popt_e[4]
                    data["sig2_e"] = popt_e[5]

                    area_e_g = abs(popt_e[0] * popt_e[2])
                    area_e_e = abs(popt_e[3] * popt_e[5])         
                    data["Pe"] = area_e_e / (area_e_g + area_e_e)
                    data["popt_e"] = popt_e
                    print(f"4WM Dark Count: {data['dark_count']:.4f}")
                    print(f"4WM Pe: {data['Pe']:.4f}")
                except Exception as e:
                    print(f"4WM Two-Gaussian fit failed: {str(e)}")

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
        params, fig = helpers.analyze_single_shot_histograms(
            data=data,
            plot=plot,
            verbose=verbose,
            span=span,
            ax=ax,
            qubit=self.cfg.expt.qubit[0],
            seq_mode=self.cfg.expt.seq_mode,
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

        # Show and save figure if requested
        if savefig:
            plt.show()
            self.save_fig(fig)

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
        n_bins = 75
        fig, ax = plt.subplots(2, 1, figsize=(6, 7))
        fig.suptitle(f"Q{self.cfg.expt.qubit[0]}")

        # Create ground state histogram
        vg, histg = helpers.create_histogram(self.data["Ig"], n_bins=n_bins)
        max_g = np.max(histg)
        ax[0].semilogy(vg, histg/max_g, color=BLUE, linewidth=2)
        ax[1].semilogy(vg, histg/max_g, color=BLUE, linewidth=2)

        # Create color palette for reset histograms
        b = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(self.data["Igr"]))

        # Create excited state histogram
        ve, histe = helpers.create_histogram(self.data["Ie"], n_bins=n_bins)
        max_e = np.max(histe)
        ax[1].semilogy(ve, histe/max_e, color=RED, linewidth=2)
        fig2, ax2 = plt.subplots(2, len(self.data["Igr"])+1, figsize=(17, 7), sharex=True, sharey=True)
        # Plot reset histograms for ground state
        ax2[0,0].plot(self.data['Ig'], self.data['Qg'], '.', markersize=1, rasterized=True)
        ax2[1,0].plot(self.data['Ie'], self.data['Qe'], '.', markersize=1, rasterized=True)
        for i in range(len(self.data["Igr"])):
            ax2[0,i+1].plot(self.data['Igr'][i], self.data['Qgr'][i], '.', markersize=1, rasterized=True)
            ax2[1,i+1].plot(self.data['Ier'][i], self.data['Qer'][i], '.', markersize=1, rasterized=True)
            v, hist = helpers.create_histogram(self.data["Igr"][i], n_bins=n_bins)
            ax[0].semilogy(v, hist/max_g, color=b[i], linewidth=1, label=f"{i+1}", rasterized=True)

            # Plot reset histograms for excited state
            v, hist = helpers.create_histogram(self.data["Ier"][i], n_bins=n_bins)
            ax[1].semilogy(v, hist/max_e, color=b[i], linewidth=1, label=f"{i+1}", rasterized=True)

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
