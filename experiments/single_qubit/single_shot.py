"""
Single-Shot Readout Experiment Module
=====================================

This module implements single-shot readout experiments for quantum processors.
It allows for discrimination between quantum states (g, e, and optionally f)
by collecting statistics on readout signals and analyzing their distributions.

The module contains two main classes:
- HistogramProgram: Defines the quantum pulse sequence for the experiment
- HistogramExperiment: Manages experiment execution, data acquisition, and analysis

Key features:
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
from ..general.qick_program import QickProgram, reset_mode_of

from ...helpers import config
from ...calib import readout_helpers as helpers

# Standard colors for plotting
BLUE = "#4053d3"  # Color for ground state
RED = "#b51d14"  # Color for excited state
GREEN = "#2ca02c"  # Color for second excited (f) state

# ====================================================== #


# Dash patterns per reset round. Once the reset has converged the later
# rounds overlap almost exactly (TVD ~0.04 on run 33), so colour alone
# makes the last one look absent.
_ROUND_STYLES = ["-", "--", ":", "-."]


def check_hardware_axis(data, verbose=True):
    """
    Warn when the raw I axis does not separate |g> from |e>.

    `analyze` re-rotates the shots in software, so it reports a healthy
    fidelity whatever the hardware is doing. `QickProgram.reset`, though,
    branches on the raw accumulated I that `read_input` returns, with no
    rotation available. The two only agree when `device.readout.phase` is
    applied exactly once, to the readout pulse.

    On run 34 the static readout path applied it twice, leaving the raw axis
    carrying `cos(phase)` of the separation -- inverted and about a third the
    size -- while single-shot still reported fidelity 0.90 and the reset
    quietly randomised the qubit instead of resetting it. This is the check
    that distinguishes the two.

    Returns a dict of the measured quantities, so they land in the saved data.
    """
    keys = ("Ig", "Qg", "Ie", "Qe")
    if not all(k in data for k in keys):
        return {}
    Ig, Qg, Ie, Qe = (np.asarray(data[k]).ravel() for k in keys)
    raw = float(np.median(Ie) - np.median(Ig))          # what the branch sees
    full = float(np.hypot(np.median(Ie) - np.median(Ig),
                          np.median(Qe) - np.median(Qg)))
    raw_q = float(np.median(Qe) - np.median(Qg))
    frac = raw / full if full else 0.0
    q_frac = raw_q / full if full else 0.0
    out = {"hw_axis_raw_sep": raw, "hw_axis_full_sep": full,
           "hw_axis_fraction": frac, "hw_axis_q_fraction": q_frac}
    if verbose and abs(frac) < 0.8:
        # Recorded, not diagnosed. On run 34 the buffered raw I carried only
        # -0.30 of the separation while the hardware branch worked fine on two
        # different channels (#662, #665), so this does not predict what
        # read_and_jump sees -- the accumulator it reads is normalised
        # differently. Useful as a flag, not as a verdict.
        print(f"  note: buffered raw I carries {frac:+.2f} of the g-e "
              f"separation ({raw:+.1f} of {full:.1f}); Q carries {q_frac:+.2f}.")
    return out


class HistogramProgram(QickProgram):
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
        - `expt.reset_mode` (optional): "passive" | "active" | "f0g1"
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

        # Configure readout parameters
        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        # Set phase based on whether active reset is enabled
        if (reset_mode_of(cfg.expt) != "passive" or cfg.expt.remeas
                or cfg.expt.calibrate):
            self.phase = cfg.device.readout.phase[cfg.expt.qubit[0]]
        else:
            self.phase = 0
        self.readout_length = cfg.expt.readout_length

        # Initialize the base program with readout configuration
        super()._initialize(cfg, readout="")

        # Define pi pulses for state preparation
        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
        # reset_f plays pi_ef ahead of each pi_ge, so it needs the pulse too.
        if cfg.expt.pulse_f or cfg.expt.reset_f:
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

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
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # Perform active reset if enabled
        # if cfg.expt.active_reset:
        #     self.reset(cfg.expt.reset)
            #self.delay_auto(10)

        # Apply pi pulse to prepare excited state if requested
        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        # Apply second pi pulse to prepare f state if requested
        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)

        # Add small delay before readout
        if cfg.expt.calibrate:
            self.delay_auto(t=0.1, tag="wait")
        else:
            self.delay_auto(t=0.01, tag="wait")

        # Apply readout pulse and trigger data acquisition
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.adc_ch], ddr4=True, pins=[0], t=self.trig_offset)

        mode = reset_mode_of(cfg.expt)
        if mode == "active":
            super().reset(cfg.expt.reset)
        elif mode == "f0g1":
            # The unconditional drive. Declared by QickProgram._initialize,
            # which super()._initialize above reaches. Unlike reset() this
            # appends no readouts, so the shot layout is unchanged and the
            # histogram machinery below needs no adjustment.
            super().reset_f0g1()
        elif mode == "while":
            # Loops on the feedback channel, which _initialize declares. That
            # channel must be triggered: a declared readout that never fires
            # leaves the averager waiting for reads that never arrive, which
            # surfaces as "exception in readout loop" on the first poll.
            super().reset_while(cfg.expt.reset)
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
            # Apply readout pulse and trigger data acquisition
            self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
            #if self.lo_ch is not None:
            #    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
            self.trigger(ros=[self.adc_ch], ddr4=True, pins=[0], t=self.trig_offset)
            # Wait for readout to complete
            #self.wait(7)
            #self.delay(14)
            self.wait_auto(cfg.expt.read_wait)
            # Add extra delay for stability
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay+10)
            

            # Read qubit state and conditionally apply π pulse
            # If I < threshold (qubit in |1⟩), apply π pulse to return to |0⟩
            # If I >= threshold (qubit in |0⟩), skip the π pulse
            self.read_and_jump(
                ro_ch=self.adc_ch,
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
            # Apply readout pulse and trigger data acquisition
                self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
                #if self.lo_ch is not None:
                #    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
                self.trigger(ros=[self.adc_ch], ddr4=True, pins=[0], t=self.trig_offset)
                # Wait for readout to complete

                self.wait_auto(cfg.expt.read_wait)
                # Add extra delay for stability
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)
            

            # Read qubit state and conditionally apply π pulse
            # If I < threshold (qubit in |1⟩), apply π pulse to return to |0⟩
            # If I >= threshold (qubit in |0⟩), skip the π pulse
            self.read_and_jump(
                ro_ch=self.adc_ch,
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
            
            self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
            self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
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


class HistogramExperiment(QickExperiment):
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
        - `reset_mode`: `"passive"`
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
            readout_length=self.cfg.device.readout.readout_length[
                qi
            ],  # Readout pulse length
            frequency=self.cfg.device.readout.frequency[qi],  # Readout frequency
            gain=self.cfg.device.readout.gain[qi],  # Readout gain
            # Which reset runs between shots: "passive" (the full T1 wait),
            # "active" (up to `reset` conditional pi pulses), or "f0g1" (the
            # unconditional all-microwave drive). setup_reset keeps the long
            # passive inter-shot delay so a reset that is not working cannot
            # quietly corrupt the calibration this class exists to produce.
            #
            # Running the calibration under the reset it will be used with is
            # the point: the |g> blob this measures is what every threshold
            # and fidelity downstream is built on, and on run 33 the f0-g1
            # drive moved that blob by several percent. Calibrating passive
            # and then running f0g1 measures one readout and uses another.
            reset_mode="passive",
            reset=self.cfg.device.readout.reset[qi],
            setup_reset=False,
            # Calibration only: re-measure after the final pi pulse so the
            # last recorded round is the true post-reset state. Costs one
            # readout per shot and nothing acts on it, so leave it off in use.
            reset_verify=False,
            # Add a pi_ef ahead of each pi_ge, for sequences that can
            # leave population in |f>. Needs reset >= 2: |f> clears in
            # one round but |e> is driven to |f> first and clears on
            # the next.
            reset_f=check_f,
            # "while excited" semantics at a fixed trip count: once a
            # round reads ground, a latch suppresses every later pi
            # pulse. Measures every round, so the count stays fixed.
            reset_latch=False,
            # Mean rounds the while loop takes, for the cycle-time
            # estimate only; `reset` is its hard cap.
            reset_while_typical=1,
            check_e=True,  # Whether to measure excited state
            check_f=check_f,  # Whether to measure second excited state
            qubit=[qi],  # Qubit index list
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[qi],  # Readout channel
            ddr4=False,  # Whether to use DDR4 memory
            remeas=False,  # Whether to do repeated measurement
            calibrate=False,
            final_delay=self.cfg.device.readout.readout_length[qi],  # Final delay
            # Constraints on the two-state (g/e) fit only. With check_f the
            # three-state joint fit is used instead, which fits tm_e and tm_f
            # separately and floats every residual population (see float_pf).
            float_heights=False,  # Let the two residual-population heights fit independently
            float_tm=False,  # Fit a separate Tm/T1 for the g and e shots
            # Three-state (check_f) fit only: let the no-pi and pi_ge shots
            # carry an |f> population. Neither has a pi_ef in front of it, so
            # turn this on only when an imperfect reset can leave |f>
            # populated going into the next shot.
            float_pf=False,
            # Three-state fit only: resolve the |e> left by an imperfect pi_ef
            # in the f shot. Only identifiable once the e and f blobs separate
            # by more than the readout width, i.e. chi < kappa.
            float_pe_in_f=False,
        )

        # Merge default and user-provided parameters
        self.cfg.expt = {**params_def, **params}

        # Derive the reset parameters for whichever mode is selected.
        if reset_mode_of(self.cfg.expt) != "passive":
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

        # Active reset shortens the inter-shot delay; setup_reset keeps the
        # passive one so a non-working reset cannot corrupt the calibration.
        final_delay = self._get_final_delay()

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
            # The reset protocol follows this scan's own preparation, not the
            # experiment-wide flag. Only the |f>-prepared scan needs the
            # pi_ef leg; applying it to the |e> scan would drive e -> f and
            # park it there for a round, making the |e> reset look far worse
            # than it is (run 33: |e> reached 17.9% with pi_ge alone but only
            # 25.7% with the pi_ef in front of it).
            cfg.expt.reset_f = self.cfg.expt.reset_f and scan_config['pulse_f']

            # Create and configure histogram program
            kwargs = {"soccfg": self.soccfg, "final_delay": final_delay, "cfg": cfg}
            # if self.cfg.expt.active_reset: 
            #     kwargs["final_wait"] = None
            # print(kwargs)
            histpro = HistogramProgram(**kwargs)

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
            if reset_mode_of(self.cfg.expt) == "active" or self.cfg.expt.remeas:
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

        # Always recorded, not just on reset runs: this run writes the
        # threshold and phase a later reset will branch on, so the number has
        # to be in the saved data even when this acquisition is passive.
        data.update(check_hardware_axis(data, verbose=True))

        # Perform detailed single-shot analysis with fitting
        try:
            if self.cfg.expt.check_f and "If" in data:
                # All three shots are fit jointly, with Tm/T1 attached to the
                # state rather than the shot so the no-pi shot's |e> residual
                # decays at tm_e (same as the prepared |e>) while |f> gets its
                # own tm_f. By default each shot is just |g> plus the state it
                # was prepared in; float_pf / float_pe_in_f add the cross-state
                # populations back.
                data2, p = helpers.fit_single_shot_gef(
                    data, plot=False,
                    float_pf=self.cfg.expt.float_pf,
                    float_pe_in_f=self.cfg.expt.float_pe_in_f,
                )
                data.update(p)
                for key in ("vhg", "histg", "vhe", "histe", "vhf", "histf"):
                    data[key] = data2[key]
                data["shots"] = self.cfg.expt.shots
                # g-f fidelity from the fitted |g> and |f> components
                fit_fid, fit_thresh = helpers.fit_fidelity_gef(data)
            else:
                # Fit single-shot data
                data2, p, paramsg, paramse2 = helpers.fit_single_shot(
                    data, plot=False,
                    float_heights=self.cfg.expt.float_heights,
                    float_tm=self.cfg.expt.float_tm,
                )

                # Update data with fit results
                data.update(p)
                data["vhg"] = data2["vhg"]
                data["histg"] = data2["histg"]
                data["vhe"] = data2["vhe"]
                data["histe"] = data2["histe"]
                data["paramsg"] = paramsg
                data["paramsg2"] = data2["paramsg2"]
                data["shots"] = self.cfg.expt.shots
                # Fidelity computed from the fitted g-shot and e-shot distributions
                fit_fid, fit_thresh = helpers.fit_fidelity(data)

            data['e_mean'] = p['e_mean']
            data['g_mean'] = p['g_mean']
            data['g_peak'] = p['g_peak']
            data['e_peak'] = p['e_peak']
            dv = self.data['ve'] - self.data['vg']
            data['e_norm'] = (self.data['e_mean']-self.data['vg'])/dv

            data['g_norm'] = (self.data['g_mean']-self.data['vg'])/dv

            data['fit_fidelity'] = fit_fid
            data['fit_threshold'] = fit_thresh
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
            self.save_config()

    def update(self, freq=True, fast=False, verbose=True, use_peak=False):
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

        # The angle is measured on top of whatever phase HistogramProgram put
        # on the pulse, so add that phase back (it is 0 for passive reset).
        expt = self.cfg.expt
        phase = self.data["angle"]
        if reset_mode_of(expt) != "passive" or expt.remeas or expt.calibrate:
            phase = (self.cfg.device.readout.phase[qi] + phase + 180) % 360 - 180

        # Update readout parameters
        config.update_readout(
            cfg_file, "phase", phase, qi, verbose=verbose
        )
        config.update_readout(
            cfg_file, "threshold", self.data["thresholds"][0], qi, verbose=verbose
        )
        config.update_readout(
            cfg_file, "fidelity", self.data["fids"][0], qi, verbose=verbose
        )
        g_key = "g_peak" if use_peak else "g_mean"
        e_key = "e_peak" if use_peak else "e_mean"
        config.update_readout(
            cfg_file, "g_mean", self.data[g_key], qi, verbose=verbose
        )
        config.update_readout(
            cfg_file, "e_mean", self.data[e_key], qi, verbose=verbose
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
        ax[0].semilogy(vg, histg/max_g, color=BLUE, linewidth=2,
                       label="|g> prep")
        ax[1].semilogy(vg, histg/max_g, color=BLUE, linewidth=2,
                       label="|g> prep (reference)")

        # Create color palette for reset histograms
        b = sns.color_palette("ch:s=-.2,r=.6", n_colors=len(self.data["Igr"]))

        # Create excited state histogram
        ve, histe = helpers.create_histogram(self.data["Ie"], n_bins=n_bins)
        max_e = np.max(histe)
        ax[1].semilogy(ve, histe/max_e, color=RED, linewidth=2, label="|e> prep")

        # The f preparation, when there is one. Worth showing next to |e>:
        # with chi >> kappa the e and f blobs overlap, so an |f> shot that
        # decays to |e> still reads excited and the distribution is visibly
        # cleaner than |e>'s. Its reset rounds go on the same panel.
        has_f = "Ifr" in self.data and "If" in self.data
        if has_f:
            vf, histf = helpers.create_histogram(self.data["If"], n_bins=n_bins)
            ax[1].semilogy(vf, histf/np.max(histf), color="darkgreen",
                           linewidth=2, ls="--", label="|f> prep")
        fig2, ax2 = plt.subplots(2, len(self.data["Igr"])+1, figsize=(17, 7), sharex=True, sharey=True)
        # Plot reset histograms for ground state
        ax2[0,0].plot(self.data['Ig'], self.data['Qg'], '.', markersize=1, rasterized=True)
        ax2[1,0].plot(self.data['Ie'], self.data['Qe'], '.', markersize=1, rasterized=True)
        for i in range(len(self.data["Igr"])):
            ax2[0,i+1].plot(self.data['Igr'][i], self.data['Qgr'][i], '.', markersize=1, rasterized=True)
            ax2[1,i+1].plot(self.data['Ier'][i], self.data['Qer'][i], '.', markersize=1, rasterized=True)
            v, hist = helpers.create_histogram(self.data["Igr"][i], n_bins=n_bins)
            ax[0].semilogy(v, hist/max_g, color=b[i], linewidth=1,
                           ls=_ROUND_STYLES[i % len(_ROUND_STYLES)],
                           label=f"after reset {i+1}", rasterized=True)

            # Plot reset histograms for excited state
            v, hist = helpers.create_histogram(self.data["Ier"][i], n_bins=n_bins)
            ax[1].semilogy(v, hist/max_e, color=b[i], linewidth=1,
                           ls=_ROUND_STYLES[i % len(_ROUND_STYLES)],
                           label=f"|e> after reset {i+1}", rasterized=True)
            if has_f:
                v, hist = helpers.create_histogram(self.data["Ifr"][i], n_bins=n_bins)
                ax[1].semilogy(v, hist/max_e, color="darkgreen", linewidth=1,
                               ls=_ROUND_STYLES[i % len(_ROUND_STYLES)],
                               alpha=0.6, label=f"|f> after reset {i+1}",
                               rasterized=True)

        ax[0].axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax[1].axhline(0.5, color="gray", linestyle="--", linewidth=1)
        # Helper function to find bin index closest to a value
        def find_bin_closest_to_value(bins, value):
            """Return the index of the histogram bin nearest to value."""
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
        # Every curve is labelled: with three preparations and three rounds
        # the panel carries nine traces, and leaving any of them out of the
        # legend makes the figure unreadable (colour alone does not say
        # whether a green trace is the |f> prep or one of its reset rounds).
        ax[0].legend(fontsize="x-small", ncol=2)
        ax[1].legend(fontsize="x-small", ncol=2)
        ax[0].set_title("Ground state")
        ax[1].set_title("Excited state")
        plt.show()
        self.save_fig(fig, "_reset_performance")


# ====================================================== #
