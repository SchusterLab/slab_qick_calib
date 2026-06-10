"""
Single-Shot Readout with Flux Pulse Experiment
===============================================

Variant of the single-shot experiment that applies a constant flux pulse
throughout the pi pulse and readout sequence. All delay_auto calls are
replaced with delay calls so the flux pulse is not waited on — it runs
concurrently with qubit pulses and readout.

Uses sweet_spot_ac gain by default.
"""

import matplotlib.pyplot as plt
import numpy as np
from qick import *
import copy
import seaborn as sns

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment
from ..general.qick_program import QickProgram

from ...helpers import config
from ...calib import readout_helpers as helpers

# Standard colors for plotting
BLUE = "#4053d3"
RED = "#b51d14"
GREEN = "#2ca02c"

# ====================================================== #


class HistogramFluxProgram(QickProgram):
    """
    Pulse sequence for single-shot readout with a concurrent flux pulse.

    The flux pulse stays on through the entire sequence: settle time,
    pi pulse(s), wait, and readout.
    """

    def __init__(self, soccfg, final_delay, cfg, final_wait=0):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg, final_wait=final_wait)

    def _initialize(self, cfg):
        """Set up readout, pi pulses (at f_ge_max), and a const flux pulse long enough to cover the whole shot."""
        cfg = AttrDict(self.cfg)
        self.add_loop("shotloop", cfg.expt.shots)

        # Configure readout parameters
        self.frequency = cfg.expt.frequency
        self.gain = cfg.expt.gain
        if cfg.expt.active_reset or cfg.expt.remeas or cfg.expt.calibrate:
            self.phase = cfg.device.readout.phase[cfg.expt.qubit[0]]
        else:
            self.phase = 0
        self.readout_length = cfg.expt.readout_length

        super()._initialize(cfg, readout="")

        # Define pi pulses for state preparation (use f_ge_max for flux sweet spot)
        q = cfg.expt.qubit[0]
        super().make_cfg_pulse(q, cfg.device.qubit.f_ge_max, "pi_ge")
        if cfg.expt.pulse_f:
            super().make_cfg_pulse(q, cfg.device.qubit.f_ef, "pi_ef")

        # Set up flux pulse
        self.flux_ch = cfg.expt.flux_chan
        self.declare_gen(self.flux_ch, nqz=1, mixer_freq=0)

        # Estimate pi pulse duration from config
        pi_cfg = cfg.device.qubit.pulses.pi_ge
        if pi_cfg.type[q] == "gauss":
            pi_dur = pi_cfg.sigma[q] * pi_cfg.sigma_inc[q]
        else:
            pi_dur = pi_cfg.get("length", pi_cfg.sigma)[q]

        # Flux length covers: settle + pi pulse(s) + wait + readout + margin
        flux_length = (cfg.expt.flux_settle + pi_dur + 0.1
                       + cfg.expt.readout_length + 0.5)

        flux_pulse = {
            "chan": self.flux_ch,
            "freq": 0,
            "phase": 0,
            "gain": cfg.expt.flux_gain,
            "length": flux_length,
            "type": "const",
        }
        super().make_pulse(flux_pulse, "flux_pulse")

        # Add initial delay for tProc setup
        self.delay(0.5)

    def _body(self, cfg):
        """Start the flux pulse, settle, optionally prepare e/f, then read out with flux still on."""
        cfg = AttrDict(self.cfg)

        # Configure readout
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # Start flux pulse (runs concurrently with everything below)
        self.pulse(ch=self.flux_ch, name="flux_pulse", t=0)
        self.delay(t=cfg.expt.flux_settle, tag="wait_flux_on")

        # Apply pi pulse to prepare excited state if requested
        if cfg.expt.pulse_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)

        # Apply second pi pulse to prepare f state if requested
        if cfg.expt.pulse_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)

        # Add small delay before readout
        if cfg.expt.calibrate:
            self.delay(t=0.1, tag="wait")
        else:
            self.delay(t=0.01, tag="wait")

        # Apply readout pulse and trigger data acquisition (flux still on)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.trigger(ros=[self.adc_ch], ddr4=True, pins=[0], t=self.trig_offset)

    def collect_shots(self, offset=0):
        """Return flattened per-shot (I, Q) arrays from the raw ADC buffer."""
        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            iq_raw = self.get_raw()
            i_shots = iq_raw[i][:, :, 0, 0]
            i_shots = i_shots.flatten()
            q_shots = iq_raw[i][:, :, 0, 1]
            q_shots = q_shots.flatten()

        return i_shots, q_shots


class HistogramFluxExperiment(QickExperiment):
    """
    Single-shot readout experiment with concurrent flux pulse.

    Same as HistogramExperiment but applies a constant flux tone throughout
    the pulse sequence. Defaults to sweet_spot_ac gain.
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
        if prefix is None:
            prefix = f"single_shot_flux_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = dict(
            shots=10000,
            reps=1,
            rounds=1,
            readout_length=self.cfg.device.readout.readout_length[qi],
            frequency=self.cfg.device.readout.frequency[qi],
            gain=self.cfg.device.readout.gain[qi],
            active_reset=False,
            reset=7,
            check_e=True,
            check_f=check_f,
            qubit=[qi],
            qubit_chan=self.cfg.hw.soc.adcs.readout.ch[qi],
            ddr4=False,
            remeas=False,
            calibrate=False,
            final_delay=self.cfg.device.readout.readout_length[qi],
            # Flux parameters
            flux_chan=self.cfg.hw.soc.dacs.flux.ch[qi],
            flux_gain=self.cfg.device.qubit.sweet_spot_ac[qi],
            flux_settle=0.05,
        )

        self.cfg.expt = {**params_def, **params}

        if self.cfg.expt.active_reset:
            super().configure_reset()

        if print:
            super().print()
            go = False
        if go:
            self.go(analyze=True, display=display, progress=progress, save=True)

    def acquire(self, progress=False, debug=False):
        """Collect single-shot I/Q data for ground (and optionally e/f) preparations at the flux point."""
        data = dict()

        if "setup_reset" in self.cfg.expt and self.cfg.expt.setup_reset:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
        elif self.cfg.expt.active_reset:
            final_delay = 1
        else:
            final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]

        if self.cfg.expt.ddr4:
            n_transfers = 1500000
            nt = n_transfers

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

        if self.cfg.expt.check_e:
            scan_configs.append({
                'name': 'excited',
                'pulse_e': True,
                'pulse_f': False,
                'data_keys': ('Ie', 'Qe', 'Ier', 'Qer', 't_e', 'iq_ddr4_e'),
                'enabled': True,
                'acquire_kwargs': {}
            })

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

        for scan_config in scan_configs:
            if debug:
                print(f"Acquiring {scan_config['name']} state data...")

            cfg = AttrDict(copy.deepcopy(dict(self.cfg)))
            cfg.expt.pulse_e = scan_config['pulse_e']
            cfg.expt.pulse_f = scan_config['pulse_f']

            kwargs = {"soccfg": self.soccfg, "final_delay": final_delay, "cfg": cfg}
            histpro = HistogramFluxProgram(**kwargs)

            if self.cfg.expt.ddr4:
                self.im[self.cfg.aliases.soc].arm_ddr4(
                    ch=self.cfg.expt.qubit_chan, nt=n_transfers
                )

            acquire_kwargs = {
                'threshold': None,
                'progress': progress,
                **scan_config['acquire_kwargs']
            }

            iq_list = histpro.acquire(
                self.im[self.cfg.aliases.soc],
                **acquire_kwargs
            )

            i_key, q_key, ir_key, qr_key, t_key, ddr4_key = scan_config['data_keys']

            data[i_key] = iq_list[0][0][:, 0]
            data[q_key] = iq_list[0][0][:, 1]

            if self.cfg.expt.active_reset or self.cfg.expt.remeas:
                data[ir_key] = iq_list[0][1:, :, 0]
                data[qr_key] = iq_list[0][1:, :, 1]

            if self.cfg.expt.ddr4:
                iq_ddr4 = self.im[self.cfg.aliases.soc].get_ddr4(nt)
                t = histpro.get_time_axis_ddr4(self.cfg.expt.qubit_chan, iq_ddr4)
                data[t_key] = t
                data[ddr4_key] = iq_ddr4

        self.data = data
        return data

    def analyze(self, data=None, span=None, verbose=False, **kwargs):
        """Rotate IQ data, fit g/e histograms, and compute fidelity and normalized state voltages."""
        if data is None:
            data = self.data

        params, _ = helpers.analyze_single_shot_histograms(data=data, plot=False, span=span, verbose=verbose)
        data.update(params)

        try:
            data2, p, paramsg, paramse2 = helpers.fit_single_shot(data, plot=False)

            data.update(p)
            data["vhg"] = data2["vhg"]
            data["histg"] = data2["histg"]
            data["vhe"] = data2["vhe"]
            data["histe"] = data2["histe"]
            data["paramsg"] = paramsg
            data["shots"] = self.cfg.expt.shots
            data['e_mean'] = p['e_mean']
            data['g_mean'] = p['g_mean']
            data['g_peak'] = p['g_peak']
            data['e_peak'] = p['e_peak']
            dv = self.data['ve'] - self.data['vg']
            data['e_norm'] = (self.data['e_mean']-self.data['vg'])/dv
            data['g_norm'] = (self.data['g_mean']-self.data['vg'])/dv
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
        """Plot the single-shot IQ clouds and histograms with fidelity annotations."""
        if data is None:
            data = self.data

        if ax is not None:
            savefig = False
        else:
            savefig = True

        params, fig = helpers.analyze_single_shot_histograms(
            data=data,
            plot=plot,
            verbose=verbose,
            span=span,
            ax=ax,
            qubit=self.cfg.expt.qubit[0],
        )

        fids = params["fids"]
        thresholds = params["thresholds"]
        angle = params["angle"]

        if "expt" not in self.cfg:
            self.cfg.expt.check_e = plot_e
            self.cfg.expt.check_f = plot_f

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

        if savefig:
            plt.show()
            self.save_fig(fig)
            self.save_config()

    def update(self, freq=True, fast=False, verbose=True, use_peak=False):
        """Write the fitted readout phase, threshold, fidelity, and g/e means back to the config file."""
        qi = self.cfg.expt.qubit[0]
        cfg_file = self.config_file

        config.update_readout(cfg_file, "phase", self.data["angle"], qi, verbose=verbose)
        config.update_readout(cfg_file, "threshold", self.data["thresholds"][0], qi, verbose=verbose)
        config.update_readout(cfg_file, "fidelity", self.data["fids"][0], qi, verbose=verbose)
        g_key = "g_peak" if use_peak else "g_mean"
        e_key = "e_peak" if use_peak else "e_mean"
        config.update_readout(cfg_file, "g_mean", self.data[g_key], qi, verbose=verbose)
        config.update_readout(cfg_file, "e_mean", self.data[e_key], qi, verbose=verbose)

        if not fast:
            config.update_readout(cfg_file, "sigma", self.data["sigma"], qi, verbose=verbose)
            config.update_readout(cfg_file, "tm", self.data["tm"], qi, verbose=verbose)

            if self.data["fids"][0] > 0.07:
                config.update_qubit(cfg_file, "tuned_up", True, qi, verbose=verbose)
            else:
                config.update_qubit(cfg_file, "tuned_up", False, qi, verbose=verbose)
                print("Readout not tuned up")


# ====================================================== #
