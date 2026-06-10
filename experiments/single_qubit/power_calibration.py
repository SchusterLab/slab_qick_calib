"""
Power Calibration Experiment

Sends a constant readout pulse at a fixed frequency and measures the IQ response.
Used for calibrating RF board power levels after configuring filters and attenuators.

The program extends AveragerProgramV2 directly (rather than QickProgram) to allow
specifying arbitrary generator and readout channels via params.
"""

import numpy as np

from qick import *
from qick.asm_v2 import AveragerProgramV2

from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment


class PowerCalibrationProgram(AveragerProgramV2):
    def __init__(self, soccfg, final_delay, cfg):
        cfg = AttrDict(cfg)
        reps = cfg.expt.reps
        super().__init__(soccfg, reps, final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        gen_ch = cfg.expt.gen_ch
        ro_ch = cfg.expt.ro_ch

        self.declare_gen(ch=gen_ch, nqz=cfg.expt.nqz)
        self.declare_readout(ch=ro_ch, length=cfg.expt.ro_len)
        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg.expt.frequency, gen_ch=gen_ch)
        self.add_pulse(
            ch=gen_ch,
            name="mypulse",
            ro_ch=ro_ch,
            style="const",
            freq=cfg.expt.frequency,
            length=cfg.expt.length,
            phase=cfg.expt.phase,
            gain=cfg.expt.gain,
            mode="periodic",
        )
        self.send_readoutconfig(ch=ro_ch, name="myro", t=0)

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)
        self.pulse(ch=cfg.expt.gen_ch, name="mypulse", t=0)
        self.trigger(ros=[cfg.expt.ro_ch], t=cfg.expt.trig_time)


class PowerCalibration(QickExperiment):
    """
    Measures IQ response of a constant readout pulse for power calibration.

    All hardware channel params (gen_ch, ro_ch, nqz) default to the qubit-indexed
    config values but can be overridden via the params dict to test arbitrary channels.
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
        save=True,
    ):
        if prefix is None:
            prefix = f"power_calibration_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "gen_ch": self.cfg.hw.soc.dacs.readout.ch[qi],
            "ro_ch": self.cfg.hw.soc.adcs.readout.ch[qi],
            "nqz": self.cfg.hw.soc.dacs.readout.nyquist[qi],
            "frequency": self.cfg.device.readout.frequency[qi],
            "gain": 0,
            "length": 0.05,
            "ro_len": 0.3,
            "trig_time": 0.45,
            "phase": 0.0,
            "reps": 1,
            "rounds": 1,
            "final_delay": 0.5,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }

        params = {**params_def, **params}
        self.cfg.expt = params

        if go:
            super().run(display=display, progress=progress, save=save, analyze=False)

    def acquire(self, progress=False):
        final_delay = self.cfg.expt.final_delay

        prog = PowerCalibrationProgram(
            soccfg=self.soccfg,
            final_delay=final_delay,
            cfg=self.cfg,
        )

        iq_list = prog.acquire(
            self.im[self.cfg.aliases.soc],
            rounds=self.cfg.expt.rounds,
            threshold=None,
            progress=progress,
        )

        iq = iq_list[0][0]
        avgi = np.squeeze(iq[..., 0])
        avgq = np.squeeze(iq[..., 1])
        amp = np.abs(avgi + 1j * avgq)

        self.data = {
            "avgi": avgi,
            "avgq": avgq,
            "amps": amp,
        }
        return self.data

    def display(self, data=None, ax=None, **kwargs):
        if data is None:
            data = self.data

        avgi = data["avgi"]
        avgq = data["avgq"]
        amp = data["amps"]

        qi = self.cfg.expt.qubit[0]
        print(f"Power Calibration Q{qi}")
        print(f"  gen_ch={self.cfg.expt.gen_ch}, ro_ch={self.cfg.expt.ro_ch}, "
              f"freq={self.cfg.expt.frequency:.3f} MHz, gain={self.cfg.expt.gain}")
        print(f"  I={float(np.mean(avgi)):.6f}, Q={float(np.mean(avgq)):.6f}, "
              f"amp={float(np.mean(amp)):.6f}")
