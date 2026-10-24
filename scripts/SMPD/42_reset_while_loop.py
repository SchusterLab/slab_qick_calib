"""Try a true while-loop active reset, self-contained; see CLAUDE.md "While-loop reset"."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2  # noqa: F401  (documents the base)

from slab_qick_calib.exp_handling.datamanagement import AttrDict
from slab_qick_calib.experiments.general.qick_experiment import QickExperiment
from slab_qick_calib.experiments.general.qick_program import QickProgram

from common import parse_args, prepare_experiment, wait_for_plots


class WhileResetProgram(QickProgram):
    """Reset by looping until the readout says ground, counting the rounds."""

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        q = cfg.expt.qubit[0]
        self.frequency = cfg.device.readout.frequency[q]
        self.gain = cfg.device.readout.gain[q]
        self.phase = cfg.device.readout.phase[q]
        self.readout_length = cfg.expt.readout_length
        super()._initialize(cfg, readout="standard")
        super().make_cfg_pulse(q, cfg.device.qubit.f_ge, "pi_ge")

        # Feedback readout: a second logical channel on the same physical
        # input, so the loop's variable trigger count lands in a buffer we
        # discard while self.adc_ch keeps exactly one trigger per rep and
        # averages correctly. PFB is static, so the frequency goes in the
        # declaration. Indices 8-9 have no tProc input and cannot be used.
        self.fb_ch = cfg.expt.feedback_ch
        # phase=0, like the shared static path: readout_pulse already carries
        # device.readout.phase, and passing it here too rotates by 2*phase --
        # which leaves this branch reading an inverted, mostly-noise axis.
        self.declare_readout(ch=self.fb_ch, length=self.readout_length,
                             freq=self.frequency, phase=0,
                             gen_ch=self.res_ch)

        # One dmem word per shot, so the address register advances across the
        # whole acquisition and must be zeroed outside the loop.
        self.addr_reg = self.add_reg(name="dmem_addr")
        self.count_reg = self.add_reg(name="round_count")
        self.write_reg(dst=self.addr_reg, src=cfg.expt.dmem_base)

        self.add_loop("shot_loop", cfg.expt.shots)

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        # Prepare |e> so the loop has something to clear. With prep_e False
        # the shot starts in |g> and the round count measures how often a
        # ground shot is misread as excited instead.
        if cfg.expt.prep_e:
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            self.delay_auto(t=0.01)

        self.write_reg(dst=self.count_reg, src=0)

        self.label("RESET_LOOP")
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(ros=[self.fb_ch], pins=[0], t=self.trig_offset)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.wait_auto(cfg.expt.read_wait)
        self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)
        # The loop runs for an unknown time, so the reference time has to be
        # pushed past the execution time or later timed instructions pile up.
        self.resync(0.05)

        # Ground: done. This is the only exit that means the reset worked.
        self.read_and_jump(ro_ch=self.fb_ch, component="I",
                           threshold=cfg.expt.threshold, test="<",
                           label="RESET_DONE")
        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        self.delay_auto(0.01)
        self.inc_reg(dst=self.count_reg, src=1)
        # Bail out at max_rounds. Without this a qubit that never reads ground
        # hangs the tProc and the board needs a reset, so it is not optional.
        self.cond_jump(label="RESET_DONE", arg1=self.count_reg, op="-",
                       arg2=cfg.expt.max_rounds, test="NS")
        self.jump("RESET_LOOP")

        self.label("RESET_DONE")
        # Per-shot round count, read back from data memory rather than the
        # averaging buffer: the loop triggers a variable number of readouts per
        # rep, so the buffer is misaligned by construction. See CLAUDE.md.
        self.write_dmem(addr=self.addr_reg, src=self.count_reg)
        self.inc_reg(dst=self.addr_reg, src=1)

        # The measurement: exactly one trigger per rep on self.adc_ch, so this
        # one is correctly averaged however long the loop ran.
        self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)
        self.resync(0.05)
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)


class WhileResetExperiment(QickExperiment):
    """Round-count distribution for a while-loop reset."""

    def __init__(self, cfg_dict, qi=0, go=False, params={}, prefix=None,
                 progress=True):
        prefix = prefix or f"reset_while_loop_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress)

        params_def = {
            "shots": 500,
            "reps": 1,
            "rounds": 1,
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "prep_e": True,
            "max_rounds": 20,
            "dmem_base": 0,
            # ADC5's PFB bank sees the same waste signal; 2-7 have a tProc
            # input, 8-9 do not. See 00_soc_map.py.
            "feedback_ch": 2,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            # Passive wait between shots: each shot must start from |g> for the
            # round count to mean anything.
            "final_delay": self.cfg.device.readout.final_delay[qi],
            "reset_mode": "passive",
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            "reset_verify": False,
            "reset_latch": False,
            "reset_f": False,
        }
        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        # threshold / read_wait / extra_delay, which the loop branches on.
        super().configure_reset()

        if go:
            self.run(progress=progress, display=True, save=True)

    def acquire(self, progress=False):
        q = self.cfg.expt.qubit[0]
        prog = WhileResetProgram(soccfg=self.soccfg,
                                 final_delay=self.cfg.expt.final_delay,
                                 cfg=self.cfg)
        prog.acquire(self.im[self.cfg.aliases.soc],
                     rounds=self.cfg.expt.rounds, threshold=None,
                     progress=progress)
        soc = self.im[self.cfg.aliases.soc]
        counts = np.asarray(
            soc.read_mem(self.cfg.expt.shots, "dmem",
                         self.cfg.expt.dmem_base)).astype(int)

        # collect_shots returns one entry per declared readout, in declaration
        # order: self.adc_ch first, the feedback channel second. Only the first
        # is meaningful -- the feedback buffer is misaligned by construction.
        i_shots, q_shots = prog.collect_shots()
        i_meas = np.asarray(i_shots[0])[..., 0].flatten()
        q_meas = np.asarray(q_shots[0])[..., 0].flatten()

        self.data = {"rounds": counts,
                     "I": i_meas, "Q": q_meas,
                     "threshold_v": self.cfg.expt.threshold_v,
                     "prep_e": self.cfg.expt.prep_e}
        return self.data

    def analyze(self, data=None, **kwargs):
        data = self.data if data is None else data
        r = np.asarray(data["rounds"])
        capped = r >= self.cfg.expt.max_rounds
        data["mean_rounds"] = float(r[~capped].mean()) if (~capped).any() else np.nan
        data["frac_capped"] = float(capped.mean())
        data["frac_zero"] = float((r == 0).mean())
        # The post-reset state, from the one correctly-averaged readout. This
        # is what the single-channel version could not measure: the loop exits
        # on a ground *read*, which a misread satisfies just as well.
        if "I" in data:
            thr = self.cfg.device.readout.threshold[self.cfg.expt.qubit[0]]
            data["p_excited_after"] = float(np.mean(np.asarray(data["I"]) > thr))
        return data

    def display(self, data=None, **kwargs):
        data = self.data if data is None else data
        r = np.asarray(data["rounds"])
        fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
        bins = np.arange(-0.5, self.cfg.expt.max_rounds + 1.5)
        ax.hist(r, bins=bins, color="#4053d3",
                label=f"{'|e> prepared' if data['prep_e'] else '|g> (control)'}, "
                      f"{r.size} shots")
        ax.axvline(self.cfg.expt.max_rounds, color="#b51d14", ls="--", lw=1,
                   label=f"cap ({self.cfg.expt.max_rounds}): did not reach |g>")
        ax.set_yscale("log")
        ax.set_xlabel("pi pulses before the readout said ground")
        ax.set_ylabel("shots")
        ax.legend(fontsize="small")
        fig.tight_layout()
        self.save_fig(fig)


if __name__ == "__main__":
    CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"
    qi = 1
    JPA_ON = True
    SHOTS = 500             # one dmem word each, so keep well under dmem depth
    MAX_ROUNDS = 20         # tProc hangs without a cap; see _body
    # Feedback index on ADC5's PFB bank (2-7 have a tProc input, 8-9 do not).
    # The measurement stays on adcs.readout.ch[qi]. Swapped from 2 to 3 to test
    # whether the loop's branching follows the channel: #664 measured index 2
    # and index 3 as equally misaligned (-0.299, -0.289 of the separation), yet
    # #662 branched correctly with 2 as the feedback.
    FEEDBACK_CH = 3
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON)

    results = {}
    for prep_e in (False, True):
        label = "|e> prepared" if prep_e else "|g> control"
        print(f"\n=== {label} ===")
        expt = WhileResetExperiment(
            cfg_dict, qi=qi, go=False,
            params={"shots": SHOTS, "prep_e": prep_e,
                    "max_rounds": MAX_ROUNDS, "feedback_ch": FEEDBACK_CH})
        expt.run(progress=not args.no_progress, display=not args.no_show,
                 save=True)
        d = expt.data
        print(f"  mean rounds (excluding capped) : {d['mean_rounds']:.3f}")
        print(f"  exited immediately (0 pulses)  : {100*d['frac_zero']:.2f} %")
        print(f"  hit the cap                    : {100*d['frac_capped']:.2f} %")
        if "p_excited_after" in d:
            print(f"  P(excited) after the reset     : "
                  f"{100*d['p_excited_after']:.2f} %   <- the measurement channel")
        results[label] = d

    g, e = results["|g> control"], results["|e> prepared"]
    print("\nThe |g> control's non-zero fraction is the false-excited rate the")
    print("fixed-round reset spends its pi pulses on:")
    print(f"  P(ground shot read as excited) ~ {100*(1-g['frac_zero']):.2f} %")
    print(f"  |e> needs {e['mean_rounds']:.2f} pulses on average")

    if not args.no_show:
        wait_for_plots()
