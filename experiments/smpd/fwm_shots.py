"""
SMPD FWM Single-Shot Histogram

Measures the single-shot readout distribution at one fixed set of four-wave
mixing parameters. There are no parameter sweeps -- just one acquisition per
sequence, differing only in which drives are on:

    g  no drives      (neither, just the wait) -- the ground reference
    e  pump + buffer  (both on) -- the conversion
    p  pump only      (pump on, buffer off, on band) -- the dark count
    d  pump only      (pump on, detuned cfg.expt.detune_p off the band)
    b  buffer only    (pump off, buffer on) -- not requested by default

Comparing them separates the contributions: the readout reports some excited
population with nothing driven at all, and the pump adds some on top. `d`
splits that pump term in two, since it is the same power off resonance, so
`p - d` is the part that needs the FWM band and `d - g` the part that does not.
Only the pump+buffer sequence contains the actual conversion.

Use the `scans` parameter to drop reference scans and save acquisition time;
'g' and 'e' are required. See CLAUDE.md "The detuned-pump scan" for why `b` is
off by default and when to put it back.

`check_f=True` adds a scan that runs the same pump + buffer sequence and
then applies a pi_ef, so the converted population is read out on the g/f axis
instead of g/e. It is an alternative readout of the conversion, not a separate
state preparation -- useful when the |f> blob is better separated than |e>.

Every sequence takes the same time: the no-drives scan (`g`) waits out the
window the drives would have occupied, so all of them see the same relaxation
history before readout.

No model is fit to the distributions. The raw histograms are reported and
plotted as acquired, alongside the thresholds and fidelities that
`helpers.analyze_single_shot_histograms` gets by counting shots.

The pulse sequence and parameter names follow the other FWM experiments
(`single_qubit/FWM_sweeps.py`, `single_qubit/qubit_spec_shot.py`): `_p` is the
pump on the qubit DAC, `_b` the buffer on readout DAC 0, and readout happens on
the waste line `qi`. `do_pump` / `do_buffer` switch the two drives, exactly as
in `QubitSpecShotProgram`.
"""

import matplotlib.pyplot as plt
import numpy as np

from ...calib import readout_helpers as helpers
from ...exp_handling.datamanagement import AttrDict
from ..general.qick_experiment import QickExperiment
from ..general.qick_program import (
    QickProgram,
    mixer_freq_for,
    reset_mode_of,
    reset_plays_pi_pulses,
    pulse_type_from_config,
)


def _reset_rounds(shots):
    """
    Pull the active-reset readouts out of a collect_shots array.

    `collect_shots` hands back `(*loop_dims, n_reads)`; readout 0 is the real
    measurement and 1.. are the rounds `QickProgram.reset()` added. Returns
    them as `(n_rounds, n_shots)` -- rounds first, matching what
    `HistogramExperiment.check_reset()` expects of `Igr` -- or None when reset
    is off or `reset=1`, neither of which produces an extra readout.
    """
    shots = np.asarray(shots)
    n_rounds = shots.shape[-1] - 1
    if n_rounds < 1:
        return None
    return np.moveaxis(shots, -1, 0)[1:].reshape(n_rounds, -1)


class SMPDHistogramProgram(QickProgram):
    """
    Pulse sequence for a single-shot FWM histogram.

    Identical in structure to `QubitSpecShotProgram` -- pump on the qubit
    channel, buffer on readout DAC 0, readout on the waste line -- except the
    hardware loop counts shots instead of stepping a frequency.
    """

    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)

        # Readout on the waste line for this qubit.
        q = cfg.expt.qubit[0]
        self.frequency = cfg.device.readout.frequency[q]
        self.gain = cfg.device.readout.gain[q]
        self.readout_length = cfg.expt.readout_length
        self.phase = cfg.device.readout.phase[q]

        super()._initialize(cfg, readout="standard")

        # Second generator for the buffer tone on readout DAC 0.
        buffer_ch = cfg.hw.soc.dacs.readout.ch[0]
        buffer_gen_args = {"ch": buffer_ch}
        if self.soccfg["gens"][buffer_ch].get("has_mixer", False):
            buffer_gen_args["mixer_freq"] = mixer_freq_for(cfg.expt.frequency_b)
        self.declare_gen(**buffer_gen_args)

        # The pump goes on the pump line, which is the qubit channel unless
        # hw.soc.dacs.pump says otherwise.
        pump_pulse = {
            "freq": cfg.expt.frequency_p,
            "gain": cfg.expt.gain_p,
            "type": cfg.expt.pulse_type_p,
            "sigma": cfg.expt.length_p,
            "phase": 0,
            "chan": self.pump_ch,
        }
        super().make_pulse(pump_pulse, "pump_pulse")

        buffer_pulse = {
            "freq": cfg.expt.frequency_b,
            "gain": cfg.expt.gain_b,
            "type": cfg.expt.pulse_type_b,
            "sigma": cfg.expt.length_b,
            "phase": 0,
            "chan": buffer_ch,
        }
        super().make_pulse(buffer_pulse, "buffer_pulse")

        # The f-readout scan transfers the converted population |e> -> |f>,
        # so it needs a calibrated e-f pi pulse. Guarded so a normal run does
        # not require f_ef / pulses.pi_ef to exist.
        # reset_f plays pi_ef ahead of each pi_ge, so it needs the pulse too.
        if cfg.expt.check_f or cfg.expt.reset_f:
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ef, "pi_ef")

        # The measurement-based resets play pi_ge after the readout, so the
        # pulse has to exist even though no scan prepares |e> deliberately.
        # Note the 'f' scan leaves population in |f>, which a pi_ge-only reset
        # conditioned on the g/e threshold does not clear.
        if reset_plays_pi_pulses(cfg.expt):
            super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

        # One loop, counting shots.
        self.add_loop("shot_loop", cfg.expt.shots)

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)

        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        gap = 0.01 if cfg.expt.sep_readout else 0.0

        if cfg.expt.do_pump:
            self.pulse(ch=self.pump_ch, name="pump_pulse", t=0)
        if cfg.expt.do_buffer:
            self.pulse(
                ch=cfg.hw.soc.dacs.readout.ch[0],
                name="buffer_pulse",
                t=cfg.expt.delay_b,
            )

        if cfg.expt.do_pump or cfg.expt.do_buffer:
            # Let the drives finish. Required whenever a transfer pulse
            # follows, so it cannot overlap the pump or buffer.
            if cfg.expt.sep_readout or cfg.expt.do_pulse_f:
                self.delay_auto(t=gap, tag="wait")
        else:
            # Nothing was played, so delay_auto has no pulse end to reference
            # and would put the readout at t=gap -- tens of microseconds
            # earlier than the other sequences. Wait out the same window the
            # drives would have occupied, so every sequence has identical
            # timing and the readout sees the same relaxation history.
            # A flat_top runs its ramps (make_pulse's default 0.02 us x 3)
            # past the flat length.
            ramp_p = 0.06 if cfg.expt.pulse_type_p == "flat_top" else 0
            ramp_b = 0.06 if cfg.expt.pulse_type_b == "flat_top" else 0
            self.delay(
                max(
                    cfg.expt.length_p + ramp_p,
                    cfg.expt.delay_b + cfg.expt.length_b + ramp_b,
                ) + gap,
                tag="wait",
            )

        if cfg.expt.do_pulse_f:
            # Move whatever the four-wave mixing converted into |e> up to |f>,
            # so the same conversion is read out on the g/f axis instead of
            # g/e. This is an alternative readout of the pump+buffer scan, not
            # a separate state preparation -- no pi_ge, which would drive the
            # unconverted population out of |g> and destroy the measurement.
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
            self.delay_auto(t=gap, tag="wait_ef")

        super().measure(cfg)


class SMPDHistogramExperiment(QickExperiment):
    """
    Single-shot FWM histogram at one set of parameters.

    Runs the sequence once per entry in `SCANS` and reports the raw shot
    distribution for each. Nothing is fit to them.

    Parameters:
    - 'shots': Number of single shots per acquisition
    - 'frequency_p', 'gain_p', 'length_p': the pump (qubit DAC)
    - 'frequency_b', 'gain_b', 'length_b', 'delay_b': the buffer (readout DAC 0)
    - 'scans': Which sequences to acquire; see SCANS
    - 'check_f': Add the pump + buffer + pi_ef scan, read out on |f>
    - 'sep_readout': Insert a short delay between the drives and readout
    - 'bins': Histogram bin count

    Default parameters are defined in the `__init__` method.
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
        min_r2=None,
        max_err=None,
    ):
        """
        Initialize the FWM histogram experiment.

        Args:
            cfg_dict: Configuration dictionary
            prefix: Prefix for data files
            progress: Whether to show progress bar
            qi: Qubit index (the waste/readout line)
            go: Whether to immediately run the experiment
            check_f: Add the pump + buffer + pi_ef scan, read out on |f>
            params: Additional parameters to override defaults
            display: Whether to display results
            min_r2: Unused, accepted for interface compatibility
            max_err: Unused, accepted for interface compatibility
        """
        if prefix is None:
            prefix = f"fwm_histogram_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

        params_def = {
            "shots": 10000,
            "reps": 1,
            "rounds": 1,
            "readout_length": self.cfg.device.readout.readout_length[qi],
            "final_delay": self.cfg.device.readout.final_delay[qi],
            # Pump, on the qubit DAC.
            "frequency_p": self.cfg.device.qubit.f_ge[qi],
            "gain_p": self.cfg.device.qubit.pulses.pi_ge.gain[qi],
            "length_p": 30,
            "pulse_type_p": pulse_type_from_config(self.cfg.device.get("fwm"), "pump_pulse_type"),
            # Buffer, on readout DAC 0.
            "frequency_b": self.cfg.device.readout.frequency[0],
            "gain_b": self.cfg.device.readout.gain[0],
            "length_b": 30,
            "pulse_type_b": pulse_type_from_config(self.cfg.device.get("fwm"), "buffer_pulse_type"),
            "delay_b": 0,
            # Sequence switches, same names as the other FWM experiments.
            # acquire() drives do_pump / do_buffer per scan; these are the
            # values cfg is left at.
            "do_pump": True,
            "do_buffer": True,
            "do_pulse_f": False,
            "sep_readout": True,
            # Adds a scan that repeats the pump + buffer sequence and then
            # transfers |e> -> |f>, reading the same conversion on the g/f axis.
            "check_f": check_f,
            # Which sequences to acquire; see SCANS. 'g' and 'e' are required,
            # the rest are reference scans and can be dropped to save
            # acquisition time.
            "scans": ("g", "e", "p", "d"),
            # How far the 'd' scan detunes the pump off the band, in MHz.
            # 30 is >=3 resonance FWHM at every gain; CLAUDE.md
            # "The detuned-pump scan".
            "detune_p": 30.0,
            # reset is the round count for reset_mode "active" only;
            # QickProgram.reset() plays up to that many conditional pi pulses.
            # setup_reset keeps the long passive inter-shot delay whatever the
            # mode, so a reset that is not working cannot quietly corrupt the
            # data.
            # Which reset runs between shots: "passive" (the full T1
            # wait), "active" (measurement-based conditional pi pulses)
            # or "f0g1" (the unconditional all-microwave reset, which
            # also clears |f> -- see QickProgram.reset_f0g1).
            "reset_mode": "passive",
            # "f0g1" only, and all optional -- everything else is
            # derived from device.f0g1. residual sets the drive length
            # as ln(1/residual)/(2*pi*Gamma), so it rescales on its own
            # as Gamma improves; f0g1_reset_length pins the length
            # instead. None means device.f0g1.reset_*, else derived.
            "f0g1_reset_residual": None,
            "f0g1_reset_gain": None,
            "f0g1_reset_gain_ef": None,
            "f0g1_reset_length": None,
            "f0g1_reset_gap": None,
            "reset": self.cfg.device.readout.reset[qi],
            "setup_reset": False,
            # Calibration only: re-measure after the final pi pulse so the
            # last recorded round is the true post-reset state. Costs one
            # readout per shot and nothing acts on it, so leave it off in use.
            "reset_verify": False,
            # Add a pi_ef ahead of each pi_ge, for sequences that can leave
            # population in |f>. Needs reset >= 2: |f> clears in one round but
            # |e> is driven to |f> first and clears on the next.
            # params.get, not the bare keyword: the runner scripts pass
            # check_f inside `params`, so keying off the __init__ default
            # left reset_f False on every real run and the f scan's |f>
            # population was never cleared by the conditional reset.
            "reset_f": params.get("check_f", check_f),
            # "while excited" semantics at a fixed trip count: once a round
            # reads ground, a latch suppresses every later pi pulse. Measures
            # every round, so the trigger count stays fixed.
            "reset_latch": False,
            # Mean rounds the while loop takes, for the cycle-time estimate
            # only; `reset` is its hard cap. See CLAUDE.md.
            "reset_while_typical": 1,
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "bins": 100,
        }

        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)

        # configure_reset() fills in threshold / read_wait / extra_delay / reset,
        # which QickProgram.reset() needs; only required when reset is enabled.
        if reset_mode_of(self.cfg.expt) != "passive":
            super().configure_reset()

        if go:
            self.run(progress=progress, display=display)

    # (label, do_pump, do_buffer, do_pulse_f, detuned, passive) per acquisition.
    # 'g' and 'e' come first so an interrupted run still has the headline pair.
    # `detuned` offsets the pump by cfg.expt.detune_p; see CLAUDE.md
    # "The detuned-pump scan". `passive` swaps the reset for the T1 wait; see
    # CLAUDE.md "The passive-wait scan".
    SCANS = (
        ("g", False, False, False, False, False),  # no drives -- the ground reference
        ("e", True, True, False, False, False),    # pump + buffer
        ("p", True, False, False, False, False),   # pump only, on band
        ("d", True, False, False, True, False),    # pump only, detuned off the band
        ("b", False, True, False, False, False),   # buffer only; not on by default
        ("f", True, True, True, False, False),     # pump + buffer, pi_ef -> read on f
        ("w", False, False, False, False, True),   # no drives, T1 wait not reset
    )

    def _scans(self):
        """The scans to run, in table order, filtered by cfg.expt.scans."""
        wanted = list(self.cfg.expt.scans)
        # check_f mirrors single_shot's flag and just adds the |f> scan.
        if self.cfg.expt.check_f and "f" not in wanted:
            wanted.append("f")
        known = {s[0] for s in self.SCANS}
        unknown = [label for label in wanted if label not in known]
        if unknown:
            raise ValueError(
                f"Unknown scan label(s) {unknown}; expected any of {sorted(known)}."
            )
        if not {"g", "e"}.issubset(wanted):
            raise ValueError(
                "scans must include 'g' (buffer only) and 'e' (pump + buffer): "
                "the rotation, threshold and fidelity all come from that pair."
            )
        return tuple(s for s in self.SCANS if s[0] in wanted)

    def acquire(self, progress=False):
        """
        Collect shots for each of the sequences in `SCANS`.

        Args:
            progress: Whether to show progress bar

        Returns:
            Data with 'Ig'/'Qg' (no drives), 'Ie'/'Qe' (pump + buffer) and,
            unless disabled, 'Ip'/'Qp' (pump only)
        """
        self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]] = (
            self.cfg.expt.final_delay
        )
        data = {}
        reset_f_requested = self.cfg.expt.reset_f
        base_frequency_p = self.cfg.expt.frequency_p
        base_reset_mode = reset_mode_of(self.cfg.expt)
        for label, do_pump, do_buffer, do_pulse_f, detuned, passive in self._scans():
            self.cfg.expt.do_pump = do_pump
            self.cfg.expt.do_buffer = do_buffer
            self.cfg.expt.do_pulse_f = do_pulse_f
            # Re-read per scan because the Program is rebuilt below, so
            # _initialize/make_pulse pick this up without a second pulse.
            self.cfg.expt.frequency_p = base_frequency_p + (
                self.cfg.expt.detune_p if detuned else 0.0)
            # The reset protocol follows this scan's own preparation. Only
            # the 'f' scan transfers population up to |f>, so only it needs
            # the pi_ef leg; on the others the pi_ef would drive |e> to |f>
            # and cost a round. The extra pulse is ~50 ns against a ~8 us
            # reset round, so the equal-timing invariant across scans still
            # holds to well under a percent.
            self.cfg.expt.reset_f = reset_f_requested and do_pulse_f
            # A passive scan drops the reset and takes the T1 wait instead, so
            # final_delay has to be recomputed with it -- _get_final_delay
            # returns 1 us under any reset and the full wait without one.
            self.cfg.expt.reset_mode = "passive" if passive else base_reset_mode
            final_delay = self._get_final_delay()
            prog = SMPDHistogramProgram(
                soccfg=self.soccfg, final_delay=final_delay, cfg=self.cfg
            )
            prog.acquire(
                self.im[self.cfg.aliases.soc],
                rounds=self.cfg.expt.rounds,
                threshold=None,
                progress=progress,
            )
            # collect_shots returns one entry per readout channel, each shaped
            # (*loop_dims, n_reads) with loop_dims = (reps, shots). Readout 0
            # is the measurement; 1.. are the active-reset rounds.
            i_shots, q_shots = prog.collect_shots()
            data[f"I{label}"] = np.asarray(i_shots[0])[..., 0].flatten()
            data[f"Q{label}"] = np.asarray(q_shots[0])[..., 0].flatten()
            i_rounds = _reset_rounds(i_shots[0])
            if i_rounds is not None:
                data[f"I{label}r"] = i_rounds
                data[f"Q{label}r"] = _reset_rounds(q_shots[0])

        # Restore the full sequence so cfg reflects the headline measurement,
        # and the on-band frequency so the saved yml records that.
        self.cfg.expt.do_pump = True
        self.cfg.expt.do_buffer = True
        self.cfg.expt.do_pulse_f = False
        self.cfg.expt.reset_f = reset_f_requested
        self.cfg.expt.frequency_p = base_frequency_p
        self.cfg.expt.reset_mode = base_reset_mode
        self.data = data
        return self.data

    _SCAN_NAMES = {
        "g": "no drives",
        "e": "pump + buffer",
        "p": "pump only",
        "d": "pump only, off band",
        "b": "buffer only",
        "n": "no drives",
        "f": "pump + buffer, f readout",
        "w": "no drives, T1 wait",
    }

    # Legend text for the histogram panel, naming each sequence by what is off.
    # 'n' is retained for runs taken before no-drives moved into the 'g' slot.
    _SCAN_LABELS = {
        "g": "no drives",
        "e": "buffer + pump",
        "p": "no buffer",
        "d": "no buffer, off band",
        "b": "no pump",
        "n": "no drives",
        "f": "buffer + pump, f readout",
        "w": "no drives, T1 wait",
    }

    # How the scans the shared plotter does not know about are drawn on top.
    # `panels` says where a trace still needs drawing: the shared plotter
    # already puts the |f> histogram on the log panel when If/Qf are present,
    # so f is only added to the linear one.
    _OVERLAY_STYLE = {
        "p": {"panels": ("linear", "log"), "color": "green", "ls": "--"},
        "d": {"panels": ("linear", "log"), "color": "darkorange", "ls": "--"},
        "b": {"panels": ("linear", "log"), "color": "tab:cyan", "ls": "-."},
        "n": {"panels": ("linear", "log"), "color": "0.35", "ls": "-."},
        "f": {"panels": ("linear",), "color": "magenta", "ls": "--"},
        "w": {"panels": ("linear", "log"), "color": "tab:purple", "ls": ":"},
    }


    def analyze(self, data=None, fit=True, span=None, verbose=False, **kwargs):
        """
        Histogram each scan's shots.

        Thresholds and fidelity come from
        `helpers.analyze_single_shot_histograms`, the same routine
        `HistogramExperiment` uses, so the numbers and the plot agree. It is
        called with `rotate=False`: the readout phase is taken as already
        calibrated, so the shots are histogrammed on I exactly as acquired.
        Histograms use the key names that routine plots
        (`vhg`/`histg`/`vhe`/`histe`).

        No model is fit to the distributions.

        Args:
            data: Data to analyze (if None, use self.data)
            fit: Unused, accepted for interface compatibility
            span: Histogram span passed through to the shared helper
            verbose: Print the IQ statistics
            **kwargs: Unused, accepted for interface compatibility

        Returns:
            Data with 'vh<label>' / 'hist<label>' per scan, plus the
            thresholds, fidelities and angle from the shared helper.
        """
        if data is None:
            data = self.data

        # Threshold and fidelity from the g and e scans. rotate=False: the
        # single-shot calibration has already been done and its phase is
        # applied to the readout pulse, so I is the discriminating axis as
        # acquired -- deriving another angle here would double-correct.
        params, _ = helpers.analyze_single_shot_histograms(
            data=data,
            plot=False,
            span=span,
            verbose=verbose,
            qubit=self.cfg.expt.qubit[0],
            rotate=False,
        )
        data.update(params)

        # Follow the data, not cfg, so a reloaded set histograms the scans it
        # actually contains whatever cfg.expt.scans now says.
        for label, *_ in self.SCANS:
            if f"I{label}" not in data:
                continue
            centers, hist = helpers.create_histogram(
                data[f"I{label}"], n_bins=self.cfg.expt.bins
            )
            data[f"vh{label}"] = centers
            data[f"hist{label}"] = hist

        return data

    def display(self, data=None, span=None, verbose=False, ax=None, plot=True, **kwargs):
        """
        Plot with the same routine `HistogramExperiment` uses.

        The helper draws the IQ panels and the raw g/e histograms with the
        threshold and fidelity; the remaining scans are overlaid here, since it
        only knows about g/e (and f).

        Args:
            data: Data to display (if None, use self.data)
            span: Histogram span passed through to the shared helper
            verbose: Print fidelity, threshold and rotation angle
            ax: Axes to plot into, or None to create a new figure
            plot: Whether to draw at all
            **kwargs: Unused, accepted for interface compatibility
        """
        if data is None:
            data = self.data

        save_fig = ax is None
        _, fig = helpers.analyze_single_shot_histograms(
            data=data,
            plot=plot,
            span=span,
            verbose=verbose,
            ax=ax,
            qubit=self.cfg.expt.qubit[0],
            # Must match analyze(), or the plotted axis would not be the one
            # the fits were taken on.
            rotate=False,
        )

        if fig is not None:
            self._overlay_extra_scans(fig, data)
            self._relabel_histogram_legend(fig)
            # Replaces the shared helper's suptitle (Figure.suptitle updates
            # the existing artist rather than adding a second one), so the pump
            # setting is on the saved PNG -- needed when a scan like
            # 24_fwm_histogram_vs_power writes several to one folder.
            fig.suptitle(
                f"Single Shot Histogram Q{self.cfg.expt.qubit[0]} -- pump "
                f"{self.cfg.expt.frequency_p:.3f} MHz, "
                f"gain {self.cfg.expt.gain_p:.4f}"
            )

        if save_fig and fig is not None:
            plt.show()
            self.save_fig(fig)
            self.save_config()

    def _relabel_histogram_legend(self, fig):
        """
        Name the histogram panel's traces by which drives were on.

        The shared plotter labels its two distributions 'g' and 'e', which here
        are really the pump-off and pump-on sequences. Relabelling the artists
        afterwards keeps `single_shot`'s own legends untouched.
        """
        # The overlays already carry their final labels; only the shared
        # helper's own two need renaming.
        for axis in fig.get_axes():
            if axis.get_ylabel() != "Probability":
                continue
            # patches too: the helper draws the raw e histogram with
            # ax.hist(), which labels a Rectangle rather than a line or a
            # collection, so it would otherwise be missed.
            artists = (list(axis.collections) + list(axis.get_lines())
                       + list(axis.patches))
            labels = {artist.get_label() for artist in artists}

            if "f" in labels:
                # With If/Qf present the helper labels its *excited* fill "f"
                # and additionally draws the raw e histogram as "e". That is
                # right for single_shot, where the fitted pair is g-f, but here
                # both of those curves are the pump+buffer data -- so name the
                # fill and keep the duplicate out of the legend.
                rename = {
                    "g": self._SCAN_LABELS["g"],
                    "f": self._SCAN_LABELS["e"],
                    "e": "_nolegend_",
                }
            else:
                rename = {
                    "g": self._SCAN_LABELS["g"],
                    "e": self._SCAN_LABELS["e"],
                }

            renamed = False
            for artist in artists:
                new_label = rename.get(artist.get_label())
                if new_label is not None:
                    artist.set_label(new_label)
                    renamed = True
            if renamed:
                axis.legend(fontsize="small", loc="upper right")

    def _overlay_extra_scans(self, fig, data):
        """
        Add the reference scans the shared plotter does not know about.

        That helper only handles g/e -- its third slot is the f state, and it
        would rotate on g-f, which is wrong here -- so the pump-only and
        no-drives traces are drawn on afterwards. Panels are found by their
        labels rather than by index, and anything not found is skipped.
        """
        extra = [
            label
            for label in self._OVERLAY_STYLE
            if f"hist{label}" in data
        ]
        if not extra:
            return

        for axis in fig.get_axes():
            linear = axis.get_ylabel() == "Probability"
            log = axis.get_title() == "Log Histogram"
            if not (linear or log):
                continue
            plot_fn = axis.plot if linear else axis.semilogy
            panel = "linear" if linear else "log"
            lo, hi = axis.get_xlim()
            for label in extra:
                style = dict(self._OVERLAY_STYLE[label])
                if panel not in style.pop("panels"):
                    continue
                centers, hist = data[f"vh{label}"], data[f"hist{label}"]
                # Dashed and on top: these traces largely coincide with the
                # buffer-only one, so solid lines would disappear under it.
                plot_fn(
                    centers,
                    hist,
                    lw=1.6,
                    zorder=10,
                    label=self._SCAN_LABELS[label],
                    **style,
                )
                # The helper sized the axes for g/e only; widen if a trace runs
                # past them so none of it is hidden.
                lo = min(lo, float(np.min(centers)))
                hi = max(hi, float(np.max(centers)))
            axis.set_xlim(lo, hi)
            # The legend is built once by _relabel_histogram_legend, after this.
