"""
QICK Program Module

This module provides base classes for quantum experiments using the QICK framework.
It extends the AveragerProgramV2 class with additional functionality for:
- Configuring readout and control channels
- Creating and managing pulse sequences
- Performing measurements
- Implementing active qubit reset

The module contains two main classes:
- QickProgram: For single-qubit experiments
- QickProgram2Q: For two-qubit experiments
"""

from qick.asm_v2 import AveragerProgramV2
from ...exp_handling.datamanagement import AttrDict
from ...helpers.pulse_shapes import (  # noqa: F401  (re-exported)
    PULSE_TYPES,
    pulse_type_from_config,
    ramp_length,
)
from qick import *


def mixer_freq_for(frequency, offset=400):
    """
    Fixed LO frequency for an interpolated ("int") generator.

    The mixer is hardware and is set once at declaration, so it cannot follow a
    sweep the way the DDS can. When `frequency` is swept -- as it is in
    resonator spectroscopy -- collapse it to the centre of the sweep, which
    leaves the scan sitting symmetrically around the LO. A plain number passes
    through unchanged.

    Only generators with a mixer need this; "full" generators ignore it, which
    is why this path only shows up on boards where the readout DAC is an int
    generator.
    """
    if getattr(frequency, "is_sweep", None) is not None and frequency.is_sweep():
        frequency = (frequency.minval() + frequency.maxval()) / 2
    return float(frequency) - offset


RESET_MODES = ("passive", "active", "f0g1", "while")



def reset_mode_of(expt_cfg):
    """
    Which reset a sequence uses: "passive", "active" or "f0g1".

    - "passive": no reset pulses; the inter-shot tail is the full
      `device.readout.final_delay` T1 wait.
    - "active": the measurement-based conditional reset, `QickProgram.reset`.
    - "while": `QickProgram.reset_while`, which loops until the readout says
      ground. Needs a second logical readout on the same input; see CLAUDE.md.
    - "f0g1": the unconditional all-microwave reset, `QickProgram.reset_f0g1`.

    `cfg.expt.reset_mode` is authoritative. `active_reset` is the legacy
    spelling and is honoured when `reset_mode` is absent, so the experiments
    that have not been migrated keep working untouched. Reading the mode
    through this helper rather than the raw key is what makes a missed
    migration site keep resetting as it did before, instead of silently not
    resetting at all -- which would quietly corrupt data rather than raise.

    Args:
        expt_cfg: a cfg.expt mapping

    Returns:
        str: one of RESET_MODES

    Raises:
        ValueError: on an unrecognised reset_mode, rather than falling through
            to "passive" and running with no reset.
    """
    mode = expt_cfg.get("reset_mode", None)
    legacy = expt_cfg.get("active_reset", None)

    if mode is None:
        return "active" if legacy else "passive"
    if mode not in RESET_MODES:
        raise ValueError(
            f"reset_mode must be one of {list(RESET_MODES)}, got {mode!r}"
        )

    # Both keys present and disagreeing is the one failure the staged
    # migration can produce: a script still passing active_reset=True to a
    # class whose params_def now defaults reset_mode to "passive" would have
    # its reset silently turned off. Raise instead -- a stopped run is far
    # cheaper than a run whose qubit was never reset.
    if legacy is not None and bool(legacy) != (mode == "active"):
        raise ValueError(
            f"reset_mode={mode!r} and active_reset={legacy!r} disagree. "
            "active_reset is the legacy spelling of reset_mode='active'; this "
            "experiment has been migrated, so drop active_reset and set "
            "reset_mode instead."
        )
    return mode


def reset_plays_pi_pulses(expt_cfg):
    """
    Whether the selected reset plays `pi_ge` (and `pi_ef` under `reset_f`).

    True for the measurement-based resets -- "active" and "while" -- which
    both branch on a readout and then drive the qubit down with the qubit
    pulses. False for "passive" (no pulses) and "f0g1" (its own two tones).

    A Program's `_initialize` must declare those pulses whenever this is
    true, or `reset()`/`reset_while()` plays a name that was never defined
    and the run dies at compile time, partway into the sweep. Testing the
    mode for equality against "active" is what let `reset_mode = "while"`
    through; `check_pi_ge.py` asserts the guards go through this helper.
    """
    return reset_mode_of(expt_cfg) in ("active", "while")


def declare_f0g1_gen(prog, cfg):
    """
    Declare the f0-g1 drive generator and return its channel.

    None of the existing declaration idioms in the tree fit this line:

    - `self.pump_ch` / `hw.soc.dacs.pump` is already the FWM pump.
    - `QubitStarkSpecProgram` omits `nqz` and derives the mixer as
      `freq - 400`, which for a few-hundred-MHz carrier is negative.

    So the nyquist zone and mixer frequency both come from the config. For a
    ~225 MHz carrier `mixer_freq: 0` is the right choice: the int4 DDS range is
    +/- f_dds/2 (~860 MHz), so the DDS reaches the carrier unaided and the
    NCO's leakage stays at DC rather than a few tens of MHz from the tone.

    A free function, not a method, because `QickSMPDProgram._initialize` does
    not chain to `QickProgram._initialize` -- so a program that needs this
    outside the normal path can still call it directly.
    """
    q = cfg.expt.qubit[0]
    if "f0g1" not in cfg.hw.soc.dacs:
        raise ValueError(
            "hw.soc.dacs.f0g1 is missing from this config, so there is no "
            "f0-g1 reset drive line. Add the section (ch, type, nyquist, "
            "mixer_freq, filter_*, attn*) mirroring hw.soc.dacs.pump, or use "
            "reset_mode 'active' or 'passive'."
        )
    ch = cfg.hw.soc.dacs.f0g1.ch[q]
    args = {"ch": ch, "nqz": cfg.hw.soc.dacs.f0g1.nyquist[q]}
    if prog.soccfg["gens"][ch].get("has_mixer", False):
        mixer = cfg.hw.soc.dacs.f0g1.get("mixer_freq")
        if mixer is None:
            raise ValueError(
                f"f0g1 DAC {ch} has a digital mixer, so "
                "hw.soc.dacs.f0g1.mixer_freq must be set."
            )
        args["mixer_freq"] = mixer[q]
    prog.declare_gen(**args)
    return ch


class QickProgram(AveragerProgramV2):
    """
    Base class for single-qubit quantum experiments using the QICK framework.

    This class extends AveragerProgramV2 to provide a higher-level interface for
    creating and running quantum experiments. It handles channel configuration,
    pulse generation, measurement, and data collection for a single qubit.

    The class is designed to be extended by specific experiment implementations
    that override the _body method to define the experiment sequence.
    """

    # Set True by programs that sweep the readout frequency, so they take
    # adcs.readout.swept_ch instead of .ch. A static PFB channel cannot
    # sweep, so the two arms are separate config entries.
    use_swept_readout = False

    def __init__(self, soccfg, final_delay=50,  cfg=None,final_wait=0,):

        """
        Initialize the QickProgram with hardware configuration and experiment parameters.

        Args:
            soccfg: System-on-chip configuration object containing hardware details
            final_delay: Delay time (in ns) after each experiment repetition
            cfg: Configuration dictionary containing experiment parameters
        """
        if cfg is None:
            cfg = {}
        self.cfg = AttrDict(cfg)  # Convert to attribute dictionary for easier access

        # Update configuration with experiment-specific parameters
        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg.expt.reps, final_delay, final_wait=final_wait, cfg=cfg)

    def save_params(self, readout='standard'):
        """Cache qubit/readout channel numbers, types, and readout parameters as attributes."""
        cfg = AttrDict(self.cfg)

        # Get qubit index from configuration
        self.qubits = cfg.expt.qubit
        q = self.qubits[0]  # Single qubit index

        # Configure hardware channels for the selected qubit
        # Programs that sweep the readout frequency need the dynamic readout:
        # the default channel may be a static PFB index. See CLAUDE.md "Swept
        # vs full-signal readout arms".
        ro = cfg.hw.soc.adcs.readout
        if self.use_swept_readout and "swept_ch" in ro:
            self.adc_ch = ro.swept_ch[q]
        else:
            self.adc_ch = ro.ch[q]  # ADC channel for readout
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q]  # DAC channel for resonator drive
        self.res_ch_type = cfg.hw.soc.dacs.readout.type[
            q
        ]  # Resonator channel type (full, mux, int)
        self.res_nqz = cfg.hw.soc.dacs.readout.nyquist[
            q
        ]  # Nyquist zone for resonator (1 for <5 GHz, 2 for >5 GHz)
        self.type = cfg.hw.soc.dacs.readout.type[q]  # Type of readout channel
        if self.use_swept_readout and "swept_type" in ro:
            self.adc_type = ro.swept_type[q]
        else:
            self.adc_type = ro.type[q]  # ADC channel type
        # Readout pulse shape for full / int gens; mux and "long" are always const.
        self.readout_pulse_type = pulse_type_from_config(cfg.device.readout, "pulse_type", q)

        self.trig_offset = cfg.device.readout.trig_offset[q]  # Trigger timing offset
        if "qubit" in cfg.hw.soc.dacs:
            self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[q]  # DAC channel for qubit drive
            self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type[q]  # Qubit channel type
            self.qubit_nqz = cfg.hw.soc.dacs.qubit.nyquist[q]  # Nyquist zone for qubit

        # Optional dedicated four-wave-mixing pump line. When hw.soc.dacs.pump
        # is absent the pump falls back to the qubit channel, so every existing
        # config keeps behaving exactly as before.
        if "pump" in cfg.hw.soc.dacs:
            self.pump_ch = cfg.hw.soc.dacs.pump.ch[q]
            self.pump_ch_type = cfg.hw.soc.dacs.pump.type[q]
            self.pump_nqz = cfg.hw.soc.dacs.pump.nyquist[q]
        elif "qubit" in cfg.hw.soc.dacs:
            self.pump_ch = self.qubit_ch
            self.pump_ch_type = self.qubit_ch_type
            self.pump_nqz = self.qubit_nqz
        else:
            self.pump_ch = None

        # Optional f0-g1 reset drive line. Resolved here, beside the pump, so
        # it exists before any _initialize override needs it. None when the
        # config has no such line, which is what declare_f0g1_gen reports on.
        if "f0g1" in cfg.hw.soc.dacs:
            self.f0g1_ch = cfg.hw.soc.dacs.f0g1.ch[q]
            self.f0g1_nqz = cfg.hw.soc.dacs.f0g1.nyquist[q]
        else:
            self.f0g1_ch = None
            self.f0g1_nqz = None

        # Configure standard readout parameters if specified
        if readout == "standard":
            self.readout_length = cfg.device.readout.readout_length[
                q
            ]  # Readout pulse length
            self.frequency = cfg.device.readout.frequency[q]  # Readout frequency
            self.gain = cfg.device.readout.gain[q]  # Readout amplitude
            self.phase = cfg.device.readout.phase[q]  # Readout phase

        # Configure local oscillator (LO) if available
        if (
            "lo" in cfg.hw.soc
            and "ch" in cfg.hw.soc.lo
            and cfg.hw.soc.lo.ch[q] != "None"
        ):
            # Set up LO channel parameters
            self.lo_ch = cfg.hw.soc.lo.ch[q]
            self.lo_nqz = cfg.hw.soc.lo.nyquist[q]
            self.mixer_freq = cfg.hw.soc.lo.mixer_freq[q]
            self.lo_gain = cfg.hw.soc.lo.gain[q]
        else:
            self.lo_ch = None  # No LO channel available

    def _initialize(self, cfg, readout="standard"):
        """
        Initialize hardware channels and configure pulses for the experiment.

        This method sets up the ADC (analog-to-digital converter) and DAC
        (digital-to-analog converter) channels for qubit control and readout.
        It also configures the readout pulse and qubit control channels.

        Args:
            cfg: Configuration dictionary
            readout: Readout configuration type (default: "standard")
        """
        self.save_params(readout)  # Save parameters for the experiment
        cfg = AttrDict(self.cfg)

        q = self.qubits[0]  # Single qubit index

        # Configure local oscillator (LO) if available
        if ("lo" in cfg.hw.soc and "ch" in cfg.hw.soc.lo and cfg.hw.soc.lo.ch[q] != "None"):
            
            # Declare LO signal generator with offset from mixer frequency, this assumes it is using int generator
            self.declare_gen(ch=self.lo_ch, nqz=self.lo_nqz, mixer_freq=self.mixer_freq - 500)

            # Create LO pulse for readout
            self.add_pulse(
                self.lo_ch,
                name="mix_pulse",
                style="const",  # Constant amplitude pulse
                length=self.readout_length,
                freq=self.mixer_freq,
                phase=0,
                gain=self.lo_gain,
            )


        if "aves" in cfg.expt: # Used to make a 2D scan from a 1D scan
            self.add_loop("ave_loop", cfg.expt.aves)

        # Set up readout generator
        if self.type == "full":

            self.declare_gen(
                ch=self.res_ch, nqz=self.res_nqz
            )  # Declare resonator signal generator

            # Create readout pulse
            pulse_args = {
                "ch": self.res_ch,
                "name": "readout_pulse",
                "style": "const",
                "ro_ch": self.adc_ch,
                "freq": self.frequency,
                "phase": self.phase,
                "gain": self.gain,
            }
            if readout == "long":
                # qick forces flat_top to oneshot, so the periodic tone stays const
                pulse_args["length"] = 1
                pulse_args["mode"] = "periodic"
            else:
                pulse_args["length"] = self.readout_length
                if self.readout_pulse_type == "flat_top":
                    pulse_args["style"] = "flat_top"
                    pulse_args["envelope"] = self.add_readout_ramp(self.res_ch)
            self.add_pulse(**pulse_args)
        elif self.type == "mux":
            self.declare_gen(
                ch=self.res_ch,
                nqz=self.res_nqz,
                ro_ch=self.adc_ch,
                mux_freqs=[self.frequency],
                mux_gains=[self.gain],
                mux_phases=[self.phase],
            )

            self.add_pulse(
                ch=self.res_ch,
                name="readout_pulse",
                style="const",
                length=self.readout_length,
                mask=[0],
            )
        elif self.type == "int":
            self.declare_gen(
                ch=self.res_ch,
                nqz=self.res_nqz,
                ro_ch=self.adc_ch,
                mixer_freq=mixer_freq_for(self.frequency),
            )

            shape = (
                {"style": "flat_top", "envelope": self.add_readout_ramp(self.res_ch)}
                if self.readout_pulse_type == "flat_top"
                else {"style": "const"}
            )
            self.add_pulse(
                self.res_ch,
                ro_ch=self.adc_ch,
                name="readout_pulse",
                length=self.readout_length,
                freq=self.frequency,
                phase=self.phase,
                gain=self.gain,
                **shape,
            )
        
        # Second logical readout on the same physical input, for the while
        # loop's variable-count feedback. The measurement channel keeps one
        # trigger per rep and so still averages correctly. Indices 8-9 have no
        # tProc input and cannot be used. See CLAUDE.md "While-loop reset".
        if reset_mode_of(cfg.expt) == "while":
            if "feedback_ch" not in cfg.hw.soc.adcs.readout:
                raise RuntimeError(
                    "reset_mode 'while' needs hw.soc.adcs.readout.feedback_ch: "
                    "a second logical readout on the same physical input, so "
                    "the loop's variable trigger count does not misalign the "
                    "measurement buffer. Run 00_soc_map.py to find one."
                )
            self.fb_ch = cfg.hw.soc.adcs.readout.feedback_ch[q]
        else:
            self.fb_ch = None

        # Configure readout settings
        if self.adc_type == "dyn":
            self.declare_readout(self.adc_ch, length=self.readout_length+self.trig_offset-0.1)  # Configure ADC for readout
            self.add_readoutconfig(ch=self.adc_ch, name="readout", freq=self.frequency, gen_ch=self.res_ch)
        elif self.adc_type == "std":
            # phase=0: readout_pulse above already carries
            # device.readout.phase, and the dyn path sets no demodulator
            # phase at all. Passing it here too rotates by 2*phase, which
            # leaves the hardware read_and_jump branching on a badly
            # misaligned axis while the software analysis still re-rotates
            # and looks fine. See CLAUDE.md "Readout phase on the static arm".
            self.declare_readout(
                ch=self.adc_ch,
                length=self.readout_length,
                freq=self.frequency,
                phase=0,
                gen_ch=self.res_ch,
            )

        if self.fb_ch is not None:
            # phase=0 for the same reason as above: the pulse carries it.
            self.declare_readout(ch=self.fb_ch, length=self.readout_length,
                                 freq=self.frequency, phase=0,
                                 gen_ch=self.res_ch)

        if "qubit" in cfg.hw.soc.dacs:
            # Set up qubit control channel
            self.declare_gen(
                ch=self.qubit_ch, nqz=self.qubit_nqz
            )  # Declare qubit signal generator

        # A dedicated pump line needs its own generator; when it shares the
        # qubit channel the declaration above already covers it. An
        # interpolated generator has a mixer, which must be given a frequency
        # at declaration -- taken from hw.soc.dacs.pump.mixer_freq, the same
        # way the JPA line carries its own mixing_freq.
        if self.pump_ch is not None and self.pump_ch != getattr(self, "qubit_ch", None):
            pump_args = {"ch": self.pump_ch, "nqz": self.pump_nqz}
            if self.soccfg["gens"][self.pump_ch].get("has_mixer", False):
                pump_mixer = cfg.hw.soc.dacs.pump.get("mixer_freq")
                if pump_mixer is None:
                    raise ValueError(
                        f"Pump DAC {self.pump_ch} has a mixer, so "
                        "hw.soc.dacs.pump.mixer_freq must be set."
                    )
                pump_args["mixer_freq"] = pump_mixer[self.qubits[0]]
            self.declare_gen(**pump_args)

        # The unconditional f0-g1 reset's two tones. Declared here so every
        # program that chains to this _initialize can select reset_mode
        # "f0g1" without knowing anything about it. Names are distinct from
        # F0G1Program's own f0g1_pulse/ef_pulse so the calibration
        # experiments can themselves run with this reset on.
        if reset_mode_of(cfg.expt) == "f0g1":
            self.f0g1_ch = declare_f0g1_gen(self, cfg)
            ramp = {
                "ramp_sigma": cfg.expt.f0g1_reset_ramp_sigma,
                "ramp_sigma_inc": cfg.expt.f0g1_reset_ramp_sigma_inc,
            }
            self.make_pulse(
                {
                    "chan": self.f0g1_ch,
                    "freq": cfg.expt.f0g1_reset_freq,
                    "gain": cfg.expt.f0g1_reset_gain,
                    "phase": 0,
                    "type": cfg.expt.f0g1_reset_pulse_type,
                    "length": cfg.expt.f0g1_reset_length,
                    **ramp,
                },
                "f0g1_reset_pulse",
            )
            # Same length and ramps as the f0-g1 tone, so the two cover
            # exactly the same window when both start at t=0.
            self.make_pulse(
                {
                    "chan": self.qubit_ch,
                    "freq": cfg.expt.f0g1_reset_freq_ef,
                    "gain": cfg.expt.f0g1_reset_gain_ef,
                    "phase": 0,
                    "type": cfg.expt.f0g1_reset_pulse_type_ef,
                    "length": cfg.expt.f0g1_reset_length,
                    **ramp,
                },
                "ef_reset_pulse",
            )

    def _body(self, cfg):
        """
        Default experiment sequence implementation.

        This method defines the basic pulse sequence for the experiment.
        It should be overridden by subclasses to implement specific experiments.

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)
        # Send readout configuration to hardware
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        # Apply readout pulse
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        # Trigger data acquisition with specified timing offset
        self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset, ddr4=True)

    def measure(self, cfg):
        """
        Perform qubit measurement.

        This method implements the standard measurement sequence:
        1. Apply readout pulse to the resonator
        2. Apply LO pulse if available
        3. Trigger data acquisition
        4. Perform active reset if enabled

        Args:
            cfg: Configuration dictionary
        """
        cfg = AttrDict(self.cfg)

        # Apply readout pulse to resonator
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        # Apply LO pulse if using QICK for it
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
        # Trigger data acquisition
        self.trigger(
            ros=[self.adc_ch],
            pins=[0],
            t=self.trig_offset,
        )
        # Reset between shots, if any. See reset_mode_of for the three modes.
        mode = reset_mode_of(cfg.expt)
        if mode == "active":
            self.reset(cfg.expt.reset)
        elif mode == "f0g1":
            self.reset_f0g1()
        elif mode == "while":
            self.reset_while(cfg.expt.reset)

    def add_readout_ramp(self, ch):
        """
        Add the flat_top readout's edge envelope on `ch` and return its name.

        Same ramp as make_pulse's flat_top default (0.02 us sigma, 3 sigma
        long). The flat part of the readout is `readout_length`, so the pulse
        runs 0.06 us longer than that. A flat_top plateau is half the
        amplitude of a const pulse at the same gain.
        """
        self.add_gauss(ch=ch, name="readout_ramp", sigma=0.02, length=0.06, even_length=True)
        return "readout_ramp"

    def make_pulse(self, pulse, name):
        """
        Create a pulse with specified parameters.

        This method creates different types of pulses (Gaussian, flat-top, or constant)
        based on the pulse parameters provided. The pulse is then added to the program
        for later execution.

        Args:
            pulse: Dictionary containing pulse parameters
            name: Name to assign to the created pulse

        Supported pulse types:
            - "gauss": Gaussian pulse with specified sigma
            - "flat_top": Flat-top pulse with Gaussian rise/fall
            - Other (default): Constant amplitude pulse
        """
        pulse = AttrDict(pulse)  # Convert to attribute dictionary

        # Common pulse parameters
        if "chan" not in pulse:
            pulse.chan = self.qubit_ch
        pulse_args = {
            "ch": pulse.chan,
            "name": name,
            "freq": pulse.freq,  # Pulse frequency
            "phase": pulse.phase,  # Pulse phase
            "gain": pulse.gain,  # Pulse amplitude
        }
        # print('pulse: {}'.format(pulse))
        # print('pulse_args: {}'.format(pulse_args))

        # Create different pulse types based on pulse.type
        if pulse.type == "gauss":
            # Gaussian pulse, with sigma = sigma, total length = sigma * sigma_inc or length
            style = "arb"  # Arbitrary waveform

            # Determine pulse length
            if "length" in pulse:
                length = pulse.length
            else:
                length = pulse.sigma * pulse.sigma_inc  # Calculate from sigma

            # Create Gaussian envelope
            self.add_gauss(
                ch=pulse.chan,
                name="ramp",
                sigma=pulse.sigma,  # Width of Gaussian
                length=length,
                even_length=True,
            )
            pulse_args["envelope"] = "ramp"  # Use Gaussian envelope

        elif pulse.type == "flat_top":
            # Flat-top pulse with Gaussian rise/fall

            # Determine pulse length
            if "length" in pulse:
                length = pulse.length
            else:
                length = pulse.sigma

            style = "flat_top"

            # Create Gaussian ramp for rise/fall
            if "ramp_sigma" not in pulse:
                pulse.ramp_sigma = 0.02
            if "ramp_sigma_inc" not in pulse:
                pulse.ramp_sigma_inc = 3
            ramp_length = pulse.ramp_sigma * pulse.ramp_sigma_inc

            self.add_gauss(
                ch=pulse.chan,
                name="ramp",
                sigma=pulse.ramp_sigma,  # Width of rise/fall
                length=ramp_length,  # Length of rise/fall
                even_length=True,
            )
            pulse_args["envelope"] = "ramp"  # Use Gaussian envelope for edges
            pulse_args["length"] = length  # Total pulse length (this is of the flat part?)

        else:
            # Default: Constant amplitude pulse

            # Determine pulse length
            if "length" in pulse:
                length = pulse.length
            else:
                length = pulse.sigma

            style = "const"  # Constant amplitude
            pulse_args["length"] = length

        # Set pulse style and add to program
        pulse_args["style"] = style
        self.add_pulse(**pulse_args)

    def make_cfg_pulse(self, q, freq, name):
        """
        Create a pulse for the specified qubit based on definition in cfg file; usually this will be a pi pulse. 

        A π pulse rotates the qubit state by 180 degrees around the X-axis,
        flipping the state between |0⟩ and |1⟩.

        Args:
            q: Qubit index
            freq: Dictionary of frequencies for each qubit
            name: Name of the pulse to create

        Returns:
            Pulse dictionary with the created pulse parameters
        """
        cfg = AttrDict(self.cfg)
        # Get pulse parameters from configuration for the specified qubit
        pulse = {key: value[q] for key, value in cfg.device.qubit.pulses[name].items()}
        # Set pulse frequency and phase
        pulse["freq"] = freq[q]
        if "phase" not in pulse:
            pulse["phase"] = 0  # Zero phase for X-rotation
        # Create the pulse using the make_pulse method
        self.make_pulse(pulse, name)
        return pulse

    def collect_shots(self, offset=0):
        """
        Collect raw measurement data from the ADC.

        This method retrieves the raw I/Q data from the ADC, normalizes it,
        and applies the specified offset correction.

        Args:
            offset: DC offset to subtract from the raw data

        Returns:
            Tuple of (i_shots, q_shots) containing the I and Q quadrature data
        """
        # Should we get multiple offsets for multiple channels?
        iq_raw = self.get_raw()  # Get raw ADC data for all channels
        i_shots_list = []
        q_shots_list = []
        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            nsamp = rocfg["length"]  # Number of samples

            # get_raw() returns (*loop_dims, n_reads, 2) per channel, where
            # loop_dims = (reps, *sweep dims). Index the trailing I/Q axis with
            # an ellipsis so this works for any number of sweep loops -- a
            # fixed [:, :, :, 0] silently keeps the I/Q axis once a program has
            # more than one sweep loop.
            # Extract and normalize I quadrature data for this channel
            i_shots_vec = iq_raw[i][..., 0] / nsamp - offset
            # Extract and normalize Q quadrature data for this channel
            q_shots_vec = iq_raw[i][..., 1] / nsamp - offset

            i_shots_list.append(i_shots_vec)
            q_shots_list.append(q_shots_vec)

        return i_shots_list, q_shots_list

    def reset(self, i):
        """
        Perform active qubit reset.

        This method implements a measurement-based reset protocol that:
        1. Reads the most recent readout result
        2. Applies a π pulse only if the qubit reads as |e>
        3. Repeats the process i times to improve reset fidelity

        So `i` rounds apply between 0 and `i` π pulses, and add `i - 1` extra
        readouts to the shot (the last round is not re-measured). That is what
        `collect_shots` sees on the trailing readout axis. With
        `cfg.expt.reset_verify` the last round is re-measured too, giving `i`
        extra readouts and making the final one the true post-reset state.

        `cfg.expt.reset_f` adds a pi_ef ahead of each pi_ge, for sequences
        that can leave population in |f>.

        `cfg.expt.reset_latch` gives "while excited" semantics at a fixed
        trip count: the loop still runs `i` rounds and measures every one, but
        once any round reads ground a latch register suppresses every later pi
        pulse. A real while loop cannot be used here -- see the comment in the
        loop body -- and the latch removes the same dominant error, a ground
        shot misread as excited on the final round.

        Args:
            i: Number of reset iterations to perform
        """
        cfg = AttrDict(self.cfg)

        if cfg.expt.reset_latch:
            # A data register, allocated rather than hardcoded: loops consume
            # data registers too, and add_reg picks a free address. Cleared
            # here so the latch is per-shot, not per-program.
            self.reset_latch = self.add_reg(name="reset_latch", allow_reuse=True)
            self.write_reg(dst=self.reset_latch, src=0)

        # Perform reset sequence i times
        for n in range(i):
            # Wait for readout to complete
            self.wait_auto(cfg.expt.read_wait)
            # Add extra delay for stability
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)

            # Read the qubit state and conditionally apply a π pulse.
            # read_and_jump jumps to the label when the test passes, so
            # test="<" jumps over the π pulse when I < threshold and the pulse
            # fires when I >= threshold. That only resets if
            # g_mean < threshold < e_mean, the ordering 12_single_shot.py
            # establishes via device.readout.phase and device.readout.threshold.
            if cfg.expt.reset_latch:
                # "while excited" semantics with a compile-time-fixed trip
                # count. A real while loop would trigger this readout a
                # data-dependent number of times per rep, and qick sizes the
                # buffer read by *counting trigger instructions*: a variable
                # count silently pulls the wrong IQ values out, with no error
                # (confirmed by the qick authors). The usual fix is a second
                # logical readout on the same physical input -- discard the
                # feedback channel, keep one trigger on the readout channel --
                # but our waste line sits on a dynamic readout (ADC6, index 1)
                # that is 1:1 with its input, so there is no second index to
                # spend. The PFB bank on ADC5 has 8 indices on one input and
                # would give it, once that signal is routed there.
                #
                # So: run a fixed `i` rounds, but latch. Once any round reads
                # ground, the latch suppresses every later pi pulse, which is
                # the part that matters -- a ground shot misread as excited on
                # the *last* round is what sets the floor, and this removes it.
                self.read_input(ro_ch=self.adc_ch)
                # Already confirmed ground on an earlier round: play nothing.
                # arg1 alone with no op tests the register itself, so this is
                # "latch != 0". It does not touch s_port_l, so the threshold
                # comparison below still sees the value just read.
                self.cond_jump(label=f"NOPULSE{n}", arg1=self.reset_latch,
                               test="NZ")
                # I < threshold: ground. Latch it, then play nothing.
                self.cond_jump(label=f"GROUND{n}", arg1="s_port_l", op="-",
                               arg2=cfg.expt.threshold, test="S")
            else:
                self.read_and_jump(
                    ro_ch=self.adc_ch,
                    component="I",  # Use I quadrature for state discrimination
                    threshold=cfg.expt.threshold,  # Threshold for state discrimination
                    test="<",  # Jump (skip the π pulse) when I < threshold, i.e. |g>
                    label=f"NOPULSE{n}",
                )

            # Everything between the jump above and NOPULSE{n} below runs
            # only when I >= threshold, i.e. not ground. A ground shot jumps
            # past the whole block and plays nothing.
            #
            # `if cfg.expt.reset_f` is a compile-time switch -- whether the
            # pi_ef is emitted into the program at all -- not a second
            # runtime branch. There is no branch between |e> and |f| because
            # the readout cannot support one: chi is large compared to kappa,
            # so the resonator response saturates and the two blobs land on
            # top of each other (0.46 sigma apart on run 33 Waymar). Since
            # the only goal is |g>, a not-ground shot is assumed to be |f>
            # and walked down f -> e -> g. Self-correcting across rounds:
            #     |f> -> pi_ef -> |e> -> pi_ge -> |g>     (one round)
            #     |e> -> pi_ef -> |f> -> pi_ge -> |f>     (fixed next round)
            # so |f> clears immediately and |e> takes two, which is why
            # reset_f wants `reset` >= 2.
            if cfg.expt.reset_f:
                self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
                self.delay_auto(0.01)
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            # Small delay for pulse completion
            self.delay_auto(0.01)
            if cfg.expt.reset_latch:
                # Skip past the latch-setting block on the pulsed path.
                self.jump(f"NOPULSE{n}")
                self.label(f"GROUND{n}")
                self.write_reg(dst=self.reset_latch, src=1)
            # Jump target for the ground case: skips both pulses above.
            self.label(f"NOPULSE{n}")

            # Re-measure so the next round has something to branch on. The
            # last round is skipped in normal use: nothing acts on a readout
            # taken after the final π pulse, so it would only cost time.
            # `reset_verify` adds it back, which is what a *calibration* needs
            # -- without it the last round's conditional pulse is unmeasured
            # and the recorded end state is one round stale. 12b_active_reset
            # sets it; production runs should not.
            if n < i - 1 or cfg.expt.reset_verify or cfg.expt.reset_latch:
                
                # Apply readout pulse
                self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
                # Trigger readout
                self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
                # Apply LO pulse if available
                if self.lo_ch is not None:
                    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
                # Small delay before next iteration
                # self.delay_auto(0.01)
            # else:
            #     self.delay_auto(0.01)  # Final small delay after last iteration

    def reset_while(self, max_rounds):
        """
        Reset by looping until the readout says ground.

        Unlike `reset`, which runs a fixed number of rounds, this exits as soon
        as a read comes back below threshold. The loop triggers `self.fb_ch`,
        a second logical readout on the same physical input, so its variable
        trigger count lands in a buffer nothing reads; the measurement on
        `self.adc_ch` keeps exactly one trigger per rep and still averages
        correctly. qick sizes buffer reads by counting trigger instructions, so
        a variable count on the measurement channel would silently misalign it.

        Validated on run 34 #662 and #665 (feedback on index 2 and 3): |e>
        needs 1.01 pulses on average, |g> 0.01, and both end at the ground
        mean.

        Args:
            max_rounds: hard cap. Without it a qubit that never reads ground
                spins the tProc forever and the board needs a reset.
        """
        cfg = AttrDict(self.cfg)
        count = self.add_reg(name="while_rounds", allow_reuse=True)
        self.write_reg(dst=count, src=0)

        self.label("RESET_WHILE")
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        self.trigger(ros=[self.fb_ch], pins=[0], t=self.trig_offset)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.wait_auto(cfg.expt.read_wait)
        self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)
        # The loop runs for an unknown time, so the reference time has to be
        # pushed past the execution time or later timed instructions pile up.
        self.resync(0.05)

        self.read_and_jump(ro_ch=self.fb_ch, component="I",
                           threshold=cfg.expt.threshold, test="<",
                           label="RESET_WHILE_DONE")
        if cfg.expt.reset_f:
            self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
            self.delay_auto(0.01)
        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        self.delay_auto(0.01)
        self.inc_reg(dst=count, src=1)
        self.cond_jump(label="RESET_WHILE_DONE", arg1=count, op="-",
                       arg2=max_rounds, test="NS")
        self.jump("RESET_WHILE")
        self.label("RESET_WHILE_DONE")
        self.resync(0.05)

    def reset_f0g1(self):
        """
        Unconditional f0-g1 reset: drive |f,0> -> |g,1> and let the photon out.

        The all-microwave reset of Magnard et al., arXiv:1801.07689. Both tones
        run for the same fixed duration; nothing is measured and nothing
        branches. Three consequences worth knowing:

        - **No extra readouts.** `reset()` appends `reset - 1` conditional
          readouts to every shot, so `n_reads` grows with the round count.
          This appends none, leaving `n_reads = 1` -- so everything downstream
          that indexes the trailing read axis is unaffected.
        - **It clears |f> natively.** `reset()` cannot: chi >> kappa puts the
          e and f blobs ~0.5 sigma apart, so no threshold separates them, which
          is why `reset_f` has to walk |f> -> |e> -> |g> over two rounds.
        - **Fixed duration**, so sequences that depend on every variant taking
          the same time (the FWM scan table) stay balanced automatically.

        Reads `f0g1_reset_gap`, `read_wait` and `extra_delay` from `cfg.expt`;
        the pulses are declared in `_initialize` from the other
        `f0g1_reset_*` keys, which `QickExperiment.configure_reset` derives
        from `device.f0g1`.
        """
        cfg = AttrDict(self.cfg)

        # The same ring-down allowance reset() uses, so the readout pulse is
        # over and the resonator has emptied before the drive starts -- stray
        # readout photons are exactly what detunes the f0-g1 transition.
        # wait_auto is deliberately absent: reset() needs it to block on the
        # readout value it branches on, and nothing here reads anything.
        self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)

        # Both at t=0 with equal length and ramps, so they cover the same
        # window. This is the ef_align="start" case of F0G1Program.
        self.pulse(ch=self.f0g1_ch, name="f0g1_reset_pulse", t=0)
        self.pulse(ch=self.qubit_ch, name="ef_reset_pulse", t=0)

        # Let the emitted photon leave before the next shot reads out.
        self.delay_auto(t=cfg.expt.f0g1_reset_gap)

    def cond_reset(self, i):
        """
        Perform active qubit reset.

        This method implements a measurement-based reset protocol that:
        1. Measures the qubit state
        2. Applies a π pulse only if the qubit is in |1⟩ state
        3. Repeats the process i times to improve reset fidelity

        Args:
            i: Number of reset iterations to perform
        """
        # Not tested, and not sure qick allows it
        n = 0
        cfg = AttrDict(self.cfg)
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
            test="<",  # Apply pulse if I < threshold
            label=f"NOPULSE0",  # Jump to this label if I >= threshold
        )

        # Apply π pulse to flip qubit from |1⟩ to |0⟩
        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        # Small delay for pulse completion
        self.delay_auto(0.01)
        # Label for conditional jump target

        # For all but the last iteration, perform another measurement
        # Trigger readout
        self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
        # Apply readout pulse
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        # Apply LO pulse if available
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        # Small delay before next iteration
        self.delay_auto(0.01)

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
            test="<",  # Apply pulse if I < threshold
            label=f"NOPULSE1",  # Jump to this label if I >= threshold
        )
        self.label(f"NOPULSE1")
        self.label(f"NOPULSE0")


class QickProgram2Q(AveragerProgramV2):
    """
    Base class for two-qubit quantum experiments using the QICK framework.

    This class extends AveragerProgramV2 to provide a higher-level interface for
    creating and running quantum experiments involving two qubits. It handles
    channel configuration, pulse generation, measurement, and data collection
    for multiple qubits.

    The class is similar to QickProgram but operates on arrays of channels and
    parameters to support multiple qubits simultaneously.
    """

    def __init__(self, soccfg, final_delay=50, cfg={}):
        """
        Initialize the QickProgram2Q with hardware configuration and experiment parameters.

        Args:
            soccfg: System-on-chip configuration object containing hardware details
            final_delay: Delay time (in ns) after each experiment repetition
            cfg: Configuration dictionary containing experiment parameters
        """
        self.cfg = AttrDict(cfg)  # Convert to attribute dictionary for easier access

        # Update configuration with experiment-specific parameters
        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg.expt.reps, final_delay, cfg=cfg)

    def _initialize(self, cfg, readout="standard"):
        """
        Initialize hardware channels and configure pulses for the experiment.

        This method sets up the ADC and DAC channels for qubit control and readout
        for all qubits involved in the experiment. It creates arrays of channels
        and parameters, with each element corresponding to a specific qubit.

        Args:
            cfg: Configuration dictionary
            readout: Readout configuration type (default: "standard")
        """
        cfg = AttrDict(self.cfg)

        # Get qubit indices from configuration
        self.qubits = cfg.expt.qubit
        soc = cfg.hw.soc  # Hardware configuration
        # Configure hardware channels for all qubits (as arrays)
        self.adc_ch = [soc.adcs.readout.ch[q] for q in self.qubits]  # ADC channels
        self.res_ch = [
            soc.dacs.readout.ch[q] for q in self.qubits
        ]  # Resonator channels
        self.res_ch_type = [
            soc.dacs.readout.type[q] for q in self.qubits
        ]  # Resonator types
        self.qubit_ch = [
            soc.dacs.qubit.ch[q] for q in self.qubits
        ]  # Qubit drive channels
        self.qubit_ch_type = [
            soc.dacs.qubit.type[q] for q in self.qubits
        ]  # Qubit types
        self.res_nqz = [
            soc.dacs.readout.nyquist[q] for q in self.qubits
        ]  # Resonator Nyquist zones
        self.qubit_nqz = [
            soc.dacs.qubit.nyquist[q] for q in self.qubits
        ]  # Qubit Nyquist zones
        self.trig_offset = [
            cfg.device.readout.trig_offset[q] for q in self.qubits
        ]  # Trigger offsets
        # Alternative: use maximum trigger offset for all channels
        # self.trig_offset = np.max(self.trig_offset)

        # Configure standard readout parameters for all qubits
        if readout == "standard":
            # Create arrays of readout parameters for each qubit
            self.readout_length = [
                cfg.device.readout.readout_length[q] for q in self.qubits
            ]  # Readout pulse lengths
            self.frequency = [
                cfg.device.readout.frequency[q] for q in self.qubits
            ]  # Readout frequencies
            self.gain = [
                cfg.device.readout.gain[q] for q in self.qubits
            ]  # Readout amplitudes
            self.phase = [
                cfg.device.readout.phase[q] for q in self.qubits
            ]  # Readout phases

        # Configure local oscillators (LO) if available
        if "lo" in cfg.hw.soc and "ch" in cfg.hw.soc.lo:
            # Create arrays of LO parameters for each qubit
            self.lo_ch = [
                cfg.hw.soc.lo.ch[q] if cfg.hw.soc.lo.ch[q] != "None" else None
                for q in self.qubits
            ]  # LO channels
            self.lo_nqz = [
                cfg.hw.soc.lo.nyquist[q] for q in self.qubits
            ]  # LO Nyquist zones
            self.mixer_freq = [
                cfg.hw.soc.lo.mixer_freq[q] for q in self.qubits
            ]  # Mixer frequencies
            self.lo_gain = [cfg.hw.soc.lo.gain[q] for q in self.qubits]  # LO amplitudes

            # Set up LO for each qubit that has one
            for q in range(len(self.qubits)):
                if self.lo_ch[q] is not None:
                    # Declare LO signal generator with offset from mixer frequency
                    self.declare_gen(
                        ch=self.lo_ch[q],
                        nqz=self.lo_nqz[q],
                        mixer_freq=self.mixer_freq[q] - 500,
                    )
                    # Create LO pulse for readout
                    self.add_pulse(
                        self.lo_ch[q],
                        name=f"mix_pulse_{q}",
                        style="const",
                        length=self.readout_length[q],
                        freq=self.mixer_freq[q],
                        phase=0,
                        gain=self.lo_gain[q],
                    )
        else:
            # No LO channels available
            self.lo_ch = [None] * len(self.qubits)

        # Set up readout and control channels for each qubit
        for q in range(len(self.qubits)):
            # Declare resonator signal generator
            self.declare_gen(ch=self.res_ch[q], nqz=self.res_nqz[q])
            # Configure ADC for readout
            self.declare_readout(self.adc_ch[q], length=self.readout_length[q])
            # Configure readout settings
            self.add_readoutconfig(
                ch=self.adc_ch[q],
                name=f"readout_{q}",
                freq=self.frequency[q],
                gen_ch=self.res_ch[q],
            )

            # Create readout pulse for this qubit
            self.add_pulse(
                ch=self.res_ch[q],
                name=f"readout_pulse_{q}",
                style="const",  # Constant amplitude pulse
                ro_ch=self.adc_ch[q],  # Associated readout channel
                length=self.readout_length[q],
                freq=self.frequency[q],
                phase=self.phase[q],
                gain=self.gain[q],
            )

            # Declare qubit control signal generator
            self.declare_gen(ch=self.qubit_ch[q], nqz=self.qubit_nqz[q])

    def _body(self, cfg):
        """
        Default experiment sequence implementation.

        This method should be overridden by subclasses to implement specific
        two-qubit experiments. The default implementation is empty.

        Args:
            cfg: Configuration dictionary
        """
        pass

    def make_pulse(self, q, pulse, name):
        """
        Create a pulse for the specified qubit.

        This method creates different types of pulses (Gaussian or constant)
        for a specific qubit in the two-qubit system.

        Args:
            q: Index of the qubit in the qubit array
            pulse: Dictionary containing pulse parameters
            name: Name to assign to the created pulse

        Supported pulse types:
            - "gauss": Gaussian pulse with specified sigma
            - Other (default): Constant amplitude pulse
        """
        pulse = AttrDict(pulse)  # Convert to attribute dictionary
        if "chan" not in pulse:
            pulse.chan = self.qubit_ch[q]
        # Common pulse parameters
        pulse_args = {
            "ch": pulse.chan,  # Channel for the specified qubit
            "name": name,
            "freq": pulse.freq,  # Pulse frequency
            "phase": pulse.phase,  # Pulse phase
            "gain": pulse.gain,  # Pulse amplitude
        }

        # Create different pulse types based on pulse.type
        if pulse.type == "gauss":
            # Gaussian pulse
            style = "arb"  # Arbitrary waveform

            # Determine pulse length
            if "length" in pulse:
                length = pulse.length
            else:
                length = pulse.sigma * pulse.sigma_inc  # Calculate from sigma

            # Create Gaussian envelope
            self.add_gauss(
                ch=pulse.chan,
                name="ramp",
                sigma=pulse.sigma,  # Width of Gaussian
                length=length,
                even_length=False,
            )
            pulse_args["envelope"] = "ramp"  # Use Gaussian envelope

        else:
            # Default: Constant amplitude pulse

            # Determine pulse length
            if "length" in pulse:
                length = pulse.length
            else:
                length = pulse.sigma

            style = "const"  # Constant amplitude
            pulse_args["length"] = length

        # Set pulse style and add to program
        pulse_args["style"] = style
        self.add_pulse(**pulse_args)

    def make_cfg_pulse(self, q, i, freq, name):
        """
        Create a π pulse for the specified qubit.

        A π pulse rotates the qubit state by 180 degrees around the X-axis,
        flipping the state between |0⟩ and |1⟩.

        Args:
            q: Qubit index in the configuration
            i: Qubit index in the channel arrays
            freq: Dictionary of frequencies for each qubit
            name: Base name of the pulse to create

        Returns:
            Pulse dictionary with the created pulse parameters
        """
        cfg = AttrDict(self.cfg)
        # Get pulse parameters from configuration for the specified qubit
        pulse = {key: value[q] for key, value in cfg.device.qubit.pulses[name].items()}
        # Set pulse frequency and phase
        pulse["freq"] = freq[q]
        pulse["phase"] = 0  # Zero phase for X-rotation
        # Create the pulse using the make_pulse method with indexed name
        self.make_pulse(i, pulse, f"{name}_{i}")
        return pulse

    def collect_shots(self, offset=[0, 0]):
        """
        Collect raw measurement data from all ADCs.

        This method retrieves the raw I/Q data from all ADCs, normalizes it,
        and applies the specified offset correction for each channel.

        Args:
            offset: List of DC offsets to subtract from each channel's raw data

        Returns:
            Tuple of (i_shots_list, q_shots_list) containing lists of I and Q
            quadrature data for each qubit
        """
        # Initialize lists to store I and Q data for each qubit
        i_shots_list, q_shots_list = [], []

        # Process each readout channel
        for i, (ch, rocfg) in enumerate(self.ro_chs.items()):
            nsamp = rocfg["length"]  # Number of samples
            iq_raw = self.get_raw()  # Get raw ADC data

            # Extract and normalize I quadrature data
            i_shots = iq_raw[i][:, :, 0, 0] / nsamp - offset[i]
            i_shots = i_shots.flatten()  # Flatten to 1D array

            # Extract and normalize Q quadrature data
            q_shots = iq_raw[i][:, :, 0, 1] / nsamp - offset[i]
            q_shots = q_shots.flatten()  # Flatten to 1D array

            # Add to lists
            i_shots_list.append(i_shots)
            q_shots_list.append(q_shots)

        return i_shots_list, q_shots_list

    def reset(self, i):
        """
        Perform active qubit reset for multiple qubits.

        This method implements a measurement-based reset protocol that:
        1. Reads the most recent readout result for each qubit
        2. Applies a π pulse only to qubits that read as |e>
        3. Repeats the process i times to improve reset fidelity

        The reset is performed in parallel for all qubits in the system.

        Args:
            i: Number of reset iterations to perform
        """
        # Perform reset sequence i times
        for n in range(i):
            cfg = AttrDict(self.cfg)
            # Wait for readout to complete
            self.wait_auto(cfg.expt.read_wait)
            # Add extra delay for stability
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)

            # Process each qubit separately
            for q in range(len(self.adc_ch)):
                # Read the qubit state and conditionally apply a π pulse.
                # read_and_jump jumps to the label when the test passes, so
                # test="<" jumps over the π pulse when I < threshold and the
                # pulse fires when I >= threshold. That only resets if
                # g_mean < threshold < e_mean for this qubit.
                self.read_and_jump(
                    ro_ch=self.adc_ch[q],
                    component="I",  # Use I quadrature for state discrimination
                    threshold=cfg.expt.threshold[
                        q
                    ],  # Threshold for state discrimination
                    test="<",  # Jump (skip the π pulse) when I < threshold, i.e. |g>
                    label=f"NOPULSE{n}_{q}",
                )
                # Reached only when I >= threshold: flip |e> back to |g>.
                self.pulse(ch=self.qubit_ch[q], name=f"pi_ge_{q}", t=0)
                # Label for conditional jump target
                self.label(f"NOPULSE{n}_{q}")

            # Small delay for pulse completion
            self.delay_auto(0.01)

            # For all but the last iteration, perform another measurement
            if n < i - 1:
                # Measure each qubit again
                for q in range(len(cfg.expt.qubit)):
                    # Apply readout pulse
                    self.pulse(ch=self.res_ch[q], name=f"readout_pulse_{q}", t=0)
                    # Apply LO pulse if available
                    if self.lo_ch[q] is not None:
                        self.pulse(ch=self.lo_ch[q], name=f"mix_pulse_{q}", t=0.0)
                    # Trigger data acquisition
                    self.trigger(
                        ros=[self.adc_ch[q]],
                        pins=[0],
                        t=self.trig_offset[q],
                    )

class QickSMPDProgram(QickProgram):
    """
    Base class for SMPD quantum experiments using the QICK framework.
    Targeted for a qubit coupled to two resonators (buffer and waste).
    Uses the 'qi' parameter from experiments to select which resonator to read out from.
    """

    def _initialize(self, cfg, readout="standard"):
        """
        Initialize hardware channels and configure pulses for the experiment.
        Sets up ADC and DAC channels for the qubit and BOTH resonators.
        The `qi` parameter in the config specifies the active readout resonator.
        """
        cfg = AttrDict(self.cfg)

        # In SMPD, the experiment's 'qubit' list actually provides the resonator index.
        self.qubits = cfg.expt.qubit
        ri = self.qubits[0]  # Resonator index chosen for readout
        qi = ri  # Qubit index now tracks the resonator index

        # We set up both resonators
        self.all_adc_chs = cfg.hw.soc.adcs.readout.ch
        self.all_res_chs = cfg.hw.soc.dacs.readout.ch
        self.all_res_ch_types = cfg.hw.soc.dacs.readout.type
        self.all_res_nqzs = cfg.hw.soc.dacs.readout.nyquist
        self.all_adc_types = cfg.hw.soc.adcs.readout.type

        # Set active channels for compatibility with existing experiments
        self.adc_ch = self.all_adc_chs[ri]
        self.res_ch = self.all_res_chs[ri]
        self.res_ch_type = self.all_res_ch_types[ri]
        self.res_nqz = self.all_res_nqzs[ri]
        self.type = self.all_res_ch_types[ri]
        self.adc_type = self.all_adc_types[ri]

        # Trigger offsets
        self.all_trig_offsets = cfg.device.readout.trig_offset
        self.trig_offset = self.all_trig_offsets[ri]

        # Configure qubit channel (two elements in config)
        if "qubit" in cfg.hw.soc.dacs:
            if isinstance(cfg.hw.soc.dacs.qubit.ch, (list, tuple)):
                self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[qi]
                self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type[qi]
                self.qubit_nqz = cfg.hw.soc.dacs.qubit.nyquist[qi]
            else:
                self.qubit_ch = cfg.hw.soc.dacs.qubit.ch
                self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type
                self.qubit_nqz = cfg.hw.soc.dacs.qubit.nyquist

        # Configure standard readout parameters
        if readout == "standard":
            self.all_readout_lengths = cfg.device.readout.readout_length
            self.all_frequencies = cfg.device.readout.frequency
            self.all_gains = cfg.device.readout.gain
            self.all_phases = cfg.device.readout.phase

            self.readout_length = self.all_readout_lengths[ri]
            self.frequency = self.all_frequencies[ri]
            self.gain = self.all_gains[ri]
            self.phase = self.all_phases[ri]

        # Configure local oscillator (LO) if available
        if (
            "lo" in cfg.hw.soc
            and "ch" in cfg.hw.soc.lo
            and (isinstance(cfg.hw.soc.lo.ch, (list, tuple)) and cfg.hw.soc.lo.ch[ri] != "None" or not isinstance(cfg.hw.soc.lo.ch, (list, tuple)) and cfg.hw.soc.lo.ch != "None")
        ):
            self.lo_ch = cfg.hw.soc.lo.ch[ri] if isinstance(cfg.hw.soc.lo.ch, (list, tuple)) else cfg.hw.soc.lo.ch
            self.lo_nqz = cfg.hw.soc.lo.nyquist[ri] if isinstance(cfg.hw.soc.lo.nyquist, (list, tuple)) else cfg.hw.soc.lo.nyquist
            self.mixer_freq = cfg.hw.soc.lo.mixer_freq[ri] if isinstance(cfg.hw.soc.lo.mixer_freq, (list, tuple)) else cfg.hw.soc.lo.mixer_freq
            self.lo_gain = cfg.hw.soc.lo.gain[ri] if isinstance(cfg.hw.soc.lo.gain, (list, tuple)) else cfg.hw.soc.lo.gain

            self.declare_gen(
                ch=self.lo_ch, nqz=self.lo_nqz, mixer_freq=self.mixer_freq - 500
            )

            self.add_pulse(
                self.lo_ch,
                name="mix_pulse",
                style="const",
                length=self.readout_length,
                freq=self.mixer_freq,
                phase=0,
                gain=self.lo_gain,
            )
        else:
            self.lo_ch = None

        if "aves" in cfg.expt:
            self.add_loop("ave_loop", cfg.expt.aves)

        declared_gens = []
        declared_adcs = []

        # Set up generators and readouts for BOTH resonators
        for i in range(len(self.all_res_chs)):
            r_ch = self.all_res_chs[i]
            a_ch = self.all_adc_chs[i]
            r_nqz = self.all_res_nqzs[i]
            r_type = self.all_res_ch_types[i]
            a_type = self.all_adc_types[i]

            if readout == "standard":
                r_freq = self.all_frequencies[i]
                r_gain = self.all_gains[i]
                r_phase = self.all_phases[i]
                r_len = self.all_readout_lengths[i]
            else:
                r_freq = self.frequency
                r_gain = self.gain
                r_phase = self.phase
                r_len = self.readout_length

            # Setup gen
            if r_type == "full":
                if r_ch not in declared_gens:
                    self.declare_gen(ch=r_ch, nqz=r_nqz)
                    self.add_readout_ramp(r_ch)
                    declared_gens.append(r_ch)
                pulse_args = {
                    "ch": r_ch,
                    "name": f"readout_pulse_{i}",
                    "style": "const",
                    "ro_ch": a_ch,
                    "freq": r_freq,
                    "phase": r_phase,
                    "gain": r_gain,
                }
                if readout == "long":
                    # qick forces flat_top to oneshot, so the periodic tone stays const
                    pulse_args["length"] = 1
                    pulse_args["mode"] = "periodic"
                else:
                    pulse_args["style"] = "flat_top"
                    pulse_args["envelope"] = "readout_ramp"
                    pulse_args["length"] = r_len
                self.add_pulse(**pulse_args)
                
                if i == ri:
                    pulse_args["name"] = "readout_pulse"
                    self.add_pulse(**pulse_args)
                    
            elif r_type == "mux":
                if r_ch not in declared_gens:
                    self.declare_gen(
                        ch=r_ch,
                        nqz=r_nqz,
                        ro_ch=a_ch,
                        mux_freqs=[r_freq],
                        mux_gains=[r_gain],
                        mux_phases=[r_phase],
                    )
                    declared_gens.append(r_ch)
                self.add_pulse(
                    ch=r_ch,
                    name=f"readout_pulse_{i}",
                    style="const",
                    length=r_len,
                    mask=[0],
                )
                if i == ri:
                    self.add_pulse(
                        ch=r_ch,
                        name="readout_pulse",
                        style="const",
                        length=r_len,
                        mask=[0],
                    )
            elif r_type == "int":
                if r_ch not in declared_gens:
                    self.declare_gen(
                        ch=r_ch,
                        nqz=r_nqz,
                        ro_ch=a_ch,
                        mixer_freq=mixer_freq_for(r_freq),
                    )
                    self.add_readout_ramp(r_ch)
                    declared_gens.append(r_ch)
                self.add_pulse(
                    r_ch,
                    ro_ch=a_ch,
                    name=f"readout_pulse_{i}",
                    style="flat_top",
                    envelope="readout_ramp",
                    length=r_len,
                    freq=r_freq,
                    phase=r_phase,
                    gain=r_gain,
                )
                if i == ri:
                    self.add_pulse(
                        r_ch,
                        ro_ch=a_ch,
                        name="readout_pulse",
                        style="flat_top",
                        envelope="readout_ramp",
                        length=r_len,
                        freq=r_freq,
                        phase=r_phase,
                        gain=r_gain,
                    )

            # Setup readout
            if a_type == "dyn":
                if a_ch not in declared_adcs:
                    self.declare_readout(a_ch, length=r_len)
                    declared_adcs.append(a_ch)
                self.add_readoutconfig(
                    ch=a_ch, name=f"readout_{i}", freq=r_freq, gen_ch=r_ch
                )
                if i == ri:
                    self.add_readoutconfig(
                        ch=a_ch, name="readout", freq=r_freq, gen_ch=r_ch
                    )
            elif a_type == "std":
                if a_ch not in declared_adcs:
                    # phase=0 for the same reason as the single-readout
                    # path above: the pulse already carries r_phase.
                    self.declare_readout(
                        ch=a_ch,
                        length=r_len,
                        freq=r_freq,
                        phase=0,
                        gen_ch=r_ch,
                    )
                    declared_adcs.append(a_ch)

        if "qubit" in cfg.hw.soc.dacs:
            if self.qubit_ch not in declared_gens:
                self.declare_gen(ch=self.qubit_ch, nqz=self.qubit_nqz)
                declared_gens.append(self.qubit_ch)

    def _body(self, cfg):
        """Default body: configure readouts, play the readout pulse, and trigger all ADCs."""
        cfg = AttrDict(self.cfg)
        if self.adc_type == "dyn":
            for a_ch, i in zip(self.all_adc_chs, range(len(self.all_adc_chs))):
                if self.all_adc_types[i] == "dyn":
                    self.send_readoutconfig(ch=a_ch, name=f"readout_{i}", t=0)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        # Trigger all readouts to satisfy `get_raw()` matching declared channels length
        self.trigger(ros=self.all_adc_chs, pins=[0], t=self.trig_offset, ddr4=True)

    def measure(self, cfg):
        """Play the readout pulse (and LO mix pulse if present), trigger the ADCs, and optionally reset."""
        cfg = AttrDict(self.cfg)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
        self.trigger(
            ros=self.all_adc_chs,
            pins=[0],
            t=self.trig_offset,
        )
        if cfg.expt.active_reset:
            self.reset(cfg.expt.reset)

    def reset(self, i):
        """
        Perform active qubit reset.

        This method implements a measurement-based reset protocol that:
        1. Reads the most recent readout result
        2. Applies a π pulse only if the qubit reads as |e>
        3. Repeats the process i times to improve reset fidelity

        So `i` rounds apply between 0 and `i` π pulses, and add `i - 1` extra
        readouts to the shot (the last round is not re-measured). That is what
        `collect_shots` sees on the trailing readout axis. With
        `cfg.expt.reset_verify` the last round is re-measured too, giving `i`
        extra readouts and making the final one the true post-reset state.

        `cfg.expt.reset_f` adds a pi_ef ahead of each pi_ge, for sequences
        that can leave population in |f>.

        `cfg.expt.reset_latch` gives "while excited" semantics at a fixed
        trip count: the loop still runs `i` rounds and measures every one, but
        once any round reads ground a latch register suppresses every later pi
        pulse. A real while loop cannot be used here -- see the comment in the
        loop body -- and the latch removes the same dominant error, a ground
        shot misread as excited on the final round.

        Args:
            i: Number of reset iterations to perform
        """
        cfg = AttrDict(self.cfg)

        if cfg.expt.reset_latch:
            # A data register, allocated rather than hardcoded: loops consume
            # data registers too, and add_reg picks a free address. Cleared
            # here so the latch is per-shot, not per-program.
            self.reset_latch = self.add_reg(name="reset_latch", allow_reuse=True)
            self.write_reg(dst=self.reset_latch, src=0)

        # Perform reset sequence i times
        for n in range(i):
            # Wait for readout to complete
            self.wait_auto(cfg.expt.read_wait)
            # Add extra delay for stability
            self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)

            # Read the qubit state and conditionally apply a π pulse.
            # read_and_jump jumps to the label when the test passes, so
            # test="<" jumps over the π pulse when I < threshold and the pulse
            # fires when I >= threshold. That only resets if
            # g_mean < threshold < e_mean, the ordering 12_single_shot.py
            # establishes via device.readout.phase and device.readout.threshold.
            if cfg.expt.reset_latch:
                # "while excited" semantics with a compile-time-fixed trip
                # count. A real while loop would trigger this readout a
                # data-dependent number of times per rep, and qick sizes the
                # buffer read by *counting trigger instructions*: a variable
                # count silently pulls the wrong IQ values out, with no error
                # (confirmed by the qick authors). The usual fix is a second
                # logical readout on the same physical input -- discard the
                # feedback channel, keep one trigger on the readout channel --
                # but our waste line sits on a dynamic readout (ADC6, index 1)
                # that is 1:1 with its input, so there is no second index to
                # spend. The PFB bank on ADC5 has 8 indices on one input and
                # would give it, once that signal is routed there.
                #
                # So: run a fixed `i` rounds, but latch. Once any round reads
                # ground, the latch suppresses every later pi pulse, which is
                # the part that matters -- a ground shot misread as excited on
                # the *last* round is what sets the floor, and this removes it.
                self.read_input(ro_ch=self.adc_ch)
                # Already confirmed ground on an earlier round: play nothing.
                # arg1 alone with no op tests the register itself, so this is
                # "latch != 0". It does not touch s_port_l, so the threshold
                # comparison below still sees the value just read.
                self.cond_jump(label=f"NOPULSE{n}", arg1=self.reset_latch,
                               test="NZ")
                # I < threshold: ground. Latch it, then play nothing.
                self.cond_jump(label=f"GROUND{n}", arg1="s_port_l", op="-",
                               arg2=cfg.expt.threshold, test="S")
            else:
                self.read_and_jump(
                    ro_ch=self.adc_ch,
                    component="I",  # Use I quadrature for state discrimination
                    threshold=cfg.expt.threshold,  # Threshold for state discrimination
                    test="<",  # Jump (skip the π pulse) when I < threshold, i.e. |g>
                    label=f"NOPULSE{n}",
                )

            # Everything between the jump above and NOPULSE{n} below runs
            # only when I >= threshold, i.e. not ground. A ground shot jumps
            # past the whole block and plays nothing.
            #
            # `if cfg.expt.reset_f` is a compile-time switch -- whether the
            # pi_ef is emitted into the program at all -- not a second
            # runtime branch. There is no branch between |e> and |f| because
            # the readout cannot support one: chi is large compared to kappa,
            # so the resonator response saturates and the two blobs land on
            # top of each other (0.46 sigma apart on run 33 Waymar). Since
            # the only goal is |g>, a not-ground shot is assumed to be |f>
            # and walked down f -> e -> g. Self-correcting across rounds:
            #     |f> -> pi_ef -> |e> -> pi_ge -> |g>     (one round)
            #     |e> -> pi_ef -> |f> -> pi_ge -> |f>     (fixed next round)
            # so |f> clears immediately and |e> takes two, which is why
            # reset_f wants `reset` >= 2.
            if cfg.expt.reset_f:
                self.pulse(ch=self.qubit_ch, name="pi_ef", t=0)
                self.delay_auto(0.01)
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            # Small delay for pulse completion
            self.delay_auto(0.01)
            if cfg.expt.reset_latch:
                # Skip past the latch-setting block on the pulsed path.
                self.jump(f"NOPULSE{n}")
                self.label(f"GROUND{n}")
                self.write_reg(dst=self.reset_latch, src=1)
            # Jump target for the ground case: skips both pulses above.
            self.label(f"NOPULSE{n}")

            # Re-measure so the next round has something to branch on. The
            # last round is skipped in normal use: nothing acts on a readout
            # taken after the final π pulse, so it would only cost time.
            # `reset_verify` adds it back, which is what a *calibration* needs
            # -- without it the last round's conditional pulse is unmeasured
            # and the recorded end state is one round stale. 12b_active_reset
            # sets it; production runs should not.
            if n < i - 1 or cfg.expt.reset_verify or cfg.expt.reset_latch:
                
                # Apply readout pulse
                self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
                # Trigger readout
                self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
                # Apply LO pulse if available
                if self.lo_ch is not None:
                    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
                # Small delay before next iteration
                # self.delay_auto(0.01)
            # else:
            #     self.delay_auto(0.01)  # Final small delay after last iteration

    def cond_reset(self, i):
        """Two-round conditional reset: measure and apply a pi pulse only when the qubit reads as |1>."""
        n = 0
        cfg = AttrDict(self.cfg)
        self.wait_auto(cfg.expt.read_wait)
        self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)

        self.read_and_jump(
            ro_ch=self.adc_ch,
            component="I",
            threshold=cfg.expt.threshold,
            test="<",
            label=f"NOPULSE0",
        )

        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        self.delay_auto(0.01)

        self.trigger(ros=self.all_adc_chs, pins=[0], t=self.trig_offset)
        self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
        if self.lo_ch is not None:
            self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
        self.delay_auto(0.01)

        self.wait_auto(cfg.expt.read_wait)
        self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)

        self.read_and_jump(
            ro_ch=self.adc_ch,
            component="I",
            threshold=cfg.expt.threshold,
            test="<",
            label=f"NOPULSE1",
        )
        self.label(f"NOPULSE1")
        self.label(f"NOPULSE0")

    def collect_shots(self, offset=0, single=True):
        """
        Return per-shot (I, Q) values from the raw ADC buffer, normalized by readout length.

        Args:
            offset: Value subtracted from each normalized sample
            single: If True, return flattened arrays for the active channel;
                if False, return lists with one entry per readout
        """
        if not single:
            i_shots = []
            q_shots = []
            
        iq_raw = self.get_raw()
        
        # Get active channel index from ro_chs keys
        idx = list(self.ro_chs.keys()).index(self.adc_ch)
        rocfg = self.ro_chs[self.adc_ch]
        nsamp = rocfg["length"]
        
        i_shots_vec = iq_raw[idx][:, :, :, 0] / nsamp - offset
        q_shots_vec = iq_raw[idx][:, :, :, 1] / nsamp - offset
        
        if single:
            i_shots = i_shots_vec.flatten()
            q_shots = q_shots_vec.flatten()
        else:
            for j in range(i_shots_vec.shape[1]):
                i_shots.append(i_shots_vec[:, j, :])
                q_shots.append(q_shots_vec[:, j, :])
                
        return i_shots, q_shots

    def make_pi_pulse(self, q, freq, name):
        """
        Create a π pulse for the qubit.
        Uses `q` (resonator index) to index into the qubit parameters.
        """
        cfg = AttrDict(self.cfg)
        pulse_cfg = cfg.device.qubit.pulses[name]
        pulse = {}
        for key, value in pulse_cfg.items():
            # For robustness handle lists/tuples passed from YAML fallback mapping
            if isinstance(value, (list, tuple)):
                pulse[key] = value[q]
            else:
                pulse[key] = value
                
        if isinstance(freq, (list, tuple)):
            pulse["freq"] = freq[q]
        else:
            pulse["freq"] = freq
            
        if "phase" not in pulse:
            pulse["phase"] = 0
            
        self.make_pulse(pulse, name)
        return pulse