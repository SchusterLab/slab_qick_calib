from qick.asm_v2 import AveragerProgramV2
from ...exp_handling.datamanagement import AttrDict
from qick import *

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


class QickProgram(AveragerProgramV2):
    """
    Base class for single-qubit quantum experiments using the QICK framework.

    This class extends AveragerProgramV2 to provide a higher-level interface for
    creating and running quantum experiments. It handles channel configuration,
    pulse generation, measurement, and data collection for a single qubit.

    The class is designed to be extended by specific experiment implementations
    that override the _body method to define the experiment sequence.
    """

    def __init__(self, soccfg, final_delay=50, cfg={}):
        """
        Initialize the QickProgram with hardware configuration and experiment parameters.

        Args:
            soccfg: System-on-chip configuration object containing hardware details
            final_delay: Delay time (in ns) after each experiment repetition
            cfg: Configuration dictionary containing experiment parameters
        """
        self.cfg = AttrDict(cfg)  # Convert to attribute dictionary for easier access

        # Update configuration with experiment-specific parameters
        self.cfg.update(self.cfg.expt)
        super().__init__(soccfg, self.cfg.expt.reps, final_delay, cfg=cfg)

    def save_params(self, readout='standard'):
        cfg = AttrDict(self.cfg)

        # Get qubit index from configuration
        self.qubits = cfg.expt.qubit
        q = self.qubits[0]  # Single qubit index

        # Configure hardware channels for the selected qubit
        self.adc_ch = cfg.hw.soc.adcs.readout.ch[q]  # ADC channel for readout
        self.res_ch = cfg.hw.soc.dacs.readout.ch[q]  # DAC channel for resonator drive
        self.res_ch_type = cfg.hw.soc.dacs.readout.type[
            q
        ]  # Resonator channel type (full, mux, int)
        self.res_nqz = cfg.hw.soc.dacs.readout.nyquist[
            q
        ]  # Nyquist zone for resonator (1 for <5 GHz, 2 for >5 GHz)
        self.type = cfg.hw.soc.dacs.readout.type[q]  # Type of readout channel
        self.adc_type = cfg.hw.soc.adcs.readout.type[q]  # ADC channel type

        self.trig_offset = cfg.device.readout.trig_offset[q]  # Trigger timing offset
        if "qubit" in cfg.hw.soc.dacs:
            self.qubit_ch = cfg.hw.soc.dacs.qubit.ch[q]  # DAC channel for qubit drive
            self.qubit_ch_type = cfg.hw.soc.dacs.qubit.type[q]  # Qubit channel type
            self.qubit_nqz = cfg.hw.soc.dacs.qubit.nyquist[q]  # Nyquist zone for qubit

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
        if (
            "lo" in cfg.hw.soc
            and "ch" in cfg.hw.soc.lo
            and cfg.hw.soc.lo.ch[q] != "None"
        ):
            
            # Declare LO signal generator with offset from mixer frequency, this assumes it is using int generator
            self.declare_gen(
                ch=self.lo_ch, nqz=self.lo_nqz, mixer_freq=self.mixer_freq - 500
            )

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


        if "aves" in cfg.expt:
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
                pulse_args["length"] = 1
                pulse_args["mode"] = "periodic"
            else:
                pulse_args["length"] = self.readout_length
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
                mixer_freq=self.frequency - 400,
            )

            self.add_pulse(
                self.res_ch,
                ro_ch=self.adc_ch,
                name="readout_pulse",
                style="const",  # Constant amplitude pulse
                length=self.readout_length,
                freq=self.frequency,
                phase=self.phase,
                gain=self.gain,
            )
        
        # Configure readout settings
        if self.adc_type == "dyn":
            self.declare_readout(self.adc_ch, length=self.readout_length)  # Configure ADC for readout
            self.add_readoutconfig(ch=self.adc_ch, name="readout", freq=self.frequency, gen_ch=self.res_ch)
        elif self.adc_type == "std":
            self.declare_readout(
                ch=self.adc_ch,
                length=self.readout_length,
                freq=self.frequency,
                phase=self.phase,
                gen_ch=self.res_ch,
            )

        if "qubit" in cfg.hw.soc.dacs:
            # Set up qubit control channel
            self.declare_gen(
                ch=self.qubit_ch, nqz=self.qubit_nqz
            )  # Declare qubit signal generator

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
        # Perform active reset if enabled
        if cfg.expt.active_reset:
            self.reset(3)  # Reset qubit state 3 times

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
                even_length=False,
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

            # Extract and normalize I quadrature data for this channel
            i_shots_vec = iq_raw[i][:, :, :, 0] / nsamp - offset
            # Extract and normalize Q quadrature data for this channel
            q_shots_vec = iq_raw[i][:, :, :, 1] / nsamp - offset

            i_shots_list.append(i_shots_vec)
            q_shots_list.append(q_shots_vec)

        return i_shots_list, q_shots_list

    def reset(self, i):
        """
        Perform active qubit reset.

        This method implements a measurement-based reset protocol that:
        1. Measures the qubit state
        2. Applies a π pulse only if the qubit is in |1⟩ state
        3. Repeats the process i times to improve reset fidelity

        Args:
            i: Number of reset iterations to perform
        """
        cfg = AttrDict(self.cfg)

        # Perform reset sequence i times
        for n in range(i):
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
            #print(cfg.expt.threshold)

            # Apply π pulse to flip qubit from |1⟩ to |0⟩
            self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
            # Small delay for pulse completion
            self.delay_auto(0.01)
            # Label for conditional jump target
            self.label(f"NOPULSE{n}")

            # For all but the last iteration, perform another measurement
            if n < i - 1:
                
                # Apply readout pulse
                self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
                # Trigger readout
                self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
                # Apply LO pulse if available
                if self.lo_ch is not None:
                    self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
                # Small delay before next iteration
                self.delay_auto(0.01)
            else:
                self.delay_auto(0.01)  # Final small delay after last iteration

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
        1. Measures the state of each qubit
        2. Applies a π pulse only to qubits that are in the |1⟩ state
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
                # Read qubit state and conditionally apply π pulse
                # If I < threshold (qubit in |1⟩), apply π pulse to return to |0⟩
                # If I >= threshold (qubit in |0⟩), skip the π pulse
                self.read_and_jump(
                    ro_ch=self.adc_ch[q],
                    component="I",  # Use I quadrature for state discrimination
                    threshold=cfg.expt.threshold[
                        q
                    ],  # Threshold for state discrimination
                    test="<",  # Apply pulse if I < threshold
                    label=f"NOPULSE{n}_{q}",  # Jump to this label if I >= threshold
                )
                # Apply π pulse to flip qubit from |1⟩ to |0⟩
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
