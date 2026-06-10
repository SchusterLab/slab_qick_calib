# SLAB QICK Calibration Experiments

This document provides a comprehensive guide to configuring and running various quantum experiments using the SLAB QICK calibration framework. The framework is designed for calibrating and characterizing superconducting qubits using the QICK (Quantum Instrumentation Control Kit) hardware.

## Table of Contents
- [Introduction](#introduction)
- [Experiment Configuration](#experiment-configuration)
- [Available Experiments](#available-experiments)
  - [Single Qubit Experiments](#single-qubit-experiments)
  - [Two Qubit Experiments](#two-qubit-experiments)
  - [Flux Experiments](#flux-experiments)
- [Common Experiment Workflows](#common-experiment-workflows)
- [Parameter Reference](#parameter-reference)

## Introduction

The SLAB QICK calibration framework provides a comprehensive set of experiments for characterizing and calibrating superconducting qubits. The framework is built on top of the QICK hardware platform and provides a high-level interface for running quantum experiments.

Before running experiments, make sure a nameserver is running on the network (on whatever computer you want to run it, open NameServer.ipynb and run with that computer's IP address). Then, on QICK board, go to pyro4 folder and open 01_server.ipynb, setting an alias for the board that you will use when connecting to the board on your client computer, and setting the ip address to that of the nameserver. This is also where you would specify the location of firmware.

## Experiment Configuration

### QICK Program vs. QICK Experiment

In this framework, it's important to understand the distinction between a `QickProgram` and a `QickExperiment`:

- **`QickProgram`**: This is a lower-level class that directly interacts with the QICK's pulse processor. It is responsible for defining the pulse sequences, acquiring data, and managing the hardware registers. Each experiment has an associated `QickProgram` that handles the hardware communication.
- **`QickExperiment`**: This is a higher-level class that encapsulates the entire experimental procedure. It handles data processing, fitting, plotting, and saving. It uses a `QickProgram` to execute the pulse sequence on the hardware. When you run an experiment, you will typically interact with the `QickExperiment` class.

This separation of concerns allows for a modular and extensible framework where the high-level experiment logic is decoupled from the low-level hardware control.

**For detailed information about writing QICK programs**, see the [QICK Demos and Documentation](https://github.com/meeg/qick_demos_sho/tree/main/tprocv2).

### Basic Configuration Setup

To set up a new configuration or data folder:
When you initialize the config, you will need to set the DAC/ADC channels for resonator and qubit. You can look at the soc printout to identify the channel number you are plugged into.

```python
# Set up paths
expt_path = 'C:\_Data\YourExperiment\2025-05-12\'
cfg_file = 'YourConfig.yml'

# Create new config and folders if needed
import os
import slab_qick_calib.config as config

new_config = False  # Set to True to create a new config
new_folder = False  # Set to True to create a new data folder

configs_dir = os.path.join(os.getcwd(), 'configs')
cfg_file_path = os.path.join(configs_dir, cfg_file)
images_dir = os.path.join(expt_path, 'images')
summary_dir = os.path.join(images_dir, 'summary')

if new_config or new_folder:
    if new_config:
        # Initialize config with 4 qubits, using 'full' type and 'bf1_soc' aliases
        config.init_config(cfg_file_path, 4, type='full', aliases='bf1_soc', t1=30)

    if not os.path.exists(expt_path):
        os.makedirs(expt_path)
        os.mkdir(images_dir)
        os.mkdir(summary_dir)
```

### Running Experiments

To run an experiment, you need to:
1. Load the configuration
2. Connect to instruments
3. Create a configuration dictionary
4. Initialize and run the experiment

```python
# Load configuration
cfg_path = os.path.join(os.getcwd(), 'configs', cfg_file)
auto_cfg = config.load(cfg_path)

# Connect to instruments
im = InstrumentManager(ns_address=ip)
soc = QickConfig(im[auto_cfg['aliases']['soc']].get_cfg())

# Create configuration dictionary
cfg_dict = {'soc': soc, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}

# Run an experiment (example: Resonator Spectroscopy)
qi = 0  # Qubit index
rspec = meas.ResSpec(cfg_dict, qi=qi, style='coarse', params={'start': 5000, 'span': 1700})
```

### Experiment Parameters

Each experiment accepts a `params` dictionary that allows you to customize the experiment. Common parameters include:

- `reps`: Number of repetitions for each experiment point
- `rounds`: Number of software averages
- `expts`: Number of experiment points
- `start`: Start value for the swept parameter
- `span`: Span of the swept parameter
- `gain`: Pulse amplitude (DAC units)
- `length`: Pulse length (μs)
- `frequency`: Pulse frequency (MHz)
- `active_reset`: Whether to use active reset

Experiment-specific parameters are documented in the respective experiment classes.
Experiments are configured using parameter dictionaries that are passed to the experiment constructor. Each experiment has a set of default parameters that can be overridden by the user.
All times are in μs. All gains are out of 1.

## Available Experiments

### Single Qubit Experiments

#### Time of Flight (ToF) Calibration

**File**: `experiments/single_qubit/tof_calibration.py`

**Description**: Measures the time it takes for the signal to run through the cables, plus latency. It determines the time that we should wait to make measurements. Time of flight (tof) is stored in parameter `cfg.device.readout.trig_offset`.

**Purpose**: Run this calibration when the wiring of the setup is changed. By calibrating this delay, we ensure that data acquisition starts at the optimal time when the signal actually arrives at the detector, avoiding dead time or missed signals. Not automated, need to figure out trig_offset by eye.

**Key Parameters**:
- `rounds`: Number of software averages for the measurement
- `readout_length`: Length of the readout pulse (μs)
- `trig_offset`: Current trigger offset for the ADC (μs)
- `gain`: Amplitude of the readout pulse (DAC units)
- `frequency`: Frequency of the readout pulse (MHz)
- `reps`: Number of averages per point. ToF largely only works for reps=1, so we crank up the rounds. 
- `qubit`: List containing the qubit index to calibrate
- `phase`: Phase of the readout pulse
- `final_delay`: Final delay after the pulse sequence (μs)
- `check_e`: Whether to excite qubit to |1⟩ state before measurement
- `use_readout`: Whether to use current readout parameters (gain and phase) -- default is max gain, phase 0. 

**Example**:
```python
tof = meas.ToFCalibrationExperiment(cfg_dict=cfg_dict, qi=0)
```

**2D Time of Flight Calibration**:

The package also provides a 2D Time of Flight calibration experiment (`ToF2D`) that performs multiple measurements over time. This can be useful for performing many iterations of the readout trace.

```python
tof_2d = meas.ToF2D(cfg_dict=cfg_dict, qi=0, params={'expts_count': 500})
```

#### Resonator Spectroscopy

**File**: `experiments/single_qubit/resonator_spectroscopy.py`

**Description**: Measures the resonant frequency of the readout resonator by sweeping the readout pulse frequency and looking for the frequency with the maximum measured amplitude. The resonator frequency is stored in the parameter `cfg.device.readout.frequency`.

The module includes:
- `ResSpecProgram`: Defines the pulse sequence for the resonator spectroscopy experiment
- `ResSpec`: Main experiment class for resonator spectroscopy
- `ResSpecPower`: 2D version that sweeps both frequency and power
- `ResSpec2D`: 2D version for repeated measurements

Note that harmonics of the clock frequency will show up as "infinitely" narrow peaks!

**Key Parameters**:
- `start`: Start frequency (MHz)
- `span`: Frequency span (MHz), if `kappa` uses 7 * `kappa` from cfg as span 
- `center`: Center frequency (MHz) - alternative to `start`
- `expts`: Number of frequency points
- `gain`: Gain of the readout resonator
- `length`: Length of the readout pulse (μs)
- `final_delay`: Delay time between repetitions (μs)
- `pulse_e`: Boolean to add π pulse prior to measurement (excite qubit)
- `pulse_f`: Boolean to add π pulse on |e⟩-|f⟩ prior to measurement (excite to 2nd level)
- `style`: 'coarse' for wide frequency scan, 'fine' for narrow scan
- `long_pulse`: Whether to use a long readout pulse, requiring 'periodic' mode where readout pulse occurs entire time, when pulse length exceeds hardware limit
- `reps`: Number of repetitions
- `rounds`: Number of software averages
- `loop`: Whether to use loop mode for acquisition, needed for non-linear sweeps
- `phase_const`: Whether to use constant phase spacing instead of frequency spacing (used with loop)
- `active_reset`: Whether to use active reset

**Style Parameters**:
- **'coarse'**: Wide frequency span for initial search
  - `start`: 6000 MHz
  - `expts`: 5000 points
  - `span`: 500 MHz
- **'fine'** (default): Narrow span for precise measurement
  - `center`: Resonator frequency from config
  - `expts`: 220 points
  - `span`: 5 MHz

**Example**:
```python
# Coarse scan to find resonators
rspecc = meas.ResSpec(cfg_dict, qi=0, style='coarse', params={'start': 5000, 'span': 1700})

# Fine scan to precisely measure a resonator
rspec = meas.ResSpec(cfg_dict, qi=0, params={'span': 5})

# Measure resonator with qubit in excited state (to measure dispersive shift)
rspec_e = meas.ResSpec(cfg_dict, qi=0, params={'pulse_e': True})
```

#### Resonator Power Spectroscopy

**File**: `experiments/single_qubit/resonator_spectroscopy.py`

**Description**: Performs a 2D sweep of both readout frequency and power to map out how the resonator response changes with power. This allows measurement of the Lamb shift and other power-dependent effects. It's useful for finding a good value for gain to park your readout at until you run readout optimization. When log sweep used, increases the number of averages to account for decrease in signal. 

**Key Parameters**:
- `rng`: Range for the gain sweep, going from max_gain to max_gain/rng (default: 100)
- `max_gain`: Maximum gain value for sweep (default: qubit.max_gain, the hardware max)
- `expts_gain`: Number of gain points (default: 20)
- `span`: Frequency span (MHz)
- `f_off`: Frequency offset of center from resonant frequency in MHz (usually negative, default: 4), to account for lamb shift. 
- `min_reps`: Minimum number of repetitions (default: 100), used for the high power points of sweep. 
- `log`: Whether to use logarithmic scaling for the gain sweep (default: True)
- `pulse_e`: Whether to apply a π pulse to excite the qubit before measurement
- `expts`: Number of frequency points (default: 200)
- `reps`: Base number of repetitions, divided by 2400 to prevent scan from taking forever. Can adjust for worse SNR. 

**Example**:
```python
# Standard power sweep
rpowspec = meas.ResSpecPower(cfg_dict, qi=0, params={'rng': 300, 'span': 5, 'f_off': 1, 'expts_gain': 30})

# Power sweep with qubit in excited state
rpowspec_e = meas.ResSpecPower(cfg_dict, qi=0, params={'pulse_e': True, 'span': 5})
```

The experiment produces a 2D plot showing the resonator response as a function of both frequency and power. The Lamb shift (difference in resonator frequency between high and low power) can be extracted from the fit.

#### Qubit Spectroscopy

**File**: `experiments/single_qubit/pulse_probe_spectroscopy.py`

**Description**: Measures the qubit frequency by applying a probe pulse with variable frequency and measuring the resulting qubit state. This allows determination of the qubit transition frequencies (f_ge and f_ef).

The module includes:
- `QubitSpecProgram`: Defines the pulse sequence for the spectroscopy experiment
- `QubitSpec`: Main experiment class for frequency spectroscopy
- `QubitSpecPower`: 2D version that sweeps both frequency and power

This experiment is particularly useful for finding qubit frequencies and characterizing the qubit spectrum as a function of probe power.

**Key Parameters**:
- `start`: Start frequency (MHz)
- `span`: Frequency span (MHz)
- `expts`: Number of frequency points
- `gain`: Probe pulse amplitude
- `length`: Probe pulse length (μs) (can be set to "t1" to use 3*T1 for GE or T1/4 for EF)
- `checkEF`: Whether to check the e-f transition
- `pulse_type`: Type of pulse ('const' or 'gauss')
- `sep_readout`: Whether to separate the probe pulse and readout
- `readout_length`: Length of the readout pulse (μs)
- `reps`: Number of repetitions
- `rounds`: Number of software averages
- `final_delay`: Delay time between repetitions (μs)
- `active_reset`: Whether to use active reset

**Style Parameters**:
- **'huge'**: Very wide frequency span with high power
  - `gain`: 80 * low_gain * spec_gain
  - `span`: 1500 MHz
  - `expts`: 1000 points
- **'coarse'**: Wide frequency span with medium power
  - `gain`: 20 * low_gain * spec_gain
  - `span`: 500 MHz
  - `expts`: 500 points
- **'medium'**: Medium frequency span with low power
  - `gain`: 5 * low_gain * spec_gain
  - `span`: 50 MHz
  - `expts`: 200 points
- **'fine'**: Narrow frequency span with very low power
  - `gain`: low_gain * spec_gain
  - `span`: 5 MHz
  - `expts`: 100 points
  - `reps`: 2 * self.reps

**Example**:
```python
# Find g-e transition with medium range
qspec = meas.QubitSpec(cfg_dict, qi=0, style='medium')

# Find e-f transition
qspec_ef = meas.QubitSpec(cfg_dict, qi=0, style='medium', params={'checkEF': True})

# Fine scan with low power
qspec_fine = meas.QubitSpec(cfg_dict, qi=0, style='fine')
```

#### Qubit Spectroscopy Power

**File**: `experiments/single_qubit/pulse_probe_spectroscopy.py`

**Description**: Performs a 2D sweep of frequency and power to map out the qubit response. This experiment is useful for identifying power-dependent broadening and multi-photon or excited state transitions. It creates a 2D map showing how the qubit spectrum changes with probe power.

**Key Parameters**:
- `span`: Frequency span (MHz)
- `expts`: Number of frequency points
- `reps`: Number of repetitions for each experiment (default: 2 * self.reps)
- `rng`: Range for logarithmic gain sweep (default: 50)
- `max_gain`: Maximum gain value of sweep (default: qubit.max_gain, max from hardware)
- `expts_gain`: Number of gain points (default: 10)
- `log`: Whether to use logarithmic gain spacing (default: True)
- `checkEF`: Whether to check the e-f transition

**Style Parameters**:
- **'coarse'**: Wide frequency span with many points
  - `span`: 800 MHz
  - `expts`: 500 points
- **'fine'**: Narrow frequency span with fewer points
  - `span`: 40 MHz
  - `expts`: 100 points

**Example**:
```python
# Standard power sweep
qspec_pow = meas.QubitSpecPower(cfg_dict, qi=0, params={'span': 40, 'expts': 100, 'max_gain': 0.4})

# Power sweep for EF transition
qspec_pow_ef = meas.QubitSpecPower(cfg_dict, qi=0, params={'checkEF': True, 'span': 20})

# Fine scan with more gain points
qspec_pow_fine = meas.QubitSpecPower(cfg_dict, qi=0, style='fine', params={'expts_gain': 20})
```

The experiment produces a 2D plot showing the qubit response as a function of both frequency and power. This can reveal features like multi-photon transitions and power broadening.

#### Rabi Experiment

**File**: `experiments/single_qubit/rabi.py`

**Description**: Measures Rabi oscillations by varying either the amplitude or length of a drive pulse. Rabi oscillations are observed by varying either the amplitude or length of a driving pulse and measuring the resulting qubit state. This allows determination of the π-pulse parameters (amplitude and duration) needed for qubit control.

The module includes:
- `RabiProgram`: Defines the pulse sequence for the Rabi experiment
- `RabiExperiment`: Main experiment class for amplitude or length Rabi oscillations
- `ReadoutCheck`: Class for checking readout parameters
- `RabiChevronExperiment`: 2D version that sweeps both frequency and amplitude/length
- `Rabi2D`: 2D version that sweeps both length and gain

**Key Parameters**:
- `expts`: Number of amplitude/length points (default: 60)
- `reps`: Number of repetitions for each experiment (default: self.reps)
- `rounds`: Number of software averages (default: self.rounds)
- `gain`: Pulse amplitude (for length sweep)
- `sigma`: Pulse width in time (for amplitude sweep)
- `max_gain`: Maximum gain used in amplitude sweep (if specified, used instead of gain)
- `max_length`: Maximum length used in length sweep (if specified, used instead of sigma)
- `sweep`: 'amp' or 'length' to specify what to sweep (default: 'amp')
- `checkEF`: Whether to check the e-f transition (default: False)
- `pulse_ge`: Boolean flag to indicate if pulse is for ground to excited state transition (default: True), used for temp measurement
- `start`: Starting point for the experiment (default: 0.003 for amp, 3*cycles2us(1) for length)
- `pulse_type`: Type of pulse used in the experiment (default: 'gauss', others are flat_top, const)
- `num_osc`: Try to set max gain or length to measure this number of oscillations (default: 2.5)
- `n_pulses`: Number of Rabi pulses to apply (default: 1)
- `active_reset`: Whether to use active qubit reset
- `loop`: Whether to use loop-based acquisition (default: False)

**Style Parameters**:
- **'fine'**: More averages for better precision
  - `rounds`: 2 * self.rounds
- **'fast'**: Fewer points for quick measurements
  - `expts`: 25

**Example**:
```python
# Amplitude Rabi
amp_rabi = meas.RabiExperiment(cfg_dict, qi=0)

# Length Rabi
len_rabi = meas.RabiExperiment(cfg_dict, qi=0, params={'sweep': 'length', 'type': 'const', 'sigma': 0.1, 'gain': 0.15})

# EF Rabi (checks e-f transition)
ef_rabi = meas.RabiExperiment(cfg_dict, qi=0, params={'checkEF': True})
```

#### Rabi Chevron

**File**: `experiments/single_qubit/rabi.py`

**Description**: Performs a 2D sweep of pulse amplitude/length and frequency to map out Rabi oscillations as a function of detuning. This experiment creates a 2D map showing how the Rabi oscillation frequency changes with detuning from the qubit frequency, creating a characteristic chevron pattern.

**Key Parameters**:
- `span_f`: Frequency span (MHz) (default: 20)
- `expts_f`: Number of frequency points (default: 30)
- `sweep`: 'amp' or 'length' to specify what to sweep (default: 'amp')
- `checkEF`: Whether to check the e-f transition (default: False)
- `start_f`: Start qubit frequency (MHz) (calculated from f_ge/f_ef and span_f)

**Example**:
```python
# Amplitude Rabi Chevron
amp_rabi_chevron = meas.RabiChevronExperiment(cfg_dict, qi=0, params={'span_f': 10})

# Length Rabi Chevron
len_rabi_chevron = meas.RabiChevronExperiment(cfg_dict, qi=0, params={'span_f': 10, 'sweep': 'length'})
```

The experiment produces a 2D plot showing the qubit response as a function of both drive frequency and amplitude/length. The chevron pattern allows visualization of how the Rabi oscillation frequency increases with detuning from the qubit frequency.

#### ReadoutCheck

**File**: `experiments/single_qubit/rabi.py`

**Description**: Class for checking readout parameters. This experiment is used to characterize the readout by sweeping either the end wait time or the gain of a qubit pulse and measuring the response.

**Key Parameters**:
- `expts`: Number of points in the sweep (default: 30)
- `reps`: Number of repetitions, multiplied by 5 (default: 5 * self.reps)
- `rounds`: Number of software averages (default: self.rounds)
- `df`: Frequency detuning in MHz (default: 50)
- `length`: Pulse length in μs (default: 10)
- `gain`: Pulse gain (default: 0.5)
- `type`: Pulse type (default: 'const')
- `max_wait`: Maximum wait time for 'end_wait' sweep (default: 10)
- `max_gain`: Maximum gain for 'gain' sweep (default: 1)
- `expt_type`: Type of sweep, 'end_wait' or 'gain' (default: 'end_wait')

**Example**:
```python
# Check end wait time
readout_check = meas.ReadoutCheck(cfg_dict, qi=0, params={'expt_type': 'end_wait'})

# Check gain
readout_check_gain = meas.ReadoutCheck(cfg_dict, qi=0, params={'expt_type': 'gain'})
```

#### T1 Experiment

**File**: `experiments/single_qubit/t1.py`

**Description**: Measures the energy relaxation time (T1) by exciting the qubit to the |1⟩ state and measuring its decay over time. T1 (energy relaxation time) is measured by:
1. Exciting the qubit to |1⟩ state with a π pulse
2. Waiting for a variable delay time
3. Measuring the qubit state
4. Fitting the decay of the |1⟩ state population to an exponential function

The module provides several experiment classes:
- `T1Program`: Low-level pulse sequence implementation
- `T1Experiment`: Standard T1 measurement
- `T1_2D`: 2D T1 measurement for stability analysis

**Key Parameters**:
- `expts`: Number of wait time points (default: 60)
- `reps`: Number of repetitions for each experiment (default: 2 * self.reps)
- `rounds`: Number of software averages (default: self.rounds)
- `start`: Start wait time (μs) (default: 0)
- `span`: Total span of wait times (μs) (default: 3.7 * T1)
- `acStark`: Whether to apply AC Stark shift during wait time (default: False)
- `active_reset`: Whether to use active qubit reset
- `qubit`: List of qubit indices to measure
- `stark_freq`: Frequency of AC Stark pulse (MHz) (if acStark=True)
- `stark_gain`: Amplitude of AC Stark pulse (if acStark=True)
- `end_wait`: Additional wait time after AC Stark pulse (μs)

**Style Parameters**:
- **'fine'**: More averages for better precision
  - `rounds`: 2 * self.rounds
- **'fast'**: Fewer points for quick measurements
  - `expts`: 30

**Example**:
```python
# Standard T1 measurement
t1 = meas.T1Experiment(cfg_dict, qi=0)

# T1 with AC Stark shift
t1_stark = meas.T1Experiment(cfg_dict, qi=0, params={'acStark': True, 'stark_freq': 4900, 'stark_gain': 0.1})

# 2D T1 measurement for stability analysis
t1_2d = meas.T1_2D(cfg_dict, qi=0, params={'sweep_pts': 100})
```

The experiment fits the data to an exponential decay function and extracts the T1 time. The result can be automatically updated in the configuration file using the `update()` method.

#### T2 Ramsey Experiment

**File**: `experiments/single_qubit/t2.py`

**Description**: Measures the phase coherence time (T2*) using a Ramsey sequence (π/2 - wait - π/2). T2 is a measure of how long a qubit maintains phase coherence in the x-y plane of the Bloch sphere. The Ramsey protocol uses two π/2 pulses separated by a variable delay time.

The module supports three main measurement protocols:
1. **Ramsey**: Uses two π/2 pulses separated by a variable delay time
2. **Echo**: Uses two π/2 pulses with one or more π pulses in between to refocus dephasing
3. **CPMG**: Uses a sequence of π pulses to dynamically decouple the qubit from the environment

Additional features include:
- AC Stark shift measurements during Ramsey experiments
- EF transition measurements (first excited to second excited state)
- Automatic frequency error detection and correction

**Key Parameters**:
- `experiment_type`: "ramsey", "echo", or "cpmg" (default: "ramsey")
- `expts`: Number of wait time points (default: 100)
- `reps`: Number of repetitions for each experiment (default: 2 * self.reps)
- `rounds`: Number of software averages (default: self.rounds)
- `start`: Start wait time (μs) (default: 0.01)
- `span`: Total span of wait times (μs) (default: 3 * T2r for Ramsey, 3 * T2e for Echo)
- `ramsey_freq`: Frequency detuning for phase advancement (MHz) (default: "smart", which sets to 1.5/T2)
- `checkEF`: Whether to check the e-f transition (default: False)
- `acStark`: Whether to apply AC Stark shift during wait time (Ramsey only) (default: False)
- `active_reset`: Whether to use active qubit reset
- `num_pi`: Number of π pulses for Echo/CPMG (default: 1 for 'echo', 0 for 'ramsey')
- `stark_freq`: Frequency of AC Stark pulse (MHz) (if acStark=True)
- `stark_gain`: Amplitude of AC Stark pulse (if acStark=True)

**Style Parameters**:
- **'fine'**: More averages for better precision
  - `rounds`: 2 * self.rounds
- **'fast'**: Fewer points for quick measurements
  - `expts`: 50

**Example**:
```python
# Standard Ramsey experiment
t2r = meas.T2Experiment(cfg_dict, qi=0)

# Ramsey with specific detuning frequency
t2r_custom = meas.T2Experiment(cfg_dict, qi=0, params={'ramsey_freq': 0.5})

# Ramsey for EF transition
t2r_ef = meas.T2Experiment(cfg_dict, qi=0, params={'checkEF': True})

# Echo experiment
t2e = meas.T2Experiment(cfg_dict, qi=0, params={'experiment_type': 'echo'})

# Echo with multiple π pulses (CPMG sequence)
t2e_cpmg = meas.T2Experiment(cfg_dict, qi=0, params={'experiment_type': 'echo', 'num_pi': 3})
```

The experiment fits the data to a decaying sinusoid and extracts both the T2 time and the frequency error. The frequency error can be used to correct the qubit frequency in the configuration. For Echo experiments, the T2E time is typically longer than the T2* time measured by the Ramsey experiment because the echo sequence refocuses dephasing caused by low-frequency noise.

#### Single Shot Readout

**File**: `experiments/single_qubit/single_shot.py`

**Description**: Performs single-shot readout measurements to characterize readout fidelity and optimize readout parameters.

**Key Parameters**:
- `shots`: Number of single-shot measurements
- `check_e`: Whether to check the excited state
- `check_f`: Whether to check the second excited state
- `active_reset`: Whether to use active reset

**Example**:
```python
shot = meas.HistogramExperiment(cfg_dict, qi=0, params={'shots': 20000})
```



#### Single Shot Optimization

**File**: `experiments/single_qubit/single_shot_opt.py`

**Description**: Optimizes readout parameters (frequency, gain, length) to maximize readout fidelity.

**Key Parameters**:
- `expts_f`: Number of frequency points
- `expts_gain`: Number of gain points
- `expts_len`: Number of readout length points
- `span_f`: Frequency span (MHz)
- `span_gain`: Gain span
- `span_len`: Readout length span (μs)

**Example**:
```python
shotopt = meas.SingleShotOptExperiment(cfg_dict, qi=0, params={'expts_f': 5, 'expts_gain': 7, 'expts_len': 5})
```

#### Stark Spectroscopy

**File**: `experiments/single_qubit/stark_spectroscopy.py`

**Description**: Measures the AC Stark shift of the qubit frequency as a function of drive power.

**Key Parameters**:
- Similar to QubitSpec, with additional Stark-specific parameters

**Example**:
```python
stark_spec = meas.StarkSpec(cfg_dict, qi=0, style='medium')
```

#### T1 Continuous Measurement

**File**: `experiments/single_qubit/t1_cont.py`

**Description**: This experiment performs continuous T1 measurements to monitor the stability of the qubit's relaxation time. It repeatedly runs a T1 experiment and plots the results over time.

**Key Parameters**:
- `t1_expts`: Number of T1 experiments to run
- `t1_reps`: Number of repetitions for each T1 experiment
- `t1_rounds`: Number of software averages for each T1 experiment

**Example**:
```python
t1_cont = meas.T1Cont(cfg_dict, qi=0, params={'t1_expts': 100})
```

#### T1 Stark Measurement

**File**: `experiments/single_qubit/t1_stark.py`

**Description**: This experiment measures the T1 relaxation time in the presence of a Stark drive applied to the readout resonator. It is useful for characterizing the impact of the Stark drive on qubit coherence.

**Key Parameters**:
- `stark_gain`: Amplitude of the Stark pulse
- `stark_freq`: Frequency of the Stark pulse (MHz)
- `wait_time`: Wait time for the T1 measurement (μs)

**Example**:
```python
t1_stark = meas.T1Stark(cfg_dict, qi=0, params={'stark_gain': 0.1, 'stark_freq': 4900})
```

#### T1 Stark Complex Measurement

**File**: `experiments/single_qubit/t1_stark_complex.py`

**Description**: This experiment performs more complex T1 measurements with multiple Stark tones and advanced pulse sequences. It enables detailed characterization of how different Stark drive configurations affect qubit relaxation, including multi-tone interference effects and complex drive patterns.

**Key Parameters**:
- Similar to T1 Stark, with additional parameters for multiple tone configurations
- Supports complex pulse sequencing for advanced Stark shift studies
- Enables investigation of multi-photon processes and higher-order effects

**Example**:
```python
t1_stark_complex = meas.T1StarkComplex(cfg_dict, qi=0, params={'stark_gain': 0.1, 'stark_freq': 4900})
```

#### T2 Ramsey Stark Measurement

**File**: `experiments/single_qubit/t2_ramsey_stark.py`

**Description**: This experiment measures the T2 Ramsey coherence time while applying a Stark drive to the readout resonator. It helps to understand how the Stark drive affects qubit dephasing.

**Key Parameters**:
- `stark_gain`: Amplitude of the Stark pulse
- `stark_freq`: Frequency of the Stark pulse (MHz)
- `ramsey_freq`: Ramsey frequency for phase advancement (MHz)

**Example**:
```python
t2_ramsey_stark = meas.T2RamseyStark(cfg_dict, qi=0, params={'stark_gain': 0.1, 'stark_freq': 4900})
```

#### Active Reset

**File**: `experiments/single_qubit/active_reset.py`

**Description**: This experiment does not do any fitting. It is designed to test the memory and the active reset.

**Key Parameters**:
- `shots`: Number of shots per experiment
- `check_e`: Whether to test the e state blob (true if unspecified)
- `check_f`: Whether to also test the f state blob
- `active_reset`: Boolean to add active reset
- `read_wait`: Wait time between measurements in μs

**Example**:
```python
reset_exp = meas.RepMeasExperiment(cfg_dict, qi=0, params={'shots': 20000, 'active_reset': True})
```

### Two Qubit Experiments

#### Two-Qubit Rabi

**File**: `experiments/two_qubit/rabi_2q.py`

**Description**: Performs Rabi oscillations on two qubits simultaneously.

**Key Parameters**:
- Similar to single-qubit Rabi, but with a list of two qubits

**Example**:
```python
rabi_2q = meas.Rabi_2Q(cfg_dict, qi=[0, 1])
```

#### Two-Qubit T1

**File**: `experiments/two_qubit/t1_2q.py`

**Description**: Measures T1 relaxation times for two qubits simultaneously.

**Key Parameters**:
- Similar to single-qubit T1, but with a list of two qubits

**Example**:
```python
t1_2q = meas.T1_2Q(cfg_dict, qi=[0, 1])
```

#### Two-Qubit T1 Continuous

**File**: `experiments/two_qubit/t1_2q_cont.py`

**Description**: This experiment continuously measures the T1 relaxation time for two qubits simultaneously. It is useful for monitoring the stability of T1 for both qubits over time.

**Key Parameters**:
- `shots`: Number of shots per experiment
- `reps`: Number of repetitions for each experiment
- `rounds`: Number of software averages
- `wait_time`: Wait time for the T1 measurement (μs)
- `active_reset`: Whether to use active reset
- `final_delay`: Delay between measurements (μs)
- `readout`: Readout pulse length (μs)
- `n_g`: Number of ground state measurements
- `n_e`: Number of excited state measurements
- `n_t1`: Number of T1 measurements

**Example**:
```python
t1_2q_cont = meas.T1Cont2QExperiment(cfg_dict, qi=[0, 1], params={'shots': 50000})
```

### Flux Experiments

**Directory**: `experiments/flux/` (plus `T1FastFlux` in `experiments/single_qubit/t1.py`)

Experiments that pulse the qubit flux line with a fast QICK DAC and sweep measurements across flux. They share common sweep parameters (`gain_start`, `gain_stop`, `expts_gain`, `direction`, `freq_span`, `lin_freq`) through the `QickFluxSweep` base class in `experiments/general/qick_flux_experiment.py`. See [README_fast_flux.md](README_fast_flux.md) for a full guide, including the flux-line predistortion pipeline.

| Experiment | File | Measures |
|------------|------|----------|
| `QubitSpecFlux` | `pulse_probe_spectroscopy_flux.py` | Qubit spectroscopy vs DC bias (2D) |
| `QubitSpecFastFlux` | `pulse_probe_spectroscopy_flux.py` | Qubit frequency vs fast flux gain; fits the flux model |
| `QubitSpecFluxSweep` | `pulse_probe_spectroscopy_flux.py` | Spectroscopy vs an arbitrary parameter (e.g. `flux_lead_time` for step-response characterization) |
| `ResSpecFastFlux` | `resonator_spectroscopy_flux.py` | Resonator frequency vs flux gain |
| `RabiFluxExperiment` | `rabi_flux.py` | Rabi oscillations with concurrent flux pulse |
| `HistogramFluxExperiment` | `single_shot_flux.py` | Single-shot readout at an operating flux point |
| `T1FastFlux` | `../single_qubit/t1.py` | Full T1 decay curves at each flux point |
| `T1FastFluxLoop` | `t1_fastflux_loop.py` | Fast adaptive single-point T1 vs flux |
| `T1ContFluxExperiment` | `t1_cont_flux.py` | T1 vs flux with inline g/e calibration |
| `T1Predist`, `QubitSpecPredist` | `t1_predistorted.py`, `qubit_spec_predistorted.py` | T1 / spectroscopy with predistorted flux envelopes |
| `T1PredistFastFlux` | `t1_predist_fastflux.py` | T1 vs flux with predistorted pulses |

The `*Repeated` variants (`T1FastFluxLoopRepeated`, `T1ContFluxRepeated`, `T1PredistFastFluxRepeated`) rerun their sweep continuously to build T1(frequency, time) heatmaps with live plotting and incremental CSV saves.

**Example**:
```python
t1_loop = meas.T1FastFluxLoop(cfg_dict, qi=2, params={
    'gain_start': 0.16, 'gain_stop': -0.28, 'expts_gain': 300,
    'lin_freq': True, 'direction': 'neg'})
```

## Common Experiment Workflows

### Resonator and Qubit Characterization

1. **Time of Flight Calibration**:
   ```python
   tof = meas.ToFCalibrationExperiment(cfg_dict=cfg_dict, qi=0)
   ```

2. **Resonator Spectroscopy**:
   ```python
   # Coarse scan to find resonators
   rspecc = meas.ResSpec(cfg_dict, qi=0, style='coarse', params={'start': 5000, 'span': 1700})
   
   # Fine scan for precise measurement
   rspec = meas.ResSpec(cfg_dict, qi=0, params={'span': 5})
   
   # Update configuration with resonator frequency
   rspec.update()
   ```

3. **Resonator Power Spectroscopy**:
   ```python
   rpowspec = meas.ResSpecPower(cfg_dict, qi=0, params={'rng': 300, 'span': 5})
   ```

4. **Qubit Spectroscopy**:
   ```python
   # Find qubit frequency
   qspec = meas.QubitSpec(cfg_dict, qi=0, style='medium')
   
   # Update configuration with qubit frequency
   if qspec.status:
       auto_cfg = config.update_qubit(cfg_path, 'f_ge', qspec.data["best_fit"][2], 0)
   ```

### Qubit Coherence Measurements

1. **Rabi Oscillations**:
   ```python
   # Amplitude Rabi to calibrate pi pulse
   amp_rabi = meas.RabiExperiment(cfg_dict, qi=0)
   
   # Update pi pulse amplitude
   config.update_qubit(cfg_path, ('pulses', 'pi_ge', 'gain'), amp_rabi.data['pi_length'], 0)
   ```

2. **T1 Measurement**:
   ```python
   t1 = meas.T1Experiment(cfg_dict, qi=0)
   
   # Update T1 value
   t1.update()
   ```

3. **T2 Ramsey Measurement**:
   ```python
   t2r = meas.T2Experiment(cfg_dict, qi=0)
   
   # Update qubit frequency and T2 value
   if t2r.status:
       config.update_qubit(cfg_path, 'f_ge', t2r.data['new_freq'], 0)
       config.update_qubit(cfg_path, 'T2r', t2r.data['best_fit'][3], 0)
   ```

4. **T2 Echo Measurement**:
   ```python
   t2e = meas.T2Experiment(cfg_dict, qi=0, params={'experiment_type': 'echo'})
   
   # Update T2E value
   if t2e.status:
       config.update_qubit(cfg_path, 'T2e', t2e.data['best_fit'][3], 0)
   ```

### Readout Optimization

1. **Single Shot Measurement**:
   ```python
   shot = meas.HistogramExperiment(cfg_dict, qi=0, params={'shots': 20000})
   
   # Update readout parameters
   shot.update()
   ```

2. **Single Shot Optimization**:
   ```python
   shotopt = meas.SingleShotOptExperiment(cfg_dict, qi=0, params={'expts_f': 5, 'expts_gain': 7, 'expts_len': 5})
   
   # Update optimized readout parameters
   shotopt.update()
   ```

## Live Plotting with Visdom

Real-time visualization for 2D experiments using Visdom. Useful for monitoring long-running experiments.

### Setup

Start the Visdom server before running experiments:
```bash
python -m visdom.server
```
Access the web interface at `http://localhost:8097` to view live plots.

### Usage

Enable live plotting in `QickExperiment2DSimple` with `live_plot=True`:

```python
# Basic live plotting
exp_2d = QickExperiment2DSimple(cfg_dict=cfg_dict, live_plot=True, qi=0)
y_sweep = [{"var": "wait_time", "pts": np.linspace(0, 50, 25)}]
data = exp_2d.acquire(y_sweep)

# Keep plots open after experiment
exp_2d = QickExperiment2DSimple(cfg_dict=cfg_dict, live_plot=True, auto_close_visdom=False, qi=0)

# T1 2D with live plotting
t1_2d = T1_2D(cfg_dict=cfg_dict, qi=0, live_plot=True, params={'sweep_pts': 100})
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `live_plot` | Enable real-time visualization | `False` |
| `auto_close_visdom` | Auto-close plots when done | `True` |

### Features

- **Real-time heatmaps**: Shows amplitude, phase, I, and Q quadratures
- **Four-panel display**: Updates after each sweep point
- **Automatic cleanup**: Plots close when experiments finish (if enabled)

### Troubleshooting

- **Server not running**: Run `python -m visdom.server` first
- **Plots not updating**: Refresh browser or check console for errors
- **Memory issues**: Use `auto_close_visdom=True` for long experiments
- **Test setup**: Run `python test_visdom_cleanup.py` to verify functionality

## Parameter Reference

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `reps` | Number of repetitions for each experiment point | Varies by experiment |
| `rounds` | Number of software averages | Varies by experiment |
| `expts` | Number of experiment points | Varies by experiment |
| `start` | Start value for the swept parameter | Varies by experiment |
| `span` | Span of the swept parameter | Varies by experiment |
| `gain` | Pulse amplitude (DAC units) | Varies by experiment |
| `length` | Pulse length (μs) | Varies by experiment |
| `frequency` | Pulse frequency (MHz) | Varies by experiment |
| `active_reset` | Whether to use active reset | False |
| `qubit` | List of qubit indices | [qi] |
| `qubit_chan` | Qubit channel for readout | From config |

### Style Parameters Summary

| Experiment | Style | Description | Key Changes |
|------------|-------|-------------|-------------|
| **ResSpec** | 'coarse' | Wide frequency scan | start: 6000 MHz, span: 500 MHz, expts: 5000 |
| | 'fine' | Narrow scan around resonator | center: from config, span: 5 MHz, expts: 220 |
| **QubitSpec** | 'huge' | Very wide scan, high power | span: 1500 MHz, gain: 80×low_gain×spec_gain |
| | 'coarse' | Wide scan, medium power | span: 500 MHz, gain: 20×low_gain×spec_gain |
| | 'medium' | Medium scan, low power | span: 50 MHz, gain: 5×low_gain×spec_gain |
| | 'fine' | Narrow scan, very low power | span: 5 MHz, gain: low_gain×spec_gain |
| **Rabi** | 'fine' | More averages | rounds: 2×self.rounds |
| | 'fast' | Fewer points | expts: 25 |
| **T1** | 'fine' | More averages | rounds: 2×self.rounds |
| | 'fast' | Fewer points | expts: 30 |
| **T2** | 'fine' | More averages | rounds: 2×self.rounds |
| | 'fast' | Fewer points | expts: 50 |

### Experiment-Specific Parameters

Each experiment has additional parameters specific to its functionality. Refer to the experiment class documentation for details.

### Units and Conventions

- **Time**: All times are in microseconds (μs)
- **Frequency**: All frequencies are in megahertz (MHz)
- **Gain**: All gains are normalized (0 to 1)
- **Phase**: All phases are in degrees unless otherwise specified
- **Power**: Power levels are specified as DAC units or normalized gains

### Configuration File Updates

Many experiments provide an `update()` method to automatically update the configuration file with measured parameters:

```python
# Example: Update resonator frequency after measurement
rspec.update(freq=True)

# Example: Update T1 time and related parameters
t1.update(rng_vals=[1, 500], first_time=True)

# Example: Update qubit frequency after Ramsey measurement
if t2r.status:
    config.update_qubit(cfg_path, 'f_ge', t2r.data['new_freq'], 0)
```

This automated updating helps maintain accurate configuration parameters as you characterize and tune your qubits.
