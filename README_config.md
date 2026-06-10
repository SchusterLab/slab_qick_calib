# Manual for `config.py`

This manual explains the parameters and functions in the `config.py` module and how they are used in various quantum experiments with superconducting qubits.

## Overview

The `config.py` module provides a comprehensive set of functions to initialize, load, save, and update configuration files for quantum experiments with superconducting qubits. It creates YAML configuration files with parameters for qubits, readout resonators, and hardware settings.

The configuration structure has three main sections:
- **`device.qubit`**: Qubit parameters (frequencies, coherence times, pulse settings)
- **`device.readout`**: Readout resonator parameters (frequency, gain, timing)
- **`hw.soc`**: Hardware configuration (ADC/DAC channels and settings)

Configuration files are stored in YAML format and loaded as `AttrDict` objects, allowing dictionary keys to be accessed as attributes (e.g., `cfg.device.qubit.T1`). This configuration system serves as the central hub that connects hardware settings, qubit properties, and experimental parameters, enabling automated tuning and optimization of quantum devices.

## Configuration Management Functions

### `load(file_name)`
Loads a YAML configuration file and returns it as an `AttrDict`, which allows accessing dictionary keys as attributes.

### `save(cfg, file_name, reload=True)`
Saves a configuration object to a YAML file. If `reload` is `True`, it reloads the file after saving and returns the updated configuration.

### `save_copy(file_name)`
Saves a copy of a configuration file with a timestamp in the filename to prevent overwriting previous configurations.

### `update_config(file_name, path, field, value, index=None, ...)`
A general-purpose function to update a specific value in the configuration file. It can update any part of the configuration, including nested fields and array elements.

**Update Helper Functions:**
- `update_qubit(file_name, field, value, qubit_i, ...)`: A wrapper for `update_config` to update qubit parameters.
- `update_readout(file_name, field, value, qubit_i, ...)`: A wrapper for `update_config` to update readout parameters.
- `update_stark(file_name, field, value, qubit_i, ...)`: A wrapper for `update_config` to update Stark shift parameters.
- `update_lo(file_name, field, value, qi, ...)`: A wrapper for `update_config` to update local oscillator parameters.

### `save_single_qubit_config(file_name, qubit_index, new_file_name)`
Extracts the configuration for a single qubit and saves it to a new file. This is useful for running single-qubit experiments without loading the full multi-qubit configuration.

## Configuration Initialization Functions

### `init_config(file_name, num_qubits, type="full", t1=50, aliases="Qick001")`
Initializes a standard configuration file for multi-qubit experiments.

| Parameter | Description | Usage |
|-----------|-------------|-------|
| `file_name` | Path to the output YAML configuration file. | Used to save the configuration |
| `num_qubits`| Number of qubits to configure. | Determines the length of parameter arrays |
| `type` | Type of readout DAC output (`full`, `mux`, `int`). | Used in hardware configuration for readout DACs |
| `t1` | Default T1 relaxation time in μs, used to set initial T1, T2, and readout delay values. | Used to initialize qubit T1, T2 values and readout delay |
| `aliases` | Identifier for the System-on-Chip (SoC). | Used in hardware configuration |

### `init_config_res(file_name, num_qubits, type="full", aliases="Qick001")`
Initializes a configuration file specifically for resonator experiments. This configuration is a simplified version of the standard config, containing only the parameters necessary for resonator characterization.

### `init_model_config(file_name, num_qubits)`
Initializes a model configuration file for storing theoretical and fitting parameters related to the quantum device. This is used for more advanced analysis and modeling.

**Parameters:**
- `file_name`: Path where the configuration will be saved (a new file with "_model" suffix is created)
- `num_qubits`: Number of qubits to configure

**Returns:** The YAML configuration string that was saved

**Note:** Creates a new file with "_model" appended before the extension (e.g., "config.yml" → "config_model.yml"). All parameters are initialized to None and filled during analysis.

### `init_stark_section(file_name, num_qubits)`
Adds a Stark shift section to an existing configuration file. Adds parameters for characterizing AC Stark shifts from drive tones, useful for experiments measuring and compensating Stark effects.

**Parameters:**
- `file_name`: Path to the configuration file
- `num_qubits`: Number of qubits (determines array lengths)

**Returns:** The updated configuration object

**Note:** Parameters represent Stark shift coefficients:
- `q`, `l`, `o`: Quadratic, linear, offset terms
- `qneg`, `lneg`, `oneg`: Negative detuning versions
- `f`, `fneg`: Frequency shifts at positive/negative detuning

### `add_pulse(file_name, pulse_name, pulse_type="gauss")`
Adds a new pulse definition to the configuration under `device.qubit.pulses`. Useful for defining custom pulse sequences beyond the default `pi_ge` and `pi_ef` pulses.

**Parameters:**
- `file_name`: Path to the configuration file
- `pulse_name`: Name for the new pulse (e.g., "pi_half", "echo", "drag")
- `pulse_type`: Type of pulse shape - "gauss", "const", or "flat_top" (default: "gauss")

**Returns:** The updated configuration object

**Default pulse parameters:**
- **gauss**: `gain`=0.15, `sigma`=0.1, `sigma_inc`=4
- **const**: `gain`=0.15, `length`=1
- **flat_top**: `gain`=0.15, `length`=0.1, `ramp_sigma`=0.02, `ramp_sigma_inc`=4

## RF Board Configuration Functions

For systems with an on-board RF front end (filters, attenuators, DC bias), four functions manage the per-qubit RF board settings under `hw.soc`. See [README_rfboard.md](README_rfboard.md) for the full workflow.

### `init_rfboard_section(file_name, num_qubits)`
Adds `filter_fc`, `filter_bw`, and `filter_type` arrays to `adcs.readout`, `dacs.readout`, and (if present) `dacs.qubit`, with bypass defaults. Idempotent — existing values are preserved.

### `init_rfboard_active_section(file_name)`
Adds the `hw.soc.rfboard_active` section that tracks the last-programmed state of each physical RF channel. Idempotent.

### `update_rfboard(file_name, path, field, value, index=None, ...)`
Updates a single RF board parameter, e.g. `update_rfboard(cfg_path, "dacs.qubit", "filter_bw", 500, index=qi)`. Convenience wrapper for `update_config` targeting `hw.soc.{path}`.

### `update_rfboard_active(file_name, channel_type, ch, settings)`
Persists a channel's active RF state to disk. Normally called internally by the `helpers/rfboard.py` functions rather than directly.

## Configuration Structure

The configuration is organized into three main sections:

1. `device.qubit`: Qubit parameters
2. `device.readout`: Readout resonator parameters
3. `hw.soc`: Hardware configuration

## Qubit Parameters (`device.qubit`)

### Coherence Times

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `T1` | Energy relaxation time (μs). | Used in `t1.py` to set the span of wait times for T1 measurements. Also used in `rabi.py` when calculating pulse lengths. |
| `T2r` | Ramsey dephasing time (μs). | Used in T2 Ramsey experiments to set appropriate measurement timescales. |
| `T2e` | Echo dephasing time (μs). | Used in T2 Echo experiments to set appropriate measurement timescales. |

### Qubit Frequencies

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `f_ge` | Ground to excited state transition frequency (MHz). | Used in `pulse_probe_spectroscopy.py` to set the center frequency for spectroscopy. Used in `rabi.py` to set the frequency of the π pulse. |
| `f_ef` | Excited to second excited state transition frequency (MHz). | Used in `pulse_probe_spectroscopy.py` when `checkEF=True` to probe the ef transition. |
| `kappa` | Qubit linewidth (MHz) (informational). | Just informative, not used anywhere. |

### Qubit Pulses (`device.qubit.pulses`)

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `pi_ge.gain`| Amplitude of the π pulse for the g→e transition. | Used in `rabi.py` and other experiments that apply π pulses to the qubit. |
| `pi_ge.sigma`| Width (std dev) of the Gaussian π pulse for g→e. | Used in `rabi.py` to set the pulse width. |
| `pi_ge.sigma_inc`| Pulse length in units of sigma. | Used to calculate the total pulse length from sigma. |
| `pi_ge.type` | Pulse shape type (e.g., "gauss"). | Determines the pulse shape in experiments. |
| `pi_ef.gain`| Amplitude of the π pulse for the e→f transition. | Used when manipulating the second excited state. |
| `pi_ef.sigma`| Width (std dev) of the Gaussian π pulse for e→f. | Used to set the pulse width for e→f transitions. |
| `pi_ef.sigma_inc`| Pulse length in units of sigma. | Used to calculate the total pulse length from sigma. |
| `pi_ef.type` | Pulse shape type (e.g., "gauss"). | Determines the pulse shape in experiments. |

### Other Qubit Parameters

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `spec_gain` | Gain scaling factor for spectroscopy for qubit to qubit variation. | Used in `pulse_probe_spectroscopy.py` to set appropriate pulse amplitudes. |
| `pop` | Thermal population (informational). | Used in analysis of qubit measurements. |
| `temp` | Qubit temperature (informational). | Used in analysis of qubit measurements. |
| `tuned_up` | Boolean flag indicating if the qubit is tuned up. | Used in experiments to determine whether to show Amps/Q in addition to I. Set by single_shot, depending on fidelity. |
| `low_gain` | Minimum gain for spectroscopy scans. | Used in `pulse_probe_spectroscopy.py` to set the minimum gain for pulses. |
| `max_gain` | Maximum gain for qubit control pulses. | Used in `pulse_probe_spectroscopy.py` and `rabi.py` to set the maximum gain for pulses. |

## Readout Parameters (`device.readout`)

### Resonator Properties

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `frequency` | Readout resonator frequency (MHz). | Used in `resonator_spectroscopy.py` to set the center frequency for spectroscopy. |
| `gain` | Readout pulse amplitude. | Used in all experiments that perform qubit readout. |
| `lamb` | Lamb shift (informational). | Used in analysis of resonator spectroscopy data. |
| `chi` | Dispersive shift (informational). | Used in analysis of resonator spectroscopy data. |
| `kappa` | Resonator linewidth (MHz). | Used in `resonator_spectroscopy.py` to determine appropriate frequency spans. |
| `qe` | External quality factor (informational). | Used in analysis of resonator spectroscopy data. |
| `qi` | Internal quality factor (informational). | Used in analysis of resonator spectroscopy data. |

### Readout Settings

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `phase` | Phase rotation for the readout signal. | Used in all experiments to correctly rotate readout so that signal is in I quadrature. |
| `readout_length`| Duration of the readout pulse (μs). | Used in all experiments that perform qubit readout. |
| `threshold` | State discrimination threshold. | Used for active reset. |
| `fidelity` | Readout fidelity (informational). | Used in analysis of readout performance. |
| `tm` | Measurement time / T1 fit from single shot (informational). | Used in analysis of readout performance. |
| `sigma` | Width (noise) of the readout histogram (informational). | Used in analysis of readout performance. |
| `rescale` | Flag to rescale readout data. | Used in readout data processing. |

### Readout Timing

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `trig_offset`| Trigger offset for readout. | Used in timing of readout pulses. |
| `final_delay`| Delay after readout before the next experiment (μs), typically set to `6 * T1`. | Used in all experiments to ensure qubit returns to ground state. Set to 6*T1 by default. |
| `active_reset`| Flag to enable active qubit reset. | Used in experiments that support active reset to improve experiment speed. |
| `reset_e` | Parameter for active reset of the excited state. | Used in active reset protocols. |
| `reset_g` | Parameter for active reset of the ground state. | Used in active reset protocols. |

### Readout Averaging

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `reps` | Per-qubit multiplier for the number of repetitions. | Allow qubit to qubit variation in number of reps used (base is reps[qi]*reps_base). Used in all experiments to set the number of repetitions. |
| `rounds` | Per-qubit multiplier for the number of software averages. | Allow qubit to qubit variation in number of rounds used. Used in all experiments to set the number of software averages. |
| `reps_base` | Base number of repetitions for the entire device. | Used to calculate appropriate repetition counts. |
| `rounds_base`| Base number of software averages for the entire device. | Used to calculate appropriate averaging counts. |
| `max_gain` | Maximum gain for the readout pulse. | Used to limit readout pulse amplitude. |

## Hardware Configuration (`hw.soc`)

### ADC Configuration

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `adcs.readout.ch`| ADC channel for readout. | Used in all experiments to specify which ADC channel to use for readout. |
| `adcs.readout.type`| ADC type (e.g., "dyn"). | Used in ADC configuration. |

### DAC Configuration

| Parameter | Description | Usage in Experiments |
|-----------|-------------|---------------------|
| `dacs.qubit.ch` | DAC channel for qubit control. | Used in all experiments to specify which DAC channel to use for qubit control. |
| `dacs.qubit.nyquist`| Nyquist zone for the qubit DAC. | Used in signal generation for qubit control. 1 for frequencies < fs/2, 2 for frequencies above. |
| `dacs.qubit.type` | Type of qubit DAC output (`full`, `mux`, `int`). | Used in signal generation for qubit control. |
| `dacs.readout.ch`| DAC channel for readout. | Used in all experiments to specify which DAC channel to use for readout. |
| `dacs.readout.nyquist`| Nyquist zone for the readout DAC. | Used in signal generation for readout. 1 for frequencies < fs/2, 2 for frequencies above. |
| `dacs.readout.type` | Type of readout DAC output (`full`, `mux`, `int`). | Used in signal generation for readout. |

## Usage Examples

### Resonator Spectroscopy

In `resonator_spectroscopy.py`, the resonator frequency is swept to find the resonance:

```python
# From ResSpec class in resonator_spectroscopy.py
params_def["center"] = self.cfg.device.readout.frequency[qi]
params_def["expts"] = 220
params_def["span"] = 5
```

The experiment uses the configured resonator frequency as the center for the frequency sweep.

### Qubit Spectroscopy

In `pulse_probe_spectroscopy.py`, the qubit frequency is swept to find the qubit transition:

```python
# From QubitSpec class in pulse_probe_spectroscopy.py
if params['checkEF']:
    params_def["start"] = self.cfg.device.qubit.f_ef[qi] - params["span"] / 2
else:
    params_def["start"] = self.cfg.device.qubit.f_ge[qi] - params["span"] / 2
```

The experiment uses either the g→e or e→f transition frequency as the center for the frequency sweep.

### Rabi Oscillations

In `rabi.py`, the amplitude or length of a qubit drive pulse is swept to observe Rabi oscillations:

```python
# From RabiExperiment class in rabi.py
if params['checkEF']:
    cfg_qub = self.cfg.device.qubit.pulses.pi_ef
    params_def["freq"] = self.cfg.device.qubit.f_ef[qi]
else:
    cfg_qub = self.cfg.device.qubit.pulses.pi_ge
    params_def["freq"] = self.cfg.device.qubit.f_ge[qi]
```

The experiment uses the configured qubit frequency and pulse parameters to generate the appropriate drive pulse.

### T1 Measurement

In `t1.py`, the delay after a π pulse is swept to measure the energy relaxation time:

```python
# From T1Experiment class in t1.py
params_def = {
    # 
    "span": 3.7 * self.cfg.device.qubit.T1[qi],  # Total span of wait times (μs), set to ~3.7*T1
    # 
}
```

The experiment uses the configured T1 time to set an appropriate span for the delay sweep.

## Updating Configuration

The `config.py` module provides functions to update the configuration based on experiment results:

```python
# From T1Experiment.update method in t1.py
def update(self, cfg_file, rng_vals=[1,500], first_time=False, verbose=True):
    qi = self.cfg.expt.qubit[0]
    if self.status: 
        config.update_qubit(cfg_file, 'T1', self.data['new_t1_i'], qi, sig=2, rng_vals=rng_vals, verbose=verbose)
        config.update_readout(cfg_file, 'final_delay', 6*self.data['new_t1_i'], qi, sig=2, rng_vals=[rng_vals[0]*10, rng_vals[1]*3], verbose=verbose)
        if first_time:
            config.update_qubit(cfg_file, 'T2r', self.data['new_t1_i'], qi, sig=2, rng_vals=[rng_vals[0], rng_vals[1]*2], verbose=verbose)        
            config.update_qubit(cfg_file, 'T2e', 2*self.data['new_t1_i'], qi, sig=2, rng_vals=[rng_vals[0], rng_vals[1]*2], verbose=verbose)
```

This allows experiments to automatically update the configuration with measured values, creating a feedback loop for qubit tuning.

## Conclusion

The `config.py` module provides a flexible and comprehensive system for managing configurations in quantum experiments. Understanding these parameters and functions is essential for effectively running, customizing, and analyzing the results of quantum experiments with this codebase. The configuration serves as the central hub that connects hardware settings, qubit properties, and experimental parameters, enabling automated tuning and optimization of quantum devices.
