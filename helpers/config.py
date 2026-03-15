"""
Configuration module for quantum experiments.

This module provides functions for loading, saving, and updating configuration files
for quantum experiments with superconducting qubits. It creates YAML configuration 
files with parameters for qubits, readout resonators, and hardware settings.

The configuration structure has three main sections:
    - device.qubit: Qubit parameters (frequencies, coherence times, pulse settings)
    - device.readout: Readout resonator parameters (frequency, gain, timing)
    - hw.soc: Hardware configuration (ADC/DAC channels and settings)

Configuration files are stored in YAML format and loaded as AttrDict objects,
allowing dictionary keys to be accessed as attributes (e.g., cfg.device.qubit.T1).
"""

import yaml
from functools import reduce
import numpy as np
from datetime import datetime

from ..exp_handling.datamanagement import AttrDict


def nested_set(dic, keys, value):
    """
    Set a value in a nested dictionary using a list of keys.
    
    This function navigates through nested dictionary levels using the provided
    keys, creating intermediate dictionaries as needed, and sets the final value.
    
    Args:
        dic: The dictionary to modify
        keys: List of keys defining the path to the target value
        value: The value to set at the target location
        
    Example:
        nested_set(cfg, ['device', 'qubit', 'T1'], 50) sets cfg['device']['qubit']['T1'] = 50
    """
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def load(file_name):
    """
    Load a YAML configuration file and return it as an AttrDict.
    
    Args:
        file_name: Path to the YAML configuration file
        
    Returns:
        AttrDict: Configuration object with dictionary keys accessible as attributes
    """
    with open(file_name, "r") as file:
        auto_cfg = AttrDict(yaml.safe_load(file))
    return auto_cfg


def save(cfg, file_name, reload=True):
    """
    Save a configuration to a YAML file.

    Args:
        cfg: Configuration object to save
        file_name: Path to save the configuration
        reload: Whether to reload the file after saving (default: True).
                This ensures the returned config matches the saved file format.

    Returns:
        AttrDict: The saved configuration if reload=True, otherwise None
    """
    # Convert to YAML format
    cfg_yaml = yaml.safe_dump(cfg.to_dict(), default_flow_style=None)

    # Write to file
    with open(file_name, "w") as modified_file:
        modified_file.write(cfg_yaml)

    # Reload if requested
    if reload:
        with open(file_name, "r") as file:
            return AttrDict(yaml.safe_load(file))
    return None


def save_copy(file_name):
    """
    Save a timestamped copy of a configuration file.
    
    Creates a backup copy with the current timestamp appended to the filename.
    Useful for preserving configuration history before making changes.

    Args:
        file_name: Path to the original configuration file

    Returns:
        AttrDict: The saved configuration
        
    Example:
        If file_name is "config.yml", creates "config_20250109_173000.yml"
    """
    # Load the configuration
    cfg = load(file_name)
    cfg_yaml = yaml.safe_dump(cfg.to_dict(), default_flow_style=None)

    # Create a new filename with timestamp
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_file_name = f"{file_name[0:-4]}_{current_time}.yml"

    # Write to the new file
    with open(new_file_name, "w") as modified_file:
        modified_file.write(cfg_yaml)

    # Reload and return
    with open(new_file_name, "r") as file:
        return AttrDict(yaml.safe_load(file))


def recursive_get(d, keys):
    """
    Get a value from a nested dictionary using a list of keys.
    
    Args:
        d: The dictionary to search
        keys: List of keys defining the path to the target value
        
    Returns:
        The value at the specified path, or empty dict if path doesn't exist
    """
    return reduce(lambda c, k: c.get(k, {}), keys, d)


def in_rng(val, rng_vals):
    """
    Clamp a value to a specified range.

    Args:
        val: The value to check
        rng_vals: A tuple/list of (min, max) values

    Returns:
        float: The value, clamped to the specified range
        
    Note:
        Prints a message if the value is clamped
    """
    if val < rng_vals[0]:
        print("Val is out of range, setting to min")
        return rng_vals[0]
    elif val > rng_vals[1]:
        print("Val is out of range, setting to max")
        return rng_vals[1]
    else:
        return val


def format_value(value, sig=4, rng_vals=None):
    """
    Format a value for storage in the configuration.
    
    Handles rounding of floats and range clamping. NaN values and non-numeric
    types (int, str, bool) are preserved without modification.

    Args:
        value: The value to format
        sig: Number of significant digits for floating point values (default: 4)
        rng_vals: Optional range limits (min, max)

    Returns:
        The formatted value with appropriate type and precision
    """
    # Skip formatting for NaN values
    if np.isnan(value):
        return value

    # Round floating point values
    if not isinstance(value, (int, str, bool)):
        value = float(round(value, sig))

    # Apply range limits if provided
    if rng_vals is not None:
        value = in_rng(value, rng_vals)

    return value


def update_config(
    file_name, path, field, value, index=None, verbose=True, sig=4, rng_vals=None
):
    """
    Update a value in a configuration file.

    This is a general-purpose update function that can update any part of the 
    configuration, including nested fields and array elements. The file is 
    automatically saved after updating.

    Args:
        file_name: Path to the configuration file
        path: Dot-separated path to the parameter section (e.g., "device.qubit", 
              "hw.soc.dacs"). Use None for top-level fields.
        field: Field name to update. Can be a string or tuple for nested fields.
        value: New value to set
        index: Optional index for array values (e.g., qubit index)
        verbose: Whether to print update information (default: True)
        sig: Number of significant digits for floating point values (default: 4)
        rng_vals: Optional range limits (min, max) to clamp the value

    Returns:
        AttrDict: The updated configuration
        
    Examples:
        update_config("cfg.yml", "device.qubit", "T1", 50.5, index=0)
        update_config("cfg.yml", "hw.soc.dacs", "ch", 3, index=1)
    """
    # Load the configuration
    cfg = load(file_name)

    # Skip if value is NaN
    if value is None or np.isnan(value):
        return cfg

    # Format the value
    value = format_value(value, sig, rng_vals)

    if path is not None:
        # Split the path into components
        path_parts = path.split(".")

        # Navigate to the target section
        section = cfg
        for part in path_parts:
            section = section[part]
    else:
        section = cfg
        
    # Update the value
    if isinstance(field, tuple):  # For nested fields
        v = recursive_get(section, field)
        old_value = v[index]
        v[index] = value
        nested_set(section, field, v)
    elif index is not None:  # For array values
        old_value = section[field][index]
        section[field][index] = value
    else:  # For scalar values
        old_value = section[field]
        section[field] = value

    # Print update information if requested
    if verbose:
        if index is not None:
            print(f"*Set cfg {path} {index} {field} to {value} from {old_value}*")
        else:
            print(f"*Set cfg {path} {field} to {value} from {old_value}*")

    # Save the updated configuration
    save(cfg, file_name)

    return cfg


def update_qubit(file_name, field, value, qubit_i, verbose=True, sig=4, rng_vals=None):
    """
    Update a qubit parameter in the configuration.
    
    Convenience wrapper for update_config that targets device.qubit parameters.

    Args:
        file_name: Path to the configuration file
        field: Qubit parameter field name (e.g., "T1", "f_ge")
        value: New value
        qubit_i: Qubit index
        verbose: Whether to print update information (default: True)
        sig: Number of significant digits (default: 4)
        rng_vals: Optional range limits (min, max)

    Returns:
        AttrDict: The updated configuration
    """
    return update_config(
        file_name, "device.qubit", field, value, qubit_i, verbose, sig, rng_vals
    )


def update_readout(
    file_name, field, value, qubit_i, verbose=True, sig=4, rng_vals=None
):
    """
    Update a readout parameter in the configuration.
    
    Convenience wrapper for update_config that targets device.readout parameters.

    Args:
        file_name: Path to the configuration file
        field: Readout parameter field name (e.g., "frequency", "gain")
        value: New value
        qubit_i: Qubit index
        verbose: Whether to print update information (default: True)
        sig: Number of significant digits (default: 4)
        rng_vals: Optional range limits (min, max)

    Returns:
        AttrDict: The updated configuration
    """
    return update_config(
        file_name, "device.readout", field, value, qubit_i, verbose, sig, rng_vals
    )


def update_stark(file_name, field, value, qubit_i, verbose=True, sig=4, rng_vals=None):
    """
    Update a Stark shift parameter in the configuration.
    
    Convenience wrapper for update_config that targets stark parameters.

    Args:
        file_name: Path to the configuration file
        field: Stark parameter field name
        value: New value
        qubit_i: Qubit index
        verbose: Whether to print update information (default: True)
        sig: Number of significant digits (default: 4)
        rng_vals: Optional range limits (min, max)

    Returns:
        AttrDict: The updated configuration
    """
    return update_config(
        file_name, "stark", field, value, qubit_i, verbose, sig, rng_vals
    )


def update_lo(file_name, field, value, qi, verbose=True, sig=4, rng_vals=None):
    """
    Update a local oscillator parameter in the configuration.
    
    Convenience wrapper for update_config that targets hw.soc.lo parameters.

    Args:
        file_name: Path to the configuration file
        field: LO parameter field name
        value: New value
        qi: Qubit/channel index
        verbose: Whether to print update information (default: True)
        sig: Number of significant digits (default: 4)
        rng_vals: Optional range limits (min, max)

    Returns:
        AttrDict: The updated configuration
    """
    return update_config(
        file_name, "hw.soc.lo", field, value, qi, verbose, sig, rng_vals
    )


def init_config(file_name, num_qubits, type="full", t1=50, aliases="Qick001", ip="", flux=False):
    """
    Initialize a configuration file for quantum experiments with qubits.
    
    Creates a complete configuration with default values for multi-qubit experiments,
    including qubit parameters, readout settings, and hardware configuration.

    Args:
        file_name: Path where the configuration will be saved
        num_qubits: Number of qubits to configure
        type: Type of readout DAC output - "full" (full bandwidth), "mux" (multiplexed),
              or "int" (interpolated) (default: "full")
        t1: Default T1 relaxation time in μs. Used to initialize T1, T2 values and 
            final_delay (default: 50)
        aliases: Identifier for the System-on-Chip (SoC) (default: "Qick001")
        ip: IP address for the device (default: "")
        flux: Whether to include flux control hardware and qubit flux parameters
              (default: False). When True, adds dacs.flux section and qubit fields
              f_ge_max, sweet_spot_dc, sweet_spot_ac.

    Returns:
        str: The YAML configuration string that was saved
        
    Note:
        - T2r (Ramsey) is initialized to t1
        - T2e (Echo) is initialized to 2*t1
        - final_delay is initialized to 6*t1
        - All qubit frequencies start at 4000 MHz (ge) and 3800 MHz (ef)
        - All readout frequencies start at 7000 MHz
    """

    # Create a helper function to initialize arrays
    def init_array(value, length=num_qubits):
        """Create an array with the same value repeated."""
        return [value] * length

    # Initialize the configuration structure
    device = {"qubit": {"pulses": {"pi_ge": {}, "pi_ef": {}}}, "readout": {}}

    # Qubit coherence parameters
    device["qubit"].update(
        {
            "T1": init_array(t1),
            "T2r": init_array(t1),
            "T2e": init_array(2 * t1),
        }
    )

    # Qubit frequency parameters
    device["qubit"].update(
        {
            "f_ge": init_array(4000),  # Ground to excited state frequency (MHz)
            "f_ef": init_array(3800),  # Excited to second excited state frequency (MHz)
            "kappa": init_array(0),    # Qubit linewidth (MHz)
            "spec_gain": init_array(1),  # Gain scaling for spectroscopy
        }
    )

    # Qubit flux parameters (only when flux control is enabled)
    if flux:
        device["qubit"].update(
            {
                "f_ge_max": init_array(4000),       # Max ge frequency at sweet spot (MHz)
                "sweet_spot_dc": init_array(0),     # DC bias value at sweet spot (V)
                "sweet_spot_ac": init_array(0),     # AC flux gain at sweet spot
            }
        )

    # Qubit pulse parameters for ge and ef transitions
    for pulse_type in ["pi_ge", "pi_ef"]:
        device["qubit"]["pulses"][pulse_type].update(
            {
                "gain": init_array(0.15),      # Pulse amplitude
                "sigma": init_array(0.1),      # Gaussian width (μs)
                "sigma_inc": init_array(4),    # Pulse truncated after this number of sigma
                "type": init_array("gauss"),   # Pulse shape
                "phase": init_array(0),        # Pulse phase (degrees)
            }
        )

    # Other qubit parameters
    device["qubit"].update(
        {
            "pop": init_array(0),           # Thermal population
            "temp": init_array(0),          # Qubit temperature
            "tuned_up": init_array(False),  # Calibration status flag
            "low_gain": 0.003,              # Minimum gain for spectroscopy
            "max_gain": 1,                  # Maximum gain for control pulses
        }
    )

    # Readout frequency and gain
    device["readout"].update(
        {
            "frequency": init_array(7000),  # Readout resonator frequency (MHz)
            "gain": init_array(0.05),       # Readout pulse amplitude
        }
    )

    # Readout resonator parameters
    device["readout"].update(
        {
            "lamb": init_array(0),      # Lamb shift
            "chi": init_array(0),       # Dispersive shift
            "kappa": init_array(0.5),   # Resonator linewidth (MHz)
            "qe": init_array(0),        # External quality factor
            "qi": init_array(0),        # Internal quality factor
        }
    )

    # Readout settings
    device["readout"].update(
        {
            "phase": init_array(0),             # Phase rotation for signal (degrees)
            "readout_length": init_array(5),    # Readout pulse duration (μs)
            "threshold": init_array(10),        # State discrimination threshold
            "fidelity": init_array(0),          # Readout fidelity
            "nstark": init_array(None),         # Stark photon number per unit gain²
            "tm": init_array(0),                # Measurement time constant
            "sigma": init_array(0),             # Readout histogram width
            "rescale": init_array(False),       # Rescaling flag
        }
    )

    # Readout timing parameters
    device["readout"].update(
        {
            "trig_offset": init_array(0.5),     # Trigger timing offset (μs)
            "final_delay": init_array(t1 * 6),  # Delay before next shot (μs)
            "active_reset": init_array(False),  # Active reset flag
            "reset": init_array(3),             # Reset parameter
        }
    )

    # Readout averaging parameters
    device["readout"].update(
        {
            "reps": init_array(1),      # Per-qubit repetition multiplier
            "rounds": init_array(1),    # Per-qubit averaging multiplier
            "max_gain": 1,              # Maximum readout gain
            "reps_base": 150,           # Base repetitions for device
            "rounds_base": 1,           # Base rounds for device
        }
    )

    # Hardware configuration
    soc = {
        "adcs": {
            "readout": {
                "ch": init_array(0),              # ADC channel numbers
                "attn": init_array(0),            # ADC attenuation (from 0-30dB)
                "type": init_array("dyn"),        # ADC type
                "filter_fc": init_array(0),       # Filter center frequency (MHz, 0 = not set)
                "filter_bw": init_array(1000),    # Filter bandwidth (MHz)
                "filter_type": init_array("bypass"),  # bandpass|lowpass|highpass|bypass
            }
        },
        "dacs": {
            "qubit": {
                "ch": init_array(1),              # DAC channel for qubit control
                "nyquist": init_array(1),         # Nyquist zone (1 or 2)
                "attn1": init_array(0),           # DAC attenuation (from 0-30dB)
                "attn2": init_array(0),           # DAC attenuation (from 0-30dB)
                "type": init_array("full"),       # DAC mode
                "filter_fc": init_array(0),       # Filter center frequency (MHz, 0 = not set)
                "filter_bw": init_array(1000),    # Filter bandwidth (MHz)
                "filter_type": init_array("bypass"),  # bandpass|lowpass|highpass|bypass
            },
            "readout": {
                "ch": init_array(0),              # DAC channel for readout
                "nyquist": init_array(2),         # Nyquist zone (1 or 2)
                "attn1": init_array(0),           # DAC attenuation (from 0-30dB)
                "attn2": init_array(0),           # DAC attenuation (from 0-30dB)
                "type": init_array(type),         # DAC mode
                "filter_fc": init_array(0),       # Filter center frequency (MHz, 0 = not set)
                "filter_bw": init_array(1000),    # Filter bandwidth (MHz)
                "filter_type": init_array("bypass"),  # bandpass|lowpass|highpass|bypass
            },
        },
    }

    # Flux DAC section (only when flux control is enabled)
    if flux:
        soc["dacs"]["flux"] = {
            "ch": init_array(0),              # DAC channel for flux control
            "dc_ch": init_array(0),           # DC bias channel number
            "dc_val": init_array(0),          # DC bias value (V)
            "nyquist": init_array(1),         # Nyquist zone (1 or 2)
            "type": init_array("int"),        # DAC mode (interpolated for DC-like flux)
            "quad_a": init_array(0),          # Quadratic coeff a: freq = a*g^2 + b*g + c
            "quad_b": init_array(0),          # Quadratic coeff b
            "quad_c": init_array(0),          # Quadratic coeff c
        }

    # Build rfboard_active from unique physical channels
    rfboard_active = {}
    for active_key, ch_list, attn_defaults in [
        ("adc", soc["adcs"]["readout"]["ch"], {"attn": 0}),
        ("dac_readout", soc["dacs"]["readout"]["ch"], {"attn1": 0, "attn2": 0}),
        ("dac_qubit", soc["dacs"]["qubit"]["ch"], {"attn1": 0, "attn2": 0}),
    ]:
        rfboard_active[active_key] = {}
        for ch in set(ch_list):
            entry = {"filter_fc": 0, "filter_bw": 0, "filter_type": "unknown", "qubit": None}
            entry.update(attn_defaults)
            rfboard_active[active_key][str(ch)] = entry
    soc["rfboard_active"] = rfboard_active

    # Assemble the complete configuration
    auto_cfg = {"device": device, "hw": {"soc": soc}, "aliases": {"soc": aliases, "ip": ip}}

    # Convert to YAML and save
    cfg_yaml = yaml.safe_dump(auto_cfg, default_flow_style=None)
    with open(file_name, "w") as modified_file:
        modified_file.write(cfg_yaml)

    return cfg_yaml


def init_config_res(file_name, num_qubits, type="full", aliases="Qick001"):
    """
    Initialize a configuration file for resonator experiments.
    
    Creates a simplified configuration with only readout parameters, used for
    resonator characterization experiments without qubit control.

    Args:
        file_name: Path where the configuration will be saved
        num_qubits: Number of resonators to configure
        type: Type of readout DAC output - "full", "mux", or "int" (default: "full")
        aliases: Identifier for the System-on-Chip (SoC) (default: "Qick001")

    Returns:
        str: The YAML configuration string that was saved
    """

    # Create a helper function to initialize arrays
    def init_array(value, length=num_qubits):
        """Create an array with the same value repeated."""
        return [value] * length

    # Initialize the configuration structure
    device = {"readout": {}}

    # Readout frequency and gain
    device["readout"].update(
        {
            "frequency": init_array(7000),  # Resonator frequency (MHz)
            "gain": init_array(0.05),       # Readout pulse amplitude
        }
    )

    # Readout resonator parameters (including high/low power measurements)
    device["readout"].update(
        {
            "kappa": init_array(0.5),       # Linewidth (MHz)
            "kappa_hi": init_array(0.5),    # Linewidth at high power
            "qe": init_array(0),            # External quality factor
            "qi": init_array(0),            # Internal quality factor (overall)
            "qi_hi": init_array(0),         # Internal Q at high power
            "qi_lo": init_array(0),         # Internal Q at low power
        }
    )

    # Readout settings
    device["readout"].update(
        {
            "phase": init_array(0),             # Phase rotation (degrees)
            "readout_length": init_array(100),  # Pulse duration (μs)
            "trig_offset": init_array(0.5),     # Trigger timing (μs)
            "final_delay": init_array(50),      # Delay between shots (μs)
        }
    )

    # Readout averaging
    device["readout"].update(
        {
            "reps": init_array(1),          # Per-resonator repetition multiplier
            "rounds": init_array(1),        # Per-resonator averaging multiplier
            "max_gain": 1,                  # Maximum readout gain
            "reps_base": 30,                # Base repetitions
            "rounds_base": 1,               # Base rounds
            "phase_inc": [1140],            # Phase increment for averaging
        }
    )

    # Hardware configuration
    soc = {
        "adcs": {
            "readout": {
                "ch": init_array(0),              # ADC channels
                "attn": init_array(0),            # ADC attenuation
                "filter_fc": init_array(0),       # Filter center frequency (MHz, 0 = not set)
                "filter_bw": init_array(1000),    # Filter bandwidth (MHz)
                "filter_type": init_array("bypass"),  # bandpass|lowpass|highpass|bypass
            }
        },
        "dacs": {
            "readout": {
                "ch": init_array(0),              # DAC channels
                "nyquist": init_array(2),         # Nyquist zone
                "attn1": init_array(0),           # DAC attenuation
                "attn2": init_array(0),           # DAC attenuation
                "type": init_array(type),         # DAC mode
                "filter_fc": init_array(0),       # Filter center frequency (MHz, 0 = not set)
                "filter_bw": init_array(1000),    # Filter bandwidth (MHz)
                "filter_type": init_array("bypass"),  # bandpass|lowpass|highpass|bypass
            },
        },
    }

    # Build rfboard_active from unique physical channels
    rfboard_active = {}
    for active_key, ch_list, attn_defaults in [
        ("adc", soc["adcs"]["readout"]["ch"], {"attn": 0}),
        ("dac_readout", soc["dacs"]["readout"]["ch"], {"attn1": 0, "attn2": 0}),
    ]:
        rfboard_active[active_key] = {}
        for ch in set(ch_list):
            entry = {"filter_fc": 0, "filter_bw": 0, "filter_type": "unknown", "qubit": None}
            entry.update(attn_defaults)
            rfboard_active[active_key][str(ch)] = entry
    soc["rfboard_active"] = rfboard_active

    # Assemble the complete configuration
    auto_cfg = {"device": device, "hw": {"soc": soc}, "aliases": {"soc": aliases}}

    # Convert to YAML and save
    cfg_yaml = yaml.safe_dump(auto_cfg, default_flow_style=None)
    with open(file_name, "w") as modified_file:
        modified_file.write(cfg_yaml)

    return cfg_yaml


def save_single_qubit_config(file_name, qubit_index, new_file_name):
    """
    Extract and save configuration for a single qubit.
    
    Creates a new configuration file containing only the parameters for the 
    specified qubit index, extracted from a multi-qubit configuration.

    Args:
        file_name: Path to the original multi-qubit configuration file
        qubit_index: Index of the qubit to extract (0-indexed)
        new_file_name: Path where the single-qubit configuration will be saved

    Returns:
        AttrDict: The single-qubit configuration
        
    Note:
        Arrays with length > 1 are reduced to single values at the specified index.
        Scalar values and single-element arrays are preserved as-is.
    """
    # Load the original configuration
    cfg = load(file_name)

    def extract_single_element(data):
        """
        Recursively extract the ith element from fields with length > 1.
        """
        if isinstance(data, list):
            return data[qubit_index] if len(data) > 1 else data
        elif isinstance(data, dict):
            return {key: extract_single_element(value) for key, value in data.items()}
        else:
            return data

    # Extract the single qubit configuration
    single_qubit_cfg = extract_single_element(cfg.to_dict())

    # Save the new configuration
    cfg_yaml = yaml.safe_dump(single_qubit_cfg, default_flow_style=None)
    with open(new_file_name, "w") as modified_file:
        modified_file.write(cfg_yaml)

    return AttrDict(single_qubit_cfg)


def init_model_config(file_name, num_qubits):
    """
    Initialize a model configuration file for theoretical parameters.
    
    Creates a configuration for storing theoretical circuit parameters and 
    statistical analysis results from qubit characterization.

    Args:
        file_name: Path where the configuration will be saved
        num_qubits: Number of qubits to configure

    Returns:
        str: The YAML configuration string that was saved
        
    Note:
        A new file is created with "_model" appended before the extension.
        All parameters are initialized to None and filled during analysis.
    """

    # Create a helper function to initialize arrays
    def init_array(value, length=num_qubits):
        """Create an array with the same value repeated."""
        return [value] * length

    # Initialize the configuration structure with circuit parameters
    auto_cfg = {
        "nqubits": num_qubits,
        "Ec": init_array(None),             # Charging energy
        "Ej": init_array(None),             # Josephson energy
        "Delta": init_array(None),          # Asymmetry parameter
        "Sum": init_array(None),            # Sum parameter
        "alpha": init_array(None),          # Anharmonicity
        "T1_purcell": init_array(None),     # Purcell-limited T1
        "g_lamb": init_array(None),         # Coupling (Lamb shift)
        "g_chi": init_array(None),          # Coupling (dispersive shift)
        "g": init_array("chi"),             # Coupling type
        "kappa_low": init_array(None),      # Low-power linewidth
        "ratio": init_array(None),          # Ratio parameter
        "ng": init_array(None),             # Gate charge
        "nreadout": init_array(None),       # Readout photon number
        "Q1": init_array(None),             # Quality factor
        "T1_nopurcell": init_array(None),   # T1 without Purcell decay
        "Tphi": init_array(None),           # Pure dephasing time
    }

    # Statistical analysis parameters
    stats_cfg = {
        "t1_mean": init_array(None),
        "t1_max": init_array(None),
        "q1_mean": init_array(None),
        "q1_max": init_array(None),
        "t2e_mean": init_array(None),
        "t2e_max": init_array(None),
        "t2r_mean": init_array(None),
        "t2r_max": init_array(None),
        "tphi_mean": init_array(None),
        "tphi_max": init_array(None),
        "t1mean_nopurcell": init_array(None),
        "t1max_nopurcell": init_array(None),
        "t1_std": init_array(None),
        "q1_std": init_array(None),
        "t2e_std": init_array(None),
        "t2r_std": init_array(None),
        "tphi_std": init_array(None),
    }
    auto_cfg["stats"] = stats_cfg

    # Convert to YAML and save with "_model" suffix
    cfg_yaml = yaml.safe_dump(auto_cfg, default_flow_style=None)
    new_file_name = f"{file_name[0:-4]}_model.yml"
    with open(new_file_name, "w") as modified_file:
        modified_file.write(cfg_yaml)

    return cfg_yaml


def init_stark_section(file_name, num_qubits):
    """
    Add a Stark shift section to an existing configuration file.
    
    Adds parameters for characterizing AC Stark shifts from drive tones.
    Useful for experiments measuring and compensating Stark effects.

    Args:
        file_name: Path to the configuration file
        num_qubits: Number of qubits, determines array lengths

    Returns:
        AttrDict: The updated configuration object
        
    Note:
        Parameters represent Stark shift coefficients:
        - q, l, o: Quadratic, linear, offset terms
        - qneg, lneg, oneg: Negative detuning versions
        - f, fneg: Frequency shifts at positive/negative detuning
    """
    # Load the existing configuration
    cfg = load(file_name)

    # Define the stark parameters
    stark_params = ['q', 'l', 'o', 'qneg', 'lneg', 'oneg', 'f', 'fneg']

    # Create the stark section with null values
    stark_section = {param: [None] * num_qubits for param in stark_params}

    # Add the stark section to the configuration
    cfg['stark'] = stark_section

    # Save the updated configuration and return it
    return save(cfg, file_name)


def add_pulse(file_name, pulse_name, pulse_type="gauss"):
    """
    Add a new pulse definition to the configuration.
    
    Adds a pulse to device.qubit.pulses with appropriate default parameters
    based on the pulse type. Useful for defining custom pulse sequences.

    Args:
        file_name: Path to the configuration file
        pulse_name: Name for the new pulse (e.g., "pi_half", "echo")
        pulse_type: Type of pulse shape - "gauss", "const", or "flat_top" 
                    (default: "gauss")

    Returns:
        AttrDict: The updated configuration object
        
    Note:
        Default pulse parameters:
        - gauss: gain=0.15, sigma=0.1, sigma_inc=4
        - const: gain=0.15, length=1
        - flat_top: gain=0.15, length=0.1, ramp_sigma=0.02, ramp_sigma_inc=4
    """
    cfg = load(file_name)
    num_qubits = len(cfg.device.qubit.T1)

    def init_array(value, length=num_qubits):
        """Create an array with the same value repeated."""
        return [value] * length

    # Create pulse definition based on type
    if pulse_type=='gauss':
        new_pulse = {
            "gain": init_array(0.15),           # Pulse amplitude
            "sigma": init_array(0.1),           # Gaussian width (μs)
            "sigma_inc": init_array(4),         # Pulse length in sigmas
            "type": init_array(pulse_type),
        }
    elif pulse_type=='const':
        new_pulse = {
            "gain": init_array(0.15),           # Pulse amplitude
            "length": init_array(1),            # Pulse duration (μs)
            "type": init_array(pulse_type),
        }
    elif pulse_type=='flat_top':
        new_pulse = {
            "gain": init_array(0.15),           # Pulse amplitude
            "length": init_array(0.1),          # Flat top duration (μs)
            "ramp_sigma": init_array(0.02),     # Ramp width (μs)
            "ramp_sigma_inc": init_array(4),    # Ramp length in sigmas
            "type": init_array(pulse_type),
        }

    # Add the new pulse to the configuration
    cfg.device.qubit.pulses[pulse_name] = new_pulse

    return save(cfg, file_name)


def init_rfboard_active_section(file_name):
    """
    Add an ``rfboard_active`` section to an existing configuration file.

    Scans ``hw.soc.{adcs.readout,dacs.readout,dacs.qubit}.ch`` to discover
    unique physical channels and creates per-channel entries under
    ``hw.soc.rfboard_active.{adc,dac_readout,dac_qubit}`` with
    ``filter_type: "unknown"`` and ``qubit: null`` defaults.

    Idempotent: existing entries are preserved.

    Args:
        file_name: Path to the configuration file

    Returns:
        AttrDict: The updated configuration object
    """
    cfg = load(file_name)
    hw = cfg["hw"]["soc"]

    if "rfboard_active" not in hw:
        hw["rfboard_active"] = {}

    active = hw["rfboard_active"]

    # Map: (config section path, active key, default attn fields)
    channel_map = [
        (hw["adcs"]["readout"], "adc", {"attn": 0}),
        (hw["dacs"]["readout"], "dac_readout", {"attn1": 0, "attn2": 0}),
    ]
    if "qubit" in hw["dacs"]:
        channel_map.append(
            (hw["dacs"]["qubit"], "dac_qubit", {"attn1": 0, "attn2": 0})
        )

    for section, active_key, attn_defaults in channel_map:
        if active_key not in active:
            active[active_key] = {}
        for ch in set(section["ch"]):
            ch_str = str(ch)
            if ch_str not in active[active_key]:
                entry = {
                    "filter_fc": 0,
                    "filter_bw": 0,
                    "filter_type": "unknown",
                    "qubit": None,
                }
                entry.update(attn_defaults)
                active[active_key][ch_str] = entry

    return save(cfg, file_name)


def update_rfboard_active(file_name, channel_type, ch, settings):
    """
    Persist a channel's active RF state to the configuration file.

    Args:
        file_name: Path to the configuration file
        channel_type: One of ``"adc"``, ``"dac_readout"``, ``"dac_qubit"``
        ch: Physical channel number (int)
        settings: Dict with keys like ``filter_fc``, ``filter_bw``,
                  ``filter_type``, ``qubit``, and attenuation fields

    Returns:
        AttrDict: The updated configuration object
    """
    cfg = load(file_name)
    hw = cfg["hw"]["soc"]

    if "rfboard_active" not in hw:
        hw["rfboard_active"] = {}
    if channel_type not in hw["rfboard_active"]:
        hw["rfboard_active"][channel_type] = {}

    hw["rfboard_active"][channel_type][str(ch)] = dict(settings)

    return save(cfg, file_name)


def init_rfboard_section(file_name, num_qubits):
    """
    Add RF board filter fields to an existing configuration file.

    Adds ``filter_fc``, ``filter_bw``, and ``filter_type`` arrays to each
    channel group (``adcs.readout``, ``dacs.qubit``, ``dacs.readout``) with
    bypass defaults.  Safe to call on configs that already have the fields
    (existing values are preserved).

    Args:
        file_name: Path to the configuration file
        num_qubits: Number of qubits, determines array lengths

    Returns:
        AttrDict: The updated configuration object
    """
    cfg = load(file_name)
    hw = cfg["hw"]["soc"]

    filter_defaults = {
        "filter_fc": [0] * num_qubits,
        "filter_bw": [1000] * num_qubits,
        "filter_type": ["bypass"] * num_qubits,
    }

    # Sections to update: (section_dict, exists_check)
    sections = [hw["adcs"]["readout"], hw["dacs"]["readout"]]
    if "qubit" in hw["dacs"]:
        sections.append(hw["dacs"]["qubit"])

    for section in sections:
        for key, default in filter_defaults.items():
            if key not in section:
                section[key] = list(default)  # copy so each section is independent

    return save(cfg, file_name)


def update_rfboard(
    file_name, path, field, value, index=None, verbose=True, sig=4, rng_vals=None
):
    """
    Update an RF board parameter in the configuration.

    Convenience wrapper for ``update_config`` targeting ``hw.soc.{path}``
    (e.g., ``"dacs.qubit"``, ``"adcs.readout"``).

    Args:
        file_name: Path to the configuration file
        path: Dot-separated sub-path under hw.soc (e.g., "dacs.qubit", "adcs.readout")
        field: Field name to update (e.g., "filter_fc", "filter_bw", "filter_type")
        value: New value
        index: Optional channel/qubit index for array values
        verbose: Whether to print update information (default: True)
        sig: Number of significant digits (default: 4)
        rng_vals: Optional range limits (min, max)

    Returns:
        AttrDict: The updated configuration
    """
    return update_config(
        file_name, f"hw.soc.{path}", field, value, index, verbose, sig, rng_vals
    )
