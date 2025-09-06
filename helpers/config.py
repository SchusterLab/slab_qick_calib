"""
Configuration module for quantum experiments.

This module provides functions for loading, saving, and updating configuration files
for quantum experiments. It also includes functions for initializing default configurations
for different types of experiments.
"""

import yaml
from functools import reduce
import numpy as np
from datetime import datetime

from ..exp_handling.datamanagement import AttrDict


def nested_set(dic, keys, value):
    """Set a nested dictionary value using a list of keys."""
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def load(file_name):
    """Load a YAML configuration file and return it as an AttrDict."""
    with open(file_name, "r") as file:
        auto_cfg = AttrDict(yaml.safe_load(file))
    return auto_cfg


def save(cfg, file_name, reload=True):
    """
    Save a configuration to a YAML file.

    Args:
        cfg: Configuration object to save
        file_name: Path to save the configuration
        reload: Whether to reload the file after saving (default: True)

    Returns:
        The saved configuration as an AttrDict if reload=True, otherwise None
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

    Save a copy of a configuration file with a timestamp in the filename.

    Args:
        file_name: Path to the original configuration file

    Returns:
        The saved configuration as an AttrDict
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
    """Get a value from a nested dictionary using a list of keys."""
    return reduce(lambda c, k: c.get(k, {}), keys, d)


def in_rng(val, rng_vals):
    """
    Ensure a value is within a specified range.

    Args:
        val: The value to check
        rng_vals: A tuple/list of (min, max) values

    Returns:
        The value, clamped to the specified range
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

    Args:
        value: The value to format
        sig: Number of significant digits for floating point values
        rng_vals: Optional range limits (min, max)

    Returns:
        The formatted value
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

    This is a general-purpose update function that can update any part of the configuration.

    Args:
        file_name: Path to the configuration file
        path: Path to the parameter section (e.g., "device.qubit", "hw.soc.lo")
        field: Field name to update
        value: New value
        index: Optional index for array values
        verbose: Whether to print update information
        sig: Number of significant digits for floating point values
        rng_vals: Optional range limits (min, max)

    Returns:
        The updated configuration
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
    """Update a qubit parameter in the configuration."""
    return update_config(
        file_name, "device.qubit", field, value, qubit_i, verbose, sig, rng_vals
    )


def update_readout(
    file_name, field, value, qubit_i, verbose=True, sig=4, rng_vals=None
):
    """Update a readout parameter in the configuration."""
    return update_config(
        file_name, "device.readout", field, value, qubit_i, verbose, sig, rng_vals
    )


def update_stark(file_name, field, value, qubit_i, verbose=True, sig=4, rng_vals=None):
    """Update a Stark shift parameter in the configuration."""
    return update_config(
        file_name, "stark", field, value, qubit_i, verbose, sig, rng_vals
    )


def update_lo(file_name, field, value, qi, verbose=True, sig=4, rng_vals=None):
    """Update a local oscillator parameter in the configuration."""
    return update_config(
        file_name, "hw.soc.lo", field, value, qi, verbose, sig, rng_vals
    )


def init_config(file_name, num_qubits, type="full", t1=50, aliases="Qick001", ip=""):
    """
    Initialize a configuration file for quantum experiments with qubits.

    Args:
        file_name: Path to save the configuration
        num_qubits: Number of qubits to configure
        type: Type of readout, default is "full"
        t1: Default T1 relaxation time in μs
        aliases: Identifier for the System-on-Chip (SoC)
        ip: IP address for the device

    Returns:
        The created configuration
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
            "f_ge": init_array(4000),
            "f_ef": init_array(3800),
            "kappa": init_array(0),
            "spec_gain": init_array(1),
        }
    )

    # Qubit pulse parameters
    for pulse_type in ["pi_ge", "pi_ef"]:
        device["qubit"]["pulses"][pulse_type].update(
            {
                "gain": init_array(0.15),
                "sigma": init_array(0.1),
                "sigma_inc": init_array(5),
                "type": init_array("gauss"),
                "phase": init_array(0),
            }
        )

    # Other qubit parameters
    device["qubit"].update(
        {
            "pop": init_array(0),
            "temp": init_array(0),
            "tuned_up": init_array(False),
            "low_gain": 0.003,
            "max_gain": 1,
        }
    )

    # Readout frequency and gain
    device["readout"].update(
        {
            "frequency": init_array(7000),
            "gain": init_array(0.05),
        }
    )

    # Readout resonator parameters
    device["readout"].update(
        {
            "lamb": init_array(0),
            "chi": init_array(0),
            "kappa": init_array(0.5),
            "qe": init_array(0),
            "qi": init_array(0),
        }
    )

    # Readout settings
    device["readout"].update(
        {
            "phase": init_array(0),
            "readout_length": init_array(5),
            "threshold": init_array(10),
            "fidelity": init_array(0),
            "tm": init_array(0),
            "sigma": init_array(0),
            "rescale": init_array(False),
        }
    )

    # Readout timing
    device["readout"].update(
        {
            "trig_offset": init_array(0.5),
            "final_delay": init_array(t1 * 6),
            "active_reset": init_array(False),
            "reset": init_array(3),
        }
    )

    # Readout averaging
    device["readout"].update(
        {
            "reps": init_array(1),
            "rounds": init_array(1),
            "max_gain": 1,
            "reps_base": 150,
            "rounds_base": 1,
        }
    )

    # Hardware configuration
    soc = {
        "adcs": {
            "readout": {
                "ch": init_array(0),
                "type": init_array("dyn"),
            }
        },
        "dacs": {
            "qubit": {
                "ch": init_array(1),
                "nyquist": init_array(1),
                "type": init_array("full"),
            },
            "readout": {
                "ch": init_array(0),
                "nyquist": init_array(2),
                "type": init_array(type),
            },
        },
    }

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

    Args:
        file_name: Path to save the configuration
        num_qubits: Number of qubits to configure
        type: Type of readout, default is "full"
        aliases: Identifier for the System-on-Chip (SoC)

    Returns:
        The created configuration
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
            "frequency": init_array(7000),
            "gain": init_array(0.05),
        }
    )

    # Readout resonator parameters
    device["readout"].update(
        {
            "kappa": init_array(0.5),
            "kappa_hi": init_array(0.5),
            "qe": init_array(0),
            "qi": init_array(0),
            "qi_hi": init_array(0),
            "qi_lo": init_array(0),
        }
    )

    # Readout settings
    device["readout"].update(
        {
            "phase": init_array(0),
            "readout_length": init_array(100),
            "trig_offset": init_array(0.5),
            "final_delay": init_array(50),
        }
    )

    # Readout averaging
    device["readout"].update(
        {
            "reps": init_array(1),
            "rounds": init_array(1),
            "max_gain": 1,
            "reps_base": 30,
            "rounds_base": 1,
            "phase_inc": [1140],
        }
    )

    # Hardware configuration
    soc = {
        "adcs": {"readout": {"ch": init_array(0)}},
        "dacs": {
            "readout": {
                "ch": init_array(0),
                "nyquist": init_array(2),
                "type": init_array(type),
            },
        },
    }

    # Assemble the complete configuration
    auto_cfg = {"device": device, "hw": {"soc": soc}, "aliases": {"soc": aliases}}

    # Convert to YAML and save
    cfg_yaml = yaml.safe_dump(auto_cfg, default_flow_style=None)
    with open(file_name, "w") as modified_file:
        modified_file.write(cfg_yaml)

    return cfg_yaml


def save_single_qubit_config(file_name, qubit_index, new_file_name):
    """
    Save a configuration file that contains only the ith element from each field
    in the configuration that has length greater than 1.

    Args:
        file_name: Path to the original configuration file
        qubit_index: Index of the qubit to extract
        new_file_name: Path to save the new configuration file

    Returns:
        The saved configuration as an AttrDict
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
    Initialize a model configuration file for quantum experiments with qubits.

    Args:
        file_name: Path to save the configuration
        num_qubits: Number of qubits to configure

    Returns:
        The created configuration
    """

    # Create a helper function to initialize arrays
    def init_array(value, length=num_qubits):
        """Create an array with the same value repeated."""
        return [value] * length

    # Initialize the configuration structure
    auto_cfg = {
        "nqubits": num_qubits,
        "Ec": init_array(None),
        "Ej": init_array(None),
        "Delta": init_array(None),
        "Sum": init_array(None),
        "alpha": init_array(None),
        "T1_purcell": init_array(None),
        "g_lamb": init_array(None),
        "g_chi": init_array(None),
        "g": init_array("chi"),
        "kappa_low": init_array(None),
        "ratio": init_array(None),
        "ng": init_array(None),
        "Q1": init_array(None),
        "T1_nopurcell": init_array(None),
        "Tphi": init_array(None),
    }

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

    # Convert to YAML and save
    cfg_yaml = yaml.safe_dump(auto_cfg, default_flow_style=None)
    new_file_name = f"{file_name[0:-4]}_model.yml"
    with open(new_file_name, "w") as modified_file:
        modified_file.write(cfg_yaml)

    return cfg_yaml


def init_stark_section(file_name, num_qubits):
    """
    Initializes a 'stark' section in a configuration file.

    This function loads an existing configuration, adds a 'stark' section with
    np.nan values for various parameters, and saves the updated configuration.

    Args:
        file_name (str): Path to the configuration file.
        num_qubits (int): The number of qubits, used to determine the length of the parameter arrays.

    Returns:
        AttrDict: The updated configuration object.
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
    Add a new pulse to the configuration.
    The new pulse will be added to the device/qubit/pulses section.

    Args:
        file_name (str): Path to the configuration file.
        pulse_name (str): Name of the new pulse to add.

    Returns:
        AttrDict: The updated configuration object.
    """
    cfg = load(file_name)
    num_qubits = len(cfg.device.qubit.T1)

    def init_array(value, length=num_qubits):
        """Create an array with the same value repeated."""
        return [value] * length

    if pulse_type=='gauss':
        new_pulse = {
            "gain": init_array(0.15),
            "sigma": init_array(0.1),
            "sigma_inc": init_array(4),
            "type": init_array(pulse_type),
        }
    elif pulse_type=='const':
        new_pulse = {
            "gain": init_array(0.15),
            "length": init_array(1),
            "type": init_array(pulse_type),
        }
    elif pulse_type=='flat_top':
        new_pulse = {
            "gain": init_array(0.15),
            "length": init_array(0.1),
            "ramp_sigma": init_array(0.02),
            "ramp_sigma_inc": init_array(4),
            "type": init_array(pulse_type),
        }

    cfg.device.qubit.pulses[pulse_name] = new_pulse

    return save(cfg, file_name)
