"""
QICK Frequency Check Utilities

This module provides utility functions for checking and validating frequency settings
in quantum experiments using the QICK (Quantum Instrumentation Control Kit) framework.
It helps identify potential issues with frequency aliasing and validates qubit and
resonator frequency configurations.
"""

from . import config
import numpy as np


def get_dac_ch_name(soc, gen_ch):
    """
    Get the physical connector name of a DAC (generator) channel.

    Args:
        soc: QICK SoC or QickConfig object (indexable, e.g. soc['gens'])
        gen_ch (int): Generator channel index

    Returns:
        str: Board-specific connector name (RFSoC4x2, ZCU111, and ZCU216
        are recognized; other boards get a generic "DAC_<tile><block>" label)
    """
    dac_ch_str = soc['gens'][gen_ch]['dac']
    tile, block = [int(c) for c in dac_ch_str]
    if soc['board']=='RFSoC4x2':
        return {'00': 'DAC_B', '20': 'DAC_A'}.get(dac_ch_str, f"DAC_{dac_ch_str}")
    elif soc['board']=='ZCU111':
        return "DAC%d_T%d_CH%d or RF board output %d" % (tile + 228, tile, block, tile*4 + block)
    elif soc['board']=='ZCU216':
        return "%d_%d, on JHC%d" % (block, tile + 228, 1 + (block%2) + 2*(tile//2))
    return f"DAC_{dac_ch_str}"

def get_adc_ch_name(soc, ro_ch):
    """
    Get the physical connector name of an ADC (readout) channel.

    Args:
        soc: QICK SoC or QickConfig object (indexable, e.g. soc['readouts'])
        ro_ch (int): Readout channel index

    Returns:
        str: Board-specific connector name (RFSoC4x2, ZCU111, and ZCU216
        are recognized; other boards get a generic "ADC_<tile><block>" label)
    """
    adc_ch_str = soc['readouts'][ro_ch]['adc']
    tile, block = [int(c) for c in adc_ch_str]
    if soc['board']=='RFSoC4x2':
        return {'00': 'ADC_D', '02': 'ADC_C', '20': 'ADC_B', '21': 'ADC_A'}.get(adc_ch_str, f"ADC_{adc_ch_str}")
    elif soc['board']=='ZCU111':
        rfbtype = "DC" if tile > 1 else "AC"
        return "ADC%d_T%d_CH%d or RF board %s input %d" % (tile + 224, tile, block//2, rfbtype, (tile%2)*2 + block//2)
    elif soc['board']=='ZCU216':
        return "%d_%d, on JHC%d" % (block, tile + 224, 5 + (block%2) + 2*(tile//2))
    return f"ADC_{adc_ch_str}"

def get_ch(soc, name, type='dac'):
    """
    Look up a channel index from (part of) its connector name.

    Args:
        soc: QICK SoC or QickConfig object
        name (str): Substring of the connector name as returned by
            get_dac_ch_name / get_adc_ch_name (e.g. "DAC_A", "0_228")
        type (str): 'dac' or 'adc' (default: 'dac')

    Returns:
        int or None: First matching channel index, or None if no match
    """
    if type == 'dac':
        for i in range(len(soc['gens'])):
            if name in get_dac_ch_name(soc, i):
                return i
    elif type == 'adc':
        for i in range(len(soc['readouts'])):
            if name in get_adc_ch_name(soc, i):
                return i
    return None

def print_dac_channels(soc):
    """Prints the available DAC channels and their names."""
    print("DAC Channels:")
    for i in range(len(soc['gens'])):
        print(f"  {i}: {get_dac_ch_name(soc, i)}")

def print_adc_channels(soc):
    """Prints the available ADC channels and their names."""
    print("ADC Channels:")
    for i in range(len(soc['readouts'])):
        print(f"  {i}: {get_adc_ch_name(soc, i)}")

def check_freqs(i, cfg_dict):
    """
    Check qubit frequencies for potential aliasing issues and validate configuration.

    This function examines the qubit frequencies (f_ge and f_ef) for a specific qubit
    and checks for potential aliasing around the Nyquist frequency. It also calculates
    the anharmonicity and provides recommendations for frequency selection.

    Args:
        i (int): Qubit index to check
        cfg_dict (dict): Configuration dictionary containing 'cfg_file' path and 'soc' object

    Returns:
        None: Results are printed to the console
    """
    # Load configuration
    auto_cfg = config.load(cfg_dict["cfg_file"])

    # Get DAC channel for the specified qubit
    dac = auto_cfg.hw.soc.dacs.qubit.ch[i]

    # Get the sampling frequency from the SoC for the DAC channel
    fs = cfg_dict["soc"]._get_ch_cfg(dac)["fs"]

    # Calculate Nyquist frequency (mirror point)
    mirror_freq = fs / 2
    print(f"Nyquist frequency: {mirror_freq} MHz")

    # Get configured qubit frequencies
    freq = auto_cfg.device.qubit.f_ge[i]  # Ground to excited state transition
    fef = auto_cfg.device.qubit.f_ef[i]  # Excited to second excited state transition

    # Calculate potential aliased frequencies
    freq_offset = freq - mirror_freq
    alt_freq = mirror_freq - freq_offset  # Aliased g-e frequency

    freq_offset_ef = fef - mirror_freq
    alt_fef = mirror_freq - freq_offset_ef  # Aliased e-f frequency

    # Print frequency information
    print(f"Possible g-e frequencies: {freq} MHz and {alt_freq} MHz")
    print(f"Possible e-f frequencies: {fef} MHz and {alt_fef} MHz")

    # Calculate anharmonicity (difference between g-e and e-f transitions)
    alpha = freq - fef
    alpha2 = freq - alt_fef
    print(f"Anharmonicity: {alpha} MHz. With aliased e-f: {alpha2} MHz")

    # Provide recommendations based on frequency relationships
    if fef < freq and alt_fef < freq and fef > alt_freq and alt_fef > alt_freq:
        print(
            "Both e-f frequencies are less than the chosen g-e frequency and greater than the aliased g-e frequency."
        )
        print("Recommendation: Current g-e frequency is the correct choice.")

    if alt_fef > freq and alt_fef > alt_freq:
        print("Aliased e-f frequency is greater than both g-e frequencies.")
        print("Recommendation: Current e-f frequency is the correct choice.")


def check_resonances(cfg_dict):
    """
    Check resonator frequencies for potential aliasing issues.

    This function examines the readout resonator frequencies and identifies
    potential aliasing around the Nyquist frequency.

    Args:
        cfg_dict (dict): Configuration dictionary containing 'cfg_file' path and 'soc' object

    Returns:
        None: Results are printed to the console
    """
    # Load configuration
    auto_cfg = config.load(cfg_dict["cfg_file"])

    # Get readout DAC channel
    ro_dac = auto_cfg.hw.soc.dacs.readout.ch[0]

    # Get the sampling frequency from the SoC
    fs = cfg_dict["soc"]._get_ch_cfg(ro_dac)["fs"]

    # Calculate Nyquist frequency (mirror point)
    mirror_freq = fs / 2
    print(f"Nyquist frequency: {mirror_freq} MHz")

    # Get configured resonator frequencies
    freq = np.array(auto_cfg.device.readout.frequency)

    # Calculate aliased frequencies
    freq_offset = freq - mirror_freq
    alt_freq = mirror_freq - freq_offset

    # Print results
    print("Configured resonator frequencies (MHz):")
    # Print frequencies on one line with 4 significant digits
    print(" ".join(f"{float(f):.7g}" for f in freq))
    print("Aliased resonator frequencies (MHz):")
    print(" ".join(f"{float(f):.7g}" for f in alt_freq))
    print('Qubit frequencies (MHz):')
    print(" ".join(f"{float(f):.7g}" for f in auto_cfg.device.qubit.f_ge))
    print("Difference between mirror resonator and qubit frequencies (MHz):")
    print(" ".join(f"{float(f):.7g}" for f in auto_cfg.device.qubit.f_ge- alt_freq))
    

def check_adc(cfg_dict):
    """
    Check if any readout frequencies alias too close to Nyquist boundaries.

    This function identifies if any readout frequencies alias to within a critical
    window around Nyquist zone boundaries, which could cause readout issues.

    Args:
        cfg_dict (dict): Configuration dictionary containing 'cfg_file' path and 'soc' object

    Returns:
        None: Warnings are printed to the console if issues are found
    """
    # Load configuration
    auto_cfg = config.load(cfg_dict["cfg_file"])

    # Get ADC sampling frequency
    fs = cfg_dict["soc"]._get_ch_cfg(ro_ch=0)["fs"]
    print("ADC sampling frequency:", fs)

    # Calculate Nyquist frequency
    nyquist_freq = fs / 2

    # Get configured resonator frequencies
    freq = np.array(auto_cfg.device.readout.frequency)

    # Define critical window size (1/16 of sampling frequency for dynamic readout)
    window_size = fs / 16

    # Check each frequency for proximity to Nyquist zone boundaries
    for i, f in enumerate(freq):
        # Find which Nyquist zone the frequency falls in
        n = 0
        while abs(f - n * nyquist_freq) > nyquist_freq:
            n += 1

        # Calculate distance to nearest Nyquist zone boundary
        alias_dist = abs(f - n * nyquist_freq)

        # Warn if frequency is too close to a Nyquist zone boundary
        if alias_dist < window_size:
            print(
                f"Warning: Qubit {i} Frequency {f} MHz aliases to within {alias_dist:.1f} MHz of Nyquist frequency"
            )
            print(f"Distance to Nyquist zone {n} boundary: {alias_dist:.1f} MHz")
            print(f"This may cause readout issues. Consider adjusting the frequency.")
