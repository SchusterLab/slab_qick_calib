"""
Qubit Parameter Analysis Module

This module provides functions for calculating various qubit parameters and properties
used in superconducting quantum circuits, including coupling strengths, coherence times,
and Hamiltonian parameters. It also includes utilities for updating configuration files
with calculated parameters.

Dependencies:
- scqubits: For transmon qubit calculations
- scipy.constants: For physical constants
- numpy: For numerical operations
- config helper: For configuration file management
"""

import scqubits as scq
from ..helpers import config 
import numpy as np
from scipy import constants as cs

# Physical constants
phi_0 = cs.h/(2*cs.e)  # Flux quantum (Wb)

def ham(cfg_path): 
    """
    Calculate and update Hamiltonian parameters for all qubits in the configuration.
    
    This function:
    1. Loads the main configuration file
    2. Calculates transmon parameters (EJ, EC, anharmonicity) using scqubits
    3. Calculates transverse coupling strengths
    4. Updates the model configuration file with calculated values
    
    Parameters:
    -----------
    cfg_path : str
        Path to the main configuration YAML file
        
    Notes:
    ------
    - Creates/updates a corresponding '_model.yml' file with calculated parameters
    - Uses scqubits.Transmon.find_EJ_EC() to determine Josephson and charging energies
    - Calculates g_chi (transverse coupling) using counter-rotating term
    """
    auto_cfg = config.load(cfg_path)
    model_name = cfg_path[0:-4] + '_model.yml'
    model_cfg = config.load(cfg_path[0:-4] + '_model.yml')
    
    # Loop through all qubits in the configuration
    for i in np.arange(len(auto_cfg.device.qubit.f_ge)):
        # Calculate anharmonicity from transition frequencies
        alpha =  auto_cfg.device.qubit.f_ef[i] - auto_cfg.device.qubit.f_ge[i]
        
        # Use scqubits to find EJ and EC from measured frequencies
        en = scq.Transmon.find_EJ_EC(auto_cfg.device.qubit.f_ge[i]/1000,alpha/1000)
        
        # Update model configuration with calculated parameters
        config.update_config(model_name, None, 'alpha', alpha, index=i, verbose=False, sig=4)
        config.update_config(model_name, None, 'Ej', en[0], index=i, verbose=False, sig=4)
        config.update_config(model_name, None, 'Ec', en[1], index=i, verbose=False, sig=4)
        config.update_config(model_name, None, 'ratio', en[0]/en[1], index=i, verbose=False, sig=4)

        # Calculate g from chi using counter-rotating term
        gX = gX_CR(auto_cfg.device.qubit.f_ge[i], auto_cfg.device.readout.frequency[i], 
                   auto_cfg.device.readout.chi[i], alpha)
        config.update_config(model_name, None, 'g_chi', gX, index=i, verbose=False, sig=4)

        #nreadout = auto_cfg.device.readout.gain[i]**2*auto_cfg.device.readout.nstark[i]
        #config.update_config(model_name, None, 'nreadout', nreadout, index=i, verbose=False, sig=2)

def delta(cfg_path):
    """
    Calculate detuning parameters and related quantities for all qubits.
    
    This function:
    1. Calculates detuning (Delta) and sum frequency (Sum) for each qubit-resonator pair
    2. Calculates longitudinal coupling strength using lamb shift data
    3. Calculates Purcell-limited T1 times
    4. Calculates critical photon numbers
    5. Updates the model configuration file
    
    Parameters:
    -----------
    cfg_path : str
        Path to the main configuration YAML file
        
    Notes:
    ------
    - Delta = -(f_qubit - f_resonator): negative detuning convention
    - Sum = f_qubit + f_resonator: sum frequency for counter-rotating term calculations
    - Uses kappa_low values from model config for T1 Purcell calculations
    """
    model_name = cfg_path[0:-4] + '_model.yml'
    auto_cfg = config.load(cfg_path)
    model_cfg = config.load(cfg_path[0:-4] + '_model.yml')
    
    # Loop through all qubits in the configuration
    for i in np.arange(len(auto_cfg.device.qubit.f_ge)):
        # Calculate detuning (negative convention: qubit below resonator)
        Delta = -(auto_cfg.device.qubit.f_ge[i] - auto_cfg.device.readout.frequency[i])
        config.update_config(model_name, None, 'Delta', Delta, index=i, verbose=False, sig=4)

        # Calculate sum frequency for counter-rotating term effects
        Sum = auto_cfg.device.qubit.f_ge[i] + auto_cfg.device.readout.frequency[i]
        config.update_config(model_name, None, 'Sum', Sum, index=i, verbose=False, sig=4)
        
        # Calculate g from Lamb shift measurements
        g_lamb = gL_CR(Delta, Sum, auto_cfg.device.readout.lamb[i])
        config.update_config(model_name, None, 'g_lamb', g_lamb, index=i, verbose=False, sig=4)
        
        if model_cfg.g[i] == "chi":
            # If g is set to "chi", use g_chi value instead
            g = model_cfg.g_chi[i]
        else:
            g = model_cfg.g_lamb[i]
        # Calculate Purcell-limited T1 using low-power kappa (in case kappa is broadened at the readout power)
        kappa_val = model_cfg.kappa_low[i]
        T1purcell = T1p(kappa_val, Delta, g)
        config.update_config(model_name, None, 'T1_purcell', T1purcell, index=i, verbose=False, sig=4)
        
        # Calculate critical photon number in resonator
        nG = ng(Delta, g)
        config.update_config(model_name, None, 'ng', nG, index=i, verbose=False, sig=4)

def cohere(cfg_path, t2_type=None):
    """
    Calculate coherence-related parameters for all qubits.

    This function:
    1. Calculates quality factors from T1 measurements
    2. Calculates pure dephasing times from T1 and T2 measurements
    3. Calculates T1 times excluding Purcell decay contributions
    4. Updates the model configuration file

    Parameters:
    -----------
    cfg_path : str
        Path to the main configuration YAML file
    t2_type : str or list of str, optional
        Which T2 to use per qubit: 'T2e', 'T2r', or a list of per-qubit types.
        If None, picks the shorter of T2e/T2r for each qubit (the one that was
        actually measured most recently is typically shorter).

    Notes:
    ------
    - Q1 = 2π·f_ge·T1: quality factor in MHz·μs units (divided by 1e6 for storage)
    - T_φ calculated from measured T1 and T2 using standard relation
    - T1_nopurcell = 1/(1/T1_measured - 1/T1_purcell): intrinsic T1 excluding Purcell decay
    """
    model_name = cfg_path[0:-4] + '_model.yml'
    model_cfg = config.load(cfg_path[0:-4] + '_model.yml')
    auto_cfg = config.load(cfg_path)
    n_qubits = len(auto_cfg.device.qubit.f_ge)

    # Resolve per-qubit T2 type
    if isinstance(t2_type, list):
        t2_types = t2_type
    elif isinstance(t2_type, str):
        t2_types = [t2_type] * n_qubits
    else:
        # Auto: pick longer of T2e/T2r for each qubit
        t2_types = []
        for i in range(n_qubits):
            t2e = auto_cfg.device.qubit.T2e[i]
            t2r = auto_cfg.device.qubit.T2r[i]
            t2_types.append('T2e' if t2e >= t2r else 'T2r')

    # Loop through all qubits in the configuration
    for i in np.arange(n_qubits):
        # Calculate quality factor Q = 2π·f·T1
        q = np.pi*2 * auto_cfg.device.qubit.f_ge[i] * auto_cfg.device.qubit.T1[i]
        config.update_config(model_name, None, 'Q1', q/1e6, index=i, verbose=False, sig=4)

        # Calculate pure dephasing time from T1 and T2 measurements
        T1_val = auto_cfg.device.qubit.T1[i]
        T2_val = getattr(auto_cfg.device.qubit, t2_types[i])[i]
        TPhi = Tphi(T1_val, T2_val)
        config.update_config(model_name, None, 'Tphi', TPhi, index=i, verbose=False, sig=4)

        # Calculate intrinsic T1 excluding Purcell decay contribution
        T1_val = auto_cfg.device.qubit.T1[i]
        T1_purcell_val = model_cfg.T1_purcell[i]
        T1nopurcell = T1_nopurcell(T1_val, T1_purcell_val)
        config.update_config(model_name, None, 'T1_nopurcell', T1nopurcell, index=i, verbose=False, sig=4)

def stats(cfg_path, tt_stats, qubit_list, t2='auto', tracking_dir=None):
    from .collections import detect_t2_type

    # Resolve per-qubit T2 type
    if isinstance(t2, list):
        t2_types = t2
    elif t2 == 'auto':
        if tracking_dir is None:
            raise ValueError("tracking_dir is required when t2='auto'")
        t2_types = [detect_t2_type(tracking_dir, q).lower() for q in qubit_list]
    else:
        t2_types = [t2] * len(qubit_list)

    pars = ['t1', 't2', 'q1', 'tphi']
    vals = ['mean', 'max', 'std']
    for i, q in enumerate(qubit_list):
        for p in pars:
            for v in vals:
                g = tt_stats[i][p][v]
                pp = t2_types[i] if p == 't2' else p
                config.update_config(cfg_path[0:-4] + '_model.yml', 'stats', f'{pp}_{v}', g, index=q, verbose=False, sig=4)

        



def lj(ej):
    """
    Calculate Josephson inductance from Josephson energy.
    
    The Josephson inductance is given by: L_J = (φ₀/2π)² / (ℏ·E_J)
    
    Parameters:
    -----------
    ej : float or None
        Josephson energy in GHz
        
    Returns:
    --------
    float or None
        Josephson inductance in nH, or None if ej is None
    """
    if ej is None:
        return None
    return (phi_0/2/np.pi)**2 / (cs.h*ej*1e9)*1e9

def gL(Delta, chiL):
    """
    Calculate longitudinal coupling strength from detuning and longitudinal dispersive shift.
    
    For longitudinal coupling: g_L = √(-Δ·χ_L)
    
    Parameters:
    -----------
    Delta : float or None
        Detuning between qubit and resonator (MHz)
    chiL : float or None
        Longitudinal dispersive shift (MHz)
        
    Returns:
    --------
    float or None
        Longitudinal coupling strength (MHz), or None if any parameter is None
    """
    if Delta is None or chiL is None:
        return None
    return np.sqrt(-Delta*chiL)

def gX(Delta, chi, alpha):
    """
    Calculate transverse coupling strength from detuning, dispersive shift, and anharmonicity.
    
    For transverse coupling in the dispersive regime:
    g_X = √(Δ·(Δ+α)/α·χ) / 2
    
    Parameters:
    -----------
    Delta : float or None
        Detuning between qubit and resonator (MHz)
    chi : float or None
        Dispersive shift (MHz)
    alpha : float or None
        Qubit anharmonicity (MHz)
        
    Returns:
    --------
    float or None
        Transverse coupling strength (MHz), or None if any parameter is None
    """
    if Delta is None or chi is None or alpha is None:
        return None
    return np.sqrt(Delta * (Delta + alpha) / alpha * chi)/2

def gX_CR(Delta, Sum, chi, alpha):
    """
    Calculate transverse coupling strength using including counter rotating terms.

    This uses both detuning and sum frequency for more accurate coupling calculation:
    g_X = √(χ/α / (1/Δ/(Δ+α) + 1/Σ/(Σ-α))) / 2
    
    Parameters:
    -----------
    Delta : float or None
        Detuning: qubit frequency - resonator frequency (MHz)
    Sum : float or None
        Sum: qubit frequency + resonator frequency (MHz)
    chi : float or None
        Dispersive shift (MHz)
    alpha : float or None
        Qubit anharmonicity (MHz)
        
    Returns:
    --------
    float or None
        Transverse coupling strength (MHz), or None if any parameter is None
    """
    if Delta is None or Sum is None or chi is None or alpha is None:
        return None
    return np.sqrt(chi/ alpha / (1/Delta/(Delta + alpha) + 1/Sum/(Sum - alpha)))/2

def gL_CR(Delta, Sum, chiL):
    """
    Calculate longitudinal coupling strength using counter rotating term.
    
    This accounts for both positive and negative detuning contributions:
    g_L = √(-χ_L / (1/Δ + 1/Σ))
    
    Parameters:
    -----------
    Delta : float or None
        Detuning: qubit frequency - resonator frequency (MHz)
    Sum : float or None
        Sum: qubit frequency + resonator frequency (MHz)
    chiL : float or None
        Longitudinal dispersive shift (MHz)
        
    Returns:
    --------
    float or None
        Longitudinal coupling strength (MHz), or None if any parameter is None
    """
    if Delta is None or Sum is None or chiL is None:
        return None
    return np.sqrt(-chiL / (1/Delta + 1/Sum))

def chi(alpha, Delta, g):
    """
    Calculate dispersive shift from anharmonicity, detuning, and coupling.
    
    In the dispersive regime: χ = α·g²/(Δ·(Δ+α))
    
    Parameters:
    -----------
    alpha : float or None
        Qubit anharmonicity (MHz)
    Delta : float or None
        Detuning between qubit and resonator (MHz)
    g : float or None
        Coupling strength (MHz)
        
    Returns:
    --------
    float or None
        Dispersive shift (MHz), or None if any parameter is None
    """
    if alpha is None or Delta is None or g is None:
        return None
    return alpha*g**2/Delta/(Delta + alpha)

def chiL(g, Delta, Sum):
    """
    Calculate longitudinal dispersive shift.
    
    For longitudinal coupling: χ_L = -g²·(1/Δ + 1/Σ)
    
    Parameters:
    -----------
    g : float or None
        Coupling strength (MHz)
    Delta : float or None
        Detuning: qubit frequency - resonator frequency (MHz)
    Sum : float or None
        Sum: qubit frequency + resonator frequency (MHz)
        
    Returns:
    --------
    float or None
        Longitudinal dispersive shift (MHz), or None if any parameter is None
    """
    if g is None or Delta is None or Sum is None:
        return None
    return -g**2*(1/Delta + 1/Sum)

# Placeholder for future dispersive regime chi calculation
#def chi_DR(alpha, Delta, Sum, g): 

def T1p(kappa, Delta, g):
    """
    Calculate Purcell-limited T1 time.
    
    Purcell decay rate: Γ_Purcell = κ·(g/Δ)²
    T1_Purcell = 1/(2π·Γ_Purcell) = (Δ/g)²/(2π·κ)
    
    Parameters:
    -----------
    kappa : float or None
        Resonator decay rate (MHz)
    Delta : float or None
        Detuning between qubit and resonator (MHz)
    g : float or None
        Coupling strength (MHz)
        
    Returns:
    --------
    float or None
        Purcell-limited T1 time (μs), or None if any parameter is None or zero
    """
    if kappa is None or Delta is None or g is None or kappa == 0 or Delta == 0 or g == 0:
        return None
    return 1/kappa*(Delta/g)**2/2/np.pi

def ng(Delta, g):
    """
    Calculate average photon number in resonator due to qubit coupling.
    
    In the dispersive regime: n̄ = (Δ/g)²/4
    
    Parameters:
    -----------
    Delta : float or None
        Detuning between qubit and resonator (MHz)
    g : float or None
        Coupling strength (MHz)
        
    Returns:
    --------
    float or None
        Average photon number, or None if any parameter is None
    """
    if Delta is None or g is None or g == 0:
        return None
    return (Delta/g)**2/4

def T1px(kappa, chi, alpha):
    """
    Calculate T1 decay time due to transverse coupling effects.
    
    T1_transverse = α/(2π·κ·χ)
    
    Parameters:
    -----------
    kappa : float or None
        Resonator decay rate (MHz)
    chi : float or None
        Dispersive shift (MHz)
    alpha : float or None
        Qubit anharmonicity (MHz)
        
    Returns:
    --------
    float or None
        T1 time due to transverse effects (μs), or None if any parameter is None
    """
    if kappa is None or chi is None or alpha is None:
        return None
    return alpha/kappa/chi/2/np.pi

def T1px_opt(chi, alpha):
    """
    Calculate optimized T1 decay time for transverse coupling.
    
    Optimized formula: T1_opt = α/(4π·χ²)
    
    Parameters:
    -----------
    chi : float or None
        Dispersive shift (MHz)
    alpha : float or None
        Qubit anharmonicity (MHz)
        
    Returns:
    --------
    float or None
        Optimized T1 time (μs), or None if any parameter is None
    """
    if chi is None or alpha is None:
        return None
    return alpha/chi**2/4/np.pi

def Tphi(T1, T2):
    """
    Calculate pure dephasing time T_φ from T1 and T2 measurements.
    
    The relationship is: 1/T2 = 1/(2·T1) + 1/T_φ
    Therefore: T_φ = 1/(1/T2 - 1/(2·T1))
    
    Parameters:
    -----------
    T1 : float or None
        Longitudinal relaxation time (μs)
    T2 : float or None
        Transverse coherence time (μs)
        
    Returns:
    --------
    float or None
        Pure dephasing time (μs), infinity if T2 ≥ 2·T1, or None if invalid parameters
    """
    if T1 is None or T2 is None or T1 == 0 or T2 == 0:
        return None
    if T2 >= 2*T1:
        return np.inf
    return 1/(1/T2 - 1/(2*T1))

def T1_nopurcell(T1_measured, T1_purcell):
    """
    Calculate intrinsic T1 time excluding Purcell decay contribution.
    
    The total decay rate is the sum of intrinsic and Purcell decay rates:
    1/T1_total = 1/T1_intrinsic + 1/T1_purcell
    Therefore: T1_intrinsic = 1/(1/T1_total - 1/T1_purcell)
    
    Parameters:
    -----------
    T1_measured : float or None
        Measured total T1 time (μs)
    T1_purcell : float or None
        Purcell-limited T1 time (μs)
        
    Returns:
    --------
    float or None
        Intrinsic T1 time excluding Purcell decay (μs), or None if invalid parameters
    """
    if T1_measured is None or T1_purcell is None or T1_measured == 0 or T1_purcell == 0:
        return None
    try:
        denominator = 1/T1_measured - 1/T1_purcell
        return 1/denominator if denominator != 0 else None
    except Exception:
        return None

