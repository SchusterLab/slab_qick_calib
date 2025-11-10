# SLAB QICK Calibration

A comprehensive package for calibrating and characterizing superconducting qubits using the QICK (Quantum Instrumentation Control Kit) framework.
Note: Primary authorship of the READMEs and comments belongs to Cline; but they have largely been checked for accuracy and edited as needed. 

## Overview

This package provides tools and utilities for calibrating and characterizing superconducting qubits using the QICK framework. It includes a wide range of experiments for measuring qubit parameters, optimizing control pulses, and characterizing qubit coherence. The package has been converted to a namespace package to allow importing functions and modules directly.

Key features:
- Single qubit characterization experiments (resonator spectroscopy, qubit spectroscopy, Rabi, T1, T2)
- Two qubit experiments
- Automated calibration workflows
- Data management and analysis tools
- Configuration management

## Installation

### Development Installation

For development, you can install the package in development mode:

```bash
# Clone the repository
git clone <repository-url>
cd slab_qick_calib

# Install in development mode
pip install -e .
```

### Regular Installation

```bash
pip install .
```

## Usage

After installation, you can import the package and its modules:

```python
# Import the package
import slab_qick_calib

# Import specific modules
from slab_qick_calib import calib
from slab_qick_calib import exp_handling
from slab_qick_calib import experiments
from slab_qick_calib import analysis
from slab_qick_calib import helpers

# Import experiments as a group (common pattern)
import slab_qick_calib.experiments as meas

# Import specific functions or classes
from slab_qick_calib.calib import qubit_tuning, measure_func
from slab_qick_calib.experiments.single_qubit import resonator_spectroscopy
from slab_qick_calib.experiments.general import qick_experiment
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
```

## Package Structure

- `analysis/`: Data analysis and fitting tools
  - `allan.py`: Allan variance analysis for measuring qubit stability and noise characterization
  - `collections.py`: Data collection, processing, and visualization tools (histograms, violin plots, time series)
  - `fitting.py`: Curve fitting functions for experiment data (exponential, sinusoidal, Lorentzian, etc.)
  - `qubit_params.py`: Theoretical qubit parameter calculations and circuit QED formulas
  - `time_series.py`: Power spectral density and time series analysis for noise characterization
- `calib/`: Calibration modules for qubit tuning
  - `measure_func.py`: Measurement functions for calibration (dispersive shift and temperature measurements)
  - `qubit_tuning.py`: Automated qubit tuning workflows with intelligent parameter optimization
  - `readout_helpers.py`: Advanced readout analysis tools including histogram fitting, IQ rotation, and discrimination metrics
  - `time_tracking.py`: Long-term stability tracking and time-based measurement protocols
- `configs/`: Configuration files for different systems
  - Various `.yml` and `.cfg` files for instrument configurations
- `exp_handling/`: Experiment handling modules for data management and analysis (slab files, so that you don't need to install slab)
  - `dataanalysis.py`: Data analysis utilities
  - `datamanagement.py`: Data storage and retrieval
  - `experiment.py`: Base experiment classes
  - `instrumentmanager.py`: Instrument management and control
- `experiments/`: Experiment implementations
  - `general/`: Base classes for QICK experiments
    - `qick_experiment.py`: Base classes for single qubit QICK experiments
    - `qick_experiment_2q.py`: Base classes for two qubit QICK experiments
    - `qick_program.py`: Base classes for QICK programs
  - `single_qubit/`: Single qubit experiments
    - `active_reset.py`: Checks of active qubit reset parameters, mostly not used
    - `pulse_probe_spectroscopy.py`: Measures qubit transition frequencies
    - `rabi.py`: Calibrates qubit control pulses
    - `resonator_spectroscopy.py`: Characterizes readout resonators
    - `single_shot.py`: Single-shot readout fidelity
    - `single_shot_opt.py`: Optimizes single-shot readout
    - `stark_spectroscopy.py`: Measures AC Stark shifts, still writing up.
    - `t1.py`: Measures energy relaxation time
    - `t1_cont.py`: Fast continuous T1 measurements 
    - `t1_stark.py`: T1 measurements with Stark shifts of qubit
    - `t1_stark_complex.py`: Complex T1 measurements with multiple Stark tones
    - `t2.py`: Measures phase coherence time (Ramsey and Echo)
    - `t2_ramsey_stark.py`: Ramsey T2 with Stark shifts
    - `tof_calibration.py`: Calibrates time of flight for readout
  - `two_qubit/`: Two qubit experiments
    - `rabi_2q.py`: Two-qubit Rabi oscillations
    - `t1_2q.py`: Two-qubit T1 measurements
    - `t1_2q_cont.py`: Continuous two-qubit T1 measurements
- `helpers/`: Utility functions and configuration helpers
  - `config.py`: Configuration file handling with YAML load/save and parameter updates
  - `handy.py`: General utility functions for plotting and data visualization
  - `qick_check.py`: QICK system verification, channel mapping, and debugging tools
- `notebooks/`: Example notebooks and tutorials
  - Various Jupyter notebooks demonstrating package usage

For detailed documentation on the experiments, see [README_experiments.md](README_experiments.md).

For detailed documentation on the config file, see [README_config.md](README_config.md).

For detailed documentation on the base classes, see [README_classes.md](README_classes.md).

For documentation on the process of tuning up qubits, see [README_qubit_tuning.md](README_qubit_tuning.md).

For information on the structure of data that is saved, see [README_data_structures.md](README_data_structures.md).

For information on connecting to the QICK board remotely, see [README_connecting_to_qick.md](README_connecting_to_qick.md).

## Analysis Tools

The package includes powerful analysis tools for characterizing qubit stability and noise:

### Allan Variance Analysis (`analysis/allan.py`)

Allan variance is a measure of frequency stability over time, useful for characterizing different noise sources affecting qubits:

```python
from slab_qick_calib.analysis import allan

# Perform Allan variance analysis on tracking data
allan.perform_analysis(tracking_data, fname='stability_analysis', param='t1', qubit_list=[0, 1])
```

**Key features:**
- **Noise model fitting**: Power law, Lorentzian, and combined models to identify noise sources
- **Overlapping Allan variance**: Improved statistics for shorter datasets
- **Automatic noise source identification**: Distinguishes white noise, flicker noise, and random walk
- **Visualization**: Time series and Allan deviation plots with fit curves

### Data Collections and Visualization (`analysis/collections.py`)

Tools for processing and visualizing large collections of measurement data:

```python
from slab_qick_calib.analysis import collections

# Process tracking data and calculate derived parameters
tt_processed, stats = collections.process_data(tracking_data, qubit_list=[0, 1, 2])

# Create histogram plots for all parameters
collections.plot_all(tracking_data, qubit_list=[0, 1], param_keys=['t1', 't2r', 'f_ge'])

# Create violin plots for statistical comparison
collections.plot_violin(tracking_data, qubit_list=[0, 1], param_keys=['t1', 't2r'])
```

**Key features:**
- **Automated data processing**: Calculates quality factors, dephasing times, and other derived parameters
- **Multi-parameter visualization**: Histograms and time series for comprehensive overview
- **Statistical analysis**: Violin plots for comparing distributions across qubits
- **Outlier filtering**: Automatic removal based on fit quality (R²) and statistical thresholds

### Time Series Analysis (`analysis/time_series.py`)

Power spectral density analysis for identifying noise sources in time-domain data:

```python
from slab_qick_calib.analysis import time_series

# Analyze power spectral density of qubit parameter fluctuations
psd_results = time_series.analyze_qubit_psd(times, values, param_name='T1')
```

**Key features:**
- **Welch PSD estimation**: Robust power spectral density calculation
- **Noise model fitting**: Power law, white noise + Lorentzian models
- **Frequency-domain analysis**: Identify characteristic noise frequencies

### Qubit Parameter Calculations (`analysis/qubit_params.py`)

Circuit QED theory functions for calculating qubit parameters:

```python
from slab_qick_calib.analysis import qubit_params

# Calculate dispersive shift
chi_calc = qubit_params.chi(alpha=-200, Delta=1000, g=50)  # MHz

# Calculate Purcell decay rate
T1_purcell = qubit_params.T1p(kappa=5, Delta=1000, g=50)  # μs
```

**Key features:**
- **Circuit QED formulas**: Dispersive shift, Purcell effect, cross-Kerr coupling
- **Hamiltonian analysis**: Extract parameters from device configuration
- **Coherence calculations**: Predict T1, T2 from theoretical models

## Calibration Tools

### Advanced Readout Analysis (`calib/readout_helpers.py`)

Sophisticated tools for analyzing and optimizing single-shot readout:

```python
from slab_qick_calib.calib import readout_helpers

# Analyze single-shot histograms with automatic IQ rotation
results, metrics = readout_helpers.analyze_single_shot_histograms(
    data, plot=True, use_f_state=True
)

# Fit histograms to extract readout parameters
iq_data, fit_results, bins, hist = readout_helpers.fit_single_shot(data, plot=True)
```

**Key features:**
- **Automatic IQ rotation**: Optimizes readout axis for maximum discrimination
- **Gaussian fitting**: Extracts ground and excited state distributions
- **T1 decay modeling**: Accounts for relaxation during measurement
- **Discrimination metrics**: Calculates fidelity, threshold, and error rates

### Long-term Stability Tracking (`calib/time_tracking.py`)

Automated protocols for monitoring qubit stability over extended periods:

```python
from slab_qick_calib.calib import time_tracking

# Run automated stability tracking for 12 hours
time_tracking.time_tracking(
    qubit_list=[0, 1], 
    cfg_dict=cfg_dict, 
    total_time=12,  # hours
    fast=True
)
```

**Key features:**
- **Automated measurements**: T1, T2, frequency tracking with periodic updates
- **Fast protocols**: Optimized measurement sequences for high time resolution
- **Statistical analysis**: Automatic calculation of means, standard deviations, and drift rates
- **Adaptive tuning**: Optional automatic recentering of qubit frequency

## Testing

You can verify that the package is installed correctly by importing it in Python:

```python
import slab_qick_calib
print("Package imported successfully!")

# Test importing key modules
from slab_qick_calib import experiments, calib, exp_handling, analysis
print("Core modules imported successfully!")
