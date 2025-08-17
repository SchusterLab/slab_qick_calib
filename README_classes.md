# QICK Program and Experiment Classes

This document provides an overview of the base classes for creating quantum experiments using the QICK framework, along with detailed guidance on how to write new experiments. These classes are designed to be extended for specific experiment implementations.

## Table of Contents
1. [Class Overview](#class-overview)
2. [How to Write a New Experiment](#how-to-write-a-new-experiment)
3. [2D Experiment Implementation Guide](#2d-experiment-implementation-guide)
4. [Practical Examples](#practical-examples)
5. [Configuration Patterns](#configuration-patterns)
6. [Analysis and Display Patterns](#analysis-and-display-patterns)

## Class Overview

## QickProgram

The `QickProgram` class is the base for single-qubit experiments. It extends `AveragerProgramV2` from the QICK library to provide a higher-level interface for pulse generation, measurement, and data collection.

### Methods

- `__init__(soccfg, final_delay=50, cfg={})`: Initializes the program with hardware and experiment configurations.
- `_initialize(cfg, readout="standard")`: Sets up ADC and DAC channels, and configures readout and qubit control pulses.
- `_body(cfg)`: Defines the main experiment sequence. This method should be overridden in subclasses.
- `measure(cfg)`: Implements the standard measurement sequence, including readout, LO pulse application, and optional active reset.
- `make_pulse(pulse, name)`: Creates a pulse (Gaussian, flat-top, or constant) and adds it to the program.
- `make_cfg_pulse(q, freq, name)`: Creates a π pulse for a specified qubit.
- `collect_shots(offset=0, single=True)`: Retrieves raw I/Q data from the ADC.
- `reset(i)`: Performs active qubit reset by measuring the qubit state and applying a conditional π pulse.

## QickProgram2Q

The `QickProgram2Q` class is the base for two-qubit experiments, extending `AveragerProgramV2` to handle multiple qubits.

### Methods

- `__init__(soccfg, final_delay=50, cfg={})`: Initializes the program for two-qubit experiments.
- `_initialize(cfg, readout="standard")`: Sets up hardware channels for all qubits involved in the experiment.
- `_body(cfg)`: Defines the main experiment sequence. This method should be overridden in subclasses.
- `make_pulse(q, pulse, name)`: Creates a pulse for a specified qubit.
- `make_cfg_pulse(q, i, freq, name)`: Creates a π pulse for a specified qubit.
- `collect_shots(offset=[0, 0])`: Retrieves raw I/Q data from all ADCs.
- `reset(i)`: Performs active qubit reset for multiple qubits in parallel.

## QickExperiment

The `QickExperiment` class is the base for running quantum experiments on QICK hardware. It handles experiment configuration, data acquisition, analysis, and visualization.

### Methods

- `__init__(cfg_dict, qi=0, prefix="QickExp", fname=None, progress=None, check_params=True)`: Initializes the experiment with configuration parameters.
- `acquire(prog_name, progress=True, get_hist=True, single=True, compact=False)`: Acquires data by running a specified `QickProgram`.
- `analyze(fitfunc, fitterfunc, data=None, fit=True, use_i=None, get_hist=True, verbose=True, inds=None, **kwargs)`: Analyzes measurement data by fitting it to a theoretical model.
- `display(data=None, ax=None, plot_all=False, title="", xlabel="", fit=True, show_hist=False, rescale=False, fitfunc=None, caption_params=[], debug=False, **kwargs)`: Displays measurement results with optional fit curves.
- `make_hist(prog, single=True)`: Generates a histogram of single-shot measurement results.
- `qubit_run(qi=0, progress=True, analyze=True, display=True, save=True, print=False, min_r2=0.1, max_err=1, disp_kwargs=None, **kwargs)`: A wrapper for `run` that handles qubit-specific configurations.
- `run(progress=True, analyze=True, display=True, save=True, min_r2=0.1, max_err=1, disp_kwargs=None, **kwargs)`: Runs the complete experiment workflow.
- `save_data(data=None, verbose=False)`: Saves experiment data to disk.
- `print()`: Prints the experimental configuration.
- `get_status(max_err=1, min_r2=0.1)`: Determines if the experiment was successful based on fit quality.
- `get_params(prog)`: Gets swept parameter values from the program.
- `check_params(params_def)`: Checks for unexpected parameters in the configuration.
- `configure_reset()`: Configures parameters for active reset.
- `get_freq(fit=True)`: Calculates the correct frequency when using mixers.
- `scale_ge()`: Scales data to represent excited state probability (0 to 1) based on histogram analysis.

## QickExperimentLoop

The `QickExperimentLoop` class extends `QickExperiment` for loop-based parameter sweeps, generally because some parameters can't be adjusted with QickSweeps -- primarily for nonlinear sweep, or parameters that can't be swept that way (as with Gaussian pulses)

### Methods

- `__init__(cfg_dict=None, prefix="QickExp", progress=False, qi=0)`: Initializes the QickExperimentLoop.
- `acquire(prog_name, x_sweep, progress=True, hist=False)`: Acquires data by running the program for each point in a parameter sweep.
- `stow_data(iq_list, data)`: Processes and stores I/Q data from a measurement.
- `make_hist(shots_i)`: Generates a histogram from collected shots.

## QickExperiment2D

The `QickExperiment2D` class extends `QickExperimentLoop` for 2D parameter sweeps with maximum generality, remaking the program for each line of the y-sweep. 

### Methods

- `__init__(cfg_dict=None, prefix="QickExp", progress=None, qi=0)`: Initializes the QickExperiment2D.
- `acquire(prog_name, y_sweep, progress=True)`: Acquires data for a 2D parameter sweep.
- `analyze(fitfunc, fitterfunc, data=None, fit=False, **kwargs)`: Analyzes 2D data by fitting each row to a model function.
- `display(data=None, ax=None, plot_both=False, plot_amps=False, title="", xlabel="", ylabel="", **kwargs)`: Displays 2D results as a heatmap.

## QickExperiment2DSimple

The `QickExperiment2DSimple` class is a simplified version of `QickExperiment2D` for experiments where the program is made once, and in each line of the y-sweep aspects of the cfg.expt list of params are changed. 

### Methods

- `__init__(cfg_dict=None, prefix="QickExp", progress=None, qi=0)`: Initializes the QickExperiment2DSimple.
- `acquire(y_sweep, progress=False)`: Acquires data for a 2D parameter sweep using a nested experiment.

## QickExperiment2DSweep

The `QickExperiment2DSweep` class extends `QickExperiment` for 2D parameter sweeps where both dimension of sweep are performed on the QICK, instead of the y-axis being swept in python. 

### Methods

- `analyze(fitfunc, fitterfunc, data=None, fit=False, **kwargs)`: Analyzes 2D data by fitting each row to a model function.
- `display(data=None, ax=None, plot_both=False, plot_amps=False, title="", xlabel="", ylabel="", **kwargs)`: Displays 2D results as a heatmap.

---

# How to Write a New Experiment

This section provides step-by-step guidance for creating new quantum experiments using the QICK framework.

## Basic Experiment Structure

Every QICK experiment consists of two main components:

1. **QickProgram**: Defines the pulse sequence that runs on the QICK hardware
2. **QickExperiment**: Handles data acquisition, analysis, and visualization

## Step-by-Step Process

### 1. Create the Program Class

First, create a program class that inherits from `QickProgram` (single qubit) or `QickProgram2Q` (multi-qubit):

```python
from ..general.qick_program import QickProgram
from qick.asm_v2 import QickSweep1D

class MyExperimentProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        
        # Add any loops for parameter sweeps
        self.add_loop("my_loop", cfg.expt.expts)
        
        # Initialize standard components
        super()._initialize(cfg, readout="standard")
        
        # Create any custom pulses
        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)
        
        # Configure readout if using dynamic ADC
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        
        # Define your pulse sequence here
        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        self.delay_auto(t=cfg.expt["wait_time"], tag="wait")
        
        # Measure the qubit
        super().measure(cfg)
```

### 2. Create the Experiment Class

Next, create an experiment class that inherits from the appropriate base class:

```python
from ..general.qick_experiment import QickExperiment
from ...analysis import fitting as fitter

class MyExperiment(QickExperiment):
    def __init__(self, cfg_dict, qi=0, go=True, params={}, prefix=None, **kwargs):
        if prefix is None:
            prefix = f"my_experiment_qubit{qi}"
        
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, qi=qi, **kwargs)
        
        # Define default parameters
        params_def = {
            "reps": 2 * self.reps,
            "rounds": self.rounds,
            "expts": 50,
            "start": 0,
            "span": 10,  # Parameter range
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        
        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        
        if go:
            super().qubit_run(qi=qi, **kwargs)

    def acquire(self, progress=False, debug=False):
        # Define parameter metadata for plotting
        self.param = {"label": "wait", "param": "t", "param_type": "time"}
        
        # Set up parameter sweep
        self.cfg.expt.wait_time = QickSweep1D(
            "my_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        
        # Run the program
        super().acquire(MyExperimentProgram, progress=progress)
        return self.data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data
        
        # Choose appropriate fitting function
        fitfunc = fitter.expfunc  # or fitter.sinefunc, etc.
        fitterfunc = fitter.fitexp  # or fitter.fitsine, etc.
        
        super().analyze(fitfunc, fitterfunc, data, **kwargs)
        return data

    def display(self, data=None, fit=True, **kwargs):
        title = f"My Experiment Q{self.cfg.expt.qubit[0]}"
        xlabel = "Parameter (units)"
        
        caption_params = [
            {"index": 0, "format": "Parameter: {val:.3f} ± {err:.3f}"},
        ]
        
        super().display(
            data=data, title=title, xlabel=xlabel, 
            fitfunc=self.fitfunc, caption_params=caption_params,
            fit=fit, **kwargs
        )
```

### 3. Key Implementation Points

- **Parameter Sweeps**: Use `QickSweep1D` for parameters swept on the QICK hardware
- **Configuration**: Always define `params_def` with sensible defaults
- **Parameter Validation**: Call `super().check_params(params_def)` to catch typos
- **Metadata**: Set `self.param` to describe what parameter is being swept, so that QICK will return the sweep points used. 
- **Fitting**: Choose appropriate fitting functions from the `fitter` module
- **Display**: Provide meaningful titles, labels, and caption parameters

---

# 2D Experiment Implementation Guide

2D experiments sweep two parameters to create a 2D map of measurement results. There are three main approaches, each suited for different scenarios.

## When to Use Each 2D Experiment Type

### QickExperiment2D
- **Use when**: Maximum flexibility needed, y-parameter affects pulse sequence
- **Characteristics**: Remakes the program for each y-sweep point
- **Examples**: Frequency vs. time stability, parameter vs. flux
- **Performance**: Slower but most flexible

### QickExperiment2DSimple  
- **Use when**: Y-parameter only affects configuration, not pulse sequence
- **Characteristics**: Uses nested experiments, changes `cfg.expt` parameters
- **Examples**: T1 vs. time, parameter vs. power
- **Performance**: Moderate speed, good for stability measurements

### QickExperiment2DSweep
- **Use when**: Both parameters can be swept on QICK hardware
- **Characteristics**: Both dimensions swept in single program execution
- **Examples**: Frequency vs. amplitude sweeps
- **Performance**: Fastest, but limited to QICK-sweepable parameters

## Implementation Patterns

### QickExperiment2D Pattern

```python
class MyExperiment2D(QickExperiment2D):
    def __init__(self, cfg_dict, qi=0, go=True, params={}, **kwargs):
        super().__init__(cfg_dict=cfg_dict, qi=qi, **kwargs)
        
        params_def = {
            "expts": 50,      # X-axis points
            "y_pts": 100,     # Y-axis points  
            "x_span": 10,     # X-axis range
            "y_span": 5,      # Y-axis range
            # ... other parameters
        }
        
        self.cfg.expt = {**params_def, **params}
        
        if go:
            super().run(**kwargs)

    def acquire(self, progress=True):
        # Define y-axis sweep
        y_pts = np.linspace(0, self.cfg.expt.y_span, self.cfg.expt.y_pts)
        y_sweep = [{"pts": y_pts, "var": "y_parameter"}]
        
        # Run 2D acquisition
        super().acquire(MyExperimentProgram, y_sweep, progress=progress)
        return self.data

    def display(self, data=None, **kwargs):
        title = "My 2D Experiment"
        xlabel = "X Parameter (units)"
        ylabel = "Y Parameter (units)"
        
        super().display(
            data=data, title=title, xlabel=xlabel, ylabel=ylabel, **kwargs
        )
```

### QickExperiment2DSimple Pattern

```python
class MyExperiment2DSimple(QickExperiment2DSimple):
    def __init__(self, cfg_dict, qi=0, go=True, params={}, **kwargs):
        super().__init__(cfg_dict=cfg_dict, **kwargs)
        
        # Create nested experiment
        self.expt = MyExperiment(cfg_dict, qi, go=False, params=params)
        
        params_def = {
            "sweep_pts": 200,  # Number of y-axis points
        }
        
        # Merge with nested experiment parameters
        params = {**params_def, **params}
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params
        
        if go:
            super().run(**kwargs)

    def acquire(self, progress=False):
        # For time-based sweeps, use "count" variable
        sweep_pts = np.arange(self.cfg.expt["sweep_pts"])
        y_sweep = [{"pts": sweep_pts, "var": "count"}]
        
        super().acquire(y_sweep, progress=progress)
        return self.data
```

## Common 2D Experiment Patterns

### Time Stability Measurements
```python
# Y-axis is time, X-axis is the experimental parameter
y_sweep = [{"pts": np.arange(num_time_points), "var": "count"}]
```

### Parameter vs. Parameter Sweeps
```python
# Both axes are experimental parameters
power_pts = np.linspace(min_power, max_power, num_points)
y_sweep = [{"pts": power_pts, "var": "power_parameter"}]
```

### Multi-Parameter Sweeps
```python
# Sweep multiple parameters simultaneously
freq_pts = np.linspace(freq_min, freq_max, num_points)
power_pts = np.linspace(power_min, power_max, num_points)
y_sweep = [
    {"pts": freq_pts, "var": "frequency"},
    {"pts": power_pts, "var": "power"}
]
```

---

# Practical Examples

## Example 1: Simple 1D Rabi Experiment

```python
from qick.asm_v2 import QickSweep1D
from ..general.qick_program import QickProgram
from ..general.qick_experiment import QickExperiment
from ...analysis import fitting as fitter

class RabiProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        self.add_loop("gain_loop", cfg.expt.expts)
        super()._initialize(cfg, readout="standard")
        
        # Create variable-amplitude pulse
        pulse = {
            "sigma": cfg.device.qubit.pulses.pi_ge.sigma[cfg.expt.qubit[0]],
            "freq": cfg.device.qubit.f_ge[cfg.expt.qubit[0]],
            "gain": cfg.expt.gain,  # This will be swept
            "phase": 0,
            "type": "gauss",
        }
        super().make_pulse(pulse, "rabi_pulse")

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        
        self.pulse(ch=self.qubit_ch, name="rabi_pulse", t=0)
        super().measure(cfg)

class RabiExperiment(QickExperiment):
    def __init__(self, cfg_dict, qi=0, go=True, params={}, **kwargs):
        super().__init__(cfg_dict=cfg_dict, prefix=f"rabi_qubit{qi}", qi=qi, **kwargs)
        
        params_def = {
            "reps": 2 * self.reps,
            "rounds": self.rounds,
            "expts": 50,
            "start": 0,
            "span": 32000,  # Gain range
            "qubit": [qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
        }
        
        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        
        if go:
            super().qubit_run(qi=qi, **kwargs)

    def acquire(self, progress=False):
        self.param = {"label": "rabi_pulse", "param": "gain", "param_type": "pulse"}
        
        self.cfg.expt.gain = QickSweep1D(
            "gain_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        
        super().acquire(RabiProgram, progress=progress)
        return self.data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data
        
        self.fitfunc = fitter.sinefunc
        self.fitterfunc = fitter.fitsine
        super().analyze(self.fitfunc, self.fitterfunc, data, **kwargs)
        
        # Extract π pulse amplitude
        data["pi_gain"] = data["best_fit"][2] + data["best_fit"][3] / 2
        return data

    def display(self, data=None, fit=True, **kwargs):
        title = f"Rabi Q{self.cfg.expt.qubit[0]}"
        xlabel = "Pulse Gain"
        
        caption_params = [
            {"index": 3, "format": "Period: {val:.1f} ± {err:.1f}"},
            {"index": 2, "format": "Offset: {val:.1f} ± {err:.1f}"},
        ]
        
        super().display(
            data=data, title=title, xlabel=xlabel,
            fitfunc=self.fitfunc, caption_params=caption_params,
            fit=fit, **kwargs
        )
```

## Example 2: 2D T1 Stability Measurement

```python
class T1_2D(QickExperiment2DSimple):
    def __init__(self, cfg_dict, qi=0, go=True, params={}, **kwargs):
        super().__init__(cfg_dict=cfg_dict, prefix=f"t1_2d_qubit{qi}", **kwargs)
        
        # Create nested T1 experiment
        from .t1 import T1Experiment
        self.expt = T1Experiment(cfg_dict, qi, go=False, params=params, check_params=False)
        
        params_def = {
            "sweep_pts": 200,  # Number of time points
        }
        
        # Merge parameters
        params = {**params_def, **params}
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = params
        
        if go:
            super().run(**kwargs)

    def acquire(self, progress=False):
        # Time-based sweep
        sweep_pts = np.arange(self.cfg.expt["sweep_pts"])
        y_sweep = [{"pts": sweep_pts, "var": "count"}]
        
        super().acquire(y_sweep, progress=progress)
        return self.data

    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data
        
        fitfunc = fitter.expfunc
        fitterfunc = fitter.fitexp
        super().analyze(fitfunc, fitterfunc, data)
        return data

    def display(self, data=None, **kwargs):
        title = f"T1 2D Q{self.cfg.expt.qubit[0]}"
        xlabel = "Wait Time (μs)"
        ylabel = "Time (hours)"
        
        super().display(
            data=data, title=title, xlabel=xlabel, ylabel=ylabel, **kwargs
        )
```

---

# Configuration Patterns

## Parameter Definition Best Practices

### Standard Parameter Structure
```python
params_def = {
    # Measurement parameters
    "reps": 2 * self.reps,           # Inner loop repetitions
    "rounds": self.rounds,            # Outer loop averages
    "expts": 50,                     # Number of sweep points
    
    # Sweep parameters
    "start": 0,                      # Sweep start value
    "span": 10,                      # Sweep range
    
    # Hardware parameters
    "qubit": [qi],                   # Qubit indices
    "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],  # Readout channel
    
    # Experiment-specific parameters
    "pulse_length": 0.1,             # Custom parameter
    "active_reset": self.cfg.device.readout.active_reset[qi],
}
```

### Parameter Validation
```python
# Always validate parameters to catch typos
self.cfg.expt = {**params_def, **params}
super().check_params(params_def)
```

### Style-Based Parameter Adjustment
```python
if style == "fine":
    params_def["rounds"] = params_def["rounds"] * 2
elif style == "fast":
    params_def["expts"] = 30
```

## Hardware Configuration Integration

### Qubit-Specific Defaults
```python
params_def = {
    "span": 3.7 * self.cfg.device.qubit.T1[qi],  # Scale with T1
    "active_reset": self.cfg.device.readout.active_reset[qi],
    "final_delay": self.cfg.device.readout.final_delay[qi],
}
```

### Multi-Qubit Parameters
```python
params_def = {
    "qubit": qi,  # qi can be [0, 1] for two qubits
    "qubit_chan": [self.cfg.hw.soc.adcs.readout.ch[q] for q in qi],
    "active_reset": np.all([self.cfg.device.readout.active_reset[q] for q in qi]),
}
```

---

# Analysis and Display Patterns

## Common Fitting Functions

### Available Fitting Functions
```python
from ...analysis import fitting as fitter

# Exponential decay (T1, T2)
fitfunc = fitter.expfunc
fitterfunc = fitter.fitexp

# Sinusoidal (Rabi, Ramsey)
fitfunc = fitter.sinefunc  
fitterfunc = fitter.fitsine

# Gaussian (spectroscopy)
fitfunc = fitter.gaussfunc
fitterfunc = fitter.fitgauss

# Lorentzian (resonances)
fitfunc = fitter.lorentzfunc
fitterfunc = fitter.fitlorentz
```

### Custom Analysis
```python
def analyze(self, data=None, **kwargs):
    if data is None:
        data = self.data
    
    # Standard fitting
    super().analyze(self.fitfunc, self.fitterfunc, data, **kwargs)
    
    # Extract derived parameters
    data["derived_param"] = some_function(data["best_fit"])
    
    return data
```

## Display Customization

### Caption Parameters
```python
caption_params = [
    {"index": 0, "format": "Offset: {val:.3f} ± {err:.3f} units"},
    {"index": 1, "format": "Amplitude: {val:.3f} ± {err:.3f} units"},
    {"index": 2, "format": "T1: {val:.3f} ± {err:.3f} μs"},
]
```

### Display Options
```python
def display(self, data=None, **kwargs):
    # For untuned qubits, show all quadratures
    if not self.cfg.device.qubit.tuned_up[qi]:
        kwargs.setdefault("plot_all", True)
    
    # For rescaled data
    if self.cfg.device.readout.rescale[qi]:
        kwargs.setdefault("rescale", True)
    
    super().display(data=data, **kwargs)
```

### 2D Display Options
```python
# Single quadrature (default)
super().display(data=data, title=title, xlabel=xlabel, ylabel=ylabel)

# Both I and Q quadratures  
super().display(data=data, plot_both=True, ...)

# Amplitude and phase
super().display(data=data, plot_amps=True, ...)
```

This comprehensive guide should help you implement new experiments efficiently while following established patterns and best practices.
