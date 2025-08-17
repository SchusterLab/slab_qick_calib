# Complete Guide to Writing New QICK Experiments

This comprehensive guide teaches you how to create quantum experiments using the `slab_qick_calib` framework. By the end of this guide, you'll understand how to write pulse sequences, manage timing, create experiment classes, and extend to 2D experiments.

## Table of Contents
1. [Framework Overview](#framework-overview)
2. [Your First Complete Experiment](#your-first-complete-experiment)  
3. [Understanding Pulse Creation](#understanding-pulse-creation)
4. [Mastering Timing Control](#mastering-timing-control)
5. [Experiment Class Architecture](#experiment-class-architecture)
6. [Configuration and Parameters](#configuration-and-parameters)
7. [Data Analysis and Display](#data-analysis-and-display)
8. [Creating 2D Experiments](#creating-2d-experiments)
9. [What to Copy vs What to Write](#what-to-copy-vs-what-to-write)
10. [Common Patterns and Examples](#common-patterns-and-examples)
11. [Troubleshooting Guide](#troubleshooting-guide)

## Framework Overview

The `slab_qick_calib` framework is built around two main components:

### QickProgram Classes
- **Purpose**: Define the actual pulse sequences that run on QICK hardware
- **Inherit from**: `QickProgram` (single qubit) or `QickProgram2Q` (two qubits)  
- **Key methods**: `_initialize()`, `_body()`
- **What they do**: Create pulses, manage timing, handle measurements

### QickExperiment Classes  
- **Purpose**: Manage experiment workflow, analysis, and visualization
- **Inherit from**: `QickExperiment`, `QickExperiment2DSimple`, etc.
- **Key methods**: `__init__()`, `acquire()`, `analyze()`, `display()`
- **What they do**: Set parameters, run programs, fit data, create plots

## Your First Complete Experiment

Let's create a simple pulse-probe experiment step by step. This experiment applies a variable-length pulse and then measures the qubit.

### Step 1: Create the QickProgram

```python
from ..general.qick_program import QickProgram
from ...exp_handling.datamanagement import AttrDict
from qick.asm_v2 import QickSweep1D

class MyPulseProbeProgram(QickProgram):
    """
    Simple pulse-probe experiment: apply a variable-length pulse then measure
    """
    
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        """Set up pulses and loops for the experiment"""
        cfg = AttrDict(self.cfg)
        
        # Standard setup: readout configuration and sweep loop
        super()._initialize(cfg, readout="standard")
        self.add_loop("pulse_loop", cfg.expt.expts)
        
        # Create the variable pulse
        pulse = {
            "freq": cfg.expt.pulse_freq,     # Pulse frequency  
            "gain": cfg.expt.pulse_gain,     # Pulse amplitude
            "length": cfg.expt.pulse_length, # This will be swept
            "phase": 0,                      # Pulse phase
            "type": "const",                 # Constant amplitude pulse
        }
        super().make_pulse(pulse, "probe_pulse")

    def _body(self, cfg):
        """Define the pulse sequence"""
        cfg = AttrDict(self.cfg)
        
        # Configure dynamic readout if needed
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
        
        # Apply the probe pulse
        self.pulse(ch=self.qubit_ch, name="probe_pulse", t=0)
        
        # Small delay before measurement
        self.delay_auto(t=0.01, tag="settle")  
        
        # Measure the qubit
        super().measure(cfg)
```

### Step 2: Create the QickExperiment

```python
from ..general.qick_experiment import QickExperiment
from ...analysis import fitting as fitter

class MyPulseProbeExperiment(QickExperiment):
    """
    Experiment class for pulse-probe measurement
    Sweeps pulse length and measures qubit response
    """
    
    def __init__(self, cfg_dict, qi=0, go=True, params={}, **kwargs):
        # Set up file naming
        prefix = f"pulse_probe_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, qi=qi, **kwargs)
        
        # Define default parameters
        params_def = {
            "expts": 50,                    # Number of length points
            "reps": self.reps,              # Measurement repetitions
            "rounds": self.rounds,          # Averaging rounds  
            "pulse_freq": self.cfg.device.qubit.f_ge[qi],  # Use qubit frequency
            "pulse_gain": 0.1,              # Low amplitude probe
            "start_length": 0.01,           # Minimum pulse length (μs)
            "max_length": 1.0,              # Maximum pulse length (μs) 
            "qubit": [qi],                  # Qubit list
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],  # Readout channel
        }
        
        # Merge with user parameters
        self.cfg.expt = {**params_def, **params}
        super().check_params(params_def)
        
        # Run experiment if requested
        if go:
            super().qubit_run(qi=qi, **kwargs)

    def acquire(self, progress=False, debug=False):
        """Run the experiment and collect data"""
        # Set up the length sweep
        self.cfg.expt["pulse_length"] = QickSweep1D(
            "pulse_loop", 
            self.cfg.expt.start_length, 
            self.cfg.expt.max_length
        )
        
        # Define what parameter we're sweeping (for plotting)
        self.param = {
            "label": "probe_pulse", 
            "param": "length", 
            "param_type": "pulse"
        }
        
        # Run the program
        super().acquire(MyPulseProbeProgram, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        """Analyze the data (no fitting for this simple example)"""
        if data is None:
            data = self.data
        # For this example, we don't fit - just return the data
        return data

    def display(self, data=None, **kwargs):
        """Display the results"""
        if data is None:
            data = self.data
            
        title = f"Pulse-Probe Q{self.cfg.expt.qubit[0]}"
        xlabel = "Pulse Length (μs)"
        
        super().display(
            data=data,
            title=title, 
            xlabel=xlabel,
            fit=False,  # No fitting for this example
            **kwargs
        )
```

### Step 3: Running Your Experiment

```python
# In a Jupyter notebook:
import os
os.chdir('/path/to/your/slab_qick_calib')

from experiments.single_qubit.my_pulse_probe import MyPulseProbeExperiment

# Load your configuration
cfg_dict = {
    "soc": soccfg,           # Your QICK configuration
    "expt_path": "data/",    # Where to save data
    "cfg_file": config_file, # Your config YAML
    "im": im                 # Instrument manager
}

# Run the experiment
exp = MyPulseProbeExperiment(
    cfg_dict=cfg_dict,
    qi=0,                    # Qubit 0
    params={
        "max_length": 2.0,   # Override default max length
        "expts": 100         # More points
    }
)

# Data is automatically saved and plotted
# Access results with exp.data
```

## Understanding Pulse Creation

Pulses are the building blocks of quantum experiments. The framework supports several pulse types:

### Basic Pulse Parameters

Every pulse needs these core parameters:
```python
pulse = {
    "freq": 5000,      # Frequency in MHz
    "gain": 0.5,       # Amplitude (0-1 for most channels)  
    "phase": 0,        # Phase in degrees
    "type": "const"    # Pulse type
}
```

### Pulse Types

#### 1. Constant Pulses (`"const"`)
```python
pulse = {
    "freq": cfg.device.qubit.f_ge[qi],
    "gain": 0.3,
    "length": 0.1,     # Duration in μs
    "phase": 0,
    "type": "const"
}
```

#### 2. Gaussian Pulses (`"gauss"`)
```python  
pulse = {
    "freq": cfg.device.qubit.f_ge[qi],
    "gain": 0.8,
    "sigma": 0.02,          # Width of Gaussian in μs
    "sigma_inc": 4,         # Total length = sigma * sigma_inc
    "phase": 0,
    "type": "gauss"
}
```

#### 3. Flat-Top Pulses (`"flat_top"`)
```python
pulse = {
    "freq": cfg.device.qubit.f_ge[qi], 
    "gain": 0.5,
    "length": 0.2,          # Total pulse length
    "ramp_sigma": 0.01,     # Rise/fall width
    "ramp_sigma_inc": 3,    # Rise/fall shaping
    "phase": 0,
    "type": "flat_top"
}
```

### Creating Pulses in Your Program

```python
def _initialize(self, cfg):
    super()._initialize(cfg, readout="standard")
    
    # Method 1: Manual pulse creation
    pulse = {
        "freq": cfg.expt.drive_freq,
        "gain": cfg.expt.drive_gain, 
        "sigma": cfg.expt.pulse_sigma,
        "phase": 0,
        "type": "gauss"
    }
    super().make_pulse(pulse, "drive_pulse")
    
    # Method 2: Use pre-configured π pulse
    super().make_cfg_pulse(
        cfg.expt.qubit[0],           # Qubit index
        cfg.device.qubit.f_ge,       # Frequency dict
        "pi_ge"                      # Name
    )
```

## Mastering Timing Control

Precise timing is crucial in quantum experiments. Here's how to manage it:

### Basic Timing Methods

```python
def _body(self, cfg):
    # Apply pulse at time t=0
    self.pulse(ch=self.qubit_ch, name="pi_pulse", t=0)
    
    # Wait for a specific time
    self.delay_auto(t=0.5, tag="wait_time")  # 0.5 μs delay
    
    # Apply another pulse (automatically after the delay)
    self.pulse(ch=self.qubit_ch, name="readout_pulse", t=0)
    
    # Measure (includes trigger timing offset automatically)
    super().measure(cfg)
```

### Swept Timing Parameters

```python
class T1Program(QickProgram):
    def _initialize(self, cfg):
        super()._initialize(cfg, readout="standard")
        
        # Create sweep loop for wait times
        self.add_loop("wait_loop", cfg.expt.expts)
        
        # π pulse to excite qubit
        super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")
    
    def _body(self, cfg):
        # Excite qubit
        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        
        # Variable wait time (swept parameter)
        self.delay_auto(t=cfg.expt.wait_time + 0.01, tag="t1_wait")
        
        # Measure decay
        super().measure(cfg)

# In the experiment's acquire method:
def acquire(self, progress=False, debug=False):
    # Create the time sweep
    self.cfg.expt.wait_time = QickSweep1D(
        "wait_loop",                    # Loop name from _initialize
        self.cfg.expt.start,           # Start time 
        self.cfg.expt.start + self.cfg.expt.span  # End time
    )
    
    # Tell the framework what we're sweeping
    self.param = {
        "label": "t1_wait",            # Tag from delay_auto
        "param": "t",                  # Time parameter
        "param_type": "time"           # Parameter type
    }
```

### Advanced Timing Patterns

```python
def _body(self, cfg):
    # Ramsey sequence with swept phase accumulation time
    self.pulse(ch=self.qubit_ch, name="pi2_pulse", t=0)  # π/2 pulse
    
    self.delay_auto(t=cfg.expt.ramsey_time, tag="ramsey_wait")
    
    # Second π/2 with swept phase
    self.pulse(ch=self.qubit_ch, name="pi2_pulse_phase", t=0) 
    
    # Short delay before measurement
    self.delay_auto(t=0.01, tag="settle")
    
    super().measure(cfg)
```

## Experiment Class Architecture

Understanding the experiment class structure helps you build robust experiments:

### The `__init__` Method

This is where you set up parameters and configuration:

```python
def __init__(self, cfg_dict, qi=0, go=True, params={}, prefix=None, 
             progress=True, display=True, style="", **kwargs):
    
    # 1. Set up the prefix for data files
    if prefix is None:
        prefix = f"my_experiment_qubit{qi}"
    
    # 2. Call parent constructor
    super().__init__(cfg_dict=cfg_dict, prefix=prefix, qi=qi, **kwargs)
    
    # 3. Define default parameters
    params_def = {
        # Measurement parameters
        "expts": 60,                    # Number of points
        "reps": self.reps,              # From base class  
        "rounds": self.rounds,          # From base class
        
        # Pulse parameters (get from device config)
        "freq": self.cfg.device.qubit.f_ge[qi],
        "gain": self.cfg.device.qubit.pulses.pi_ge.gain[qi],
        "sigma": self.cfg.device.qubit.pulses.pi_ge.sigma[qi],
        
        # Experiment-specific parameters
        "start": 0,
        "span": 10,
        
        # Hardware parameters  
        "qubit": [qi],
        "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
    }
    
    # 4. Handle style modifications
    if style == "fine":
        params_def["rounds"] *= 2      # More averages
    elif style == "fast": 
        params_def["expts"] = 30       # Fewer points
        
    # 5. Merge parameters
    self.cfg.expt = {**params_def, **params}
    super().check_params(params_def)
    
    # 6. Run if requested
    if go:
        super().qubit_run(qi=qi, display=display, progress=progress, **kwargs)
```

### The `acquire` Method

This method runs your program and collects data:

```python
def acquire(self, progress=False, debug=False):
    """Acquire experimental data"""
    
    # 1. Set up parameter sweeps
    if self.cfg.expt.sweep_type == "frequency":
        self.cfg.expt.freq = QickSweep1D(
            "freq_loop",
            self.cfg.expt.start_freq, 
            self.cfg.expt.end_freq
        )
        self.param = {"label": "drive_pulse", "param": "freq", "param_type": "pulse"}
        
    elif self.cfg.expt.sweep_type == "amplitude":
        self.cfg.expt.gain = QickSweep1D(
            "amp_loop",
            self.cfg.expt.min_gain,
            self.cfg.expt.max_gain  
        )
        self.param = {"label": "drive_pulse", "param": "gain", "param_type": "pulse"}
    
    # 2. Run the program
    super().acquire(MyProgram, progress=progress)
    
    return self.data
```

### The `analyze` Method

This method fits your data and extracts parameters:

```python
def analyze(self, data=None, fit=True, **kwargs):
    """Analyze data and extract parameters"""
    if data is None:
        data = self.data
    
    if fit:
        # Choose the fitting function
        if self.cfg.expt.fit_type == "exponential":
            self.fitfunc = fitter.expfunc      # exp(-t/T1)
            self.fitterfunc = fitter.fitexp
            
        elif self.cfg.expt.fit_type == "sinusoidal": 
            self.fitfunc = fitter.sinfunc      # A*sin(2πft + φ) + offset
            self.fitterfunc = fitter.fitsin
            
        elif self.cfg.expt.fit_type == "lorentzian":
            self.fitfunc = fitter.lorenfunc    # Lorentzian lineshape
            self.fitterfunc = fitter.fitloren
        
        # Perform the fit
        super().analyze(data, fit=fit, **kwargs)
        
        # Extract experiment-specific parameters
        if self.cfg.expt.fit_type == "exponential":
            data["T1"] = data["best_fit"][2]   # Decay constant
        elif self.cfg.expt.fit_type == "sinusoidal":
            data["frequency"] = data["best_fit"][1]  # Oscillation frequency
            data["pi_length"] = fitter.fix_phase(data["best_fit"])
    
    return data
```

### The `display` Method

This method creates plots:

```python
def display(self, data=None, fit=True, **kwargs):
    """Display results"""
    if data is None:
        data = self.data
    
    # Set up plot labels
    title = f"My Experiment Q{self.cfg.expt.qubit[0]}"
    
    if self.cfg.expt.sweep_type == "frequency":
        xlabel = "Frequency (MHz)"
    elif self.cfg.expt.sweep_type == "amplitude": 
        xlabel = "Gain (DAC units)"
    else:
        xlabel = "Time (μs)"
    
    # Set up fit parameter display
    caption_params = []
    if fit and "T1" in data:
        caption_params.append({
            "index": 2,  # Parameter index in fit
            "format": "T₁ = {val:.3f} ± {err:.3f} μs"
        })
    
    # Create the plot
    super().display(
        data=data,
        title=title,
        xlabel=xlabel, 
        fit=fit,
        fitfunc=self.fitfunc,
        caption_params=caption_params,
        **kwargs
    )
```

## Configuration and Parameters

### Parameter Hierarchy

Parameters are resolved in this order (highest priority first):
1. **Direct `params` argument**: `MyExperiment(cfg_dict, params={"gain": 0.5})`
2. **Default parameters**: Defined in `params_def` in `__init__`
3. **Device configuration**: Values from YAML config files

### Accessing Configuration Values

```python
# In your program or experiment:
cfg = AttrDict(self.cfg)

# Device parameters (from YAML config)
qubit_freq = cfg.device.qubit.f_ge[qi]
pi_gain = cfg.device.qubit.pulses.pi_ge.gain[qi]
readout_freq = cfg.device.readout.frequency[qi]

# Hardware parameters
qubit_dac = cfg.hw.soc.dacs.qubit.ch[qi]
readout_adc = cfg.hw.soc.adcs.readout.ch[qi]

# Experiment parameters
num_experiments = cfg.expt.expts
sweep_start = cfg.expt.start
```

### Common Parameter Patterns

```python
# Time-based experiments (T1, T2, etc.)
params_def = {
    "start": 0,                    # Start time (μs)
    "span": 3 * self.cfg.device.qubit.T1[qi],  # Based on expected T1
    "expts": 60,                   # Number of time points
}

# Frequency sweeps (spectroscopy)
params_def = {
    "start_freq": self.cfg.device.qubit.f_ge[qi] - 10,  # Start 10 MHz below
    "span_freq": 20,               # 20 MHz span
    "expts": 100,                  # Frequency resolution
}

# Amplitude sweeps (Rabi)  
params_def = {
    "max_gain": self.cfg.device.qubit.pulses.pi_ge.gain[qi] * 2,
    "start_gain": 0.01,            # Minimum gain for linearity
    "num_oscillations": 2.5,       # How many Rabi oscillations to see
}
```

## Data Analysis and Display

### Fitting Functions Available

The framework includes several pre-built fitting functions:

```python
from ...analysis import fitting as fitter

# Exponential decay: A*exp(-t/τ) + offset
self.fitfunc = fitter.expfunc
self.fitterfunc = fitter.fitexp

# Sinusoidal: A*sin(2πft + φ)*exp(-t/T₂) + offset  
self.fitfunc = fitter.sinfunc
self.fitterfunc = fitter.fitsin

# Lorentzian lineshape: A/((f-f₀)² + Γ²) + offset
self.fitfunc = fitter.lorenfunc  
self.fitterfunc = fitter.fitloren

# Gaussian lineshape: A*exp(-(f-f₀)²/2σ²) + offset
self.fitfunc = fitter.gaussfunc
self.fitterfunc = fitter.fitgauss
```

### Custom Analysis Example

```python
def analyze(self, data=None, fit=True, **kwargs):
    if data is None:
        data = self.data
        
    if fit:
        # Standard fitting
        self.fitfunc = fitter.expfunc
        self.fitterfunc = fitter.fitexp
        super().analyze(data, **kwargs)
        
        # Custom analysis - extract coherence time
        T1 = data["best_fit"][2]  # Time constant from exponential fit
        data["T1_seconds"] = T1 * 1e-6  # Convert μs to seconds
        
        # Calculate quality factor if we know the frequency
        if "resonance_freq" in self.cfg.expt:
            f0 = self.cfg.expt.resonance_freq * 1e6  # MHz to Hz
            data["Q_factor"] = 2 * np.pi * f0 * T1 * 1e-6
            
        # Custom uncertainty analysis
        T1_err = np.sqrt(data["fit_err_avgi"][2, 2])  # Diagonal of covariance matrix
        data["T1_uncertainty"] = T1_err
        
    return data
```

### Advanced Display Options

```python
def display(self, data=None, fit=True, plot_all=False, show_hist=False, **kwargs):
    if data is None:
        data = self.data
        
    # Multi-panel display
    if plot_all:
        # Shows I, Q, and amplitude on separate subplots
        super().display(data=data, fit=fit, plot_all=True, **kwargs)
    else:
        # Single panel with custom formatting
        title = f"T₁ Measurement Q{self.cfg.expt.qubit[0]}"
        xlabel = "Wait Time (μs)"
        
        # Custom fit parameter display
        caption_params = [
            {"index": 2, "format": "T₁ = {val:.3f} ± {err:.3f} μs"},
            {"index": 0, "format": "Amplitude = {val:.3f}"},
            {"index": 1, "format": "Offset = {val:.3f}"}
        ]
        
        super().display(
            data=data,
            title=title,
            xlabel=xlabel,
            fit=fit,
            fitfunc=self.fitfunc,
            caption_params=caption_params,
            show_hist=show_hist,  # Show single-shot histogram
            **kwargs
        )
```

## Creating 2D Experiments

2D experiments sweep two parameters to create heatmap visualizations. The framework provides `QickExperiment2DSimple` for this purpose.

### Basic 2D Experiment Structure

```python
from ..general.qick_experiment import QickExperiment2DSimple

class My2DExperiment(QickExperiment2DSimple):
    """
    2D experiment: sweeps frequency (outer) vs amplitude (inner)
    """
    
    def __init__(self, cfg_dict, qi=0, go=True, params={}, **kwargs):
        prefix = f"my_2d_experiment_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, **kwargs)
        
        # Define 2D-specific parameters
        params_def = {
            "freq_span": 20,      # MHz
            "freq_expts": 30,     # Points in frequency
            "start_freq": self.cfg.device.qubit.f_ge[qi] - 10,
        }
        
        # Create the inner experiment (handles x-axis sweep)
        self.expt = MyInnerExperiment(
            cfg_dict, qi=qi, go=False, params=params, check_params=False
        )
        
        # Merge parameters
        params = {**params_def, **params}  
        self.cfg.expt = {**self.expt.cfg.expt, **params}
        
        if go:
            super().run(**kwargs)
    
    def acquire(self, progress=False, debug=False):
        """Acquire 2D data"""
        # Create frequency points for y-axis sweep
        freq_points = np.linspace(
            self.cfg.expt.start_freq,
            self.cfg.expt.start_freq + self.cfg.expt.freq_span,
            self.cfg.expt.freq_expts
        )
        
        # Define the outer (y-axis) sweep
        y_sweep = [{"pts": freq_points, "var": "freq"}]
        
        # Run the 2D acquisition
        super().acquire(y_sweep, progress=progress)
        return self.data
    
    def display(self, data=None, fit=False, **kwargs):
        """Display 2D heatmap"""
        if data is None:
            data = self.data
            
        title = f"2D Experiment Q{self.cfg.expt.qubit[0]}"
        xlabel = "Amplitude (DAC units)"  # From inner experiment
        ylabel = "Frequency (MHz)"        # From outer sweep
        
        super().display(
            data=data,
            title=title,
            xlabel=xlabel, 
            ylabel=ylabel,
            fit=fit,
            **kwargs
        )
```

### Time-Stability 2D Experiments

A common 2D experiment tracks parameter changes over time:

```python
class StabilityExperiment(QickExperiment2DSimple):
    """Track T1 vs time to monitor stability"""
    
    def __init__(self, cfg_dict, qi=0, go=True, params={}, **kwargs):
        prefix = f"t1_stability_qubit{qi}"
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, **kwargs)
        
        params_def = {
            "time_points": 100,    # Number of time points
        }
        
        # The inner experiment is a T1 measurement
        from .t1 import T1Experiment
        self.expt = T1Experiment(
            cfg_dict, qi=qi, go=False, params=params, check_params=False
        )
        
        params = {**params_def, **params}
        self.cfg.expt = {**self.expt.cfg.expt, **params}
        
        if go:
            super().run(**kwargs)
    
    def acquire(self, progress=False, debug=False):
        # Create dummy time points - actual time will be recorded automatically
        time_points = np.arange(self.cfg.expt.time_points)
        
        # Use "count" variable - this tells the framework to use actual time
        y_sweep = [{"pts": time_points, "var": "count"}]
        
        super().acquire(y_sweep, progress=progress)
        return self.data
    
    def analyze(self, data=None, fit=True, **kwargs):
        """Analyze each T1 measurement separately"""
        if data is None:
            data = self.data
            
        if fit:
            # Fit each T1 trace
            self.fitfunc = fitter.expfunc
            self.fitterfunc = fitter.fitexp
            super().analyze(data, **kwargs)
            
            # Extract T1 vs time
            if "fit_avgi" in data:
                t1_values = [fit[2] for fit in data["fit_avgi"]]  # Time constant
                data["t1_vs_time"] = np.array(t1_values)
                data["mean_t1"] = np.mean(t1_values)
                data["std_t1"] = np.std(t1_values)
        
        return data
```

## What to Copy vs What to Write

### What You Can Copy Directly

**From existing experiments:**
- Basic `_initialize()` structure
- Standard readout configuration: `super()._initialize(cfg, readout="standard")`
- Loop setup: `self.add_loop("sweep_loop", cfg.expt.expts)`  
- Measurement: `super().measure(cfg)`
- Parameter structure in `__init__`: `params_def` dictionary setup
- Standard imports: `AttrDict`, `QickSweep1D`, base classes
- File naming patterns: `prefix = f"experiment_name_qubit{qi}"`
- Basic display elements: `title`, `xlabel` setup

**From the framework base classes:**
- Readout configuration (already handled by `super()._initialize()`)
- Data acquisition loop (`super().acquire()`)
- Basic fitting infrastructure (`super().analyze()`)
- Plot generation (`super().display()`)
- File I/O and data management

### What You Must Write Yourself

**Program-specific (`QickProgram` subclass):**
- Pulse sequence logic in `_body()`
- Experiment-specific pulse creation in `_initialize()`
- Custom timing patterns and delays
- Conditional logic (if/else for different experiment modes)
- Custom pulse parameters and calculations

**Experiment-specific (`QickExperiment` subclass):**
- Parameter default values in `params_def`
- Sweep configuration in `acquire()`
- Custom analysis calculations in `analyze()` 
- Experiment-specific fit parameter extraction
- Custom plot labels, titles, and formatting
- Parameter validation and error checking

**Always customize:**
- All parameter values and ranges
- Pulse frequencies, amplitudes, and timing
- Fit functions and analysis methods
- Plot titles and axis labels
- File naming prefixes

## Common Patterns and Examples

### Pattern 1: Simple Parameter Sweep

Most experiments follow this pattern - sweep one parameter and measure the response:

```python
# In QickProgram._initialize():
self.add_loop("sweep_loop", cfg.expt.expts)
pulse = {"freq": cfg.expt.freq, "gain": cfg.expt.gain, ...}
super().make_pulse(pulse, "my_pulse")

# In QickProgram._body():
self.pulse(ch=self.qubit_ch, name="my_pulse", t=0)
super().measure(cfg)

# In QickExperiment.acquire():
self.cfg.expt.gain = QickSweep1D("sweep_loop", start, end)
self.param = {"label": "my_pulse", "param": "gain", "param_type": "pulse"}
super().acquire(MyProgram, progress=progress)
```

### Pattern 2: Time Evolution Experiments

For T1, T2, and similar coherence measurements:

```python
# Program structure:
def _initialize(self, cfg):
    super()._initialize(cfg, readout="standard")
    self.add_loop("time_loop", cfg.expt.expts)
    # Create excitation pulse
    super().make_cfg_pulse(cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge")

def _body(self, cfg):
    # Excite qubit
    self.pulse(ch=self.qubit_ch, name="pi_ge", t=0) 
    # Wait variable time
    self.delay_auto(t=cfg.expt.wait_time, tag="evolution")
    # Optional: apply second pulse for Ramsey/echo
    # self.pulse(ch=self.qubit_ch, name="pi2_ge", t=0)
    # Measure
    super().measure(cfg)

# Experiment structure:
def acquire(self, progress=False):
    self.cfg.expt.wait_time = QickSweep1D(
        "time_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
    )
    self.param = {"label": "evolution", "param": "t", "param_type": "time"}
    super().acquire(MyTimeProgram, progress=progress)
```

### Pattern 3: Spectroscopy Experiments

For finding resonances and measuring lineshapes:

```python
# Program structure - sweep frequency
def _initialize(self, cfg):
    super()._initialize(cfg, readout="standard")
    self.add_loop("freq_loop", cfg.expt.expts)
    # Create probe pulse with swept frequency
    pulse = {
        "freq": cfg.expt.freq,      # This will be swept
        "gain": cfg.expt.probe_gain,
        "length": cfg.expt.probe_length,
        "type": "const"
    }
    super().make_pulse(pulse, "probe_pulse")

def _body(self, cfg):
    # Apply probe pulse  
    self.pulse(ch=self.qubit_ch, name="probe_pulse", t=0)
    super().measure(cfg)

# Experiment structure:
def acquire(self, progress=False):
    self.cfg.expt.freq = QickSweep1D(
        "freq_loop",
        self.cfg.expt.center_freq - self.cfg.expt.span/2,
        self.cfg.expt.center_freq + self.cfg.expt.span/2
    )
    self.param = {"label": "probe_pulse", "param": "freq", "param_type": "pulse"}
    self.param = {"label": "wait", "param": "t", "param_type": "time"}
    self.param = {"label": "readout_pulse", "param": "freq", "param_type": "pulse"}
    self.param = {"label": "qubit_pulse", "param": "freq", "param_type": "pulse"}
    super().acquire(SpectroscopyProgram, progress=progress)

# Analysis - fit to Lorentzian
def analyze(self, data=None, fit=True, **kwargs):
    if data is None:
        data = self.data
    if fit:
        self.fitfunc = fitter.lorenfunc
        self.fitterfunc = fitter.fitloren  
        super().analyze(data, **kwargs)
        # Extract center frequency
        data["resonance_freq"] = data["best_fit"][1]
        data["linewidth"] = data["best_fit"][2]
    return data
```

### Pattern 4: Two-Pulse Sequences

For Ramsey, spin echo, and related experiments:

```python
def _body(self, cfg):
    # First pulse (π/2)
    self.pulse(ch=self.qubit_ch, name="pi2_pulse", t=0)
    
    # Evolution time
    self.delay_auto(t=cfg.expt.tau, tag="evolution")
    
    # Second pulse (π/2 with possible phase)
    self.pulse(ch=self.qubit_ch, name="pi2_pulse_2", t=0)
    
    # Measurement
    super().measure(cfg)
```

### Pattern 5: Active Reset Integration

For experiments requiring high-fidelity initialization:

```python
# In __init__, add active reset to default parameters:
params_def = {
    # ... other parameters ...
    "active_reset": self.cfg.device.readout.active_reset[qi],
}

# The framework handles reset automatically in acquire() if active_reset=True
# Your pulse sequence runs after reset is complete
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. "Parameter not found in config"
**Problem**: `KeyError` when accessing `cfg.device.qubit.something[qi]`

**Solutions**:
- Check that your YAML config file has the required parameters
- Verify the parameter path: `cfg.device.qubit.f_ge` vs `cfg.device.qubit.freq.f_ge`
- Use `print(cfg.device.qubit.keys())` to see available parameters
- Check that qubit index `qi` exists in your config

#### 2. "QickSweep1D loop not found"
**Problem**: `RuntimeError` about missing sweep loop

**Solutions**:
- Ensure `self.add_loop("loop_name", cfg.expt.expts)` is in `_initialize()`
- Loop name in `QickSweep1D("loop_name", ...)` must match exactly
- Check that `cfg.expt.expts` is properly set

#### 3. "Pulse not found" 
**Problem**: `KeyError` when trying to use a pulse

**Solutions**:
- Verify pulse was created in `_initialize()` with `super().make_pulse()`
- Check pulse name spelling: `"pi_ge"` vs `"pi_pulse"`
- Ensure pulse creation happens before `_body()` is called

#### 4. Timing Issues
**Problem**: Pulses not properly synchronized

**Solutions**:
- Use `t=0` for relative timing (pulses happen in sequence)
- Use `self.delay_auto()` for explicit delays
- Check that all pulses use consistent timing scheme
- Verify trigger timing with `self.trig_offset`

#### 5. Poor Fit Quality
**Problem**: `R² < 0.1` or high parameter errors

**Solutions**:
- Check that you're using the right fit function for your data
- Verify sweep range covers the interesting physics
- Increase number of points (`expts`) or averages (`rounds`)
- Check for obvious outliers in your data
- Try different initial guess parameters

#### 6. Data Not Saving
**Problem**: Experiment runs but no data files created

**Solutions**:
- Check that `expt_path` exists and is writable
- Verify `save=True` in your `run()` call
- Check file permissions in the target directory
- Look for error messages about file I/O

#### 7. 2D Experiment Issues
**Problem**: 2D experiment doesn't work as expected

**Solutions**:
- Ensure inner experiment has `go=False` and `check_params=False`
- Verify that parameter names in `y_sweep` match experiment parameters
- Check that both x and y sweep ranges are reasonable
- Make sure inner experiment can run successfully by itself

### Debugging Techniques

#### 1. Print Configuration
```python
# In your experiment __init__ or acquire:
print("Experiment configuration:")
for key, value in self.cfg.expt.items():
    print(f"  {key}: {value}")
```

#### 2. Test Program Creation
```python
# Test your program without running:
prog = MyProgram(soccfg=soccfg, final_delay=50, cfg=cfg)
print("Program created successfully")
print(f"Program pulses: {list(prog.pulses.keys())}")
```

#### 3. Check Sweep Parameters
```python
# In acquire(), after creating sweep:
print(f"Sweep range: {self.cfg.expt.my_param}")
print(f"Number of points: {self.cfg.expt.expts}")
```

#### 4. Validate Data Shape
```python
# In analyze():
print(f"Data shape - xpts: {data['xpts'].shape}, avgi: {data['avgi'].shape}")
print(f"X range: {np.min(data['xpts']):.3f} to {np.max(data['xpts']):.3f}")
```

#### 5. Test Individual Components
```python
# Test just the program:
prog = MyProgram(soccfg, 50, cfg)
# Test just parameter setup:
exp = MyExperiment(cfg_dict, qi=0, go=False)
print(exp.cfg.expt)
```

### Performance Optimization

#### 1. Faster Experiments
```python
# Use style="fast" for quick tests:
exp = MyExperiment(cfg_dict, style="fast")  # Fewer points

# Or adjust parameters directly:
exp = MyExperiment(cfg_dict, params={"expts": 20, "rounds": 1})
```

#### 2. Higher Quality Data
```python
# Use style="fine" for publication-quality data:
exp = MyExperiment(cfg_dict, style="fine")  # More averages

# Or increase averaging:
exp = MyExperiment(cfg_dict, params={"rounds": 10})
```

#### 3. Memory Management
For very long experiments or 2D scans:
- Save data frequently during acquisition
- Use `compact=True` in `acquire()` to store less data
- Clear old variables with `del` when done

## Best Practices Summary

### 1. Experiment Design
- Start with the simplest version that works
- Test individual components before combining
- Use existing experiments as templates
- Follow consistent naming conventions

### 2. Parameter Management
- Always define sensible defaults in `params_def`
- Use device config values as starting points
- Validate parameter ranges and combinations
- Document non-obvious parameter choices

### 3. Code Organization
- Keep pulse sequence logic in `_body()` simple and clear
- Put complex calculations in `analyze()`
- Use descriptive variable names and comments
- Follow the established file structure

### 4. Testing and Validation
- Test with `go=False` before running full experiments
- Validate sweep ranges make physical sense
- Check fit quality and parameter errors
- Compare results with known values when possible

### 5. Data Management
- Use descriptive file prefixes
- Save intermediate results for long experiments  
- Back up important data immediately
- Document experiment conditions and parameters

This guide should give you everything you need to successfully create new QICK experiments. Start with simple examples, understand the patterns, and gradually build up to more complex experiments. The framework is designed to handle the tedious parts so you can focus on the physics!
