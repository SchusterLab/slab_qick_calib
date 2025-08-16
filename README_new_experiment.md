# How to Write a New Experiment Program

This guide explains how to create a new experiment program using the `slab_qick_calib` framework. It covers the structure of an experiment, how to define pulses, configure the readout, and manage timing.

## Core Concepts

The framework is built on two main classes:

1.  `QickProgram`: This is a low-level class that inherits from `qick.AveragerProgramV2`. It is responsible for defining the pulse sequences that run directly on the QICK hardware. You will typically inherit from this class to create the specific pulse sequence for your experiment.

2.  `QickExperiment`: This is a higher-level class that wraps the `QickProgram` and handles the experiment logic, data acquisition, analysis, and plotting. You will inherit from this class to define the parameters and logic for your specific experiment.

There are two main approaches to structuring your experiment:

1.  **Inherit from `QickExperiment` and a custom `QickProgram`**: This is the most common approach. You create a `QickProgram` subclass to define the pulse sequence and a `QickExperiment` subclass to manage the experiment.
2.  **Inherit directly from `QickProgram`**: For very simple experiments or for quick tests, you might choose to inherit directly from `QickProgram` and not use the `QickExperiment` wrapper. This gives you more direct control but requires you to handle data acquisition and analysis manually.

## Writing a New Experiment: A Step-by-Step Guide

Let's walk through the process of creating a new experiment, using a Rabi experiment as an example.

### 1. Create a `QickProgram` Subclass

First, create a class that inherits from `QickProgram`. This class will define the pulse sequence.

```python
from ..general.qick_program import QickProgram
from ...exp_handling.datamanagement import AttrDict

class MyRabiProgram(QickProgram):
    def __init__(self, soccfg, final_delay, cfg):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        q = cfg.expt.qubit[0]

        # Initialize standard readout
        super()._initialize(cfg, readout="standard")

        # Add a sweep loop
        self.add_loop("sweep_loop", cfg.expt.expts)

        # Define the qubit pulse
        pulse = {
            "sigma": cfg.expt.sigma,
            "freq": cfg.expt.freq,
            "gain": cfg.expt.gain,
            "phase": 0,
            "type": "gauss",
        }
        super().make_pulse(pulse, "qubit_pulse")

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)

        # Apply the qubit pulse
        self.pulse(ch=self.qubit_ch, name="qubit_pulse", t=0)

        # Perform measurement
        super().measure(cfg)
```

**Key Methods:**

*   `__init__(self, soccfg, final_delay, cfg)`: The constructor. It's important to call the parent `__init__` method.
*   `_initialize(self, cfg)`: This is where you set up your experiment.
    *   **Readout:** `super()._initialize(cfg, readout="standard")` sets up the standard readout. If you need a custom readout, you can pass other options or implement your own logic here. For example, you might want to use a longer readout pulse for better SNR, which you can configure by setting `readout="long"`.
    *   **Pulses:** Use `self.make_pulse(...)` or `self.make_pi_pulse(...)` to define the pulses you'll need. These methods take a dictionary of pulse parameters from your config.
    *   **Sweeps:** If you are sweeping a parameter, you can use `self.add_loop(...)` to create a sweep loop.
*   `_body(self, cfg)`: This method defines the core logic of your pulse sequence. You will use methods like `self.pulse(...)`, `self.delay_auto(...)`, and `self.measure(...)` to construct your sequence.

### 2. Create a `QickExperiment` Subclass

Next, create a class that inherits from `QickExperiment`. This class will manage the experiment.

```python
from ..general.qick_experiment import QickExperiment

class MyRabiExperiment(QickExperiment):
    def __init__(self, cfg_dict, qi=0, go=True, params={}, ...):
        # ... (see rabi.py for full example)
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, ...)
        # Define default parameters for your experiment
        params_def = {
            "expts": 60,
            "reps": self.reps,
            "rounds": self.rounds,
            "sweep": "amp",
            # ... other parameters
        }
        self.cfg.expt = {**params_def, **params}
        if go:
            super().qubit_run(...)

    def acquire(self, progress=False, debug=False):
        # Configure the sweep
        if self.cfg.expt.sweep == "amp":
            self.cfg.expt["gain"] = QickSweep1D(...)
            self.param = {"label": "qubit_pulse", "param": "gain", "param_type": "pulse"}
        # ...
        super().acquire(MyRabiProgram, progress=progress)
        return self.data

    def analyze(self, data=None, fit=True, **kwargs):
        # ... (fitting logic)
        return data

    def display(self, data=None, fit=True, **kwargs):
        # ... (plotting logic)
```

**Key Methods:**

*   `__init__(...)`: Sets up the experiment parameters. You define default parameters in a `params_def` dictionary and merge them with any user-provided `params`.
*   `acquire(...)`: This method is responsible for running the experiment. It configures the sweep using `QickSweep1D` and then calls `super().acquire(MyRabiProgram, ...)` to run the program.
*   `analyze(...)`: This is where you fit your data and extract relevant parameters.
*   `display(...)`: This method handles plotting the results.

## Managing Timing

Precise timing is critical in quantum experiments. Here’s how to manage it in your `QickProgram`:

*   **`self.pulse(ch=..., name=..., t=...)`**: The `t` parameter of the `pulse` method specifies the time at which the pulse should be triggered. This time is relative to the start of the experiment sequence.
*   **`self.delay_auto(t=..., tag=...)`**: This method introduces a delay of a specified time `t`. The time is automatically converted to the correct units for the hardware. You can add a `tag` to the delay for easier debugging.
*   **Registers for Timing**: For more complex timing, you can use registers to store time values and update them during the experiment. This is useful for experiments where the delay is swept, like a T1 measurement.

In the `T1Program` example, the wait time is swept using a loop:

```python
class T1Program(QickProgram):
    def _initialize(self, cfg):
        # ...
        self.add_loop("wait_loop", cfg.expt.expts)
        # ...

    def _body(self, cfg):
        # ...
        self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
        self.delay_auto(t=cfg.expt["wait_time"] + 0.01, tag="wait")
        self.measure(cfg)
```

In the `T1Experiment` class, the `acquire` method sets up the sweep:

```python
class T1Experiment(QickExperiment):
    def acquire(self, progress=False, debug=False):
        self.param = {"label": "wait", "param": "t", "param_type": "time"}
        self.cfg.expt.wait_time = QickSweep1D(
            "wait_loop", self.cfg.expt.start, self.cfg.expt.start + self.cfg.expt.span
        )
        super().acquire(T1Program, progress=progress)
        return self.data
```

## Default vs. Custom Behavior

The framework provides default implementations for many common tasks, but you can customize them as needed.

*   **`measure(self, cfg)` in `QickProgram`**: This method provides a standard measurement sequence. You can override it in your `QickProgram` subclass if you need a different measurement protocol.
*   **`analyze(...)` and `display(...)` in `QickExperiment`**: These methods provide generic analysis and plotting functionality. For most experiments, you will want to override these to implement fitting and plotting specific to your experiment.
*   **Readout Configuration**: The `_initialize` method in `QickProgram` sets up a standard readout. You can customize this by either passing `readout="long"` for a longer readout pulse or by implementing your own readout configuration logic in your `_initialize` method.

By understanding these core concepts and following the examples, you can create new and complex experiments with the `slab_qick_calib` framework.

## Specifying Readout in `_initialize`

In the `_initialize` method of your `QickProgram` subclass, you have fine-grained control over the readout configuration. The `resonator_spectroscopy.py` experiment provides a good example of this.

In `ResSpecProgram._initialize`, the readout type is chosen based on the pulse length:

```python
class ResSpecProgram(QickProgram):
    def _initialize(self, cfg):
        # ...
        # Choose readout type based on pulse length
        readout = 'long' if cfg.expt.long_pulse else 'custom'
            
        # Initialize with appropriate readout type
        super()._initialize(cfg, readout=readout)
        # ...
```

If `long_pulse` is `True`, the `readout` parameter is set to `'long'`, which configures a longer readout pulse for better signal-to-noise ratio. If not, it's set to `'custom'`, and the readout parameters are taken directly from the `cfg.expt` dictionary. This allows you to define custom readout lengths and gains for specific experiments.

## Experiment Parameters: `params` vs. YAML Config

When you create an instance of a `QickExperiment`, you can provide a `params` dictionary to override the default parameters for that specific run. This is useful for changing parameters "in situ" without modifying the main configuration files.

The experiment parameters are determined in the following order of precedence:

1.  **`params` dictionary**: Parameters passed directly to the experiment's `__init__` method have the highest priority.
2.  **`params_def` dictionary**: Default parameters defined within the experiment's `__init__` method.
3.  **YAML Configuration File**: The base parameters are loaded from the YAML configuration file specified in your `cfg_dict`.

This hierarchy allows you to have a stable base configuration in your YAML file while still having the flexibility to change parameters for individual experiment runs. For example, you could run a Rabi experiment with a different `max_gain` without changing the `pi_ge` gain in your main config file.

## Analyzing and Displaying Results

The `analyze` and `display` methods in your `QickExperiment` subclass are where you process and visualize your data. Let's look at the `T1Experiment` from `t1.py` as an example.

### The `analyze` Method

The `analyze` method is responsible for fitting the data and extracting key parameters.

```python
class T1Experiment(QickExperiment):
    def analyze(self, data=None, **kwargs):
        if data is None:
            data = self.data

        # Fit to exponential decay function
        self.fitfunc = fitter.expfunc
        self.fitterfunc = fitter.fitexp
        super().analyze(self.fitfunc, self.fitterfunc, data, **kwargs)

        # Extract T1 time from fit parameters
        data["new_t1"] = data["best_fit"][2]
        data["new_t1_i"] = data["fit_avgi"][2]
        return data
```

In this example, the `analyze` method:
1.  Sets the `fitfunc` (the model to fit to) and `fitterfunc` (the fitting algorithm) from the `analysis.fitting` module.
2.  Calls `super().analyze(...)` to perform the fit.
3.  Extracts the T1 time from the fit results and adds it to the `data` dictionary.

### The `display` Method

The `display` method is responsible for plotting the data and the fit results.

```python
class T1Experiment(QickExperiment):
    def display(self, data=None, fit=True, ...):
        # ...
        title = f"$T_1$ Q{qubit}"
        xlabel = "Wait Time ($\\mu$s)"

        caption_params = [
            {"index": 2, "format": "$T_1$ fit: {val:.3} $\\pm$ {err:.2} $\\mu$s"},
        ]

        super().display(
            data=data,
            # ...
            title=title,
            xlabel=xlabel,
            fit=fit,
            fitfunc=self.fitfunc,
            caption_params=caption_params,
            # ...
        )
```

In this example, the `display` method:
1.  Sets the plot `title` and `xlabel`.
2.  Defines `caption_params`, which specifies how to format the fit results for the plot legend.
3.  Calls `super().display(...)` to create the plot. The parent `display` method handles the actual plotting logic, using the parameters you provide.

## Creating a 2D Experiment with `QickExperiment2DSimple`

For 2D experiments, where you sweep two parameters, you can use the `QickExperiment2DSimple` class. This class is designed for nested experiments, where an "outer" experiment sweeps one parameter (the y-axis) and an "inner" experiment sweeps another parameter (the x-axis).

The `RabiChevronExperiment` in `rabi.py` is a good example. It sweeps the qubit frequency on the y-axis and the pulse amplitude or length on the x-axis.

### Structure of a `QickExperiment2DSimple`

Here's a simplified look at the structure:

```python
from ..general.qick_experiment import QickExperiment2DSimple

class RabiChevronExperiment(QickExperiment2DSimple):
    def __init__(self, cfg_dict, qi=0, go=True, params={}, ...):
        # ...
        super().__init__(cfg_dict=cfg_dict, prefix=prefix, ...)

        # Create an instance of the inner experiment (but don't run it)
        self.expt = RabiExperiment(
            cfg_dict, qi=qi, go=False, params=params, check_params=False...
        )
        # ...
        if go:
            super().run(...)

    def acquire(self, progress=False, debug=False):
        # Create the y-axis sweep points (frequency)
        freqpts = np.linspace(...)
        ysweep = [{"pts": freqpts, "var": "freq"}]

        # Call the parent acquire method
        super().acquire(ysweep, progress=progress)
        return self.data
```

**Key Points:**

*   **Inheritance**: The outer experiment class inherits from `QickExperiment2DSimple`.
*   **`__init__`**:
    *   It creates an instance of the inner experiment (in this case, `RabiExperiment`) with `go=False` to prevent it from running immediately.
    *   This inner experiment instance is stored as `self.expt`.
*   **`acquire`**:
    *   It defines the y-axis sweep points (`ysweep`).
    *   It then calls `super().acquire(ysweep, ...)` which handles the looping. For each point in the `ysweep`, it updates the `self.expt.cfg.expt` dictionary with the new parameter value (e.g., `freq`) and then calls `self.expt.acquire()`. This is how the parameters of the inner experiment are modified at each step of the outer (y-axis) sweep.
