# RF Board Usage Guide

This guide explains how to use the on-board RF front end of QICK boards equipped with an RF companion board: ADMV8818 programmable bandpass filters, digital step attenuators, and DC bias DACs. All of the helper code lives in [helpers/rfboard.py](helpers/rfboard.py), with config-file support in [helpers/config.py](helpers/config.py).

## Overview

On systems where several qubits share physical RF channels (a common readout line, a shared qubit-drive DAC), the RF board hardware must be reprogrammed whenever you switch which qubit you are measuring:

- **Bandpass filters (ADMV8818)**: one per generator (DAC) output and per readout (ADC) input. Each is centered on the relevant readout or qubit frequency to suppress out-of-band tones.
- **Step attenuators**: two per generator output (`attn1`, `attn2`) and one per readout input (`attn`), each 0–30 dB, used to set signal levels without changing pulse gains.
- **DC bias DACs**: slow bias outputs used for qubit flux offset (distinct from the fast flux RF DACs — see [README_fast_flux.md](README_fast_flux.md)).

**Unit convention**: the config file stores all frequencies in **MHz**; the low-level `soc.rfb_*` methods expect **GHz**. The helper functions in `helpers/rfboard.py` perform this conversion internally — if you call `soc.rfb_set_gen_filter(...)` yourself, you must divide by 1000.

All helper functions take `soc` (the QICK SoC proxy) and `cfg` (the loaded configuration AttrDict) as explicit arguments; nothing relies on notebook globals.

## Quick start

The single function most users need is `activate_qubit_rf()`. It reads the per-qubit filter, attenuator, and DC-bias settings from the config arrays and programs all of them at once:

```python
from qick import QickConfig
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
from slab_qick_calib.helpers import config, rfboard

cfg_path = 'configs/sample_50_rfboard.yml'
auto_cfg = config.load(cfg_path)

im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'], port=8888)
soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())

# Switch all shared RF channels to qubit 2's settings
qi = 2
auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)
```

With `verbose=True` (the default) it prints a summary of what was programmed:

```
Activated qubit 2: ADC ch 2 (fc=6917, bw=500, bandpass), DAC-ro ch 0 (fc=6917, bw=100, bandpass), DAC-qb ch 1 (fc=4373, bw=1000, bandpass), DC bias ch 2 (val=0.0 V)
```

Passing `cfg_file` makes the function (a) reload the config from disk first, so manual YAML edits are picked up, and (b) persist the resulting active state back to the file. The updated config is returned, so re-assign it as shown above.

You can restrict which channel types are touched:

```python
# Only retune the readout path, leave qubit drive and DC bias alone
auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path,
                                     channels=("adc", "dac_readout"))
```

Valid entries are `"adc"`, `"dac_readout"`, `"dac_qubit"`, and `"dc_bias"` (all four by default). Channel types whose config sections don't exist (e.g. no `dacs.qubit` or no `dacs.flux.dc_ch`) are skipped silently.

## Config file structure

The RF board settings live under `hw.soc` in the config YAML as per-qubit arrays. Index `[qi]` in each array gives that qubit's settings; several qubits may share the same physical channel number with different filter/attenuator values.

```yaml
hw:
  soc:
    adcs:
      readout:
        ch: [2, 1]                      # physical ADC channel per qubit
        attn: [0, 0]                    # readout attenuator (dB, 0-30)
        filter_fc: [6917, 7741]         # filter center frequency (MHz)
        filter_bw: [500, 500]           # filter bandwidth (MHz)
        filter_type: [bandpass, bandpass]  # "bandpass" or "bypass"
    dacs:
      readout:
        ch: [0, 1]                      # physical DAC channel per qubit
        attn1: [0, 0]                   # first TX attenuator (dB)
        attn2: [0, 0]                   # second TX attenuator (dB)
        filter_fc: [6917, 7741]
        filter_bw: [100, 100]
        filter_type: [bandpass, bandpass]
      qubit:                            # optional: qubit drive DAC section
        ch: [1, 0]
        attn1: [0, 0]
        attn2: [0, 0]
        filter_fc: [4373, 4357]
        filter_bw: [1000, 1000]
        filter_type: [bandpass, bandpass]
      flux:                             # optional: flux section
        ch: [9, 10]                     # fast flux RF DAC channel per qubit
        dc_ch: [2, 3]                   # DC bias channel per qubit
        dc_val: [0.0, 0.0]              # DC bias value (V)
        type: [int, int]                # generator type: 'int' or 'full'
    rfboard_active:                     # runtime state, written by the helpers
      adc:
        "2": {filter_fc: 6917, filter_bw: 500, filter_type: bandpass, qubit: 0, attn: 0}
      dac_readout:
        "0": {filter_fc: 6917, filter_bw: 100, filter_type: bandpass, qubit: 0, attn1: 0, attn2: 0}
      dac_qubit:
        "1": {filter_fc: 4373, filter_bw: 1000, filter_type: bandpass, qubit: 0, attn1: 0, attn2: 0}
```

### Initializing the sections

For a config that doesn't yet have the filter fields, two idempotent helpers in `helpers/config.py` create them with safe (bypass) defaults:

```python
from slab_qick_calib.helpers import config

config.init_rfboard_section(cfg_path, num_qubits=4)     # adds filter_fc/filter_bw/filter_type arrays
config.init_rfboard_active_section(cfg_path)            # adds the rfboard_active state section
```

### The `rfboard_active` section

Because channels are shared, the config's per-qubit arrays describe what each qubit *wants*, not what the hardware is currently doing. The `rfboard_active` section records what was last programmed on each *physical channel*, keyed by channel number, including which qubit it was set up for. Query it with:

```python
rfboard.get_active_rf_state(auto_cfg)                       # full dict
rfboard.get_active_rf_state(auto_cfg, "dac_qubit")          # one channel type
rfboard.get_active_rf_state(auto_cfg, "adc", ch=2)          # one channel entry
```

If `get_active_rf_state(...)["qubit"]` is not the qubit you're about to measure, run `activate_qubit_rf()` first.

## Function reference

### High level

| Function | Purpose |
|----------|---------|
| `activate_qubit_rf(qubit, soc, cfg, cfg_file=None, channels=(...), verbose=True)` | Switch all shared channels to one qubit's stored settings. The main entry point. |
| `apply_config_rf_settings(soc, cfg, qubit=None, cfg_file=None)` | Push stored settings for one qubit (or all qubits, in array order) to hardware. With `qubit=None`, later qubits overwrite earlier ones on shared channels — mostly useful at bring-up. |
| `get_active_rf_state(cfg, channel_type=None, ch=None)` | Read back the `rfboard_active` state. |

### Per-chain setup

These set one RF chain explicitly, with optional overrides of the config values. Useful when tuning up filter bandwidths or attenuations interactively before committing them to the config:

```python
# Readout chain: filters on ADC + DAC centered on device.readout.frequency[qi]
info = rfboard.setup_readout_chain(qi, soc, auto_cfg, bw_adc=500, bw_dac=100,
                                   atten1_dac=10, atten2_dac=0, atten_adc=5)

# Qubit drive chain: filter on the qubit DAC centered on device.qubit.f_ge[qi]
info = rfboard.setup_qubit_drive_chain(qi, soc, auto_cfg, bw=1000,
                                       center_freq=4500)   # MHz, overrides f_ge
```

Any parameter left as `None` falls back to the per-qubit config arrays. Pass `cfg_file=` to persist the resulting state to `rfboard_active`.

### Low level

`set_bandpass_rf(soc, gen_ch, ro_ch, fc_MHz, bw_MHz, ...)` programs a bandpass on one generator/readout channel pair with extra safety features:

- Validates `bw_MHz > 0` and that the lower band edge is positive; raises `ValueError` otherwise.
- Warns if the band edges fall outside the ADMV8818's usable span (~2–18 GHz).
- Sleeps `settle_ms` (default 30 ms) after programming, then calls `soc.clear_interrupts(error_on_interrupt=False)` to clear sticky flags raised while the filter settles.
- Returns the `(f_lo, f_hi)` band edges in MHz.

`bypass_all_filters(soc, gen_channels, ro_channels, cfg=None, cfg_file=None)` puts every listed channel into bypass — a good known state for debugging signal-chain problems.

### Persisting tuned values

After settling on good filter/attenuator values interactively, write them into the per-qubit arrays so `activate_qubit_rf()` uses them from then on:

```python
config.update_rfboard(cfg_path, "dacs.readout", "attn1", 10, index=qi)
config.update_rfboard(cfg_path, "adcs.readout", "filter_bw", 500, index=qi)
config.update_rfboard(cfg_path, "dacs.qubit", "filter_type", "bandpass", index=qi)
```

## DC flux bias

Two equivalent ways to set the slow flux bias:

```python
# 1. Helper: reads dc_ch from config; optionally updates dc_val in config
rfboard.set_dc_bias(qi, soc, auto_cfg, dc_val=0.15, cfg_file=cfg_path)

# 2. Direct: useful inside sweep loops where you don't want config writes
dc_ch = auto_cfg['hw']['soc']['dacs']['flux']['dc_ch'][qi]
soc.rfb_set_bias(dc_ch, 0.15)
```

The direct form is what flux-sweep experiments use internally — for example `QubitSpecFlux` in [experiments/flux/pulse_probe_spectroscopy_flux.py](experiments/flux/pulse_probe_spectroscopy_flux.py) and the DC sweep mode of [experiments/single_qubit/resonator_spectroscopy.py](experiments/single_qubit/resonator_spectroscopy.py) call `soc.rfb_set_bias()` once per sweep point. Remember to return the bias to its resting value when a sweep finishes.

## How the rest of the package uses it

- **Automated tuning** ([calib/qubit_tuning.py](calib/qubit_tuning.py)): `tune_up_qubit()` calls `activate_qubit_rf()` before characterizing each qubit, so multi-qubit tune-up loops switch the shared chains automatically.
- **Long-term tracking** ([calib/time_tracking.py](calib/time_tracking.py)): the tracking loop activates each qubit's RF settings before its measurement block when cycling through qubits.
- **Scripts** ([scripts/](scripts/)): every standalone runner (e.g. `scripts/run_t1_fastflux_loop.py`, `run_t1_predist_fastflux.py`) activates the target qubit right after connecting, before constructing experiments.

The standard preamble for any script or notebook is therefore:

```python
auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)
soc.rfb_set_bias(auto_cfg['hw']['soc']['dacs']['flux']['dc_ch'][qi], 0.0)  # if flux bias needed
```

## Troubleshooting

- **No signal after switching qubits**: check `get_active_rf_state()` — a filter programmed for another qubit's frequency will block your tones entirely. Re-run `activate_qubit_rf()`.
- **ADMV8818 range warnings**: the filter only works between roughly 2 and 18 GHz. Below ~2 GHz, use `filter_type: bypass` for that channel instead of a bandpass.
- **Glitches right after programming filters**: the readout chain needs time to settle. `set_bandpass_rf()` already waits 30 ms and clears interrupts; if you program filters via raw `soc.rfb_*` calls, do the same (`time.sleep(0.03)` then `soc.clear_interrupts(error_on_interrupt=False)`).
- **Config and hardware out of sync**: the YAML arrays are *requests*; `rfboard_active` is the *last programmed state*. After editing the YAML by hand, run `activate_qubit_rf(..., cfg_file=cfg_path)` (it reloads from disk first) to push your edits to hardware.
- **No `dacs.qubit` or `dacs.flux` section**: these are optional. `activate_qubit_rf()` and `apply_config_rf_settings()` skip channel types whose sections are missing; `set_dc_bias()` raises `ValueError` if there is no flux section.
