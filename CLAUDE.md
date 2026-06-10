# CLAUDE.md

User-facing guides: [README_rfboard.md](README_rfboard.md) (RF board filters/attenuators/DC bias)
and [README_fast_flux.md](README_fast_flux.md) (fast flux sweeps and predistortion).

## Running experiments from scripts

### Connecting to hardware
The RFSoC is accessed via Pyro4 proxies through a nameserver. The nameserver must
be running on this PC first (run `start_nameserver.bat` or `python nameserver.py`).

```python
from qick import QickConfig
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
from slab_qick_calib.helpers import config, rfboard

cfg_path = 'configs/sample_50_rfboard.yml'   # relative to repo root
auto_cfg = config.load(cfg_path)

# Port 8888 is required — the nameserver listens there, not the default 9090
im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'], port=8888)
soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())

cfg_dict = {'soc': soccfg, 'expt_path': r'C:\_Data\sample_50_new',
            'cfg_file': cfg_path, 'im': im}
```

### Activating a qubit
Before running experiments, activate the qubit's RF settings (attenuators, filters, DC bias):
```python
qi = 2
auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)
# Set DC flux bias if needed
soc.rfb_set_bias(auto_cfg['hw']['soc']['dacs']['flux']['dc_ch'][qi], 0.0)
```

### Running an experiment
All experiment classes take `cfg_dict` and follow the same pattern:
```python
import slab_qick_calib.experiments as meas

# go=True runs acquire+analyze+display immediately
expt = meas.QubitSpec(cfg_dict, qi=2, style="medium", params={...})

# Or go=False for manual control:
expt = meas.QubitSpec(cfg_dict, qi=2, style="medium", go=False, params={...})
data = expt.acquire(progress=True)
data = expt.analyze(data=data)
expt.display()
expt.save_data(data=data)
```

### Non-interactive scripts (no Jupyter)
- Use `matplotlib.use('Agg')` before any imports to prevent `plt.show()` from blocking
- Use `go=False` and call `acquire`/`analyze` manually, skip `display()`
- Save plots with `fig.savefig()` instead of showing them
- The `display()` method on experiments calls `plt.show()` internally and will hang

### Key config values (sample_50_rfboard.yml)
- `aliases.ip`: nameserver IP (10.108.30.23)
- `aliases.soc`: Pyro name of the board (bf1_soc)
- `hw.soc.dacs.flux.ch[qi]`: fast flux DAC channel per qubit
- `hw.soc.dacs.flux.dc_ch[qi]`: DC bias channel per qubit
- `hw.soc.dacs.flux.type[qi]`: generator type ('int' = interpolated, 'full' = full-speed)
- `device.qubit.f_ge[qi]`, `device.qubit.sweet_spot_ac[qi]`: qubit params

### Generator info at runtime
```python
gen_cfg = soccfg['gens'][ch]
gen_cfg['type']           # e.g. 'axis_sg_int4_v2'
gen_cfg['f_fabric']       # envelope sample rate in MHz (430.08 for int gens)
gen_cfg['samps_per_clk']  # 1 for int, 16 for full-speed
gen_cfg['maxlen']          # max envelope samples (8192 or 16384)
soccfg.get_maxv(ch)       # max int16 value with safety margin
```

## Experiment code style
- All experiment parameter defaults must be declared in `params_def` in the Experiment `__init__`, not buried as `.get()` fallbacks elsewhere.
- Program classes and `acquire`/`analyze`/`display` methods should use direct attribute access on `cfg.expt` (e.g. `cfg.expt.length`), never `cfg.expt.get("key", default)`. By the time the Program runs, every key it needs must already exist.
- `.get()` on `params` dict (before merge into `cfg.expt`) is fine for pre-init logic. `.get()` on `data` dicts (checking optional analysis results) is also fine.

## Notebooks
- All notebooks should have autoreload at the top:
```python
%load_ext autoreload
%autoreload 2
```

## Plotting
- Do not add titles to plots. Existing titles in user code should be left as-is.

## Predistorted flux pulses

Flux lines have exponential distortions (reflections, bias-tee droop) characterized by
a step response: `s(t) = α₀ + Σ αᵢ·exp(-t/τᵢ)`. The compensation pipeline is:

### 1. Characterize the step response
Run `QubitSpecFluxSweep` sweeping `flux_lead_time` to measure qubit frequency vs settling time.
Fit with `fit_step_response_flux()` from `analysis/fitting.py` to extract `alpha0`, `alphas`, `taus`.
See `notebooks_analysis/analyze_flux_step_response.ipynb` for a worked example.

### 2. Generate predistorted envelopes (`analysis/flux_filter.py`)
Three functions produce int16 `idata` arrays ready for `prog.add_envelope()`:

- `make_full_predistorted_step(alphas, taus, alpha0, duration_us, fs_mhz)` — all corrections in one envelope
- `make_slow_ramp(alphas, taus, alpha0, duration_us, fs_mhz, tau_threshold_us=0.5)` — only τ > threshold (smooth ramp, good first test)
- `make_fast_correction(alphas, taus, alpha0, fs_mhz, duration_ns=200)` — only τ ≤ threshold (short transient)

Each returns a dict with `idata`, `gain_scale`, `waveform`, `time_us`, `n_samples`.
The `gain_scale` factor is the peak of the predistorted waveform — multiply your nominal flux gain by it to get the QICK pulse gain.

### 3. Use in a QICK program
The int generator (`axis_sg_int4_v2`, `samps_per_clk=1`) plays arb envelopes at `f_fabric` rate
with 4× hardware interpolation, giving ~19 µs max at 8192 samples or ~38 µs at 16384.

```python
fs_mhz = self.soccfg['gens'][flux_ch]['f_fabric']
maxv = self.soccfg.get_maxv(flux_ch)

env = make_slow_ramp(alphas, taus, alpha0, duration_us=total_us, fs_mhz=fs_mhz, maxv=maxv)
self.add_envelope(flux_ch, name="flux_predist", idata=env["idata"])
self.add_pulse(ch=flux_ch, name="flux_pulse", style="arb", envelope="flux_predist",
               freq=0, phase=0, gain=nominal_gain * env["gain_scale"],
               outsel="product")
```

### 4. Ready-made experiment
`experiments/flux/qubit_spec_predistorted.py` provides `QubitSpecPredist`, a drop-in
replacement for `QubitSpec` with predistorted flux. Required params:

```python
QubitSpecPredist(cfg_dict, qi=2, style="medium", params={
    "flux_alphas": [0.0484, -0.131, -1.999],
    "flux_taus": [0.018, 9.077, 12534.6],   # µs
    "flux_alpha0": 1.0969,
    "flux_gain": -0.35,
    "flux_slow_only": True,    # True = only τ > 0.5 µs terms
})
```

### Hardware constraints (ZCU216)
- Int gen ch 9 (qubit 2 flux): `f_fabric=430.08 MHz`, `maxlen=8192` (19 µs max envelope)
- Envelope length must fit in memory: `duration_us * f_fabric < maxlen`
- For longer pulses, use `mode="periodic"` on a const segment or chain multiple pulses
- `maxv_scale=0.9` (prevents interpolation overshoot) — already handled by `soccfg.get_maxv()`
