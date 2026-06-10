# Fast Flux Sweep Scans

This guide covers experiments that pulse a qubit's flux line with the QICK RF DACs ("fast flux") and sweep measurements across flux: spectroscopy vs flux, the T1-vs-flux family, and the predistortion pipeline that compensates flux-line distortions. The shared sweep machinery lives in [experiments/general/qick_flux_experiment.py](experiments/general/qick_flux_experiment.py); the experiments themselves are in [experiments/flux/](experiments/flux/) (plus `T1FastFlux` in [experiments/single_qubit/t1.py](experiments/single_qubit/t1.py)).

For the slow DC bias DACs (static flux offsets), see [README_rfboard.md](README_rfboard.md).

## 1. Hardware and config

Fast flux pulses are played on ordinary QICK generator channels, configured per qubit under `hw.soc.dacs.flux`:

```yaml
hw:
  soc:
    dacs:
      flux:
        ch: [9, 10]        # fast flux RF DAC channel per qubit
        dc_ch: [2, 3]      # slow DC bias channel per qubit (rfboard)
        dc_val: [0.0, 0.0] # resting DC bias (V)
        type: [int, int]   # generator type: 'int' or 'full'
```

Two generator types matter for envelope-based (arb) pulses:

| | `int` (`axis_sg_int4_v2`) | `full` (full-speed) |
|---|---|---|
| `samps_per_clk` | 1 | 16 |
| envelope sample rate | `f_fabric` (430.08 MHz on ZCU216), 4× interpolated in hardware | full DAC rate |
| `maxlen` | 8192 or 16384 samples | larger |
| max arb envelope | ~19 µs (8192) / ~38 µs (16384) | shorter in time |

Query at runtime:

```python
gen_cfg = soccfg['gens'][flux_ch]
gen_cfg['f_fabric']        # envelope sample rate (MHz)
gen_cfg['maxlen']          # max envelope samples
soccfg.get_maxv(flux_ch)   # max safe int16 amplitude (includes 0.9 interpolation margin)
```

An arb envelope must satisfy `duration_us * f_fabric < maxlen`. For longer constant flux, use a `const` pulse (no envelope memory needed) or `mode="periodic"`.

## 2. Flux models and gain↔frequency conversion

Sweeping flux gain moves the qubit frequency along its dispersion curve. The package keeps a per-qubit flux model in the config so sweeps can be specified in frequency rather than raw gain. The model classes live in [analysis/fitting.py](analysis/fitting.py):

- `TransmonFluxConverter` — transmon SQUID model `f(g) = (f_max + E_C)·(cos²φ + d²sin²φ)^¼ − E_C`, built from config keys `hw.soc.dacs.flux.transmon_{f_max, E_C, g_period, g_offset, d}[qi]`. Fit with `fit_transmon_flux(gain, freq, E_C)` (E_C held fixed).
- `QuadraticFluxConverter` — simple `f(g) = a·g² + b·g + c` from config keys `quad_a/quad_b/quad_c[qi]`. Adequate near the sweet spot.
- `flux_converter_from_config(cfg_flux, cfg_qubit, qi, direction)` — factory that prefers the transmon model and falls back to quadratic; returns `None` if neither is configured.

Both converters expose `gain_to_freq(g)`, `freq_to_gain(f)`, `sweet_spot`, and `f_max`. The `direction` argument (`'pos'`/`'neg'`) picks which side of the sweet spot the inverse lookup uses.

`QubitSpecFastFlux` measures freq(gain) and (with `update=True`) writes the fitted model parameters back into the config, so later sweeps can use frequency-based ranges.

### Common sweep parameters

Every flux-gain sweep class derives from `QickFluxSweep` and accepts:

| Param | Meaning |
|---|---|
| `gain_start` | first flux gain (default: `device.qubit.sweet_spot_ac[qi]`) |
| `gain_stop` | last flux gain (default: sweet spot ± 0.4 depending on `direction`) |
| `expts_gain` | number of gain points (default 50) |
| `direction` | `'pos'` or `'neg'` — which way to sweep from the sweet spot |
| `freq_span` | sweep depth in MHz; with a flux model this overrides `gain_stop` |
| `lin_freq` | if True and a model exists, space points linearly in *frequency* instead of gain |
| `flux_converter` | pass a converter explicitly instead of loading from config |

`lin_freq=True` is usually what you want for T1-vs-frequency maps: equal frequency steps give uniform coverage of the dispersion even though the gain axis is nonlinear.

## 3. Spectroscopy vs flux

All in [experiments/flux/pulse_probe_spectroscopy_flux.py](experiments/flux/pulse_probe_spectroscopy_flux.py) unless noted.

- **`QubitSpecFlux`** — 2D spectroscopy: probe frequency × *DC bias*. Sets the slow bias per row via `soc.rfb_set_bias()`, optionally recentering the readout per row (`recenter_res`, `tune_resonator`, or a `bias_table_V`/`f0_table_MHz` interpolation table). This is the DC counterpart, useful for mapping the full flux period.
- **`QubitSpecFastFlux`** — qubit spectroscopy at each *fast flux gain* point. Tracks the qubit by re-centering each scan on the previous fit; fits quadratic (and optionally transmon, `fit_model='transmon'`) models to freq(gain) and can write them to the config (`update=True`). Supports co-swept drive power/length via `spec_gain_pts` / `spec_length_pts` for deep sweeps where the optimal drive changes.
- **`QubitSpecFluxSweep`** — generalization that sweeps any inner-experiment parameter (`sweep_var`, e.g. `"flux_gain"`, `"flux_lead_time"`, `"gain"`) with explicit `sweep_pts` or `sweep_start/sweep_stop/expts_sweep`. With `predistorted=True` the inner experiment is `QubitSpecPredist`. Sweeping `flux_lead_time` is how the flux **step response** is characterized (section 6).
- **`ResSpecFastFlux`** ([resonator_spectroscopy_flux.py](experiments/flux/resonator_spectroscopy_flux.py)) — resonator spectroscopy at each gain point; maps the resonator frequency / dispersive shift vs flux.

```python
import slab_qick_calib.experiments as meas

qspec = meas.QubitSpecFastFlux(cfg_dict, qi=2, params={
    "gain_start": 0.0,
    "gain_stop": -0.3,
    "expts_gain": 60,
    "direction": "neg",
    "update": True,        # store fitted flux model in config
})
```

## 4. T1 vs flux

Three strategies, trading speed against robustness:

### Full decay curves: `T1FastFlux` ([experiments/single_qubit/t1.py](experiments/single_qubit/t1.py))
Runs a complete `T1Experiment` (full wait-time sweep + exponential fit) at each gain point. Slowest but most trustworthy; adapts the wait-time span between points based on the previous fit (`start_t1` sets the initial span). `T1FastFluxRepeated` reruns the sweep to build a T1(frequency, time) heatmap.

### Adaptive single-point: `T1FastFluxLoop` ([experiments/flux/t1_fastflux_loop.py](experiments/flux/t1_fastflux_loop.py))
Measures the excited-state population at a *single* wait time per gain point and inverts the exponential:

```
T1 = -wait_time / ln(population)
```

After each point the wait time is reset to `new_T1 / 2` (the sensitivity optimum), so the scan tracks T1 as it varies across the dispersion — typically an order of magnitude faster than full decay curves. `t1_max` bounds the tracker against bad fits. Requires good g/e voltage calibration to convert readout voltage to population; `T1FastFluxLoopRepeated` interleaves calibration scans (every 10–50 sweeps, or single-shot histograms with `calibration="single_shot"`), live-plots the accumulating heatmap, and saves an incremental CSV.

### Inline calibration: `T1ContFluxExperiment` ([experiments/flux/t1_cont_flux.py](experiments/flux/t1_cont_flux.py))
Builds calibration into the pulse program itself. Per repetition the program plays:

1. `n_e` short-wait shots (π + `calib_wait_short` flux + readout) → excited-state voltage `ve`
2. `n_g` long-wait shots (π + `calib_wait_long` flux + readout) → ground-state voltage `vg`
3. `n_t1` shots at the adaptive wait time → T1 data

Every gain point therefore carries its own `ve`/`vg`, immune to slow drifts and flux-dependent readout shifts. Defaults: `n_e=3`, `n_g=2`, `n_t1=8`, `calib_wait_short=0.05` µs, `calib_wait_long=20` µs. `T1ContFluxRepeated` adds the heatmap/repeat machinery with no separate calibration passes needed.

```python
expt = meas.T1FastFluxLoop(cfg_dict, qi=2, params={
    "gain_start": 0.16,
    "gain_stop": -0.28,
    "expts_gain": 300,
    "lin_freq": True,
    "t1_max": 10,          # µs, cap on the adaptive tracker
    "direction": "neg",
})
```

(adapted from [scripts/run_t1_fastflux_loop.py](scripts/run_t1_fastflux_loop.py); see also `run_t1_fastflux_loop_repeated.py` and `run_t1_cont_flux.py` in the same directory).

## 5. Other concurrent-flux experiments

- **`RabiFluxExperiment` / `RabiFluxChevronExperiment`** ([rabi_flux.py](experiments/flux/rabi_flux.py)) — Rabi oscillations with a constant flux pulse held on for the entire sequence (settle → drive → readout). All `delay_auto` calls are replaced by explicit `delay` so the program never waits for the flux pulse to end.
- **`HistogramFluxExperiment`** ([single_shot_flux.py](experiments/flux/single_shot_flux.py)) — single-shot g/e histograms with the flux pulse on through settle, π pulse, and readout. Use this to get the `ve`/`vg` calibration at the actual operating flux.

Both take `flux_gain`, `flux_chan`, and a `flux_settle` wait between the flux edge and the first qubit pulse.

## 6. Predistorted flux pulses

A nominally square flux pulse arrives at the qubit distorted by reflections and bias-tee droop. The distortion is modeled as a step response

```
s(t) = α₀ + Σᵢ αᵢ·exp(-t/τᵢ)
```

and compensated by playing a pre-distorted envelope. The full pipeline:

### 6.1 Characterize the step response

Sweep the delay between the flux edge and a spectroscopy probe (`QubitSpecFluxSweep` with `sweep_var="flux_lead_time"`) to get qubit frequency vs settling time. Then fit in *normalized flux space*, which linearizes the nonlinear frequency↔flux map via the flux converter:

```python
from slab_qick_calib.analysis import fitting as fitter

params, s_data, s_fit = fitter.fit_step_response_flux(
    tdata, fdata, flux_converter, flux_gain_target,
    n_exp=3,        # number of exponential terms
    s0=1.0,         # constrain s(0); or fix alpha0 (0.0 = pure bias-tee high-pass)
)
# params: {"alpha0": ..., "alphas": [...], "taus": [...], ...}
```

(`fit_step_response` is the simpler frequency-space variant; both follow arXiv:2503.04610.) The fitted values can be stored per qubit in the config as `hw.soc.dacs.flux.step_alphas / step_taus / step_alpha0`, where `QubitSpecFluxSweep(predistorted=True)` and `T1Predist` will pick them up automatically.

### 6.2 Generate predistorted envelopes ([analysis/flux_filter.py](analysis/flux_filter.py))

The inverse filter is computed analytically (`step_response_to_tf` → bilinear transform in `compute_inverse_iir`) and applied to a step. Three ready-made generators return int16 `idata` arrays for `prog.add_envelope()`:

| Function | Contents | Use |
|---|---|---|
| `make_full_predistorted_step(alphas, taus, alpha0, duration_us, fs_mhz)` | all corrections | simplest: one envelope does everything |
| `make_slow_ramp(..., tau_threshold_us=0.5)` | only τ > threshold | smooth ramp; good first test |
| `make_fast_correction(..., duration_ns=200)` | only τ ≤ threshold | short transient overlapped on a const pulse |

Each returns `{"idata", "waveform", "gain_scale", "time_us", "n_samples"}`. **`gain_scale` matters**: the predistorted waveform overshoots 1, so the envelope is normalized to fit in int16 — multiply your nominal flux gain by `gain_scale` when setting the QICK pulse gain. `validate_filter()` round-trips a step through inverse and forward filters to check residuals.

### 6.3 Ready-made experiments

- **`QubitSpecPredist`** ([qubit_spec_predistorted.py](experiments/flux/qubit_spec_predistorted.py)) — drop-in `QubitSpec` replacement with a predistorted flux envelope (flux lead/trail times around the probe pulse):

  ```python
  meas.QubitSpecPredist(cfg_dict, qi=2, style="medium", params={
      "flux_alphas": [0.0484, -0.131, -1.999],
      "flux_taus": [0.018, 9.077, 12534.6],   # µs
      "flux_alpha0": 1.0969,
      "flux_gain": -0.35,
      "flux_slow_only": True,    # only τ > flux_tau_threshold terms
  })
  ```

- **`T1Predist`** ([t1_predistorted.py](experiments/flux/t1_predistorted.py)) — T1 with predistorted flux during the wait. Because an arb envelope's length is fixed at program load, the wait-time sweep runs as a Python loop, rebuilding the program with a correctly-sized envelope per point. Supports `flux_negative_reset=True` to play an inverted pulse after readout, discharging the bias tee between shots.
- **`T1PredistFastFlux` / `T1PredistFastFluxRepeated`** ([t1_predist_fastflux.py](experiments/flux/t1_predist_fastflux.py)) — `T1FastFlux` with `T1Predist` as the inner experiment; same sweep parameters, plus the `flux_*` predistortion params forwarded to the inner experiment. See [run_t1_predist_fastflux.py](run_t1_predist_fastflux.py) at the repo root for a complete runner.

### 6.4 Using an envelope in your own program

```python
fs_mhz = self.soccfg['gens'][flux_ch]['f_fabric']
maxv = self.soccfg.get_maxv(flux_ch)

env = make_slow_ramp(alphas, taus, alpha0, duration_us=total_us,
                     fs_mhz=fs_mhz, maxv=maxv)
self.add_envelope(flux_ch, name="flux_predist", idata=env["idata"])
self.add_pulse(ch=flux_ch, name="flux_pulse", style="arb", envelope="flux_predist",
               freq=0, phase=0, gain=nominal_gain * env["gain_scale"],
               outsel="product")
```

## 7. Choosing an experiment

| Goal | Use |
|---|---|
| Map qubit freq vs flux, fit flux model | `QubitSpecFastFlux` (`update=True`) |
| Map resonator vs flux | `ResSpecFastFlux` |
| Characterize flux-line step response | `QubitSpecFluxSweep` sweeping `flux_lead_time`, then `fit_step_response_flux` |
| Quick T1 vs flux survey | `T1FastFluxLoop` (`lin_freq=True`) |
| Long T1(f, t) stability maps | `T1ContFluxRepeated` (inline calibration) or `T1FastFluxLoopRepeated` |
| Gold-standard T1 at each flux | `T1FastFlux` |
| Clean flux steps despite line distortion | predistortion pipeline + `T1Predist` / `QubitSpecPredist` |
| Readout calibration at operating flux | `HistogramFluxExperiment` |
