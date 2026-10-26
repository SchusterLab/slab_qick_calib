# SMPD script runner — agent orientation

Operational usage is in [README.md](README.md). This file is the map: how the
scripts, experiment classes, configs, and data layout fit together.

## The four moving parts

| Part | Location | Role |
|---|---|---|
| Runner scripts | `scripts/SMPD/NN_*.py` | Thin, editable "cells". Pick `qi`, build a `params` dict, instantiate one experiment class, run it. |
| Shared harness | [common.py](common.py) | Arg parsing, hardware connect, run-folder allocation, plot handling. Every script imports from here. |
| Experiment classes | `experiments/**` | The actual QICK programs + acquire/analyze/display/update logic. |
| Config YAML | `configs/*.yml` | Per-run device state. Read at startup, written back by `update()` calls. |

Scripts are the script-ified form of cells in `notebooks/SMPDs/tests_smpd_v2_waymar_rfboard.ipynb`
(and the `tune_qubits_smpd_v2_run*.ipynb` family). When notebook and script
disagree, the notebook is usually the newer experiment; the script is the
reproducible version.

## Runner script anatomy

Every script is the same five moves:

```python
"""Docstring becomes the --help description."""
import slab_qick_calib.experiments as meas
from common import finish_experiment, parse_args, prepare_experiment

if __name__ == "__main__":
    CONFIG = "..._bfg33_smpd_v2_yuvi_rfboard.yml"   # <- picks the device
    qi = 1                       # qubit / waste line index
    buffer_line_qi = 0           # buffer line index (readout index 0)
    JPA_ON = True
    args = parse_args(__doc__, config=CONFIG)
    run_dir, cfg_path, auto_cfg, soc, cfg_dict = prepare_experiment(
        args, __file__, qi, JPA_ON, additional_readout_qis=(0,)
    )
    params = {...}               # <- the only part normally edited
    expt = meas.SomeExperiment(cfg_dict, qi=qi, go=False, params=params)
    finish_experiment(expt, args)
    if not args.no_update_config:
        expt.update()
```

`go=False` + `finish_experiment` is the house style — never `go=True` in a
script, because `finish_experiment` owns display/save/plot-window behavior.
One consequence worth knowing: `qubit_run()` is never reached, so anything it
would have set up has to happen in the experiment's `__init__` instead. That is
why every reset-capable class calls `configure_reset()` there (see Active
reset) — `T1Experiment` had it behind `if go:` and so silently never configured
reset from a script.

**`CONFIG` is how you switch device.** It names the config file, and the run
number and device name are read back out of that filename
(`common.run_and_device_from_config`), so editing that one line moves the data
folder too. `common.py` holds no device default at all — changing device must
never mean editing the shared harness. `--config <other>.yml` overrides it for
a one-off run without editing the file.

### `common.py` contract

- `parse_args(__doc__, config=CONFIG)` → flags shared by all scripts:
  `--config`, `--run-number`, `--device-name`, `--port` (9090),
  `--no-update-config`, `--no-show`, `--no-progress`. `config` is a **required
  keyword-only Python argument**, supplied by the `CONFIG` constant at the top
  of each script's `__main__` — there is deliberately no default config in
  `common.py`, so a script that forgets it raises `TypeError` rather than
  running against whichever device a fallback named.
- `run_and_device_from_config(config_name)` → `("33", "Yuvi")` from
  `..._bfg33_smpd_v2_yuvi_rfboard.yml`. `--run-number` / `--device-name` default
  to `None` and are filled from this, so the config filename alone decides which
  `Data/run<N>_<Device>/` folder a run lands in. It raises on a name that does
  not match `YYYY_MM_DD_bfg<run>_..._<device>_rfboard.yml` rather than guessing.
- `prepare_experiment(args, __file__, qi, jpa_on, additional_readout_qis=())`:
  1. allocates the numbered run folder (see Data layout),
  2. copies the running script into the metadata folder,
  3. loads the YAML, connects `InstrumentManager` → `soc` → `soccfg`,
  4. calls `rfboard.activate_qubit_rf(qi, ...)` for attenuators/filters/DC bias,
  5. activates `dac_readout` only for each extra index in `additional_readout_qis`
     (needed whenever a second line — e.g. the buffer at index 0 — must transmit),
  6. optionally turns on the JPA pump from `hw.soc.dacs.jpa[qi]`.
  Returns `(run_dir, cfg_path, auto_cfg, soc, cfg_dict)`.
- `cfg_dict` keys: `soc`, `expt_path` (run root), `data_path`, `metadata_path`,
  `cfg_file`, `im`. `data_path`/`metadata_path` are what redirect saves into the
  per-experiment numbered folder.
- `finish_experiment(expt, args, wait=True)` → `expt.run(display, save, progress)`
  then blocks on open figures unless `--no-show`.
- `fwm_gain_for_frequency(auto_cfg, pump_frequency)` → pump gain from
  `device.fwm.pump_power_fit` (see FWM section).
- `effective_final_delay(cfg.expt)` → the inter-shot tail actually spent, in us.
  Active reset replaces the passive `final_delay` with its own much shorter
  sequence, so anything quoting a cycle time or duty cycle must go through this
  rather than reading `final_delay`. See the Active reset section.
- `experiment_name_for_script()` maps script filename → folder-name label.
  **Adding a new numbered script means adding an entry to that dict**, otherwise
  the folder falls back to the bare filename stem.
- `suppress_duplicate_config(experiment, metadata_path)` — see below.

### One config yml per run

`save_config()` writes `<prefix>_000NN.yml`, so a script that builds one
experiment per point writes one yml per point: **421 of them, 1.5 MB, for a
full script-26 grid**. Consecutive copies differ by the single swept line, and
that value is already in each point's h5 (`h5.load_config()`, which is what
`fit_shared_means.py` and `fwm_qc.load_saved_run` actually read) and in
`qc_log.csv`. One copy is the run's device record; the rest are noise.

`common.suppress_duplicate_config(experiment, metadata_path)`, called right
after constructing each experiment, no-ops that instance's `save_config` if the
folder already holds a yml. It patches the **instance**, not the class, so it
covers both ways a script persists the config: an explicit `save_config()` call
(26/37/40, 12c) and the one inside `display()` (23/24). Single-point scripts are
unaffected — their folder is empty, so the first write lands.

Applied to the scripts that produced many: `23` and `26`/`40` (421), `37` (85),
`12c` (20), `24` (10). Deliberately **not** applied to the small-N loops —
`36` (8), `35b` (4), `25` and `18` (2) — where a yml per point is useful
documentation and costs nothing. The kept copy is the first point's, so its
`expt` block shows that point's swept value, not a run-wide setting; read the
swept axis from the h5 or `qc_log.csv`, not from the yml.

## Experiment class hierarchy

All in `experiments/`, auto-re-exported flat by [`experiments/__init__.py`](../../experiments/__init__.py)
— so `meas.FourWMPowerExperiment` works regardless of which subpackage it lives
in. `general/` is imported first so it can serve as a superclass.

- `QickExperiment` (`experiments/general/qick_experiment.py`) — base. Owns
  `reps`/`rounds` from `device.readout`, `acquire`, `analyze`, `display`,
  `run`, `save_data`, `save_fig`, `save_config`, `make_hist`, `check_params`.
- `QickExperimentLoop` — software-looped 1D.
- `QickExperiment2DBase` → `QickExperiment2D`, `QickExperiment2DSweep`, and
  **`QickExperiment2DSimple`** — the last wraps an existing 1D experiment
  (`self.expt = Inner1D(...)`) and sweeps one of its `cfg.expt` keys as the
  y-axis. All the FWM 2D experiments use this pattern.
- `QickProgram` (`experiments/general/qick_program.py`) — pulse-sequence base;
  `make_pulse`, `make_cfg_pulse`, `measure`, reset helpers.

### The 2DSimple idiom

```python
self.expt = QubitSpecShot(cfg_dict, qi=qi, go=False, params=params,
                          check_params=False)   # inner 1D, never run directly
params = {**self.expt.cfg.expt, **params}       # inherit inner defaults
self.cfg.expt = params
...
def acquire(self, progress=False):
    ysweep = [{"pts": gainpts, "var": "gain_p"}]   # y-axis = an inner cfg.expt key
    super().acquire(ysweep, progress=progress)
```

The y-axis points land in `data["ypts"]` and also `data["<var>_pts"]`
(e.g. `data["gain_p_pts"]`, `data["frequency_b_pts"]`).

### The P(e) display layout

`QickExperiment2DBase.display_pe()` is the shared 2D display for shot-based
experiments: amplitude and phase stacked on the left, **P(e) spanning both
gridspec rows on the right** so it sits vertically centered beside them. Used by
all three FWM 2D experiments (scripts 18/19/20) — their own `display()` methods
just build a title and delegate.

It falls back to the plain `QickExperiment2DBase.display()` when `data` has no
`'p_e'` key, so a run without shots still plots. If `data` carries `'band_freq'`
(only script 18 fits one) the trajectory is overlaid on the P(e) panel, labelled
with the chosen model. Axis labels come from `self.xlabel` / `self.ylabel`,
which each experiment sets in `acquire()`.

`p_e` itself is computed in each experiment's `analyze()` by thresholding raw
shots against `device.readout.threshold[qi]`, then `shots` is dropped from
`data` to keep the h5 small (`save_shots: True` retains it in script 20).

**Script 20 draws a second figure.** `FWM_gainb_tb_sweep.analyze()` also
reduces its P(e) map to a conversion efficiency at each buffer length
(`data["slopes"]`) plus the plateau average (`data["efficiency"]`), and
`display_efficiency()` plots efficiency vs buffer length — saved with an
`_efficiency` filename suffix next to the P(e) panels. This is the same
reduction script 23 applies at every pump power, so both share
`_efficiency_from_pe` (see below); script 20 is the single-pump-power view of
it. When `n_photon_per_gain` is unavailable the reduction is skipped with a
printed note and only the P(e) figure is drawn.

## FWM experiments

Naming convention across all FWM code: suffix `_p` = **pump** (on the qubit DAC),
`_b` = **buffer** (on readout DAC/ADC index 0), `_r` = **readout/waste**
(readout index 1). So `frequency_p`/`gain_p`/`length_p`, `frequency_b`/`gain_b`/
`length_b`/`delay_b`, etc.

**Pulse shapes come from the config, not the scripts.** Each is `"flat_top"`
(make_pulse's default 0.02 us x 3 ramps on top of the flat `length`) or
`"const"` (square, no ramps); anything else raises, and a missing key means
`flat_top`. Helpers: `helpers/pulse_shapes.py` (`pulse_type_from_config`,
`ramp_length`), re-exported by `qick_program.py`.

| shape | config key | becomes | also used by |
|---|---|---|---|
| readout | `device.readout.pulse_type[qi]` | `QickProgram.readout_pulse_type` | — |
| FWM pump | `device.fwm.pump_pulse_type` | `pulse_type_p` | script-39 T1 pump |
| FWM buffer | `device.fwm.buffer_pulse_type` | `pulse_type_b` | script-38 Ramsey buffer, script-17 Stark tone (`stark_pulse_type`) |
| f0-g1 drive | `device.f0g1.pulse_type` | `pulse_type_f0g1`, `f0g1_reset_pulse_type` | — |
| e-f drive | `device.f0g1.pulse_type_ef` | `pulse_type_ef`, `f0g1_reset_pulse_type_ef`, `EFRabiPower`'s / script 29's `pulse_type` | — |

They are defaults, so a `params` value still wins, and the resolved value lands
in `cfg.expt` -- which is what `fit_shared_means.py`, `fwm_qc` and scripts
26/37 read back to rescale the photon calibration. The tones that share a key
share it because they share a calibration: the Stark and Ramsey photon numbers
only transfer to the FWM buffer if all three are played with the same shape.
The two f0-g1 reset tones must match each other (`f0g1_reset_params` and
`ResetTimeScan` raise otherwise), or one drive outlasts the other.

**qick plays a flat_top's plateau at half the amplitude of a const pulse at the
same gain** (the flat segment's gain register is halved; both use the
DDS-only output path). So switching a key between the two is 2x in amplitude
and 4x in photons: `device.fwm.pump_power_fit`, `buffer_power_fit`, photon
calibrations and `device.readout.gain` taken with the other shape are off
until re-measured. Every timing sum that adds ramps goes through
`ramp_length`, so a const drive adds none.

Readout exceptions that are always const: mux gens (no envelope memory),
`readout="long"` (periodic, which qick does not allow for flat_top), the LO
`mix_pulse`, and `QickProgram2Q`. The script-17 qubit probe's `pulse_type`
stays const, and the probe ends at the end of the Stark plateau.

| Script | Class | File | Sweep (x → y) |
|---|---|---|---|
| `18_fwm_power.py` | `FourWMPowerExperiment` (alias `QubitSpecShotPower`) | `single_qubit/qubit_spec_shot.py:551` | pump freq → pump gain |
| `19_fwm_buffer.py` | `QubitSpecShotBuffer` | `single_qubit/qubit_spec_shot.py:851` | pump freq → buffer freq |
| `20_fwm_buffer_gain_length.py` | `FWM_gainb_tb_sweep` | `single_qubit/FWM_sweeps.py:779` | buffer gain → buffer length |
| `21_fwm_histogram.py` | `SMPDHistogramExperiment` | `smpd/fwm_shots.py` | no sweep — histogram at one parameter set |
| `23_fwm_efficiency_vs_power.py` | `SMPDHistogramExperiment` x 30 | `smpd/fwm_shots.py` | full histogram suite over pump gain x buffer gain |
| `25_chi.py` | `ResSpec` x2 | `single_qubit/resonator_spectroscopy.py` | resonator freq, in |g> and |e> |
| `24_fwm_histogram_vs_power.py` | `SMPDHistogramExperiment` × N | `smpd/fwm_shots.py` | one histogram per point along the band |
| `26_fwm_efficiency_vs_power_checked.py` | `SMPDHistogramExperiment` × N | `smpd/fwm_shots.py` + [fwm_qc.py](fwm_qc.py) | pump gain × buffer gain at one fixed buffer length, every point QC'd and retaken if it fails |
| `40_fwm_efficiency_vs_buffer_length.py` | `SMPDHistogramExperiment` × N | `smpd/fwm_shots.py` + [fwm_qc.py](fwm_qc.py) | script 26 plus a buffer-length axis on the outside; summary rows by `length_b` per pump gain, one curve per pump gain in the figure |
| `37_fwm_efficiency_vs_buffer_frequency.py` | `SMPDHistogramExperiment` × 85 | `smpd/fwm_shots.py` + [fwm_qc.py](fwm_qc.py) | script 26's QC at one pump power, over buffer frequency × buffer gain; summary rows by `frequency_b` (`reduce_to_nep(by=...)`), Ramsey photon calibration rescaled to the applied attenuation/pulse. `fit_shared_means.py` is copied in and auto-detects the frequency sweep axis |
| `39_t1_pump.py` | `T1Experiment` with `pump=True` | `single_qubit/t1.py` | wait time, pump on throughout; never writes T1 to config |
| `17_stark_spectroscopy.py` | `StarkSpec` | `single_qubit/stark_spectroscopy.py` | qubit freq → buffer gain |
| `17b_stark_probe_saturation.py` | `QubitSpecPower` | `single_qubit/pulse_probe_spectroscopy.py` | qubit freq → probe gain, no buffer: picks script 17's probe gain (see "Probe power and the PNS comb") |
| `17c_stark_shift_offresonant.py` | `StarkSpec` (`stark_shift`) | `single_qubit/stark_spectroscopy.py` | qubit freq → buffer gain, buffer off resonance: the Stark photon calibration, writes `nstark` |
| `42_buffer_ringdown.py` | `ToFCalibrationExperiment` (`drive_length`) | `single_qubit/tof_calibration.py` | buffer ADC trace through a 2 µs drive and its free decay → κ (energy decay) |
| `38_t2_photon_calibration.py` | `T2PhotonCal` | `single_qubit/t2.py` | buffer gain (Ramsey at each), drive at f_g by default |
| `41_buffer_dephasing_spectroscopy.py` | `BufferDephasingSpec` | `single_qubit/t2.py` | buffer frequency (Ramsey at each) → f_g, chi, κ, photons per gain² |
| `22_readout_optimization.py` | `SingleShotOptExperiment` + `HistogramExperiment` | `single_qubit/single_shot_opt.py` | freq × gain × length |

Related 1D siblings in `FWM_sweeps.py`: `FWM_tb_sweep`, `FWM_gainb_sweep`,
`FWM_tp_sweep`. `single_qubit/qubit_buffer_spec_shot.py` is a further variant.

### Single-shot histogram (script 21)

`SMPDHistogramExperiment` is the simplest of the FWM experiments: no sweeps at
all. Its program is `QubitSpecShotProgram`'s sequence with the frequency loop
replaced by a shot loop, and it runs **four** acquisitions at one fixed
parameter set, differing only in which drives are on (the `SCANS` table):

| label | pump | buffer | weight | legend |
|---|---|---|---|---|
| `g` | off | on | `dark_count` — buffer-induced | no pump |
| `e` | on | on | `Pe` — the conversion | buffer + pump |
| `p` | on | off | `pump_background` — pump-induced | no buffer |
| `n` | off | off | `readout_background` — misassignment + thermal | no drives |

The three reference scans are what let you attribute the excited population.
`cfg.expt.scans` selects which to run; `'g'` and `'e'` are required since the
threshold and fidelity come from that pair.

> **The table above is the old scan set.** `g` is now *no drives*, the
> buffer-only sequence moved to `b` and is off by default, and `d` (pump only,
> detuned off the band) was added. See
> ["The detuned-pump scan"](#the-detuned-pump-scan) for the current table and
> why. Runs before that change still load — `n` stays defined everywhere.

**The fit is the T1-decay state model, not plain Gaussians.** Each histogram is
`helpers.combined_state_model`: a ground-level gaussian plus an excited
gaussian carrying an erf T1-decay tail, the excited weight being
`1 - mag_ground`. `analyze()` stores per scan `ground_hist_<label>` and
`excited_hist_<label>` (which sum to the total fit), `mag_ground_<label>`, and
the weight.

**Only the population split is fit.** `vg`, `ve`, `sigma` and `tm` come from
the single-shot calibration — `device.readout.g_mean` / `e_mean` / `sigma` /
`tm`, written by `12_single_shot.py` — so every scan and every pump power is
measured against identical state levels and decay. Override via params if
needed; a missing config entry raises pointing at script 12.

> **Stale.** The working tree's `experiments/smpd/fwm_shots.py` fits nothing —
> `analyze()` histograms each scan and reads a counting fidelity off the g/e
> pair, and its docstring says so explicitly. The fitting described in this
> section was real at run 32 and has since been removed. The claim that pinning
> the levels from the config holds the error to 0.005 does **not** hold on run
> 33 `#138`, where the stored calibration was a full σ from that run's own
> blobs; see the script-26 section below for the measurement.

Two things this deliberately avoids:

- **`helpers.fit_single_shot` is not used**, despite being the codebase's own
  joint T1 fit. It seeds the excited level with the e-shot's *weighted mean*,
  which is fine when state preparation is near-complete but lands between the
  two blobs when a shot is only partly excited — as FWM shots are. From there
  it converges to a degenerate solution: on synthetic data with `ve=13.5,
  Tm/T1=0.18` and a 55%-excited shot it returned `ve=62.3, Tm/T1=2.21`,
  inflating every weight. It is correct for `single_shot`'s own near-complete
  preparation; just not for this regime.
- **Refitting the state parameters per run** was the next attempt and also
  fails: pinning them on the pump+buffer scan needs a well-populated excited
  blob, so `Pe` came out 0.166 against a true 0.115 at the low-power end of a
  sweep — exactly where `24_fwm_histogram_vs_power` starts. Taking them from
  the calibration holds the worst error across a full sweep to 0.005.

**Every sequence takes the same time.** With no drives, `delay_auto` has no
pulse end to reference and would put the readout at `t=0.01` — tens of µs
earlier than the others, so the qubit would have had no time to relax. The
`n` scan therefore uses an explicit `self.delay(max(length_p, delay_b +
length_b) + gap)` instead: `delay()` increments the reference time, whereas
`delay_auto()` *sets* it, so mixing the two would silently discard the wait.

`analyze()` reads threshold/fidelity from
`helpers.analyze_single_shot_histograms` — the same routine `single_shot` uses
— then histograms I on one shared set of bins and stores the keys that
routine's `seq_mode='fwm'` branch plots: `vhg`/`histg` (and `e`, `p`, `n`),
`mag1_g`/`mu1_g`/`sig1_g`/`mag2_g`/… plus the weights and
`separation_sigma_*`. It iterates `SCANS` filtered by which `I*` keys are
present, so a reloaded data set analyses the scans it actually contains rather
than whatever `cfg.expt.scans` currently says.

**No rotation.** Both the analysis and the display pass `rotate=False`. The
readout phase is taken as already calibrated — `12_single_shot.py` /
`22_readout_optimization.py` have run and `device.readout.phase` is applied to
the readout pulse — so I is the discriminating axis as acquired. Deriving an
angle from the g/e medians here would double-correct, and with a small dark
count that angle is unreliable anyway. `rotate` defaults to `True` in the
shared helper so `single_shot`, whose job is to *find* the rotation, is
unaffected.

`display()` delegates to that same helper for the standard 2×2 figure, then
`_overlay_extra_scans()` adds the `p` (green dashed) and `n` (grey dash-dot)
raw traces, and `_overlay_state_fits()` draws every scan's total fit (black)
and its excited component (dotted, scan colour) on both histogram panels. The
helper only ever draws the g-shot's own fit and nothing for `p`/`n`, hence the
overlays. `_relabel_histogram_legend()` then renames its `g`/`e` legend entries
to the names in the table above, leaving `single_shot`'s own legends alone.
The log panel is where the excited components are readable — they stack in the
expected order (pump+buffer > pump only > buffer only > no drives).

### Histogram vs pump power (script 24)

`24_fwm_histogram_vs_power.py` is a plain `for` loop in the script — no new
experiment class. It steps **equally in frequency** between the band
frequencies at `GAIN_MIN` and `GAIN_MAX` (the gain range the power scan
covers), gets the gain at each step back from
`common.fwm_gain_for_frequency`, and runs a full four-sequence
`SMPDHistogramExperiment` there. For a quadratic fit the gain steps are
therefore uneven — equal frequency spacing is the requested axis.

**Each experiment must be constructed inside the loop, after the previous one
has saved.** `Experiment.__init__` picks its filename with
`get_next_filename`, which scans the run folder for the highest existing
index — so building all five up front gives every one of them
`fwm_histogram_qubit1_00000` and four of the five figures get overwritten.
Built in sequence they come out `_00000` … `_00004`, one PNG each in
`metadata_and_images/`.

After the loop it saves `excited_vs_pump_power.png`: the `excited_hist_e`
component from every step, overlaid and coloured by pump gain. Curves rather
than a 2D map because each run histograms over its own range, so a pcolormesh
would need interpolating onto a common axis first.

`finish_experiment(..., wait=False)` keeps the loop moving instead of blocking
per step, and each iteration closes the previous step's window before the next
plot appears — so only one figure is ever on screen while all five PNGs are
still written. Figures are closed by number (`plt.close(fignum)` over a saved
`plt.get_fignums()`), never via `plt.figure(num)`, which re-raises the window
and causes the flashing described above. `wait_for_plots()` at the end blocks
on the final figure.

### Pulse sequence (`QubitSpecShotProgram`, `qubit_spec_shot.py:37`)

Declares a second generator on `hw.soc.dacs.readout.ch[0]` for the buffer
(mixer freq set to `frequency_b - 400` when the gen has a mixer), builds
`qubit_pulse` from the `_p` params and `buffer_pulse` from the `_b` params.
Body: optional `pi_ge` (if `checkEF`) → `qubit_pulse` (if `do_pump`) →
`buffer_pulse` at `t=delay_b` (if `do_buffer`) → `delay_auto` (if `sep_readout`)
→ `measure`. `do_pump`/`do_buffer` are the switches for background checks.

### Expected FWM frequency

`18_fwm_power.py` computes the scan center from config:

```
f_fwm ≈ f_ge[qi] + f_readout[waste] - chi[waste] - f_readout[buffer]
```

Scripts 19–21 **hardcode** `pump_frequency = 5210` and derive `gain_p` from the
parabola fit (see below).

### FWM analysis → config (new, and the current focus)

`_track_fwm_band()` (`qubit_spec_shot.py`) finds the bright FWM band in the
`P(e)` map robustly. Naive per-row `argmax` is only a seed; a polynomial
trajectory model is then fit with iterative sigma-clipping, and `P(e)` is
re-read in a ±`window_pts` window around the fitted trajectory so stray bright
points elsewhere in a row can never be reported as the band maximum.

Five candidate models, all `freq = Σ coeffᵢ·designᵢ + c` through the shared
sigma-clipping loop `_fit_band_poly()`:

| `pump_power_fit_type` | Model | `designs` | stored as |
|---|---|---|---|
| `quadratic` | `freq = a·gain² + c` (vertex forced to zero gain) | `[gain²]` | `[a, c]` |
| `linear` | `freq = a·gain + c` | `[gain]` | `[a, c]` |
| `linear_quadratic` | `freq = a·gain² + b·gain + c` (vertex free) | `[gain², gain]` | `[a, b, c]` |
| `cubic` | `freq = a·gain³ + b·gain² + c·gain + d` | `[gain³, gain², gain]` | `[a, b, c, d]` |
| `quartic` | `freq = a·gain⁴ + b·gain³ + c·gain² + d·gain + e` | `[gain⁴, gain³, gain², gain]` | `[a, b, c, d, e]` |

With `cfg.expt.pump_power_fit_type = "auto"` (the default in `params_def`) all
five are fit and one is chosen; set it to a model name in a script's `params`
to force that one and skip the selection.

**How the winner is picked**, in four stages, because the models do not all
have the same number of free parameters and are not all nested in each other:

1. **`quadratic` vs `linear`** — both two-parameter, so they are compared
   directly by **median absolute residual over all seed points**, not over each
   fit's own inliers. That matters: sigma-clipping lets a model discard the
   points it fits worst, so scoring on inliers would let the wrong model win by
   throwing away its own evidence.
2. **`linear_quadratic` against that winner** by an **F-test at 95% on the one
   extra parameter**. It contains both two-parameter models as special cases
   (`b = 0` and `a = 0`), so it can always reach at least as small a residual —
   a direct residual comparison would hand it every scan. The F-test is the
   textbook comparison for nested models, and the same rule `fit_ef_doublet`
   uses to accept a doublet over a single Lorentzian. It uses R² over all seed
   points, for the same reason the median residual does.
3. **`cubic` against `linear_quadratic`**, by the same F-test, **only if stage 2
   adopted `linear_quadratic`**. `cubic` nests `linear_quadratic` by exactly one
   parameter (the `gain³` term), so that is the only baseline the test is valid
   against — comparing it to `quadratic` or `linear` directly would be testing
   two added terms at once, which the F-test's degrees of freedom don't cover.
   One consequence: a scan whose vertex term wasn't justified at stage 2 never
   gets a chance at `cubic` in `auto`. Force `fit_type='cubic'` to fit it
   regardless.
4. **`quartic` against `cubic`**, the same way again, **only if stage 3
   adopted `cubic`**, for the same nesting reason one level up. Reaching
   `quartic` in `auto` therefore needs *three* F-tests to all clear 95% on the
   same noise draw, and a 5-parameter fit has little contrast left in a
   typical ~30–70-point seed set to resolve — in practice `auto` lands on
   `quartic` rarely to never, and that's closer to correct than not: each
   stage's gate looks at the *lower-order truncation's own shape*, so a
   `quartic` that is itself perfectly monotonic can still have a best-fit
   parabola whose vertex falls inside `(0, 1)` (the parabola cuts across an
   inflection the quartic has and the parabola doesn't), which correctly
   blocks the whole chain at stage 2 regardless of noise. Treat an
   `auto`-chosen `quartic` as worth a second look rather than a settled
   result; force it deliberately if that is really the fit wanted, and expect
   it to not necessarily beat `cubic` or `linear_quadratic` on residual even
   then — on run 34 Waymar `#196`, forcing `quartic` gave 0.257 MHz against
   `linear_quadratic`'s 0.163 MHz.

**The gain² coefficient is constrained to be ≤ 0**, in every model that has
one, selection or not. The pump Stark shift grows with power, so the band can
only bend downward. With one inequality constraint on a linear model the
solution is either the unconstrained fit or the one on the boundary, so
`_fit_band_poly(..., nonpositive_lead=True)` just drops the `gain²` column and
refits when the unconstrained fit comes out positive — no optimizer. On the
boundary `linear_quadratic` *is* the linear model, stored as `[0, b, c]`, and
`band_gain_from_fit` inverts it as one. **`cubic` and `quartic` carry no such
constraint** — there's no equally simple physical argument for the sign of a
higher-order term, and enforcing concavity of a cubic or quartic over an
interval needs real constrained optimization rather than the "drop a column
and see" trick the quadratic terms use.

This is not hypothetical. Forcing `linear_quadratic` on run 33 Waymar wrote
`pump_power_fit: [6.4563, -16.535, 5218.9857]` — `a = +6.46`, concave up. It
falls monotonically over the gain range and its vertex is at gain 1.28, so no
other gate catches it; the pump gains it yields are simply wrong.

The constraint only ever binds on `linear_quadratic`, and effectively cannot on
`quadratic`: with its vertex pinned at zero gain, the sign of `a` there is the
sign of the *net* frequency change, not of the curvature, so a band that falls
overall always fits with `a < 0` however it bends.

**And a turning-point gate that auto applies and forcing does not**, for
`linear_quadratic`, `cubic`, and `quartic`. A `linear_quadratic` fit whose
vertex `-b/2a` falls inside the playable gain range `(0, 1)` describes a band
that doubles back, so `band_gain_from_fit` cannot invert it and every script
that reads the fit would raise. `cubic`'s and `quartic`'s analogous check is
on their *derivative*'s real roots — a quadratic (`3a·gain² + 2b·gain + c = 0`)
for `cubic`, a cubic (`4a·gain³ + 3b·gain² + 2c·gain + d = 0`, via `np.roots`
rather than a closed form) for `quartic` — since each turns over wherever its
slope hits zero, not at a single vertex. At 95% the F-test accepts a spurious
extra term on about 1 scan in 20, and a spurious one lands its turning point in
range often — measured on synthetic quadratic data at `F = 4.35` against
`F_crit = 4.24` with the vertex at gain 0.18. Auto must not pick a model the
rest of the tree cannot use. Forcing any of these models still gives it to
you; the inversion then says why it is unusable.

Adds to `data`: `band_fit_type`, **`band_fit_coeffs`** (the coefficient list in
the order above — this is what `update()` writes), `band_fit_a` / `band_fit_b`
/ `band_fit_c` (the first three coefficients positionally; `c` is always the
model's offset regardless of how many coefficients it has, and `b` is NaN for
the two-parameter models), `band_freq`, `band_p_e`, `band_freq_idx`, every
candidate's score as `band_resid_<name>` (MHz), `band_f_stat` / `band_f_crit`
from stage 2's comparison, `band_f_stat_cubic` / `band_f_crit_cubic` from
stage 3's, and `band_f_stat_quartic` / `band_f_crit_quartic` from stage 4's
(each pair NaN when its stage didn't run). `analyze()` prints the choice, all
five residuals, and every F-test verdict whose stage ran; `display()` labels
the overlaid trajectory with the chosen model.

`FourWMPowerExperiment.update()` writes two config fields via
`config.update_fwm()`:

- **`device.fwm.pump_power_fit`** = `band_fit_coeffs`, so **2 to 5 elements
  depending on the model**
- **`device.fwm.pump_power_fit_type`** = the model name

The type tag is what lets consumers invert the right form, and since it now also
sets the expected list *length*, the two must always be read together — never
`a, c = ...pump_power_fit` at a call site. `band_frequency_from_fit` /
`band_gain_from_fit` (both in `qubit_spec_shot.py`, both public) are the only
two places that unpack it; they raise if the coefficients and the tag are out of
step, which is what a hand-edited config produces.

Check all of this offline, with no hardware:

```
python check_band_fit.py    # the five models, auto, the guards, the inversion
```

`config.update_fwm(file_name, field, value, ...)` (`helpers/config.py`) is the
FWM-specific writer. Unlike `update_qubit`/`update_readout`/`update_stark`,
`device.fwm` is **not indexed per qubit** — there is one FWM device — and it
accepts a scalar, a list (each element formatted to `sig` digits), or a string
tag stored verbatim. It `setdefault`s the `fwm` section, so no config init step
is needed.

Reading it back: `auto_cfg.device.fwm.pump_power_fit`. The section was
originally named `4wm`; it was renamed to `fwm` precisely because a key starting
with a digit cannot be reached by attribute access.

### Consuming the fit (scripts 19–21)

Scripts 19/20/21 hardcode their **pump frequency** and derive the **pump gain**
from the stored fit via `common.fwm_gain_for_frequency(auto_cfg, pump_frequency)`,
which delegates to `band_gain_from_fit`:

```
quadratic:         gain = sqrt((f - c) / a)
linear:            gain = (f - c) / a
linear_quadratic:  gain = (-b ± sqrt(b² - 4a(c - f))) / 2a
cubic:             gain = min non-negative real root of
                          a·gain³ + b·gain² + c·gain + (d - f) = 0
quartic:           gain = min non-negative real root of
                          a·gain⁴ + b·gain³ + c·gain² + d·gain + (e - f) = 0
```

The inversion lives in one place rather than at each call site precisely
because of that third line — `common.fwm_frequency_for_gain`,
`FWMEfficiencyVsPower._pump_points` and script 24 all used to carry their own
copy of the dispatch, which was fine for two closed forms and is not for a
branch selection. `cubic` and `quartic` go through `np.roots` rather than a
closed form (`_real_roots`, which filters out the complex roots `np.roots`
always returns, keeping those within a tolerance of the real axis), since
there was no reason to hand-code Cardano's or Ferrari's formula once a
general root finder was already needed.

**Which root**: the one *nearest zero gain*, for `linear_quadratic`, `cubic`,
and `quartic` alike. The offset term anchors the fit at gain 0, so that is the
branch the measured data is continuously connected to. The case where more
than one root is in range cannot arise, because the turning-point gate above
rejects a band that folds back inside `(0, 1)` — `linear_quadratic`'s own
vertex for that model, the roots of `cubic`'s or `quartic`'s derivative for
those — and if one is forced through anyway, `band_gain_from_fit` raises
rather than picking a side.

It prints the model, its coefficients, and the resulting gain before the sweep
starts, and flags (but does not clamp) a gain above `device.qubit.max_gain` — a
wildly large gain means the fit is stale, which is worth seeing. A gain that
would come out negative, or a frequency past the turning point, raises: that
frequency is unreachable on the fitted band.

**Refresh order matters**: run `18_fwm_power.py` (with `update()`, i.e. without
`--no-update-config`) before 19–21, or they compute gains from a stale fit.

### Efficiency vs pump power (script 23)

`FWMEfficiencyVsPower` (`single_qubit/FWM_sweeps.py`) is the only 3-deep
experiment in the tree: it walks the band and runs a whole
`FWM_gainb_tb_sweep` at each pump power, so `data["p_e"]` is
`(pump_expts, length_b_expts, expts)`.

It is a plain `QickExperiment` with an explicit `for` loop in `acquire()` — the
`QickExperiment2DSimple` idiom only handles one nested level. Two gotchas it has
to work around:

- **The pump must reach the innermost config.** `QickExperiment2DSimple.acquire`
  only propagates its own y-sweep variable, so writing `gain_p` on the middle
  level never reaches the Program. `_set_pump()` writes both
  `self.expt.cfg.expt` (so the middle level's title is right) and
  `self.expt.expt.cfg.expt` (what the Program actually reads).
- **The nested sweep's interim saves are disabled** (`save_interim = False`),
  or it would rewrite its own file at every pump step and keep only the last.

Rather than sweeping pump frequency independently, the pump *gain* is stepped
and the frequency comes from whichever band fit model is stored (`a·gain² + c`,
`a·gain + c`, or one of the three higher-order forms — see "FWM analysis →
config"), so every point sits on the band. Raises at construction if
`device.fwm` has no fit.

**The efficiency metric** (`_efficiency_from_pe`) mirrors
`notebooks/SMPDs/load_lengthb_gainb_apr16_late.py`, the offline version, and is
shared by scripts 20 and 23:

1. Buffer gain → mean photon number `n = gain_b² · nstark`, then to
   `1 - P(0) = 1 - exp(-n)` (Poisson; equals `1 - poisson.pmf(0, n)` with no
   scipy needed).
2. At each buffer length, the slope of `P(e)` vs `1 - P(0)` over the linear
   window `[slope_lo, slope_hi]` = `[0.05, 0.2]`.
3. Average those slopes over buffer lengths past `length_b_min` (40 µs), where
   conversion has plateaued. Falls back to all lengths if the scan is shorter,
   so a short run returns a number instead of NaN.

Then `fit_efficiency_vs_power()` fits the trend to
`η = 4A·C/(1 + κᵢ/κ_c + C)²` with `C = (gain_p/scale)²`, printing
`scale`, `κᵢ/κ_c`, `A`. Non-fatal if it fails to converge.

**Dependency**: step 1 needs `device.readout.nstark[qi]` — photons per unit
gain², written by `17_stark_spectroscopy.py`'s `update()`. It is **not** in the
current config, so run script 17 first or pass `n_photon_per_gain` in `params`.
`analyze()` raises with that instruction rather than guessing.

### Photon number from Ramsey dephasing (scripts 38, 41)

Every buffer photon calibration is stated in one unit: **on-resonance photons
per gain²**, i.e. photons for a drive at f_g = `device.readout.frequency[0]`
(the |g>-dressed buffer resonance, where Stark and FWM drive), at a recorded
buffer attenuation and pulse shape. The Stark/PNS comb measures that directly;
the Ramsey routes convert to it.

**The physics** (`buffer_dephasing_per_photon`, `single_qubit/t2.py`). A buffer
drive at f_d builds `α_g = ε/(κ/2 − iΔ_g)` with the qubit in g and
`α_e = ε/(κ/2 − iΔ_e)` in e (Δ from f_g and f_e = f_g + chi[0]). In a Ramsey the
qubit is in both, so the fringe

- **dephases** at `Γ = (κ/2)·|α_g − α_e|²` (1/T2 rises), and
- **shifts** by `δ = chi·Re(α_g α_e*)` (sign is convention-dependent; fits use
  its magnitude or a free sign).

Both are per `n_on = ε²/(κ/2)²`. Units: `κ = 2π·kappa[0]` (config kappa is
the FWHM in MHz), chi[0] the full signed g→e shift. **The rate per photon
depends on where you drive**: at f_g, α_e ≈ 0 and Γ ≈ (κ/2)·n_on, which needs κ
only linearly; at the midpoint f_g + chi/2 the fields are equal and opposite,
Γ ≈ 2κ·n_mid, but n_mid is ~1/32 of n_on for the same gain, so converting a
midpoint run to f_g goes as ~χ²/κ³. `ramsey_photons_per_rate(kappa, chi,
detuning)` returns (photons at the drive, photons at f_g) per 1/µs;
`detuning=None` is the midpoint, the old formula exactly.

**Script 38** (`T2PhotonCal`) sweeps buffer gain at one `frequency_b`, now
**defaulting to f_g** (until 2026-10-16 it defaulted to the midpoint; every
run-34 script-38 run up to #500 drove at f_g − 0.680 MHz). It appends a closing
gain-0 Ramsey as a drift check (#500 had an 85 kHz qubit excursion mid-run that
nothing caught), stores each point's programmed `ramsey_freq`, and fits both
1/T2 and the shift vs gain². Their ratio depends only on κ (given chi), so it
also prints a **self-consistent κ** (`ramsey_kappa_from_ratio`). `update()`
writes `device.readout.nramsey[qi]` (on-resonance photons per gain², from
1/T2) and in `device.fwm`: `buffer_dephasing_ramsey` (slope, 1/µs per gain²),
`buffer_shift_ramsey` (MHz per gain²), and tags `_attn_db`, `_pulse`,
`_detuning` (frequency_b − f_g, MHz). It stores slopes, not photons, so **every
κ and χ in the chain is the config's current `device.readout`**:
`fit_shared_means` and `fwm_qc.load_calibration` convert with
`ramsey_photons_per_rate(kappa[0], chi[0], _detuning)` -- a missing `_detuning`
means the legacy midpoint -- and use the same κ (and `kappa_c[0]`) for the
arrival flux.

**The shift is the more reproducible observable.** On run 34, #333/#371/#500
gave shift slopes of −7791/−7806/−7834 kHz per gain² (1%), while their 1/T2
slopes scattered 12.4–17.1 (37%), and #333's 1/T2 curves upward.

**Script 41** (`BufferDephasingSpec`) is buffer spectroscopy read out through
the qubit: Ramsey at one gain over `frequency_b_pts` (default: ±0.35 MHz around
f_g and f_e in 50 kHz steps, plus the midpoint and one point outside each), with
a gain-0 reference every `ref_every` points as the drift baseline. Γ(f) has a
peak of width ~κ at each dressed resonance, δ(f) a dispersive shape, and
`fit_buffer_dephasing_spectrum` fits both for **f_g, chi, κ and n_on with no
config input**. On synthetic data at script 41's settings (70 kHz steps, 10000 reps) it recovers photons per gain² to ~3%, κ to ~2%, chi to ~3 kHz. The shift's sign is chosen with fixed block weights before the per-block rescaling; choosing it after (the first version) let the wrong sign win by de-weighting both blocks, biasing n/gain² ~2× low.
`update()` writes `device.fwm.buffer_photon_ramsey_spec` (photons per gain² at
f_g), `buffer_ramsey_spec_fit` = [f_g, chi, κ] and `_attn_db` / `_pulse`;
it deliberately leaves `device.readout` alone.

> **Buffer ResSpec cannot calibrate Waymar's buffer κ/f_g as is.** Its phase is
> not delay-corrected (~1150 rad across 2 MHz), so the complex fit fails and
> `update()` writes nonsense: run 34 #533–#536 walked `frequency[0]` from
> 6682.23 to 6683.06 and κ to 0.08–7.6, each scan re-centred on the last bad
> fit. A plain Lorentzian on `amps` gives f0 = 6682.26–6682.27, FWHM 0.19–0.21.
> Use script 41 for κ/chi/f_g.

**Pooling runs (optional).** `analysis/combine_t2_photon_cal.py` fits several
script-38 runs jointly: one shared slope in a reference gain, one 1/T2(0)
intercept per run, each run's gain² rescaled by its applied attenuation and
pulse shape (`const` = 4× `flat_top`). The runs must share kappa_b, chi_b and
drive detuning. It prints the joint slope; pasting it into
`device.fwm.buffer_dephasing_ramsey` (with `_attn_db` = its `--ref-attn`,
`_pulse: flat_top`, `_detuning` as printed) is manual, and the next script-38
run overwrites it.

**Stark calibration: the off-resonant line shift (script 17c).**
`17c_stark_shift_offresonant.py` runs `StarkSpec` (`fit_type="stark_shift"`)
with the buffer at 6681.5503 MHz, about f_g − 0.74. That far off resonance the
qubit line shifts instead of splitting into photon-number lines. The shift
slope (MHz per gain²) converts to on-resonance photons per gain² through
`buffer_dephasing_per_photon`, the same physics as the Ramsey routes, using
`kappa[0]`, `chi_buffer` and `stark_freq − frequency[0]`. `update()` writes
`device.readout.nstark[qi]`, which scripts 20/23 read. It has no attn/pulse
tags: it is stated at 17c's buffer setting (10 dB, flat_top).

The PNS comb measures the same number without needing κ. See "Probe power and
the PNS comb".

**Choosing the calibration.** `fit_shared_means.py --calibration
ramsey_spec|ramsey|stark` (default `ramsey_spec`) picks the calibration. So
does the `PHOTON_CAL` constant in scripts 26 and 40, default `ramsey_spec`.
Script 37 always uses `ramsey_spec`. The keys:

| source | key | form |
|---|---|---|
| `ramsey_spec` | `buffer_photon_ramsey_spec` (script 41) | photons per gain² at f_g, as stored |
| `ramsey` | `buffer_dephasing_ramsey` (script 38) | slope, converted with config kappa/chi at `_detuning`; carries the T2 window bias below |
| `stark` | `buffer_power_fit` (refit_two_peak, PNS) | coefficients as stored; the per-row comb in refit_two_peak is unreliable at weak probe |

Each source is rescaled to the attenuation and pulse actually applied, using
its `_attn_db` / `_pulse` tags. The applied setting comes from
`fwm_qc.buffer_attenuation_db`, or per h5 from `rfboard_active` and
`pulse_type_b` in fit_shared_means. So one calibration serves runs at any
setting. A 10 dB mismatch is 10× in photons: the tags carry the result, they
do not decorate it.

**κ is the crux, and only the ring-down measures it (script 42).**
`42_buffer_ringdown.py` plays a 2 µs pulse at f_g on the buffer and records the
buffer ADC through the free decay. It fits a single-mode reflection model three
ways: the full trace, the drive-on segment, and the decay alone.
- **Use the decay.** On run 34 (#643) it is a clean single exponential:
  κ = 0.182 ± 0.003 from any start time, with a negligible second component.
- **The driven parts are distorted.** The drive-on segment gives 0.28, ResSpec
  gives 0.24 (its complex fit also returns κ_c > κ, which is unphysical), and
  41's free κ gave 0.207. Anything measured while the drive is on contains the
  directly reflected input and whatever else is in the input path.
- **Keep the drive linear.** A const pulse gives ~1000 photons per gain² at f_g.
  At gain 0.3 (#642, ~90 photons, near n_crit ~100) the qubit-induced Kerr made
  the decay chirp and the segments disagree. Use gain 0.05.

**Run 34 result (2026-10-17): 315 ± 10 on-resonance photons per gain² at
10 dB, flat_top.** `device.readout[0]` holds κ = κ_c = 0.182 (ring-down) and
f_g = 6682.295 (ring-down, 41 and the comb agree). ResSpec's 6682.259 sits
36 kHz low, which would cost ~13% of the photons at κ/2 = 91 kHz. Every route
agrees at κ = 0.182:

| measurement | photons / gain² at f_g |
|---|---|
| #333 / #371 / #500, Ramsey shift at f_g − 0.744 | 316 ± 4 / 317 ± 12 / 318 ± 9 |
| #555, script 17c Stark-program line shift, same drive | 326 (stored as `nstark`) |
| #554, script 41 refit with κ fixed: both / shift / dephasing | 308 ± 10 / 313 ± 13 / 299 ± 17 (308 stored as `buffer_photon_ramsey_spec`) |
| #553, script 38 1/T2 at f_g, envelope exp × Gaussian (T_g = 20 µs) | 296 ± 30 (232 with the exponential envelope) |
| #549–#551, PNS comb, global shared-area fit (κ-free) | ~326 ± 20 |

#555 shows the Stark and Ramsey programs give the same field per gain. The
qubit line shift at the same drive agrees to 3%. So three different
observables agree:
- the photon-number-split line weights, which need no κ;
- the dispersive line shift, which goes as 1/κ²;
- the measurement-induced dephasing, which goes as 1/κ.

**Before κ was measured,** the shift routes read 250 with 41's κ = 0.207 and
185 with ResSpec's 0.24, which looked like PNS overcounting by 1.3–1.8×.

**The T2 window bias.** T2PhotonCal and BufferDephasingSpec fit exponential
Ramsey envelopes. The intrinsic decay has a Gaussian part (T2E 26 µs against
T2R 15 µs; envelope T_g ≈ 20–40 µs). Buffer-on points use windows shrunk to
3·T2, so they capture less of the Gaussian part than the gain-0 references,
and ΔΓ comes out low: 20–35% on #553. Shift-based numbers are not affected.
The bias also pulls 41's free κ toward 0.207.

χ: 25_chi gives −1.293, the comb −1.283, 41 −1.28 to −1.30, and the config
holds −1.3029. The earlier −1.36 was never measured in run 34.

**Photon-calibration analysis tools** (all in `analysis/`, run from there with
run numbers; `--run run<N>_<Device>` picks the data folder):

| script | does |
|---|---|
| `compare_photon_cal.py` | Stark comb and Ramsey runs on the on-resonance axis; probe saturation from a 17b run |
| `pns_global.py` | global PNS fit (shared area scale, n̄ = c0 + c2·gain², finite-pulse lines, `--voigt`) |
| `pns_lineshape.py` | finite-pulse and Voigt line shapes, the comb, a per-row fit (shared vs free amplitude) |
| `pns_line_areas.py` | main-line area vs buffer gain (e.g. #555 off resonance) |
| `refit_dephasing_spec.py` | script-41 refit with κ fixed (`--kappa`) or free; both / shift / dephasing only |
| `ramsey_envelope_refit.py` | script-38 traces with exp × Gaussian envelopes (removes the window bias) |
| `sim_photon_cal.py` | master-equation check of the Ramsey formula and the PNS fit (qutip) |
| `combine_t2_photon_cal.py` | pool several script-38 runs on one slope |

**Comparing them.** `analysis/compare_photon_cal.py --stark N --ramsey N ...
[--exclude RUN:i-j] [--saturation N]` puts Stark (both comb normalizations)
and Ramsey (1/T2 and shift, config vs PNS chi, config vs self-consistent κ) on
the on-resonance axis. It also reports probe saturation from a 17b run. It
takes each Ramsey run's detuning from that run's own saved `frequency[0]`, so
pass `--chi` / `--kappa` with the current values when the config has moved.

### Probe power and the PNS comb

The comb (`refit_two_peak.py`) weights each photon-number line by Poisson(n).
Which property of a line carries that weight depends on the probe:

- **Weak probe: the area is ∝ P(n), for any pulse length.** The n-th line is
  broader (κ-broadened), so its height is P(n)·γ₀/γₙ. `--normalization area`
  (the default) models this.
- **Saturated: the height is ∝ P(n).** `--normalization height` is the old
  model. Between the two neither is right, and the narrow n=0 line compresses
  first, biasing n̄ high.

**P(e) < 0.5 is not a criterion.** It holds at any power, and P(e) = 0.25 is
already strongly saturated. Measure the saturation with
`17b_stark_probe_saturation.py`, using script 17's exact probe.

- With a **100 µs** probe (#547), s at the old gain of 0.1 was 5–25. That is
  the steady state, `s = Ω²T₁T₂`.
- With the **20 µs** probe now in script 17 (#548), the probe is coherent:
  peak height ∝ sin²(θ/2), with a π rotation at gain 0.110. The compression of
  the n=0 peak below the weak-drive limit is 10.5% / 6% / 2.7% at gains
  0.04 / 0.03 / 0.02.
- At 20 µs, widths are pulse-limited (~20 kHz HWHM), so the width-based s
  that `--saturation` prints is not meaningful. Use heights.

**Fit form.** At weak probe, refit_two_peak's per-row fit (Lorentzian lines,
free amplitude per row) is underdetermined: #549 gave 153 (height) vs 377
(area). What works is a global fit:
- n̄ = c0 + c2·gain² imposed across rows;
- one line-area scale shared by all rows, so n=0 depletion counts;
- the exact finite-pulse weak-drive line shape;
- χ′ fixed to 0.

On master-equation-simulated spectra it recovers c2 to about 5%. On
#549–#551 it gives ~326, matching the κ = 0.182 routes. It is
`analysis/pns_global.py` (`python pns_global.py 549 550 551 [--voigt]`);
refit_two_peak still does the per-row fit.

Checks that came out negative:
- A Voigt line (Gaussian qubit frequency noise) leaves c2 unchanged.
- #555's n=0 area at gain 0.1 is 0.86–0.89 of the zero-gain area. That is not
  an anomaly: off resonance, the main line keeps e^{−|α_g−α_e|²} ≈ 0.77 of its
  weight and the rest goes to sidebands.

Script 17 is left set up for weak-probe PNS: 20 µs probe, f0g1 reset,
10 kHz steps. Its `stark_length` lead is +3 µs, and at κ = 0.182 the field
holds ~67% of its photons when the probe starts (photons rise as
(1−e^{−κt/2})²), ~97% averaged over the probe. So #549–#551 and #555 read ~3%
low. Scripts 17 and 17c now use +10 µs.

### The detuned-pump scan

The scan table is `g e p d b f`
([fwm_shots.py](../../experiments/smpd/fwm_shots.py)), with `SCANS` rows of
`(label, do_pump, do_buffer, do_pulse_f, detuned)`:

| key | pump | buffer | role |
|---|---|---|---|
| `g` | off | off | no drives — the ground reference |
| `e` | on | on | the conversion |
| `p` | on | off, on band | the dark count |
| `d` | on, `+detune_p` | off | non-resonant pump background |
| `b` | off | on | buffer only — **not requested by default** |
| `f` | on | on, then `pi_ef` | the conversion on the g/f axis |

**What `d` is for.** `p` conflates the *resonant* FWM dark count with
*non-resonant* pump-induced excitation — heating, quasiparticles, amplifier
noise. Since `p_dark` is the numerator of the NEP, that conflation sits under
the headline number. `d` is the same pump power off resonance, so `p - d` is
the part that needs the band and `d - g` the part that does not.

**`detune_p = 30.0` MHz, not 10.** The resonance FWHM *grows with pump gain* —
measured from run 34 `#472`'s band map: 0.6–2 MHz at gain 0.2–0.4, 2–3 at
0.5–0.6, 4–6 at 0.7–0.8, **8–9 MHz at 0.9–1.0**. At 10 MHz the gain-1.0 scan
would sit ~1.1 FWHM out, still on the shoulder with real conversion. 30 MHz is
3.2× the worst FWHM, clears the band manifold's 5225.7 MHz intercept at every
gain, and costs 2.7% in Stark shift (pump–qubit detuning is 1087 MHz). Nothing
else is within 1 GHz: `f_ge` 4133, `f_ef` 3889, buffer 6682, waste 7780.

**Why no-drives moved into the `g` slot.** `g` was buffer-on/pump-off and `n`
was all-drives-off; they differ only by the buffer tone. Over 687 accepted
points in six run-34 sweeps, `p_g - p_n` has median **+0.0002 (+0.2σ)** against
a 30 k-shot σ of 0.0009, with **symmetric** scatter (p5 −0.0118, p95 +0.0116) —
a real buffer-induced excitation would be one-sided positive. The 24.6% of
points beyond 5σ have `p_n` median 0.037 against 0.024 for the rest, i.e. they
are the known readout excursions, not the buffer. So the buffer tone alone
excites nothing measurable, and the two scans merged.

`g` had to be the surviving key: it is the "ground" slot
`analyze_single_shot_histograms` reads, `_scans()` hard-requires it, and
`fit_shared_means` seeds `vg`/`sigma` from its pooled histogram. Folding
no-drives into it keeps every one of those consumers working untouched, and
makes the reference slightly *cleaner* since no buffer tone is present. `g` now
takes the explicit-delay branch in `_body` — the one written for no-drives — so
it still waits out `max(length_p + ramp_p, delay_b + length_b + ramp_b) + gap`
and sees the same relaxation history as every other scan.

> **The merge is bounded to `gain_b` ≤ 0.03**, the range those runs used. At the
> 0.15–0.2 buffer gains earlier runs used, a direct buffer-induced excitation
> would scale with buffer gain and so land in the **efficiency slope**, not the
> free intercept — indistinguishable from conversion. That is why the
> buffer-only sequence stays in the table as `b`: add it back to `scans` if
> buffer gains go up again.

**`g` carries the floor gate.** Being no-drives it should sit near the thermal
population at every point of the grid whatever the sweep is doing, which is what
makes an absolute bar on it well-founded — see
["Why the gates are absolute"](#why-the-gates-are-absolute).

**Per script**, set explicitly in each `params`:

| script | `scans` | + `check_f` | total |
|---|---|---|---|
| 26 pump power | `("g", "e", "p", "d")` | `f` | 5 |
| 37 buffer frequency | `("g", "e", "p")` | `f` | 4 |
| 40 buffer length | `("g", "e", "p")` | `f` | 4 |

37's pump is fixed, so a `d` scan would be one repeated number; 40's pump only
changes duration. Both therefore run four scans, 20% cheaper than five. `p`
cannot be dropped from any of them: it is `DARK_SCAN` → `p_dark` → `n_equiv`,
`nep`, and `analysis/fit_shared_means.py:1175` hard-`KeyError`s without it
where `fwm_qc` merely NaNs out.

**Per-scan pump frequency** works because `acquire()` already rebuilds
`SMPDHistogramProgram` inside its loop, so `_initialize`/`make_pulse` re-run per
scan — setting `cfg.expt.frequency_p` in the loop is enough, with no second
pulse. It is restored to the on-band value afterwards so the saved yml records
that; the offset is recoverable from `cfg.expt.detune_p`.

**Backward compatibility.** `n` and `b` stay listed in `fwm_qc.SCANS`,
`fit_shared_means.SCANS` and the label/colour tables, so every saved run-33/34
set still loads, replots and reduces unchanged — verified by replaying `#531`
(72 points, old-style `g` plus `n`) through the new gates: 0 rejections, and
`reduce_to_nep` never reads `g` or `n` anyway. One cosmetic consequence:
`NEP_WATT_SCAN` moved from `n` to `d`, and a four-scan run has no spare grid
column, so 37/40 lose that one cell from `fit_shared_means_grid.png` —
`fwm_summary.png` still draws the NEP.

### QC'd efficiency sweep (script 26, [fwm_qc.py](fwm_qc.py))

Script 26 runs a pump gain × buffer gain grid at one fixed buffer length and
checks all five scans of every point
before moving on, retaking the point up to `MAX_ATTEMPTS = 3` times if any gate
fails. It exists because script 23 checks nothing: on run 33 `#138` a quarter of
the 130 points are unusable, including a ten-point block at pump gain 0.8 where
the ground blob moved ~1σ across all five scans at once and the no-drives
fraction went 0.03 → 0.25, and none of that was visible until
`fit_shared_means.py` was run hours later.

**What the gates check** — the reference scans are *repeated identical
measurements* over part of the grid, which is what makes them canaries:

| gate | scans | test |
|---|---|---|
| `peak` | all five | ground-blob argmax within `PEAK_TOL` σ of `device.readout.g_mean` |
| `width` | all five | ground-blob HWHM-derived σ within `[WIDTH_LO, WIDTH_HI]` × the config σ |
| `floor` | `g`, `n` | no-drives fraction **below** `GROUND_MAX_PE` (0.05), absolute |
| `pump` | `p`, `d` | pump-on background **below** `PUMP_MAX_PE` (0.10), absolute |
| `ordering` | `e` | `p_e ≥ max(p_g, p_p, p_n) - ORDER_TOL` |
| `anchor` | `e` | at `gain_b = 0` the pump+buffer sequence *is* the pump-only one, so `p_e ≈ p_p` |

`f` is **not** gated — only `peak`/`width`, which it shares with every scan.
`p_f/p_e` is still logged to `qc_log.csv` as a diagnostic; see below for why it
is not a tolerance.

**Nothing is fitted, deliberately.** Two model-based versions were built and
measured against `#138` first, and both failed on that data:

- Pinning `vg`/`ve`/`sigma`/`t_ratio` from `device.readout` and projecting — the
  approach the script-21 section above describes. That config was a full σ away
  from `#138`'s own blobs, biasing the `e`-scan fractions by −0.048 against
  `fit_shared_means.csv` (rms 0.076, not the 0.005 that section claims), so a
  residual gate would have measured the staleness of the YAML.
- Fitting those four shared parameters on the leading points, as
  `fit_shared_means.py` does over a whole run. The grid starts at its lowest
  pump *and* buffer gain, where every scan is nearly all ground and `ve`/the
  decay tail are unconstrained: unbounded the fit walked `t_ratio` to 31 and
  called every fraction 1.0; bounded it sat on the bound with `ve` swinging
  13→21 depending on how many leading points it saw.

What replaced it needs no levels of its own — the thresholded fraction, how
`p_e` is computed everywhere else in this tree. The cost is that fractions are
biased low by the T1 continuum the threshold splits, which is why the NEP
summary is labelled indicative and `fit_shared_means.py` stays the authority.

### The passive-wait scan

The `w` scan is a ground-state readout that **waits instead of resetting**: no
drives, and the f0g1 reset replaced by the full `final_delay` T1 wait. It
exists to separate the two things hiding in the `p_g` floor -- genuine thermal
population, versus population the f0g1 reset is failing to clear (or adding).
Comparing `p_w` with `p_g` inside the same point makes that apples to apples;
comparing across runs with `reset_mode` switched globally confounds it with
drift.

Mechanically it is the sixth field of the `SCANS` table. `acquire()` sets
`cfg.expt.reset_mode` per scan and recomputes `final_delay`, because
`_get_final_delay()` returns 1 us under any reset and the full passive wait
without one; the requested mode is restored afterwards so the saved yml records
it. Everything else follows from the table -- `LOG_COLUMNS` and `log_row`
derive from `fwm_qc.SCANS`, so `p_w` appears in `qc_log.csv` automatically.

**Do not expect `p_w` < `p_g`.** The T1 wait only reaches thermal equilibrium,
whereas f0g1 is an active cooling protocol that can go below it. `p_w` above
`p_g` means the reset is working, not failing. This is why `w` is deliberately
absent from `FLOOR_SCANS`: gating a diagnostic whose expected value may sit at
or above `GROUND_MAX_PE` would reject good data for a non-reason.

**Cost.** At 436.83 us and 30 k shots the scan adds 13.1 s per point against
~1 us per shot on the reset path, so on script 26's 18 x 8 grid it is about
31 minutes. Note that `w` is drive-independent -- it is the same measurement at
every point of the sweep -- so a subset, or `ground_monitor.py` run in both
modes, gets the same comparison far more cheaply. In-sweep buys interleaving
against drift, nothing else.

## The peak gate is a gross-excursion catch

`PEAK_TOL` is 1.25 sigma, not the 0.75 it started at, because the peak estimate
is quantized and the gate was firing on that quantization rather than on drift.

`ground_peak_and_width` reports the argmax bin centre, and `bin_grid` spans 24
sigma in `QC_BINS` = 120 bins, so the answer can only land on a 0.2 sigma
lattice (the odd multiples of 0.1 sigma). Worse, the argmax itself is noisy: at
30 k shots the peak bin holds ~2394 counts whose Poisson noise is 2.0%, exactly
the 2.0% gap to the bin one step over, so which bin wins is near a coin flip.
The reported peak therefore jitters by +/-0.11 sigma on a perfectly stable
qubit.

That jitter is harmless against a centred reference and fatal against a stale
one. Run 34 #625 had q1's `g_mean` about 0.65 sigma stale, putting the mode at
-0.7 sigma with the bar at 0.75 -- one bin of jitter away. Simulating a
stationary blob at -0.65 sigma reproduces the observed peak histogram
(-0.9/-0.7/-0.5/-0.3 = 4/33/16/1 against 4.4/31.5/17.9/0.2) and the 7%
per-scan reject rate, which compounds over four scans into the 28% of attempts
that were actually retaken. None of it was instability.

The response is to loosen the bar rather than refine the estimator, because the
gate is nearly redundant. What corrupts `p_e` is the blob moving *toward* the
threshold, and `p_g` responds to that far faster than the peak does: with q1's
4.84 sigma blob separation the threshold sits 1.445 sigma above `g_mean`, so a
drift of only **+0.2 sigma** toward it already takes Gaussian `p_g` to 0.05 and
trips `GROUND_MAX_PE`. The floor gate is roughly 4x more sensitive than the
peak gate in the one direction that matters; drift the other way leaves `p_e`
alone. At 1.25 sigma the peak gate tolerates a reference up to ~0.8 sigma stale
and still catches a gross excursion, which is all it is now for.

Two things not done, deliberately. Parabolic refinement of the argmax only
improves the jitter from 0.114 to 0.086 sigma, not worth the code. A windowed
centroid is 23x more precise (0.005 sigma) but biased by the +/-2.5 sigma
window's asymmetric truncation, compressing a real 1.5 sigma drift to 1.17 --
it trades sensitivity where the gate is supposed to have it.

Note the interaction with the floor gate when refreshing the reference:
`12_single_shot.py` re-optimizes `threshold` along with `g_mean`, and with
fidelity at 0.78 the overlap contribution to `p_g` is already a large part of
the observed 0.037, so check that `p_g` still clears 0.05 with margin
afterwards.

## Why the gates are absolute

**Every gate is now a fixed bar with no history.** The `profile` and `drift`
gates compared each scan against a trailing median of *accepted* points, and
run 34 `#621` showed that design failing in both directions at once:

- **It missed the thing it was for.** Points 82→84 ramped `p_g` 0.039 → 0.092 →
  0.059 → **0.160** and all three were accepted, because each step was inside
  `DRIFT_TOL = 0.04` of the running median. A differential gate cannot catch a
  drift slower than its own tolerance per step, and it *banks* every bad point
  into the reference on the way.
- **Then it dead-ended.** By point 85 the trailing median of `p` was **0.1180**,
  poisoned by those banked points. From attempt 2 the data was healthy again
  (`p_p` 0.045–0.051, the pre-excursion level) but was compared against 0.118
  and rejected — ten attempts on point 85, then straight into 86, all on good
  data. The ground-state check had already recovered the qubit (33 checks,
  164 s, 0.164 → 0.047); the gates blocked anyway.
- **And `profile` was not a second opinion.** It fired on exactly the same
  attempts (tv 0.055–0.072) because its median *histogram* came from the same
  poisoned window — the same stale reference expressed as a shape distance.

The absolute bars fix both ends. Replayed against `#621`: 82/83/84 and
85-attempt-1 are all caught (`floor:g`, and `pump:p` on two of them), while
85-attempts-2-10 and 86 all pass. `#610` stays at 0 rejections; `#611` flags 6
of 126, all `floor:g` at 0.050–0.053 against that run's median of 0.0234.

**Where the numbers come from.** `GROUND_MAX_PE = 0.05` is reused from the
ground-state check, so the in-sequence floor and the out-of-sequence health
probe share one definition of healthy; pooled over run 34, `p_g` has median
0.0255 and p95 0.0453, so the bar rejects 3.6% of previously accepted points.
`PUMP_MAX_PE = 0.10` is twice that — `p_p` and `p_d` sit legitimately above the
floor — and against a pooled `p_p` median of 0.031 and p99 of 0.069 it rejects
0.4%.

**What this removes.** `QCMonitor` is stateless: no `_history`, `_GROUPING`,
`_recent` or `accept()`, and no `DRIFT_*`, `PROFILE_TOL` or `MIN_HISTORY`. That
also closes the `MIN_HISTORY = 4` blind spot — the first four points of every
group used to be ungated, which is how `#528` banked two bad points at the
start of a run — and makes script 40's per-length monitor unnecessary, so it is
built once.

**Why `f` has no gate (a removed one, kept here as a record).** `f` reads the
same conversion on the g/f axis, so `p_f/p_e` looks like it should be a
constant of the run — both scans are the *same* state preparation, and `f`'s
population is derived from `e`'s by the `pi_ef`, so past the background-limited
regime the ratio asymptotes to `eta_ef * A_f / A_e`, a property of that pulse
and the readout rather than of the conversion. (It sits slightly *above* 1,
~1.04–1.09, because `analyze_point` thresholds every scan against the same
stored g/e `threshold` and the |f> blob clears it more completely. That is what
`check_f` is for.)

It was gated on a trailing median and the premise does not survive the data.
Two separate structures, both real:

- **It depends on conversion level.** On run 33 `#671`, binned by `p_e`
  quartile: 1.337 → 1.162 → 1.088 → 1.035, with the spread shrinking 0.83 →
  0.12. The small-signal limit is the ratio of the two scans' *backgrounds*,
  not of their signals, so a trailing median mixes points with genuinely
  different expected ratios. `RATIO_MIN_PE` did not cut deep enough to remove
  it.
- **It depends on pump power beyond that.** At *matched* `p_e` on `#671`:
  1.30–1.37 for pump gain ≥ 0.8 against 1.04–1.08 below, while the readout was
  provably stable (`width_g` identical at every point, `peak_g` inside one bin,
  `fid_ge` healthy) and the measured backgrounds were flat or lower. The
  absolute excess `p_f − p_e` jumps +0.012 → +0.066.

So the gate fired on physics, 7 of its 8 firings upward. The leading
explanation for the pump-power part is **pump-induced relaxation**: if T1 falls
at high pump gain, more converted population decays into the sub-threshold
continuum, the `e` readout loses it and the `f` readout — starting a level
higher — retains it. Untested; it predicts a dependence on `readout_length`.

If that is right it matters beyond QC: **`eta_e` under-reports conversion at
high pump gain**, which is where the best NEP comes from, and `eta_f` is the
more faithful one there. Worth settling with T1 vs pump gain before quoting
high-pump efficiencies.

### Ground-state recovery

**A failed point holds the sweep until |g⟩ is healthy again.** Retaking
immediately was the wrong response: when a gate fires, the qubit or readout is
usually in a bad state, so the retake lands in the same bad state and burns
`MAX_ATTEMPTS` acquisitions before skipping the point (run 34 `#520`'s log is
all of this). Now every failure calls `fwm_qc.wait_for_ground`, which blocks on
a ground-only readout until the fraction past `device.readout.threshold` is
under `GROUND_MAX_PE = 0.05` — a level this device should always reach —
`GROUND_CONSECUTIVE = 3` times in a row.

**The sweep will not start on an unhealthy qubit either.** Before the first
point, each script runs the same check with `consecutive=1` and blocks until it
passes once. Run 34 `#528` is why: it began with the no-drives background
already at `p_n` ≈ 0.10–0.12 and banked **two points as `ok`** before anything
fired — at the time the background gates were differential and needed four
accepted points before engaging, and `peak`/`width` do not see a raised
background at all. The third point failed only through `ordering` — `p_e` 0.084
below
`p_n` 0.119, a conversion signal beneath its own no-drives floor. One pass is
enough to start; a mid-sweep failure needs the full streak, since starting is a
one-off while a failure means something just moved. The pre-sweep row in
`ground_waits.csv` is marked `index = -1`, `attempt = 0`.

**The streak is the point, and it has to span time.** One pass can be a
fluctuation, or the low half of something oscillating: a single-pass gate
resumes on the first good look even when the next one would have failed. The
streak resets to zero on any failure, and the settle runs between *every*
check, pass or fail — three back-to-back acquisitions would take 0.3 s and
measure nothing a single longer acquisition would not. So a clean recovery
costs `(GROUND_CONSECUTIVE - 1) * GROUND_SETTLE` = 10 s, which is also the
floor on every failure even when the ground state was never the problem.

The gates keep their job. They detect; the ground state decides when it is safe
to carry on. Nothing about `QCMonitor` or the six gates changed.

**Unbounded by design.** There is no give-up path: the run does not continue
until |g⟩ passes, warning every `GROUND_WARN_EVERY = 10` *failed* checks with the
current fraction and the elapsed time so an unattended stall is diagnosable
from the log. A readout that has genuinely drifted will not recover on its own,
so a long stall means a human is needed — which is the intent, not a bug.

**`MAX_ATTEMPTS` still caps retakes of the point.** A point can keep failing a
gate with a perfectly healthy ground state — `ordering` at `gain_b = 0`, say —
and that must not become an infinite loop. The wait sits *before* the
`if last:` branch, so a point being skipped waits too: "the run doesn't
continue until the ground state passes" covers the skip path.

**`GROUND_SHOTS = 5000`, `GROUND_SETTLE = 5.0 s`.** 5000 shots give σ = 0.0031
at p = 0.05, so the bar is 8.3σ off a healthy 0.03 floor and cannot be crossed
by counting noise; the check costs ~0.1 s of acquisition, so the settle
dominates and gives whatever drifted a chance to recover instead of hammering
the readout.

Two traps the implementation has to avoid, both verified:

- **Inherit the sweep's reset params, don't re-derive them.**
  `GROUND_RESET_KEYS` filters the sweep's own `params` into the check. The
  sweeps pass `f0g1_reset_gap: 0.0` against the config's much longer default,
  so a check built from config defaults would clear the resonator *more*
  completely than the sweep does and read a **better** ground state than the
  sweep ever sees — masking the problem it exists to catch.
- **Rotate only when the reset is passive.** `HistogramProgram` puts
  `device.readout.phase` on the pulse iff `reset_mode != "passive"`
  ([single_shot.py:100-104](../../experiments/single_qubit/single_shot.py#L100-L104)).
  `ground_monitor.py` rotates in software because it runs passive; 26/37/40 run
  `active`/`f0g1`, where rotating again would double-rotate. With the hardware
  phase applied the fraction is directly comparable to the sweep's own `p_n`.

**Why `HistogramExperiment` and not `SMPDHistogramExperiment`.** The latter's
`_scans()` hard-requires `g` and `e`, and its `n` scan still pays the full
buffer window per shot. `HistogramExperiment(check_e=False)` acquires `Ig`/`Qg`
alone — the same thing `ground_monitor.py` and `12c` already use. Its
`analyze()` must not be called: it reads `data["Ie"]` unconditionally and
raises, so `ground_excited_fraction` does its own thresholding.

**Nothing is saved.** The check is built once and re-acquired, and
`save_data`/`save_config` are never called, so no file appears. It reserves a
`single_shot_qubit<qi>_00000.h5` name at construction — a different prefix from
the sweep's `fwm_histogram_*`, so it cannot disturb the sweep's indices.

**Output**: `ground_waits.csv` in `metadata_and_images/`, columns `index,
attempt, checks, failures, elapsed_s, p_e_first, p_e_final` (script 37 inserts
`frequency_b`). `wait_for_ground` returns those fields as a dict, so a row is
`{"index": ..., "attempt": ..., **wait}`, and it is written only when
`failures` is non-zero — i.e. when the sweep was actually held up rather than
merely verified. It is a separate file rather than new `qc_log.csv`
columns because an attempt's row is written *before* the wait happens, and
script 37 already wraps `log_row` to inject its own extra column.

**A point that never passes QC is skipped, not kept.** It gets a `status =
failed_skipped` row in `qc_log.csv` for its final attempt, its histograms and
metrics go into `qc_rejects.h5` like every other rejected attempt, but nothing
is saved to `data/` or `metadata_and_images/` and it never enters the QC
history or the accepted list. Letting it into the history would slide the
reference onto the excursion and quietly pass everything after it — the exact
failure the script exists to prevent — and saving its h5 would leave an
unusable point sitting in the data folder indistinguishable from a real one.
The cost is that a *sustained* excursion spends three attempts on every point
it covers; the end-of-run tally says how many were skipped.

**Save path.** `finish_experiment` is not used, because `QickExperiment.run`
saves before anything can inspect the data, so it would have written every
reject's 150 k shots. The script calls
`run(analyze=True, display=False, save=False)` — still the class's own
acquire/analyze — then `save_data()` + `save_config()` only when a point
passes. `save_config()` has to be explicit because the per-point yml is
normally written inside `display()`, which this script never calls. Since
`self.fname` is chosen in `Experiment.__init__` but the h5 is created lazily, a
skipped attempt never claims an index — but a whole skipped *point* means no
attempt of it ever wrote, so **the saved files are `_00000 … _00N` contiguous
in save order, not necessarily one per grid point** — a run with any
`failed_skipped` points has fewer h5 files than grid points, and the mapping
from filename index to `(gain_p, gain_b)` has to come from `qc_log.csv`, not
from position in the grid.

**Plots are opt-in, `--plots none|rejects|all`, default `none`** — a full grid
is 130 figures and 130 windows to juggle, and any point replots from its h5
with `plot_log_histogram.py`. The default draws only the end-of-run
`qc_nep_summary.png`, saved with `fig.savefig` rather than `save_fig` (which
calls `tight_layout()`, conflicting with the figure's constrained layout, and a
blocking `plt.show()` that only `common.run_experiment` patches).

**`--plot-every N`, default 10**, is the middle ground: the five scan
histograms for every Nth *kept* point, via `fwm_qc.plot_point`, saved as
`qc_point_<index>.png`. A periodic look at the distributions during a long run
without a figure per point. `0` disables it, and it is skipped under
`--plots all`, which already draws every point. Because it comes from the
stored histograms it is the same figure a rejected attempt gets and the same
one `fwm_qc --replot-rejects` makes.

The two higher `--plots` modes draw *different* figures, on purpose:

- `rejects` → `fwm_qc.plot_point()`, the five scan histograms linear and log,
  captioned with the gates that failed. Drawn from the stored histograms, not
  the shots, because a rejected attempt's shots are deliberately not kept —
  which is also why `fwm_qc.py --replot-rejects <qc_rejects.h5>` produces the
  identical figure, same function and same filename, from a run taken with
  `--plots none`. Choosing `none` therefore costs no information, only the
  convenience of having the figure immediately.
- `all` → additionally `SMPDHistogramExperiment.display()` for every *kept*
  point, i.e. exactly the figure script 23 writes, IQ panels included. That one
  needs the raw shots, so it is only available for a point that passed.

`display()` is wrapped in `common.nonblocking_plots()`, a context manager
extracted from `run_experiment` so there is one copy of the `plt.show` patch
rather than two. Windows are closed by number at the top of each point (never
via `plt.figure(num)`, which re-raises and makes them flash), so one point is
on screen at a time; under `--no-show` they are closed as soon as the PNG is
written. Note `--plots` and `--no-show` are independent: `--plots` decides
whether figures are produced and saved, `--no-show` only whether the run blocks
on them at the end.

**Outputs** in `metadata_and_images/#<n>_FWM_efficiency_vs_power_checked_.../`:
`qc_log.csv` (one row per *attempt*, so the reject rate is recoverable),
`qc_rejects.h5` (one group per rejected attempt: the five histograms and the
metrics, ~600 floats, not the shots), `qc_summary.csv`, `qc_nep_summary.png`,
and — with `--plots` above `none` — `qc_reject_<index>_attempt<n>.png` and the
per-point PNGs.

**Recalibrating the tolerances.** `fwm_qc.py` runs standalone and read-only
over a saved folder, printing the distribution of every gate statistic and
which points it would have retaken. Do this after any change to the readout —
the constants are set from run 33 `#138` and are specific to that readout and
30 k shots.

**Pass `--config` unless the data matches `fwm_qc.CONFIG`.** The reference frame
(`g_mean`, `sigma`, `threshold`) comes from the config, and `--data-dir` says
nothing about which device produced it, so the two can silently disagree.
Checking Waymar's `#138` against the yuvi default reports 642 `width` failures
out of 650 — width ratios of 0.5–0.7, which read as a catastrophically narrow
readout and are only a `sigma` from the wrong fridge. With the matching config
it is 18.

```
python fwm_qc.py --data-dir Data/run33_Waymar/data/#138_FWM_efficiency_vs_power_q1_091826
```

### Buffer-length sweep (script 40)

`40_fwm_efficiency_vs_buffer_length.py` is script 26 with a buffer-length
axis on the outside. 26 stays the fixed-length version; 40 is for the length
trade-off. It currently runs **one pump power** (`PUMP_GAIN`, wrapped as
`PUMP_GAINS = [PUMP_GAIN]`); the loops and the per-pump reduction are kept, so
a pump scan is just a longer list. Everything in the script-26 section above applies -- same gates,
same `--plots`, same `PHOTON_CAL`, same outputs -- plus the following.

**The buffer length is the outermost axis.** `length_p` tracks `length_b` — the
pump has to cover the whole buffer window or the tail of the buffer pulse sees
no pump and cannot convert — and that moves the reference scans with it: `p`
waits out `length_p`, `g` waits out `max(length_p, delay_b + length_b)`
(`fwm_shots._body`). That used to need a fresh `QCMonitor` per length so their
trailing histories were not pooled across lengths; with the gates now absolute
there is no history, so the monitor is built once. Outermost rather than
innermost so the pump × gain grid inside one length runs as it would alone.

Log-spaced lengths, because both effects are multiplicative: arrivals go as
`length_b`, and `eta_qubit = (T1/t)(1 - exp(-t/T1))` falls with it. Measured
efficiency should track `eta_qubit` if decay before readout is the limit, which
is the comparison the sweep is for — hence the sloping ceiling on the figure's
efficiency panel. Short windows are penalised by dead time (duty
`L / (L + dead)`), long ones by decay, so `duty * eta_qubit` peaks near
`L* = sqrt(2 * dead * T1_pump)`, and the grid is `N_LENGTHS` points spanning
`LENGTH_SPAN` geometrically centred there.

- **`device.fwm.T1_pump` is required.** It is the T1 with the pump on, which is
  what the qubit sees during the window; `39_t1_pump.py` writes it (with the
  pump gain/frequency it was taken at). The script reads it from the config
  file *before* `prepare_experiment`, and stops if it is missing, NaN or ≤ 0 —
  no run folder, no hardware.
- **`dead` is measured from a probe**, an `SMPDHistogramExperiment` built with
  the sweep's own `params` and never run: period minus `length_b`, i.e.
  `readout_length + effective_final_delay` (+ `delay_b`). So it follows
  `RESET_MODE`. On bfg34 qi=1 that is 41.0 us active (7.0 readout + 34.0 tail
  with `reset: 5`), 10.6 us f0g1, 402 us passive.
- **The grid is `LENGTH_START`, `LENGTH_END`, `LENGTH_CENTER`, `N_LENGTHS`.**
  Two log-spaced runs, start → centre and centre → end, so the centre is
  always a point; the `N_LENGTHS` points split between the sides by their
  share of the log range, so the spacing is about even. `LENGTH_CENTER = None`
  uses `sqrt(2 dead T1_pump)` (printed either way for comparison); a centre
  outside (start, end) stops the script. Lengths are rounded to whole us and
  deduplicated, so a narrow range can come out with fewer points. Keep
  `LENGTH_END` under the hard limit: a pulse longer than 2**16 fabric cycles
  fails at compile time (~152 us at 430 MHz).

**`reduce_to_nep` is per group, not per run.** `t_d` and `cycle_us` were
global, which a length sweep breaks since both feed `n_equiv` and the NEP. Each
row now carries its own `length_b`, `cycle_us`, `eta_qubit` and `eta_ceiling`;
points may carry `cycle_us` and the scalar argument is the fallback. `info`
still holds the whole-run medians for single-length callers (script 37).

Script 40 calls it once per pump gain with `by="length_b"` and `t1=T1_pump`:
the efficiency at a length is the initial slope over *that length's* own
buffer gains. Pooling lengths at one gain instead would mix different
`eta_qubit` into one slope.

**The analysis is efficiency vs length, not NEP.** Per pump gain,
`fwm_qc.plot_efficiency_vs_length` draws `qc_efficiency_vs_length_p<gain>.png`
(and `fit_shared_means` its `summary/fwm_summary.png`): efficiency (e and f
readouts) and efficiency × duty against length, each with the ceiling
`kappa_c/kappa * ringup_efficiency(L, 1, T1_pump, 1/(2 pi kappa))` -- the same
ring-up × survival average as the fit, at the impedance-matched ring-up: the
ceiling is the Gamma = kappa limit, where the pump doubles the linewidth, so
tau is half the pump-off `1/(pi kappa)`. In length mode the per-length
ceilings in the CSVs, `summary.txt` and "% of ceiling" use it too. Dark count,
`n_equiv` and NEP are still in the CSVs, just not plotted.

**One fit of the window dynamics, to every point** (`fit_window_dynamics`, a
copy in `fwm_qc` -- `reduce_to_nep(dynamics=True)` -- and in
`fit_shared_means`). The excited population over the window obeys

    dP/dt = r(t) (1 - P) - P / T,    r(t) = A (N / L) (1 - e^(-t/tau))^2

with N the photons arriving in the window (they arrive at the steady-state
flux from t = 0, which is what the photon axis counts), the buffer field
ringing up as `1 - e^(-t/tau)` so conversion goes as its square, and `1 - P`
blocking a second excitation. Measured `c + (1 - c) P(L)`. A, T, tau are
shared across lengths; the background c is per length, since the pump-only
floor moves between them. The points drawn at each length are the
linear-regime efficiency there, `A_L * <ring-up x survival>`, from refitting
only A at that length with T, tau and c held, so points and curve are the same
model. `T` near `T1_pump` means decay before readout is the length limit;
`tau` should sit at or below the pump-off ring-up `1/(pi kappa)` (printed),
since the pump adds a loss channel.

**Pump-gain and frequency runs (23/26/37) use the same model** in
`fit_shared_means`, since 2026-10-17. Before that they took a straight line
through the 5 lowest-arrival points. At a 41 µs window those points reach
N ≈ 0.56, where the response already bends over, so the line read ~20% low.
On #632 at pump 0.6 the line gave 0.53; the model gives 0.665, matching #630
at the same length and settings.

How a single-length run uses the model:
- One window length cannot separate T and tau from A, so `efficiency_vs_pump`
  fits A and c at each sweep value with them held: T = `device.fwm.T1_pump`
  (else `device.qubit.T1`), tau = 1/(2π κ).
- η = A·ringup_efficiency(L, 1, T, tau) moves only 2–4% across T = 50–73 µs and
  tau = 0.9–1.8 µs.
- The ceiling is `kappa_c/kappa * ringup_efficiency(L, 1, T, 1/(2π κ))` for
  every sweep, with the pump-on T. Before, single-length runs used
  (T1/T_d)(1 − e^(−T_d/T1)) with the pump-off T1.

`T1_pump` is measured at one pump gain (0.6 on run 34). For a pump-gain sweep
the ceiling is therefore only exact there. At weaker pump T1 is longer, so
#632 reads 104–111% of it at pump 0.45–0.55 (99% against the pump-off
ceiling). A per-pump ceiling needs T1 vs pump gain (script 39 at several
gains). fwm_qc's live reduction in scripts 26/37 still uses the straight
line; it is indicative only.

Why not per-length fits. Buffer gains are fixed across lengths and arrivals
go as gain² × L, so at 80 us even the lowest gains reach ~1.4 photons and
P(e) ~ 0.55. Two simpler reductions were tried on run 34 and both biased T low,
checked by simulating known T1 on `#522`'s own photon grid:

| per-length reduction, then fit eta(L) | true T1 69 us gives | data (`#520`/`#522`) |
|---|---|---|
| initial slope over the 5 lowest arrivals | -- | 27 / -- us |
| saturating `c + (1 - c)(1 - e^(-eta N))` | 50 noiseless, 33 ± 4 with noise | 31–35 us |
| this global dynamics fit | 67.6 ± 4.0 (35 -> 34.9 ± 1.3) | 59–71 us |

The saturating form lets P(e) climb to 1, but once L ~ T1 a decayed qubit is
re-excited and P(e) levels off below 1 (0.86 at 61 us), so it flattens eta at
long L -- 0.60 against a true 0.82 at 61 us -- which reads as fast decay.

Run 34 (pump 0.6, f0g1 reset), global fit: T 59–71 us in `#518`/`#519`/`#520`/
`#522` against `T1_pump` = 69 us (`#523`), tau 0.8–1.5 us (best `#522`, 1.0 ±
0.1, which starts at 2 us) against 1.30 us pump-off, efficiency × duty peaking
at 36–38 us. A is 1.08–1.25, above `kappa_c/kappa`, so the photon axis is
still suspect.

**`fit_shared_means.py` has a length mode** (auto-detected, like the
frequency sweep; `--length-b` picks the per-histogram figure). Every quantity
that was one number per run -- `t_d`, `eta_qubit`, ceiling, cycle, duty --
is per length, the ceiling uses `device.fwm.T1_pump` (falling back to
`device.qubit.T1` with a note), and `summary/fwm_summary.png` becomes the
same two panels with the A, T fit; dark count and NEP stay in the grid
figure's bottom row and the CSV. Pump-gain and frequency runs are unchanged
(checked: identical stdout and `fwm_summary.csv` on run 34 `#232`).

`sqrt(2 dead T1_pump)` (the dotted line) is the `dead << L` approximation of
the optimum. With active reset `dead` (41 us) is comparable to `L`, and it
overshoots: on synthetic bfg34 numbers the ceiling peaks at ~58 us against the
line's 69. The exact optimum of `duty * eta_qubit` solves
`L + dead = T (exp(L/T) - 1)`, whose small-L expansion is `L^2 = 2 dead T`.

## f0-g1 unconditional reset (scripts 27+, [f0g1_reset.py](../../experiments/smpd/f0g1_reset.py))

All-microwave unconditional reset following Magnard et al., arXiv:1801.07689. A
drive at `f_ge + f_ef - f_waste` (**225.54 MHz** on run 33 Waymar) converts
|f,0> -> |g,1>; the waste resonator then emits the photon at `kappa`. Unlike the
paper the *readout* resonator doubles as the reset resonator, which its closing
paragraph argues is preferable.

`kappa[1] = 1.1843` MHz, so the best achievable rate is `Gamma = kappa/3 =
395 kHz`, reached once `g~ >~ 0.5 MHz` with `Omega_ef ~ 0.29 MHz`
(`check_f0g1_config.py` tabulates this).

| Script | Class | Sweep | Writes |
|---|---|---|---|
| `27_f0g1_spectroscopy.py` | `F0G1Spec` | f0-g1 drive frequency | `device.f0g1.frequency` |
| `28_f0g1_stark_shift.py` | `F0G1SpecPower` | f0-g1 freq -> f0-g1 gain | `device.f0g1.stark_fit` |
| `29_ef_rabi_flat_top.py` | `RabiExperiment` (existing) | e-f flat-top length | nothing -- see below |
| `30_ef_rabi_vs_power.py` | `EFRabiPower` | e-f length -> e-f gain | `device.f0g1.omega_ef_fit` |
| `31_ef_spec_under_f0g1.py` | `EFSpecF0G1` | e-f probe frequency | nothing (single-power check) |
| `32_ef_stark_shift_vs_f0g1_power.py` | `EFSpecF0G1Power` | e-f freq -> f0-g1 gain | `device.f0g1.ef_stark_fit` |
| `33_f0g1_rabi.py` | `F0G1Rabi` | f0-g1 drive duration | nothing (single-power check) |
| `34_f0g1_rabi_vs_power.py` | `F0G1RabiPower` | duration -> f0-g1 gain | `device.f0g1.g_tilde_fit`, `kappa` |

| `35_reset_time.py` | `ResetTimeScan` | reset time, x3 preparations | nothing (the measurement) |
| `36_reset_rate_vs_ef_gain.py` | `ResetTimeScan` x N | reset time, per e-f gain | nothing (measures the Omega_ef scale) |
| `37_reset_rate_map.py` | `ResetTimeScan` x N | reset time, per (g~, Omega_ef) point | nothing (the Fig. 3a map) |

All four calibration steps plus the reset measurement are written. 34's
`analyze()` closes the loop: it prints the `(f0g1 gain, e-f gain)` pair that
lands on the Fig. 3a red curve, which 35 then derives for itself.
`check_f0g1_config.py` recomputes the same thing from the config alone.

Measured so far on run 33 Waymar:

- 27: the transition is at **225.142 MHz**, 0.4 MHz off the design value.
- 28: `stark_fit = [-0.064, 225.142]`, i.e. **no measurable shift** -- there is
  attenuation on the f0-g1 line limiting the dynamic range. **Rerun 28 once
  that comes off**, because 31/32 and step 4 all track the drive frequency
  through this fit and will otherwise hold it essentially fixed.
- 30: `omega_ef_fit = [10.8768, 0.0852]`, so `Omega_ef/2pi = 10.88*gain + 0.085`
  MHz. The reset plateau wants `Omega_ef ~ 0.29 MHz`, i.e. e-f gain ~0.019.

### The drive line

DAC14, combined onto the qubit line, configured as `hw.soc.dacs.f0g1` (same
shape as `dacs.pump`). Two settings are not the house default and both matter:

- **`mixer_freq: 0`.** The v2 programs use `ABSOLUTE_FREQS`, so the DDS is
  handed `freq - mixer_freq`, and an int4 DDS reaches +/- `f_dds/2` (~860 MHz)
  on its own. 225 MHz therefore needs no NCO at all. The usual
  `mixer_freq_for` convention (`freq - 400`) exists only to reach GHz carriers
  while keeping the DDS away from DC, where NCO leakage would sit on the
  carrier; here it would *create* that problem, putting leakage at 175 MHz,
  50 MHz from the tone.
- **`filter_type: bypass`.** The RF board's ADMV8818 spans 2-18 GHz. Anything
  else and the tone never reaches the sample.

`nyquist: 1` (fs/2 = 3440 MHz). `rfboard.activate_qubit_rf` grew a `dac_f0g1`
block and `"dac_f0g1"` in its default `channels`, guarded on the section
existing -- the Yuvi config has no f0g1 line, so it is skipped there and
`F0G1Spec` raises a message naming the missing section rather than failing
inside the Program.

`declare_f0g1_gen()` does the declaration, because none of the three existing
idioms fits: `dacs.pump`/`self.pump_ch` is already the 5.2 GHz FWM pump, and
`QubitStarkSpecProgram`'s version omits `nqz` and derives the mixer as
`freq - 400`, i.e. negative here.

### Pulse shape: `flat_top` by default

`const` in `make_pulse` is a hard square pulse -- no envelope. The gaussian-edged
flat pulse is the separate `flat_top` type, where `length` is the **flat**
portion and `ramp_sigma`/`ramp_sigma_inc` (0.02 us / 3) are the edges. The f0-g1
and e-f drives use `device.f0g1.pulse_type` / `pulse_type_ef`, both `flat_top`
on bfg34 (see "FWM experiments" for the keys). With `const` the ramp terms in
the nesting delay, the e-f pi length (`t_pi` minus ramps) and the reset tail
drop to zero. A `flat_top` length sweep runs as a hardware
`QickSweep1D` and `param="total_length"` reports flat + ramp, a constant offset
that the Rabi fits absorb -- unlike `gauss` + length, which is forced into slow
software-loop mode.

### `readout_gap`

Defaults to `3 / device.readout.kappa[qi]` (2.5 us). Not cosmetic: the drive's
whole purpose is to put a photon in the *readout* resonator, so measuring before
it has leaked away measures the photon rather than the qubit.

### e-f spectroscopy under the f0-g1 drive (scripts 31/32)

`F0G1Program` grew two switches rather than getting a second program:
`ef_probe_on` plays a flat-top e-f probe on the qubit channel, and `shelve_ge`
adds the trailing pi_ge. Steps 1/4 leave both off.

**The nesting.** The f0-g1 tone starts first and the probe is pushed out by
`f0g1_total - ef_total` so both *end* together (the `StarkSpec` idiom). This
uses `self.delay`, which *increments* the reference time -- `delay_auto` *sets*
it and would discard the f0-g1 pulse's own start. Both totals include the
flat-top ramps. `EFSpecF0G1.__init__` raises if the probe is longer than the
drive, since that would emit a negative delay and silently reorder the
sequence.

**Why step 3 needs the shelve and steps 1/4 do not.** Its contrast is e-vs-f,
which this readout cannot resolve. Tracing both outcomes with the f0-g1 drive
on resonance:

```
off resonance  |e> --(no ef rotation)--> |e> --pi_ge--> |g>        reads g
on resonance   |e> --ef pi--> |f> --f0g1--> |g,1> -> |g,0>
                                     --pi_ge--> |e>               reads not-g
```

so the paths separate even though the on-resonance population arrives in |g>
*before* the shelving pulse rather than sitting in |f>. The shelve is placed
after `readout_gap` so it is not detuned by the emitted photon's own Stark
shift.

**The e-f pi length is derived, not stored.** `t_pi = 1/(2*Omega_ef)` from
`device.f0g1.omega_ef_fit` at whatever `gain_ef` is in use, on the same
total-length axis the rate was fit against -- so the flat part is `t_pi` minus
the ramps. There is deliberately no `pulses.pi_ef_flat_top` config section to
keep in sync, and `device.qubit.pulses.pi_ef` stays the gaussian prep pulse.
Raises if `t_pi` is shorter than the ramps (i.e. `gain_ef` too high).

**`gain_ef` sets the linewidth, not the answer.** The measurement reads the
dip's *position*, so an imperfect pi costs contrast and nothing else. Weaker
probe -> narrower line (Fourier limit ~`2*Omega_ef`) -> better centre, until
there is no dip left to fit.

**Both drive frequencies track their own fits.** The f0-g1 tone is held on its
Stark-shifted resonance via `f0g1_frequency_for_gain` (inverting step 1's
`stark_fit`), co-varied with the gain as a second ysweep entry in the 2D
version -- the same trick scripts 23/24 use to walk the FWM band. Driving it
off resonance would under-report the e-f shift. Both that helper and
`omega_ef_for_gain` raise naming the script to run (28, 30) rather than
proceeding with a NaN.

**The e-f line is a doublet, and the fit reads its centre.** The f0-g1 drive
coherently couples |f,0> to |g,1>, so |f,0> hybridizes into two dressed states
at +/- g~ and the e-f probe sees both. The pair is split by **2*g~** and the
*centre* is the bare (Stark-shifted) e-f frequency -- which is what the reset
needs, since Eq. 1 assumes each drive resonant with its bare transition and
produces the splitting as an *output*. A single Lorentzian lands on a peak or
between them; driving a peak is `delta_ef = +/- g~` and costs most of Gamma.
Measured on synthetic data at gain 1.0 (`g~ = 0.82`): the doublet centre is
2 kHz out, a single Lorentzian **810 kHz** out.

`fit_ef_doublet` therefore fits both `fitlor` and `two_lorentzians` per row and
keeps the doublet only when the pair is believable. Six free parameters against
four means the doublet can always fit at least as well, usually by parking a
small satellite on noise, so a better R2 is not evidence. Four gates:

| gate | why |
|---|---|
| amplitudes same sign | both dressed states carry half the |f,0> character |
| within 4x of each other | a genuine pair is nominally 1:1 |
| separation > its own linewidth | below `2*g~ ~ kappa/2` the states merge and the fit is degenerate |
| F-test at 95% on the 2 extra parameters | the models are nested, so this is the textbook comparison |

**The F-test replaced a "R2 must improve 20%" rule**, which ignored how many
points a row had. On `#364` (100 points) that rule rejected the gain-0.8 row at
`F = 8.9` against `F_crit = 3.1`. Adds `ef_stark_center` / `_splitting` /
`_gamma` / `_r2` / `_npeaks` per row, and fits `splitting = m*gain + b` to give
`g_tilde_split_m/_b/_resid` -- an **independent measurement of g~**, a frequency
rather than a rate. It is printed against `device.f0g1.g_tilde_fit` from step 4
and deliberately **not written to the config**: the two should be reconciled by
hand, not silently overwritten.

That splitting fit is gated on rows surviving the sigma-clipping, not on rows
entering it. With 3 resolved rows, 4 passes at 3 sigma can discard all of them
while `_fit_band_model` still returns its last iteration's coefficients --
on `#364` that read `g~ = 1.088*gain - 0.404`, a negative intercept, which is
unphysical since g~ must vanish with the drive.

**Why the paper fits a single Gaussian and never mentions the doublet.** Their
Fig. 2e probes with a **140 ns square e-f pi-pulse**, i.e. `Omega_ef/2pi =
3.57 MHz` and a probe linewidth `~2*Omega_ef = 7.1 MHz`. Added in quadrature
with the dressed linewidth `kappa/2 = 4.5 MHz` that is an 8.4 MHz line against
a `2*g~ = 9.6 MHz` splitting at their `Vf0g1 = 444 mV` -- **ratio 1.14, one
broad merged dip**. Ours is a 26 us probe at `gain_ef = 0.001`, a 188x narrower
line, giving `1.65 / 0.66 = 2.49` and a resolved pair.

It is not primarily a kappa difference (their `2g~/(kappa/2) = 2.13` against our
2.49 -- comparable). It is the probe.

Their choice is **unbiased by design, not an oversight**: a probe broad enough
to cover both dressed states measures the intensity-weighted centroid, which by
the sum rule `sum_i |<f,0|i>|^2 E_i = <f,0|H|f,0>` *is* the bare Stark-shifted
e-f frequency -- exactly the quantity Eq. 1 wants, since they state they
"calibrate all drive frequencies to enforce `delta_ef = delta_f0g1 = 0`".
Simulated in their regime, a single Gaussian on the merged line returns the
centre to 0.000 MHz at `delta_f0g1 = 0`, and is only 0.59 MHz out even with a
3 MHz f0-g1 mis-set.

So there are two valid strategies, and the trade-off is worth knowing:

| | probe | centre | g~ | risk |
|---|---|---|---|---|
| paper | strong, short (pi in 140 ns) | centroid, robust | from the Rabi only | none for the centre |
| ours | weak, long (26 us) | needs the doublet fit | **also from the splitting** | a single-peak fit lands on a lobe, `-g~` |

A deliberately *broad* e-f probe is therefore a legitimate fallback if the
doublet fit is ever in doubt -- it trades the independent g~ measurement for a
centre that needs no model selection. What is *not* safe is the middle ground
we were in: a narrow probe that resolves the pair, fitted with one Lorentzian.

### Two sampling guards on scripts 31/32

Both print at construction from `device.f0g1`, and both were calibrated against
a scan that failed:

- **span vs splitting.** `2*g~` at `gain_f0g1` must fit inside the window, or
  the pair is clipped and the centre is pulled toward one peak. Silent at
  script 31's shipped `gain_f0g1 = 0.1` (0.22 MHz splitting in a 1 MHz span);
  fires the moment the gain is raised to look at the pair directly, where the
  splitting is 1.65 MHz.
- **step vs linewidth.** Resolving the pair is limited by the *linewidth*, not
  the splitting -- two peaks 1.6 MHz apart are still one broad line if neither
  is sampled. The probe's Fourier limit `~2*Omega_ef` is the narrowest the line
  can be and is known before the run, so it is the yardstick; the guard wants
  >= 3 points across it. `#364` ran 101 kHz steps against a 130 kHz linewidth,
  i.e. **1.3 points per linewidth**, and 7 of its 10 rows fitted as single
  lines. Script 32 then shipped `span = 4, expts = 300` (13 kHz); it now ships
  `span = 8, expts = 150` (see "Runner notes, scripts 27–37").

**`#364` predates those settings and needs re-running** before the splitting
can measure g~; it is the scan the doublet was discovered in, not one the fit
can work with. Its splitting grows 6.8x with f0-g1 gain (0.21 -> 1.43 MHz),
which is what rules out photon-number splitting as the cause -- that would sit
at a constant `2|chi| = 0.20 MHz`.

**Sign cross-check.** `analyze()` warns if `ef_stark_fit[0]` and
`stark_fit[0]` have opposite signs: one drive pushes both transitions the same
way, so a sign flip means the two runs saw different drive amplitudes (most
likely a changed attenuation), not physics. Magnitudes need not match.

### No shelving pulse in scripts 27/28

The observable is "did the population return to |g>". `chi >> kappa` puts the
|e> and |f> blobs ~0.5 sigma apart, so both read as not-g, which is all that is
needed. Step 3 will need a trailing `pi_ge` (its contrast is e-vs-f); steps 1
and 4 do not. `prep` is validated against `g`/`e`/`f` at construction, because
an unrecognised value would silently run the |g> sequence and look like a real
measurement with no f population in it.

### The e-f drive (scripts 29/30)

Step 2 reuses `RabiExperiment` / `Rabi2D` outright -- `checkEF` already gives
`pi_ge -> swept e-f pulse -> pi_ge -> measure`, and the trailing pi_ge is the
shelving pulse that turns an e-vs-f contrast into g-vs-f. `EFRabiPower` is
`Rabi2D` with `yval="gain"` plus the rate extraction, the linear fit and
`update()`; it raises if `yval`/`sweep` are not `gain`/`length`.

Two traps, both load-bearing:

- **Script 29 must not call `update()`.** `RabiExperiment.update()` writes a
  length sweep's result into `device.qubit.pulses.pi_ef.sigma` -- the
  *gaussian* prep pulse's width, used by every f-state experiment in the tree
  including 29's own state preparation. The flat-top drive is calibrated as a
  rate-vs-gain line by script 30 instead, and step 3 derives its pi length
  from that: `t_pi = 1/(2*Omega_ef)`. So there is no `pulses.pi_ef_flat_top`
  section and nothing to keep in sync.
- **The time axis is fixed across rows of script 30.** `max_length =
  2*num_osc*length` is computed once at construction and the y loop only
  rewrites `gain`, while `Omega_ef` is proportional to gain. Keep the gain
  range to ~4x: below `1/(window*m)` the bottom rows contain less than one
  period and `analyze` drops them, and far above it the top rows are
  undersampled.

`Omega_ef/2pi` is read straight off `fitsin`'s frequency parameter
(`fit_avgi[i][1]`), which is already in MHz because the x axis is the pulse
length in us. For a `flat_top` sweep that x axis is `total_length` = flat +
0.06 us of ramp, a constant offset the sine absorbs as a phase.

### f0-g1 Rabi and the Eq.-C2 fit (scripts 33/34)

Same sequence as 27/28 (`prep="f"`, no probe, no shelve) with the drive's flat
duration swept instead of its frequency, and the frequency held on the step-1
Stark curve. `self.param` uses `total_length`, so the x axis is flat + 0.06 us
of ramp -- a constant offset the model's `t0` absorbs.

**Why a new fit model was needed.** `fitter.fit_f0g1_rabi` /
`f0g1_rabi_func` implement the paper's Eq. C2:

```
Pf(t) = exp(-kappa*t/2) * |cosh(W*t/2) + kappa/(2W)*sinh(W*t/2)|^2
W     = sqrt(kappa^2/4 - (2*g~)^2)      # complex: real = overdamped
```

`W` flips between real and imaginary at `2*g~ = kappa/2`, and the reset's
`kappa/3` plateau sits on the **overdamped** side, where the trace is a pure
decay with no oscillation frequency for a sine to read `g~` off. Measured on
synthetic data at `g~ = 0.2 MHz, kappa = 1.18 MHz`: `fitdecaysin` reports
0.075 MHz where `2*g~ = 0.4 MHz` (off by 5x), while this model returns 0.2000.

Validated against the paper as well as in the limits: exact agreement with
`cos^2(2*pi*g~*t)` as `kappa -> 0` (3e-9), finite at the critically damped
point, and round-trips `g~` within 10% and `kappa` within 20% over
`g~ = 0.2 - 4.8 MHz` across both regimes, including the paper's own
configuration A (`kappa/2pi = 9, g~/2pi = 4.8`).

**`gamma_ef` is dropped from Eq. C2, deliberately.** It enters only through
`(kappa +/- gamma)`, and `gamma_ef = 1/T1_ef ~ 0.04/us` against
`kappa ~ 7/us` here -- a 0.5% correction, far below the scatter on `g~`. That
removes a parameter no data in this sweep constrains, and avoids needing to
pin `T1_ef`, which is not in the config.

**Seeding is a grid search, not an FFT.** For the same reason: in the
overdamped regime there is no peak, and an FFT seed lands on noise. A 60-point
log grid over `g~` with `kappa` from a decay-time estimate gets `curve_fit`
into the right basin.

**Three guards, because a bad `g_tilde_fit` would poison step 5 silently.**
`analyze()` sets `g_tilde_fit_ok` and `update()` refuses to write unless it is
true:

- a **non-positive slope** is not a measurement -- `g~` cannot fall as the
  drive gets stronger. The usual cause is `max_length` being too short to see
  the transfer, which at `2*g~ << kappa` takes about `kappa/(4*g~^2)`: ~1 us at
  `g~ = 0.2 MHz` but ~100 us at `g~ = 0.02 MHz`. Caught on synthetic data where
  an unconstrained fit returned `-0.689` MHz per unit gain and would otherwise
  have reported a *negative* e-f gain as the operating point.
- a fitted `kappa` more than 3x off the config's warns (stale config, or rows
  that are not constraining the fit).
- the usual per-row `min_r2` gate plus sigma-clipping.

`kappa` is fit **per row**, not shared across rows as the paper constrains it.
Per-row first is deliberate: the scatter is the honest error bar, and a `kappa`
that drifts with power is a signal that the frequency tracking is wrong rather
than something to average away. The lower panel of the `_g_tilde` figure is
where to look.

### The reset measurement (script 35)

Both drives on together for a time `t_r`, then readout, for three
preparations. `F0G1Program` gained `ef_align`: `"end"` nests a short probe
inside a longer drive (step 3), `"start"` runs both from `t=0` for the same
swept duration (step 5). Two `QickSweep1D` objects on the same `sweep_loop`
step the two lengths together, so the pair is on for exactly `t_r` at every
point.

**The observable is the thresholded "not g" fraction, which *is* `Pe + Pf`** --
exactly the paper's `Pexc`. Their three-state assignment matrix is not
available here (chi >> kappa puts the e and f blobs ~0.5 sigma apart) and is
not attempted; g-vs-not-g is both what the readout supports and what `Pexc`
needs.

**The `g` preparation is the floor and is mandatory.** `preps` must contain
`g` and `e`, enforced at construction. The |g> blob's tail above
`device.readout.threshold` sits at a few percent on run 33 Waymar (the
threshold is 1.6 sigma above `g_mean`), so nothing below that is measurable and
every other trace is quoted against it. Same logic as `12b_active_reset.py`
reporting the |g> floor alongside the absolute |e> population.

**The inter-shot reset is `passive`, deliberately (35, 35b, 36).** `active`
leaves `device.readout.reset_e` = 8.6% in |e>, against 1.64%
(`device.qubit.pop`) at thermal equilibrium after the passive 313 us wait. At
the 1% level this measures, that 5x dirtier start would show as the |g> trace
decaying from ~14% rather than sitting at the readout floor. `f0g1` would
measure the reset against itself. Switch to `active` only to shorten a long
survey, and compare the |g> trace against a passive run before trusting any
steady-state number.

**Same-timing baseline.** Script 35 can rerun preparations with both drive
gains at zero; pulse slots, `readout_gap` and the passive delay stay
identical, so the difference is the drives alone. `g` is the one that matters:
a flat |g> trace by itself cannot tell drive-induced excitation from readout
misassignment (on run 33 `#370` that ambiguity cost real time). Adding `e`
separates the reset from plain T1 decay, though at 2 us against T1 = 75 us
that is only ~3%. Each entry costs one acquisition.

The three preparations are a Python loop, not a sweep axis: `prep` selects
which pi pulses the Program *declares*, so it cannot be a `QickSweep1D`. Each
pass is a full hardware time sweep, so it costs three acquisitions, not
`3*expts`.

**The operating point is derived, and the derivation is on thin ice.** With
run 33's measured `g_tilde_fit = [0.1588, 0.0219]`, g~ reaches only 0.18 MHz at
full gain where the `kappa/3` plateau wants ~0.3-0.45 MHz, so `Gamma` tops out
at ~43% of the ceiling (0.11 of 0.26 MHz) and the optimal `Omega_ef` is
correspondingly tiny -- **0.047 MHz**. Two consequences the code handles
explicitly:

- Inverting `omega_ef_fit = [10.8768, 0.0852]` for 0.047 MHz gives a
  **negative** gain, because the target is below the fitted intercept.
  `ef_gain_for_omega` falls back to the through-origin form `gain = omega/m`
  (Omega_ef must vanish at zero drive, so a non-zero intercept is a fit
  artifact from the lowest-gain rows) and says which form it used. It never
  returns a negative gain -- the DAC would clip it to zero and the e-f drive
  would silently be absent.
- Because that intercept is comparable to the target, construction prints a
  NOTE that the e-f gain rests on extrapolating step 2's fit ~10x below its
  measured range, and that `gain_ef` should be *scanned* around the derived
  value rather than trusted. `Gamma` is not monotonic in `Omega_ef` -- the
  paper's configuration C is slower than B purely by overshooting it
  (verified: 1.40 vs 2.78 MHz at their kappa).

`cfg.expt.omega_ef_target` records the Omega_ef the gain was chosen *for*, and
`analyze()` predicts `Gamma` from that rather than recomputing forward through
`omega_ef_fit` -- which would re-apply the intercept the inversion just dropped
and quote a different rate (0.090 vs 0.112 MHz) than the one being aimed at.

**Timescales, and the honest caveat.** `analyze()` fits the |e> trace to a
single exponential (not the paper's master equation -- the oscillatory
structure they resolve needs a rate an order of magnitude above this) and
compares the result with `reset_rate`, with the floor-subtracted steady state,
and with T1. At Gamma = 0.11 MHz the 1% time is ~6.5 us against T1 = 52 us, so
the reset is ~8x faster than doing nothing; construction warns when that ratio
drops below ~3. That is the number that improves when the f0-g1 attenuation
comes off.

### `type` / `pulse_type` in `_configure_pulse_params` (fixed)

`RabiExperiment` used to raise `KeyError: 'type'` if `type` was in `params`:
the config-copy loop skipped keys the caller supplied, leaving `params_def`
without `type`, which the next line read unconditionally. Worse, passing
`type='flat_top'` *without* `pulse_type` would have silently played a gaussian,
because the Program reads `pulse_type`.

Both are now fixed in `rabi.py` and its near-copy `flux/rabi_flux.py`:

- the config-copy loop populates `params_def` **unconditionally** -- it is the
  defaults dict, and every later merge is `{**params_def, **params}`, so a
  caller's value still wins. This also stops `check_params` reporting
  `gain`/`length`/`ramp_sigma*` as "unexpected parameters" on every run.
- `type` and `pulse_type` are now symmetric aliases: setting either sets the
  other, and setting both to *different* values raises rather than silently
  preferring one.

Verified unchanged for scripts 04b (`const`), 05 (`gauss`) and 13b (config
default): same `type`/`pulse_type`/`sweep`/`loop`/`start`. One deliberate
behaviour change: `Rabi2D`'s display label now reads `flat_top` instead of
`gauss` for a flat-top sweep.

### Two rejection stages in every 2D fit here

Shared by `F0G1SpecPower` and `EFRabiPower` via `_per_row_r2`. A row can be bad
in two different ways, so both are checked:

1. **`min_r2 = 0.1`** (matching `QickExperiment.run`'s own floor) drops rows
   where the Lorentzian did not describe the data. `fitlor` returns `[nan]*4`
   only when `curve_fit` *raises*; a row with no peak at all converges happily
   onto a noise spike, lands its centre inside the window with a plausible
   linewidth, and is identifiable only by R2 ~ 0. Measured on synthetic data:
   without this gate two dead rows out of twelve pulled a recovered
   `a = -12.0` to `-13.15`; with it, to `-11.99`.
2. **Sigma-clipping** via the FWM band tracker's own `_fit_band_model`, shared
   rather than reimplemented since every model here is
   `value = slope*design + offset` (design = `gain**2` for a Stark shift,
   `gain` for a Rabi rate). This catches values that are real fits but off the
   line.

`EFRabiPower` adds a third, model-specific gate: a fitted period longer than
the scanned window is an extrapolation, not a measurement -- there is no full
oscillation in the data to set the rate, and `fitsin`'s lower frequency bound
of 1e-3 MHz makes that reachable.

All of them print what they dropped.

### Checking it without hardware

```
python check_f0g1_config.py            # section shape, design vs measured freq,
                                       # every fit evaluated, Gamma vs g~
python check_f0g1_config.py --config <path-to-any.yml>
python check_pi_ge.py                  # f0g1_reset.py is in its FILES list
python verify_active_reset.py          # every mode; F0G1Spec where dacs.f0g1 exists
```

`reset_rate` / `optimal_omega_ef` implement the paper's Eq. 1 and its Fig. 3a
red curve. They reproduce the paper at `kappa/2pi = 9 MHz`: configs A
(`Omega_ef, g~` = 3, 4.8) and B (1.5, 2.9) both give `Gamma = kappa/3 = 3.0`
while C (3, 2.9) gives 1.40, and the predicted optima 3.04 / 1.40 match the
paper's chosen 3 / 1.5.

### `reset_rate` is not what a reset measurement reports

`reset_rate` returns the **asymptotic eigenvalue** `2*min|Im(lambda)|`. Every
reset measurement here instead fits a single exponential to a finite trace, and
the two are different numbers: the onset delay drags the fit *below* the
eigenvalue at weak drive, the fast initial transient pushes it *above* at
strong drive. On run 34 `#292` they differ by up to 2.9x with no measurement
error involved at all -- at `g~ = 0.276, Omega_ef = 0.121` the eigenvalue is
0.3053 and the same-grid fit is 0.1039.

`helpers.f0g1.fitted_reset_rate(g_tilde, omega_ef, kappa, t)` is the
fit-comparable prediction: it evolves Eq. 1 from `|e,0>` and fits
`P(not g) = |c_e|^2 + |c_f|^2` with the same `fitter.fitexp` the experiments
use, on the run's own time axis. **Compare measurements against this, not
`reset_rate`.** Scripts 36 and 37 both do.

Two implementation notes. The damping sign is `-i*kappa/2`, opposite
`reset_rate`'s `+i*kappa/2` -- that one only ever takes `|Im(lambda)|`, so the
sign is free there and is not free here. And the non-Hermitian evolution is
exact rather than an approximation to a Lindblad solve: an emitted photon
leaves the qubit in `|g,0>`, which is dark and never returns, so the no-jump
trajectory is the whole story. Checked against a 9x9 Liouvillian with the same
parameters: identical to 4e-5.

Using the eigenvalue is what made run 34 `#292` look unintelligible. Its
measured/predicted ratios spanned 0.09-4.09 (44x); recomputed against
`fitted_reset_rate` on each point's own grid they are 0.26-0.46 (1.8x) over the
five resolved points, i.e. a single uniform factor rather than scatter.

### Fitting the ripples

Under strong drive the reset trace *rings* -- `#446`'s best point shows eight
oscillations over two decades -- and a single exponential throws all of it
away. `helpers.f0g1.fit_reset_trajectory(t, y, g~, Omega_ef, kappa, reps)`
fits the measured trace to `reset_trajectory` (Eq. 1 from `|e,0>`) with g~,
Omega_ef, kappa, a floor and an amplitude free. `ResetTimeScan.analyze` runs
it **alongside** the exponential, not instead: scripts 36/37 key off
`reset_gamma`, so that number keeps its meaning. Results land in `data` as
`traj_*`, and `display` overlays the curve in violet.

It works. On run 34 it fits to within shot noise -- rms 0.006-0.015 against
0.005-0.009 expected at 3000 reps -- and recovers all three parameters from one
trace. `#446`'s five trusted points give g~ at 0.91-0.96 of script 34's
`g_tilde_fit` and Omega_ef at 0.47-0.55 of `omega_ef_fit`, the same ~2x
overestimate scripts 36/37 find by two other routes. **This is what settles
kappa**: the ripple damping measures it directly, giving 1.04-1.52 against the
config's 0.9209, i.e. nearer `device.readout.kappa[1]` = 1.3448. It still
trends with f0-g1 gain, which remains unexplained.

**Two guards, both of which fired on real data.**

*The ripple gate.* Without an oscillation the three parameters trade off
freely: `#446`'s four weak-drive points fit monotonic decays well (rms as low
as 0.0071) while returning kappa = 2.3-2.5 and g~ up to 1.9x the config. The
gate is a measured rebound from the **first local minimum** to the next local
maximum, in shot-noise units, thresholded at 3 sigma. It must be the first
minimum, not the global one -- the global minimum normally sits in the noisy
tail, where the "rebound" is just scatter, which read 0.009 on traces that
visibly ring and 0.000 on one whose last point was lowest. On `#446` the metric
separates cleanly, 0.0-0.3 sigma for the monotonic traces against 3.9-29 sigma
for the ringing ones with nothing in between, so the cut is not finely
balanced. A rejected fit is still printed and still drawn, faded, labelled
"not trusted" -- it is a diagnostic, not a measurement. The gate is
deliberately conservative and will reject a sound overdamped fit (`#446`'s
f0g1 0.55 / mult 0.8 looked fine and was rejected); that is the safe direction.

*Multi-start seeding and bound detection.* `omega_ef` is normally the config's
nominal rate, ~2x what the drive delivers, and seeding there sent `#446`'s
third point to kappa pinned at its upper bound with rms 0.043 where a 0.5x seed
reached 0.0091 -- and it passed the ripple gate, since that only asks whether
the trace rings. The fit now tries both Omega_ef seeds against two kappa seeds
and keeps the lowest rms, and any of g~/Omega_ef/kappa landing on a bound marks
the fit untrusted.

**`37_reset_rate_map.py --reanalyze DATA_DIR`** re-fits a finished run and
**remakes every PNG**, with no hardware and no new run folder. Each point is
reloaded with `ResetTimeScan.from_h5file` (which bypasses `__init__`), then
re-analysed and re-displayed, so the per-point figures pick up whatever
`analyze` now computes rather than only the map being redrawn. Two things it
has to do by hand: set `metadata_path`, which `from_h5file` leaves unset and
whose absence would send `save_fig` to a `metadata_and_images` folder *inside*
`data/`; and wrap `display` in `common.nonblocking_plots()`, since `save_fig`
calls `plt.show()`. Note it also rewrites each point's `.yml`, because
`ResetTimeScan.display` calls `save_config`.

### Runner notes, scripts 27–37

How to run each step and what it found. These used to be the scripts'
docstrings; the physics is in the sections above.

**27 (f0-g1 spectroscopy).** The design frequency `f_ge + f_ef - f_waste`
(~225 MHz on Waymar) uses bare transition frequencies and is good to a few MHz,
hence `style="coarse"` (200 MHz wide). On a first run start at high gain -- the
line saturates and is easy to find -- and come down once it is located. If
nothing appears at gain 1.0 the problem is the line, not the power: check the
tone reaches the sample. `prep="g"` is the control: with no |f> population, a
feature that survives is not f0-g1.

**28 (f0-g1 Stark shift).** Fits `a*gain^2 + c`, vertex forced to zero drive,
into `device.f0g1.stark_fit`, which 31–35 read to keep the drive on resonance.
Run 27 with `update()` first, or the window is centred on the design frequency
and the peak can walk out of it at high power; the expected shift is ~10 MHz,
so the span needs several times that. `FIXED_AREA` holds `gain*length`
constant so every row gets a comparable Rabi angle, as the paper does --
otherwise high-power rows saturate and broaden. The lowest gains cannot reach
the angle within the length cap; `analyze()` reports how many rows were
clipped.

**29 (e-f flat-top Rabi)** is the exploratory one: it confirms the flat-top
e-f drive oscillates and fixes a gain and time window for 30. The reset wants
`Omega_ef ~ kappa/4 ~ 0.3 MHz`. Extrapolating from the gaussian pi_ef (sigma
0.023 us at gain 0.78) puts that around gain 0.05–0.1, an order of magnitude at
best across pulse shapes. `max_length = 2*num_osc*length`, so length ~1 us is a
5 us window, ~2.5 periods at 0.5 MHz.

**30 (Omega_ef vs gain).** Fits `Omega_ef/2pi = m*gain + b` into
`device.f0g1.omega_ef_fit`. With the fixed time axis, the bottom row needs
`START_Y > 1/(window*m)` for one full period and the top row wants at most ~5
periods, so `START_Y + SPAN_Y < 5x` that. The defaults assume `m ~ 5 MHz` per
unit gain, extrapolated from the gaussian pi_ef. The "Unexpected parameters
found in params" line at startup is `RabiExperiment` noise (it never puts
`length` in its `params_def`). Script 36 later found these rates ~2x too high.

**31 (e-f spectroscopy under the f0-g1 drive)** is the single-power check that
the dip is there and the window is the right width; 32 produces the
calibration. At `gain_ef = 0.05` the stored `omega_ef_fit` gave
`Omega_ef/2pi ~ 0.63 MHz`, a 0.8 us pi and a ~1.3 MHz line.

**32 (e-f Stark shift vs f0-g1 power)** fits `a*gain^2 + c` into
`device.f0g1.ef_stark_fit`.

- **Timing.** The paper's step 3 fixes both lengths and varies only the f0-g1
  amplitude (a 420 ns f0-g1 tone around a 140 ns e-f pi); their step 1 instead
  holds `Vf0g1*tr` constant. Here both lengths derive from `GAIN_EF` (the probe
  is a pi, `t_pi = 1/(2*Omega_ef)`, and the f0-g1 tone is `f0g1_length_ratio`
  times that), so `GAIN_EF` pins the timing.
- **Pin `GAIN_EF` to the operating Omega_ef**, the Fig. 3a optimum at full
  f0-g1 gain: 0.539 MHz, `gain_ef ~ 0.046` (`plot_reset_rate.py` and
  `check_f0g1_config.py` print it). That measures the shift at the strength
  the reset uses. It also shortens the sequence from 23 us to 2.8 us (82% → 10%
  of T1_ef ~28 us), so decay stops eating the contrast -- why the gain-1.0 row
  of `#366` was a shallow blob. And it puts `2*g~/linewidth` at 1.30 against
  the paper's 1.34: the doublet merges, and a single symmetric fit returns the
  intensity-weighted centroid, which by the sum rule is the bare Stark-shifted
  frequency, with no two-peak model selection. At the Fig. 3a optimum that
  ratio is 1.07–1.39 for every kappa and g~ tried, because the optimum fixes
  `Omega_ef/g~` and the plateau fixes `kappa/g~`.
- **Span** must cover `2*g~` plus two linewidths, ~4.2 MHz at full gain, so 8
  leaves baseline. `#366` ran span 4: the feature filled the window, the
  two-peak separation ran into its bound, and through the `a*gain^2` leverage
  that row flipped the whole fit's sign. The class warns when span is too
  narrow. If the f0-g1 attenuation has come off since 28 ran, rerun 28 first,
  then widen the span until the dip stays inside at the top gain.
- **Expts** needs ~5 points per 1.27 MHz linewidth, so 150 over 8 MHz (53 kHz)
  is ample. The surplus buys centroid SNR, which is what the reset cares about
  (+/-100 kHz costs 6% of Gamma).

**33 (f0-g1 Rabi).** Run it at one gain to see the timescale, then set 34's
window. `MAX_LENGTH` is a tradeoff: at `2*g~ << kappa` the transfer is
Purcell-like at `4*g~^2/kappa` (~1.2 us at g~ = 0.2 MHz, kappa = 1.18 MHz), so
too short a window leaves the decay incomplete, while too long under-resolves
the fast rows (at g~ = 1 MHz the period is 0.5 us). It prints g~, kappa, the
damping regime and the best reachable rate, and writes nothing.

**34 (g~ vs gain).** Fits `g~/2pi = m*gain + b` into
`device.f0g1.g_tilde_fit`, and `device.f0g1.kappa` takes the median of the
per-row kappa. `MAX_LENGTH` (from 33) must cover the slowest, lowest-gain row.
Without frequency tracking the drive would detune further at each step and read
as g~ saturating. Constraining kappa across rows, as the paper does, is a
worthwhile second pass once the per-row values look flat.

**35 (reset time).** Needs 27–34 run with `update()`. `GAIN_F0G1` defaults to
full scale and `GAIN_EF` to `optimal_omega_ef` at the resulting g~; set either
to walk the Fig. 3a surface (e.g. the paper's configuration C). Set
`MAX_LENGTH` to 2–3x the printed time to 1% residual.

**35b (where the |g> floor comes from)** is a diagnostic and writes nothing. On
run 33 `#370` the |g> preparation sat at 11.35% "not g" after the drives,
against 3.88% from 30 k |g> shots of the single-shot calibration `#377`.
Thresholding (9.2%) and the blob mean (11.8%: `avgi_g = -9.64` between
`vg = -12.67` and `ve = 12.98`, no threshold involved) agree that most of the
gap is real excitation. It is already there at the shortest drive time (74 ns)
and does not grow (13.5% → 11.3%).

35b runs the same |g> preparation four times: both drives, f0-g1 only, e-f
only, and neither. Frequencies, lengths, `readout_gap`, reps and the passive
delay are identical in all four, and a zero-gain pulse keeps its slot, so the
differences against `neither` attribute the floor with no readout model. If
`neither` already sits near 11%, the drives are exonerated and `readout_gap` is
the next suspect. `preps` keeps `e` because `ResetTimeScan` requires it, and
the "barely beating doing nothing" / "extrapolating step 2's fit" warnings on
the drive-off cases are harmless. Result: the f0-g1 tone adds ~2.7% at full
scale and the e-f tone nothing (see "Choosing the f0g1 gains").

**36 (reset rate vs e-f gain).** On run 33 at f0-g1 gain 0.74, `gain_ef =
0.0325` (the predicted optimum) gave Gamma 0.11–0.14 MHz and 0.0460 ("past" it)
gave 0.20–0.22, so the optimum is not where the model puts it. Inverting the
measured Gamma at script 34's g~: nominal Omega_ef 0.383 → implied 0.188
(0.49x), 0.539 → 0.246 (0.46x). The e-f drive acts at ~half its calibrated
rate. That is an `omega_ef_fit` error -- script 30 fits with `fitsin`, which
has no decay term and returned R² 0.43–0.79, against R² ≥ 0.99 for 34's g~ --
not a failure of Eq. 1. 36 runs a full `ResetTimeScan` per e-f gain and fits
one scale factor `s` in `Omega_ef -> s*(m*gain + b)`, the correction to apply
to `omega_ef_fit`. The e-f side is both the weak leg and the one Gamma is most
sensitive to, since Gamma is non-monotonic in Omega_ef. `READOUT_GAP` is a
short 2 us because a long gap lets drive-induced heating develop (tau ~14 us,
~+4%), inflating every trace.

The `fitsin` explanation above is **wrong** and is kept only so it is not
re-derived: tested on synthetic data, `fitsin` recovers the frequency to 0.3%
at R² = 0.42, so the poor Rabi fit does not produce the 2x. The origin of the
factor is still unknown. It is not a pulse-shape mismatch either -- run 34
`#267` measured `omega_ef_fit` with `flat_top` and the reset drives `flat_top`,
so the half-amplitude plateau factor does not apply.

**What the scale factor is, and is not.** It is "the number that makes Eq. 1
reproduce the observed reset rate, holding g~ and kappa at script 34's values"
-- not an independent measurement of the drive. Four things could produce it:
Omega_ef really being 2x lower, g~ being wrong, kappa being wrong, or Eq. 1
being incomplete. On run 34 `#292` a single Omega_ef scale fits best by a wide
margin (rms 0.014 against 0.032-0.052 for a g~ scale, a kappa shift, or both),
and the per-point implied scale is 0.42-0.59 across 7 resolved points spanning
11x in g~ and 17x in Omega_ef. That is tight, but it is still model selection
among a restricted set.

**Do not paste the scale into `omega_ef_fit` on that basis alone.** Scripts
31/32 derive the e-f pi length as `t_pi = 1/(2*Omega_ef)` from the same fit, so
halving it doubles `t_pi` and turns their pi pulse into a 2pi pulse. Script 36
is what settles it, because the *peak position* calibrates the drive on its
own: at the optimum the drive is at `Omega_ef_opt(g~, kappa)` by definition, so
the peak needs neither the absolute Gamma nor kappa to be right. At
`GAIN_F0G1 = 0.74` the two hypotheses are well separated -- the peak sits at
`gain_ef = 0.035` if script 30 is right, `0.075` if the reset data is -- and
the shipped `GAIN_EF_LIST` brackets both. Note `EFRabiPower.update()` rewrites
`omega_ef_fit`, so a hand-set value is lost the next time 30 runs without
`--no-update-config`.

**Settled on run 34 `#293`: the scale is 0.51.** Eight e-f gains at f0-g1 gain
0.74 gave a clean peak with turnover on both sides. Two independent routes
agree to 3%: the whole-curve fit gives `0.510` at 10% rms over 8 points, and
the **peak position alone** gives `0.524` -- measured peak `gain_ef = 0.0706`
against the as-calibrated model's `0.0363`. The peak route is the stronger
evidence, since it uses neither the absolute Gamma nor kappa; checked across
kappa 0.92-1.59 the predicted peak moves only 0.0363 -> 0.0395 (10%) while the
Omega_ef scale moves it 2.4x, so the two do not trade off.

So `omega_ef_fit` reads about **2x high** and should be `[14.82, 0.0220]`.
Note this run also prefers `kappa = 0.9209` strongly (rms 0.103, against 0.415
at 1.345 and 0.565 at 1.59) -- the peak *height* constrains kappa, and it
points back at the config value, not at the ~1.6 that `#292`'s g~ trend
wanted. Those two are not yet reconciled: `#293` is one g~, `#292` spans g~,
and only `#292` sees the trend.

**`--reanalyze DATA_DIR`** re-fits a saved run's h5 files and redraws, with no
hardware and no new run folder -- the analysis is `analyze()` / `draw()` /
`load_run()` at module level for exactly this. `gain_f0g1` is read back from
the data rather than taken from `GAIN_F0G1`, so re-analysing an old run cannot
silently use today's constant. The figure goes to the matching
`metadata_and_images/` folder. It prints `OPTIMAL e-f GAIN`, the peak from a
parabola through the top three points in log gain.

`MAX_LENGTH`/`EXPTS` are 6.0 us / 150, not the original 5.0 / 50: at the
predicted optimum `tau ~ 0.18 us`, and 50 points over 5 us gave under 2 samples
per time constant, which is the same under-sampling that invalidated `#292`'s
fastest point.

**37 (reset rate map).** `37_reset_rate_map.py` -- note the number collides
with `37_fwm_efficiency_vs_buffer_frequency.py`; both are registered in
`experiment_name_for_script` by full filename, so only the spoken name is
ambiguous. One `ResetTimeScan` per chosen `(g~, Omega_ef)` point, drawn against
Fig. 3a. Its time window is **per point**, from `window_for()`: a single fixed
window cannot work here, because Gamma is non-monotonic in Omega_ef, so an
Omega_ef calibration error makes points below the optimum read slower than
predicted and points above it read *faster* -- the sign flips across the
optimum. `window_for` therefore brackets `SCALE_RANGE = (0.4, 0.7, 1.0)`,
taking the span from the slowest case (`N_TAU` time constants) and the sampling
from the fastest (`PTS_PER_TAU` per tau). On `#292`'s six points that gives
every one of them >=3 tau of span and >=4 samples per tau, against 0.8
samples per tau for the worst point under the old flat 20 us / 50 pts.

**`OMEGA_EF_SCALE = 0.51` is applied in the script**, from `#293`, rather than
written to `device.f0g1.omega_ef_fit` -- scripts 31/32 derive their e-f pi
length from that same fit, so halving it would turn their pi into a 2pi. The
script asks `ef_gain_for_omega` for `mult * om_opt / OMEGA_EF_SCALE`, the gain
that *delivers* the target, and then records `OMEGA_EF_SCALE *
omega_ef_for_gain(...)`, what is actually delivered. Everything downstream --
the window, the prediction, the map's y axis -- is therefore in true Omega_ef.

That also makes `mult` mean something again. In `#292` it was a multiple of an
optimum computed from a 2x-high fit, which is exactly why all four of its
"mult = 1.0" points sat at about *half* the real optimum and read as though
Gamma tracked g~ instead of plateauing. `SCALE_RANGE` now brackets 1.0 as
`(0.7, 1.0, 1.4)` -- residual uncertainty on an already-corrected rate, since
`#292`'s per-point implied scale was 0.42-0.59 against the 0.51 applied here.

**The shipped grid now walks DOWN in f0-g1 amplitude**, which is the direction
nothing has measured. `#446` (f0g1 0.30/0.40/0.55 x mult 0.8/1.3/1.8) plus
`#293`'s e-f scan at 0.74 settled the upper half: **time to 1% residual is flat
at ~2 us all the way from f0g1 1.00 down to 0.40**, because the faster Gamma at
high drive is offset by more ringing to settle out. Optimising `mult` at each
g~ gives 1.93 us at gain 1.00 and 1.71 us at 0.40 -- within the +/-0.5 us
jitter of the metric (see below) -- for 6x less deposited power. Eq. 1 says
that holds to gain ~0.18, a 20x saving, then falls off the plateau at
`g~ = 0.28*kappa` = 0.288, i.e. gain 0.157.

So the grid is f0g1 0.20/0.28/0.40 x mult 1.0/1.3/1.6, plus `(0.15, 1.3)` as a
below-the-plateau control and `(0.55, 1.3)` as an anchor against `#446`.
Eleven points, ~41 min. `mult = 1.8` is dropped: it lost to 1.3 at every gain
in `#446`. Derived gain_ef runs 0.009-0.056, which at the low end is a long
extrapolation of `omega_ef_fit` -- that is tolerable here only because the
trajectory fit measures Omega_ef back out of each trace independently.

**The figure of merit is time to a residual, not Gamma**, and it must be the
*last* time the trace is above the threshold, not the first crossing: these
traces ring, so a first-crossing metric reports a transient dip. Measured
first-crossing on `#446` claimed its fastest point settled in 0.47 us when it
actually finished 2.6% above the floor. The metric carries ~+/-0.5 us of
intrinsic jitter near the optimum, because the last crossing jumps
discontinuously as a ripple peak crosses the threshold -- so differences below
about 0.5 us are not real.

**The old "err high rather than low on gain_ef" rule is reversed.** It was
written when Omega_ef was believed 2x larger. Measured, the penalty is worse
above the optimum (mult 2.5 -> 5.5 us) than below (mult 0.75 -> 2.9 us), and
the config's own derived `reset_gain_ef` sits at **mult ~0.51** -- on the
rising low side, needing roughly double.

The summary table and the figure are module-level `summarize(rows)` and
`draw_map(rows, kappa)` rather than inline in `__main__`, so a finished run can
be redrawn from its h5 files without re-acquiring -- load the runner with
`importlib.util.spec_from_file_location` (`37_...` is not an importable module
name), rebuild `rows` as `(g~, Omega_ef, pred, meas, gain_f0g1, gain_ef,
mult)`, and call them. `reset_rate_map.png` goes to `metadata_path`, not
`run_dir`: `run_dir` is the *data* folder, and every other image in this tree
lives under `metadata_and_images/`. Run 34 `#292` predates both that fix and
`fitted_reset_rate`, so its figure was regenerated in place.

### Stark spectroscopy fit modes (script 17)

In the SMPD wiring the Stark tone sits on the buffer line, so `StarkSpec`'s
y axis *is* buffer gain (labelled as such). `cfg.expt.fit_type` picks the
analysis:

- **`stark_shift`** (class default) — one Lorentzian per buffer gain, then a
  quadratic fit of the pulled frequency. The gain² coefficient converts to
  on-resonance photons per gain² through `buffer_dephasing_per_photon(kappa[0],
  chi_buffer, stark_freq − frequency[0])`. Only meaningful with the Stark
  tone off resonance, as in script 17c; it warns within 2κ of f_g. This is the
  mode whose `update()` writes `device.readout.nstark`, which scripts 20 and 23
  depend on. It used to divide by `chi[stark_qubit]`, the waste line's
  placeholder −0.1, not the buffer's χ.
- **`two_peak`** — fits the first two photon-number peaks with
  `two_lorentzians` (shared linewidth, separation seeded at
  `chi_buffer = device.readout.chi[0]` and bounded to 0.3–3× it). Adds
  `peak_y0`/`peak_amp0`/`peak_amp1`/`peak_freq`/`peak_delta`/`peak_gamma` —
  the full parameter set, so any row's fit can be redrawn — with NaN where a
  row failed. `update()` no-ops in this mode.

  `display_peak_amplitudes()` draws three panels, saved with a
  `_peak_amplitudes` suffix: the two amplitudes vs buffer gain, the fitted
  separation vs `chi_buffer`, and the measured spectrum plus fitted curve at
  the first, middle and last buffer gain **that converged** (so a failed row is
  never shown as an example). Those three gains are marked with dotted vlines
  on the amplitude panel. The first two panels share the buffer-gain x axis;
  the third is a frequency cut, hence the explicit gridspec rather than
  `subplots(sharex=True)`.

Two things the two-peak fit needs to work:

- **The frequency span must cover `f_ge + chi_buffer`.** With
  `start = f_ge - span/2 + df`, script 17's `span=3, df=-1` spans
  `f_ge-2.5 … f_ge+0.5`, which contains the n=1 peak at `f_ge-1.64`. Narrow the
  span and `peak_amp1` gets extrapolated from a tail instead of fit.
- **The brightest peak is not always n=0.** Once the buffer fills, n=1
  dominates; since `delta` is sign-bounded, seeding on the wrong peak makes the
  fit unreachable. `fit_two_lorentzians` tries both peak assignments and keeps
  the lower residual.

The fitted-separation panel is the sanity check: it should sit on
`chi_buffer`. At the lowest gains there is no n=1 peak to constrain it, so
`peak_delta` drifts to its bound there — expected, not a fit failure.

### Buffer gain → photon number (`device.fwm.buffer_power_fit`)

`Data/run32_Waymar/metadata_and_images/#351_.../refit_two_peak.py` is an
offline re-analysis of a `two_peak` Stark run that goes further than the
experiment does: instead of two free peak amplitudes it fits the whole
**photon-number-splitting comb** — Poisson-weighted Lorentzians at
`f0 - (2n·chi + chi'·n²)` with width `kappa/2 + n·Δγ` — so each buffer gain
yields a photon number `n_bar` rather than a pair of amplitudes. The model is
the one in that folder's `analysis.ipynb`.

The comb geometry (`f0`, `chi`, `chi_prime`, `delta_gamma`) is shared across
all buffer gains; `n_bar`, `kappa`, amplitude and baseline are per gain.
**`kappa` has to be per-gain**: the comb broadens peaks only with photon
*number*, but the n=0 peak broadens with *occupation* (measurement-induced
dephasing), and sharing it visibly under-fits the sharp low-gain peak.

`n_bar` is then fit against gain in the form `--gain-fit` names (default
`constant_quadratic`, `n_bar = c0 + c2·gain²`, c0 ≥ 0) and written to
`device.fwm.buffer_power_fit` / `buffer_power_fit_type`, with
`buffer_power_fit_attn_db` / `_pulse` from the run's applied buffer setting so
consumers can rescale it like the Ramsey calibration. `--normalization
area|height` picks the comb's line model (see "Probe power and the PNS comb");
the printout is saved as `<h5 stem>_refit.txt`. **`c2` is not
`device.readout.nstark`**, which scripts 20 and 23 use and script 17c writes
from the off-resonant line shift. Run 34's calibration (315 ± 10 per gain² at
f_g, 10 dB flat_top) is in "Photon number from Ramsey dephasing (scripts 38, 41)".

`#447_.../fit_shared_means.py` reads those coefficients back to put its
excited fractions on a `1 - P(0) = 1 - exp(-n_bar)` axis, whose slope reads as
a conversion efficiency. Both scripts read the fit from the config rather than
hardcoding it, so revising the Stark fit cannot leave a stale copy behind.

**Open discrepancy**: on run 32 those slopes reach 1.65 at pump gain 1.0,
i.e. above unity, which no per-photon conversion can give. The two runs drive
the buffer at the same frequency through the same line at identical
attenuation, so it is not a wiring mismatch — either the PNS `n_bar`
under-counts by ~2× or the `e`-scan excited fraction has a contribution that
is not per-photon conversion. Unresolved; don't read efficiencies off that
axis until it is.

## Config YAML (`configs/`)

`config.load()` returns an `AttrDict`, so `auto_cfg.device.qubit.f_ge[qi]`
works. Naming is `YYYY_MM_DD_bfg<run>_<device>_rfboard.yml` — the date the
cooldown started plus the fridge run number
(`2026_09_21_bfg33_smpd_v2_waymar_rfboard.yml` is run 33, Waymar). That
filename is authoritative: `run_and_device_from_config` parses
`bfg<run>` and the `<device>` token before `_rfboard` out of it to build
`Data/run<N>_<Device>/`, so a config must be named to the convention or the
scripts refuse to start. A `*_model.yml` sibling, when present, holds model/fit
metadata.

Top-level sections:

- `aliases` — `ip` (nameserver), `soc` (Pyro name).
- `device.qubit` — `f_ge`, `f_ef`, `f_fh`, `T1`, `T2r`, `T2e`, `kappa`, `pop`,
  `temp`, `spec_gain`, `tuned_up`, `pulses.*` (per-qubit lists), plus scalars
  `low_gain`, `max_gain`.
- `device.readout` — per-line lists: `frequency`, `gain`, `phase`,
  `readout_length`, `trig_offset`, `threshold`, `g_mean`/`e_mean`/`sigma`,
  `fidelity`, `chi`, `kappa`, `lamb`, `final_delay`, `reps`, `rounds`, `reset`,
  `pulse_type`, `active_reset` (legacy classes only), `reset_threshold`, `reset_ringdown`,
  `reset_e`/`reset_g`, `jpa`; scalars `reps_base`, `rounds_base`.
  **Index 0 = buffer line, index 1 = waste/qubit readout line** in the SMPD
  scripts.
- `device.fwm` — non-indexed FWM params: `pump_power_fit` with
  `pump_power_fit_type: quadratic|linear|linear_quadratic|cubic|quartic` (pump
  gain ↔ band frequency; `[a, c]` for the first two, `[a, b, c]` for the
  third, `[a, b, c, d]` for `cubic`, `[a, b, c, d, e]` for `quartic`), and
  `buffer_power_fit: [c1, c2]` with `buffer_power_fit_type: linear_quadratic`
  (buffer gain → photon number; see below), and `pump_pulse_type` /
  `buffer_pulse_type` (see "FWM experiments").
- `device.f0g1` — non-indexed f0-g1 reset calibration (`frequency`,
  `stark_fit`, `omega_ef_fit`, `ef_stark_fit`, `g_tilde_fit`, `kappa`), the
  drive shapes `pulse_type` / `pulse_type_ef`, and the reset operating point
  `reset_gain`, `reset_gain_ef`, `reset_length`, `reset_residual`, `reset_gap`
  (null = derive; see "Reset modes").
- `hw.soc.dacs` — `qubit`, `readout`, `flux`, `jpa` (each with `ch`, `nyquist`,
  `mixer_freq`/`mixing_freq`, and for jpa `pump_gain`, `pump_freq`, `jpa_dc_val`,
  `attn1`, `attn2`).
- `hw.soc.adcs.readout.ch`.
- `hw.soc.rfboard_active` / rfboard section — attenuator, filter, and DC bias
  state applied by `rfboard.activate_qubit_rf`.

### Config writers (`helpers/config.py`)

`update_qubit`, `update_readout`, `update_stark`, `update_lo` (all take
`qubit_i`), `update_fwm` (no index), `update_rfboard`, `update_rfboard_active`,
plus the generic `update_config` and the `init_*` builders. All of them
`load()` → mutate → `save()` the YAML in place, and print a
`*Set cfg ... to X from Y*` line unless `verbose=False`. Values are rounded to
`sig=4` significant digits by `format_value`.

Experiment classes reach the YAML via `self.config_file` (from
`cfg_dict["cfg_file"]`), which is what their `update()` methods pass in.

## Data layout

`scripts/SMPD/Data/run<N>_<Device>/` — currently `run33_Waymar` and
`run33_Yuvi`, the two devices in the bfg33 cooldown; `run32_Waymar` is the
previous cooldown (451 experiments), kept for reference. Both name parts come
from the config filename, so the folder follows the `CONFIG` constant
automatically. The run folder is created on demand by `prepare_run` — never
make it by hand.

```
run33_Waymar/
  data/#<n>_<experiment_name>_q<qi>_<HHMMSS>/<prefix>_00000.h5
  metadata_and_images/#<n>_<experiment_name>_q<qi>_<HHMMSS>/
      <prefix>_00000.png      <- save_fig()
      <prefix>_00000.yml      <- save_config(), the cfg actually used
      NN_script_name.py       <- copy of the script that produced it
```

- `<n>` auto-increments per `(run-number, device-name)` pair
  (`next_experiment_number` scans `data/` for `#<int>_` folder names).
- `<experiment_name>` comes from `common.experiment_name_for_script`.
- `<prefix>` is the experiment class's own `prefix` (e.g.
  `qubit_spectroscopy_power__qubit1`), set in each class's `__init__`.
- `save_fig`/`save_config` write to `metadata_path`, falling back to
  `expt_path/metadata_and_images` when it's absent (i.e. notebook runs).
- Reload a saved run with `QickExperiment.from_h5file(fname)`.

## Two devices on one board: Waymar and Yuvi

The bfg33 cooldown has both SMPDs wired simultaneously, on disjoint physical
channels of the shared RF board. Switching between them is the `CONFIG`
constant and nothing else; because the channels do not overlap, activating one
device never disturbs the other's filter/attenuator state.

| Line | Waymar (`..._waymar_rfboard.yml`) | Yuvi (`..._yuvi_rfboard.yml`) |
|---|---|---|
| Qubit drive | DAC3 (A2) | DAC2 (A13) |
| Buffer, readout index 0 | DAC1 (A3) | DAC9 (A11) |
| Waste, readout index 1 | DAC0 (A16) | DAC8 (A9) |
| FWM pump | DAC10 (D6) | DAC11 (D8) |
| JPA pump | DAC13 (B14) | DAC12 (B3) |
| ADC buffer | ADC5 (C6) — see below | none |
| ADC waste | ADC4 (C5) -> `ch` 0 | ADC6 (C9) -> `ch` 1 |

### The ADC ports are the sharp edge here

`hw.soc.adcs.readout.ch` is the **QICK readout index**, and on this firmware it
is *not* the ADC port number. Read the real mapping off the board rather than
assuming one:

```python
for i, r in enumerate(soccfg["readouts"]):
    tile, block = [int(a) for a in r["adc"]]
    print(i, "-> ADC port", 4 * (tile - 1) + block, r["ro_type"])
print("
".join(soccfg["extra_description"]))   # which daughter cards exist
```

As of bfg33 that gives:

| RF-board port | QICK readout index | type | can sweep frequency? |
|---|---|---|---|
| ADC4 | 0 | `axis_dyn_readout_v1` | **yes** |
| ADC5 | 2–9 | `axis_pfb_readout_v4` (one static PFB bank, 8 channels on one ADC) | **no** |
| ADC6 | 1 | `axis_dyn_readout_v1` | **yes** |
| ADC7 | — | no readout instantiated | — |

Only ADC slots 2 and 3 have RF In cards (ports 4–5 and 6–7); slots 0 and 1 are
empty, so readout index 10 exists in the firmware but has no physical input.

**There are therefore only two frequency-swept readouts on the whole box.**
Resonator spectroscopy, `ResSpec`, and anything else that sweeps the readout
frequency can only run on ADC4 and ADC6. Pointing `adcs.readout.ch` at a PFB
index fails at `declare_readout` with *"readout N is static (PYNQ-configured) -
frequency must be set in declaration"*.

Yuvi's waste line is on ADC6, i.e. `ch: [1, 1]` — index 1 is the real one and
index 0 mirrors it because Yuvi has no buffer digitizer, so an accidental
`qi = 0` activation cannot detune the waste readout.

> **Waymar's config is stale as a result.** It still says `ch: [1, 0]`, i.e.
> buffer on readout 1 = ADC6 — which is now Yuvi's line. Waymar's buffer moved
> to ADC5, the static PFB bank, so it cannot do a swept readout there at all.
> Fix the Waymar config (or the cabling) before running Waymar again.

DAC numbers in `hw.soc.dacs.*.ch` *are* used directly — only the ADC side has
this indirection.

Two consequences of Yuvi's wiring:

- **Yuvi's readout DACs are interpolated generators** (8/9), where Waymar's
  (0/1) are full-speed, hence `dacs.readout.type: [int, int]`. No `mixer_freq`
  key is needed for them — an `int` readout gen gets `frequency - 400`
  automatically (`qick_program.py`). Only `dacs.pump` needs an explicit one.
- **Yuvi's buffer is not digitized.** There is no ADC on that line, so both
  indices of `adcs.readout.ch` point at the waste readout. Arrays stay 2-long,
  which all the indexing assumes, and nothing passes `None` into
  `soc.rfb_set_ro_filter`. But the scripts that default to `qi = 0` to measure
  the buffer — `02`, `02b`, `03`, `04`, `25` — must be run with `qi = 1` on
  Yuvi. FWM is unaffected: the buffer is drive-only there and the readout is on
  the waste line. There is no spare swept readout to patch a buffer ADC into
  (ADC5 is the static PFB bank, ADC7 has no readout), so this needs either a
  firmware change or giving up one of Waymar's.

`device.fwm` is deliberately **absent** from the Yuvi config rather than copied
from Waymar — an inherited band fit would silently give wrong pump gains, so
scripts 19–26 raise until `18_fwm_power.py` has run and written a real one.
`config.update_fwm` `setdefault`s the section, so nothing has to create it.

Yuvi's `device.qubit` / `device.readout` numbers are carried over from
`2026_08_16_bfg31_smpd_v2_yuvi_rfboard.yml` (run 31, the last real Yuvi tune-up)
as starting guesses, with `tuned_up: [false, false]`. Expected FWM pump from
those: ~5149 MHz.

**The hardcoded frequencies in the runner scripts are Waymar values** —
`01_tof` 7790, `02` 6682.2, `10_t1` 7781, `19`/`20`/`21` pump 5210/5220. They
are not derived from the config and must be hand-edited when running Yuvi
(waste ~7741, buffer ~6917).

## Offline analysis scripts (`analysis/`)

`analysis/` holds the canonical copy of the after-the-fact re-analyses.
`prepare_run` copies the relevant one into each run's
`metadata_and_images/#<n>_.../` folder alongside the runner script, so every
run carries the version that matches it and the copy sits next to its own data:

| Runner | gets a copy of |
|---|---|
| `17_stark_spectroscopy.py` | `refit_two_peak.py` |
| `23_fwm_efficiency_vs_power.py` | `fit_shared_means.py` |
| `26_fwm_efficiency_vs_power_checked.py` | `fit_shared_means.py` |
| `37_fwm_efficiency_vs_buffer_frequency.py` | `fit_shared_means.py` |

`fit_shared_means.py` also writes a `summary/` subfolder next to its other
output: `fwm_summary.png` (efficiency | dark count | noise-equivalent n̄ | NEP),
`fwm_summary.csv` (one row per sweep value), and `summary.txt` (calibration,
ceiling, duty cycle, best efficiency and NEP, and a warning if the shared fit
sat on a decay-ratio bound).

**Across runs: `analysis/summarize_fwm_runs.py`.** Runs from `analysis/`
directly (not copied anywhere). It finds every complete script-23/26 grid in a
fridge run, re-reduces each with the *current* `fit_shared_means.py` (so old
runs get the Ramsey calibration too), and writes everything under
`Data/<run>/metadata_and_images/FWM_summary/`: `nep_vs_pump_time.png`,
`nep_active_vs_f0g1.png`, `fwm_runs_summary.csv`, and `runs/<run>/` with each
run's full fit output. It never writes into a run's own folder. Per-run fits
are cached and only redone when `fit_shared_means.py` is newer, or with
`--refit`. The folder name has no leading `#`, so `next_experiment_number`
ignores it.

The mapping is `analysis_scripts_for_script()` in [common.py](common.py), the
same shape as `experiment_name_for_script`. A missing source warns rather than
failing the run — losing an offline re-analysis is not worth aborting an
acquisition over.

**Run them from the folder they were copied into, with no arguments.** Both
resolve `DATA_DIR` as `HERE.parents[1] / "data" / HERE.name`, i.e. their own
run's h5 files:

```
cd Data/run33_Yuvi/metadata_and_images/#42_stark_spectroscopy_q1_101530
python refit_two_peak.py --gain-max 0.3
```

Running the copy in `analysis/` directly does *not* work — `HERE.parents[5]`
is meaningless there, so `DEFAULT_CONFIG` comes out None and `--config`
becomes required. That copy is a template.

### Neither script hardcodes a config any more

They used to carry `DEFAULT_CONFIG = "..._waymar_rfboard.yml"`. With two SMPDs
on one box that was a live hazard: `refit_two_peak` ends with
`config.update_fwm(...)`, so run against Yuvi data it would have written Yuvi's
`buffer_power_fit` into **Waymar's** YAML, and `fit_shared_means` would have
read Waymar's `g_mean`/`e_mean`/`sigma` as the reference frame for Yuvi shots.

`_config_for_run()` now derives it from the run folder the copy was dropped
into — `run33_Yuvi` -> `configs/*bfg33_*_yuvi_rfboard.yml` — which is the
inverse of `common.run_and_device_from_config`. If nothing matches it returns
None and `--config` is required, rather than guessing.

## Starting a new run

A "run" is one fridge cooldown. Switching to a new one is a new config plus a
few constants; nothing under `Data/` needs creating.

1. **Copy the previous config** in `configs/` to a new
   `YYYY_MM_DD_bfg<run>_<device>_rfboard.yml`. Copy rather than start fresh:
   the old `device.qubit.f_ge`, `device.readout.frequency`, rfboard
   attenuator/filter settings and JPA bias are the starting guesses for the
   retune, and the `hw.soc` wiring is usually unchanged. What *is* stale after
   a cooldown, and gets overwritten as you work up the script sequence:
   `T1`/`T2r`/`T2e`, `threshold`/`g_mean`/`e_mean`/`sigma`/`fidelity`,
   `pulses.pi_ge`/`pi_ef` gains, `chi`, `lamb`, and both
   `device.fwm.*_power_fit` pairs.
2. **Point the scripts at it** — update the `CONFIG` constant at the top of
   each `NN_*.py` you are going to run, plus `plot_log_histogram.py` and
   `fwm_qc.py`. Nothing else: the run number and device name come out of the
   filename, so there is no second or third constant to keep in sync, and
   **`common.py` is never edited**.
3. **`README.md`** and the run-folder mentions in this file.
4. Leave the old config in place. The per-experiment YAML copy under
   `metadata_and_images/` is the record of what any single shot was taken with,
   but the dated config is what you reload when comparing runs.

`prepare_run` does `mkdir(parents=True, exist_ok=True)` on
`Data/run<N>_<Device>/`, and `#<n>` numbering restarts at 1 because
`next_experiment_number` only scans that folder's own `data/`. The first script
run against a new config filename materializes everything.

Retune order after a cooldown follows the script numbering: `01_tof` →
`02_resonator_spectroscopy` → `03_qubit_spectroscopy` → `04_rabi` →
`12_single_shot` → `12b_active_reset` (only needed before running anything with
`RESET_MODE = "active"`), then the JPA trio (`14a` coarse bias survey, then
`14` fine tuning, then `15` pump on), then the FWM sequence from
`18_fwm_power` onward — scripts 19–21, 23 and 24 read
`device.fwm.pump_power_fit`, so 18 has to run and `update()` first.

### JPA script order

`14a_jpa_bias_scan.py` (`JPABiasScan`, `experiments/jpa/jpa_bias_scan.py`) is
the coarse survey and **runs before `14_jpa_fine_tuning.py`, even though it
sorts after it** -- there was no free number below 14 in the JPA group, and
keeping the three JPA scripts adjacent won over strict sort order. It sweeps
the DC bias, taking a waste-resonance sweep at each value, and deliberately
leaves `rspec_update` off: it would only write back the bias it was handed, not
a measured optimum. `14` is what picks the operating point and writes
`hw.soc.dacs.jpa`.

## Readout length at constant SNR (script 12d)

`12d_readout_length_constant_snr.py` sweeps `readout_length` with
`gain = gain_cfg * sqrt(L_cfg / L)`, i.e. **gain² × length constant, not
gain × length**. QICK gain is amplitude, so photon number ∝ gain²; with
L ≫ 1/κ (~0.2 us on qi=1) the g–e separation goes as gain and the noise as
1/sqrt(L), so SNR ∝ gain·sqrt(L). Holding gain·L instead would give
SNR ∝ 1/sqrt(L) and quadruple the power per halving of L: from 7.5 us that
reaches gain 0.15 at 2 us, where run 34 `#188` already shows readout-induced
transitions (outside fraction 2.8% at 0.14, 12% at 0.23) and >30% response
compression. Constant gain²·L reaches 0.11 at 1 us and 0.155 at 0.5 us, so
expect the short end to depart from constant SNR anyway; the SNR panel is
the check, not an assumption.

NEP ∝ sqrt(Γ_dark)/(η·duty), with Γ = p_dark/cycle and duty = W/cycle, is
sqrt(p_dark·cycle)/(η·W) at fixed window W. Cycle = W + L + reset tail
(`effective_final_delay`, so active reset's per-round readouts are counted).
Two bounds are reported: dark = readout P(e|g) (the readout's own effect on
the dark count), and dark fixed (pump-limited, so only cycle and η move).
η is 1 − P(g|e); thresholds are the histogram's fidelity-max ones.

## NEP-optimal readout (script 12e)

`12e_readout_nep_optimization.py` is the 3D version of 12d. It sweeps frequency ×
length × gain and minimizes the same fixed-window NEP, scoring the **g–f
(shelved) readout**: dark = P(f-side|g) + `P_PUMP_EXCESS`, η = 1 − P(g|f).
Both are counted at `thresholds[1]` on the analysis's g→f rotated axis.
`P_PUMP_EXCESS = 0.006` is the mean p_p − p_g at zero buffer gain in run 34
`#519/#520/#522` (p_g 0.034–0.039 against p_p 0.035–0.048). The pump adds
little, so the readout's own false positives and residual excitation are most
of the dark count, which is why this optimizes NEP and not fidelity.

- Grid. Gain steps are half the length steps, so the constant-gain²·L line
  through the config runs through grid nodes, and 12d's sweep is the diagonal
  of each frequency slice.
  - `#613`, the ~4 h wide scan: length = L_cfg·2^(k/6), k = −9..9 (2.5–20
    us); gain = g_cfg·2^(j/12), j = −12..12 (×2 either side); 21
    frequencies over 2κ. 9975 points.
  - A narrow scan around `#613`'s optimum (~40 min + waits) is `RATIO`
    2^(1/12), k and j −6..6 (length ×1.41, gain ×1.19 either side), 9
    frequencies over 0.75κ (74 kHz steps). 1521 points. In `#613` the cells
    within 3% of the optimum spanned the whole 2κ in frequency, 7–11 us and
    gain 0.05–0.09, so the surface is flat there and the finer step is
    warranted.
  - **The grid is relative to the config, which 12e itself rewrites.** The
    same k range is not the same range twice: `#613` ran 2.5–20 us from
    `L_cfg` = 7.07, and with `L_cfg` now 8.909 it spans 3.15–25.2 us. Compare
    runs by the saved `readout_nep_grid.json`, not by the step constants.
- Timing, from `#608`: 0.33 s + 3 × shots × (L + reset tail) per point. The
  tail is length-independent only for `f0g1`/`passive`; under `active`/`while`
  it carries `readout_length` once per round, so the estimate recomputes it
  per length. Broad grid at 10 k shots: ~2.9 h f0g1, ~4.1 h while (the while
  tail runs 11.7 us at 3.1 us of readout to 55.8 us at 25.2 us).
- **Raw shots are saved**, one h5 per point via `save_data` (reference
  points too, prefix `readout_nep_reference_qubit*`). The metrics are stored
  next to the shots, and the CSV `file` column names each h5. ~0.95 MB per
  point at 20k shots, ~1.5 GB for the narrow grid.
  - `#613` saved only per-point histograms (`readout_nep_histograms.h5`),
    because raw shots for 9975 points would have been ~11 GB. Its CSV `file`
    column is blank.
- **Ground-state check, as in scripts 26/37/40** (`fwm_qc.make_ground_check` /
  `wait_for_ground`, CLAUDE.md "Ground-state recovery"). It looks at |g>-only
  shots at the *config* readout, so it sees both a hot qubit and a JPA losing
  gain.
  - Before the first point it blocks until one pass.
  - Every `REF_EVERY` points it measures the reference and takes one ground
    look. The check fails if the ground look is ≥ `GROUND_MAX_PE`, or if the
    reference is more than `Z_FLAG` σ from the median of the references that
    passed so far (after `MIN_GOOD_REFS`).
  - On a failure, the run waits for the full `GROUND_CONSECUTIVE` streak, then
    retakes that block's points. Each point is retaken at most
    `MAX_RETAKES` times, after which it is kept and left flagged.
  - Waits are logged to `ground_waits.csv`.
  - `#613` had no such check: 83 of 201 references failed, with bursts of
    P(f|g) doubling, and SNR dropping as far as 2.6 against 5.7 normally.
- **Tolerating instabilities.** Points run in random order, so a drift episode
  lands on scattered points instead of a contiguous block that looks like
  structure.
  - Reference measurements are logged to `readout_nep_reference.csv`.
  - NEP is normalized by the median of the references that *passed*. Using
    all of them made `#613` read ~2.8% low.
  - Flags:
    - `unstable`: the reference that *closes* the point's block (the next
      one in time) is more than `Z_FLAG` σ off that session's median. Only
      the closing reference counts, because a failed check is followed by a
      wait for |g>, so what comes after it starts clean. A block that was
      retaken after its check is therefore not flagged.
    - `neighbour`: more than `Z_FLAG`·hypot(own error, local MAD) off its
      3×3×3 neighbour median. These can be real sharp features (edge cases) as
      well as glitches. Interior cells only; on the boundary the neighbourhood
      is one-sided, and in testing it flagged every steep edge.
  - A failed acquisition (Pyro hiccup) is skipped and left missing; three in a
    row stops the run.
  - CSVs are appended and flushed per point, so a killed run is a complete
    folder to `--rescan` with `--select missing`.
- **`--rescan RUN`** (folder path or `#NNN`) re-acquires `--select`
  {missing, flagged, unstable, all} of an earlier run into a new folder,
  optionally restricted to a box (`--frequency/--length/--gain LO HI`).
  - It uses that run's `readout_nep_grid.json` (grid, reference, shots,
    window, `P_PUMP_EXCESS`), not the current config, so the merged data stay
    comparable.
  - The merged `readout_nep.csv` keeps the last row per point. Its `source`
    and `file` columns name the folder and h5 that hold the point. Replaced
    rows, including in-run retakes, go to `readout_nep_replaced.csv`.
  - The per-session reference medians are printed so a shift between
    sessions is visible.
  - `#608` predates the grid json and cannot be rescanned.
- **The optimum is the argmin of the neighbour-median map** (3×3×3 median
  including self, unstable points excluded), not of the raw points. `#608`'s
  raw argmin was a single cell 5% below a flat ~1.0 neighbourhood, and was
  written to the config (7780.3868 MHz, 0.06, 7.0711 us).
- Plots:
  - Corner plot (`readout_nep_corner.png`): 2D panels show the *best over the
    hidden axis* of the median map, not a slice, so they answer "best
    achievable here". The diagonal adds the slice through the optimum and the
    raw points on it.
  - The constant-gain²·L line is on the (length, gain) panel.
  - `readout_nep_maps.png`: raw SNR, P(f|g), P(g|f) and NEP over (length,
    gain) at the optimum frequency.
  - `readout_nep_flags.png`: reference NEP vs time, and where the flagged
    points sit.
- Writes frequency/gain/length to the config, then reruns a `check_f`
  histogram there and calls `.update()`, unless `--no-update-config`.
- Why not script 22 (`SingleShotOptExperiment`): it maximizes fidelity and
  keeps no per-point P(e|g)/P(g|f) or cycle time. Its grids are linear, and it
  caps gain at `device.qubit.max_gain`. Script 22 also calls `update()` even
  with `--no-update-config`. Left unchanged as the fidelity optimizer.

## Reset modes

`cfg.expt.reset_mode` selects which reset runs between shots. Resolved in one
place, `reset_mode_of()` in
[qick_program.py](../../experiments/general/qick_program.py), which
`QickProgram.measure`, `QickExperiment._get_final_delay` and
`common.effective_final_delay` all read — they are the only three consumers.

| mode | what happens after the readout | inter-shot tail (run 33 Waymar) |
|---|---|---|
| `passive` | nothing; the full `device.readout.final_delay` T1 wait | ~313–490 us |
| `active` | `QickProgram.reset` — up to `reset` conditional pi pulses, each on its own fresh readout | 19.0 us |
| `f0g1` | `QickProgram.reset_f0g1` — one fixed unconditional drive of both tones | 12.8 us |
| `while` | `QickProgram.reset_while` — loops until a read says ground, capped at `reset` rounds; see below | 23.2 us (bfg34 qi=1, mean) |

A script picks only the mode. Every parameter comes from the config (next
section):

```python
RESET_MODE = "f0g1"     # "passive" | "active" | "f0g1" | "while"
...
params = {..., "reset_mode": RESET_MODE}
```

**Calibrate in the mode you measure in.** `12_single_shot.py` sets the
threshold, fidelity, `g_mean` and `sigma` that every downstream experiment
discriminates against, and on run 33 the f0-g1 drive shifted the |g> blob by
several percent. Calibrating passive and then running f0g1 measures one readout
and uses another.

### Reset parameters live in the config

| key | used by | missing / null / NaN means |
|---|---|---|
| `device.readout.reset[qi]` | active | (required) number of rounds |
| `device.readout.reset_threshold[qi]` | active | `device.readout.threshold[qi]` |
| `device.readout.reset_ringdown[qi]` | active, f0g1 | `read_wait + extra_delay = 0.1 + 0.2 + 1.55/kappa` |
| `device.f0g1.reset_gain` | f0g1 | `device.qubit.max_gain` (full scale) |
| `device.f0g1.reset_gain_ef` | f0g1 | the Fig. 3a optimum at that gain's g~ |
| `device.f0g1.reset_length` | f0g1 | derived from `reset_residual` at the calibrated Gamma |
| `device.f0g1.reset_residual` | f0g1 | 0.01 |
| `device.f0g1.reset_gap` | f0g1 | `3 / kappa` |

`device.f0g1` is one device, so its keys are scalars; the `device.readout`
ones are per line. The `f0g1` keys are all null on bfg34 -- the right values
are not known yet, so everything is derived.

**Precedence: a script's `params` > the config > derived**, resolved in
`QickExperiment.configure_reset`. A `None` in `params` falls through to the
config. Only three kinds of script pass reset parameters at all:

- the reset measurements (`12b`, `35`, `35b`, `36`), which sweep them -- that
  is the measurement, not an operating point;
- `26` and `37`, which shorten the gap and ring-down (see below);
- nothing else. To change an operating point, edit the config.

### Why `f0g1` is worth choosing, beyond speed

- **It clears |f>.** The conditional reset cannot: `chi >> kappa` puts the e
  and f blobs ~0.5 sigma apart, so no threshold separates them, which is why
  `reset_f` has to walk `|f> -> |e> -> |g>` over two rounds and why the note
  below about `check_f` existed. The f0-g1 drive removes |f> natively.
- **It emits no extra readouts.** `reset()` appends `reset - 1` conditional
  readouts per shot — 13.1 us of the 19.0 above is readout. `reset_f0g1`
  appends none, so `n_reads` stays 1 and nothing downstream that indexes the
  trailing read axis changes.
- **Fixed duration**, so the FWM scan table's equal-time invariant holds
  automatically. The rule to keep: never gate the reset on a per-scan switch
  (`do_pump`, `do_pulse_f`) or vary its length per scan.

### The `f0g1` drive is derived

`configure_reset` calls `helpers.f0g1.f0g1_reset_params`, which reads
`device.f0g1` and follows `ResetTimeScan`'s own derivation: f0-g1 tone at full
scale (g~ is below the `kappa/3` plateau, so more coupling is strictly better),
`Omega_ef` from the Fig. 3a optimum at that g~, both tones on their
Stark-shifted resonances. It raises naming the script to run (28 / 30 / 34) if
a fit is missing, rather than proceeding with a NaN. The two drive
*frequencies* are never set by hand: they track the gains through `stark_fit`
and `ef_stark_fit`.

The length is the setting that matters (run 33 numbers):

| `device.f0g1` | drive | tail |
|---|---|---|
| — (default, 1% residual) | 6.10 us | 12.8 us |
| `reset_residual: 1e-3` | 9.15 us | 15.9 us |
| `reset_residual: 1e-4` | 12.21 us | 18.9 us |
| `reset_length: 40.0` | 40.0 us | 46.7 us |

`reset_residual` keeps the length *derived* — it rescales on its own as
Gamma improves — while `reset_length` pins it, and then `reset_residual` is
ignored. The run log prints which values were derived and which were handed
in, so it records what the run actually did. Section 8 of
`check_reset_modes.py` tests the precedence.

### Choosing the f0g1 gains

**`reset_gain` (the f0-g1 tone).** Full scale is the right default while g~ is
below the `kappa/3` plateau, but Gamma saturates, so lowering it is close to
free. On run 34 the plateau starts at `g~ = 0.401 MHz` (`g~ >= 0.28*kappa`),
i.e. f0-g1 gain 0.458:

| gain | 1.00 | 0.70 | 0.50 | 0.35 | 0.25 | 0.15 |
|---|---|---|---|---|---|---|
| Gamma, % of kappa/3 | 100 | 100 | 100 | 39 | 18 | 7 |

Gain 0.46 is 2.2x less amplitude and 4.8x less power for the same rate. That
matters beyond headroom: `35b_reset_floor` measured the f0-g1 tone adding
~2.7% to the |g> floor at full scale (the e-f tone added nothing), and in the
FWM scripts that floor is the dark count the NEP is computed from.

Full scale is not the fastest either. Eq. 1 gives the time to a 1% residual as
1.55 us at gain 1.0, **1.02 us at 0.74** (the fastest -- the onset delay beats
the marginally lower rate, and it deposits ~2.8x less energy), 2.21 us at 0.5,
2.95 us at 0.46, then 7.5 us at 0.35 and 15 us at 0.25.

Two things `reset_gain` drags along:

- `reset_gain_ef` is derived from g~, so a lower f0-g1 gain lowers the e-f gain
  too. Pin `reset_gain_ef` to move only one.
- A pinned `reset_length` does not grow to compensate a weaker drive. If it is
  shorter than the time above, the reset simply does not finish.

**`reset_gain_ef` (the e-f tone)** is the setting most worth scanning. The
derived Fig. 3a optimum can sit far below the range `30_ef_rabi_vs_power` could
measure, so it is an extrapolation, and Gamma is steeply asymmetric about it
(run 33, fraction of the best Gamma):

| gain_ef | 0.002 | 0.003 | 0.0046 | 0.006 | 0.008 | 0.012 | 0.020 |
|---|---|---|---|---|---|---|---|
| Gamma | 10% | 24% | 100% | 96% | 89% | 77% | 57% |

Erring high is much safer than low, so scan upward from the derived value.

### Gap and ring-down in scripts 26/37

Both scripts pass `f0g1_reset_gap: 0.0` and `extra_delay: 1.0 - 0.1` instead
of the config's safe defaults.

- **Gap 0** is safe *there specifically*: the reset is followed by
  `final_delay` (1 us) and then the next shot's whole buffer window
  (`length_b`) before anything is read; at 25 us that is ~236 photon lifetimes
  even with no gap. The default
  `3/kappa` (2.13 us, ~16 lifetimes on run 34) only duplicates a wait that
  already exists. Do **not** copy it into a sequence where the readout closely
  follows the reset: script 12's next readout is ~1 us later, only ~8
  lifetimes.
- **Ring-down** is `read_wait + extra_delay`, the wait for the readout
  resonator to empty before the reset; a leftover photon would Stark-shift the
  f0-g1 transition out from under the tone. The default
  `0.1 + (0.2 + 1.55/kappa) = 1.58 us` is 12 photon lifetimes; 1.0 us is 7.6,
  suppressing leftover photons by 5e-4. That is ample: even 100 readout photons
  leave 0.05 behind, and that Stark shift is far inside a linewidth. It also
  shortens the per-round ring-down of the `active` reset, which reads the same
  two keys -- same physics, same margin. To make it the default everywhere,
  set `device.readout.reset_ringdown` instead.

`f0g1_reset_gap`, the photon-escape window, defaults to `3 / kappa` with kappa
in MHz — that is `2*pi*3/kappa_ang`, i.e. ~19 photon lifetimes on run 33,
generous. It is 4.05 us of the 12.8 above, so trimming it is the cheapest
saving available if the tail ever matters more than the margin. The formula is
inherited from the `f0g1_reset.py` classes and is left alone for consistency
with the data already taken through them.

### The migration off `active_reset`

`reset_mode` is authoritative; `active_reset` is honoured as the legacy
spelling when `reset_mode` is absent. **Every class a numbered script uses is
migrated:** the base classes, `single_shot`, `single_shot_opt`, `fwm_shots`,
`FWM_sweeps`, `qubit_spec_shot`, `f0g1_reset`, `t1`, `t2`, `rabi`,
`pulse_probe_spectroscopy` (`QubitSpec`), `resonator_spectroscopy`
(`ResSpec`), `stark_spectroscopy` and the `jpa/` classes. Still legacy, and
working through the fallback: the `flux/` and `two_qubit/` families, `rb`,
`t1_cont`, `t1_stark*`, `t2_ramsey_stark`, `photon_number_splitting`,
`active_reset.py` and `qubit_buffer_spec_shot`.

Migrated classes default to `"passive"`. **That changed the default of `T1`,
`T2` and `Rabi`**, which used to take `device.readout.active_reset[qi]` --
`true` for Waymar qi=1 -- so a notebook call that names no mode now runs
passive. It also fixed `16_temperature.py`, whose `RabiExperiment` (built in
`calib/measure_func.py`, with `go=True`) had been running active reset on
qi=1 despite measuring residual excited population. `device.readout.active_reset`
is now read only by the legacy classes.

**Passing both keys with different meanings raises.** A script passing
`active_reset=True` to a migrated class (which defaults `reset_mode` to
`"passive"`) would otherwise have its reset silently turned off.
`reset_mode_of` refuses instead: a stopped run is far cheaper than a run whose
qubit was never reset. `verify_active_reset` picks the spelling per class from
its own `MIGRATED` set.

`16_temperature.py` deliberately has no `RESET_MODE`: residual excited
population *is* what it measures. `01`, `14`, `14a` and `15` have none either
(no qubit state involved).

### How the branch works, and the one way it can go wrong

`read_and_jump(test="<", label=NOPULSE)` compiles to
`cond_jump(arg1=I, op='-', arg2=threshold, test='S')` — **jump over the pi pulse
when `I < threshold`**. So the pulse fires on `I >= threshold`, which resets the
qubit only while `g_mean < threshold < e_mean`. Both bfg33 configs satisfy this
(Waymar `-11.60 < -3.89 < 16.29`, Yuvi `-3.87 < -1.23 < 2.53`).

If that ordering ever inverts — a flipped readout phase, a different LO sideband
— reset would pi-pulse the *ground* state on every shot, which is far worse than
no reset. There is deliberately **no guard for this in `configure_reset()`**:
threshold and phase belong to `12_single_shot.py`, and a check here would paper
over a readout that needs recalibrating. `12b_active_reset.py` prints the
ordering, and `verify_active_reset.py` asserts it offline.

### `setup_reset`, the staging mode

`setup_reset=True` runs the reset sequence *and* keeps the long passive delay, so
the shots are still a valid readout calibration even if the reset does nothing.
That is how `12b` measures whether reset works before anything depends on it.
All three cases live in `QickExperiment._get_final_delay()`.

### Calibrating it (`12b_active_reset.py`)

The script form of the "# Active reset" section of
`notebooks/SMPDs/tune_qubits_smpd_v3waymar.ipynb`. Baseline with `setup_reset`,
then a `threshold_v` sweep over +/-2 sigma, then a `final_delay` sweep at
`reset=1`, then `readout_helpers.plot_reset`. It writes `reset_e` / `reset_g`
(`reset_e << 1` means the excited population is gone; `reset_g ~ 1` means reset
is not exciting the ground state) and **reports** the best threshold rather than
writing it — `12_single_shot.py` owns that value, and overwriting it would
change what every other experiment discriminates against.

Run it per device, not per cooldown. Yuvi's readout fidelity is 0.40 against
Waymar's 0.72, so expect the two to want different `RESET_ROUNDS`.

### What `reset = N` actually does

Not N pi pulses in a row. Each round takes its **own fresh readout**, so each
pi pulse is a new decision:

```
measure()                            M0   <- the real measurement, readout index 0
round 0:  branch on M0 -> maybe pi ; measure M1   (index 1)
round 1:  branch on M1 -> maybe pi ; measure M2   (index 2)
round 2:  branch on M2 -> maybe pi ; (no re-measure)
```

So N rounds give N-1 extra readouts, and a typical excited shot spends exactly
one pi pulse: M0 reads excited, the pulse fires, M1 reads ground, rounds 1 and
2 skip. Extra rounds are **retries**, not repetition — they exist to catch a
pi pulse that failed, a re-excitation, or a misread.

The last round is not re-measured **in use**: nothing acts on a readout taken
after the final pulse, so it would only cost time. `cfg.expt.reset_verify`
adds it back, which is what a *calibration* needs — without it the last
recorded round is the state before the final conditional pulse, i.e. one
round stale. `12b_active_reset.py` sets it; production runs should not.
(A while-loop reset that can stop early needs it always — see below.)

**The last round sets the floor.** Its pi pulse is unverified, so a ground
shot misread as excited there gets a spurious pi pulse and the shot ends
*excited*, with nothing to catch it. That probability is just the g blob's
tail above the threshold:

| | thr above g | P(ground read excited) | P(excited read ground) |
|---|---|---|---|
| Waymar qi=1 | 1.60 sigma | 5.5 % | ~0 % |
| Yuvi qi=1 | 0.58 sigma | 28 % | 20 % |

`device.readout.threshold` is the *fidelity*-optimal value, pushed down
towards g to catch the e blob's T1-decay tail. For reset that is the wrong
direction — it over-calls excited. Adding rounds does not fix this, because
the floor is set by the final round whatever N is. On Yuvi, with 28 % either
way, reset may not be usable at all until the readout improves. (Gaussian
estimate; the e-blob figure is optimistic because it ignores the decay tail,
but the g-blob one governs the floor and is sound.)

### `device.readout.reset_threshold`

Because of the above the two thresholds are separate config fields.
`QickExperiment.reset_threshold(qi)` returns `reset_threshold` when it is set
and falls back to `threshold` when the key is missing or NaN, so old configs
keep working and only the qubits you have calibrated carry a value.
`configure_reset()` uses it for `threshold_v`; an explicit `threshold_v` in a
script's `params` still overrides both.

Why they differ, in one line each:

- `threshold` minimises `P(g→e) + P(e→g)` with equal weight — the right
  objective for *labelling* a shot, and it asks about the past ("what state
  was prepared?"), so it wants the decayed shots counted as excited.
- reset is dominated by `P(g→e)` alone, which fires a pi pulse on a ground
  shot and *creates* excitation, re-injected every round. It asks about the
  present ("is it excited now, as the pulse lands?"), so a shot that decayed
  mid-readout must *not* be pulsed.

Same decay tail, opposite sign. On run 33 Waymar the optimum moves ~1.4σ, from
−2.21 to about +4.4; measured from `#198`, the `|e>` population left after
reset goes 18.2 % → 10.0 %.

`12b_active_reset.py` writes it. Its figure of merit is the **absolute** `|e>`
population after the last round, not that minus the `|g>` floor: a reset
threshold near or below `g_mean` pi-pulses nearly every shot and drives both
preparations to the same high value, so the difference collapses to ~1 % while
both sit at ~42 %. The floor is still reported and plotted, since `|e>` minus
floor estimates what is really left behind — useful to read, unsafe to
optimise.

**The integer threshold is normalized by the readout length actually played.**
`configure_reset` converts volts to the tProc integer as
`threshold_v * readout_length * f_fabric`, and takes `readout_length` from
`cfg.expt`, falling back to the config. It used to read
`device.readout.readout_length[qi]` unconditionally, which is the same number
for every script that does not sweep it — but `12d` and `12e` do. The
accumulated I scales with the integration length, so over 12e's 8x length range
the branch would have fired against a threshold off by up to 2.8x: resetting
*every* shot at the long end and *no* shot at the short end, with nothing in
the saved shots to show for it. Case 5 of `check_reset_threshold.py` pins the
scaling.

Check the fallback without hardware:

```
python check_reset_threshold.py     # absent / NaN / set / override / length scaling
```

Those extra readouts sit on the trailing axis of `collect_shots`, index 0 being
the real measurement. Everything that consumes shots already selects index 0:
`DataProcessor.process_iq_data`, `make_hist`, and `QickExperiment.add_p_e`
(`read_index=0`). The rounds themselves are kept only where a histogram is
saved — `Igr`/`Qgr` in `HistogramExperiment`, `I<label>r`/`Q<label>r` in
`SMPDHistogramExperiment` via `fwm_shots._reset_rounds`.

**Duty cycle.** The inter-shot tail is no longer `final_delay`, so anything
quoting a cycle time must use `common.effective_final_delay(cfg.expt)`. On
Waymar that is 335 -> 19 us, taking script 26's duty cycle from 0.081 to 0.540;
reading `final_delay` directly there would have overstated the NEP sevenfold.

**`check_f` and the `active` reset do not compose.** The `f` scan leaves
population in |f>, which a `pi_ge`-only reset conditioned on the g/e threshold
will not clear. `reset_f` is the partial answer (see below); `reset_mode =
"f0g1"` is the real one, since the f0-g1 drive removes |f> natively.

A related bug was live until this was wired up: `fwm_shots.py`'s `params_def`
set `"reset_f": check_f` from the **`__init__` keyword**, not the merged
`params` value. Every runner script passes `check_f` inside `params`, so on
every real run `check_f=True` but `reset_f=False` — the `f` scan ran and its
|f> population was never cleared. Fixed to `params.get("check_f", check_f)`.

### Resetting out of |f> (`reset_f`)

Sequences that end with a pi_ef -- the FWM `f` scan, `single_shot` with
`check_f` -- can leave population in |f>, which a pi_ge-only reset does not
touch. `cfg.expt.reset_f` puts a pi_ef ahead of each pi_ge.

**It does not branch on e vs f, and cannot.** chi is large compared to kappa
here, so the resonator response saturates and the e and f blobs land on top of
each other -- 0.46 sigma apart on run 33 Waymar (medians 16.3 and 18.4 against
sigma 4.6). `analyze_single_shot_histograms` does report a `thresholds[2]`
"ef threshold", but at ~12 it sits below *both* blobs and separates nothing;
do not build on it. Since the only goal is |g>, assume |f> and walk down:

```
|f>  -> pi_ef -> |e> -> pi_ge -> |g>     one round
|e>  -> pi_ef -> |f> -> pi_ge -> |f>     fixed on the next round
|g>  -> (reads below threshold, skipped entirely)
```

So |f> clears immediately, |e> costs an extra round, and it converges either
way. **`reset_f` therefore wants `reset` >= 2** -- with `reset = 1` an |e>
shot ends in |f>, worse than not resetting.

It defaults to `check_f` in the two classes that have one
(`SMPDHistogramExperiment`, `HistogramExperiment`) and to `False` elsewhere.
Leaving it on for the non-f scans of a `fwm_shots` run is deliberate: it costs
a pulse but keeps every scan's timing identical, which that class depends on.

### Checking it without hardware

```
python check_reset_modes.py      # every mode: what it derives, what it costs
python verify_active_reset.py    # every class, both configs, every mode
python check_pi_ge.py            # pi_ge under the active reset, pi_ef under reset_f
python check_reset_threshold.py  # reset_threshold absent / NaN / set / override
```

`check_reset_modes.py` is the one to run after touching any of this. It asserts
the inter-shot tail per mode (script 26's NEP input), that the f0g1 tail is
*exactly* its own parts with no readout term, that an uncalibrated device
refuses f0g1 mode naming the script to run, and that the legacy `active_reset`
path still resolves the way it used to.

`check_pi_ge.py` matters because `QickProgram.reset()`/`reset_while()` play
pulses by name — `pi_ge` always and `pi_ef` under `reset_f` — and most Programs
declared them only under `checkEF` / `pulse_e` / `pulse_f` / `checkFH`, so
turning reset on would have faulted on an undefined pulse name.

**The guards must go through `reset_plays_pi_pulses(cfg.expt)`**, not
`reset_mode_of(cfg.expt) == "active"`. The equality test reads as correct and
is not: it excludes `reset_mode = "while"`, which plays `pi_ge` through
`reset_while`. That is what faulted script 26 with *"trying to play pulse
pi_ge, but it hasn't been defined"* — after the sweep had started, since the
pulse table is built at compile time. The checker deliberately **rejects** the
equality spelling, so the gap cannot come back; legacy classes
(`qubit_buffer_spec_shot`, `rb`) still read `cfg.expt.active_reset` and are
accepted on that spelling. It walks the AST rather than
building the Programs (which needs real hardware), understands `if`/`elif`
chains, and checks the *declaration* sites only — widening a `_body` guard by
mistake would inject a state-preparation pulse into the sequence.

### The while-loop version (`reset_mode = "while"`)

Reads out until the qubit is actually in |g>, with no fixed round count:
`QickProgram.reset_while(max_rounds)`. The sequence is
*measure -> while excited -> reset + measure*, capped at
`device.readout.reset[qi]` rounds -- uncapped, a qubit that never reads
ground hangs the tProc and the board needs a reset, so the cap is not
optional.

**Two logical readouts on one physical input.** QICK sizes buffer reads by
counting trigger instructions *statically*, so a variable count per shot
silently misaligns the data (confirmed by the QICK authors). The loop
therefore triggers a dedicated feedback readout whose buffer is discarded,
while the measurement keeps exactly one trigger per rep and averages
correctly. That needs `hw.soc.adcs.readout.feedback_ch[qi]`: another readout
index on the same ADC -- ADC5's PFB bank, indices 2-7 (8-9 have
`tproc_in = -1` and cannot feed `read_input`). `00_soc_map.py` prints the
map and the per-readout `avg_maxlen`, which bounds shots x reads-per-shot
and so is the tightest constraint on `shots` in this mode.

**The loop runs for an unknown time**, so `resync(0.05)` pushes the reference
time past the execution time after each pass; without it the later timed
instructions pile up. This is the sanctioned fix for a data-dependent loop.

Three things that are *not* like the fixed-N reset:

- **Every round is verified**, since the loop acts on its last readout. The
  `if n < i - 1` guard has no analogue here.
- **No reset rounds reach `collect_shots`.** They land on the feedback
  channel, which is declared *after* the measurement, so `iq_list[0]` is
  still the measurement and `[..., 0]` still selects it. Nothing downstream
  changed, and `_reset_rounds` returns None because there is no trailing
  read axis to slice.
- **The tail is a distribution, not a number.** `effective_final_delay`
  quotes the mean, `base + (read_wait + extra_delay + readout_length) *
  (1 + reset_while_typical)`; run 34 measured 1.0 rounds from |e> and 0.01
  from |g>, hence `reset_while_typical = 1`. Note the tail *contains*
  `readout_length`, so it grows with a swept length -- 11.7 us at 3.1 us
  against 55.8 us at 25.2 us on bfg34 qi=1.

Validated on run 34: `#662`/`#665` reset both |g> and |e> preparations to the
ground mean at alpha = 0.4-1.0%, and `42_reset_while_loop.py` is the
self-contained version that measures the round-count distribution.

**Every `_body` that dispatches reset itself needs the `while` branch.**
`HistogramProgram` has its own dispatch covering only `active`/`f0g1`, so the
feedback channel was declared and never triggered -- a declared readout that
never fires leaves the averager waiting for reads that never arrive, which
surfaces as *"exception in readout loop"* on the first poll, not as anything
naming the reset. Programs that call `super().measure(cfg)` (`fwm_shots` and
most others) get all four modes for free.

`QickProgram.cond_reset` remains the abandoned first attempt -- both its
`label()` calls are emitted at the end so the second branch falls through,
and its docstring says "Not tested". Dead code, not a starting point.

## Conventions specific to this tree

- Line-index vs qubit-index: scripts name them explicitly
  (`buffer_line_qi`, `waste_line_qi`, `stark_line_qi`, `readout_line_qi`) rather
  than reusing `qi`. Keep that habit; it's the main source of confusion here.
- Every param a Program touches must exist in `cfg.expt` before it runs — see
  the root [CLAUDE.md](../../CLAUDE.md) rule about `params_def` and no
  `cfg.expt.get(...)`.
- `check_params=False` on inner 1D experiments of a 2DSimple wrapper is
  deliberate: the outer class passes extra y-sweep keys the inner one doesn't
  know about.
- Nameserver must be running before any script (`start_nameserver.bat`).
  Note `--port` defaults to 9090 here, unlike the 8888 in the root CLAUDE.md
  notebook snippet.
- Plots: no titles (root rule). `finish_experiment` keeps windows open; pass
  `--no-show` for unattended runs.
- **Keep comments and docstrings short.** The scripts were already cleaned up
  once for over-long ones; don't regrow them. A comment is one line saying
  *what* or flagging a trap, plus a pointer (`CLAUDE.md "<section>"`) when
  there is more to say. The *why* — derivations, measured numbers, history,
  rejected alternatives — goes in this file, not in the code. Script
  docstrings are one line (it becomes `--help`).
