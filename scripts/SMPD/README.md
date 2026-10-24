# SMPD script runner

The numbered scripts are the script-based equivalent of the succession in
`notebooks/SMPDs/tests_smpd_v2_waymar_rfboard.ipynb`.

Start the nameserver, then run from the repository root — no flags needed:

```powershell
python scripts/SMPD/10_t1.py --include-prerequisites
```

**Which device a script runs against is the `CONFIG` constant at the top of
that script**, e.g.

```python
if __name__ == "__main__":
    CONFIG = "2026_09_21_bfg33_smpd_v2_yuvi_rfboard.yml"
```

The run number and device name are read out of that filename, so editing that
one line switches the hardware *and* the output folder together. `common.py`
holds no device default and is never edited to change device. For a one-off run
against the other device without editing the file, pass
`--config 2026_09_21_bfg33_smpd_v2_waymar_rfboard.yml`.

The run is saved next to the runner in `scripts/SMPD/Data/run33_Yuvi/`
(or `run33_Waymar/`).

Each experiment creates a matching numbered folder under both parents:

- `data/#1_time_of_flight_q1_133556/*.h5` contains acquired and analyzed data.
- `metadata_and_images/#1_time_of_flight_q1_133556/*.png` contains plots, and the YAML configuration is stored there too.
- The executed script is copied into the matching `metadata_and_images/#1_time_of_flight_q1_133556/` folder.

The number resets automatically for each run-number / device-name combination,
i.e. for each config file. The number is followed by the experiment name and
start time.

Run individual stages directly, for example `01_tof.py`,
`02_resonator_spectroscopy.py`, `03_qubit_spectroscopy.py`, `04_rabi.py`,
`10_t1.py`, `11_t2.py`, `11b_t2_echo.py`, `12_single_shot.py`, or
`13_ef_spectroscopy.py`, `13b_ef_rabi.py`, or `13c_ef_ramsey.py`. Use `--update-config` to enable the notebook's
fit-to-config updates. Plots open through Matplotlib's default backend and
are still saved. Use `--no-show` for unattended runs.

Additional calibration protocols are available as:

- `17_stark_spectroscopy.py`
- `18_fwm_power.py`
- `19_fwm_buffer.py`
- `20_fwm_buffer_gain_length.py`
- `21_fwm_histogram.py`
- `22_readout_optimization.py`
- `23_fwm_efficiency_vs_power.py`
- `24_fwm_histogram_vs_power.py`
- `25_chi.py`
- `26_fwm_efficiency_vs_power_checked.py`
- `37_fwm_efficiency_vs_buffer_frequency.py`

`26_fwm_efficiency_vs_power_checked.py` is `23` with quality control: it checks
all five scans of every point before moving on and retakes the point (up to 3
attempts) if any of them looks wrong, then writes an efficiency / dark count /
noise-equivalent n̄ / NEP summary at the end.

Plotting is opt-in with `--plots`:

| | |
|---|---|
| `--plots none` (default) | only the end-of-run NEP summary figure |
| `--plots rejects` | plus a histogram figure per rejected attempt |
| `--plots all` | plus the full per-point figure for every kept point, as `23` draws |

`none` costs no information: the histograms of every rejected attempt are saved
to `qc_rejects.h5` either way, so the failure figures can be drawn afterwards
with the same function that would have drawn them live.

The gates live in `fwm_qc.py`, which also runs standalone to check a saved run,
to recalibrate the tolerances after any change to the readout, and to replot
rejects:

```
python fwm_qc.py --data-dir Data/run33_Waymar/data/#138_FWM_efficiency_vs_power_q1_091826
python fwm_qc.py --replot-rejects <run folder>/qc_rejects.h5
```

Both are read-only — they read saved files and need no nameserver. Pass
`--config` whenever the data is not from the device `fwm_qc.CONFIG` names: the
reference frame comes from the config, so a mismatched device reports a flood
of `width` failures that look like a readout problem and are not.

## Offline re-analyses

Scripts 17, 23 and 26 drop an analysis script into their own
`metadata_and_images/#<n>_.../` folder. Run it there, with no arguments — it
finds its own data and its own device config:

```powershell
cd Data/run33_Yuvi/metadata_and_images/#42_stark_spectroscopy_q1_101530
python refit_two_peak.py --gain-max 0.3     # buffer-gain cutoff for the n_bar fit
```

- `refit_two_peak.py` (from 17) — photon-number-splitting fit of a `two_peak`
  Stark run. `--gain-max` sets the buffer gain above which points are plotted
  but not fitted. Writes `device.fwm.buffer_power_fit`; pass
  `--no-update-config` to skip that.
- `fit_shared_means.py` (from 23 / 26) — shared-level fit of the FWM shot
  histograms, and the authority on excited fractions.

The canonical versions live in `analysis/`; edit them there and the next run
picks them up.

JPA operation is split into three scripts, run in this order:

- `14a_jpa_bias_scan.py` — coarse survey. Sweeps the JPA DC bias and takes a
  waste-resonance sweep at each one, so you can see where the JPA turns on.
  Writes nothing to the YAML; edit `DC_START` / `DC_STOP` at the top to set the
  range. **Runs before `14`, despite sorting after it.**
- `14_jpa_fine_tuning.py` — 2D pump-gain x DC-bias optimisation. Writes the best
  pump gain and DC bias into the YAML.
- `15_jpa_on.py` — reads those values back and starts the periodic pump tone.