#%%
cfg_file='sample_50_rfboard.yml' # Configuration file name
expt_path = 'C:\\_Data\\sample_50\\' # Experiment data path

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

np.set_printoptions(legacy="1.25")
from qick import QickConfig

from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
import slab_qick_calib.experiments as meas
from slab_qick_calib.calib import qubit_tuning, measure_func
from slab_qick_calib.calib.time_tracking import time_tracking
from slab_qick_calib.analysis import qubit_params 
from slab_qick_calib.helpers import rfboard, qick_check, config, handy

# %load_ext autoreload
# %autoreload 2

# Set color palette and font size
handy.config_figs()
import qick
import os

configs_dir = os.path.join(os.getcwd(),'../', 'configs')

cfg_file_path = os.path.join(configs_dir, cfg_file)
images_dir = os.path.join(expt_path, 'images')
summary_dir = os.path.join(images_dir, 'summary')
# Results config file
cfg_path = os.path.join(os.getcwd(),'..', 'configs', cfg_file)
auto_cfg = config.load(cfg_path)

# Connect to instruments
im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'])
print(im)

soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())

cfg_dict = {'soc': soccfg, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}
# raise Exception('stop')
# %%
print(soccfg)

# %%
update = True

# Per-qubit experiment params (RF attenuation/filter settings come from config)
qubit_list = [2]
flux = 0.
soc.rfb_set_bias(3, float(flux))
for qi in qubit_list:
    # Activate this qubit's RF settings from config
    auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)

    rspec = meas.ResSpec(cfg_dict, qi=qi, params={'span':'kappa','rounds':2})

    if update:
        rspec.update()
        auto_cfg = config.load(cfg_path)
# %%
update=False
flux = 0.
soc.rfb_set_bias(3, float(flux))

qubit_list = [2]
for qi in qubit_list: 
    filter_center_freq = 3400
    bw = 1050
    # set up filter and attenuation  
    rfboard.setup_qubit_drive_chain(qi, soc, auto_cfg, bw=bw, atten1_dac=30, atten2_dac=25, center_freq=filter_center_freq)
    start_freq = filter_center_freq - 100/2
    params={'start':start_freq, 
            'span':100, 
            'gain':0.5, 
            'expts':200, 
            'reps':1000, 
            'rounds':10, 
            'sep_readout': False, 
            'length':5, 
            'readout_length':10}
    
    qspec=meas.QubitSpec(cfg_dict, qi=qi, style='fine', params=params)
    #qspec=meas.QubitSpec(cfg_dict, qi=qi, style='fine')

    if update and qspec.status: 
        auto_cfg = config.update_qubit(cfg_path, 'f_ge', qspec.data["best_fit"][2], qi)
        auto_cfg = config.update_qubit(cfg_path, 'kappa',2*qspec.data["best_fit"][3], qi)
    elif update:
        print(f'Bad qubit! qi={qi}')
# %%

import time
qi = 2
fluxes = np.arange(0,0.2,0.05)
print(fluxes)
#%%

# ---- Configuration ----
from datetime import datetime
from scipy.optimize import curve_fit
import json

bias_chan = 3
flux_start, flux_stop = 0.07, 0.55
coarse_step = 0.05          # V, coarse flux step for scouting
fine_step = 0.005             # V, fine flux step
freq_margin = 1.5             # multiplier on model-predicted span per fine step
filter_start_freq = 3400
filter_bw = 1150
qspec_common = dict(gain=0.6, expts=201, reps=1000, rounds=10,
                    sep_readout=True, length=2, readout_length=4)
rspec_params = dict(span='kappa', expts=80)  # fewer points for speed

# Create unique subfolder for this run
run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
flux_sweep_dir = os.path.join(expt_path, f"{run_timestamp}_fluxsweep_Q{qi}")
os.makedirs(os.path.join(flux_sweep_dir, 'images'), exist_ok=True)
cfg_dict_flux = {**cfg_dict, 'expt_path': flux_sweep_dir}
print(f"Flux sweep data folder: {flux_sweep_dir}")

# ---- Step 1: Coarse sweep with dynamic tracking ----
coarse_fluxes = np.arange(flux_start, flux_stop + coarse_step/2, coarse_step)
coarse_qfreqs = []
coarse_rfreqs = []
coarse_freqs = []   # spectral data for color plot
coarse_amps = []
coarse_span = 150  # wide window for coarse scan
last_known_freq = None

print("=== Step 1: Coarse sweep ===")
for flux in tqdm(coarse_fluxes, desc="Coarse sweep"):
    soc.rfb_set_bias(bias_chan, float(flux))
    time.sleep(0.1)

    # Resonator (with retry)
    auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)
    for attempt in range(3):
        try:
            rspec = meas.ResSpec(cfg_dict_flux, qi=qi, params=rspec_params, display = True)
            rspec.update()
            break
        except Exception as e:
            print(f"  ResSpec failed (attempt {attempt+1}): {e}")
            soc.tproc.reset(); time.sleep(0.5)
    plt.close('all')
    auto_cfg = config.load(cfg_path)
    coarse_rfreqs.append(auto_cfg.device.readout.frequency[qi])

    # Qubit spec with tracking: center on last known freq if available
    search_center = last_known_freq if last_known_freq is not None else (filter_start_freq + coarse_span / 2)
    found_qubit = False
    max_shifts = 5  # shift down by half span each time before giving up

    for shift_attempt in range(max_shifts + 1):
        # Update RF filter to track the search window
        rfboard.setup_qubit_drive_chain(qi, soc, auto_cfg, bw=filter_bw,
            atten1_dac=30, atten2_dac=25, center_freq=float(search_center))
        start = search_center - coarse_span / 2
        params = {**qspec_common, 'start': start, 'span': coarse_span}

        # Try up to 3 times at this window (hardware + fit quality retries)
        good_fit = False
        for attempt in range(3):
            try:
                qspec = meas.QubitSpec(cfg_dict_flux, qi=qi, style='fine', params=params)
            except Exception as e:
                print(f"  QubitSpec exception (attempt {attempt+1}): {e}")
                soc.tproc.reset(); time.sleep(0.5)
                continue
            plt.close('all')

            # Check fit quality via parameter error
            fit_err = qspec.data.get('fit_err', None)
            if (qspec.data.get('best_fit') is not None
                    and fit_err is not None and fit_err < 0.2):
                found_freq = qspec.data['best_fit'][2]
                good_fit = True
                break
            elif attempt < 2:
                print(f"  Flux {flux:.4f}V: fit_err={fit_err}, retrying ({attempt+1}/3)")
                soc.tproc.reset(); time.sleep(0.3)

        if good_fit:
            found_qubit = True
            break

        # Bad or no fit after 3 tries — shift search window down by half span
        if shift_attempt < max_shifts:
            search_center -= coarse_span * 0.8
            print(f"  Flux {flux:.4f}V: fit not good enough, shifting search to {search_center:.1f} MHz")

    if found_qubit:
        last_known_freq = found_freq
        coarse_qfreqs.append(found_freq)
        print(f"  Flux {flux:.4f}V: qubit at {found_freq:.1f} MHz")
    else:
        coarse_qfreqs.append(np.nan)
        print(f"  Flux {flux:.4f}V: SKIPPED — no good fit found")

    coarse_freqs.append(qspec.data['xpts'].copy())
    coarse_amps.append(qspec.data['amps'].copy())

coarse_qfreqs = np.array(coarse_qfreqs)
coarse_rfreqs = np.array(coarse_rfreqs)
print(f"Coarse qubit freqs: {coarse_qfreqs}")

# Save coarse metadata
metadata = {
    'run_timestamp': run_timestamp,
    'qubit': qi,
    'bias_chan': bias_chan,
    'flux_start': flux_start,
    'flux_stop': flux_stop,
    'coarse_step': coarse_step,
    'fine_step': fine_step,
    'freq_margin': freq_margin,
    'filter_start_freq': filter_start_freq,
    'filter_bw': filter_bw,
    'qspec_common': qspec_common,
    'rspec_params': {k: str(v) if not isinstance(v, (int, float)) else v
                     for k, v in rspec_params.items()},
    'coarse_fluxes_V': coarse_fluxes.tolist(),
    'coarse_qfreqs_MHz': coarse_qfreqs.tolist(),
    'coarse_rfreqs_MHz': coarse_rfreqs.tolist(),
}
with open(os.path.join(flux_sweep_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved coarse metadata to {flux_sweep_dir}")

# %% ---- Step 2: Fit transmon model & plan fine windows ----
# coarse_fluxes = coarse_fluxes[:-2]
# coarse_qfreqs = coarse_qfreqs[:-2]
# coarse_rfreqs = coarse_rfreqs[:-2]

#%%
# SQUID transmon: f_01 = sqrt(8*E_J(Phi)*E_C) - E_C
# E_J(Phi) = E_J_Sigma * sqrt(cos^2(phi) + d^2*sin^2(phi))
# => f(V) = (f_max + E_C) * (cos^2(phi) + d^2*sin^2(phi))^(1/4) - E_C
# where phi = pi*(V - V_offset) / V_period, d = junction asymmetry
def transmon_flux(V, f_max, E_C, V_period, V_offset, d):
    phi = np.pi * (V - V_offset) / V_period
    return (f_max + E_C) * (np.cos(phi)**2 + d**2 * np.sin(phi)**2)**0.25 - E_C

# Filter out skipped (NaN) points for fitting
valid_mask = ~np.isnan(coarse_qfreqs)
valid_fluxes = coarse_fluxes[valid_mask]
valid_qfreqs = coarse_qfreqs[valid_mask]
print(f"Using {valid_mask.sum()}/{len(coarse_qfreqs)} valid coarse points for fit")

# Initial guesses from data
f_max_guess = np.max(valid_qfreqs)
E_C_guess = 200.0  # MHz, typical transmon charging energy
V_range = valid_fluxes[-1] - valid_fluxes[0]
V_period_guess = V_range * 2  # assume we see less than half a period
V_offset_guess = valid_fluxes[np.argmax(valid_qfreqs)]
d_guess = 0.1

p0 = [f_max_guess, E_C_guess, V_period_guess, V_offset_guess, d_guess]
bounds = ([0, 10, 0.01, -5, 0], [np.inf, 1000, 50, 5, 1])

try:
    popt, pcov = curve_fit(transmon_flux, valid_fluxes, valid_qfreqs, p0=p0, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))
    print(f"Fit: f_max={popt[0]:.2f} MHz, E_C={popt[1]:.1f} MHz, "
          f"V_period={popt[2]:.4f} V, V_offset={popt[3]:.4f} V, d={popt[4]:.4f}")
    fit_ok = True
except RuntimeError as e:
    print(f"Fit failed: {e}, falling back to linear interpolation")
    fit_ok = False

# Plot coarse data: color plot + tracked peaks + fit + resonator
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Coarse color plot (stitch coarse spectra)
coarse_fmin = min(f.min() for f in coarse_freqs)
coarse_fmax = max(f.max() for f in coarse_freqs)
n_coarse_grid = 300
coarse_freq_grid = np.linspace(coarse_fmin, coarse_fmax, n_coarse_grid)
coarse_amps_grid = np.full((len(coarse_fluxes), n_coarse_grid), np.nan)
for i, (freqs_i, amps_i) in enumerate(zip(coarse_freqs, coarse_amps)):
    mask = (coarse_freq_grid >= freqs_i.min()) & (coarse_freq_grid <= freqs_i.max())
    coarse_amps_grid[i, mask] = np.interp(coarse_freq_grid[mask], freqs_i, amps_i)
for i in range(len(coarse_fluxes)):
    valid = ~np.isnan(coarse_amps_grid[i])
    if valid.any():
        coarse_amps_grid[i, valid] -= np.nanmin(coarse_amps_grid[i, valid])

ax = axes[0]
pc = ax.pcolormesh(coarse_fluxes, coarse_freq_grid, coarse_amps_grid.T,
                   cmap='viridis', shading='auto', rasterized=True)
fig.colorbar(pc, ax=ax, label='Amp (a.u.)')
ax.plot(coarse_fluxes, coarse_qfreqs, 'r*', ms=10, label='Tracked peaks')
V_dense = np.linspace(flux_start, flux_stop, 500)
if fit_ok:
    ax.plot(V_dense, transmon_flux(V_dense, *popt), 'r-', lw=1.5, alpha=0.7, label='Fit')
ax.set_xlabel('Flux Bias (V)')
ax.set_ylabel('Qubit Frequency (MHz)')
ax.set_title(f'Coarse Spectroscopy Q{qi}')
ax.legend()

# Tracked peaks + fit
ax = axes[1]
ax.plot(coarse_fluxes, coarse_qfreqs, 'ko', ms=5, label='Coarse data')
if fit_ok:
    ax.plot(V_dense, transmon_flux(V_dense, *popt), 'r-', lw=2, label='Transmon fit')
ax.set_xlabel('Flux Bias (V)')
ax.set_ylabel('Qubit Frequency (MHz)')
ax.set_title(f'Coarse Peaks + Fit Q{qi}')
ax.legend()

# Resonator
ax = axes[2]
ax.plot(coarse_fluxes, coarse_rfreqs, 'bo-', ms=4)
ax.set_xlabel('Flux Bias (V)')
ax.set_ylabel('Resonator Frequency (MHz)')
ax.set_title(f'Resonator vs Flux Q{qi}')
plt.tight_layout()
fig.savefig(os.path.join(flux_sweep_dir, f"{run_timestamp}_coarse_fit_Q{qi}.png"),
            dpi=200, bbox_inches='tight')
plt.show()

# Compute fine flux points
# spacing_mode: 'freq' = constant frequency spacing (requires fit),
#               'flux' = constant flux spacing
spacing_mode = 'freq'
# use_fit_centers: when spacing_mode='flux', True = fit model, False = coarse interp
use_fit_centers = False

flux_start_fine = 0.07
flux_stop_fine = 0.5
freq_step_MHz = 2.0         # target frequency step (for spacing_mode='freq')
n_fine_per_coarse = 50       # flux subdivisions (for spacing_mode='flux')
min_span = 20.0              # MHz — generous minimum to not miss the qubit
span_margin = 3.0            # multiplier on freq step for scan span

if spacing_mode == 'freq' and fit_ok:
    # Invert the fit: sample f(V) densely, then pick V values at equal freq intervals
    V_dense = np.linspace(flux_start_fine, flux_stop_fine, 10000)
    f_dense = transmon_flux(V_dense, *popt)

    # f(V) is not monotonic — split into branches at the sweet spot (max freq)
    i_max = np.argmax(f_dense)
    f_max_val = f_dense[i_max]
    f_min_val = min(f_dense[0], f_dense[-1])

    # Target frequencies: equally spaced from min to max
    target_freqs = np.arange(f_min_val, f_max_val + freq_step_MHz / 2, freq_step_MHz)

    # For each branch (descending left, descending right of sweet spot),
    # interpolate to find V at each target freq
    fine_fluxes_list = []
    fine_centers_list = []

    # Left branch: V_dense[0..i_max], f is increasing
    if i_max > 0:
        V_left = V_dense[:i_max + 1]
        f_left = f_dense[:i_max + 1]
        mask_left = (target_freqs >= f_left[0]) & (target_freqs <= f_left[-1])
        if mask_left.any():
            V_at_f_left = np.interp(target_freqs[mask_left], f_left, V_left)
            fine_fluxes_list.append(V_at_f_left)
            fine_centers_list.append(target_freqs[mask_left])

    # Right branch: V_dense[i_max..end], f is decreasing → flip for interp
    if i_max < len(V_dense) - 1:
        V_right = V_dense[i_max:]
        f_right = f_dense[i_max:]
        mask_right = (target_freqs >= f_right[-1]) & (target_freqs <= f_right[0])
        if mask_right.any():
            V_at_f_right = np.interp(target_freqs[mask_right], f_right[::-1], V_right[::-1])
            fine_fluxes_list.append(V_at_f_right)
            fine_centers_list.append(target_freqs[mask_right])

    # Combine and sort by flux
    fine_fluxes = np.concatenate(fine_fluxes_list)
    fine_centers = np.concatenate(fine_centers_list)
    sort_idx = np.argsort(fine_fluxes)
    fine_fluxes = fine_fluxes[sort_idx]
    fine_centers = fine_centers[sort_idx]

    # Span = freq_step * margin (constant, since points are equally spaced in freq)
    fine_spans = np.full_like(fine_fluxes, freq_step_MHz * span_margin * freq_margin)
    fine_spans = np.clip(fine_spans, min_span, None)
    fit_freqs = fine_centers.copy()

    print(f"Fine sweep: freq spacing mode, {len(fine_fluxes)} points, "
          f"df={freq_step_MHz} MHz, freq range [{f_min_val:.1f}, {f_max_val:.1f}] MHz")

elif spacing_mode == 'freq' and not fit_ok:
    print("WARNING: freq spacing requires fit — falling back to flux spacing")
    spacing_mode = 'flux'

if spacing_mode == 'flux':
    fine_step = coarse_step / n_fine_per_coarse
    fine_fluxes = np.arange(flux_start_fine, flux_stop_fine + fine_step / 2, fine_step)

    if use_fit_centers and fit_ok:
        fine_centers = transmon_flux(fine_fluxes, *popt)
        fit_freqs = fine_centers.copy()
        print("Fine centers: using fit model")
    else:
        fine_centers = np.interp(fine_fluxes, valid_fluxes, valid_qfreqs,
                                 left=np.nan, right=np.nan)
        extrap_mask = np.isnan(fine_centers)
        if fit_ok and extrap_mask.any():
            fine_centers[extrap_mask] = transmon_flux(fine_fluxes[extrap_mask], *popt)
        elif extrap_mask.any():
            fine_centers[fine_fluxes < coarse_fluxes[0]] = coarse_qfreqs[0]
            fine_centers[fine_fluxes > coarse_fluxes[-1]] = coarse_qfreqs[-1]
        fit_freqs = transmon_flux(fine_fluxes, *popt) if fit_ok else None
        print("Fine centers: using coarse interpolation")

    # Spans from derivative
    if fit_ok:
        df_dV = np.gradient(transmon_flux(fine_fluxes, *popt), fine_fluxes)
    else:
        df_dV = np.gradient(fine_centers, fine_fluxes)
    fine_spans = np.abs(df_dV) * fine_step * span_margin * freq_margin
    fine_spans = np.clip(fine_spans, min_span, None)

    print(f"Fine sweep: flux spacing mode, {len(fine_fluxes)} points")

metadata['fit_ok'] = fit_ok
if fit_ok:
    metadata['fit_params'] = {'f_max': popt[0], 'E_C': popt[1],
                               'V_period': popt[2], 'V_offset': popt[3], 'd': popt[4]}
metadata['fine_fluxes_planned_V'] = fine_fluxes.tolist()
metadata['fine_centers_planned_MHz'] = fine_centers.tolist()
metadata['fine_spans_planned_MHz'] = fine_spans.tolist()
with open(os.path.join(flux_sweep_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# Save coarse data + fit to a reloadable file
coarse_save = dict(
    coarse_fluxes=coarse_fluxes,
    coarse_qfreqs=coarse_qfreqs,
    coarse_rfreqs=coarse_rfreqs,
    fine_fluxes=fine_fluxes,
    fine_centers=fine_centers,
    fine_spans=fine_spans,
    fit_ok=np.array(fit_ok),
)
if fit_ok:
    coarse_save['fit_popt'] = popt
    coarse_save['fit_pcov'] = pcov
# Save coarse spectral data as object arrays
coarse_save['coarse_freqs'] = np.array(coarse_freqs, dtype=object)
coarse_save['coarse_amps'] = np.array(coarse_amps, dtype=object)
np.savez(os.path.join(flux_sweep_dir, 'coarse_fit_data.npz'), **coarse_save)
print(f"Saved coarse + fit data to {flux_sweep_dir}/coarse_fit_data.npz")

print(f"Planned {len(fine_fluxes)} fine points, span range: "
      f"{fine_spans.min():.1f} - {fine_spans.max():.1f} MHz")
#%%
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
axes[0].plot(fine_fluxes, fine_centers, 'o-', ms=2, label='Centers (used)')
axes[0].plot(coarse_fluxes, coarse_qfreqs, 'r*', ms=10, label='Coarse data')
if fit_freqs is not None:
    V_plot = np.linspace(flux_start_fine, flux_stop_fine, 500)
    axes[0].plot(V_plot, transmon_flux(V_plot, *popt), 'g--', alpha=0.5, label='Fit')
axes[0].set_xlabel('Flux (V)'); axes[0].set_ylabel('Freq (MHz)')
axes[0].set_title(f'Planned centers ({spacing_mode} spacing)')
axes[0].legend()
axes[1].plot(fine_fluxes, fine_spans, 's-', ms=2)
axes[1].set_xlabel('Flux (V)'); axes[1].set_ylabel('Span (MHz)')
axes[1].set_title('Planned scan span')
axes[2].plot(fine_fluxes[:-1], np.diff(fine_fluxes) * 1e3, 'o-', ms=2)
axes[2].set_xlabel('Flux (V)'); axes[2].set_ylabel('Flux step (mV)')
axes[2].set_title('Flux step size')
plt.tight_layout(); plt.show()

# %% ---- Reload coarse + fit from saved file (skip Steps 1-2) ----
# Uncomment and set flux_sweep_dir to reload a previous run:
flux_sweep_dir_saved = os.path.join(expt_path, "20260311_113513_fluxsweep_Q2")

reload_data = np.load(os.path.join(flux_sweep_dir_saved, 'coarse_fit_data.npz'), allow_pickle=True)
coarse_fluxes = reload_data['coarse_fluxes']
coarse_qfreqs = reload_data['coarse_qfreqs']
coarse_rfreqs = reload_data['coarse_rfreqs']
fine_fluxes = reload_data['fine_fluxes']
fine_centers = reload_data['fine_centers']
fine_spans = reload_data['fine_spans']
fit_ok = bool(reload_data['fit_ok'])
if fit_ok:
    popt = reload_data['fit_popt']
    pcov = reload_data['fit_pcov']
    print(f"Fit: f_max={popt[0]:.2f} MHz, E_C={popt[1]:.1f} MHz, "
          f"V_period={popt[2]:.4f} V, V_offset={popt[3]:.4f} V, d={popt[4]:.4f}")
coarse_freqs = list(reload_data['coarse_freqs'])
coarse_amps = list(reload_data['coarse_amps'])
with open(os.path.join(flux_sweep_dir, 'metadata.json'), 'r') as f:
    metadata = json.load(f)
print(f"Reloaded coarse + fit data from {flux_sweep_dir}")
print(f"  {len(coarse_fluxes)} coarse points, {len(fine_fluxes)} fine points planned")

# %% ---- Step 3+4: Fine sweep with live plots ----
all_fine_fluxes = []
all_fine_freqs = []
all_fine_amps = []
all_fine_avgi = []
all_fine_avgq = []
all_fine_rfreqs = []
all_fine_rfreqs_spec = []
all_fine_ramps = []
all_fine_ravgi = []
all_fine_ravgq = []

qspec_common = dict(gain=0.2, expts=201, reps=1500, rounds=10,
                    sep_readout=True, length=2, readout_length=4)
rspec_params = dict(span='kappa', )

# Pre-compute frequency grids from planned centers/spans
n_freq_grid = 500
q_fmin = min(fine_centers - fine_spans / 2)
q_fmax = max(fine_centers + fine_spans / 2)
q_freq_grid = np.linspace(q_fmin, q_fmax, n_freq_grid)

# Resonator grid — estimate from config (will be refined after first point)
r_freq_est = auto_cfg.device.readout.frequency[qi]
r_kappa_est = auto_cfg.device.readout.kappa[qi] if hasattr(auto_cfg.device.readout, 'kappa') else 2.0
r_span_est = max(r_kappa_est * 10, 5.0)
r_freq_grid = np.linspace(r_freq_est - r_span_est, r_freq_est + r_span_est, n_freq_grid)

# Pre-allocate stitched arrays
n_flux = len(fine_fluxes)
amps_stitched = np.full((n_flux, n_freq_grid), np.nan)
avgi_stitched = np.full((n_flux, n_freq_grid), np.nan)
avgq_stitched = np.full((n_flux, n_freq_grid), np.nan)
ramps_stitched = np.full((n_flux, n_freq_grid), np.nan)
ravgi_stitched = np.full((n_flux, n_freq_grid), np.nan)
ravgq_stitched = np.full((n_flux, n_freq_grid), np.nan)

# --- Create live figures (Jupyter-compatible) ---
from IPython.display import display

fig_q, axes_q = plt.subplots(2, 2, figsize=(16, 10))
dummy_q = np.full((n_freq_grid, n_flux), np.nan)
pc_qamp = axes_q[0, 0].pcolormesh(fine_fluxes, q_freq_grid, dummy_q,
    cmap='viridis', shading='auto', rasterized=True)
fig_q.colorbar(pc_qamp, ax=axes_q[0, 0], label='Amplitude (a.u.)')
if fit_ok:
    axes_q[0, 0].plot(fine_fluxes, transmon_flux(fine_fluxes, *popt), 'r-', lw=1, alpha=0.7, zorder = -1)
axes_q[0, 0].set_xlabel('Flux Bias (V)'); axes_q[0, 0].set_ylabel('Frequency (MHz)')
axes_q[0, 0].set_title(f'Qubit Amplitude Q{qi}')

pc_qi = axes_q[0, 1].pcolormesh(fine_fluxes, q_freq_grid, dummy_q,
    cmap='RdBu_r', shading='auto', rasterized=True)
fig_q.colorbar(pc_qi, ax=axes_q[0, 1], label='I (a.u.)')
axes_q[0, 1].set_xlabel('Flux Bias (V)'); axes_q[0, 1].set_ylabel('Frequency (MHz)')
axes_q[0, 1].set_title(f'Qubit I Quadrature Q{qi}')

pc_qq = axes_q[1, 0].pcolormesh(fine_fluxes, q_freq_grid, dummy_q,
    cmap='RdBu_r', shading='auto', rasterized=True)
fig_q.colorbar(pc_qq, ax=axes_q[1, 0], label='Q (a.u.)')
axes_q[1, 0].set_xlabel('Flux Bias (V)'); axes_q[1, 0].set_ylabel('Frequency (MHz)')
axes_q[1, 0].set_title(f'Qubit Q Quadrature Q{qi}')

axes_q[1, 1].plot(coarse_fluxes, coarse_qfreqs, 'r*', ms=8, label='Coarse peaks')
if fit_ok:
    V_dense = np.linspace(fine_fluxes.min(), fine_fluxes.max(), 500)
    axes_q[1, 1].plot(V_dense, transmon_flux(V_dense, *popt), 'r-', lw=1.5, label='Fit')
axes_q[1, 1].set_xlabel('Flux Bias (V)'); axes_q[1, 1].set_ylabel('Qubit Frequency (MHz)')
axes_q[1, 1].set_title(f'Qubit Peaks + Fit Q{qi}')
axes_q[1, 1].legend()
fig_q.suptitle(f'Qubit Flux Sweep Q{qi}', fontsize=14)
fig_q.tight_layout()

fig_r, axes_r = plt.subplots(2, 2, figsize=(16, 10))
dummy_r = np.full((n_freq_grid, n_flux), np.nan)
pc_ramp = axes_r[0, 0].pcolormesh(fine_fluxes, r_freq_grid, dummy_r,
    cmap='viridis', shading='auto', rasterized=True)
fig_r.colorbar(pc_ramp, ax=axes_r[0, 0], label='Amplitude (a.u.)')
axes_r[0, 0].set_xlabel('Flux Bias (V)'); axes_r[0, 0].set_ylabel('Frequency (MHz)')
axes_r[0, 0].set_title(f'Resonator Amplitude Q{qi}')

pc_ri = axes_r[0, 1].pcolormesh(fine_fluxes, r_freq_grid, dummy_r,
    cmap='RdBu_r', shading='auto', rasterized=True)
fig_r.colorbar(pc_ri, ax=axes_r[0, 1], label='I (a.u.)')
axes_r[0, 1].set_xlabel('Flux Bias (V)'); axes_r[0, 1].set_ylabel('Frequency (MHz)')
axes_r[0, 1].set_title(f'Resonator I Quadrature Q{qi}')

pc_rq = axes_r[1, 0].pcolormesh(fine_fluxes, r_freq_grid, dummy_r,
    cmap='RdBu_r', shading='auto', rasterized=True)
fig_r.colorbar(pc_rq, ax=axes_r[1, 0], label='Q (a.u.)')
axes_r[1, 0].set_xlabel('Flux Bias (V)'); axes_r[1, 0].set_ylabel('Frequency (MHz)')
axes_r[1, 0].set_title(f'Resonator Q Quadrature Q{qi}')

rfreq_line, = axes_r[1, 1].plot([], [], 'bo-', ms=3)
axes_r[1, 1].set_xlabel('Flux Bias (V)'); axes_r[1, 1].set_ylabel('Resonator Frequency (MHz)')
axes_r[1, 1].set_title(f'Resonator Fitted Freq Q{qi}')
axes_r[1, 1].set_xlim(fine_fluxes.min(), fine_fluxes.max())
fig_r.suptitle(f'Resonator Flux Sweep Q{qi}', fontsize=14)
fig_r.tight_layout()

# Initial display — get handles for in-place updates
plt.close(fig_q); plt.close(fig_r)
dh_q = display(fig_q, display_id=True)
dh_r = display(fig_r, display_id=True)

# Track our figure numbers so we don't close them
live_fig_nums = {fig_q.number, fig_r.number}

def close_other_figs():
    """Close all figures except our live plots."""
    for num in plt.get_fignums():
        if num not in live_fig_nums:
            plt.close(num)

def update_live_plots(i_done):
    """Rebuild stitched arrays and update pcolormesh for completed points."""
    # Only stitch the latest point (previous ones already done)
    idx = i_done - 1
    freqs_i = all_fine_freqs[idx]
    mask = (q_freq_grid >= freqs_i.min()) & (q_freq_grid <= freqs_i.max())
    amps_stitched[idx, mask] = np.interp(q_freq_grid[mask], freqs_i, all_fine_amps[idx])
    avgi_stitched[idx, mask] = np.interp(q_freq_grid[mask], freqs_i, all_fine_avgi[idx])
    avgq_stitched[idx, mask] = np.interp(q_freq_grid[mask], freqs_i, all_fine_avgq[idx])
    valid = ~np.isnan(amps_stitched[idx])
    if valid.any():
        amps_stitched[idx, valid] -= np.nanmin(amps_stitched[idx, valid])

    pc_qamp.set_array(amps_stitched.T.ravel())
    pc_qamp.set_clim(np.nanmin(amps_stitched), np.nanmax(amps_stitched))
    pc_qi.set_array(avgi_stitched.T.ravel())
    pc_qi.set_clim(np.nanmin(avgi_stitched), np.nanmax(avgi_stitched))
    pc_qq.set_array(avgq_stitched.T.ravel())
    pc_qq.set_clim(np.nanmin(avgq_stitched), np.nanmax(avgq_stitched))

    freqs_i = all_fine_rfreqs_spec[idx]
    mask = (r_freq_grid >= freqs_i.min()) & (r_freq_grid <= freqs_i.max())
    ramps_stitched[idx, mask] = np.interp(r_freq_grid[mask], freqs_i, all_fine_ramps[idx])
    ravgi_stitched[idx, mask] = np.interp(r_freq_grid[mask], freqs_i, all_fine_ravgi[idx])
    ravgq_stitched[idx, mask] = np.interp(r_freq_grid[mask], freqs_i, all_fine_ravgq[idx])
    valid = ~np.isnan(ramps_stitched[idx])
    if valid.any():
        ramps_stitched[idx, valid] -= np.nanmin(ramps_stitched[idx, valid])

    pc_ramp.set_array(ramps_stitched.T.ravel())
    pc_ramp.set_clim(np.nanmin(ramps_stitched), np.nanmax(ramps_stitched))
    pc_ri.set_array(ravgi_stitched.T.ravel())
    pc_ri.set_clim(np.nanmin(ravgi_stitched), np.nanmax(ravgi_stitched))
    pc_rq.set_array(ravgq_stitched.T.ravel())
    pc_rq.set_clim(np.nanmin(ravgq_stitched), np.nanmax(ravgq_stitched))

    rfreq_line.set_data(all_fine_fluxes[:i_done], all_fine_rfreqs[:i_done])
    axes_r[1, 1].relim(); axes_r[1, 1].autoscale_view()

    # Update in-place in Jupyter
    dh_q.update(fig_q)
    dh_r.update(fig_r)

print("=== Step 3: Fine sweep (live) ===")
for i_flux in tqdm(range(len(fine_fluxes)), desc="Fine sweep"):
    flux = fine_fluxes[i_flux]
    center = fine_centers[i_flux]
    span = fine_spans[i_flux]
    start = center - span / 2

    # Set bias — save plots from previous iteration during settling time
    soc.rfb_set_bias(bias_chan, float(flux))
    if i_flux > 0:
        fig_q.savefig(os.path.join(flux_sweep_dir, 'images',
            f"qubit_flux_Q{qi}_step{i_flux-1:04d}.png"), dpi=150, bbox_inches='tight')
        fig_r.savefig(os.path.join(flux_sweep_dir, 'images',
            f"resonator_flux_Q{qi}_step{i_flux-1:04d}.png"), dpi=150, bbox_inches='tight')
    time.sleep(0.1)

    # Resonator (with retry)
    auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)
    for attempt in range(3):
        try:
            rspec = meas.ResSpec(cfg_dict_flux, qi=qi, params=rspec_params, display=False)
            rspec.update()
            break
        except Exception as e:
            print(f"  ResSpec failed (attempt {attempt+1}): {e}")
            soc.tproc.reset(); time.sleep(0.5)
    close_other_figs()
    auto_cfg = config.load(cfg_path)
    all_fine_rfreqs.append(auto_cfg.device.readout.frequency[qi])
    all_fine_rfreqs_spec.append(rspec.data['xpts'].copy())
    all_fine_ramps.append(rspec.data['amps'].copy())
    all_fine_ravgi.append(rspec.data['avgi'].copy())
    all_fine_ravgq.append(rspec.data['avgq'].copy())

    # Update resonator grid after first point (now we know the actual freq range)
    if i_flux == 0:
        r_freq_grid = np.linspace(rspec.data['xpts'].min(), rspec.data['xpts'].max(), n_freq_grid)
        for ax_row in axes_r[:2]:
            for ax in ax_row[:2] if isinstance(ax_row, np.ndarray) else [ax_row]:
                for coll in ax.collections:
                    coll.remove()
        dummy_r = np.full((n_freq_grid, n_flux), np.nan)
        pc_ramp = axes_r[0, 0].pcolormesh(fine_fluxes, r_freq_grid, dummy_r,
            cmap='viridis', shading='auto', rasterized=True)
        pc_ri = axes_r[0, 1].pcolormesh(fine_fluxes, r_freq_grid, dummy_r,
            cmap='RdBu_r', shading='auto', rasterized=True)
        pc_rq = axes_r[1, 0].pcolormesh(fine_fluxes, r_freq_grid, dummy_r,
            cmap='RdBu_r', shading='auto', rasterized=True)
        ramps_stitched = np.full((n_flux, n_freq_grid), np.nan)
        ravgi_stitched = np.full((n_flux, n_freq_grid), np.nan)
        ravgq_stitched = np.full((n_flux, n_freq_grid), np.nan)

    # Qubit spec
    rfboard.setup_qubit_drive_chain(qi, soc, auto_cfg, bw=filter_bw,
        atten1_dac=30, atten2_dac=25, center_freq=float(center))
    params = {**qspec_common, 'start': start, 'span': span}
    for attempt in range(3):
        try:
            qspec = meas.QubitSpec(cfg_dict_flux, qi=qi, style='fine', params=params, display=False)
            break
        except Exception as e:
            print(f"  QubitSpec failed (attempt {attempt+1}): {e}")
            soc.tproc.reset(); time.sleep(0.5)
    close_other_figs()

    all_fine_fluxes.append(float(flux))
    all_fine_freqs.append(qspec.data['xpts'].copy())
    all_fine_amps.append(qspec.data['amps'].copy())
    all_fine_avgi.append(qspec.data['avgi'].copy())
    all_fine_avgq.append(qspec.data['avgq'].copy())

    # Live update plots
    update_live_plots(i_flux + 1)

# Final save
fig_q.savefig(os.path.join(flux_sweep_dir, f"{run_timestamp}_qubit_flux_Q{qi}.png"),
              dpi=200, bbox_inches='tight')
fig_r.savefig(os.path.join(flux_sweep_dir, f"{run_timestamp}_resonator_flux_Q{qi}.png"),
              dpi=200, bbox_inches='tight')
np.savez(os.path.join(flux_sweep_dir, 'fine_sweep_data.npz'),
         fluxes=np.array(all_fine_fluxes),
         rfreqs=np.array(all_fine_rfreqs),
         freqs=np.array(all_fine_freqs, dtype=object),
         amps=np.array(all_fine_amps, dtype=object),
         avgi=np.array(all_fine_avgi, dtype=object),
         avgq=np.array(all_fine_avgq, dtype=object),
         rfreqs_spec=np.array(all_fine_rfreqs_spec, dtype=object),
         ramps=np.array(all_fine_ramps, dtype=object),
         ravgi=np.array(all_fine_ravgi, dtype=object),
         ravgq=np.array(all_fine_ravgq, dtype=object))
metadata['fine_fluxes_V'] = [float(f) for f in all_fine_fluxes]
metadata['fine_rfreqs_MHz'] = [float(f) for f in all_fine_rfreqs]
with open(os.path.join(flux_sweep_dir, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved fine sweep data to {flux_sweep_dir}")
# %%
for flux in np.linspace(flux,0,10):
    print(f"Setting flux bias to {flux:.2f} V")
    soc.rfb_set_bias(3, float(flux))
    time.sleep(0.1)  # wait for bias to settle

# %%
