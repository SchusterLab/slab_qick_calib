#%% ---- Configuration ----
import numpy as np
import matplotlib.pyplot as plt
import json
import os

flux_sweep_dir = r'C:\_Data\sample_50\20260311_113513_fluxsweep_Q2'  # <-- set this
n_freq_grid = 500  # interpolation grid size
flux_clip = None   # number of flux points to use (None = all, e.g. 50 = first 50)

#%% ---- Load data ----
# Coarse + fit
coarse = np.load(os.path.join(flux_sweep_dir, 'coarse_fit_data.npz'), allow_pickle=True)
coarse_fluxes = coarse['coarse_fluxes']
coarse_qfreqs = coarse['coarse_qfreqs']
coarse_rfreqs = coarse['coarse_rfreqs']
fine_fluxes_planned = coarse['fine_fluxes']
fine_centers = coarse['fine_centers']
fine_spans = coarse['fine_spans']
fit_ok = bool(coarse['fit_ok'])
if fit_ok:
    popt = coarse['fit_popt']
    pcov = coarse['fit_pcov']

def transmon_flux(V, f_max, E_C, V_period, V_offset, d):
    phi = np.pi * (V - V_offset) / V_period
    return (f_max + E_C) * (np.cos(phi)**2 + d**2 * np.sin(phi)**2)**0.25 - E_C

# Fine sweep
fine = np.load(os.path.join(flux_sweep_dir, 'fine_sweep_data.npz'), allow_pickle=True)
fluxes = fine['fluxes']
rfreqs = fine['rfreqs']
freqs = [np.asarray(a, dtype=float) for a in fine['freqs']]
amps = [np.asarray(a, dtype=float) for a in fine['amps']]
avgi = [np.asarray(a, dtype=float) for a in fine['avgi']]
avgq = [np.asarray(a, dtype=float) for a in fine['avgq']]
rfreqs_spec = [np.asarray(a, dtype=float) for a in fine['rfreqs_spec']]
ramps = [np.asarray(a, dtype=float) for a in fine['ramps']]
ravgi = [np.asarray(a, dtype=float) for a in fine['ravgi']]
ravgq = [np.asarray(a, dtype=float) for a in fine['ravgq']]

with open(os.path.join(flux_sweep_dir, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

n_flux_total = len(fluxes)
flux_clip = 50
if flux_clip is not None:
    flux_clip = min(flux_clip, n_flux_total)
    fluxes = fluxes[:flux_clip]
    rfreqs = rfreqs[:flux_clip]
    freqs = freqs[:flux_clip]
    amps = amps[:flux_clip]
    avgi = avgi[:flux_clip]
    avgq = avgq[:flux_clip]
    rfreqs_spec = rfreqs_spec[:flux_clip]
    ramps = ramps[:flux_clip]
    ravgi = ravgi[:flux_clip]
    ravgq = ravgq[:flux_clip]

n_flux = len(fluxes)
print(f"Loaded {n_flux_total} fine sweep points from {flux_sweep_dir}"
      + (f" (clipped to {n_flux})" if flux_clip else ""))
if fit_ok:
    print(f"Fit: f_max={popt[0]:.2f} MHz, E_C={popt[1]:.1f} MHz, "
          f"V_period={popt[2]:.4f} V, V_offset={popt[3]:.4f} V, d={popt[4]:.4f}")

#%% ---- Stitch onto common grids ----
# Qubit grid
q_fmin = min(f.min() for f in freqs)
q_fmax = max(f.max() for f in freqs)
q_freq_grid = np.linspace(q_fmin, q_fmax, n_freq_grid)

# Resonator grid
r_fmin = min(f.min() for f in rfreqs_spec)
r_fmax = max(f.max() for f in rfreqs_spec)
r_freq_grid = np.linspace(r_fmin, r_fmax, n_freq_grid)

amps_stitched = np.full((n_flux, n_freq_grid), np.nan)
avgi_stitched = np.full((n_flux, n_freq_grid), np.nan)
avgq_stitched = np.full((n_flux, n_freq_grid), np.nan)
ramps_stitched = np.full((n_flux, n_freq_grid), np.nan)
ravgi_stitched = np.full((n_flux, n_freq_grid), np.nan)
ravgq_stitched = np.full((n_flux, n_freq_grid), np.nan)

for i in range(n_flux):
    # Qubit
    mask = (q_freq_grid >= freqs[i].min()) & (q_freq_grid <= freqs[i].max())
    amps_stitched[i, mask] = np.interp(q_freq_grid[mask], freqs[i], amps[i])
    avgi_stitched[i, mask] = np.interp(q_freq_grid[mask], freqs[i], avgi[i])
    avgq_stitched[i, mask] = np.interp(q_freq_grid[mask], freqs[i], avgq[i])
    valid = ~np.isnan(amps_stitched[i])
    if valid.any():
        amps_stitched[i, valid] -= np.nanmin(amps_stitched[i, valid])

    # Resonator
    mask = (r_freq_grid >= rfreqs_spec[i].min()) & (r_freq_grid <= rfreqs_spec[i].max())
    ramps_stitched[i, mask] = np.interp(r_freq_grid[mask], rfreqs_spec[i], ramps[i])
    ravgi_stitched[i, mask] = np.interp(r_freq_grid[mask], rfreqs_spec[i], ravgi[i])
    ravgq_stitched[i, mask] = np.interp(r_freq_grid[mask], rfreqs_spec[i], ravgq[i])
    valid = ~np.isnan(ramps_stitched[i])
    if valid.any():
        ramps_stitched[i, valid] -= np.nanmin(ramps_stitched[i, valid])

#%% ---- Plot qubit flux sweep ----
fig_q, axes_q = plt.subplots(2, 2, figsize=(16, 10))

pc = axes_q[0, 0].pcolormesh(fluxes, q_freq_grid, amps_stitched.T,
    cmap='viridis', shading='auto', rasterized=True)
fig_q.colorbar(pc, ax=axes_q[0, 0], label='Amplitude (a.u.)')
if fit_ok:
    axes_q[0, 0].plot(fluxes, transmon_flux(fluxes, *popt), 'r-', lw=1, alpha=0.7)
axes_q[0, 0].set_xlabel('Flux Bias (V)'); axes_q[0, 0].set_ylabel('Frequency (MHz)')
axes_q[0, 0].set_title('Qubit Amplitude')

pc = axes_q[0, 1].pcolormesh(fluxes, q_freq_grid, avgi_stitched.T,
    cmap='RdBu_r', shading='auto', rasterized=True)
fig_q.colorbar(pc, ax=axes_q[0, 1], label='I (a.u.)')
axes_q[0, 1].set_xlabel('Flux Bias (V)'); axes_q[0, 1].set_ylabel('Frequency (MHz)')
axes_q[0, 1].set_title('Qubit I Quadrature')

pc = axes_q[1, 0].pcolormesh(fluxes, q_freq_grid, avgq_stitched.T,
    cmap='RdBu_r', shading='auto', rasterized=True)
fig_q.colorbar(pc, ax=axes_q[1, 0], label='Q (a.u.)')
axes_q[1, 0].set_xlabel('Flux Bias (V)'); axes_q[1, 0].set_ylabel('Frequency (MHz)')
axes_q[1, 0].set_title('Qubit Q Quadrature')

axes_q[1, 1].plot(coarse_fluxes, coarse_qfreqs, 'r*', ms=8, label='Coarse peaks')
# if fit_ok:
#     V_dense = np.linspace(fluxes.min(), fluxes.max(), 500)
#     axes_q[1, 1].plot(V_dense, transmon_flux(V_dense, *popt), 'r-', lw=1.5, label='Fit')
axes_q[1, 1].set_xlabel('Flux Bias (V)'); axes_q[1, 1].set_ylabel('Qubit Frequency (MHz)')
axes_q[1, 1].set_title('Qubit Peaks + Fit')
axes_q[1, 1].legend()
fig_q.suptitle('Qubit Flux Sweep', fontsize=14)
fig_q.tight_layout()
plt.show()

#%% ---- Plot resonator flux sweep ----
fig_r, axes_r = plt.subplots(2, 2, figsize=(16, 10))

pc = axes_r[0, 0].pcolormesh(fluxes, r_freq_grid, ramps_stitched.T,
    cmap='viridis', shading='auto', rasterized=True)
fig_r.colorbar(pc, ax=axes_r[0, 0], label='Amplitude (a.u.)')
axes_r[0, 0].set_xlabel('Flux Bias (V)'); axes_r[0, 0].set_ylabel('Frequency (MHz)')
axes_r[0, 0].set_title('Resonator Amplitude')

pc = axes_r[0, 1].pcolormesh(fluxes, r_freq_grid, ravgi_stitched.T,
    cmap='RdBu_r', shading='auto', rasterized=True)
fig_r.colorbar(pc, ax=axes_r[0, 1], label='I (a.u.)')
axes_r[0, 1].set_xlabel('Flux Bias (V)'); axes_r[0, 1].set_ylabel('Frequency (MHz)')
axes_r[0, 1].set_title('Resonator I Quadrature')

pc = axes_r[1, 0].pcolormesh(fluxes, r_freq_grid, ravgq_stitched.T,
    cmap='RdBu_r', shading='auto', rasterized=True)
fig_r.colorbar(pc, ax=axes_r[1, 0], label='Q (a.u.)')
axes_r[1, 0].set_xlabel('Flux Bias (V)'); axes_r[1, 0].set_ylabel('Frequency (MHz)')
axes_r[1, 0].set_title('Resonator Q Quadrature')

axes_r[1, 1].plot(fluxes, rfreqs, 'bo-', ms=3)
axes_r[1, 1].set_xlabel('Flux Bias (V)'); axes_r[1, 1].set_ylabel('Resonator Frequency (MHz)')
axes_r[1, 1].set_title('Resonator Fitted Freq')
fig_r.suptitle('Resonator Flux Sweep', fontsize=14)
fig_r.tight_layout()
plt.show()

#%% ---- IQ rotation: rotate all signal into I ----
# At each flux point, find the optimal rotation angle that maximizes
# the signal variance in I (equivalent to finding the axis of maximum
# signal). Then plot the rotated-I data.

def optimal_iq_angle(i_data, q_data):
    """Find rotation angle that maximizes variance in the rotated-I axis."""
    theta = 0.5 * np.arctan2(2 * np.mean(i_data * q_data) - 2 * np.mean(i_data) * np.mean(q_data),
                              np.var(i_data) - np.var(q_data))
    return theta

def rotate_iq(i_data, q_data, theta):
    """Rotate I/Q data by angle theta."""
    i_rot = i_data * np.cos(theta) + q_data * np.sin(theta)
    q_rot = -i_data * np.sin(theta) + q_data * np.cos(theta)
    return i_rot, q_rot

# Rotate raw data at each flux point
rotated_i_list = []
rotated_q_list = []
angles = []

for i in range(n_flux):
    theta = optimal_iq_angle(avgi[i], avgq[i])
    i_rot, q_rot = rotate_iq(avgi[i], avgq[i], theta)
    # Convention: ensure the peak feature is positive (flip if needed)
    if np.abs(i_rot.min()) > np.abs(i_rot.max()):
        i_rot = -i_rot
        q_rot = -q_rot
        theta += np.pi
    rotated_i_list.append(i_rot)
    rotated_q_list.append(q_rot)
    angles.append(theta)

angles = np.array(angles)

# Stitch rotated I onto common grid
roti_stitched = np.full((n_flux, n_freq_grid), np.nan)
for i in range(n_flux):
    mask = (q_freq_grid >= freqs[i].min()) & (q_freq_grid <= freqs[i].max())
    roti_stitched[i, mask] = np.interp(q_freq_grid[mask], freqs[i], rotated_i_list[i])

# Same for resonator
r_rotated_i_list = []
r_angles = []

for i in range(n_flux):
    theta = optimal_iq_angle(ravgi[i], ravgq[i])
    i_rot, q_rot = rotate_iq(ravgi[i], ravgq[i], theta)
    if np.abs(i_rot.min()) > np.abs(i_rot.max()):
        i_rot = -i_rot
        theta += np.pi
    r_rotated_i_list.append(i_rot)
    r_angles.append(theta)

r_angles = np.array(r_angles)

r_roti_stitched = np.full((n_flux, n_freq_grid), np.nan)
for i in range(n_flux):
    mask = (r_freq_grid >= rfreqs_spec[i].min()) & (r_freq_grid <= rfreqs_spec[i].max())
    r_roti_stitched[i, mask] = np.interp(r_freq_grid[mask], rfreqs_spec[i], r_rotated_i_list[i])

#%% ---- Plot rotated-I data ----
fig_rot, axes_rot = plt.subplots(2, 2, figsize=(16, 10))

pc = axes_rot[0, 0].pcolormesh(fluxes, q_freq_grid, roti_stitched.T,
    cmap='RdBu_r', shading='auto', rasterized=True)
fig_rot.colorbar(pc, ax=axes_rot[0, 0], label='Rotated I (a.u.)')
# if fit_ok:
#     axes_rot[0, 0].plot(fluxes, transmon_flux(fluxes, *popt), 'k-', lw=1, alpha=0.7)
axes_rot[0, 0].set_xlabel('Flux Bias (V)'); axes_rot[0, 0].set_ylabel('Frequency (MHz)')
axes_rot[0, 0].set_title('Qubit Rotated I')

axes_rot[0, 1].plot(fluxes, np.degrees(angles), 'o-', ms=3)
axes_rot[0, 1].set_xlabel('Flux Bias (V)'); axes_rot[0, 1].set_ylabel('Rotation Angle (deg)')
axes_rot[0, 1].set_title('Qubit IQ Rotation Angle')

pc = axes_rot[1, 0].pcolormesh(fluxes, r_freq_grid, r_roti_stitched.T,
    cmap='RdBu_r', shading='auto', rasterized=True)
fig_rot.colorbar(pc, ax=axes_rot[1, 0], label='Rotated I (a.u.)')
axes_rot[1, 0].set_xlabel('Flux Bias (V)'); axes_rot[1, 0].set_ylabel('Frequency (MHz)')
axes_rot[1, 0].set_title('Resonator Rotated I')

axes_rot[1, 1].plot(fluxes, np.degrees(r_angles), 'o-', ms=3)
axes_rot[1, 1].set_xlabel('Flux Bias (V)'); axes_rot[1, 1].set_ylabel('Rotation Angle (deg)')
axes_rot[1, 1].set_title('Resonator IQ Rotation Angle')


fig_rot.suptitle('IQ-Rotated Flux Sweep', fontsize=14)
fig_rot.tight_layout()

plt.show()

#%% ---- Baseline-subtracted, strictly positive rotated-I ----
n_bl = 10  # number of edge points to average for baseline

# Qubit: subtract baseline per flux point, then make positive
roti_bl = roti_stitched.copy()
for i in range(n_flux):
    valid = ~np.isnan(roti_bl[i])
    idx_valid = np.where(valid)[0]
    if len(idx_valid) < 2 * n_bl:
        continue
    bl = np.mean(np.concatenate([roti_bl[i, idx_valid[:n_bl]],
                                  roti_bl[i, idx_valid[-n_bl:]]]))
    roti_bl[i, valid] -= bl
roti_bl = np.where(np.isnan(roti_bl), np.nan, np.abs(roti_bl))

# Resonator: same
r_roti_bl = r_roti_stitched.copy()
for i in range(n_flux):
    valid = ~np.isnan(r_roti_bl[i])
    idx_valid = np.where(valid)[0]
    if len(idx_valid) < 2 * n_bl:
        continue
    bl = np.mean(np.concatenate([r_roti_bl[i, idx_valid[:n_bl]],
                                  r_roti_bl[i, idx_valid[-n_bl:]]]))
    r_roti_bl[i, valid] -= bl
r_roti_bl = np.where(np.isnan(r_roti_bl), np.nan, np.abs(r_roti_bl))

fig_bl, axes_bl = plt.subplots(1, 2, figsize=(16, 5))

pc = axes_bl[0].pcolormesh(fluxes, q_freq_grid, roti_bl.T,
    cmap='inferno', shading='auto', rasterized=True)
fig_bl.colorbar(pc, ax=axes_bl[0], label='|Rotated I| (a.u.)')
# if fit_ok:
#     axes_bl[0].plot(fluxes, transmon_flux(fluxes, *popt), 'c-', lw=1, alpha=0.7)
axes_bl[0].set_xlabel('Flux Bias (V)'); axes_bl[0].set_ylabel('Frequency (MHz)')
axes_bl[0].set_title('Qubit |Rotated I| (baseline subtracted)')

pc = axes_bl[1].pcolormesh(fluxes, r_freq_grid, r_roti_bl.T,
    cmap='inferno', shading='auto', rasterized=True)
fig_bl.colorbar(pc, ax=axes_bl[1], label='|Rotated I| (a.u.)')
axes_bl[1].set_xlabel('Flux Bias (V)'); axes_bl[1].set_ylabel('Frequency (MHz)')
axes_bl[1].set_title('Resonator |Rotated I| (baseline subtracted)')

fig_bl.tight_layout()
plt.show()

#%% ---- Save all figures ----

fig_q.savefig(os.path.join(flux_sweep_dir, f'qubit_flux_sweep_clipped{flux_clip}.png'), dpi=600, bbox_inches='tight')
fig_r.savefig(os.path.join(flux_sweep_dir, f'resonator_flux_sweep_{flux_clip}.png'), dpi=600, bbox_inches='tight')
fig_rot.savefig(os.path.join(flux_sweep_dir, f'iq_rotated_flux_sweep_{flux_clip}.png'), dpi=600, bbox_inches='tight')
fig_bl.savefig(os.path.join(flux_sweep_dir, f'iq_rotated_baseline_sub_{flux_clip}.png'), dpi=1000, bbox_inches='tight')
fig_bl.savefig(os.path.join(flux_sweep_dir, f'iq_rotated_baseline_sub_{flux_clip}.pdf'),  bbox_inches='tight')

print(f"Saved plots to {flux_sweep_dir}")

#%% ---- Per-trace normalized baseline-subtracted rotated-I ----
# Each flux trace is independently normalized to [0, 1]
roti_norm = roti_bl.copy()
for i in range(n_flux):
    valid = ~np.isnan(roti_norm[i])
    if not valid.any():
        continue
    peak = np.nanmax(roti_norm[i])
    if peak > 0:
        roti_norm[i, valid] /= peak

r_roti_norm = r_roti_bl.copy()
for i in range(n_flux):
    valid = ~np.isnan(r_roti_norm[i])
    if not valid.any():
        continue
    peak = np.nanmax(r_roti_norm[i])
    if peak > 0:
        r_roti_norm[i, valid] /= peak

fig_norm, axes_norm = plt.subplots(1, 2, figsize=(16, 5))

pc = axes_norm[0].pcolormesh(fluxes, q_freq_grid, roti_norm.T,
    cmap='inferno', shading='auto', rasterized=True, vmin=0, vmax=1)
fig_norm.colorbar(pc, ax=axes_norm[0], label='Normalized signal')
# if fit_ok:
#     axes_norm[0].plot(fluxes, transmon_flux(fluxes, *popt), 'c-', lw=1, alpha=0.7)
axes_norm[0].set_xlabel('Flux Bias (V)'); axes_norm[0].set_ylabel('Frequency (MHz)')
axes_norm[0].set_title('Qubit (per-trace normalized)')

pc = axes_norm[1].pcolormesh(fluxes, r_freq_grid, r_roti_norm.T,
    cmap='inferno', shading='auto', rasterized=True, vmin=0, vmax=1)
fig_norm.colorbar(pc, ax=axes_norm[1], label='Normalized signal')
axes_norm[1].set_xlabel('Flux Bias (V)'); axes_norm[1].set_ylabel('Frequency (MHz)')
axes_norm[1].set_title('Resonator (per-trace normalized)')

fig_norm.tight_layout()
fig_norm.savefig(os.path.join(flux_sweep_dir, 'iq_rotated_baseline_sub_normalized.pdf'),  bbox_inches='tight')
fig_norm.savefig(os.path.join(flux_sweep_dir, 'iq_rotated_baseline_sub_normalized.png'),  bbox_inches='tight')

plt.show()

# %%
