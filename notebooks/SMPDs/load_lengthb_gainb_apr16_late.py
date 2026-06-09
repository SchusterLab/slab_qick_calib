"""
Load all 4WM_sweep_lengthb_gainb files from a selected dataset.
Set DATASET = 'apr16' or 'may29' to switch between datasets.
"""

import os
import json
import datetime
import numpy as np
from scipy.stats import poisson
from slab_qick_calib.exp_handling.datamanagement import SlabFile, AttrDict
import matplotlib.pyplot as plt

DATASET = 'may29'  # 'apr16' or 'may29'

if DATASET == 'apr16':
    DATA_DIR = r'C:\_Data\SMPD\2026_04_11_v2_calibration\data'
    TIME_START = datetime.datetime(2026, 4, 16, 23, 45, 0)
    TIME_END   = None  # no upper bound
elif DATASET == 'may29':
    DATA_DIR   = r'C:\_Data\SMPD\2026_05_01_v2_run28_JPA\data'
    TIME_START = datetime.datetime(2026, 5, 29, 2, 50, 0)
    TIME_END   = datetime.datetime(2026, 5, 29, 7, 30, 0)

# Find all lengthb_gainb files in the time window
files = sorted(
    f for f in os.listdir(DATA_DIR)
    if 'lengthb_gainb' in f and f.endswith('.h5')
)

late_files = []
for fname in files:
    fpath = os.path.join(DATA_DIR, fname)
    mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
    if mtime >= TIME_START and (TIME_END is None or mtime <= TIME_END):
        late_files.append((fname, fpath, mtime))

# print(f"Found {len(late_files)} files modified after {CUTOFF.strftime('%I:%M %p')}:\n")
# for fname, _, mtime in late_files:
#     print(f"  {mtime.strftime('%m/%d %I:%M:%S %p')}  {fname}")

# Load each file
datasets = []
for fname, fpath, mtime in late_files:
    with SlabFile(fpath, 'r') as f:
        data = {k: np.array(f[k]) for k in f.keys()}
        data['attrs'] = f.get_dict()
        cfg = f.load_config()
    datasets.append({'fname': fname, 'mtime': mtime, 'data': data, 'cfg': cfg})
    # print(f"Loaded {fname}: keys = {list(data.keys())}")

# load up most recent stark spectroscopy from the same folder
for fname in os.listdir(DATA_DIR):
    if 'stark_spectroscopy' in fname and fname.endswith('.h5'):
        fpath = os.path.join(DATA_DIR, fname)
        with SlabFile(fpath, 'r') as f:
            data = {k: np.array(f[k]) for k in f.keys()}
            data['attrs'] = f.get_dict()
            cfg = f.load_config()
        stark_dataset = {'fname': fname, 'mtime': datetime.datetime.fromtimestamp(os.path.getmtime(fpath)), 'data': data, 'cfg': cfg}
print(fpath)
n_photon_conversion = stark_dataset['data']['n']
print(n_photon_conversion)
# n_photon_conversion = 285

# print(f"\nDone. {len(datasets)} datasets loaded into `datasets`.")
gain_ps = [dataset['cfg']['expt']['gain_p'] for dataset in datasets]
max_slope = []
for i, dataset in enumerate(datasets):
    length_b = dataset['data']['length_b_pts']
    gain_b = dataset['data']['xpts']
    p_e = dataset['data']['p_e']
    gain_p = dataset['cfg']['expt']['gain_p']
    freq_p = dataset['cfg']['expt']['frequency_p']

    # convert gain_b to 1-P(0) using Poisson statistics
    n_photons = gain_b**2 * n_photon_conversion
    gain_b_converted = 1 - poisson.pmf(0, n_photons)
    if i == 0:
        plt.figure()
        plt.plot(gain_b, gain_b_converted, marker='o')
        plt.xlabel('gain_b')
        plt.ylabel('1 - P(0)')
        plt.title('Conversion from gain_b to 1-P(0)')
        plt.grid()
        plt.show()

    # compute slope of p_e vs 1-P(0) between 0.05 and 0.2 at each length_b
    # print(n_photons)
    # print(gain_b_converted)
    slopes = []
    if i % 8 == 0:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for j in range(len(length_b)):
        idx_05 = np.argmin(np.abs(gain_b_converted - 0.05))
        idx_02 = np.argmin(np.abs(gain_b_converted - 0.2))
        if idx_02 > idx_05:
            coeffs = np.polyfit(gain_b_converted[idx_05:idx_02], p_e[j, idx_05:idx_02], 1)
            slopes.append(coeffs[0])
        else:
            slopes.append(np.nan)

        if i % 8 == 0 and j % 10 == 0:
            axes[1].plot(gain_b_converted, p_e[j], marker='o')
            axes[1].set_xlabel('1 - P(0)')
            axes[1].set_ylabel('p_e')
            axes[1].set_ylim(0, 1)
            axes[1].set_title(f"gain_p = {gain_p:.1f}")
            axes[1].grid()

    # find the average slope at later length_b values (greater than 40 us)
    later_slopes = [s for s, lb in zip(slopes, length_b) if lb > 40]
    avg_later_slope = np.nanmean(later_slopes)
    max_slope.append(avg_later_slope)

    print(datasets[i]['cfg']['expt']['gain_p'])
    if i % 8 == 0:
        mesh = axes[0].pcolormesh(length_b, gain_b, dataset['data']['p_e'], vmin=0, vmax=1)
        fig.colorbar(mesh, ax=axes[0], label='p_e')
        axes[0].set_xlabel('buffer length (us)')
        axes[0].set_ylabel('buffer gain')
        axes[0].set_title(f"gain_p = {gain_p:.1f}")

        axes[2].plot(length_b, slopes)
        axes[2].plot([40,50], [avg_later_slope, avg_later_slope], 'r--', label=f'avg slope > 40 us = {avg_later_slope:.3f}')
        axes[2].set_ylim(0,2)
        axes[2].set_xlabel('buffer length (us)')
        axes[2].set_ylabel('slope of p_e vs 1-P(0)')
        axes[2].set_title(f"gain_p = {gain_p:.1f}")
        plt.tight_layout()
        plt.show()

plt.figure()
plt.plot(gain_ps, max_slope, marker='o')
plt.xlim(left=0)
xlims = plt.xlim()
x = np.linspace(0,xlims[1])

x_ind = np.argmax(100 * max_slope)
scale = x[x_ind]*1.7

# y = 4*(x/scale)**2/(1+kappa_ratio+(x/scale)**2)**2
# fit the data to the function y = 4*A*(x/scale)**2/(1+(0.08/2)+(x/scale)**2)**2, where scale, kappa_ratio, and A are free parameters
from scipy.optimize import curve_fit
def fit_func(x, scale, kappa_ratio, A):
    return 4*A*(x/scale)**2/(1+kappa_ratio+(x/scale)**2)**2

# throw out fourth-to-last point which seems like an outlier
gain_ps_fit = np.array(gain_ps)
max_slope_fit = np.array(max_slope)
gain_ps_fit = gain_ps_fit[np.array(gain_ps) < 0.7]
max_slope_fit = max_slope_fit[np.array(gain_ps) < 0.7]
popt, pcov = curve_fit(fit_func, gain_ps_fit, max_slope_fit, p0=[scale, 0.08, np.max(max_slope_fit)], bounds=(0, np.inf))


plt.plot(x,fit_func(x, popt[0], popt[1], popt[2]),label=r'$\eta_\mathrm{4wm} = \frac{4\mathcal{C}}{(1+\kappa_i/\kappa_c + \mathcal{C})^2}$' +'\n'+r'$\mathcal{C}\propto (\mathrm{pump\;amplitude})^2$')
plt.plot([],[],ls='',marker='',label = f'Fitted parameters:\nscale = {popt[0]:.3f}\n$\kappa_i/\kappa_c$ = {popt[1]:.3f}\nA = {popt[2]:.3f}')
plt.xlabel('gain_p')
plt.ylabel('efficiency')
plt.legend()
plt.show()