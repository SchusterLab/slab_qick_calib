import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from qick import QickConfig
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
from slab_qick_calib.helpers import config, rfboard
import slab_qick_calib.experiments as meas

cfg_path = 'C:\\_Lib\\python\\slab_qick_calib\\configs\\2026_06_08_smpd_v2_run29.yml'
expt_path = 'C:\\_Data\\SMPD\\2026_06_01_Yuvi_v2_run29'
auto_cfg = config.load(cfg_path)

im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'], port=9090)
soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())

qi = 1
auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)
cfg_dict = {'soc': soccfg, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}

gain = 0.03
fc_set = 5000

# filter_bw sweep values (MHz)
bw_values = [100, 200, 500, 1000, 2000, 5000]

# Set use_yml=True to update hw.soc.adcs/dacs.readout.filter_bw in the .yml and
# apply via apply_config_rf_settings. Set False to use rfboard.set_bandpass_rf directly.
use_yml = False

# Two-segment scan: below SPLIT_FREQ uses ATT_NORMAL, above uses ATT_HIGH_FREQ.
# ATT_NORMAL is read from the config; ATT_HIGH_FREQ reduces both to 0 dB for better
# SNR at high frequencies.
# Amplitude correction: 10^(total_dB_reduction / 20) normalises the high-freq segment
# back to the same scale as the low-freq segment.
SPLIT_FREQ    = 7000
ATT_NORMAL    = (auto_cfg.hw.soc.dacs.readout.attn1[qi],
                 auto_cfg.hw.soc.dacs.readout.attn2[qi])
ATT_HIGH_FREQ = (5, 5)
ATT_CORRECTION = 10 ** ((sum(ATT_NORMAL) - sum(ATT_HIGH_FREQ)) / 20)

out_dir = os.path.join(expt_path, 'filter_bw_sweep')
os.makedirs(out_dir, exist_ok=True)


def set_filter(fc, bw, att1=None, att2=None):
    a1 = ATT_NORMAL[0] if att1 is None else att1
    a2 = ATT_NORMAL[1] if att2 is None else att2
    if use_yml:
        auto_cfg.hw.soc.adcs.readout.filter_fc[qi] = fc
        auto_cfg.hw.soc.adcs.readout.filter_bw[qi] = bw
        auto_cfg.hw.soc.dacs.readout.filter_fc[qi] = fc
        auto_cfg.hw.soc.dacs.readout.filter_bw[qi] = bw
        auto_cfg.hw.soc.dacs.readout.attn1[qi] = a1
        auto_cfg.hw.soc.dacs.readout.attn2[qi] = a2
        config.save(auto_cfg, cfg_path)
        rfboard.apply_config_rf_settings(soc, auto_cfg, qubit=qi, cfg_file=cfg_path)
    else:
        rfboard.set_bandpass_rf(soc, qi, qi, fc, bw, tx_att1=a1, tx_att2=a2)


def measure_stitched(fc, bw):
    """Two scans stitched at SPLIT_FREQ. High-freq segment uses ATT_HIGH_FREQ and is
    amplitude-corrected to match the low-freq attenuation setting."""
    set_filter(fc, bw)
    rspec_lo = meas.ResSpec(cfg_dict, qi=qi, params={
        'center': 4750, 'span': 8500, 'gain': gain, 'reps': 50, 'expts': 500
    })

    set_filter(fc, bw, att1=ATT_HIGH_FREQ[0], att2=ATT_HIGH_FREQ[1])
    rspec_hi = meas.ResSpec(cfg_dict, qi=qi, params={
        'center': 4750, 'span': 8500, 'gain': gain, 'reps': 50, 'expts': 500
    })

    xpts_lo = np.array(rspec_lo.data['xpts'])
    xpts_hi = np.array(rspec_hi.data['xpts'])
    amps_lo = np.array(rspec_lo.data['amps'])
    amps_hi = np.array(rspec_hi.data['amps'])

    mask_lo = xpts_lo < SPLIT_FREQ
    mask_hi = xpts_hi >= SPLIT_FREQ

    # Divide to undo the extra signal from lower attenuation
    amps_hi_corrected = amps_hi / ATT_CORRECTION

    xpts = np.concatenate([xpts_lo[mask_lo], xpts_hi[mask_hi]])
    amps = np.concatenate([amps_lo[mask_lo], amps_hi_corrected[mask_hi]])
    return xpts, amps


# --- Baseline (taken once) ---
print("Taking baseline (fc=10300, bw=17099)...")
xpts_baseline, amps_baseline = measure_stitched(10300, 17099)
S21_db_baseline = 20 * np.log10(amps_baseline)

# --- Sweep ---
for filter_bw_set in bw_values:
    print(f"filter_bw_set = {filter_bw_set} MHz...")
    xpts, amps = measure_stitched(fc_set, filter_bw_set)
    S21_db = 20 * np.log10(amps)

    lower_cutoff  = fc_set - filter_bw_set / 2
    higher_cutoff = fc_set + filter_bw_set / 2

    plt.figure()
    plt.plot(xpts, S21_db, label=f'fc={fc_set} MHz, bw={filter_bw_set} MHz')
    plt.plot(xpts_baseline, S21_db_baseline, label='largest bandwidth setting:\nfc=10300 MHz, bw=17099 MHz')
    plt.plot(xpts, S21_db - S21_db_baseline, label='current - baseline')
    plt.axvline(lower_cutoff,  color='r', linestyle='--', label='HPF cutoff: {:.1f} MHz'.format(lower_cutoff))
    plt.axvline(higher_cutoff, color='r', linestyle='--', label='LPF cutoff: {:.1f} MHz'.format(higher_cutoff))
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('|ADC units|^2 (dB)')
    plt.title(f'loopback filter test, channel {qi}')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.ylim(-100, 80)
    plt.savefig(os.path.join(out_dir, f'{qi}_bw{filter_bw_set}_S21.png'), bbox_inches='tight')
    plt.close()

print(f"Done. Plots saved to {out_dir}")

# --- Make video (1 second per frame) ---
import imageio

frame_paths = [os.path.join(out_dir, f'{qi}_bw{filter_bw_set}_S21.png') for filter_bw_set in bw_values]
video_path = os.path.join(out_dir, f'{qi}_bw_sweep.mp4')

with imageio.get_writer(video_path, fps=1) as writer:
    for path in frame_paths:
        writer.append_data(imageio.imread(path))

print(f"Video saved to {video_path}")
