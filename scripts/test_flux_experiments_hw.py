"""
Hardware test for flux experiment variants.
Connects to nameserver and runs each flux experiment with minimal settings.

Usage: python test_flux_experiments_hw.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

img_dir = os.path.join(os.path.dirname(__file__), 'flux_test_images')
os.makedirs(img_dir, exist_ok=True)

from qick import QickConfig
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

cfg_file = 'sample_50_rfboard.yml'
expt_path = 'C:\\_Data\\sample_50_flux_test\\'
cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', cfg_file)
auto_cfg = config.load(cfg_path)

os.makedirs(expt_path, exist_ok=True)

print('Connecting to nameserver...')
im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'], port=8888)
print(im)

soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())
cfg_dict = {'soc': soccfg, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}

qi = 2  # Qubit with nonzero sweet_spot_ac
print(f'sweet_spot_ac[{qi}] = {auto_cfg["device"]["qubit"]["sweet_spot_ac"][qi]}')

results = {}

# ============================================================
# 1. Resonator spectroscopy with flux
# ============================================================
print('\n=== 1. Resonator Spectroscopy with Flux ===')
try:
    rspec = meas.ResSpec(cfg_dict, qi=qi, params={
        'flux': True,
        'span': 'kappa',
        'expts': 50,
    }, display=False)
    rspec.display()
    plt.savefig(os.path.join(img_dir, 'resspec_flux.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    results['resspec_flux'] = 'PASSED'
    print(f'  PASSED')
except Exception as e:
    results['resspec_flux'] = f'FAILED: {e}'
    print(f'  FAILED: {e}')

# ============================================================
# 2. Rabi with flux (amplitude sweep)
# ============================================================
print('\n=== 2. Rabi Flux (amp sweep) ===')
try:
    rabi = meas.RabiFluxExperiment(cfg_dict, qi=qi, params={
        'expts': 20,
    }, display=False)
    rabi.display()
    plt.savefig(os.path.join(img_dir, 'rabi_flux_amp.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    pi_len = rabi.data.get('pi_length', 'N/A')
    results['rabi_flux_amp'] = f'PASSED (pi={pi_len})'
    print(f'  PASSED - pi_length: {pi_len}')
except Exception as e:
    results['rabi_flux_amp'] = f'FAILED: {e}'
    print(f'  FAILED: {e}')

# ============================================================
# 3. Rabi with flux (length sweep)
# ============================================================
print('\n=== 3. Rabi Flux (length sweep) ===')
try:
    rabi_len = meas.RabiFluxExperiment(cfg_dict, qi=qi, params={
        'expts': 20,
        'sweep': 'length',
        'pulse_type': 'const',
        'max_length': 0.2,
    }, display=False)
    rabi_len.display()
    plt.savefig(os.path.join(img_dir, 'rabi_flux_len.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    pi_len = rabi_len.data.get('pi_length', 'N/A')
    results['rabi_flux_len'] = f'PASSED (pi={pi_len})'
    print(f'  PASSED - pi_length: {pi_len}')
except Exception as e:
    results['rabi_flux_len'] = f'FAILED: {e}'
    print(f'  FAILED: {e}')

# ============================================================
# 4. Single shot with flux
# ============================================================
print('\n=== 4. Single Shot Flux ===')
try:
    shot = meas.HistogramFluxExperiment(cfg_dict, qi=qi, params={
        'shots': 2000,
    }, display=False)
    shot.display()
    plt.savefig(os.path.join(img_dir, 'single_shot_flux.png'), dpi=150, bbox_inches='tight')
    plt.close('all')
    fid = shot.data.get('fids', [None])[0]
    results['single_shot_flux'] = f'PASSED (fid={fid})'
    print(f'  PASSED - fidelity: {fid}')
except Exception as e:
    results['single_shot_flux'] = f'FAILED: {e}'
    print(f'  FAILED: {e}')

# ============================================================
# Summary
# ============================================================
print('\n========== SUMMARY ==========')
for name, result in results.items():
    print(f'  {name}: {result}')
print('=============================')
