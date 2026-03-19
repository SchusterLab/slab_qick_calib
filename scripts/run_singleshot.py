"""
Run Rabi flux chevron (length sweep) at the sweet spot.

Usage: python run_singleshot.py
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qick import QickConfig
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

cfg_file = 'sample_50_claude.yml'
expt_path = 'C:\\_Data\\sample_50_claude\\'
cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', cfg_file)
auto_cfg = config.load(cfg_path)

os.makedirs(expt_path, exist_ok=True)

img_dir = os.path.join(os.path.dirname(__file__), 'chevron_images')
os.makedirs(img_dir, exist_ok=True)

print('Connecting to nameserver...')
im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'], port=8888)

soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())
cfg_dict = {'soc': soccfg, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}

qi = 2

print(f'\n=== Rabi Flux Chevron (length sweep) qi={qi} ===')
chevron = meas.RabiFluxChevronExperiment(cfg_dict, qi=qi, params={
    'sweep': 'length',
    'pulse_type': 'const',
    'span_f': 200,
    'expts_f': 40,
    'expts': 200,
    'max_length': 0.5,
    'gain': 0.1,
})
chevron.display(fit=False)
plt.savefig(os.path.join(img_dir, 'rabi_flux_chevron_length.png'), dpi=150, bbox_inches='tight')
plt.close('all')

print('Image saved to scripts/chevron_images/rabi_flux_chevron_length.png')
