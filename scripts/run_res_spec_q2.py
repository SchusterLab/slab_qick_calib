"""Run resonator spectroscopy on qi=2 and update readout frequency."""
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

im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'], port=8888)
soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())
cfg_dict = {'soc': soccfg, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}

qi = 2

print(f'\n=== Resonator Spectroscopy qi={qi} ===')
rspec = meas.ResSpec(cfg_dict, qi=qi, style='fine', progress=False)

img_dir = os.path.join(os.path.dirname(__file__), 'chevron_images')
os.makedirs(img_dir, exist_ok=True)
plt.savefig(os.path.join(img_dir, 'res_spec_q2.png'), dpi=150, bbox_inches='tight')
plt.close('all')

if 'freq' in rspec.data:
    print(f'Resonator frequency: {rspec.data["freq"]:.4f} MHz')
elif 'res_freq' in rspec.data:
    print(f'Resonator frequency: {rspec.data["res_freq"]:.4f} MHz')
else:
    print('Available data keys:', list(rspec.data.keys()))
