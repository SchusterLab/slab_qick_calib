"""Run a coarse resonator spectroscopy experiment."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qick import QickConfig
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import config

cfg_file = 'sample_50_rfboard.yml'
expt_path = 'C:\\_Data\\sample_50_new\\'
cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', cfg_file)
auto_cfg = config.load(cfg_path)

im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'])
soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())
cfg_dict = {'soc': soccfg, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}

print('=== Running coarse resonator spectroscopy ===')
qi = 2
params = {'start': 3000, 'span': 5000, 'reps': 1000, 'gain': 0.1, 'expts': 1000}
rspecc = meas.ResSpec(cfg_dict, qi=qi, style='coarse', progress=False, params=params)
print('Coarse peaks:', rspecc.data['coarse_peaks'])
print('Done!')
