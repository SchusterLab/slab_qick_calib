"""
Setup script replicating notebook cells up to check_qick_issues.
Run this first, then import cfg_dict to run experiments.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script use
import matplotlib.pyplot as plt

from qick import QickConfig

from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import rfboard, qick_check, config, handy

# --- Cell 2: Config ---
cfg_file = 'sample_50_rfboard.yml'
expt_path = 'C:\\_Data\\sample_50_new\\'
max_t1 = 5

# --- Cell 7: Paths ---
configs_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
cfg_file_path = os.path.join(configs_dir, cfg_file)
images_dir = os.path.join(expt_path, 'images')
summary_dir = os.path.join(images_dir, 'summary')

print('Data will be stored in', expt_path)

# --- Cell 9: Connect to instruments ---
cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', cfg_file)
auto_cfg = config.load(cfg_path)

im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'])
print(im)

soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())

cfg_dict = {'soc': soccfg, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}

print(soccfg)

# --- Checks ---
print("\n=== Checking frequencies ===")
qick_check.check_freqs(0, cfg_dict)

print("\n=== Checking resonances ===")
qick_check.check_resonances(cfg_dict)

fnyq = cfg_dict['soc']._get_ch_cfg(ro_ch=0)['f_dds'] / 2
clock_tick = 1e3 * cfg_dict['soc'].cycles2us(1)
print(f'\nADC Nyquist frequency is {fnyq} MHz')
print(f'1 clock tick is {clock_tick} ns')

print("\n=== Checking ADC ===")
qick_check.check_adc(cfg_dict)

print("\n=== Setup complete. cfg_dict is ready. ===")


if __name__ == '__main__':
    # If run directly, just do setup. You can add experiment calls below.
    # Example:
    # result = meas.some_experiment(cfg_dict, qubit=0)
    pass
