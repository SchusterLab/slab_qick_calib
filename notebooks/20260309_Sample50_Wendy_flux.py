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

%load_ext autoreload
%autoreload 2

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
# %%
print(soccfg)

# %%
update = False

# Per-qubit experiment params (RF attenuation/filter settings come from config)
qubit_list = [2]

for qi in qubit_list:
    # Activate this qubit's RF settings from config
    auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)

    rspec = meas.ResSpec(cfg_dict, qi=qi, params={'span':'kappa',})

    if update:
        rspec.update()
        auto_cfg = config.load(cfg_path)
# %%
update=False

qubit_list = [2]
for qi in qubit_list: 
    filter_center_freq = 3420
    bw = 750
    # set up filter and attenuation  
    rfboard.setup_qubit_drive_chain(qi, soc, auto_cfg, bw=bw, atten1_dac=30, atten2_dac=25, center_freq=filter_center_freq)
    start_freq = filter_center_freq - 100/2
    params={'start':start_freq, 'span':100, 'gain':0.5, 'expts':200, 'reps':1000, 'rounds':10, 'sep_readout': True, 'length':1, 'readout_length':8}
    
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
for flux in [0,0.05,0.1]:
    print(f"Setting flux bias to {flux:.2f} V")
    soc.rfb_set_bias(3, flux)
    time.sleep(0.1)  # wait for bias to settle


    #Resonator measurement
    # Per-qubit experiment params (RF attenuation/filter settings come from config)

    # Activate this qubit's RF settings from config
    auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)

    rspec = meas.ResSpec(cfg_dict, qi=qi, params={'span':'kappa',})
    rspec.update()
    auto_cfg = config.load(cfg_path)
    
    filter_start_freq = 3420
    bw = 1150
    # set up filter and attenuation  
    rfboard.setup_qubit_drive_chain(qi, soc, auto_cfg, bw=bw, atten1_dac=30, atten2_dac=25, center_freq=filter_center_freq-bw/2)
    start_freq = filter_start_freq
    params={'start':start_freq, 'span':100, 'gain':0.5, 'expts':200, 'reps':1000, 'rounds':10, 'sep_readout': True, 'length':1, 'readout_length':8}
    
    qspec=meas.QubitSpec(cfg_dict, qi=qi, style='fine', params=params)
    plt.show()
# %%
