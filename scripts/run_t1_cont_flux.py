"""Run T1ContFlux experiment."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qick import QickConfig
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import rfboard, config

cfg_file = 'sample_50_rfboard.yml'
expt_path = 'C:\\_Data\\sample_50_new\\'
cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', cfg_file)
auto_cfg = config.load(cfg_path)

im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'], port=8888)
soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())
cfg_dict = {'soc': soccfg, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}

qi = 2
params = {
    "gain_start": 0.16,
    "gain_stop": -0.28,
    "expts_gain": 300,
    "lin_freq": True,
    "t1_max": 10,
    "reps": 2000,
    "direction": "neg",
}
auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)
expt = meas.T1ContFluxExperiment(cfg_dict, qi=qi, params=params)
