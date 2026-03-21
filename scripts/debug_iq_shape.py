"""Debug: check iq_list shape from T1ContFluxProgram."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')

from qick import QickConfig
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
import slab_qick_calib.experiments as meas
from slab_qick_calib.helpers import rfboard, config
from slab_qick_calib.experiments.flux.t1_cont_flux import T1ContFluxProgram
from slab_qick_calib.exp_handling.datamanagement import AttrDict

cfg_file = 'sample_50_rfboard.yml'
expt_path = 'C:\\_Data\\sample_50_new\\'
cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', cfg_file)
auto_cfg = config.load(cfg_path)

im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'], port=8888)
soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())
cfg_dict = {'soc': soccfg, 'expt_path': expt_path, 'cfg_file': cfg_path, 'im': im}

qi = 2
auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)

# Minimal experiment setup to get cfg
from slab_qick_calib.experiments.general.qick_experiment import QickExperimentLoop
expt = QickExperimentLoop(cfg_dict=cfg_dict, prefix="debug", qi=qi, check_params=False)

expt.cfg.expt = {
    "reps": 100,
    "rounds": 1,
    "n_e": 2,
    "n_g": 1,
    "n_t1": 8,
    "calib_wait_short": 0.04,
    "calib_wait_long": 15.0,
    "wait_time": 0.5,
    "flux": True,
    "flux_chan": expt.cfg.hw.soc.dacs.flux.ch[qi],
    "flux_gain": 0.15,
    "flux_readout_wait": 0.1,
    "final_delay": expt.cfg.device.qubit.T1[qi] * 6,
    "qubit": [qi],
    "qubit_chan": expt.cfg.hw.soc.adcs.readout.ch[qi],
    "active_reset": False,
}

final_delay = expt.cfg.device.readout.final_delay[qi]
prog = T1ContFluxProgram(soccfg=soccfg, final_delay=final_delay, cfg=expt.cfg)
iq_list = prog.acquire(im[expt.cfg.aliases.soc], rounds=1, threshold=None, progress=False)

print(f"type(iq_list): {type(iq_list)}")
print(f"len(iq_list): {len(iq_list)}")
for i, item in enumerate(iq_list):
    arr = np.array(item)
    print(f"  iq_list[{i}] shape: {arr.shape}, dtype: {arr.dtype}")

iq_array = np.array(iq_list)
print(f"\nnp.array(iq_list) shape: {iq_array.shape}")
