import numpy as np
from qick import *

from ...exp_handling.datamanagement import AttrDict
from datetime import datetime
from ...analysis import fitting as fitter
from ..general.qick_experiment_2q import QickExperiment2Q
from ..general.qick_program import QickProgram2Q
from qick.asm_v2 import QickSweep1D
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt


# class T1ContProgram(QickProgram):
#     def __init__(self, soccfg, final_delay, cfg):
#         super().__init__(soccfg, final_delay=final_delay, cfg=cfg)

#     def _initialize(self, cfg):
#         cfg = AttrDict(self.cfg)
#         self.add_loop("shot_loop", cfg.expt.shots)
#         super()._initialize(cfg, readout="standard")

#         super().make_cfg_pulse(
#             cfg.expt.qubit[0], cfg.device.qubit.f_ge, "pi_ge"
#         )

#     def _body(self, cfg):

#         cfg = AttrDict(self.cfg)

#         self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)
#         # First, n ground state measurement
#         self.delay_auto(t=0.01, tag=f"readout0_delay_1")
#         for i in range(cfg.expt.n_g):
#             self.measure(cfg)
#         self.delay_auto(t=cfg.expt['readout']+0.01, tag=f"readout_delay_1_{i}")

#         # Then, m excited state measurement
#         for i in range(cfg.expt.n_e):
#             self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
#             self.delay_auto(t=0.01, tag=f"wait0_{i}")
#             self.measure(cfg)
#             if cfg.expt.active_reset:
#                 self.reset(3,0,i)
#                 self.delay_auto(t=cfg.expt['readout'] + 0.01, tag=f"final_delay_0_{i}")
#             else:
#                 self.delay_auto(t=cfg.expt["final_delay"] + 0.01, tag=f"final_delay_1_{i}")

#         for i in range(cfg.expt.n_t1):
#             self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
#             self.delay_auto(t=cfg.expt["wait_time"] + 0.01, tag=f"wait_{i}")
#             self.measure(cfg)
#             if cfg.expt.active_reset:
#                 self.reset(3,1,i)
#                 self.delay_auto(t=cfg.expt['readout'] + 0.01, tag=f"final_delay_{i}")
#             else:
#                 self.delay_auto(t=cfg.expt["final_delay"] + 0.01, tag=f"final_delay_{i}")


#     def measure(self, cfg):

#         self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
#         if self.lo_ch is not None:
#             self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.01)
#         self.trigger(ros=[self.adc_ch],pins=[0],t=self.trig_offset)


#     def reset(self, i,j,k):
#         # Perform active reset i times
#         cfg = AttrDict(self.cfg)
#         for n in range(i):
#             self.wait_auto(cfg.expt.read_wait)
#             self.delay_auto(cfg.expt.read_wait + cfg.expt.extra_delay)

#             # read the input, test a threshold, and jump if it is met [so, if i<threshold, doesn't do pi pulse]
#             self.read_and_jump(ro_ch=self.adc_ch, component='I', threshold=cfg.expt.threshold, test='<', label=f'NOPULSE{n}{j}{k}')

#             self.pulse(ch=self.qubit_ch, name="pi_ge", t=0)
#             self.delay_auto(0.01)
#             self.label(f"NOPULSE{n}{j}{k}")

#             if n<i-1:
#                 self.trigger(ros=[self.adc_ch], pins=[0], t=self.trig_offset)
#                 self.pulse(ch=self.res_ch, name="readout_pulse", t=0)
#                 if self.lo_ch is not None:
#                     self.pulse(ch=self.lo_ch, name="mix_pulse", t=0.0)
#                 self.delay_auto(0.01)

#     def collect_shots(self, offset=0):
#         return super().collect_shots(offset=0)


# class T1ContExperiment(QickExperiment):
#     """
#     self.cfg.expt: dict
#         A dictionary containing the configuration parameters for the T1 experiment. The keys and their descriptions are as follows:
#         - span (float): The total span of the wait time sweep in microseconds.
#         - expts (int): The number of experiments to be performed.
#         - reps (int): The number of repetitions for each experiment (inner loop)
#         - rounds (int): The number of rounds for the experiment (outer loop)
#         - qubit (int): The index of the qubit being used in the experiment.
#         - qubit_chan (int): The channel of the qubit being read out.
#     """

#     def __init__(
#         self,
#         cfg_dict,
#         qi=0,
#         go=True,
#         params={},
#         prefix=None,
#         progress=True,
#         style="",
#         disp_kwargs=None,
#         min_r2=None,
#         max_err=None,
#         display=True,
#     ):

#         if prefix is None:
#             prefix = f"t1_cont_qubit{qi}"

#         super().__init__(cfg_dict=cfg_dict, prefix=prefix, progress=progress, qi=qi)

#         params_def = {
#             "shots": 50000,
#             'reps': 1,
#             "rounds": self.rounds,
#             "wait_time": self.cfg.device.qubit.T1[qi],
#             'active_reset': self.cfg.device.readout.active_reset[qi],
#             'final_delay': self.cfg.device.qubit.T1[qi]*6,
#             'readout': self.cfg.device.readout.readout_length[qi],
#             'n_g':1,
#             'n_e':2,
#             'n_t1':7,
#             "qubit": [qi],
#             "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
#         }

#         self.cfg.expt = {**params_def, **params}
#         super().check_params(params_def)
#         if self.cfg.expt.active_reset:
#             super().configure_reset()

#         if not self.cfg.device.qubit.tuned_up[qi] and disp_kwargs is None:
#             disp_kwargs = {'plot_all': True}
#         if go:
#             super().run(display=display, progress=progress, min_r2=min_r2, max_err=max_err, disp_kwargs=disp_kwargs)

#     def acquire(self, progress=False, get_hist=True):
#         self.param = {"label": "wait_0", "param": "t", "param_type": "time"}

#         if 'active_reset' in self.cfg.expt and self.cfg.expt.active_reset:
#             final_delay = self.cfg.device.readout.readout_length[self.cfg.expt.qubit[0]]
#         else:
#             final_delay = self.cfg.device.readout.final_delay[self.cfg.expt.qubit[0]]
#         prog = T1ContProgram(soccfg=self.soccfg,final_delay=final_delay,cfg=self.cfg,)

#         now = datetime.now()
#         current_time = now.strftime("%Y-%m-%d %H:%M:%S")
#         current_time = current_time.encode("ascii", "replace")

#         iq_list = prog.acquire(
#             self.im[self.cfg.aliases.soc],
#             rounds=self.cfg.expt.rounds,
#             threshold=None,
#             load_pulses=True,
#             progress=progress,
#         )
#         xpts = self.get_params(prog)


#         if get_hist:
#             v, hist = self.make_hist(prog)

#         data = {
#             "xpts": xpts,
#             "start_time": current_time,
#         }

#         nms = ['g','e','t1']
#         if self.cfg.expt.active_reset:
#             start_ind = [0, self.cfg.expt.n_g, self.cfg.expt.n_g+self.cfg.expt.n_e*3, len(iq_list[0])]
#         else:
#             start_ind = [0, self.cfg.expt.n_g, self.cfg.expt.n_g+self.cfg.expt.n_e, len(iq_list[0])]
#         iq_array = np.array(iq_list)
#         for i in range(3):
#             nm=nms[i]
#             if self.cfg.expt.active_reset:
#                 inds = np.arange(start_ind[i], start_ind[i+1], 3)
#             else:

#                 inds = np.arange(start_ind[i], start_ind[i+1])
#             #data['amps_'+nm] = np.abs(iq_array[0,inds,0].dot([1, 1j]))
#             #data['phases_'+nm] = np.angle(iq_array[0,inds,:].dot([1, 1j]))
#             data['avgi_'+nm] = iq_array[0,inds,:, 0]
#             data['avgq_'+nm] = iq_array[0,inds,:, 1]

#         if get_hist:
#             data["bin_centers"] = v
#             data["hist"] = hist

#         for key in data:
#             data[key] = np.array(data[key])
#         self.data = data

#         return self.data

#     def analyze(self, data=None, **kwargs):
#         if data is None:
#             data = self.data

#         # fitparams=[y-offset, amp, x-offset, decay rate]

#         return data

#     def display(
#         self, data=None, fit=True, plot_all=False, ax=None, show_hist=True, rescale=False,savefig=True,**kwargs
#     ):
#         qubit = self.cfg.expt.qubit[0]
#         nexp = self.cfg.expt.n_g+self.cfg.expt.n_e+self.cfg.expt.n_t1
#         n_t1 = self.cfg.expt.n_t1
#         pi_time = 0.4 # Fix me
#         if self.cfg.expt.active_reset:
#             n_reset = 3
#             pulse_length = self.cfg.expt.readout*(self.cfg.expt.n_g+n_reset*(self.cfg.expt.n_e+n_t1))+self.cfg.expt.wait_time*n_t1+nexp*pi_time
#         else:
#             pulse_length = self.cfg.expt.readout*nexp+self.cfg.expt.wait_time*n_t1+self.cfg.expt.final_delay*(self.cfg.expt.n_e+n_t1)+nexp*pi_time
#         pulse_length = pulse_length/1e6
#         navg = 100
#         nred = int(np.floor(navg/10))

#         if show_hist:  # Plot histogram of shots if show_hist is True
#             fig2, ax = plt.subplots(1, 1, figsize=(3, 3))
#             ax.hist(data['avgi_e'].flatten(), bins=50, alpha=0.6, color='r', label='Excited State', density=True)
#             #ax.legend()
#             ax.hist(data['avgi_g'].flatten(), bins=50, alpha=0.6, color='b', label='ground State', density=True)

#             # try:
#             #     ax.plot(data['bin_centers'], two_gaussians_decay(data['bin_centers'], *data['hist_fit']), label='Fit')
#             # except:
#             #     pass
#             ax.set_xlabel("I [ADC units]")
#             ax.set_ylabel("Probability")
#         m=0.2
#         fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

#         for i in range(len(data['avgi_e'])):
#             ax[0].plot(data['avgi_e'][i],'b.',markersize=m)
#             ax[1].plot(data['avgq_e'][i],'b.',markersize=m)

#         for i in range(len(data['avgi_g'])):
#             ax[0].plot(data['avgi_g'][i],'k.',markersize=m)
#             ax[1].plot(data['avgq_g'][i],'k.',markersize=m)
#         for i in range(len(data['avgi_t1'])):
#             ax[0].plot(data['avgi_t1'][i],'r.',markersize=m)
#             ax[1].plot(data['avgq_t1'][i],'r.',markersize=m)

#         t1_data = data['avgi_t1'].transpose().flatten()
#         g_data = data['avgi_g'].transpose().flatten()
#         e_data = data['avgi_e'].transpose().flatten()

#         fig, ax = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
#         smoothed_t1_data = uniform_filter1d(t1_data, size=navg*self.cfg.expt.n_t1)
#         smoothed_t1_data=smoothed_t1_data[::nred*self.cfg.expt.n_t1]
#         npts = len(smoothed_t1_data)
#         times = np.arange(npts)*pulse_length*nred
#         ax[0].plot(times,smoothed_t1_data, 'k.-', linewidth=0.1, markersize=m, label='Smoothed T1 Data')
#         smoothed_g_data = uniform_filter1d(g_data, size=navg*self.cfg.expt.n_g)
#         smoothed_g_data=smoothed_g_data[::nred*self.cfg.expt.n_g]
#         ax[1].plot(times,smoothed_g_data, 'k.-',linewidth=0.1,markersize=m, label='Smoothed g Data')
#         smoothed_e_data = uniform_filter1d(e_data, size=navg*self.cfg.expt.n_e)
#         smoothed_e_data=smoothed_e_data[::nred*self.cfg.expt.n_e]
#         ax[2].plot(times,smoothed_e_data, 'k.-',linewidth=0.1,markersize=m, label='Smoothed e Data')
#         dv = smoothed_e_data - smoothed_g_data
#         pt1 = (smoothed_t1_data-smoothed_g_data)/dv
#         ax[3].plot(times,pt1, 'k.-',linewidth=0.1,markersize=m, label='Smoothed e Data')
#         ax[3].axhline(np.exp(-1), color='r', linestyle='--', label='$e^{-1}$')


#         ax[0].set_ylabel('I (ADC), $T =T_1$')
#         ax[1].set_ylabel('I (ADC), $g$ state')
#         ax[2].set_ylabel('I (ADC), $e$ state')
#         ax[3].set_ylabel('$(v_{t1}-v_g)/(v_e-v_g)$')

#         fig, ax = plt.subplots(1, 1, figsize=(15, 4))
#         t1m = - 1 / np.log(pt1)
#         ax.plot(times, t1m, 'k.-', linewidth=0.1, markersize=m, label='T1 Data')
#         ax.set_xlabel('Time (s)')
#         ax.set_ylabel('$T_1/$tau')

#         fig2, ax = plt.subplots(1, 1, figsize=(14, 4))
#         ax.plot(times,smoothed_t1_data, 'k.-', linewidth=0.1, markersize=m, label='Smoothed T1 Data')
#         ax2 = ax.twinx()
#         ax2.plot(times,smoothed_e_data, 'b.-',linewidth=0.1,markersize=m, label='Smoothed e Data')
#         ax3 = ax.twinx()
#         ax3.plot(times,smoothed_g_data, 'r.-',linewidth=0.1,markersize=m, label='Smoothed e Data')

#         if savefig:
#             fig.tight_layout()
#             imname = self.fname.split("\\")[-1]
#             fig.savefig(
#                 self.fname[0 : -len(imname)] + "images\\" + imname[0:-3] + ".png"
#             )
#             plt.show()

#         #ax[3].legend()

#         #ax.legend()
