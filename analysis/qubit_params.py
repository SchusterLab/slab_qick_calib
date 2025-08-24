import scqubits as scq
from ..helpers import config 
import numpy as np
from scipy import constants as cs

phi_0 = cs.h/(2*cs.e)

def lj(ej): return (phi_0/2/np.pi)**2 / (cs.h*ej*1e9)*1e9
def gL(Delta, chiL): return np.sqrt(-Delta*chiL)
def gX(Delta, chi, alpha): return np.sqrt(Delta * (Delta + alpha) / alpha * chi)/2
def gX_CR(Delta, Sum, chi, alpha): return np.sqrt(chi/ alpha / (1/Delta/(Delta + alpha) + 1/Sum/(Sum - alpha)))/2
def gL_CR(Delta, Sum, chiL): return np.sqrt(-chiL / (1/Delta + 1/Sum))
def chi(alpha, Delta, g): return alpha*g**2/Delta/(Delta + alpha)
def chiL(g, Delta, Sum): return -g**2*(1/Delta + 1/Sum)
#def chi_DR(alpha, Delta, Sum, g): 
def T1p(kappa, Delta, g): return 1/kappa*(Delta/g)**2/2/np.pi
def ng(Delta, g): return (Delta/g)**2/4
def T1px(kappa, chi, alpha): return alpha/kappa/chi/2/np.pi
def T1px_opt(chi, alpha): return alpha/chi**2/4/np.pi
def Tphi(T1, T2): return 1/(1/T2 - 1/(2*T1))

def ham(cfg_path): 
    auto_cfg = config.load(cfg_path)
    model_name = cfg_path[0:-4] + '_model.yml'
    model_cfg = config.load(cfg_path[0:-4] + '_model.yml')
    for i in np.arange(len(auto_cfg.device.qubit.f_ge)):
        alpha =  auto_cfg.device.qubit.f_ef[i] - auto_cfg.device.qubit.f_ge[i]
        en = scq.Transmon.find_EJ_EC(auto_cfg.device.qubit.f_ge[i]/1000,alpha/1000)
        config.update_config(model_name, None, 'alpha', alpha, index=i, verbose=False, sig=4)
        config.update_config(model_name, None, 'Ej', en[0], index=i, verbose=False, sig=4)
        config.update_config(model_name, None, 'Ec', en[1], index=i, verbose=False, sig=4)
        config.update_config(model_name, None, 'ratio', en[0]/en[1], index=i, verbose=False, sig=4)

        gX = gX_CR(auto_cfg.device.qubit.f_ge[i], auto_cfg.device.readout.frequency[i], 
                   auto_cfg.device.readout.chi[i], alpha)
        config.update_config(model_name, None, 'g_chi', gX, index=i, verbose=False, sig=4)
    # update the model configuration with the calculated EJ and EC, alpha values

def delta(cfg_path):
    model_name = cfg_path[0:-4] + '_model.yml'
    auto_cfg = config.load(cfg_path)
    model_cfg = config.load(cfg_path[0:-4] + '_model.yml')
    for i in np.arange(len(auto_cfg.device.qubit.f_ge)):
        Delta = -(auto_cfg.device.qubit.f_ge[i] - auto_cfg.device.readout.frequency[i])
        config.update_config(model_name, None, 'Delta', Delta, index=i, verbose=False, sig=4)
        Sum = auto_cfg.device.qubit.f_ge[i] + auto_cfg.device.readout.frequency[i]
        config.update_config(model_name, None, 'Sum', Sum, index=i, verbose=False, sig=4)
        g_lamb = gL_CR(Delta, Sum, auto_cfg.device.readout.lamb[i])
        config.update_config(model_name, None, 'g_lamb', g_lamb, index=i, verbose=False, sig=4)
        #T1purcell = T1p(auto_cfg.device.readout.kappa[i], Delta, g_lamb)
        T1purcell = T1p(model_cfg.kappa_low[i], Delta, g_lamb)
        config.update_config(model_name, None, 'T1_purcell', T1purcell, index=i, verbose=False, sig=4)
        nG = ng(Delta, g_lamb)
        config.update_config(model_name, None, 'ng', nG, index=i, verbose=False, sig=4)



def cohere(cfg_path): 
    model_name = cfg_path[0:-4] + '_model.yml'
    model_cfg = config.load(cfg_path[0:-4] + '_model.yml')
    auto_cfg = config.load(cfg_path)
    for i in np.arange(len(auto_cfg.device.qubit.f_ge)):
        q = np.pi*2 * auto_cfg.device.qubit.f_ge[i] * auto_cfg.device.qubit.T1[i]
        config.update_config(model_name, None, 'Q1', q/1e6, index=i, verbose=False, sig=4)
        TPhi = Tphi(auto_cfg.device.qubit.T1[i], auto_cfg.device.qubit.T2e[i])
        config.update_config(model_name, None, 'Tphi', TPhi, index=i, verbose=False, sig=4)
        T1nopurcell = 1/( 1/auto_cfg.device.qubit.T1[i] - 1/model_cfg.T1_purcell[i])
        config.update_config(model_name, None, 'T1_nopurcell', T1nopurcell, index=i, verbose=False, sig=4)
