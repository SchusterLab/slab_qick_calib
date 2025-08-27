import scqubits as scq
from ..helpers import config 
import numpy as np
from scipy import constants as cs

phi_0 = cs.h/(2*cs.e)

def lj(ej):
    if ej is None:
        return None
    return (phi_0/2/np.pi)**2 / (cs.h*ej*1e9)*1e9

def gL(Delta, chiL):
    if Delta is None or chiL is None:
        return None
    return np.sqrt(-Delta*chiL)

def gX(Delta, chi, alpha):
    if Delta is None or chi is None or alpha is None:
        return None
    return np.sqrt(Delta * (Delta + alpha) / alpha * chi)/2

def gX_CR(Delta, Sum, chi, alpha):
    if Delta is None or Sum is None or chi is None or alpha is None:
        return None
    return np.sqrt(chi/ alpha / (1/Delta/(Delta + alpha) + 1/Sum/(Sum - alpha)))/2

def gL_CR(Delta, Sum, chiL):
    if Delta is None or Sum is None or chiL is None:
        return None
    return np.sqrt(-chiL / (1/Delta + 1/Sum))

def chi(alpha, Delta, g):
    if alpha is None or Delta is None or g is None:
        return None
    return alpha*g**2/Delta/(Delta + alpha)

def chiL(g, Delta, Sum):
    if g is None or Delta is None or Sum is None:
        return None
    return -g**2*(1/Delta + 1/Sum)

#def chi_DR(alpha, Delta, Sum, g): 
def T1p(kappa, Delta, g):
    if kappa is None or Delta is None or g is None:
        return None
    return 1/kappa*(Delta/g)**2/2/np.pi

def ng(Delta, g):
    if Delta is None or g is None:
        return None
    return (Delta/g)**2/4

def T1px(kappa, chi, alpha):
    if kappa is None or chi is None or alpha is None:
        return None
    return alpha/kappa/chi/2/np.pi

def T1px_opt(chi, alpha):
    if chi is None or alpha is None:
        return None
    return alpha/chi**2/4/np.pi

def Tphi(T1, T2):
    if T1 is None or T2 is None or T1 == 0 or T2 == 0:
        return None
    if T2 >= 2*T1:
        return np.inf
    return 1/(1/T2 - 1/(2*T1))

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
        kappa_val = model_cfg.kappa_low[i]
        if kappa_val is None or kappa_val == 0 or Delta is None or Delta == 0 or g_lamb is None or g_lamb == 0:
            T1purcell = None
        else:
            T1purcell = T1p(kappa_val, Delta, g_lamb)
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
        T1_val = auto_cfg.device.qubit.T1[i]
        T2e_val = auto_cfg.device.qubit.T2e[i]
        if T1_val is None or T2e_val is None or T1_val == 0 or T2e_val == 0:
            TPhi = None
        else:
            TPhi = Tphi(T1_val, T2e_val)
        config.update_config(model_name, None, 'Tphi', TPhi, index=i, verbose=False, sig=4)
        T1_val = auto_cfg.device.qubit.T1[i]
        T1_purcell_val = model_cfg.T1_purcell[i]
        if T1_val is None or T1_purcell_val is None or T1_val == 0 or T1_purcell_val == 0:
            T1nopurcell = None
        else:
            try:
                denominator = 1/T1_val - 1/T1_purcell_val
                T1nopurcell = 1/denominator if denominator != 0 else None
            except Exception:
                T1nopurcell = None
        config.update_config(model_name, None, 'T1_nopurcell', T1nopurcell, index=i, verbose=False, sig=4)
