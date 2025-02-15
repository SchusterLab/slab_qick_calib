import matplotlib.pyplot as plt
import numpy as np
from qick import *

from exp_handling.datamanagement import AttrDict
from tqdm import tqdm_notebook as tqdm

import scipy as sp

import fitting as fitter

"""
Averager program that takes care of the standard pulse loading for basic X, Y, Z +/- pi and pi/2 pulses.
"""
class CliffordAveragerProgram(AveragerProgram):
    # def update(self):
    #     pass

    def __init__(self, soccfg, cfg):
        self.cfg = AttrDict(cfg)
        self.cfg.update(self.cfg.expt)

        # copy over parameters for the acquire method
        self.cfg.reps = cfg.expt.reps
        
        super().__init__(soccfg, self.cfg)

    """
    Wrappers to load and play pulses.
    If play is false, must specify all parameters and all params will be saved (load params).

    If play is true, uses the default values saved from the load call, temporarily overriding freq, phase, or gain if specified to not be None. Sets the pulse registers with these settings and plays the pulse. If you want to set freq, phase, or gain via registers/update,
    be sure to set the default value to be None at loading time.

    If play is True, registers will automatically be set regardless of set_reg flag.
    If play is False, registers will be set based on value of set_reg flag, but pulse will not be played.
    """
    def handle_const_pulse(self, name, waveformname=None, ch=None, length=None, freq_MHz=None, phase_deg=None, gain=None, reload=True, play=False, set_reg=False, flag=None, phrst=0):
        """
        Load/play a constant pulse of given length.
        """
        if name is not None and (name not in self.pulse_dict.keys() or reload):
            assert ch is not None
            self.pulse_dict.update({name:dict(ch=ch, name=name, type='const', length=length, freq_MHz=freq_MHz, phase_deg=phase_deg, gain=gain, flag=flag)})
        if play or set_reg:
            assert name in self.pulse_dict.keys()
            # if not (ch == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None: params['freq_MHz'] = freq_MHz
            if phase_deg is not None: params['phase_deg'] = phase_deg
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='const', freq=self.freq2reg(params['freq_MHz'], gen_ch=params['ch']), phase=self.deg2reg(params['phase_deg'], gen_ch=params['ch']), gain=params['gain'], length=params['length'], phrst=phrst)
            if play:
                self.pulse(ch=params['ch'])
                self.sync_all()

    def handle_gauss_pulse(self, name, waveformname=None, ch=None, sigma=None, freq_MHz=None, phase_deg=None, gain=None, reload=True, play=False, set_reg=False, flag=None, phrst=0):
        """
        Load/play a gaussian pulse of length 4 sigma on channel ch
        """
        if name not in self.pulse_dict.keys() or reload:
            assert None not in [ch, sigma]
            if waveformname is None: waveformname = name
            self.pulse_dict.update({name:dict(ch=ch, name=name, waveformname=waveformname, type='gauss', sigma=sigma, freq_MHz=freq_MHz, phase_deg=phase_deg, gain=gain, flag=flag)})
            if reload or waveformname not in self.pulses[ch].keys():
                self.add_gauss(ch=ch, name=waveformname, sigma=sigma, length=sigma*4)
                # print('added gauss pulse', name, 'on ch', ch)
        if play or set_reg:
            # if not (ch == sigma == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None: params['freq_MHz'] = freq_MHz
            if phase_deg is not None: params['phase_deg'] = phase_deg
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='arb', freq=self.freq2reg(params['freq_MHz'], gen_ch=params['ch']), phase=self.deg2reg(params['phase_deg'], gen_ch=params['ch']), gain=params['gain'], waveform=params['waveformname'], phrst=phrst)
            if play:
                # print('playing gauss pulse', params['name'], 'on ch', params['ch'])
                self.pulse(ch=params['ch'])
                self.sync_all()

    def handle_flat_top_pulse(self, name, waveformname=None, ch=None, sigma=3, flat_length=None, freq_MHz=None, phase_deg=None, gain=None, reload=True, play=False, set_reg=False, flag=None, phrst=0):
        """
        Plays a gaussian ramp up (2*sigma), a constant pulse of length flat_length+4*sigma,
        plus a gaussian ramp down (2*sigma) on channel ch.
        By default: sigma=3 clock cycles
        """
        if name not in self.pulse_dict.keys() or reload:
            assert None not in [ch, sigma, flat_length]
            if waveformname is None: waveformname = name
            self.pulse_dict.update({name:dict(ch=ch, name=name, waveformname=waveformname, type='flat_top', sigma=sigma, flat_length=flat_length, freq_MHz=freq_MHz, phase_deg=phase_deg, gain=gain, flag=flag)})
            if reload or waveformname not in self.pulses[ch].keys():
                # print('all waveforms')
                # for i_ch in range(len(self.pulses)):
                #     print(self.pulses[i_ch].keys())
                self.add_gauss(ch=ch, name=waveformname, sigma=sigma, length=sigma*4)
                # print('added', waveformname, 'ch', ch)
                # print(self.gen_chs.keys())
        if play or set_reg:
            # if not (ch == name == sigma == length == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            assert name in self.pulse_dict.keys()
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None: params['freq_MHz'] = freq_MHz
            if phase_deg is not None: params['phase_deg'] = phase_deg
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='flat_top', freq=self.freq2reg(params['freq_MHz'], gen_ch=params['ch']), phase=self.deg2reg(params['phase_deg'], gen_ch=params['ch']), gain=params['gain'], waveform=params['waveformname'], length=params['flat_length'], phrst=phrst)
            if play:
                self.pulse(ch=params['ch'])
                self.sync_all()

    def handle_mux4_pulse(self, name, ch=None, mask=None, length=None, reload=True, play=False, set_reg=False, flag=None):
        """
        Load/play a constant pulse of given length on the mux4 channel.
        """
        if name is not None and reload: # and name not in self.pulse_dict.keys():
            assert ch is not None
            assert ch == 6, 'Only ch 6 on q3diamond supports mux4 currently!'
            self.pulse_dict.update({name:dict(ch=ch, name=name, type='mux4', mask=mask, length=length, flag=flag)})
        if play or set_reg:
            assert name in self.pulse_dict.keys()
            params = self.pulse_dict[name].copy()
            if mask is not None: params['mask'] = mask
            if length is not None: params['length'] = length
            self.set_pulse_registers(ch=params['ch'], style='const', length=length, mask=mask)
            if play:
                self.pulse(ch=params['ch'])
                self.sync_all()

    # mu, beta are dimensionless
    def add_adiabatic(self, ch, name, mu, beta, period_us):
        period = self.us2cycles(period_us, gen_ch=ch)
        gencfg = self.soccfg['gens'][ch]
        maxv = gencfg['maxv']*gencfg['maxv_scale']
        samps_per_clk = gencfg['samps_per_clk']
        length = np.round(period) * samps_per_clk
        period *= samps_per_clk
        t = np.arange(0, length)
        iamp, qamp = fitter.adiabatic_iqamp(t, amp_max=1, mu=mu, beta=beta, period=period)
        self.add_pulse(ch=ch, name=name, idata=maxv*iamp, qdata=maxv*qamp)

    def handle_adiabatic_pulse(self, name, waveformname=None, ch=None, mu=None, beta=None, period_us=None, freq_MHz=None, phase_deg=None, gain=None, reload=True, play=False, set_reg=False, flag=None, phrst=0):
        """
        Load/play an adiabatic pi pulse on channel ch
        """
        if name not in self.pulse_dict.keys() or reload:
            assert None not in [ch, mu, beta, period_us]
            if waveformname is None: waveformname = name
            self.pulse_dict.update({name:dict(ch=ch, name=name, waveformname=waveformname, type='adiabatic', mu=mu, beta=beta, period_us=period_us, freq_MHz=freq_MHz, phase_deg=phase_deg, gain=gain, flag=flag)})
            if reload or waveformname not in self.pulses[ch].keys():
                self.add_adiabatic(ch=ch, name=waveformname, mu=mu, beta=beta, period_us=period_us)
                # print('added gauss pulse', name, 'on ch', ch)
        if play or set_reg:
            # if not (ch == sigma == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None: params['freq_MHz'] = freq_MHz
            if phase_deg is not None: params['phase_deg'] = phase_deg
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='arb', freq=self.freq2reg(params['freq_MHz'], gen_ch=params['ch']), phase=self.deg2reg(params['phase_deg'], gen_ch=params['ch']), gain=params['gain'], waveform=params['waveformname'], phrst=phrst)
            if play:
                # print('playing gauss pulse', params['name'], 'on ch', params['ch'])
                self.pulse(ch=params['ch'])
                self.sync_all()

    # I_mhz_vs_us, Q_mhz_vs_us = functions of time in us, in units of MHz
    # times_us = times at which I_mhz_vs_us and Q_mhz_vs_us are defined
    def add_IQ(self, ch, name, I_mhz_vs_us, Q_mhz_vs_us, times_us):
        gencfg = self.soccfg['gens'][ch]
        maxv = gencfg['maxv']*gencfg['maxv_scale'] - 1
        samps_per_clk = gencfg['samps_per_clk']
        times_cycles = np.linspace(0, self.us2cycles(times_us[-1], gen_ch=ch), len(times_us))
        times_samps = samps_per_clk * times_cycles
        IQ_scale = max((np.max(np.abs(I_mhz_vs_us)), np.max(np.abs(Q_mhz_vs_us))))
        I_func = sp.interpolate.interp1d(times_samps, I_mhz_vs_us/IQ_scale, kind='linear', fill_value='extrapolate')
        Q_func = sp.interpolate.interp1d(times_samps, Q_mhz_vs_us/IQ_scale, kind='linear', fill_value='extrapolate')
        t = np.arange(0, np.round(times_samps[-1]))
        iamps = I_func(t)
        qamps = Q_func(t)
        plt.plot(maxv*iamps, '.-')
        # plt.plot(times_samps, I_func(times_samps), '.-')
        plt.plot(maxv*qamps, '.-')
        plt.axhline(maxv)
        # plt.plot(times_samps, Q_func(times_samps), '.-')
        plt.show()
        self.add_pulse(ch=ch, name=name, idata=maxv*iamps, qdata=maxv*qamps)        

    def handle_IQ_pulse(self, name, waveformname=None, ch=None, I_mhz_vs_us=None, Q_mhz_vs_us=None, times_us=None, freq_MHz=None, phase_deg=None, gain=None, reload=True, play=False, set_reg=False, flag=None, phrst=0):
        """
        Load/play an arbitrary IQ pulse on channel ch
        """
        if name not in self.pulse_dict.keys() or reload:
            assert ch is not None and I_mhz_vs_us is not None and Q_mhz_vs_us is not None and times_us is not None
            if waveformname is None: waveformname = name
            self.pulse_dict.update({name:dict(ch=ch, name=name, waveformname=waveformname, type='IQpulse', I_mhz_vs_us=I_mhz_vs_us, Q_mhz_vs_us=Q_mhz_vs_us, times_us=times_us, freq_MHz=freq_MHz, phase_deg=phase_deg, gain=gain, flag=flag)})
            if reload or waveformname not in self.pulses[ch].keys():
                self.add_IQ(ch=ch, name=waveformname, I_mhz_vs_us=I_mhz_vs_us, Q_mhz_vs_us=Q_mhz_vs_us, times_us=times_us)
        if play or set_reg:
            # if not (ch == sigma == None):
            #     print('Warning: you have specified a pulse parameter that can only be changed when loading.')
            params = self.pulse_dict[name].copy()
            if freq_MHz is not None: params['freq_MHz'] = freq_MHz
            if phase_deg is not None: params['phase_deg'] = phase_deg
            if gain is not None: params['gain'] = gain
            self.set_pulse_registers(ch=params['ch'], style='arb', freq=self.freq2reg(params['freq_MHz'], gen_ch=params['ch']), phase=self.deg2reg(params['phase_deg'], gen_ch=params['ch']), gain=params['gain'], waveform=params['waveformname'], phrst=phrst)
            if play:
                self.pulse(ch=params['ch'])
                self.sync_all()


    """
    Clifford pulse defns. extra_phase is given in deg. flag can be used to identify certain pulses.
    If play=False, just loads pulse.
    special: adiabatic, pulseiq
    """
    # General drive: Omega cos((wt+phi)X) -> Delta/2 Z + Omega/2 (cos(phi) X + sin(phi) Y)
    def X_pulse(self, q, pihalf=False, divide_len=False, neg=False, extra_phase=0, play=False, name='X', flag=None, special=None, phrst=0, reload=True, **kwargs):
        # q: qubit number in config
        f_ge = self.cfg.device.qubit.f_ge[q]
        gain = self.cfg.device.qubit.pulses.pi_ge.gain[q]
        phase_deg = self.overall_phase[q] + extra_phase
        sigma = self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.sigma[q], gen_ch=self.qubit_chs[q])
        waveformname = 'pi_ge'
        type = self.cfg.device.qubit.pulses.pi_ge.type[q]
        if special:
            if special == 'adiabatic':
                gain = self.cfg.device.qubit.pulses.pi_ge_adiabatic.gain[q]
                period_us = self.cfg.device.qubit.pulses.pi_ge_adiabatic.period[q]
                mu = self.cfg.device.qubit.pulses.pi_ge_adiabatic.mu[q]
                beta = self.cfg.device.qubit.pulses.pi_ge_adiabatic.beta[q]
                if 'adiabatic' not in name : name = name + '_adiabatic'
                waveformname = 'pi_ge_adiabatic'
                type = 'adiabatic'
            elif special == 'pulseiq':
                type = 'pulseiq'
                gain = self.cfg.device.qubit.pulses.pi_ge_IQ.gain[q]
                waveformname = 'pi_ge_IQ'
                assert 'I_mhz_vs_us' in kwargs.keys() and 'Q_mhz_vs_us' in kwargs.keys() and 'times_us' in kwargs.keys()
                I_mhz_vs_us = kwargs['I_mhz_vs_us']
                Q_mhz_vs_us = kwargs['Q_mhz_vs_us']
                times_us = kwargs['times_us']
        if pihalf:
            if divide_len:
                sigma = sigma // 2
                waveformname += 'half'
            else: gain = gain // 2
            name += 'half'
        if neg: phase_deg -= 180
        if type == 'const':
            self.handle_const_pulse(name=f'{name}_q{q}', ch=self.qubit_chs[q], waveformname=f'{waveformname}_q{q}', length=sigma, freq_MHz=f_ge, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload) 
        elif type == 'gauss':
            self.handle_gauss_pulse(name=f'{name}_q{q}', ch=self.qubit_chs[q],waveformname=f'{waveformname}_q{q}', sigma=sigma, freq_MHz=f_ge, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload) 
        elif type == 'adiabatic':
            assert not pihalf, 'Cannot do pihalf pulse with adiabatic'
            self.handle_adiabatic_pulse(name=f'{name}_q{q}', ch=self.qubit_chs[q], waveformname=f'{waveformname}_q{q}', mu=mu, beta=beta, period_us=period_us, freq_MHz=f_ge, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'pulseiq':
            assert not pihalf, 'Cannot do pihalf pulse with pulseiq'
            self.handle_IQ_pulse(name=f'{name}_q{q}', ch=self.qubit_chs[q], waveformname=f'{waveformname}_q{q}', I_mhz_vs_us=I_mhz_vs_us, Q_mhz_vs_us=Q_mhz_vs_us, times_us=times_us, freq_MHz=f_ge, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'flat_top':
            assert False, 'flat top not checked yet'
            flat_length = self.us2cycles(self.cfg.device.qubit.pulses.pi_ge.length[q], gen_ch=self.qubit_chs[q]) - 3*4
            self.handle_flat_top_pulse(name=f'{name}_q{q}', ch=self.qubit_chs[q], waveformname=f'{waveformname}_q{q}', sigma=sigma, flat_length=flat_length, freq_MHz=f_ge, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload) 
        else: assert False, f'Pulse type {type} not supported.'

    def Y_pulse(self, q, pihalf=False, divide_len=True, neg=False, extra_phase=0, adiabatic=False, play=False, flag=None, phrst=0, reload=True):
        # the sign of the 180 does not matter, but the sign of the pihalf does!
        self.X_pulse(q, pihalf=pihalf, divide_len=divide_len, neg=not neg, extra_phase=90+extra_phase, play=play, name='Y', flag=flag, adiabatic=adiabatic, phrst=phrst, reload=reload)

    def Z_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False, reload=None):
        dac_type = self.qubit_ch_types[q]
        assert not dac_type == 'mux4', "Currently cannot set phase for mux4!"
        phase_adjust = 180
        if pihalf: phase_adjust = 90 # the sign of the 180 does not matter, but the sign of the pihalf does!
        if neg: phase_adjust *= -1
        if play: self.overall_phase[q] += phase_adjust + extra_phase

    def initialize(self):
        self.cfg = AttrDict(self.cfg)
        self.cfg.update(self.cfg.expt)
        if 'qubits' in self.cfg.expt: self.qubits = self.cfg.expt.qubits
        else: self.qubits = range(4)
        self.pulse_dict = dict()
        self.num_qubits_sample = len(self.cfg.device.qubit.f_ge)

        # all of these saved self.whatever instance variables should be indexed by the actual qubit number as opposed to qubits_i. this means that more values are saved as instance variables than is strictly necessary, but this is overall less confusing
        self.adc_chs = self.cfg.hw.soc.adcs.readout.ch
        self.res_chs = self.cfg.hw.soc.dacs.readout.ch
        self.res_ch_types = self.cfg.hw.soc.dacs.readout.type
        self.qubit_chs = self.cfg.hw.soc.dacs.qubit.ch
        self.qubit_ch_types = self.cfg.hw.soc.dacs.qubit.type

        self.overall_phase = [0]*self.num_qubits_sample

        self.q_rps = [self.ch_page(ch) for ch in self.qubit_chs] # get register page for qubit_ch
        self.f_res_reg = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.readout.frequency, self.res_chs)]

        self.f_ge_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_ge, self.qubit_chs)]
        self.f_ef_regs = [self.freq2reg(f, gen_ch=ch) for f, ch in zip(self.cfg.device.qubit.f_ef, self.qubit_chs)]
        self.f_res_regs = [self.freq2reg(f, gen_ch=gen_ch, ro_ch=adc_ch) for f, gen_ch, adc_ch in zip(self.cfg.device.readout.frequency, self.res_chs, self.adc_chs)]
        self.f_Q1_ZZ_regs = [self.freq2reg(f, gen_ch=self.qubit_chs[1]) for f in self.cfg.device.qubit.f_Q1_ZZ]

        self.readout_lengths_dac = [self.us2cycles(length, gen_ch=gen_ch) for length, gen_ch in zip(self.cfg.device.readout.readout_length, self.res_chs)]
        self.readout_lengths_adc = [1+self.us2cycles(length, ro_ch=ro_ch) for length, ro_ch in zip(self.cfg.device.readout.readout_length, self.adc_chs)]

        # declare res dacs, add readout pulses
        if self.res_ch_types[0] == 'mux4': # only supports having all resonators be on mux, or none
            assert np.all([ch == 6 for ch in self.res_chs])
            mask = [0, 1, 2, 3] # indices of mux_freqs, mux_gains list to play
            mux_freqs = self.cfg.device.readout.frequency
            mux_gains = self.cfg.device.readout.gain
            self.declare_gen(ch=6, nqz=self.cfg.hw.soc.dacs.readout.nyquist[0], mixer_freq=self.cfg.hw.soc.dacs.readout.mixer_freq[0], mux_freqs=mux_freqs, mux_gains=mux_gains, ro_ch=0)
            self.handle_mux4_pulse(name=f'measure', ch=6, length=max(self.readout_lengths_dac), mask=mask, play=False, set_reg=True)
        else:
            for q in self.qubits:
                mixer_freq = 0
                if self.res_ch_types[q] == 'int4':
                    mixer_freq = self.cfg.hw.soc.dacs.readout.mixer_freq[q]
                self.declare_gen(ch=self.res_chs[q], nqz=self.cfg.hw.soc.dacs.readout.nyquist[q], mixer_freq=mixer_freq)
                self.handle_const_pulse(name=f'measure{q}', ch=self.res_chs[q], length=self.readout_length[q], freq_MHz=self.cfg.device.readout.frequency[q], phase=0, gain=self.cfg.device.readout.gain[q], play=False, set_reg=True)

        # get aliases for the sigmas we need in clock cycles
        self.pi_sigmas_us = self.cfg.device.qubit.pulses.pi_ge.sigma
        self.pi_ef_sigmas_us = self.cfg.device.qubit.pulses.pi_ef.sigma
        self.pi_Q1_ZZ_sigmas_us = self.cfg.device.qubit.pulses.pi_Q1_ZZ.sigma
        self.pi_ge_types = self.cfg.device.qubit.pulses.pi_ge.type
        self.pi_ef_types = self.cfg.device.qubit.pulses.pi_ef.type
        self.pi_Q1_ZZ_types = self.cfg.device.qubit.pulses.pi_Q1_ZZ.type

        # declare qubit dacs, add qubit pi_ge pulses
        for q in range(len(self.pi_ge_types)):
            mixer_freq = 0
            if self.qubit_ch_types[q] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.qubit.mixer_freq[q]
            if self.qubit_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.qubit_chs[q], nqz=self.cfg.hw.soc.dacs.qubit.nyquist[q], mixer_freq=mixer_freq)
            self.X_pulse(q=q, play=False, reload=True)

            # assume ef pulses are gauss
            pi_ef_sigma_cycles = self.us2cycles(self.pi_ef_sigmas_us[q], gen_ch=self.qubit_chs[q])
            self.add_gauss(ch=self.qubit_chs[q], name=f"pi_ef_qubit{q}", sigma=pi_ef_sigma_cycles, length=pi_ef_sigma_cycles*4)
            if q != 1:
                pi_Q1_ZZ_sigma_cycles = self.us2cycles(self.pi_Q1_ZZ_sigmas_us[q], gen_ch=self.qubit_chs[1])
                self.add_gauss(ch=self.qubit_chs[1], name=f"qubit1_ZZ{q}", sigma=pi_Q1_ZZ_sigma_cycles, length=pi_Q1_ZZ_sigma_cycles*4)

        # declare adcs - readout for all qubits everytime, defines number of buffers returned regardless of number of adcs triggered
        for q in range(self.num_qubits_sample):
            self.declare_readout(ch=self.adc_chs[q], length=self.readout_lengths_adc[q], freq=self.cfg.device.readout.frequency[q], gen_ch=self.res_chs[q])



    # """
    # Collect shots for all adcs, rotates by given angle (degrees), and averages over shot_avg adjacent shots
    # Returns avgi, avgq, avgi_err, avgq_err which avgi/q are avg over shot_avg and avgi/q_err is (std dev of each group of shots)/sqrt(shot_avg)
    # """
    # def get_shots(self, angle=None, shot_avg=1, verbose=True, return_err=False):
    #     if angle == None: angle = [0]*len(self.cfg.device.qubit.f_ge)
    #     bufi = np.array([
    #         self.di_buf[i]*np.cos(np.pi/180*angle[i]) - self.dq_buf[i]*np.sin(np.pi/180*angle[i])
    #         for i, ch in enumerate(self.ro_chs)])
    #     avgi = []
    #     avgi_err = []
    #     for bufi_ch in bufi:
    #         # drop extra shots that aren't divisible into averages
    #         new_bufi_ch = np.copy(bufi_ch[:len(bufi_ch) - (len(bufi_ch) % shot_avg)])
    #         # average over shots_avg number of consecutive shots
    #         new_bufi_ch = np.reshape(new_bufi_ch, (len(new_bufi_ch)//shot_avg, shot_avg))
    #         new_bufi_ch_err = np.std(new_bufi_ch, axis=1) / np.sqrt(shot_avg)
    #         new_bufi_ch = np.average(new_bufi_ch, axis=1)
    #         avgi_err.append(new_bufi_ch_err)
    #         avgi.append(new_bufi_ch)
    #     avgi = np.array(avgi)
    #     avgi = np.array([avgi[i]/ro.length for i, (ch, ro) in enumerate(self.ro_chs.items())])
    #     avgi_err = np.array([avgi[i]/ro.length for i, (ch, ro) in enumerate(self.ro_chs.items())])
    #     # print(avgi[self.cfg.expt.qubits[1]])
    #     if verbose: print([np.median(avgi[i]) for i in range(4)])

    #     bufq = np.array([
    #         self.di_buf[i]*np.sin(np.pi/180*angle[i]) + self.dq_buf[i]*np.cos(np.pi/180*angle[i])
    #         for i, ch in enumerate(self.ro_chs)])
    #     avgq = []
    #     avgq_err = []
    #     for bufq_ch in bufq:
    #         # drop extra shots that aren't divisible into averages
    #         new_bufq_ch = np.copy(bufq_ch[:len(bufq_ch) - (len(bufq_ch) % shot_avg)])
    #         # average over shots_avg number of consecutive shots
    #         new_bufq_ch = np.reshape(new_bufq_ch, (len(new_bufq_ch)//shot_avg, shot_avg))
    #         new_bufq_ch_err = np.std(new_bufq_ch, axis=1) / np.sqrt(shot_avg)
    #         new_bufq_ch = np.average(new_bufq_ch, axis=1)
    #         assert(np.shape(new_bufq_ch_err) == np.shape(new_bufq_ch))
    #         avgq_err.append(new_bufq_ch_err)
    #         avgq.append(new_bufq_ch)
    #     avgq = np.array(avgq)
    #     avgq = np.array([avgq[i]/ro.length for i, (ch, ro) in enumerate(self.ro_chs.items())])
    #     avgq_err = np.array([avgq[i]/ro.length for i, (ch, ro) in enumerate(self.ro_chs.items())])

    #     if return_err: return avgi, avgq, avgi_err, avgq_err
    #     else: return avgi, avgq

    """
    Collect shots for all adcs, rotates by given angle (degrees), separate based on threshold (if not None), and averages over all shots (i.e. returns data[num_chs, 1] as opposed to data[num_chs, num_shots]) if requested.
    Returns avgi, avgq, avgi_err, avgq_err which avgi/q are avg over shot_avg and avgi/q_err is (std dev of each group of shots)/sqrt(shot_avg)
    """
    def get_shots(self, angle=None, threshold=None, avg_shots=False, verbose=False, return_err=False):
        buf_len = len(self.di_buf[0])

        if angle is None: angle = [0]*len(self.cfg.device.qubit.f_ge)
        bufi = np.array([
            self.di_buf[i]*np.cos(np.pi/180*angle[i]) - self.dq_buf[i]*np.sin(np.pi/180*angle[i])
            for i, ch in enumerate(self.ro_chs)])
        bufi = np.array([bufi[i]/ro['length'] for i, (ch, ro) in enumerate(self.ro_chs.items())])
        if threshold is not None: # categorize single shots
            bufi = np.array([np.heaviside(bufi[ch] - threshold[ch], 0) for ch in range(len(self.adc_chs))])
        avgi = np.average(bufi, axis=1) # [num_chs]
        bufi_err = np.std(bufi, axis=1) / np.sqrt(buf_len) # [num_chs]
        if verbose: print([np.median(bufi[i]) for i in range(4)])

        bufq = np.array([
            self.di_buf[i]*np.sin(np.pi/180*angle[i]) + self.dq_buf[i]*np.cos(np.pi/180*angle[i])
            for i, ch in enumerate(self.ro_chs)])
        bufq = np.array([bufq[i]/ro['length'] for i, (ch, ro) in enumerate(self.ro_chs.items())])
        avgq = np.average(bufq, axis=1) # [num_chs]
        bufq_err = np.std(bufq, axis=1) / np.sqrt(buf_len) # [num_chs]
        if verbose: print([np.median(bufq[i]) for i in range(4)])

        if avg_shots:
            idata = avgi
            qdata = avgq
        else:
            idata = bufi
            qdata = bufq

        if return_err: return idata, qdata, bufi_err, bufq_err
        else: return idata, qdata 

    """
    If post_process == 'threshold': uses angle + threshold to categorize shots into 0 or 1 and calculate the population
    If post_process == 'scale': uses angle + ge_avgs to scale the average of all shots on a scale of 0 to 1. ge_avgs should be of shape (num_total_qubits, 4) and should represent the pre-rotation Ig, Qg, Ie, Qe
    If post_process == None: uses angle to rotate the i and q and then returns the avg i and q
    """
    def acquire_rotated(self, soc, progress, angle=None, threshold=None, ge_avgs=None, post_process=None, verbose=False):
        avgi, avgq = self.acquire(soc, load_pulses=True, progress=progress, debug=False)
        if post_process == None: 
            avgi_rot, avgq_rot, avgi_err, avgq_err = self.get_shots(angle=angle, avg_shots=True, verbose=verbose, return_err=True)
            if angle is None: return avgi_rot, avgq_rot
            else: return avgi_rot, avgi_err
        elif post_process == 'threshold':
            assert threshold is not None
            popln, avgq_rot, popln_err, avgq_err = self.get_shots(angle=angle, threshold=threshold, avg_shots=True, verbose=verbose, return_err=True)
            return popln, popln_err
        elif post_process == 'scale':
            assert ge_avgs is not None
            avgi_rot, avgq_rot, avgi_err, avgq_err = self.get_shots(angle=angle, avg_shots=True, verbose=verbose, return_err=True)

            ge_avgs_rot = [None]*4
            for q, angle_q in enumerate(angle):
                Ig_q, Qg_q, Ie_q, Qe_q = ge_avgs[q]
                ge_avgs_rot[q] = [
                    Ig_q*np.cos(np.pi/180*angle_q) - Qg_q*np.sin(np.pi/180*angle_q),
                    Ie_q*np.cos(np.pi/180*angle_q) - Qe_q*np.sin(np.pi/180*angle_q)
                ]
            ge_avgs_rot = np.asarray(ge_avgs_rot)
            # print(avgi_rot, ge_avgs_rot)
            avgi_rot -= ge_avgs_rot[:,0]
            avgi_rot /= ge_avgs_rot[:,1] - ge_avgs_rot[:,0]
            avgi_err /= ge_avgs_rot[:,1] - ge_avgs_rot[:,0]
            return avgi_rot, avgi_err
        else:
            assert False, 'Undefined post processing flag, options are None, threshold, scale'

# ===================================================================== #

"""
Take care of extra clifford pulses for qutrits.
"""
class QutritAveragerProgram(CliffordAveragerProgram):
    def Xef_pulse(self, q, pihalf=False, divide_len=True, name='X_ef', neg=False, extra_phase=0, play=False, flag=None, phrst=0, reload=True):
        ch = self.qubit_chs[q]
        f_ef_MHz = self.cfg.device.qubit.f_ef[q]
        gain = self.cfg.device.qubit.pulses.pi_ef.gain[q]
        phase_deg = self.overall_phase[q] + extra_phase
        sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_ef.sigma[q], gen_ch=ch)
        waveformname = 'pi_ef'
        if pihalf:
            if divide_len:
                sigma_cycles = sigma_cycles // 2
                waveformname += 'half'
            else: gain = gain // 2
            name += 'half'
        if neg: phase_deg -= 180
        type = self.cfg.device.qubit.pulses.pi_ef.type[q]
        if type == 'const':
            self.handle_const_pulse(name=f'{name}_q{q}', ch=ch, waveformname=f'{waveformname}_q{q}', length=sigma_cycles, freq_MHz=f_ef_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'gauss':
            self.handle_gauss_pulse(name=f'{name}_q{q}', ch=ch, waveformname=f'{waveformname}_q{q}', sigma=sigma_cycles, freq_MHz=f_ef_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'flat_top':
            sigma_ramp_cycles = 3
            flat_length_cycles = sigma_cycles - sigma_ramp_cycles*4
            self.handle_flat_top_pulse(name=f'{name}_q{q}', ch=ch, waveformname=f'{waveformname}_q{q}', sigma=sigma_ramp_cycles, flat_length=flat_length_cycles, freq_MHz=f_ef_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        else: assert False, f'Pulse type {type} not supported.'
    
    def Yef_pulse(self, q, pihalf=False, neg=False, extra_phase=0, play=False, flag=None, phrst=0, reload=True):
        # the sign of the 180 does not matter, but the sign of the pihalf does!
        self.Xef_pulse(q, pihalf=pihalf, neg=not neg, extra_phase=90+extra_phase, play=play, name='Y_ef', flag=flag, phrst=phrst, reload=reload)

    def initialize(self):
        super().initialize()
        # declare qubit ef pulses 
        # print(self.gen_chs)
        for q in range(self.num_qubits_sample):
            self.Xef_pulse(q=q, play=False)

# ===================================================================== #
"""
Multiple inheritence testing
"""
# class Clifford():
#     def xpulse(self):
#         print('normal clifford')

#     def ypulse(self):
#         print('y')
#         self.xpulse()

# class CliffordEF(Clifford):
#     def xefpulse(self):
#         print('ef')

# class CliffordEgGf(CliffordEF):
#     def xpulse(self):
#         super().xpulse()
#         print('EgGf')

# class SimRB(Clifford):
#     def clifford(self, flag=None):
#         if flag == 'X': self.xpulse()
#         elif flag == 'Y': self.ypulse()
    
# class RBEgGf(CliffordEgGf, SimRB):
#     pass

# rbeggf = RBEgGf()
# print(RBEgGf.__mro__)
# rbeggf.clifford(flag='X')
# rbeggf.clifford(flag='Y')

"""
Replace the X/Y/Z pulses with an effective TLS represented by the Eg-Gf pulse.
WARNING: this means if you try to call X pulse it will play the Eg-Gf pulse, not a normal qubit X pulse!
"""
class CliffordEgGfAveragerProgram(QutritAveragerProgram):
    # self.overall_phase keeps track of the EgGf phase insetad of the e-g pulse phase

    def XEgGf_pulse(self, qDrive, qNotDrive, pihalf=False, divide_len=True, name='X_EgGf', neg=False, extra_phase=0, play=False, flag=None, phrst=0, reload=True):
        # convention is waveformname is pi_EgGf_qNotDriveqDrive
        if qDrive == 1:
            ch = self.swap_chs[qNotDrive]
            f_EgGf_MHz = self.cfg.device.qubit.f_EgGf[qNotDrive]
            gain = self.cfg.device.qubit.pulses.pi_EgGf.gain[qNotDrive]
            phase_deg = self.overall_phase[qNotDrive] + extra_phase
            sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf.sigma[qNotDrive], gen_ch=ch)
            type = self.cfg.device.qubit.pulses.pi_EgGf.type[qNotDrive]
            waveformname = 'pi_EgGf'
        else:
            ch = self.swap_Q_chs[qDrive]
            f_EgGf_MHz = self.cfg.device.qubit.f_EgGf_Q[qDrive]
            gain = self.cfg.device.qubit.pulses.pi_EgGf_Q.gain[qDrive]
            phase_deg = self.overall_phase[qDrive] + extra_phase
            sigma_cycles = self.us2cycles(self.cfg.device.qubit.pulses.pi_EgGf_Q.sigma[qDrive], gen_ch=ch)
            type = self.cfg.device.qubit.pulses.pi_EgGf_Q.type[qDrive]
            waveformname = 'pi_EgGf'
        if pihalf:
            if divide_len:
                sigma_cycles = sigma_cycles // 2
                waveformname += 'half'
            else: gain = gain // 2
            name += 'half'
        if neg: phase_deg -= 180
        if type == 'const':
            self.handle_const_pulse(name=f'{name}_{qNotDrive}{qDrive}', ch=ch, waveformname=f'{waveformname}_{qNotDrive}{qDrive}', length=sigma_cycles, freq_MHz=f_EgGf_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'gauss':
            self.handle_gauss_pulse(name=f'{name}_{qNotDrive}{qDrive}', ch=ch, waveformname=f'{waveformname}_{qNotDrive}{qDrive}', sigma=sigma_cycles, freq_MHz=f_EgGf_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        elif type == 'flat_top':
            sigma_ramp_cycles = 3
            flat_length_cycles = sigma_cycles - sigma_ramp_cycles*4
            self.handle_flat_top_pulse(name=f'{name}_{qNotDrive}{qDrive}', ch=ch, waveformname=f'{waveformname}_{qNotDrive}{qDrive}', sigma=sigma_ramp_cycles, flat_length=flat_length_cycles, freq_MHz=f_EgGf_MHz, phase_deg=phase_deg, gain=gain, play=play, flag=flag, phrst=phrst, reload=reload)
        else: assert False, f'Pulse type {type} not supported.'
        # print('ch keys', self.gen_chs.keys())

    def YEgGf_pulse(self, qDrive, qNotDrive, pihalf=False, neg=False, extra_phase=0, play=False, flag=None, phrst=0, reload=True):
        # the sign of the 180 does not matter, but the sign of the pihalf does!
        self.XEgGf_pulse(qDrive, qNotDrive, pihalf=pihalf, neg=not neg, extra_phase=90+extra_phase, play=play, name='Y_EgGf', flag=flag, phrst=phrst, reload=reload)

    def ZEgGf_pulse(self, qDrive, qNotDrive, pihalf=False, neg=False, extra_phase=0, play=False, reload=None):
        if qDrive == 1: dac_type = self.swap_ch_types[qNotDrive]
        else: dac_type = self.swap_Q_ch_types[qDrive]
        assert not dac_type == 'mux4', "Currently cannot set phase for mux4!"
        phase_adjust = 180
        if pihalf: phase_adjust = 90 # the sign of the 180 does not matter, but the sign of the pihalf does!
        if neg: phase_adjust *= -1
        if play:
            if qDrive == 1: self.overall_phase[qNotDrive] += phase_adjust + extra_phase
            else: self.overall_phase[qDrive] += phase_adjust + extra_phase

    def initialize(self):
        self.swap_chs = self.cfg.hw.soc.dacs.swap.ch
        self.swap_ch_types = self.cfg.hw.soc.dacs.swap.type
        self.swap_Q_chs = self.cfg.hw.soc.dacs.swap_Q.ch
        self.swap_Q_ch_types = self.cfg.hw.soc.dacs.swap_Q.type
        super().initialize()
        for q in self.qubits:
            if q==1: continue
            mixer_freq = 0
            if self.swap_ch_types[q] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.swap.mixer_freq[q]
            if self.swap_chs[q] not in self.gen_chs:
                self.declare_gen(ch=self.swap_chs[q], nqz=self.cfg.hw.soc.dacs.swap.nyquist[q], mixer_freq=mixer_freq)
            # else: print('nqz', self.gen_chs[self.swap_chs[q]]['nqz'])
            mixer_freq=0
            if self.swap_Q_ch_types[q] == 'int4':
                mixer_freq = self.cfg.hw.soc.dacs.swap_Q.mixer_freq[q]
            if self.swap_Q_chs[q] not in self.gen_chs: 
                self.declare_gen(ch=self.swap_Q_chs[q], nqz=self.cfg.hw.soc.dacs.swap_Q.nyquist[q], mixer_freq=mixer_freq)
        self.sync_all(100)