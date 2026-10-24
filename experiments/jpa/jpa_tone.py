"""Helpers for manually parking the JPA pump tone."""

from qick.asm_v2 import AveragerProgramV2

from ...exp_handling.datamanagement import AttrDict


class SetPeriodicToneProgram(AveragerProgramV2):
    """Tiny QICK program that starts a periodic constant tone on one generator."""

    def __init__(self, soccfg, final_delay=0.01, cfg=None):
        cfg = AttrDict(cfg)
        super().__init__(soccfg, cfg.expt.reps, final_delay, cfg=cfg)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        ch = int(cfg.expt.ch)

        gen_args = {"ch": ch, "nqz": int(cfg.expt.nqz)}
        if self.soccfg["gens"][ch].get("has_mixer", False):
            gen_args["mixer_freq"] = float(cfg.expt.mixer_freq)

        self.declare_gen(**gen_args)
        self.add_pulse(
            ch=ch,
            name="tone",
            style="const",
            freq=float(cfg.expt.freq),
            phase=0,
            gain=float(cfg.expt.gain),
            length=float(cfg.expt.length),
            mode="periodic",
        )

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)
        self.pulse(ch=int(cfg.expt.ch), name="tone", t=0)


def set_periodic_tone(
    soc,
    soccfg,
    ch,
    freq_mhz,
    gain,
    nqz=2,
    mixer_freq_mhz=None,
    length=1.0,
):
    """Start or update a periodic QICK tone."""
    ch = int(ch)
    if mixer_freq_mhz is None and soccfg["gens"][ch].get("has_mixer", False):
        mixer_freq_mhz = float(freq_mhz) - 400.0

    cfg = {
        "expt": {
            "ch": ch,
            "freq": float(freq_mhz),
            "gain": float(gain),
            "nqz": int(nqz),
            "mixer_freq": mixer_freq_mhz,
            "length": float(length),
            "reps": 1,
        }
    }
    prog = SetPeriodicToneProgram(soccfg, cfg=cfg)
    prog.run(soc, load_prog=True, load_envelopes=True, start_src="internal")
    return prog


def stop_periodic_tone(soc, soccfg, ch, freq_mhz, nqz=2, mixer_freq_mhz=None):
    """Stop a periodic tone by replacing it with zero gain."""
    return set_periodic_tone(
        soc,
        soccfg,
        ch=ch,
        freq_mhz=freq_mhz,
        gain=0.0,
        nqz=nqz,
        mixer_freq_mhz=mixer_freq_mhz,
    )


def _jpa_value(jpa, name, idx, default=None):
    if not hasattr(jpa, name):
        return default
    value = getattr(jpa, name)
    if isinstance(value, (list, tuple)):
        return value[idx]
    return value


def _set_jpa_value(jpa, name, idx, value):
    if not hasattr(jpa, name):
        return
    current = getattr(jpa, name)
    if isinstance(current, list):
        current[idx] = value
    elif isinstance(current, tuple):
        updated = list(current)
        updated[idx] = value
        setattr(jpa, name, tuple(updated))
    else:
        setattr(jpa, name, value)


def turn_on_jpa(
    soc,
    soccfg,
    cfg,
    pump_gain=None,
    dc_bias_v=None,
    idx=1,
    verbose=True,
    pump_ch=None,
    pump_freq_mhz=None,
    mixer_freq_mhz=None,
    pump_nqz=None,
    attn1_db=None,
    attn2_db=None,
    filter_fc_mhz=None,
    filter_bw_mhz=None,
    filter_type=None,
):
    """
    Apply JPA RF/DC settings from cfg.hw.soc.dacs.jpa and start the pump tone.

    Optional keyword overrides let a notebook park the exact point found by a
    fine-tuning scan without first writing it to YAML.

    This is intended for manual parking/checking. It launches a periodic QICK
    tone, so later QICK programs may overwrite or stop it when they configure
    the tProc.
    """
    jpa = cfg.hw.soc.dacs.jpa

    pump_ch = int(_jpa_value(jpa, "ch", idx) if pump_ch is None else pump_ch)
    pump_freq = float(_jpa_value(jpa, "pump_freq", idx) if pump_freq_mhz is None else pump_freq_mhz)
    mixer_freq = float(
        _jpa_value(jpa, "mixing_freq", idx, pump_freq - 400.0)
        if mixer_freq_mhz is None
        else mixer_freq_mhz
    )
    nqz = int(_jpa_value(jpa, "nyquist", idx, 2) if pump_nqz is None else pump_nqz)
    attn1 = float(_jpa_value(jpa, "attn1", idx, 15.0) if attn1_db is None else attn1_db)
    attn2 = float(_jpa_value(jpa, "attn2", idx, 0.0) if attn2_db is None else attn2_db)
    filter_fc = float(_jpa_value(jpa, "filter_fc", idx, pump_freq) if filter_fc_mhz is None else filter_fc_mhz)
    filter_bw = float(_jpa_value(jpa, "filter_bw", idx, 1000.0) if filter_bw_mhz is None else filter_bw_mhz)
    filter_type = _jpa_value(jpa, "filter_type", idx, "bandpass") if filter_type is None else filter_type
    dc_ch = int(_jpa_value(jpa, "jpa_dc_ch", idx))

    if pump_gain is None:
        pump_gain = float(_jpa_value(jpa, "pump_gain", idx))
    if dc_bias_v is None:
        dc_bias_v = float(_jpa_value(jpa, "jpa_dc_val", idx))

    soc.rfb_set_bias(dc_ch, float(dc_bias_v))

    try:
        soc.rfb_set_gen_filter(
            pump_ch,
            fc=filter_fc / 1000.0,
            bw=filter_bw / 1000.0,
            ftype=filter_type,
        )
    except TypeError:
        soc.rfb_set_gen_filter(pump_ch, fc=filter_fc / 1000.0, ftype=filter_type)

    soc.rfb_set_gen_rf(pump_ch, attn1, attn2)
    set_periodic_tone(
        soc,
        soccfg,
        ch=pump_ch,
        freq_mhz=pump_freq,
        gain=float(pump_gain),
        nqz=nqz,
        mixer_freq_mhz=mixer_freq,
    )

    _set_jpa_value(jpa, "ch", idx, pump_ch)
    _set_jpa_value(jpa, "pump_freq", idx, pump_freq)
    _set_jpa_value(jpa, "mixing_freq", idx, mixer_freq)
    _set_jpa_value(jpa, "nyquist", idx, nqz)
    _set_jpa_value(jpa, "attn1", idx, attn1)
    _set_jpa_value(jpa, "attn2", idx, attn2)
    _set_jpa_value(jpa, "filter_fc", idx, filter_fc)
    _set_jpa_value(jpa, "filter_bw", idx, filter_bw)
    _set_jpa_value(jpa, "filter_type", idx, filter_type)
    _set_jpa_value(jpa, "pump_gain", idx, float(pump_gain))
    _set_jpa_value(jpa, "jpa_dc_val", idx, float(dc_bias_v))

    state = {
        "pump_ch": pump_ch,
        "pump_freq_mhz": pump_freq,
        "mixer_freq_mhz": mixer_freq,
        "pump_gain": float(pump_gain),
        "attn1_db": attn1,
        "attn2_db": attn2,
        "jpa_dc_ch": dc_ch,
        "jpa_dc_val": float(dc_bias_v),
    }
    if verbose:
        print(
            f"JPA ON: bias={1e3 * state['jpa_dc_val']:.3f} mV, "
            f"pump={state['pump_freq_mhz']:.3f} MHz, "
            f"gain={state['pump_gain']:.5f}, "
            f"attn={state['attn1_db']}+{state['attn2_db']} dB"
        )
    return state


def get_jpa_state(soc, cfg, idx=1, verbose=True):
    """
    Report the JPA's current DC bias (read back from hardware) alongside the
    pump settings last recorded in cfg.hw.soc.dacs.jpa.

    The pump gain/freq/channel can't be read back from the board — they're
    whatever was last written via turn_on_jpa()/update_jpa_config(), so this
    is stale if someone parked the pump manually with an override gain.
    """
    jpa = cfg.hw.soc.dacs.jpa

    pump_ch = int(_jpa_value(jpa, "ch", idx))
    pump_freq = float(_jpa_value(jpa, "pump_freq", idx))
    pump_gain = float(_jpa_value(jpa, "pump_gain", idx))
    dc_ch = int(_jpa_value(jpa, "jpa_dc_ch", idx))
    dc_val_recorded = float(_jpa_value(jpa, "jpa_dc_val", idx))

    dc_bias_readback_v = float(soc.rfb_get_bias(dc_ch))

    state = {
        "pump_ch": pump_ch,
        "pump_freq_mhz": pump_freq,
        "pump_gain": pump_gain,
        "jpa_dc_ch": dc_ch,
        "jpa_dc_val_recorded": dc_val_recorded,
        "dc_bias_readback_v": dc_bias_readback_v,
    }
    if verbose:
        print(
            f"JPA state: pump ch {pump_ch} @ {pump_freq:.3f} MHz, gain={pump_gain:.5f} "
            f"(recorded, not read back), DC bias ch {dc_ch} readback="
            f"{1e3 * dc_bias_readback_v:.3f} mV (recorded={1e3 * dc_val_recorded:.3f} mV)"
        )
    return state


def turn_off_jpa(soc, soccfg, cfg, idx=1, off_bias_v=0.0, verbose=True):
    """Set the JPA pump gain to zero and optionally reset the DC bias."""
    jpa = cfg.hw.soc.dacs.jpa
    pump_ch = int(_jpa_value(jpa, "ch", idx))
    pump_freq = float(_jpa_value(jpa, "pump_freq", idx))
    mixer_freq = float(_jpa_value(jpa, "mixing_freq", idx, pump_freq - 400.0))
    nqz = int(_jpa_value(jpa, "nyquist", idx, 2))
    dc_ch = int(_jpa_value(jpa, "jpa_dc_ch", idx))

    stop_periodic_tone(
        soc,
        soccfg,
        ch=pump_ch,
        freq_mhz=pump_freq,
        nqz=nqz,
        mixer_freq_mhz=mixer_freq,
    )
    soc.rfb_set_bias(dc_ch, float(off_bias_v))

    if verbose:
        print(f"JPA OFF: bias={1e3 * float(off_bias_v):.3f} mV, pump gain=0")

