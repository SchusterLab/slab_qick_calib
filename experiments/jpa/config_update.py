"""Helpers for persisting JPA settings into the YAML config."""

import numpy as np

from ...exp_handling.datamanagement import AttrDict
from ...helpers import config


def _scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _pair(value):
    return [0, _scalar(value)]


def _get(expt, key, default=None):
    try:
        return expt.get(key, default)
    except AttributeError:
        return default


def should_update_yaml(expt):
    return bool(_get(expt, "rspec_update", _get(expt, "update", False)))


def build_jpa_section(expt, data=None):
    """Build the hw.soc.dacs.jpa section with the active value in index 1."""
    data = {} if data is None else data

    pump_freq = _get(expt, "pump_freq_mhz", _get(expt, "pump_freq", np.nan))
    mixing_freq = _get(expt, "pump_mixer_freq_mhz", _get(expt, "pump_mixer_freq", np.nan))
    pump_gain = data.get("best_pump_gain", _get(expt, "pump_gain", _get(expt, "jpa_pump_gain", np.nan)))
    dc_val = data.get("best_dc_bias_v", _get(expt, "jpa_dc_val", _get(expt, "dc_bias_v", np.nan)))

    return {
        "attn1": _pair(_get(expt, "pump_attn1_db", 15.0)),
        "attn2": _pair(_get(expt, "pump_attn2_db", 0.0)),
        "ch": _pair(_get(expt, "pump_ch", 13)),
        "filter_bw": _pair(_get(expt, "pump_filter_bw_mhz", 1000.0)),
        "filter_fc": _pair(_get(expt, "pump_filter_fc_mhz", pump_freq)),
        "filter_type": _pair(_get(expt, "pump_filter_type", "bandpass")),
        "nyquist": _pair(_get(expt, "pump_nqz", 2)),
        "type": _pair(_get(expt, "pump_type", "int")),
        "pump_freq": _pair(pump_freq),
        "mixing_freq": _pair(mixing_freq),
        "pump_gain": _pair(pump_gain),
        "jpa_dc_ch": _pair(_get(expt, "bias_chan", _get(expt, "jpa_dc_ch", np.nan))),
        "jpa_dc_val": _pair(dc_val),
    }


def update_jpa_config(cfg_file, expt, data=None, cfg=None, verbose=True):
    """
    Persist JPA RF/DC settings under hw.soc.dacs.jpa.

    The list format is intentionally [0, active_value] because this setup uses
    the second entry for the real JPA path.
    """
    jpa_section = build_jpa_section(expt, data=data)
    cfg = config.load(cfg_file) if cfg is None else cfg
    cfg.hw.soc.dacs.jpa = AttrDict(jpa_section)
    saved = config.save(cfg, cfg_file)

    if verbose:
        print(
            "Updated hw.soc.dacs.jpa in "
            f"{cfg_file}: pump ch {jpa_section['ch'][1]}, "
            f"pump {jpa_section['pump_freq'][1]} MHz, "
            f"gain {jpa_section['pump_gain'][1]}, "
            f"DC ch {jpa_section['jpa_dc_ch'][1]}, "
            f"DC {jpa_section['jpa_dc_val'][1]} V"
        )
    return saved
