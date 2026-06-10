"""
RF board helper functions for configuring ADMV8818 bandpass filters and attenuators.

This module provides functions for setting up readout and qubit drive chains
on QICK boards with on-board RF filtering (ADMV8818) and attenuators.

All functions take ``soc`` and ``cfg`` as explicit arguments (no notebook globals).
Frequencies in the config are stored in MHz; the ``soc.rfb_*`` methods expect GHz,
so conversion is handled internally.
"""

import time


def _update_active_state(cfg, channel_type, ch, filter_fc, filter_bw, filter_type,
                         qubit=None, **attn_kwargs):
    """
    Update ``cfg.hw.soc.rfboard_active`` in memory for a single channel.

    Args:
        cfg: Configuration AttrDict
        channel_type: ``"adc"``, ``"dac_readout"``, or ``"dac_qubit"``
        ch: Physical channel number (int)
        filter_fc: Center frequency in MHz
        filter_bw: Bandwidth in MHz
        filter_type: Filter type string (``"bandpass"``, ``"bypass"``, etc.)
        qubit: Qubit index that was applied (default: None)
        **attn_kwargs: Attenuation fields (``attn``, ``attn1``, ``attn2``)
    """
    hw = cfg.hw.soc
    if "rfboard_active" not in hw:
        hw["rfboard_active"] = {}
    if channel_type not in hw["rfboard_active"]:
        hw["rfboard_active"][channel_type] = {}

    entry = {
        "filter_fc": filter_fc,
        "filter_bw": filter_bw,
        "filter_type": filter_type,
        "qubit": qubit,
    }
    entry.update(attn_kwargs)
    hw["rfboard_active"][channel_type][str(ch)] = entry


def setup_readout_chain(
    qubit,
    soc,
    cfg,
    bw_adc=None,
    bw_dac=None,
    atten1_dac=None,
    atten2_dac=None,
    atten_adc=None,
    cfg_file=None,
):
    """
    Configure the readout RF path for a given qubit.

    Sets bandpass filters on ADC and DAC readout channels centered on the
    readout frequency from the config, and sets attenuator values.

    When attenuation or bandwidth parameters are ``None`` (the default),
    values are read from the per-qubit config arrays.  Pass explicit values
    to override.

    Args:
        qubit: Qubit index
        soc: QICK SoC object
        cfg: Configuration AttrDict
        bw_adc: ADC filter bandwidth in MHz (default: from config, fallback 1000)
        bw_dac: DAC filter bandwidth in MHz (default: from config, fallback 100)
        atten1_dac: DAC attenuator 1 in dB (default: from config)
        atten2_dac: DAC attenuator 2 in dB (default: from config)
        atten_adc: ADC attenuator in dB (default: from config)
        cfg_file: Optional path to persist active state to disk (default: None)

    Returns:
        dict: Channel info ``{"ADC_ch", "DAC_ch", "readout_freq"}``
    """
    hw = cfg.hw.soc
    adc_ch = hw.adcs.readout.ch[qubit]
    dac_ch = hw.dacs.readout.ch[qubit]
    readout_freq = cfg.device.readout.frequency[qubit]

    # Read defaults from config when not explicitly provided
    if bw_adc is None:
        bw_adc = hw.adcs.readout.filter_bw[qubit] if hasattr(hw.adcs.readout, "filter_bw") else 1000
    if bw_dac is None:
        bw_dac = hw.dacs.readout.filter_bw[qubit] if hasattr(hw.dacs.readout, "filter_bw") else 100
    if atten1_dac is None:
        atten1_dac = hw.dacs.readout.attn1[qubit]
    if atten2_dac is None:
        atten2_dac = hw.dacs.readout.attn2[qubit]
    if atten_adc is None:
        atten_adc = hw.adcs.readout.attn[qubit]

    # Set bandpass filters (MHz -> GHz)
    soc.rfb_set_ro_filter(adc_ch, fc=readout_freq / 1000, ftype="bandpass", bw=bw_adc / 1000)
    soc.rfb_set_gen_filter(dac_ch, fc=readout_freq / 1000, ftype="bandpass", bw=bw_dac / 1000)

    # Set attenuators
    soc.rfb_set_gen_rf(dac_ch, atten1_dac, atten2_dac)
    soc.rfb_set_ro_rf(adc_ch, atten_adc)

    if cfg_file is not None:
        from . import config
        _update_active_state(cfg, "adc", adc_ch, readout_freq, bw_adc, "bandpass",
                             qubit=qubit, attn=atten_adc)
        _update_active_state(cfg, "dac_readout", dac_ch, readout_freq, bw_dac, "bandpass",
                             qubit=qubit, attn1=atten1_dac, attn2=atten2_dac)
        config.update_rfboard_active(cfg_file, "adc", adc_ch,
                                     cfg.hw.soc.rfboard_active["adc"][str(adc_ch)])
        config.update_rfboard_active(cfg_file, "dac_readout", dac_ch,
                                     cfg.hw.soc.rfboard_active["dac_readout"][str(dac_ch)])

    return {"ADC_ch": adc_ch, "DAC_ch": dac_ch, "readout_freq": readout_freq}


def setup_qubit_drive_chain(
    qubit,
    soc,
    cfg,
    bw=None,
    atten1_dac=None,
    atten2_dac=None,
    center_freq=None,
    cfg_file=None,
):
    """
    Configure the qubit drive RF path for a given qubit.

    Sets bandpass filter on the qubit DAC channel centered on f_ge from
    the config (or a custom center frequency) and sets attenuator values.

    When attenuation or bandwidth parameters are ``None`` (the default),
    values are read from the per-qubit config arrays.  Pass explicit values
    to override.

    Args:
        qubit: Qubit index
        soc: QICK SoC object
        cfg: Configuration AttrDict
        bw: Filter bandwidth in MHz (default: from config, fallback 1000)
        atten1_dac: DAC attenuator 1 in dB (default: from config)
        atten2_dac: DAC attenuator 2 in dB (default: from config)
        center_freq: Override center frequency in MHz (default: use f_ge from config)
        cfg_file: Optional path to persist active state to disk (default: None)

    Returns:
        dict: Channel info ``{"DAC_ch", "drive_freq"}``
    """
    hw = cfg.hw.soc
    dac_ch = hw.dacs.qubit.ch[qubit]
    drive_freq = cfg.device.qubit.f_ge[qubit]

    # Read defaults from config when not explicitly provided
    if bw is None:
        bw = hw.dacs.qubit.filter_bw[qubit] if hasattr(hw.dacs.qubit, "filter_bw") else 1000
    if atten1_dac is None:
        atten1_dac = hw.dacs.qubit.attn1[qubit]
    if atten2_dac is None:
        atten2_dac = hw.dacs.qubit.attn2[qubit]

    fc = center_freq if center_freq is not None else drive_freq

    # Set bandpass filter (MHz -> GHz)
    soc.rfb_set_gen_filter(dac_ch, fc=fc / 1000, ftype="bandpass", bw=bw / 1000)

    # Set attenuators
    soc.rfb_set_gen_rf(dac_ch, atten1_dac, atten2_dac)

    if cfg_file is not None:
        from . import config
        _update_active_state(cfg, "dac_qubit", dac_ch, fc, bw, "bandpass",
                             qubit=qubit, attn1=atten1_dac, attn2=atten2_dac)
        config.update_rfboard_active(cfg_file, "dac_qubit", dac_ch,
                                     cfg.hw.soc.rfboard_active["dac_qubit"][str(dac_ch)])

    return {"DAC_ch": dac_ch, "drive_freq": drive_freq}


def set_dc_bias(qubit, soc, cfg, dc_val=None, cfg_file=None, verbose=True):
    """
    Set the DC bias for a given qubit's flux line.

    Reads ``dc_ch`` and ``dc_val`` from ``cfg.hw.soc.dacs.flux``.
    If ``dc_val`` is provided it overrides the config value and updates it.

    Args:
        qubit: Qubit index
        soc: QICK SoC object
        cfg: Configuration AttrDict
        dc_val: Override DC bias voltage (default: from config)
        cfg_file: Optional path to persist updated dc_val to disk
        verbose: Print what was set (default: True)

    Returns:
        dict: ``{"dc_ch", "dc_val"}``
    """
    hw = cfg.hw.soc
    if not hasattr(hw.dacs, "flux"):
        raise ValueError("No flux DAC section in config")

    dc_ch = hw.dacs.flux.dc_ch[qubit]
    if dc_val is None:
        dc_val = hw.dacs.flux.dc_val[qubit]
    else:
        hw.dacs.flux.dc_val[qubit] = dc_val

    soc.rfb_set_bias(dc_ch, float(dc_val))

    if verbose:
        print(f"DC bias qubit {qubit}: ch={dc_ch}, val={dc_val} V")

    if cfg_file is not None:
        from . import config
        config.save(cfg, cfg_file)

    return {"dc_ch": dc_ch, "dc_val": dc_val}


def set_bandpass_rf(
    soc,
    gen_ch,
    ro_ch,
    fc_MHz,
    bw_MHz,
    tx_att1=None,
    tx_att2=None,
    ro_att=None,
    settle_ms=30,
    enforce_limits=True,
    verbose=True,
):
    """
    Configure ADMV8818 bandpass on both TX (gen_ch) and RX (ro_ch).

    Lower-level function with frequency validation, error handling,
    and settle time.

    Args:
        soc: QICK SoC object
        gen_ch: Generator channel index (DAC)
        ro_ch: Readout channel index (ADC)
        fc_MHz: Bandpass center frequency in MHz
        bw_MHz: Bandpass bandwidth in MHz
        tx_att1: Optional TX attenuator 1 in dB
        tx_att2: Optional TX attenuator 2 in dB
        ro_att: Optional RX attenuator in dB
        settle_ms: Pause after programming filters in ms (default: 30)
        enforce_limits: Warn if outside ADMV8818 2-18 GHz span (default: True)
        verbose: Print what's being applied (default: True)

    Returns:
        tuple: (f_lo_MHz, f_hi_MHz) — the filter band edges

    Raises:
        ValueError: If bw_MHz <= 0 or lower edge <= 0
    """
    if bw_MHz <= 0:
        raise ValueError("bw_MHz must be > 0.")
    f_lo = fc_MHz - bw_MHz / 2.0
    f_hi = fc_MHz + bw_MHz / 2.0
    if f_lo <= 0:
        raise ValueError("Lower edge must be > 0 MHz.")

    # ADMV8818 sanity (datasheet span is ~2-18 GHz)
    if enforce_limits:
        if f_lo < 2000:
            print(
                f"WARNING: lower edge {f_lo:.1f} MHz is below ~2000 MHz "
                "— outside the ADMV8818's typical span (2–18 GHz)."
            )
        if f_hi > 18000:
            print(
                f"WARNING: upper edge {f_hi:.1f} MHz is above ~18000 MHz "
                "— outside the ADMV8818's typical span (2–18 GHz)."
            )

    # Optional on-board attenuators
    if tx_att1 is not None or tx_att2 is not None:
        try:
            soc.rfb_set_gen_rf(gen_ch=gen_ch, att1=tx_att1 or 0, att2=tx_att2 or 0)
        except Exception as e:
            if verbose:
                print("rfb_set_gen_rf skipped/failed:", e)

    if ro_att is not None:
        try:
            soc.rfb_set_ro_rf(ro_ch=ro_ch, att=ro_att)
        except Exception as e:
            if verbose:
                print("rfb_set_ro_rf skipped/failed:", e)

    # Program ADMV8818 bandpass on both sides (MHz -> GHz)
    fc_GHz = fc_MHz / 1000.0
    bw_GHz = bw_MHz / 1000.0

    try:
        soc.rfb_set_gen_filter(gen_ch=gen_ch, ftype="bandpass", fc=fc_GHz, bw=bw_GHz)
    except Exception as e:
        if verbose:
            print("rfb_set_gen_filter failed:", e)

    try:
        soc.rfb_set_ro_filter(ro_ch=ro_ch, ftype="bandpass", fc=fc_GHz, bw=bw_GHz)
    except Exception as e:
        if verbose:
            print("rfb_set_ro_filter failed:", e)

    # Let the RO decimator / ADMV settle, then clear sticky flags
    time.sleep(max(0, settle_ms) / 1000.0)
    try:
        soc.clear_interrupts(error_on_interrupt=False)
    except Exception:
        pass

    if verbose:
        print(
            f"Band-pass set: fc={fc_MHz:.1f} MHz, bw={bw_MHz:.1f} MHz "
            f"(edges {f_lo:.1f}–{f_hi:.1f} MHz). "
            f"gen_ch={gen_ch}, ro_ch={ro_ch}"
        )

    return f_lo, f_hi


def bypass_all_filters(soc, gen_channels, ro_channels, cfg=None, cfg_file=None):
    """
    Bypass (disable) filters on all specified generator and readout channels.

    Args:
        soc: QICK SoC object
        gen_channels: List of generator (DAC) channel indices to bypass
        ro_channels: List of readout (ADC) channel indices to bypass
        cfg: Optional configuration AttrDict for updating active state in memory
        cfg_file: Optional path to persist active state to disk
    """
    for ch in gen_channels:
        soc.rfb_set_gen_filter(ch, fc=0, ftype="bypass")
    for ch in ro_channels:
        soc.rfb_set_ro_filter(ch, fc=0, ftype="bypass")

    if cfg is not None:
        for ch in ro_channels:
            _update_active_state(cfg, "adc", ch, 0, 0, "bypass", attn=0)
        for ch in gen_channels:
            # Mark both dac_readout and dac_qubit if they exist
            _update_active_state(cfg, "dac_readout", ch, 0, 0, "bypass",
                                 attn1=0, attn2=0)
            if hasattr(cfg.hw.soc.dacs, "qubit"):
                _update_active_state(cfg, "dac_qubit", ch, 0, 0, "bypass",
                                     attn1=0, attn2=0)

        if cfg_file is not None:
            from . import config
            active = cfg.hw.soc.rfboard_active
            for ch in ro_channels:
                config.update_rfboard_active(cfg_file, "adc", ch,
                                             active["adc"][str(ch)])
            for ch in gen_channels:
                if "dac_readout" in active and str(ch) in active["dac_readout"]:
                    config.update_rfboard_active(cfg_file, "dac_readout", ch,
                                                 active["dac_readout"][str(ch)])
                if "dac_qubit" in active and str(ch) in active["dac_qubit"]:
                    config.update_rfboard_active(cfg_file, "dac_qubit", ch,
                                                 active["dac_qubit"][str(ch)])


def apply_config_rf_settings(soc, cfg, qubit=None, cfg_file=None):
    """
    Read stored filter/attn settings from config and push them to hardware.

    If ``qubit`` is specified, only that qubit's channels are configured.
    Otherwise all channels are configured.

    Args:
        soc: QICK SoC object
        cfg: Configuration AttrDict (must have hw.soc.adcs/dacs with filter fields)
        qubit: Optional qubit index to configure (default: all)
        cfg_file: Optional path to persist active state to disk (default: None)
    """
    # Reload config from disk so that manual yml edits are picked up
    if cfg_file is not None:
        from . import config
        cfg = config.load(cfg_file)

    hw = cfg.hw.soc

    if qubit is not None:
        indices = [qubit]
    else:
        indices = range(len(hw.adcs.readout.ch))

    for qi in indices:
        # --- ADC readout ---
        adc_ch = hw.adcs.readout.ch[qi]
        adc_ftype = hw.adcs.readout.filter_type[qi]
        adc_fc = hw.adcs.readout.filter_fc[qi]
        adc_bw = hw.adcs.readout.filter_bw[qi]
        adc_attn = hw.adcs.readout.attn[qi]
        if adc_ftype != "bypass":
            soc.rfb_set_ro_filter(
                adc_ch,
                fc=adc_fc / 1000,
                ftype=adc_ftype,
                bw=adc_bw / 1000,
            )
        else:
            soc.rfb_set_ro_filter(adc_ch, fc=0, ftype="bypass")
        soc.rfb_set_ro_rf(adc_ch, adc_attn)
        _update_active_state(cfg, "adc", adc_ch, adc_fc, adc_bw, adc_ftype,
                             qubit=qi, attn=adc_attn)

        # --- DAC readout ---
        dac_ro_ch = hw.dacs.readout.ch[qi]
        dac_ro_ftype = hw.dacs.readout.filter_type[qi]
        dac_ro_fc = hw.dacs.readout.filter_fc[qi]
        dac_ro_bw = hw.dacs.readout.filter_bw[qi]
        dac_ro_attn1 = hw.dacs.readout.attn1[qi]
        dac_ro_attn2 = hw.dacs.readout.attn2[qi]
        if dac_ro_ftype != "bypass":
            soc.rfb_set_gen_filter(
                dac_ro_ch,
                fc=dac_ro_fc / 1000,
                ftype=dac_ro_ftype,
                bw=dac_ro_bw / 1000,
            )
        else:
            soc.rfb_set_gen_filter(dac_ro_ch, fc=0, ftype="bypass")
        soc.rfb_set_gen_rf(dac_ro_ch, dac_ro_attn1, dac_ro_attn2)
        _update_active_state(cfg, "dac_readout", dac_ro_ch, dac_ro_fc, dac_ro_bw,
                             dac_ro_ftype, qubit=qi, attn1=dac_ro_attn1, attn2=dac_ro_attn2)

        # --- DAC qubit (only if section exists) ---
        if hasattr(hw.dacs, "qubit"):
            dac_q_ch = hw.dacs.qubit.ch[qi]
            dac_q_ftype = hw.dacs.qubit.filter_type[qi]
            dac_q_fc = hw.dacs.qubit.filter_fc[qi]
            dac_q_bw = hw.dacs.qubit.filter_bw[qi]
            dac_q_attn1 = hw.dacs.qubit.attn1[qi]
            dac_q_attn2 = hw.dacs.qubit.attn2[qi]
            if dac_q_ftype != "bypass":
                soc.rfb_set_gen_filter(
                    dac_q_ch,
                    fc=dac_q_fc / 1000,
                    ftype=dac_q_ftype,
                    bw=dac_q_bw / 1000,
                )
            else:
                soc.rfb_set_gen_filter(dac_q_ch, fc=0, ftype="bypass")
            soc.rfb_set_gen_rf(dac_q_ch, dac_q_attn1, dac_q_attn2)
            _update_active_state(cfg, "dac_qubit", dac_q_ch, dac_q_fc, dac_q_bw,
                                 dac_q_ftype, qubit=qi, attn1=dac_q_attn1, attn2=dac_q_attn2)

        # --- DC bias (only if flux section with dc_ch exists) ---
        if hasattr(hw.dacs, "flux") and hasattr(hw.dacs.flux, "dc_ch"):
            dc_ch = hw.dacs.flux.dc_ch[qi]
            dc_val = hw.dacs.flux.dc_val[qi]
            soc.rfb_set_bias(dc_ch, float(dc_val))

    if cfg_file is not None:
        from . import config
        config.save(cfg, cfg_file)


def activate_qubit_rf(qubit, soc, cfg, cfg_file=None,
                      channels=("adc", "dac_readout", "dac_qubit", "dc_bias"), verbose=True):
    """
    Switch shared RF channels to a specific qubit's ideal settings.

    Reads per-qubit filter/attn settings from the config arrays
    (``filter_fc``, ``filter_bw``, ``filter_type``, ``attn*``) and programs
    the hardware, then updates ``rfboard_active`` in memory (and on disk
    if ``cfg_file`` is provided).

    Args:
        qubit: Qubit index
        soc: QICK SoC object
        cfg: Configuration AttrDict
        cfg_file: Optional path to persist active state to disk
        channels: Tuple of channel types to activate
                  (default: all three — ``"adc"``, ``"dac_readout"``, ``"dac_qubit"``)
        verbose: Print summary of what was activated (default: True)

    Returns:
        AttrDict or cfg: Updated cfg (reloaded from disk if ``cfg_file`` given)
    """
    # Reload config from disk so that manual yml edits are picked up
    if cfg_file is not None:
        from . import config
        cfg = config.load(cfg_file)

    hw = cfg.hw.soc
    parts = []

    if "adc" in channels:
        adc_ch = hw.adcs.readout.ch[qubit]
        adc_fc = hw.adcs.readout.filter_fc[qubit]
        adc_bw = hw.adcs.readout.filter_bw[qubit]
        adc_ftype = hw.adcs.readout.filter_type[qubit]
        adc_attn = hw.adcs.readout.attn[qubit]

        if adc_ftype != "bypass":
            soc.rfb_set_ro_filter(adc_ch, fc=adc_fc / 1000, ftype=adc_ftype,
                                  bw=adc_bw / 1000)
        else:
            soc.rfb_set_ro_filter(adc_ch, fc=0, ftype="bypass")
        soc.rfb_set_ro_rf(adc_ch, adc_attn)

        _update_active_state(cfg, "adc", adc_ch, adc_fc, adc_bw, adc_ftype,
                             qubit=qubit, attn=adc_attn)
        parts.append(f"ADC ch {adc_ch} (fc={adc_fc}, bw={adc_bw}, {adc_ftype})")

    if "dac_readout" in channels:
        dac_ro_ch = hw.dacs.readout.ch[qubit]
        dac_ro_fc = hw.dacs.readout.filter_fc[qubit]
        dac_ro_bw = hw.dacs.readout.filter_bw[qubit]
        dac_ro_ftype = hw.dacs.readout.filter_type[qubit]
        dac_ro_attn1 = hw.dacs.readout.attn1[qubit]
        dac_ro_attn2 = hw.dacs.readout.attn2[qubit]

        if dac_ro_ftype != "bypass":
            soc.rfb_set_gen_filter(dac_ro_ch, fc=dac_ro_fc / 1000, ftype=dac_ro_ftype,
                                   bw=dac_ro_bw / 1000)
        else:
            soc.rfb_set_gen_filter(dac_ro_ch, fc=0, ftype="bypass")
        soc.rfb_set_gen_rf(dac_ro_ch, dac_ro_attn1, dac_ro_attn2)

        _update_active_state(cfg, "dac_readout", dac_ro_ch, dac_ro_fc, dac_ro_bw,
                             dac_ro_ftype, qubit=qubit, attn1=dac_ro_attn1, attn2=dac_ro_attn2)
        parts.append(f"DAC-ro ch {dac_ro_ch} (fc={dac_ro_fc}, bw={dac_ro_bw}, {dac_ro_ftype})")

    if "dac_qubit" in channels and hasattr(hw.dacs, "qubit"):
        dac_q_ch = hw.dacs.qubit.ch[qubit]
        dac_q_fc = hw.dacs.qubit.filter_fc[qubit]
        dac_q_bw = hw.dacs.qubit.filter_bw[qubit]
        dac_q_ftype = hw.dacs.qubit.filter_type[qubit]
        dac_q_attn1 = hw.dacs.qubit.attn1[qubit]
        dac_q_attn2 = hw.dacs.qubit.attn2[qubit]

        if dac_q_ftype != "bypass":
            soc.rfb_set_gen_filter(dac_q_ch, fc=dac_q_fc / 1000, ftype=dac_q_ftype,
                                   bw=dac_q_bw / 1000)
        else:
            soc.rfb_set_gen_filter(dac_q_ch, fc=0, ftype="bypass")
        soc.rfb_set_gen_rf(dac_q_ch, dac_q_attn1, dac_q_attn2)

        _update_active_state(cfg, "dac_qubit", dac_q_ch, dac_q_fc, dac_q_bw,
                             dac_q_ftype, qubit=qubit, attn1=dac_q_attn1, attn2=dac_q_attn2)
        parts.append(f"DAC-qb ch {dac_q_ch} (fc={dac_q_fc}, bw={dac_q_bw}, {dac_q_ftype})")

    if "dc_bias" in channels and hasattr(hw.dacs, "flux") and hasattr(hw.dacs.flux, "dc_ch"):
        dc_ch = hw.dacs.flux.dc_ch[qubit]
        dc_val = hw.dacs.flux.dc_val[qubit]
        soc.rfb_set_bias(dc_ch, float(dc_val))
        parts.append(f"DC bias ch {dc_ch} (val={dc_val} V)")

    if verbose:
        print(f"Activated qubit {qubit}: " + ", ".join(parts))

    if cfg_file is not None:
        from . import config
        return config.save(cfg, cfg_file)

    return cfg


def get_active_rf_state(cfg, channel_type=None, ch=None):
    """
    Return the current ``rfboard_active`` state from the config.

    Args:
        cfg: Configuration AttrDict
        channel_type: Optional — ``"adc"``, ``"dac_readout"``, or ``"dac_qubit"``.
                      If ``None``, returns the full active dict.
        ch: Optional physical channel number (int). Requires ``channel_type``.

    Returns:
        dict: Full active state, a single channel-type dict, or a single
              channel entry depending on arguments.
    """
    active = cfg.hw.soc.get("rfboard_active", {})
    if channel_type is None:
        return active
    section = active.get(channel_type, {})
    if ch is None:
        return section
    return section.get(str(ch), {})
