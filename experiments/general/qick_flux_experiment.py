"""
Base class for experiments that sweep over flux gain.

QickFluxSweep extends QickExperiment2DSimple with two helper methods
(_init_flux_sweep and _build_gain_freq_pts) that are duplicated across
every flux-gain sweep class (QubitSpecFastFlux, T1FastFlux, T2FastFlux,
T1PredistFastFlux).  Centralising them here removes ~40 lines of
identical boilerplate from each subclass.
"""

import numpy as np

from ..general.qick_experiment import QickExperiment2DSimple
from ...analysis import fitting as fitter


class QickFluxSweep(QickExperiment2DSimple):
    """
    Base class for 2D experiments that sweep an inner experiment over flux gain.

    Subclasses inherit two helpers:

    _init_flux_sweep(qi, params, default_offset=0.4)
        Reads the flux model from config (or from ``params["flux_converter"]``),
        computes the sweet-spot gain and the gain_stop value, and stores the
        converter as ``self._flux_converter``.  Returns ``(sweet_spot, gain_stop)``.

    _build_gain_freq_pts()
        Builds ``self.gain_pts`` and ``self.freq_pts`` from ``self.cfg.expt``
        (uses ``gain_start``, ``gain_stop``, ``expts_gain``, ``lin_freq``) and
        ``self._flux_converter``.
    """

    def _init_flux_sweep(self, qi, params, default_offset=0.4):
        """Load flux converter and resolve gain sweep endpoints.

        Reads ``cfg.hw.soc.dacs.flux`` and ``cfg.device.qubit`` from the already-
        initialised ``self.cfg``.  Supports ``flux_converter``, ``freq_span``,
        ``gain_stop``, and ``direction`` keys in *params*.

        Args:
            qi: Qubit index.
            params: Raw params dict (read-only; use .get(), not .pop()).
            default_offset: Gain range from sweet spot when no other constraint
                is given (default 0.4).

        Returns:
            (sweet_spot, gain_stop): Both as floats.

        Side-effect:
            Sets ``self._flux_converter``.
        """
        cfg_flux = getattr(
            getattr(getattr(getattr(self.cfg, "hw", None), "soc", None), "dacs", None),
            "flux", None,
        )
        cfg_qubit = getattr(getattr(self.cfg, "device", None), "qubit", None)

        sweet_spot = 0.0
        if cfg_qubit is not None and hasattr(cfg_qubit, "sweet_spot_ac"):
            sweet_spot = cfg_qubit.sweet_spot_ac[qi]

        direction = params.get("direction", "pos")
        sign = 1 if direction == "pos" else -1

        converter = params.get("flux_converter", None)
        if converter is None:
            converter = fitter.flux_converter_from_config(cfg_flux, cfg_qubit, qi, direction)

        freq_span = params.get("freq_span", None)

        if freq_span is not None and converter is not None and "gain_stop" not in params:
            f_sweet = converter.gain_to_freq(sweet_spot)
            f_target = f_sweet - freq_span
            gain_stop = float(converter.freq_to_gain(f_target))
        elif "gain_stop" not in params:
            gain_stop = sweet_spot + sign * default_offset
        else:
            gain_stop = params["gain_stop"]

        self._flux_converter = converter
        return sweet_spot, gain_stop

    def _build_gain_freq_pts(self):
        """Build ``self.gain_pts`` and ``self.freq_pts`` from ``cfg.expt``.

        Reads ``gain_start``, ``gain_stop``, ``expts_gain``, and ``lin_freq``
        from ``self.cfg.expt``, together with ``self._flux_converter``.

        When ``lin_freq`` is True and a converter is available, points are
        spaced linearly in frequency; otherwise they are spaced linearly in
        gain.  ``self.freq_pts`` is set to None when no converter is available.
        """
        cfg_e = self.cfg.expt
        converter = self._flux_converter

        if cfg_e["lin_freq"] and converter is not None:
            freq_start = converter.gain_to_freq(cfg_e["gain_start"])
            freq_stop = converter.gain_to_freq(cfg_e["gain_stop"])
            self.freq_pts = np.linspace(
                float(freq_start), float(freq_stop), int(cfg_e["expts_gain"])
            )
            self.gain_pts = converter.freq_to_gain(self.freq_pts)
        else:
            self.gain_pts = np.linspace(
                cfg_e["gain_start"], cfg_e["gain_stop"], int(cfg_e["expts_gain"])
            )
            if converter is not None:
                self.freq_pts = converter.gain_to_freq(self.gain_pts)
            else:
                self.freq_pts = None
