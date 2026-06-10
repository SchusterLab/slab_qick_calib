"""
Resonator Spectroscopy vs Fast Flux Gain

Sweeps the resonator spectroscopy across flux gain values to map the
dispersive shift and resonator frequency as a function of flux.
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ..general.qick_flux_experiment import QickFluxSweep
from ..single_qubit.resonator_spectroscopy import ResSpec

XLABEL = "Readout Frequency (MHz)"


class ResSpecFastFlux(QickFluxSweep):
    """
    ResSpecFastFlux: 2D resonator spectroscopy vs fast flux gain.

    Runs a ResSpec at each flux gain point, collecting a 2D map of resonator
    amplitude vs (frequency, flux gain) and the fitted resonator frequency at
    each gain point.

    Sweep parameters (passed via params dict):
        gain_start: Starting flux gain value (default: sweet_spot_ac)
        gain_stop: Ending flux gain value (default: sweet_spot_ac + 0.4)
        expts_gain: Number of gain points (default: 50)
        direction: 'pos' or 'neg' — sweep direction from sweet spot (default: 'pos')
        freq_span: Frequency span in MHz. When provided with a flux model, converts
            to a gain range. Overrides gain_stop.
        lin_freq: If True and a flux model is available, space points linearly in
            frequency instead of linearly in gain (default: True)

    All other params are forwarded to the inner ResSpec experiment (span, expts,
    gain, reps, etc.).
    """

    _inner_expt_class = ResSpec

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        progress=True,
        display=True,
    ):
        if prefix is None:
            prefix = f"resspec_fastflux_qubit{qi}"

        super().__init__(cfg_dict=cfg_dict, prefix=prefix, qi=qi)

        sweet_spot, gain_stop = self._init_flux_sweep(qi, params)
        direction = params.get("direction", "pos")
        freq_span = params.get("freq_span", None)

        params_def = {
            "gain_start": sweet_spot,
            "gain_stop": gain_stop,
            "expts_gain": 50,
            "lin_freq": False,
            "freq_span": freq_span,
            "direction": direction,
        }

        params = {**params_def, **params, "flux": True}

        self.expt = self._inner_expt_class(
            cfg_dict, qi=qi, go=False, params=params, check_params=False
        )
        params = {**self.expt.cfg.expt, **params}
        self.cfg.expt = {**params_def, **params}
        self.expt.cfg.expt = {**self.expt.cfg.expt, **params}

        self._qi = qi
        self._cfg_dict = cfg_dict

        self._build_gain_freq_pts()

        if go:
            self.acquire(progress=progress, display=display)
            self.display()

    def acquire(self, progress=True, display=False):
        """
        Run a ResSpec at each flux gain point and fit the resonator frequency.

        Returns:
            dict: 2D amps/avgi/avgq/phases (gain × frequency), the shared
            frequency axis, gain_pts, freq_pts (if a flux model exists), and
            res_freq_list with the fitted resonator frequency per gain point
        """
        data = {}
        res_freq_list = []

        for i, g in enumerate(tqdm(self.gain_pts, disable=not progress)):
            self.expt.cfg.expt["flux_gain"] = float(g)

            data_new = self.expt.acquire(progress=False)
            try:
                self.expt.analyze(data=data_new)
            except Exception:
                pass

            for key in ("avgi", "avgq", "amps", "phases", "xpts", "freq"):
                if key in data_new:
                    if i == 0:
                        data[key] = []
                    data[key].append(data_new[key])

            # Extract fitted resonator frequency (absolute MHz)
            if "freq_fit" in data_new and data_new["freq_fit"] is not None:
                res_freq = float(data_new["freq_fit"][0])
            else:
                res_freq = np.nan
            res_freq_list.append(res_freq)

            data["gain_pts"] = self.gain_pts
            if self.freq_pts is not None:
                data["freq_pts"] = self.freq_pts
            data["res_freq_list"] = np.array(res_freq_list)

            if self.save_interim:
                self.save_data(data=data)

        # Convert per-row lists to 2D arrays
        for k in ("avgi", "avgq", "amps", "phases"):
            if k in data:
                data[k] = np.array(data[k])

        # All rows share the same frequency axis; keep only the first
        if "xpts" in data:
            data["xpts"] = np.array(data["xpts"][0])
        if "freq" in data:
            data["freq"] = np.array(data["freq"][0])

        data["ypts"] = self.gain_pts
        self.data = data
        return data

    def display(self, data=None, **kwargs):
        """Plot the amplitude map vs (frequency, flux gain) and the fitted resonator frequency vs gain."""
        if data is None:
            data = self.data

        gain_pts = data["gain_pts"]
        # Use absolute frequency array for x-axis when available
        freq_axis = data.get("freq", data.get("xpts"))
        amps = data["amps"]
        qi = self.cfg.expt["qubit"][0]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # --- Color plot: amplitude vs (frequency, flux gain) ---
        ax = axes[0]
        im = ax.pcolormesh(
            freq_axis, gain_pts, amps, cmap="viridis", shading="auto", rasterized=True
        )
        plt.colorbar(im, ax=ax, label="Amplitude (ADC units)")
        ax.set_xlabel(XLABEL)
        ax.set_ylabel("Flux Gain")
        ax.tick_params(top=True, bottom=True, right=True)

        # --- 1D plot: resonator frequency vs flux gain ---
        ax = axes[1]
        ax.plot(gain_pts, data["res_freq_list"], "o-", markersize=3)
        ax.set_xlabel("Flux Gain")
        ax.set_ylabel("Resonator Frequency (MHz)")
        ax.tick_params(top=True, bottom=True, right=True)

        self.save_fig(fig)
        self.save_config()
        plt.show()

        return fig
