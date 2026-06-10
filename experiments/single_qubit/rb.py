"""
Single-Qubit Randomized Benchmarking (RB)

Measures average single-qubit Clifford gate fidelity by applying random
sequences of Clifford gates of varying depths, followed by a computed
recovery gate, then measuring survival probability.

The survival probability decays exponentially with sequence depth:
    p(m) = A * alpha^m + B

The average gate fidelity is F = (1 + alpha) / 2, and the error per
Clifford (EPC) is r = (1 - alpha) / 2.

Each (depth, seed) combination requires a unique random gate sequence,
so a new QICK program is compiled per point (outer Python loop).
"""

import numpy as np
from tqdm import tqdm

from ...analysis import fitting as fitter
from ..general.qick_experiment import QickExperiment, DataProcessor, get_current_time_string
from ..general.qick_program import QickProgram
from ...exp_handling.datamanagement import AttrDict


# ====================================================== #
# Single-Qubit Clifford Group
# ====================================================== #

# The 24 single-qubit Clifford gates decomposed into physical gates.
# Physical gate naming: pi_P = pi pulse at phase P degrees,
#                       pi2_P = pi/2 pulse at phase P degrees.
# Gates are listed in application order (first in list applied first).
# Derived from 2x2 unitary representations; verified for closure,
# inverses, and associativity.

CLIFFORD_GATES = [
    [],                                    # C0:  I
    ["pi_0"],                              # C1:  X
    ["pi_90"],                             # C2:  Y
    ["pi2_0"],                             # C3:  X/2
    ["pi2_90"],                            # C4:  Y/2
    ["pi2_180"],                           # C5:  -X/2
    ["pi2_270"],                           # C6:  -Y/2
    ["pi_0", "pi_90"],                     # C7:  Z (up to global phase)
    ["pi_0", "pi2_90"],                    # C8
    ["pi_0", "pi2_270"],                   # C9
    ["pi_90", "pi2_0"],                    # C10
    ["pi_90", "pi2_180"],                  # C11
    ["pi2_0", "pi2_90"],                   # C12
    ["pi2_0", "pi2_270"],                  # C13
    ["pi2_90", "pi2_0"],                   # C14
    ["pi2_90", "pi2_180"],                 # C15
    ["pi2_180", "pi2_90"],                 # C16
    ["pi2_180", "pi2_270"],               # C17
    ["pi2_270", "pi2_0"],                  # C18
    ["pi2_270", "pi2_180"],               # C19
    ["pi2_0", "pi2_90", "pi2_0"],          # C20
    ["pi2_0", "pi2_90", "pi2_180"],        # C21
    ["pi2_0", "pi2_270", "pi2_0"],         # C22
    ["pi2_0", "pi2_270", "pi2_180"],       # C23
]

# Multiplication table: CLIFFORD_MULT_TABLE[i, j] = index of C_j @ C_i
# (i.e., apply C_i first, then C_j)
CLIFFORD_MULT_TABLE = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    [1, 0, 7, 5, 8, 3, 9, 2, 4, 6, 11, 10, 16, 17, 19, 18, 12, 13, 15, 14, 23, 22, 21, 20],
    [2, 7, 0, 10, 6, 11, 4, 1, 9, 8, 3, 5, 17, 16, 18, 19, 13, 12, 14, 15, 21, 20, 23, 22],
    [3, 5, 11, 1, 12, 0, 13, 10, 16, 17, 2, 7, 8, 9, 20, 21, 4, 6, 22, 23, 19, 18, 15, 14],
    [4, 9, 6, 14, 2, 15, 0, 8, 1, 7, 18, 19, 20, 23, 10, 11, 22, 21, 3, 5, 17, 12, 13, 16],
    [5, 3, 10, 0, 16, 1, 17, 11, 12, 13, 7, 2, 4, 6, 23, 22, 8, 9, 21, 20, 14, 15, 18, 19],
    [6, 8, 4, 18, 0, 19, 2, 9, 7, 1, 14, 15, 21, 22, 3, 5, 23, 20, 10, 11, 12, 17, 16, 13],
    [7, 2, 1, 11, 9, 10, 8, 0, 6, 4, 5, 3, 13, 12, 15, 14, 17, 16, 19, 18, 22, 23, 20, 21],
    [8, 6, 9, 19, 7, 18, 1, 4, 0, 2, 15, 14, 23, 20, 11, 10, 21, 22, 5, 3, 13, 16, 17, 12],
    [9, 4, 8, 15, 1, 14, 7, 6, 2, 0, 19, 18, 22, 21, 5, 3, 20, 23, 11, 10, 16, 13, 12, 17],
    [10, 11, 5, 7, 17, 2, 16, 3, 13, 12, 0, 1, 9, 8, 21, 20, 6, 4, 23, 22, 15, 14, 19, 18],
    [11, 10, 3, 2, 13, 7, 12, 5, 17, 16, 1, 0, 6, 4, 22, 23, 9, 8, 20, 21, 18, 19, 14, 15],
    [12, 17, 13, 20, 11, 21, 3, 16, 5, 10, 22, 23, 19, 14, 2, 7, 15, 18, 1, 0, 6, 8, 9, 4],
    [13, 16, 12, 22, 3, 23, 11, 17, 10, 5, 20, 21, 18, 15, 1, 0, 14, 19, 2, 7, 8, 6, 4, 9],
    [14, 15, 19, 9, 20, 4, 23, 18, 22, 21, 6, 8, 1, 7, 17, 12, 2, 0, 13, 16, 5, 3, 11, 10],
    [15, 14, 18, 4, 22, 9, 21, 19, 20, 23, 8, 6, 2, 0, 16, 13, 1, 7, 12, 17, 10, 11, 3, 5],
    [16, 13, 17, 23, 10, 22, 5, 12, 3, 11, 21, 20, 14, 19, 7, 2, 18, 15, 0, 1, 9, 4, 6, 8],
    [17, 12, 16, 21, 5, 20, 10, 13, 11, 3, 23, 22, 15, 18, 0, 1, 19, 14, 7, 2, 4, 9, 8, 6],
    [18, 19, 15, 8, 21, 6, 22, 14, 23, 20, 4, 9, 7, 1, 12, 17, 0, 2, 16, 13, 11, 10, 5, 3],
    [19, 18, 14, 6, 23, 8, 20, 15, 21, 22, 9, 4, 0, 2, 13, 16, 7, 1, 17, 12, 3, 5, 10, 11],
    [20, 21, 23, 17, 19, 12, 14, 22, 15, 18, 13, 16, 5, 10, 6, 8, 11, 3, 9, 4, 0, 1, 7, 2],
    [21, 20, 22, 12, 15, 17, 18, 23, 19, 14, 16, 13, 11, 3, 4, 9, 5, 10, 8, 6, 2, 7, 1, 0],
    [22, 23, 21, 16, 18, 13, 15, 20, 14, 19, 12, 17, 10, 5, 8, 6, 3, 11, 4, 9, 7, 2, 0, 1],
    [23, 22, 20, 13, 14, 16, 19, 21, 18, 15, 17, 12, 3, 11, 9, 4, 10, 5, 6, 8, 1, 0, 2, 7],
])

# Inverse table: CLIFFORD_INVERSE[i] = j such that C_j @ C_i = I
CLIFFORD_INVERSE = np.array([
    0, 1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 11, 19, 15, 17, 13, 18, 14, 16, 12,
    20, 23, 22, 21,
])


# ====================================================== #
# Helper Functions
# ====================================================== #

def generate_rb_sequence(depth, rng):
    """Generate a random Clifford sequence with recovery gate.

    Args:
        depth: Number of random Cliffords (not counting recovery).
        rng: numpy random Generator instance.

    Returns:
        List of Clifford indices (length depth + 1, last is recovery).
    """
    seq = rng.integers(0, 24, size=depth)
    composed = 0  # identity
    for c in seq:
        composed = CLIFFORD_MULT_TABLE[composed, c]
    recovery = CLIFFORD_INVERSE[composed]
    return list(seq) + [recovery]


def decompose_sequence(clifford_indices):
    """Convert a list of Clifford indices to a flat list of pulse names.

    Args:
        clifford_indices: List of integers in [0, 23].

    Returns:
        List of pulse name strings (e.g. ["pi2_0", "pi_90", ...]).
    """
    gates = []
    for idx in clifford_indices:
        gates.extend(CLIFFORD_GATES[idx])
    return gates


def rb_decay(m, A, B, alpha):
    """RB decay model: p(m) = A * alpha^m + B."""
    return A * np.power(alpha, m) + B


def fit_rb_decay(xdata, ydata, fitparams=None):
    """Fit data to the RB decay model.

    Args:
        xdata: Sequence depths.
        ydata: Measured values (e.g. average I quadrature).
        fitparams: Optional initial guesses [A, B, alpha].

    Returns:
        Tuple of (pOpt, pCov, pInit).
    """
    if fitparams is None:
        fitparams = [None, None, None]
    if fitparams[0] is None:
        fitparams[0] = ydata[0] - ydata[-1]  # A
    if fitparams[1] is None:
        fitparams[1] = ydata[-1]              # B
    if fitparams[2] is None:
        fitparams[2] = 0.99                   # alpha

    bounds = ([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 1.0])
    return fitter.generic_fit(rb_decay, xdata, ydata, fitparams, bounds=bounds)


# ====================================================== #
# RB Program
# ====================================================== #

class RBProgram(QickProgram):
    """Pulse program for a single RB gate sequence.

    Plays a fixed list of physical gates (from cfg.expt.gate_list)
    followed by measurement. A new program instance is created for
    each random sequence.
    """

    def __init__(self, soccfg, final_delay, cfg, final_wait=0):
        super().__init__(soccfg, final_delay=final_delay, cfg=cfg, final_wait=0)

    def _initialize(self, cfg):
        cfg = AttrDict(self.cfg)
        super()._initialize(cfg, readout="standard")

        q = cfg.expt.qubit[0]

        # Build base pulse parameters from config
        pi_pulse = {
            key: value[q]
            for key, value in cfg.device.qubit.pulses.pi_ge.items()
        }
        pi_pulse["freq"] = cfg.device.qubit.f_ge[q]

        # Create pi gate pulses (full gain) at phase 0 and 90
        for phase in [0, 90]:
            pulse = dict(pi_pulse)
            pulse["phase"] = phase
            super().make_pulse(pulse, f"pi_{phase}")

        # Create pi/2 gate pulses (half gain) at phases 0, 90, 180, 270
        for phase in [0, 90, 180, 270]:
            pulse = dict(pi_pulse)
            pulse["gain"] = pi_pulse["gain"] / 2
            pulse["phase"] = phase
            super().make_pulse(pulse, f"pi2_{phase}")

        # pi_ge pulse for active reset
        if cfg.expt.active_reset:
            super().make_cfg_pulse(q, cfg.device.qubit.f_ge, "pi_ge")

    def _body(self, cfg):
        cfg = AttrDict(self.cfg)
        if self.adc_type == "dyn":
            self.send_readoutconfig(ch=self.adc_ch, name="readout", t=0)

        for i, gate_name in enumerate(cfg.expt.gate_list):
            self.pulse(ch=self.qubit_ch, name=gate_name, t=0)
            self.delay_auto(t=0.01, tag=f"g{i}")

        super().measure(cfg)


# ====================================================== #
# RB Experiment
# ====================================================== #

class RBExperiment(QickExperiment):
    """Single-qubit randomized benchmarking experiment.

    For each sequence depth, generates num_seeds random Clifford sequences,
    runs each on hardware, and averages the survival probability. Fits the
    decay to extract average Clifford gate fidelity.

    Usage:
        RBExperiment(cfg_dict, qi=0)
        RBExperiment(cfg_dict, qi=0, params={
            "depths": [1, 3, 5, 10, 20, 50, 100, 200],
            "num_seeds": 40,
            "rng_seed": 42,
        })
    """

    def __init__(
        self,
        cfg_dict,
        qi=0,
        go=True,
        params={},
        prefix=None,
        fname=None,
        progress=True,
        display=True,
        disp_kwargs=None,
    ):
        if prefix is None:
            prefix = f"rb_qubit{qi}"

        super().__init__(
            cfg_dict=cfg_dict, prefix=prefix, fname=fname, progress=progress, qi=qi,
        )

        cfg_qub = self.cfg.device.qubit.pulses.pi_ge

        params_def = {
            "reps": 2 * self.reps,
            "rounds": self.rounds,
            "depths": [1, 2, 5, 10, 20, 50, 100],
            "num_seeds": 20,
            "rng_seed": None,
            "qubit": [qi],
            "active_reset": self.cfg.device.readout.active_reset[qi],
            "qubit_chan": self.cfg.hw.soc.adcs.readout.ch[qi],
            "freq": self.cfg.device.qubit.f_ge[qi],
            "gate_list": [],
        }

        # Copy pi_ge pulse params from config
        for key in cfg_qub:
            params_def[key] = cfg_qub[key][qi]

        params = {**params_def, **params}
        self.cfg.expt = params

        super().check_params(params_def)

        super().qubit_run(
            qi=qi,
            go=go,
            display=display,
            progress=progress,
            disp_kwargs=disp_kwargs,
        )

    def acquire(self, progress=False):
        depths = np.array(self.cfg.expt.depths)
        num_seeds = self.cfg.expt.num_seeds

        rng = np.random.default_rng(self.cfg.expt.rng_seed)

        final_delay = self._get_final_delay()
        current_time = get_current_time_string()

        avgi_all = np.zeros((len(depths), num_seeds))
        avgq_all = np.zeros((len(depths), num_seeds))

        total = len(depths) * num_seeds
        pbar = tqdm(total=total, disable=not progress)

        for di, depth in enumerate(depths):
            for si in range(num_seeds):
                # Generate random sequence
                cliff_seq = generate_rb_sequence(int(depth), rng)
                gate_list = decompose_sequence(cliff_seq)

                # Update config with this sequence's gate list
                self.cfg.expt.gate_list = gate_list

                # Create and run program
                prog = RBProgram(
                    soccfg=self.soccfg,
                    final_delay=final_delay,
                    cfg=self.cfg,
                )
                iq_list = prog.acquire(
                    self.im[self.cfg.aliases.soc],
                    rounds=self.cfg.expt.rounds,
                    threshold=None,
                    progress=False,
                )

                # Extract single IQ point (no sweep, so shape is [1, 1, ...])
                iq = iq_list[0][0]
                avgi_all[di, si] = np.squeeze(iq[..., 0])
                avgq_all[di, si] = np.squeeze(iq[..., 1])

                pbar.update(1)

        pbar.close()

        # Average over seeds
        avgi = np.mean(avgi_all, axis=1)
        avgq = np.mean(avgq_all, axis=1)
        amps = np.abs((avgi + 1j * avgq))

        data = {
            "xpts": depths,
            "depths": depths,
            "avgi": avgi,
            "avgq": avgq,
            "amps": amps,
            "avgi_all": avgi_all,
            "avgq_all": avgq_all,
            "avgi_std": np.std(avgi_all, axis=1),
            "start_time": current_time,
        }

        for key in data:
            data[key] = np.array(data[key])

        self.data = data
        return data

    def analyze(self, data=None, fit=True, verbose=True, **kwargs):
        if data is None:
            data = self.data

        if not fit:
            return data

        xdata = data["xpts"].astype(float)
        ydata = data["avgi"].astype(float)

        pOpt, pCov, pInit = fit_rb_decay(xdata, ydata)

        data["fit_avgi"] = pOpt
        data["fit_err_avgi"] = pCov
        data["fit_init_avgi"] = pInit

        if not np.any(np.isnan(pOpt)):
            alpha = pOpt[2]
            data["alpha"] = alpha
            data["fidelity"] = (1 + alpha) / 2
            data["epc"] = (1 - alpha) / 2

            # R-squared
            y_pred = rb_decay(xdata, *pOpt)
            ss_res = np.sum((ydata - y_pred) ** 2)
            ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
            data["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            data["fit_err"] = np.mean(np.abs(np.diag(pCov[:3, :3])))

            if verbose:
                print(f"Avg Clifford fidelity: {data['fidelity']:.6f}")
                print(f"Error per Clifford:    {data['epc']:.2e}")
                print(f"Alpha (depol param):   {alpha:.6f}")
        else:
            data["alpha"] = np.nan
            data["fidelity"] = np.nan
            data["epc"] = np.nan
            data["r2"] = 0.0
            data["fit_err"] = np.inf

        self.fitfunc = rb_decay
        self.fitterfunc = fit_rb_decay
        self.status = data["r2"] > 0.5

        return data

    def display(self, data=None, fit=True, ax=None, savefig=True, **kwargs):
        import matplotlib.pyplot as plt

        if data is None:
            data = self.data

        save_fig = ax is None
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        else:
            fig = None
            if not isinstance(ax, list):
                ax = [ax]

        if isinstance(ax, list):
            a = ax[0]
        else:
            a = ax

        depths = data["xpts"]
        avgi = data["avgi"]

        a.plot(depths, avgi, "o", markersize=5)

        if fit and "fit_avgi" in data and not np.any(np.isnan(data["fit_avgi"])):
            p = data["fit_avgi"]
            m_fine = np.linspace(0, np.max(depths) * 1.05, 200)
            label = (
                f"F = {data['fidelity']:.4f}\n"
                f"EPC = {data['epc']:.2e}\n"
                f"$\\alpha$ = {p[2]:.4f}"
            )
            a.plot(m_fine, rb_decay(m_fine, *p), "-", label=label)
            a.legend()

        if "avgi_std" in data:
            a.errorbar(
                depths, avgi, yerr=data["avgi_std"] / np.sqrt(data.get("num_seeds", 1)),
                fmt="none", capsize=3, color="C0", alpha=0.5,
            )

        a.set_xlabel("Sequence Depth (Cliffords)")
        a.set_ylabel("I (ADC units)")

        if save_fig and savefig:
            self.save_fig(fig)
