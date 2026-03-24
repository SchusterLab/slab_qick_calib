"""
Test script for predistorted flux pulse spectroscopy.

Connects to the RFSoC, runs QubitSpecPredist on qubit 2, and displays results.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from qick import QickConfig

from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
from slab_qick_calib.helpers import config, rfboard, handy

# ── Configuration ──────────────────────────────────────────────────────
cfg_file = 'sample_50_rfboard.yml'
expt_path = r'C:\_Data\sample_50_new'
qi = 2  # qubit index

# Step-response fit parameters (from analyze_flux_step_response notebook)
flux_alphas = [0.0484, -0.1310, -1.9992]
flux_taus   = [0.018,   9.077,  12534.640]  # µs
flux_alpha0 = 1.0969

# ── Connect to hardware ───────────────────────────────────────────────
configs_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
cfg_path = os.path.join(configs_dir, cfg_file)
auto_cfg = config.load(cfg_path)

print("Connecting to RFSoC...")
im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'], port=8888)
print(im)

soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())
print(soccfg)

cfg_dict = {
    'soc': soccfg,
    'expt_path': expt_path,
    'cfg_file': cfg_path,
    'im': im,
}

# ── Activate qubit RF settings ────────────────────────────────────────
auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)

# Set DC flux bias
dc_flux = 0.0
dc_chan = auto_cfg['hw']['soc']['dacs']['flux']['dc_ch'][qi]
soc.rfb_set_bias(dc_chan, float(dc_flux))
print(f"DC bias ch {dc_chan} set to {dc_flux} V")

# ── Check envelope memory ─────────────────────────────────────────────
flux_ch = auto_cfg['hw']['soc']['dacs']['flux']['ch'][qi]
gen_cfg = soccfg['gens'][flux_ch]
print(f"\nFlux generator (ch {flux_ch}):")
print(f"  Type: {gen_cfg['type']}")
print(f"  f_fabric: {gen_cfg['f_fabric']:.1f} MHz")
print(f"  samps_per_clk: {gen_cfg['samps_per_clk']}")
print(f"  maxlen: {gen_cfg['maxlen']}")
print(f"  maxv (with scale): {soccfg.get_maxv(flux_ch)}")

# Estimate envelope size
flux_lead = 0.035  # µs
flux_trail = 0.025  # µs
qubit_length = 10   # µs
total_us = qubit_length + flux_lead + flux_trail
n_samps = int(total_us * gen_cfg['f_fabric'])
print(f"\n  Envelope for {total_us:.3f} µs pulse: {n_samps} samples "
      f"({'OK' if n_samps < gen_cfg['maxlen'] else 'TOO LONG'})")

# ── Run predistorted flux spectroscopy ────────────────────────────────
from slab_qick_calib.experiments.flux.qubit_spec_predistorted import QubitSpecPredist

print("\n" + "="*60)
print("Running QubitSpecPredist (slow-only first)...")
print("="*60)

import matplotlib
matplotlib.use('Agg')  # non-interactive backend

try:
    expt = QubitSpecPredist(cfg_dict, qi=qi, style="medium", go=False, params={
        "flux_alphas": flux_alphas,
        "flux_taus": flux_taus,
        "flux_alpha0": flux_alpha0,
        "flux_slow_only": True,   # start with slow terms only
        "flux_gain": -0.35,
        "length": 10,
    })

    # Run acquire + analyze separately (no display blocking)
    print("Acquiring...")
    data = expt.acquire(progress=True)
    print("Analyzing...")
    data = expt.analyze(data=data)

    print("\nExperiment completed successfully!")
    print(f"  Data keys: {list(data.keys())}")
    if 'best_fit' in data:
        bf = data['best_fit']
        print(f"  Fitted freq: {bf[2]:.4f} MHz")
        print(f"  Kappa: {bf[3]:.4f} MHz")
    print(f"  xpts range: [{data['xpts'].min():.2f}, {data['xpts'].max():.2f}] MHz")
    print(f"  amps range: [{data['amps'].min():.4f}, {data['amps'].max():.4f}]")

    # Save plot to file
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data['xpts'], data['amps'])
    if 'best_fit' in data and not np.any(np.isnan(data['best_fit'])):
        from slab_qick_calib.analysis.fitting import lorfunc
        xfit = np.linspace(data['xpts'].min(), data['xpts'].max(), 500)
        ax.plot(xfit, lorfunc(xfit, *data['best_fit']), 'r-', lw=2)
        ax.axvline(data['best_fit'][2], color='gray', ls='--', alpha=0.5)
    ax.set_xlabel("Qubit Frequency (MHz)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    plot_path = os.path.join(expt_path, "data", "test_predistorted_flux.png")
    fig.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved to: {plot_path}")

    # Also save data
    expt.save_data(data=data)
    print(f"  Data saved to: {expt.fname}")

except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
