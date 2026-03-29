"""Run T1PredistFastFlux experiment from CLI."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from qick import QickConfig
from slab_qick_calib.exp_handling.instrumentmanager import InstrumentManager
from slab_qick_calib.helpers import config, rfboard
import slab_qick_calib.experiments as meas
from slab_qick_calib.analysis.fitting import expfunc

# --- Connect to hardware ---
cfg_path = 'configs/sample_50_rfboard.yml'
auto_cfg = config.load(cfg_path)

im = InstrumentManager(ns_address=auto_cfg['aliases']['ip'], port=8888)
soc = im[auto_cfg['aliases']['soc']]
soccfg = QickConfig(soc.get_cfg())

cfg_dict = {'soc': soccfg, 'expt_path': r'C:\_Data\sample_50_new',
            'cfg_file': cfg_path, 'im': im}

# --- Activate qubit 2 ---
qi = 2
auto_cfg = rfboard.activate_qubit_rf(qi, soc, auto_cfg, cfg_file=cfg_path)
soc.rfb_set_bias(auto_cfg['hw']['soc']['dacs']['flux']['dc_ch'][qi], 0.0)

OUT_DIR = r'C:\_Data\sample_50_new'
SCANS_DIR = os.path.join(OUT_DIR, 't1_predist_scans')
os.makedirs(SCANS_DIR, exist_ok=True)

# --- Run the full gain sweep ---
print("=" * 60)
print("Running T1PredistFastFlux sweep...")
print("=" * 60)
expt = meas.T1PredistFastFlux(cfg_dict, qi=2, go=False, params={
    "flux_negative_reset": True,
    "flux_slow_only": False,
    "expts_gain": 200,
    "expts": 50,
    "direction": "neg",
})
data = expt.acquire(progress=True, display=False)

# --- Save individual T1 scan plots with fits ---
gain_pts = data["gain_pts"]
t1_list = data["t1_list"]
freq_pts = data.get("freq_pts", None)

from slab_qick_calib.analysis.fitting import fitexp

for i in range(len(gain_pts)):
    avgi = np.array(data["avgi"][i], dtype=float)
    xpts = np.array(data["xpts"][i], dtype=float)

    fig, ax = plt.subplots()
    ax.plot(xpts, avgi, "o-", rasterized=True)

    # Use stored fit params (offset, amp, t1); refit if not available
    offset_val = float(data["offset_list"][i])
    amp_val = float(data["amp_list"][i])
    t1_val = float(data["t1_list"][i])
    fit_params = np.array([offset_val, amp_val, t1_val])

    if not np.all(np.isfinite(fit_params)):
        try:
            fp = fitexp(xpts, avgi)
            if fp is not None and len(fp) >= 3:
                fit_params = fp
                t1_val = fp[2]
        except Exception:
            pass

    if np.all(np.isfinite(fit_params)):
        caption = f"$T_1$ fit: {t1_val:.3f} $\\mu$s"
        ax.plot(xpts, expfunc(xpts, *fit_params), label=caption, rasterized=True)
        ax.legend()

    ax.set_xlabel("Wait Time ($\\mu$s)")
    ax.set_ylabel("I (a.u.)")
    fig.tight_layout()
    fig.savefig(os.path.join(SCANS_DIR, f"t1_scan_{i:02d}.png"), dpi=120)
    plt.close(fig)

print(f"Individual scan plots saved to {SCANS_DIR}")

# --- Summary plots (with IQR outlier rejection) ---
t1_arr = np.array(t1_list, dtype=float)
mask = np.isfinite(t1_arr)

def plot_with_outlier_rejection(ax, x, y, mask):
    if mask.sum() > 2:
        q1, q3 = np.percentile(y[mask], [25, 75])
        iqr = q3 - q1
        inlier = mask & (y >= q1 - 1.5 * iqr) & (y <= q3 + 1.5 * iqr)
        ax.plot(x[inlier], y[inlier], ".-")
        ax.plot(x[~inlier & mask], y[~inlier & mask], "rx", ms=6)
        inlier_vals = y[inlier]
        if len(inlier_vals) > 0:
            ymin, ymax = inlier_vals.min(), inlier_vals.max()
            margin = (ymax - ymin) * 0.1 if ymax > ymin else abs(ymax) * 0.1 or 0.1
            ax.set_ylim(ymin - margin, ymax + margin)
    else:
        ax.plot(x[mask], y[mask], ".-")

fig, ax = plt.subplots()
plot_with_outlier_rejection(ax, gain_pts, t1_arr, mask)
ax.set_xlabel("Flux Gain")
ax.set_ylabel("$T_1$ ($\\mu$s)")
fig.savefig(os.path.join(OUT_DIR, 't1_predist_vs_gain.png'), dpi=150)
plt.close(fig)

if freq_pts is not None:
    fig, ax = plt.subplots()
    plot_with_outlier_rejection(ax, freq_pts, t1_arr, mask)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("$T_1$ ($\\mu$s)")
    fig.savefig(os.path.join(OUT_DIR, 't1_predist_vs_freq.png'), dpi=150)
    plt.close(fig)

# Waterfall
if "avgi" in data and len(data["avgi"]) > 0:
    curves = data["avgi"]
    xpts_list = data.get("xpts", None)
    n = len(curves)
    spacing = 0.4
    fig, ax = plt.subplots(figsize=(8, max(4, n * spacing * 0.25)))
    for i, curve in enumerate(curves):
        y = np.array(curve, dtype=float)
        x = np.array(xpts_list[i], dtype=float) if xpts_list is not None else np.arange(len(y))
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        if ymax > ymin:
            y_norm = (y - ymin) / (ymax - ymin) * spacing
        else:
            y_norm = np.zeros_like(y)
        ax.plot(x, y_norm + i * spacing, "-", lw=0.8)
    ax.set_xlabel("Wait Time ($\\mu$s)")
    use_freq = freq_pts is not None
    label_pts = freq_pts if use_freq else gain_pts
    ax.set_ylabel("Frequency (MHz)" if use_freq else "Flux Gain")
    max_ticks = 20
    step = max(1, n // max_ticks)
    tick_idx = np.arange(0, n, step)
    ax.set_yticks(tick_idx * spacing)
    ax.set_yticklabels([f"{label_pts[j]:.1f}" for j in tick_idx])
    ax.set_ylim(-0.2, n * spacing)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 't1_predist_waterfall.png'), dpi=150)
    plt.close(fig)

# Fit parameter plots
if freq_pts is not None:
    x_pts = freq_pts
    x_label = "Frequency (MHz)"
else:
    x_pts = gain_pts
    x_label = "Flux Gain"

fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
panels = [
    (axes[0, 0], data.get("amp_list", []), "I Amplitude"),
    (axes[0, 1], data.get("offset_list", []), "I Offset"),
    (axes[1, 0], data.get("q_amp_list", []), "Q Amplitude"),
    (axes[1, 1], data.get("q_offset_list", []), "Q Offset"),
    (axes[2, 0], data.get("s_amp_list", []), "Rescaled Amplitude"),
    (axes[2, 1], data.get("s_offset_list", []), "Rescaled Offset"),
]
for ax, vals, label in panels:
    vals = np.array(vals, dtype=float)
    if vals.size == 0:
        ax.set_ylabel(label)
        continue
    pmask = np.isfinite(vals)
    if pmask.sum() > 2:
        q1, q3 = np.percentile(vals[pmask], [25, 75])
        iqr = q3 - q1
        inlier = pmask & (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
        ax.plot(x_pts[inlier], vals[inlier], ".-")
        ax.plot(x_pts[~inlier & pmask], vals[~inlier & pmask], "rx", ms=6)
        inlier_vals = vals[inlier]
        if len(inlier_vals) > 0:
            ymin, ymax = inlier_vals.min(), inlier_vals.max()
            margin = (ymax - ymin) * 0.1 if ymax > ymin else abs(ymax) * 0.1 or 0.1
            ax.set_ylim(ymin - margin, ymax + margin)
    else:
        ax.plot(x_pts, vals, ".-")
    ax.set_ylabel(label)
axes[2, 0].set_xlabel(x_label)
axes[2, 1].set_xlabel(x_label)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 't1_predist_fit_params.png'), dpi=150)
plt.close(fig)

print("=" * 60)
print("Done! Plots saved to:")
print(f"  {OUT_DIR}\\t1_predist_vs_gain.png")
print(f"  {OUT_DIR}\\t1_predist_vs_freq.png")
print(f"  {OUT_DIR}\\t1_predist_waterfall.png")
print(f"  {OUT_DIR}\\t1_predist_fit_params.png")
print(f"  {SCANS_DIR}\\t1_scan_00.png .. t1_scan_{len(gain_pts)-1:02d}.png")
print("=" * 60)
