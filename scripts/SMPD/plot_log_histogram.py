"""Replot the log-histogram panel from a saved single-shot or FWM histogram file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from slab_qick_calib.calib.readout_helpers import COLORS
from slab_qick_calib.exp_handling.datamanagement import SlabFile
from slab_qick_calib.experiments.smpd.fwm_shots import SMPDHistogramExperiment

from common import run_and_device_from_config

# Which run folder to read from, named the same way the runner scripts name it.
CONFIG = "2026_10_05_bfg34_smpd_v2_waymar_rfboard.yml"

# Drawing style per scan. g/e/f use the shared single-shot colours so the panel
# matches the 2x2 figure; p/n keep the FWM overlay styles.
STYLES = {
    "g": {"color": COLORS["ground"], "ls": "-"},
    "e": {"color": COLORS["excited"], "ls": "-"},
    # "f": {"color": COLORS["f_state"], "ls": "-"},
    "p": {"color": "green", "ls": "--"},
    "n": {"color": "0.35", "ls": "-."},
}
PLAIN_LABELS = {"g": "|g>", "e": "|e>", "f": "|f>"}


def find_data_file(run_dir, number, index):
    """Locate <run>/data/#<number>_*/..._<index>.h5."""
    matches = sorted((run_dir / "data").glob(f"#{number}_*"))
    if not matches:
        raise SystemExit(f"No folder #{number}_* under {run_dir / 'data'}")
    if len(matches) > 1:
        raise SystemExit(
            f"Several folders match #{number}_*: {[m.name for m in matches]}"
        )
    folder = matches[0]
    files = sorted(folder.glob(f"*_{int(index):05d}.h5"))
    if not files:
        raise SystemExit(f"No file ending _{int(index):05d}.h5 in {folder}")
    return files[0]


def histograms(data, bins=100):
    """
    Histogram every scan present, from the raw shots, on shared bins.

    Deliberately not the saved `hist*` arrays: `single_shot` with check_f
    feeds the f-state shots into its excited slot, so the stored `histe` is
    actually the |f> distribution and no `histf` is written at all. Going back
    to the I/Q arrays avoids inheriting that mislabelling.

    The shots are stored unrotated, so the stored rotation angle is reapplied
    here to land on the axis the original figure plotted. FWM histogram files
    are taken with rotate=False, where that angle is 0 and this is a no-op.
    """
    theta = np.deg2rad(float(data.get("angle", 0.0)))
    rotated = {}
    for scan in STYLES:
        if f"I{scan}" not in data:
            continue
        rotated[scan] = (
            data[f"I{scan}"] * np.cos(theta) - data[f"Q{scan}"] * np.sin(theta)
        )
    if not rotated:
        raise SystemExit("No I/Q shot arrays in this file")

    edges = np.histogram_bin_edges(
        np.concatenate(list(rotated.values())), bins=bins
    )
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, {
        scan: np.histogram(values, bins=edges, density=True)[0]
        for scan, values in rotated.items()
    }, np.rad2deg(theta)


def describe(expt):
    """One line of the settings that matter, for whichever experiment it was."""
    if "frequency_p" in expt:
        return (
            f"pump {expt['frequency_p']:.3f} MHz gain {expt['gain_p']:.4f}, "
            f"buffer {expt['frequency_b']:.3f} MHz gain {expt['gain_b']:.4f}"
        )
    return (
        f"readout {expt['frequency']:.3f} MHz gain {expt['gain']}, "
        f"{expt['readout_length']:.3f} us, {expt['shots']} shots"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", help="experiment number, the #N in the folder name")
    parser.add_argument("index", nargs="?", default="0",
                        help="data file index, the _000NN suffix (default 0)")
    default_run, default_device = run_and_device_from_config(CONFIG)
    parser.add_argument("--run-number", default=default_run)
    parser.add_argument("--device-name", default=default_device)
    parser.add_argument("--file", help="path to a .h5, instead of number/index")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--out", help="output PNG path")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    run_dir = SCRIPT_DIR / "Data" / f"run{args.run_number}_{args.device_name}"
    path = Path(args.file) if args.file else find_data_file(
        run_dir, args.number, args.index
    )
    print(f"Reading {path}")

    with SlabFile(str(path), "r") as h5:
        data = {key: h5[key][()] for key in h5.keys()}
        cfg = h5.load_config()
    expt = cfg["expt"]

    centers, hists, angle = histograms(data, bins=args.bins)
    # FWM files carry the pump-only / no-drives scans and name them by which
    # drives were on; a plain single-shot file is just the bare states.
    is_fwm = "p" in hists or "n" in hists
    labels = SMPDHistogramExperiment._SCAN_LABELS if is_fwm else PLAIN_LABELS
    print(f"  scans: {sorted(hists)}   rotation applied: {angle:.3f} deg")

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for scan, hist in hists.items():
        ax.plot(centers, hist, lw=1.2, label=labels.get(scan, scan),
                    **STYLES[scan])

    thresholds = np.atleast_1d(data.get("thresholds", []))
    names = ("g-e", "g-f", "e-f")
    for value, name in zip(thresholds[:1], names):
        ax.axvline(float(value), color=COLORS["threshold"], ls="--", lw=1,
                   label=f"{name} threshold {float(value):.2f}")

    fids = np.atleast_1d(data.get("fids", []))
    if len(fids):
        ax.plot([], [], ls="none", marker="",
                label=" ".join(f"{n} {v:.3f}" for n, v in zip(names, fids)))

    ax.set_xlabel("I (ADC levels)",fontsize=14)
    ax.set_ylabel("Probability",fontsize=14)
    ax.tick_params(axis="both", labelsize=14)      # both axes
    # ax.tick_params(axis="y", labelsize=14)         # just y

    # ax.legend(fontsize="small")
    # fig.suptitle(f"{path.parent.name} / {path.stem}\n{describe(expt)}")
    fig.suptitle('four-wave mixing: single photon detection',fontsize= 16)
    fig.tight_layout()

    out = Path(args.out) if args.out else (
        run_dir / "metadata_and_images" / path.parent.name
        / f"{path.stem}_log_histogram.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
