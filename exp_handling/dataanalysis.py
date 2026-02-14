# -*- coding: utf-8 -*-
"""
Data analysis utilities for quantum control experiments.

This module provides functions for data analysis, plotting, file handling,
and signal processing commonly used in quantum control experiments.

Original code by David Schuster
Taken from the slab repository

Created on Sat Jul 30 14:04:03 2011
@author: Dave
"""

import glob
from pathlib import Path
import time
import dateutil
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import (
    arange,
    array,
    argsort,
    cos,
    convolve,
    genfromtxt,
    loadtxt,
    ones,
    pi,
    sin,
    sqrt,
    transpose,
    zeros,
    arctan2,
)
from scipy.signal import decimate


from .datamanagement import get_script

###################################################

#### General utility functions for unit conversions ####


def dBm_to_W(dBm):
    """
    Convert power from dBm to Watts.
    
    Args:
        dBm (float): Power in dBm
        
    Returns:
        float: Power in Watts
    """
    return 10.0 ** (dBm / 10.0) * 1e-3


# Create alias for backward compatibility
dBmtoW = dBm_to_W


def W_to_dBm(W):
    """
    Convert power from Watts to dBm.
    
    Args:
        W (float): Power in Watts
        
    Returns:
        float: Power in dBm
    """
    return np.log(W * 1e3) / np.log(10) * 10.0


def dBm_to_V(dBm):
    """
    Convert power from dBm to RMS voltage (assuming 50 ohm impedance).
    
    Args:
        dBm (float): Power in dBm
        
    Returns:
        float: RMS voltage in Volts
    """
    return sqrt(dBm_to_W(dBm) * 50) * sqrt(2)


# Create alias for backward compatibility
dBmtoV = dBm_to_V

######################## Plotting helpers

figureStyles = [
    {
        "name": "default",
        "width": 8,
        "height": 8.0 / 1.61,
        "title_size": 12,
        "label_size": 8,
        "tick_size": 7,
        "wspace": 0.5,
        "tight": True,
        "dpi": 72,
        "transparent": False,
    },
    {
        "name": "prl1",
        "width": 3.4,
        "height": 3.4,
        "title_size": 12,
        "label_size": 8,
        "tick_size": 7,
        "wspace": 0.5,
        "tight": True,
        "dpi": 300,
    },
    {
        "name": "prl2",
        "width": 7,
        "height": 3.4,
        "title_size": 12,
        "label_size": 8,
        "tick_size": 7,
        "wspace": 0.5,
        "tight": True,
        "dpi": 300,
    },
    {
        "name": "prl21",
        "width": 1.2,
        "height": 3.4,
        "title_size": 12,
        "label_size": 8,
        "tick_size": 7,
        "wspace": 0.5,
        "tight": True,
        "dpi": 300,
    },
    {
        "name": "small",
        "width": 5,
        "height": 5.0 / 1.61,
        "title_size": 12,
        "label_size": 8,
        "tick_size": 7,
        "wspace": 0.5,
        "tight": True,
    },
    {
        "name": "medium",
        "width": 8,
        "height": 8.0 / 1.61,
        "title_size": 12,
        "label_size": 8,
        "tick_size": 7,
        "wspace": 0.5,
        "tight": True,
    },
    {
        "name": "big",
        "width": 16,
        "height": 16.0 / 1.61,
        "title_size": 22,
        "label_size": 16,
        "tick_size": 14,
        "wspace": 0.5,
        "tight": True,
    },
    {
        "name": "ppt",
        "width": 16,
        "height": 16.0 / 1.61,
        "title_size": 48,
        "label_size": 36,
        "tick_size": 32,
        "wspace": 0.5,
        "tight": True,
    },
    {
        "name": "square",
        "width": 8,
        "height": 8,
        "title_size": 26,
        "label_size": 18,
        "tick_size": 16,
        "wspace": 0.2,
        "tight": True,
    },
    {
        "name": "wide",
        "width": 16,
        "height": 6,
        "title_size": 26,
        "label_size": 18,
        "tick_size": 16,
        "wspace": 0.2,
        "tight": True,
    },
    {
        "name": "r4",
        "width": 32,
        "height": 6,
        "title_size": 26,
        "label_size": 18,
        "tick_size": 16,
        "wspace": 0.2,
        "tight": True,
    },
]
figureStyleDict = {}
for style in figureStyles:
    figureStyleDict[style["name"]] = style


def get_figure_style(style=None):
    """Returns profile of figure style with name _style_, if style=None returns the whole style dictionary"""
    if style is None:
        return figureStyleDict
    else:
        return figureStyleDict[style]


def figure_style(style=None, **kwargs):
    """
    Style figure (change dimensions, fontsizes etc, according to _style_ profile:
    Valid profiles can be located in figureStyleDict and include: 'default', 'big', 'small', 'prl1','prl2', 'ppt'
    """
    matplotlib.rc("font", **{"family": "sans-serif"})
    s = figureStyleDict["default"]
    if style is not None:
        s.update(figureStyleDict[style])
    s.update(kwargs)
    f = plt.gcf()
    ax = plt.gca()
    plt.tight_layout()
    f.set_figwidth(s["width"])
    f.set_figheight(s["height"])
    f.subplots_adjust(wspace=s["wspace"])
    for ax in f.axes:
        ax.title.set_fontsize(s["title_size"])
        for item in [ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(s["label_size"])
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(s["tick_size"])


def save_styled_fig(fname, style=None, **kwargs):
    s = figureStyleDict["default"]
    if style is not None:
        s.update(figureStyleDict[style])
    s.update(kwargs)
    figure_style(style, **kwargs)
    plt.savefig(fname, dpi=s["dpi"], transparent=s["transparent"], bbox_inches="tight")


######################## File handling


def save_script(expt_path, prefix):
    """This function saves the file of the current running script to a file in the standard file naming convention:
    expt_path\date_prefix_number"""
    tag = date_tag(None)
    ii = next_path_index(expt_path, prefix)
    fname = str(Path(expt_path) / ("%s_%s_%03d.py" % (tag, prefix, ii)))
    fw = open(fname, "w")
    fw.write(get_script())
    fw.close()
    return fname


def find_closest_index(a, v):
    return argsort(abs(a - v))[0]


def load_ppms_file(filename):
    return transpose(
        array(
            genfromtxt(
                filename,
                delimiter=",",
                usecols=(3, 8, 10),
                skip_header=33,
                missing_values="",
                filling_values=1e-8,
            )
        )
    )


########### Network Analyzer Analysis Functions ####
def load_nwa_file(filename):
    """return three arrays: frequency magnitude and phase"""
    return transpose(loadtxt(filename, skiprows=3, delimiter=","))


def load_nwa_dir(datapath):
    fnames = sorted(Path(datapath).glob("*.CSV"))
    prefixes = [f.name for f in fnames]
    fnames = [str(f) for f in fnames]
    data = [load_nwa_file(fname) for fname in fnames]
    return prefixes, data


def load_nwa_list(datapath, header_list):
    fnames = sorted(Path(datapath).glob("*.CSV"))
    prefixes = []
    fnames2 = []
    for fname in fnames:
        prefix = fname.name
        number = prefix.split("_")[0]
        if int(number) in header_list:
            prefixes.append(prefix)
            fnames2.append(str(fname))
    #            print 'filename', fname
    #    print 'filenames 2',fnames2
    data = [load_nwa_file(fname) for fname in fnames2]
    return prefixes, data


############## Experiment helpers #####################


def next_file_index(datapath, prefix, suffix=""):
    """Searches directories for files of the form *_prefix* and returns next number
    in the series"""

    dirlist = sorted(Path(datapath).glob(prefix + "_*" + suffix))
    if len(dirlist) > 0:
        try:
            ii = int(dirlist[-1].name.split("_")[-1][0:-3]) + 1
        except:
            ii = 0
    else:
        ii = 0
    return ii


def current_file_index(datapath, prefix="", suffix=".h5"):
    """Searches directories for files of the form *_prefix* and returns current number
    in the series"""

    dirlist = sorted(Path(datapath).glob(prefix + "_*" + suffix))
    try:
        ii = int(dirlist[-1].name.split("_")[0])
    except:
        ii = 0
    return ii


def date_tag(date_str=None):
    if date_str is None:
        lt = time.localtime()
        return "%02d%02d%02d" % (lt.tm_year % 100, lt.tm_mon, lt.tm_mday)
    else:
        lt = dateutil.parser.parse(date_str)
        return "%02d%02d%02d" % (lt.year % 100, lt.month, lt.day)


def next_path_index(expt_path, prefix="", suffix=".h5"):
    dirlist = sorted(Path(expt_path).glob("*" + prefix + "*[0-9][0-9][0-9]"))
    if not dirlist:
        return 0
    return int(dirlist[-1].name.split("_")[-1]) + 1


def current_path_index(expt_path, prefix=""):
    dirlist = sorted(Path(expt_path).glob("*" + prefix + "*[0-9][0-9][0-9]"))
    if not dirlist:
        return 0
    return int(dirlist[-1].name.split("_")[-1])


def get_next_filename(datapath, prefix, suffix=""):
    ii = next_file_index(datapath, prefix, suffix)
    return prefix + "_%05d" % (ii) + suffix


def get_current_filename(datapath, prefix, suffix=""):
    ii = current_file_index(datapath, prefix)
    return prefix + "_%05d" % (ii) + suffix


def make_datapath(expt_path, prefix, date_str=None, new_file=False):
    """Automatically makes a new folder in the experiment folder with new index"""
    tag = date_tag(date_str)
    if new_file:
        ii = next_path_index(expt_path, prefix)
    else:
        ii = current_path_index(expt_path, prefix)
    datapath = Path(expt_path) / ("%s_%s_%03d" % (tag, prefix, ii))
    datapath.mkdir()
    return str(datapath)


def current_datapath(expt_path, prefix, date_str=None):
    tag = date_tag(date_str)
    ii = next_path_index(expt_path, prefix) - 1
    return str(Path(expt_path) / ("%s_%s_%03d" % (tag, prefix, ii)))


def tic():
    global last_tic
    last_tic = time.time()


def toc(log=False):
    global last_tic
    t = time.time()
    if log:
        print("Tic-Toc: %.0f ms" % ((t - last_tic) * 1000.0))
    return t - last_tic


def digital_homodyne(
    time_pts, ch1_pts, ch2_pts=None, IFfreq=1, dfactor=1.0, AmpPhase=False
):
    """digital_homodyne computes I/Q or Amp/Phase as a function of time of a particular frequency component.
    @param time_pts: time of each sample
    @param ch1_pts: Scope channel 1 points
    @param ch2_pts: Scope channel 2 points (optional)
    @param IFfreq: Frequency to extract info for
    @param dfactor: decimation factor number of periods to include per point
    @param AmpPhase: returns I/Q if False (default) or Amp/Phase if True
    """

    df = int(dfactor / (time_pts[1] - time_pts[0]) / IFfreq)
    cospts = (2.0 / sqrt(df)) * cos(2.0 * pi * IFfreq * time_pts)
    sinpts = (2.0 / sqrt(df)) * sin(2.0 * pi * IFfreq * time_pts)
    in2 = ones(df, dtype=float) / sqrt(df)

    I1pts = decimate(convolve(ch1_pts * cospts, in2, mode="valid"), df, n=2)
    Q1pts = decimate(convolve(ch1_pts * sinpts, in2, mode="valid"), df, n=2)
    dtpts = arange(len(I1pts)) * ((time_pts[1] - time_pts[0]) * df) + time_pts[0]

    if ch2_pts is not None:
        I2pts = decimate(convolve(ch2_pts * cospts, in2, mode="valid"), df, n=2)
        Q2pts = decimate(convolve(ch2_pts * sinpts, in2, mode="valid"), df, n=2)

    if AmpPhase:
        amp1pts = sqrt(I1pts**2 + Q1pts**2)
        phi1pts = arctan2(I1pts, Q1pts) * 180.0 / pi
        if ch2_pts is not None:
            amp2pts = sqrt(I2pts**2 + Q2pts**2)
            phi2pts = arctan2(I2pts, Q2pts) * 180.0 / pi
            return dtpts, amp1pts, phi1pts, amp2pts, phi2pts
        else:
            return dtpts, amp1pts, phi1pts
    else:
        if ch2_pts is not None:
            return dtpts, I1pts, Q1pts, I2pts, Q2pts
        else:
            return dtpts, I1pts, Q1pts


def heterodyne(
    time_pts, ch1_pts, ch2_pts=None, IFfreq=1, AmpPhase=True, anti_alias=True
):
    """heterodyne computes I/Q or Amp/Phase as a function of time of a particular frequency component.
    @param time_pts: time of each sample
    @param ch1_pts: Scope channel 1 points
    @param ch2_pts: Scope channel 2 points (optional)
    @param IFfreq: Frequency to extract info for
    @param AmpPhase: returns I/Q if False (default) or Amp/Phase if True
    """
    if ch2_pts is None:
        ch2_pts = zeros(len(time_pts))

    # anti alias makes sure that there are an integer number of IFFreq cycles in the array of points
    if anti_alias:
        stop_pt = int(
            np.floor(len(time_pts) * (IFfreq * (time_pts[1] - time_pts[0])))
            / (IFfreq * (time_pts[1] - time_pts[0]))
        )
        print(stop_pt)
    else:
        stop_pt = len(time_pts)

    cospts = cos(2.0 * pi * IFfreq * time_pts[:stop_pt])
    sinpts = sin(2.0 * pi * IFfreq * time_pts[:stop_pt])
    A1 = (2.0 * np.sum(cospts * ch1_pts[:stop_pt])) / stop_pt
    B1 = (2.0 * np.sum(sinpts * ch1_pts[:stop_pt])) / stop_pt
    A2 = (2.0 * np.sum(cospts * ch2_pts[:stop_pt])) / stop_pt
    B2 = (2.0 * np.sum(sinpts * ch2_pts[:stop_pt])) / stop_pt
    if AmpPhase:
        return (
            sqrt(A1**2 + B1**2),
            arctan2(A1, B1) * 180.0 / pi,
            sqrt(A2**2 + B2**2),
            arctan2(A2, B2) * 180.0 / pi,
        )
    else:
        return A1, B1, A2, B2
