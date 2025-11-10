import numpy as np
import matplotlib.units as munits
import matplotlib.dates as mdates
from matplotlib.dates import AutoDateFormatter, AutoDateLocator
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import os
from pathlib import Path


def get_parameter_config():
    """
    Get the parameter configuration dictionary that maps parameter keys to their properties.
    
    Returns:
    --------
    dict
        Dictionary mapping parameter keys to configuration with 'key', 'r2_scan', and 'label' fields
    """
    return {
        "t1": {
            "key": "t1",
            "r2_scan": "t1_r2",
            "label": "$T_1$ ($\mu$s)"
        },
        "t2r": {
            "key": "t2r",
            "r2_scan": "t2r_r2",
            "label": "$T_{2,R}$ ($\mu$s)"
        },
        "t2": {
            "key": "t2",
            "r2_scan": "t2_r2",
            "label": "$T_{2,E}$ ($\mu$s)"
        },
        "f_ge": {
            "key": "f_ge",
            "r2_scan": "t2_r2",
            "label": "$f-\overline{f}$ (kHz), Qubit "
        },
        "fidelity": {
            "key": "fidelity",
            "r2_scan": None,
            "label": "Log(1 -Readout Fidelity)"
        },
        "kappa": {
            "key": "kappa",
            "r2_scan": "rspec_r2",
            "label": "Resonator $\kappa$ (MHz)"
        },
        "frequency": {
            "key": "frequency",
            "r2_scan": "rspec_r2",
            "label": "$f-\overline{f}$ (kHz), Resonator "
        },
        "pi_length": {
            "key": "pi_length",
            "r2_scan": "amp_rabi_r2",
            "label": "$\pi_{gain}/\overline{\pi_{gain}}$"
        },
        "tphi": {
            "key": "tphi",
            "r2_scan": "t2_r2",
            "label": "$T_{\phi}$ ($\mu$s)"
        },
        "tsphi": {
            "key": "tsphi",
            "r2_scan": "t2_r2",
            "label": "$T_{\phi,R}$ ($\mu$s)"
        },
        "t2_r2": {
            "key": "t2_r2",
            "r2_scan": None,
            "label": "$T_2 \, R^2$"
        },
        "t2et1": {
            "key": "t2et1",
            "r2_scan": "t2_r2",
            "label": "$T_{2,E}/2 T_1$"
        },
        "t1_off": {
            "key": "t1_off",
            "r2_scan": "t1_r2",
            "label": "$T_1$ offset"
        },
        "t1_amp": {
            "key": "t1_amp",
            "r2_scan": "t1_r2",
            "label": "$T_1$ amplitude"
        },
        "t2r_off": {
            "key": "t2r_off",
            "r2_scan": "t2r_r2",
            "label": "$T_{2,R}$ offset"
        },
        "t2r_amp": {
            "key": "t2r_amp",
            "r2_scan": "t2r_r2",
            "label": "$T_{2,R}$ amplitude"
        },
        "q": {
            "key": "q",
            "r2_scan": "t1_r2",
            "label": "$Q_1$"
        },
        "phase": {
            "key": "phase",
            "r2_scan": None,
            "label": "Phase"
        }, 
        "Gamma1": {
            "key": "Gamma1",
            "r2_scan": "t1_r2",
            "label": "$\Gamma_1$ (ms$^{-1}$)"
        },
        "Gamma2r": {
            "key": "Gamma2r",
            "r2_scan": "t2r_r2",
            "label": "$\Gamma_{2,R}$ (ms$^{-1}$)"
        },
        "Gamma2": {
            "key": "Gamma2",
            "r2_scan": "t2_r2",
            "label": "$\Gamma_{2,E}$ (ms$^{-1}$)"
        },
        "Gamma2e": {
            "key": "Gamma2e",
            "r2_scan": "t2_r2",
            "label": "$\Gamma_{2,E}$ (ms$^{-1}$)"
        }, 
        "Gammaphi": {
            "key": "Gammaphi",
            "r2_scan": "t2_r2",
            "label": "$\Gamma_{\phi}$ (ms$^{-1}$)"
        },
        "Gammasphi": {
            "key": "Gammasphi",
            "r2_scan": "t2_r2",
            "label": "$\Gamma_{\phi,R}$ (ms$^{-1}$)"
        },
    }

def process_data(tt, qubit_list=None):

    if qubit_list is None:
        qubit_list = [i for i in range(len(tt))]

    b = {}

    for j, qi in enumerate(qubit_list):
        for k in list(tt[j].keys()):
            # try:
            tt[j][k] = np.array(tt[j][k])
            # except Exception:
            #     pass

        tt[j]["q"] = tt[j]["t1"] * tt[j]["f_ge"] * 2 * np.pi
        if "t2r" in tt[j].keys():
            tt[j]["t2rt1"] = tt[j]["t2r"] / 2 / tt[j]["t1"]
            tsphi = 1 / (1 / tt[j]["t2r"] - 1 / tt[j]["t1"] / 2)
            tsphi[(tsphi < 0) | (tsphi > 1000)] = np.nan
            tt[j]["tsphi"] = tsphi
            tt[j]["q2"] = tt[j]["tsphi"] * tt[j]["f_ge"] * 2 * np.pi
        if "t2" in tt[j].keys() or "t2e" in tt[j].keys():
            tt[j]["t2et1"] = tt[j]["t2"] / 2 / tt[j]["t1"]
            tphi = 1 / (1 / tt[j]["t2"] - 1 / tt[j]["t1"] / 2)
            tphi[(tphi < 0) | (tphi > 1000)] = np.nan
            tt[j]["tphi"] = tphi
            tt[j]["q2"] = tt[j]["tphi"] * tt[j]["f_ge"] * 2 * np.pi

    for key in tt[0].keys():
        for j, qi in enumerate(qubit_list):
            if j == 0:
                b[key + "_mn"] = np.zeros(len(qubit_list))
                b[key + "_std"] = np.zeros(len(qubit_list))
            b[key + "_mn"][j] = np.nanmean(tt[j][key])
            b[key + "_std"][j] = np.nanstd(tt[j][key])

    return tt, b


def plot_all(tt, qubit_list=None, fname='def', param_keys=None, use_mean=False, nbins=40, plot_time=False):

    if qubit_list is None:
        qubit_list = [i for i in range(len(tt))]
    if param_keys is None:
        param_keys = ["t1", "t2r", "t2", "f_ge", "fidelity", "kappa", "frequency", "pi_length", "tphi", "tsphi", "t2_r2"]
    
    parameter_config = get_parameter_config()
    
    nrows, ncols, sz = calculate_subplot_layout(len(param_keys))
    mpl.rcParams["lines.markersize"] = 1
    mpl.rcParams.update({"font.size": 11})

    for j, qi in enumerate(qubit_list):
        fig2, ax2 = plt.subplots(nrows, ncols, figsize=sz)
        fig2.suptitle(f"Tracking Histograms Qubit {qi}")
        ax2 = ax2.flatten()
        xval = tt[j]["time"]
        xval = pd.to_datetime(tt[j]["time"], unit="s")
        # xval = pd.to_datetime(tt[j]['time'])
        # xval = xval.dt.strftime('%d %H:%M:%s')  # Format time as Day HH:MM
        #fig3, ax3 = plt.subplots(1, 1)
        xval = (xval - xval[0]) / np.timedelta64(1, "h")
        if plot_time:
            fig, ax = plt.subplots(nrows, ncols, figsize=sz)
            ax = ax.flatten()
            fig.suptitle(f"Qubit {qi}")
        i = 0

        for param_key in param_keys:
            if param_key not in parameter_config:
                print(f"Warning: Parameter '{param_key}' not found in configuration")
                continue
                
            config = parameter_config[param_key]
            key = config["key"]
            r2_key = config["r2_scan"]
            
            # Skip if key doesn't exist in data
            if key not in tt[j]:
                continue
                
            # Remove outliers, those with values far away from mean or small r^2
            if r2_key is not None and r2_key in tt[j]:
                # Ignore data at least 0.08 lower r2 than mean and 4 std away from mean
                inds = (tt[j][r2_key] > np.nanmean(np.array(tt[j][r2_key])) - 0.08) 
                inds2 = (np.abs(tt[j][key] - np.nanmean(tt[j][key])) < np.nanstd(tt[j][key]) * 4)
                inds_true = np.logical_and(inds, inds2)

                inds = np.where(inds_true)[0]
                print(
                    f"Number of bad inds for {key} is {np.sum(~np.array(inds_true, dtype=bool))}"
                )
            else:
                inds = np.arange(len(tt[j][key]))

            # Set up axis labels and data style
            if key in ["f_ge", "frequency"]:
                y = 1e3 * (tt[j][key][inds] - np.nanmean(tt[j][key][inds]))
            elif key in ["pi_length"] or use_mean:
                y = tt[j][key][inds] / np.nanmean(tt[j][key][inds])
            elif key in ["t2r_amp", "t1_amp"]:
                y = np.abs(tt[j][key][inds])
            elif key in ["fidelity"]:
                y = np.log10(1 - tt[j][key][inds])
            else:
                y = tt[j][key][inds]

            # if key in ['t1','t2r','t2e','tphi','tsphi']:
            # y =np.log(y)
            # if key=='t1':
            #     ax[2].plot(xval[inds], 2*tt[j]['t1'][inds], 'r.', label='$2 T_1$')
            #     ax[2].legend()
            if plot_time:
                ax[i].xaxis.set_major_locator(
                    plt.MaxNLocator(5)
                )  # Limit the number of ticks to avoid overcrowding
                # ax[k].set_xticklabels(xval, rotation=45, ha='right')  # Rotate labels for better readability
                # Plot data, set titles.
                ax[i].plot(
                    xval[inds], np.array(y), "o", linewidth=1
                )  # , label=leg_list[k])
                # ax[k].legend()

                ax[i].set_ylabel(config["label"])
                ax[i].set_xlabel("Time (hr)")

            # Histogram plot
            ax2[i].hist(np.array(y), bins=nbins, label=int(qi), density=True)
            ax2[i].set_xlabel(config["label"])

            # if key == "t1":
            #     ax3.plot(
            #         xval[inds], tt[j]["t1"][inds], ".", markersize=3, label="$T_1$"
            #     )
            #     ax3.xaxis.set_major_locator(
            #         plt.MaxNLocator(5)
            #     )  # Limit the number of ticks to avoid overcrowding

            # if key == "f_ge":
            #     ax3_twin = ax3.twinx()
            #     ax3_twin.plot(
            #         xval[inds],
            #         tt[j]["f_ge"][inds],
            #         ".",
            #         label="f_Q$",
            #         color="tab:orange",
            #     )
            i += 1

        if plot_time:
            fig.tight_layout()
            #fig.savefig(os.path.join("images", f"{qi}_tracking_{fname}.png"))

        fig2.tight_layout()
        #fig2.savefig(os.path.join("images", f"{qi}_histogram_{fname}.png"))





def calculate_subplot_layout(n_plots, max_cols=5, subplot_width=3.5, subplot_height=3):
    """
    Calculate optimal subplot layout and figure size based on number of plots.

    Parameters:
    -----------
    n_plots : int
        Number of plots to display
    max_cols : int, optional
        Maximum number of columns (default: 5)
    subplot_width : float, optional
        Width of each subplot in inches (default: 3)
    subplot_height : float, optional
        Height of each subplot in inches (default: 2.5)

    Returns:
    --------
    tuple
        (n_rows, n_cols, figsize) where figsize is (width, height) in inches
    """
    if n_plots <= 0:
        raise ValueError("Number of plots must be positive")

    if n_plots == 1:
        n_rows, n_cols = 1, 1
    elif n_plots <= 3:
        n_rows, n_cols = 1, n_plots
    elif n_plots <= max_cols:
        n_rows, n_cols = 2, int(np.ceil(n_plots / 2))
    else:
        # Try to make layout as square as possible
        n_cols = min(max_cols, int(np.ceil(np.sqrt(n_plots))))
        n_rows = int(np.ceil(n_plots / n_cols))

        # Adjust if we can reduce rows by increasing columns slightly
        if n_cols < max_cols:
            alt_cols = n_cols + 1
            alt_rows = int(np.ceil(n_plots / alt_cols))
            if alt_rows < n_rows and alt_cols <= max_cols:
                n_cols = alt_cols
                n_rows = alt_rows

    # Calculate figure size
    fig_width = n_cols * subplot_width
    fig_height = n_rows * subplot_height
    figsize = (fig_width, fig_height)

    return n_rows, n_cols, figsize


def nice_dates():
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    formats = [
        "%y",  # ticks are mostly years
        "%b",  # ticks are mostly months
        "%d",  # ticks are mostly days
        "%H",  # hrs
        "%H:%M",  # min
        "%S.%f",
    ]  # secs
    formatter.formats = formats
    # these are mostly just the level above...
    zero_formats = [""] + formats[:-1]
    formatter.zero_formats = zero_formats
    # ...except for ticks that are mostly hours, then it is nice to have
    # month-day:
    formatter.zero_formats[3] = "%d-%b"
    offset_formats = [
        "",
        "%Y",
        "%b %Y",
        "%d %b %Y",
        "%d %b %Y",
        "%d %b %Y %H:%M",
    ]
    formatter.offset_formats = offset_formats

    converter = mdates.ConciseDateConverter(
        formats=formats, zero_formats=zero_formats, offset_formats=offset_formats
    )

    munits.registry[np.datetime64] = converter
    munits.registry[datetime.date] = converter
    munits.registry[datetime.datetime] = converter

    return locator


def plot_violin(tt, qubit_list=None, fname='def', param_keys=None, use_mean=False, csv_path="data.csv"):
    """
    Create violin plots for tracking data parameters across qubits.
    
    Parameters:
    -----------
    tt : list
        Tracking data for each qubit
    qubit_list : list, optional
        List of qubit indices to plot
    fname : str, optional
        Filename prefix for saving plots
    param_keys : list, optional
        List of parameter keys to plot
    use_mean : bool, optional
        Whether to normalize by mean values
    """
    
    if qubit_list is None:
        qubit_list = [i for i in range(len(tt))]
    if param_keys is None:
        param_keys = ["t1", "t2r", "t2", "f_ge", "fidelity", "kappa", "frequency", "pi_length", "tphi", "tsphi", "t2_r2", "t2et1", "t1_off", "t1_amp", "t2r_off", "t2r_amp", "q", "phase"]
    
    parameter_config = get_parameter_config()
    
    nrows, ncols, sz = calculate_subplot_layout(len(param_keys))
    
    # Collect data for all qubits and parameters
    violin_data = []
    qubit_labels = []
    param_labels = []
    
    for param_key in param_keys:
        if param_key not in parameter_config:
            print(f"Warning: Parameter '{param_key}' not found in configuration")
            continue
            
        config = parameter_config[param_key]
        key = config["key"]
        r2_key = config["r2_scan"]
        
        param_data = []
        param_qubit_labels = []
        
        for j, qi in enumerate(qubit_list):
            if key not in tt[qi]:
                continue
                
            # Remove outliers, same logic as plot_all
            if r2_key is not None and r2_key in tt[qi]:
                inds = (tt[qi][r2_key] > np.nanmean(np.array(tt[qi][r2_key])) - 0.08)
                #inds2 = (np.abs(tt[qi][key] - np.nanmean(tt[qi][key])) < np.nanstd(tt[qi][key]) * 4)
                #inds_true = np.logical_and(inds, inds2)
                inds_true = inds
                inds = np.where(inds_true)[0]
            else:
                inds = np.arange(len(tt[qi][key]))
            
            # Process data same as plot_all
            if key in ["f_ge", "frequency"]:
                y = 1e3 * (tt[qi][key][inds] - np.nanmean(tt[qi][key][inds]))
            elif key in ["pi_length"] or use_mean:
                y = tt[qi][key][inds] / np.nanmean(tt[qi][key][inds])
            elif key in ["t2r_amp", "t1_amp"]:
                y = np.abs(tt[qi][key][inds])
            elif key in ["fidelity"]:
                y = np.log10(1 - tt[qi][key][inds])
            else:
                y = tt[qi][key][inds]

            # Remove NaN values
            y_clean = y[~np.isnan(y)]
            if len(y_clean) > 0:
                param_data.extend(y_clean)
                param_qubit_labels.extend([f'Q{qi}'] * len(y_clean))
        
        if len(param_data) > 0:
            violin_data.append(param_data)
            qubit_labels.append(param_qubit_labels)
            param_labels.append(config["label"])
    
    # Create violin plots
    fig, axes = plt.subplots(nrows, ncols, figsize=sz)
    

    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i, (data, qubit_labs, param_label) in enumerate(zip(violin_data, qubit_labels, param_labels)):
        if i >= len(axes):
            break
            
        # Create DataFrame for seaborn
        df = pd.DataFrame({
            'Value': data,
            'Qubit': qubit_labs
        })
        
        # Create violin plot
        sns.violinplot(data=df, x='Qubit', y='Value', ax=axes[i])
        axes[i].set_ylabel(param_label)
        axes[i].set_xlabel('Qubit')
        axes[i].tick_params(axis='x', rotation=45)
        
        # # Add mean markers
        # for qubit in df['Qubit'].unique():
        #     qubit_data = df[df['Qubit'] == qubit]['Value']
        #     mean_val = qubit_data.mean()
        #     x_pos = df['Qubit'].unique().tolist().index(qubit)
        #     axes[i].plot(x_pos, mean_val, 'ro', markersize=4)
    
    # Hide unused subplots
    for i in range(len(violin_data), len(axes)):
        axes[i].set_visible(False)
    
    fig.tight_layout()

    # Save plots
    #file_path = Path(filename)
    qlist = [str(q) for q in qubit_list]
    qstr = ",".join(qlist)
    plist = ",".join(param_keys)
    pstr = plist.replace(" ", "")
    file_path = Path(csv_path)

    new_filename = file_path.name.rsplit(".", 1)[0] + "_violin" + qstr + "_" + pstr + ".png"
    fig.savefig(file_path.parent / "images" / new_filename)

    return fig, axes


def plot_sets(d, xvals, yvals, cols=4, nrep=10, fit_func=None, params=None):
    sns.set_palette("coolwarm", nrep)
    nsets = len(d)
    rows = int(np.ceil(nsets / nrep / cols))

    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    ax = ax.flatten()

    for i in range(int(nsets / nrep)):
        for j in range(nrep):
            if fit_func is None:
                ax[i].plot(
                    d[i * nrep + j][xvals],
                    d[i * nrep + j][yvals],
                    ".-",
                    markersize=2,
                    linewidth=0.5,
                )
            else:
                ax[i].plot(
                    d[i * nrep + j][xvals], d[i * nrep + j][yvals], "k.", markersize=1
                )
                ax[i].plot(
                    d[i * nrep + j][xvals],
                    fit_func(d[i * nrep + j][xvals], *params[i * nrep + j]),
                    linewidth=0.5,
                )

    fig.tight_layout()
    return fig, ax


def add_losses(tt): 
    par_list = ['t1','t2r','t2','t2e', 'tphi', 'tsphi']
    for j in range(len(tt)):
        for par in par_list:
            if par in tt[j].keys():
                tt[j]['Gamma' + par[1:]] = 1000 / tt[j][par]
    return tt
        
