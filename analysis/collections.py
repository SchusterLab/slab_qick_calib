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


def plot_all(tt, qubit_list=None, fname='def', data_inds=None, use_mean=False, nbins=40, plot_time=False):

    if qubit_list is None:
        qubit_list = [i for i in range(len(tt))]
    if data_inds is None:
        data_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    nrows, ncols, sz = calculate_subplot_layout(len(data_inds))

    key_list = [
        "t1",
        "t2r",
        "t2",
        "f_ge",
        "fidelity",
        "kappa",
        "frequency",
        "pi_length",
        "tphi",
        "tsphi",
        "t2_r2",
        "t2et1",
        "t1_off",
        "t1_amp",
        "t2r_off",
        "t2r_amp",
        "q",
        "phase",
    ]

    r2_scan = [
        "t1_r2",
        "t2r_r2",
        "t2_r2",
        "t2_r2",
        None,
        "rspec_r2",
        "rspec_r2",
        "amp_rabi_r2",
        "t2_r2",
        "t2_r2",
        None,
        "t2_r2",
        "t1_r2",
        "t1_r2",
        "t2r_r2",
        "t2r_r2",
        "t1_r2",
        None,
    ]
    labels = [
        "$T_1$ ($\mu$s)",
        "$T_{2,R}$ ($\mu$s)",
        "$T_{2,E}$ ($\mu$s)",
        "$f-\overline{f}$ (kHz), Qubit ",
        "Log(1 -Readout Fidelity)",
        "Resonator $\kappa$ (MHz)",
        "$f-\overline{f}$ (kHz), Resonator ",
        "$\pi_{gain}/\overline{\pi_{gain}}$",
        "$T_{\phi}$ ($\mu$s)",
        "$T_{\phi,R}$ ($\mu$s)",
        "$T_2 \, R^2$",
        "$T_{2,E}/2 T_1$",
        "$T_1$ offset",
        "$T_1$ amplitude",
        "$T_{2,R}$ offset",
        "$T_{2,R}$ amplitude",
        "$Q_1$",
        "Phase",
    ]
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

        for k in data_inds:
            key = key_list[k]
            # print(k)
            # Remove outliers, those with values far away from mean or small r^2
            if r2_scan[k] is not None:
                # Ignore data at least 0.08 lower r2 than mean and 4 std away from mean
                inds = (tt[j][r2_scan[k]] > np.nanmean(np.array(tt[j][r2_scan[k]])) - 0.08) 
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

                ax[i].set_ylabel(labels[k])
                ax[i].set_xlabel("Time (hr)")

            # Histogram plot
            ax2[i].hist(np.array(y), bins=nbins, label=int(qi), density=True)
            ax2[i].set_xlabel(labels[k])

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


def plot_violin(tt, qubit_list=None, fname='def', data_inds=None, use_mean=False):
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
    data_inds : list, optional
        Indices of parameters to plot
    use_mean : bool, optional
        Whether to normalize by mean values
    """
    
    if qubit_list is None:
        qubit_list = [i for i in range(len(tt))]
    if data_inds is None:
        data_inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    
    nrows, ncols, sz = calculate_subplot_layout(len(data_inds))

    key_list = [
        "t1",
        "t2r", 
        "t2",
        "f_ge",
        "fidelity",
        "kappa",
        "frequency",
        "pi_length",
        "tphi",
        "tsphi",
        "t2_r2",
        "t2et1",
        "t1_off",
        "t1_amp",
        "t2r_off",
        "t2r_amp",
        "q",
        "phase",
    ]

    r2_scan = [
        "t1_r2",
        "t2r_r2",
        "t2_r2",
        "t2_r2",
        None,
        "rspec_r2",
        "rspec_r2",
        "amp_rabi_r2",
        "t2_r2",
        "t2_r2",
        None,
        "t2_r2",
        "t1_r2",
        "t1_r2",
        "t2r_r2",
        "t2r_r2",
        "t1_r2",
        None,
    ]
    
    labels = [
        "$T_1$ ($\mu$s)",
        "$T_{2,R}$ ($\mu$s)",
        "$T_{2,E}$ ($\mu$s)",
        "$f-\overline{f}$ (kHz), Qubit ",
        "Log(1 -Readout Fidelity)",
        "Resonator $\kappa$ (MHz)",
        "$f-\overline{f}$ (kHz), Resonator ",
        "$\pi_{gain}/\overline{\pi_{gain}}$",
        "$T_{\phi}$ ($\mu$s)",
        "$T_{\phi,R}$ ($\mu$s)",
        "$T_2 \, R^2$",
        "$T_{2,E}/2 T_1$",
        "$T_1$ offset",
        "$T_1$ amplitude",
        "$T_{2,R}$ offset",
        "$T_{2,R}$ amplitude",
        "$Q_1$",
        "Phase",
    ]
    
    # Collect data for all qubits and parameters
    violin_data = []
    qubit_labels = []
    param_labels = []
    
    for k in data_inds:
        key = key_list[k]
        param_data = []
        param_qubit_labels = []
        
        for j, qi in enumerate(qubit_list):
            if key not in tt[j]:
                continue
                
            # Remove outliers, same logic as plot_all
            if r2_scan[k] is not None:
                inds = (tt[qi][r2_scan[k]] > np.nanmean(np.array(tt[qi][r2_scan[k]])) - 0.08) 
                inds2 = (np.abs(tt[qi][key] - np.nanmean(tt[qi][key])) < np.nanstd(tt[qi][key]) * 4)
                inds_true = np.logical_and(inds, inds2)
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
            param_labels.append(labels[k])
    
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
        
        # Add mean markers
        for qubit in df['Qubit'].unique():
            qubit_data = df[df['Qubit'] == qubit]['Value']
            mean_val = qubit_data.mean()
            x_pos = df['Qubit'].unique().tolist().index(qubit)
            axes[i].plot(x_pos, mean_val, 'ro', markersize=4)
    
    # Hide unused subplots
    for i in range(len(violin_data), len(axes)):
        axes[i].set_visible(False)
    
    fig.tight_layout()
    #fig.savefig(os.path.join("images", f"violin_plots_{fname}.png"), dpi=150, bbox_inches='tight')
    
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
