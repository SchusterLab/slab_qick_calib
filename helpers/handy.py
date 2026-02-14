from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_many(d,title='', save_path=None, chan='avgi',yax=None, norm=False, horz_line=None):
    nrows = int(np.ceil(len(d)/4))
    fig, ax = plt.subplots(nrows,4, figsize=(12,3*nrows))
    ax = ax.flatten()
    fig.suptitle(title)
    for i in range(len(d)):
        if norm==True: 
            for j in range(len(d[i].data[chan])):
                d[i].data[chan][j,:] = d[i].data[chan][j,:] / np.median(d[i].data[chan][j,:])
            
        ax[i].pcolormesh(d[i].data['xpts'],d[i].data['ypts'],d[i].data[chan])
        ax[i].set_title(f"Q{i}")
        if horz_line is not None:
            ax[i].axhline(horz_line[i], color='red')
        if yax=='log':
            ax[i].set_yscale('log')
    
    fig.tight_layout()
    if save_path is not None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = str(Path(save_path) / "images" / "summary" / (title + "_" + current_time + ".png"))
        plt.savefig(fname)

def plot_many_limited(d,row:int = 1,title='',save = True, save_path=None, chan='avgi',yax=None, norm=False, horz_line=None, individial_fig_size = (4,4), title_list = None, xlabel = None, sensitivity = 0.1):
    plot_number = len(d)
    col_number = int(np.ceil(plot_number/row))
    fig, ax = plt.subplots(row,col_number, figsize=(individial_fig_size[0]*col_number,individial_fig_size[1]*row))
    if plot_number == 1:
        ax = np.array([ax])
    else:
        ax = np.array(ax)
    ax = ax.flatten()
    fig.suptitle(title)

    # Normalize data across all plots for consistent colormap
    all_data = np.concatenate([d[i].data[chan].flatten() for i in range(len(d))])
    vmin, vmax = np.min(all_data), np.max(all_data)
    total_range = np.absolute(vmax - vmin)
    average = np.mean(all_data)
    vmax = average + (sensitivity * total_range)
    vmin = average - (sensitivity * total_range)

    for i in range(len(d)):
        if norm==True: 
            for j in range(len(d[i].data[chan])):
                d[i].data[chan][j,:] = d[i].data[chan][j,:] / np.median(d[i].data[chan][j,:])
            
        pcm = ax[i].pcolormesh(d[i].data['xpts'],d[i].data['ypts'],d[i].data[chan], vmin=vmin, vmax=vmax)
        
        if title_list is not None:
            ax[i].set_title(title_list[i])
        else:
            ax[i].set_title(f"Plot {i}")
        if horz_line is not None:
            ax[i].axhline(horz_line[i], color='red')
        if yax=='log':
            ax[i].set_yscale('log')
        if xlabel is not None:
            ax[i].set_xlabel(xlabel)
    
    fig.tight_layout()
    fig.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)  # Add a shared colorbar
    if save_path is not None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = str(Path(save_path) / "images" / "summary" / (title + "_" + current_time + ".png"))
        if save:
            plt.savefig(fname)



def config_figs():

    # Set seaborn color palette
    colors = ["#0869c8", "#b51d14", '#ddb310', '#658b38', '#7e1e9c', '#75bbfd', '#cacaca']
    sns.set_palette(sns.color_palette(colors))

    # Figure parameters
    plt.rcParams['figure.figsize'] = [8, 4]
    plt.rcParams.update({'font.size': 13})
