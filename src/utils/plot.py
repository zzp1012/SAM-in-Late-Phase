# import basic libs
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_hist(data: np.ndarray,
              save_path: str,
              fname: str,
              title: str = None,
              xlabel: str = None,
              ylabel: str = None,
              bins: int = 20,
              figsize: tuple = (7, 5)) -> None:
    """plot the histogram.
    Args:
        data (np.ndarray): the data.
        save_path (str): the path to save the figure.
        fname (str): the file name of the figure.
        title (str): the title of the figure.
        xlabel (str): the label of x axis.
        ylabel (str): the label of y axis.
        bins (int): the number of bins.
        figsize (tuple): the size of the figure.
    """
    # Initialise the figure and axes.
    fig, ax = plt.subplots(figsize=figsize)
    # Draw the histogram.
    ax.hist(data, bins=bins)
    ax.grid()
    ax.set(xlabel = xlabel, ylabel = ylabel, title = title)
    # save the fig
    path = os.path.join(save_path, fname)
    plt.savefig(path)
    plt.close()


def plot_multiple_hist(data_dict: dict,
                       save_path: str,
                       fname: str,
                       title: str = None,
                       xlabel: str = None,
                       ylabel: str = None,
                       bins: int = 20,
                       density: bool = False,
                       figsize: tuple = (7, 5)) -> None:
    """plot the multiple histograms.

    Args:
        data_dict (dict): the dictionary of the data.
        save_path (str): the path to save the figure.
        fname (str): the file name of the figure.
        title (str): the title of the figure.
        xlabel (str): the label of x axis.
        ylabel (str): the label of y axis.
        bins (int): the number of bins.
        density (bool): whether to plot the density.
        figsize (tuple): the size of the figure.
    """
    # Initialise the figure and axes.
    fig, ax = plt.subplots(figsize=figsize)
    # setup the bins
    max_val = max([max(data) for data in data_dict.values()])
    min_val = min([min(data) for data in data_dict.values()])
    bins_ = np.linspace(min_val, max_val, bins)

    # Draw the histogram.
    for idx, (label, data) in enumerate(data_dict.items()):
        plt.hist(data, bins=bins_, alpha=0.5, label=label, density=density)
    
    plt.grid()
    plt.legend(loc='upper right')
    
    # set the x and y labels and title
    ax.set(xlabel = xlabel, ylabel = ylabel, title = title)

    # save the fig
    path = os.path.join(save_path, fname)
    plt.savefig(path)
    plt.close()


def plot_twinx_curves(yleft: np.ndarray,
                      yright: np.ndarray,
                      x: np.ndarray,
                      save_path: str,
                      fname: str,
                      xlabel: str,
                      yleftlabel: str,
                      yrightlabel: str,
                      title: str,
                      yleftlim: list,
                      yrightlim: list,
                      figsize: tuple = (7, 6)) -> None:
    """plot two curves in one figure.

    Args:
        yleft (np.ndarray): the left y axis.
        yright (np.ndarray): the right y axis.
        x (np.ndarray): the x axis.
        save_path (str): the path to save the figure.
        fname (str): the file name of the figure.
        xlabel (str): the label of x axis.
        yleftlabel (str): the label of left y axis.
        yrightlabel (str): the label of right y axis.
        title (str): the title of the figure.
        yleftlim (list): the range of left y axis.
        yrightlim (list): the range of right y axis.
        figsize (tuple): the size of the figure.
    """
    # Initialise the figure and axes.
    fig, axleft = plt.subplots(figsize=figsize)
    axright = axleft.twinx()

    # Draw all the lines in the same plot, assigning a label for each one to be
    #   shown in the legend.
    axleft.plot(x, yleft, label=yleftlabel, c="r")
    axleft.scatter(x, yleft, s=5, c="k")
    axright.plot(x, yright, label=yrightlabel, c="b")
    axright.scatter(x, yright, s=5, c="k")
    axleft.grid(), axright.grid()

    # add the x and y labels and title
    axleft.set(xlabel = xlabel, title = title)
    axleft.set_ylabel(yleftlabel, color="r")
    axright.set_ylabel(yrightlabel, color="b")

    # Add a legend, and position it on the lower right (with no box)
    axleft.legend(frameon=False, loc="upper left")
    axright.legend(frameon=False, loc="upper right")

    # set ylim
    axleft.set_ylim(yleftlim)
    axright.set_ylim(yrightlim)
    
    # save the fig
    path = os.path.join(save_path, fname)
    fig.savefig(path)
    plt.close()


def plot_multiple_curves(Y: dict,
                         x: np.ndarray,
                         save_path: str, 
                         fname: str,
                         title: str = None,
                         xlabel: str = None,
                         ylabel: str = None,
                         xticks: list = None,
                         ylim: list = None,
                         rot: int = 0,
                         logscale: bool = False,
                         figsize: tuple = (7, 5),) -> None:
    """plot curves in one figure for each key in dictionary.
    Args:
        Y (dict): the dictionary of the curves.
        x (np.ndarray): the x axis.
        save_path (str): the path to save the figure.
        fname (str): the file name of the figure.
        title (str): the title of the figure.
        xlabel (str): the label of x axis.
        ylabel (str): the label of y axis.
        xticks (list): the ticks of x axis.
        ylim (list): the range of y axis.
        rot (int): the rotation of xticks.
        logsacle (bool): whether to use log scale.
        figsize (tuple): the size of the figure.
    """
    # Initialise the figure and axes.
    fig, ax = plt.subplots(figsize=figsize)
    # Draw all the lines in the same plot, assigning a label for each one to be
    #   shown in the legend.
    for label, y in Y.items():
        ax.plot(x, np.array(y), label=label, marker="o", linestyle="-")
        if logscale:
            ax.set_yscale('log')
    
    ax.grid()
    ax.set(xlabel = xlabel, ylabel = ylabel, title = title)
    
    # Add a legend, and position it on the lower right (with no box)
    plt.legend(frameon=True, prop={'size': 10})
    plt.tight_layout()
    plt.ylim(ylim)
    
    if xticks:
        plt.xticks(range(len(xticks)), xticks)
    # rotate the xticks
    plt.xticks(rotation=rot)
    plt.subplots_adjust(bottom=0.15)

    # save the fig
    path = os.path.join(save_path, fname)
    fig.savefig(path)
    plt.close()


def plot_multiple_curves_v2(data: dict,
                            save_path: str, 
                            fname: str,
                            title: str = None,
                            xlabel: str = None,
                            ylabel: str = None,
                            xticks: list = None,
                            ylim: list = None,
                            rot: int = 0,
                            figsize: tuple = (7, 5)) -> None:
    """plot curves in one figure for each key in dictionary.
    Args:
        data (dict): the dictionary of the curves.
        save_path (str): the path to save the figure.
        fname (str): the file name of the figure.
        title (str): the title of the figure.
        xlabel (str): the label of x axis.
        ylabel (str): the label of y axis.
        xticks (list): the ticks of x axis.
        ylim (list): the range of y axis.
        rot (int): the rotation of xticks.
        figsize (tuple): the size of the figure.
    """
    # Initialise the figure and axes.
    fig, ax = plt.subplots(figsize=figsize)
    # Draw all the lines in the same plot, assigning a label for each one to be
    #   shown in the legend.
    for label, (x, y) in data.items():
        ax.plot(x, y, label=label, marker="o", linestyle="-")
    
    ax.grid()
    ax.set(xlabel = xlabel, ylabel = ylabel, title = title)
    
    # Add a legend, and position it on the lower right (with no box)
    plt.legend(frameon=True, prop={'size': 10})
    plt.tight_layout()
    plt.ylim(ylim)
    
    if xticks:
        plt.xticks(range(len(xticks)), xticks)
    # rotate the xticks
    plt.xticks(rotation=rot)
    plt.subplots_adjust(bottom=0.15)

    # save the fig
    path = os.path.join(save_path, fname)
    fig.savefig(path)
    plt.close()


def plot_multiple_bars(Y: dict, 
                       xticks: list,  
                       save_path: str, 
                       fname: str,
                       title: str,
                       xlabel: str, 
                       ylabel: str,
                       ylim: list = None,
                       width: float = 0.2,
                       rot: int = 45,
                       figsize: tuple = (7, 6)) -> None:
    """
    Create a bar plot with multiple bars and a legend.

    Args:
        Y (dict): the dictionary of the curves.
        xticks (list): the x axis.
        title (str): the title of the figure.
        save_path (str): the path to save the figure.
        fname (str): the file name of the figure.
        xlabel (str): the label of x axis.
        ylabel (str): the label of y axis.
        ylim (list): the range of y axis.
        width (float): the width of the bar.
        rot (int): the rotation of the xticks.
    """

    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(xticks))

    # Plot each data series
    for i, (label, y) in enumerate(Y.items()):
        ax.bar(x + i*width, y, width=width, label=label)

    # set the y axis range
    if ylim is not None:
        ax.set_ylim(ylim)

    # Add labels, title, and legend
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation=rot, ha='right')
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(frameon=True, prop={'size': 10})

    # Automatically adjust the layout to prevent overlapping x-axis ticks
    fig.tight_layout()

    # save the fig
    path = os.path.join(save_path, fname)
    fig.savefig(path)
    plt.close()
