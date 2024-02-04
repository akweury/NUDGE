# Created by shaji at 29/01/2024

import os
import datetime
import torch
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def plot_line_chart(data, path, labels, x=None, title=None, x_scale=None, y_scale=None, y_label=None, show=False,
                    x_label=None,
                    log_y=False, cla_leg=False, figure_size=None):
    """ data with shape a*b, a is the number of lines, b is the data of each line """
    if data.shape[1] <= 1:
        return
    if figure_size is not None:
        plt.figure(figsize=figure_size)

    if y_scale is None:
        y_scale = [1, 1]
    if x_scale is None:
        x_scale = [1, 1]

    for i, row in enumerate(data):
        if x is None:
            x = np.arange(row.shape[0]) * x_scale[1]
        y = row
        plt.plot(x, y, label=labels[i], lw=1)

    if title is not None:
        plt.title(title)

    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)

    if log_y:
        plt.yscale('log')

    plt.legend()
    plt.grid(True)

    if not os.path.exists(str(path)):
        os.mkdir(path)
    plt.savefig(
        str(Path(path) / f"{title}_{y_label}_{date_now}_{time_now}.png"))

    if show:
        plt.show()
    if cla_leg:
        plt.cla()

    if show:
        plt.show()


def plot_head_maps(data, row_labels=None, path=None, title=None, y_label=None, x_label=None, col_labels=None, ax=None,
                   cbar_kw=None, cbarlabel="", figure_size=None, **kwargs):
    if figure_size is not None:
        plt.figure(figsize=figure_size)

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    if title is not None:
        plt.title(title)

    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)

    if not os.path.exists(str(path)):
        os.mkdir(path)
    plt.savefig(
        str(Path(path) / f"{title}_{y_label}_{date_now}_{time_now}.png"))

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def plot_scatter(data, labels, name, path, log_x=False, log_y=False, cla_leg=True):
    for d_i in range(len(data)):
        # Create a scatter plot
        x = data[d_i][:, 0]
        y = data[d_i][:, 1]
        plt.scatter(x, y, label=labels[d_i])

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'{name}')

    if log_y:
        plt.yscale('log')

    if log_x:
        plt.xscale('log')
    # Add a legend
    plt.legend()

    # save the plot
    plt.savefig(str(Path(path) / f"{name}_{date_now}_{time_now}_scatter.png"))

    if cla_leg:
        plt.cla()


def plot_decision_boundary(x_tensor, y_tensor, model, path, name, log_x=False, log_y=False, cla_leg=True):
    model.eval()
    with torch.no_grad():
        x_min, x_max = x_tensor[:, 0].min() - 1, x_tensor[:, 0].max() + 1
        y_min, y_max = x_tensor[:, 1].min() - 1, x_tensor[:, 1].max() + 1
        xx, yy = torch.meshgrid(torch.arange(x_min, x_max, 0.01), torch.arange(y_min, y_max, 0.01), indexing='ij')
        grid_tensor = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
        Z = model(grid_tensor).detach().argmax(dim=1).numpy().reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

    # Plot the data points
    plt.scatter(x_tensor[:, 0], x_tensor[:, 1], c=y_tensor.argmax(dim=1).numpy(), edgecolors='k', marker='o',
                cmap=plt.cm.Paired)
    plt.xlabel('X')
    plt.ylabel('Y')
    if log_y:
        plt.yscale('log')

    if log_x:
        plt.xscale('log')

    plt.title(f'Decision Boundary {name}')

    plt.savefig(str(Path(path) / f"{name}_decision_boundary.png"))

    if cla_leg:
        plt.cla()
