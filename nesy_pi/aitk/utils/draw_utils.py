# Created by shaji at 29/01/2024

import os
import datetime
import torch
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from matplotlib.backends.backend_agg import FigureCanvasAgg

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

def plot_to_np_array():
    fig = plt.gcf()
    canvas = FigureCanvasAgg(fig)
    # Render the plot to the FigureCanvasAgg
    canvas.draw()

    # Get the RGB values of the rendered plot
    width, height = fig.get_size_inches() * fig.get_dpi()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return image_array


def plot_line_chart(data, path, labels, x=None, title=None, x_scale=None, y_scale=None, y_label=None, show=False,
                    x_label=None, log_y=False, cla_leg=False, figure_size=None, conf_interval=False, color=None, line_width=1):
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
        plt.plot(x, y, label=labels[i], lw=line_width, color=color[i])

    if title is not None:
        plt.title(title, fontsize=22)

    if y_label is not None:
        plt.ylabel(y_label, fontsize=20)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=20)

    if log_y:
        plt.yscale('log')

    # Calculate standard deviation for confidence intervals
    if conf_interval:
        std_dev = torch.tensor(np.std(y.numpy()))
        # Plot confidence intervals
        lower_bound = y - 1.96 * std_dev
        upper_bound = y + 1.96 * std_dev
        plt.fill_between(np.arange(len(y)), lower_bound, upper_bound,
                         color='gray', alpha=0.3, label='95% Confidence Interval')

    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    if not os.path.exists(str(path)):
        os.mkdir(path)
    plt.savefig(
        str(Path(path) / f"{title}_{y_label}.png"))

    plot_array = plot_to_np_array()

    if show:
        plt.show()

    if cla_leg:
        plt.cla()
    matplotlib.pyplot.close()
    return plot_array


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


def plot_scatter(data, labels, name, path, log_x=False, log_y=False, cla_leg=True, figure_size=None):
    if figure_size is not None:
        plt.figure(figsize=figure_size)

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
    filename = str(Path(path) / f"{name}_scatter.png")
    plt.savefig(filename)
    # print(f" Scatter plot saved to {filename}")
    plot_array = plot_to_np_array()

    if cla_leg:
        plt.cla()

    matplotlib.pyplot.close()
    return plot_array


def plot_histogram(data, labels, name, path, log_x=False, log_y=False, figure_size=None, cla_leg=True):
    if figure_size is not None:
        plt.figure(figsize=figure_size)

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(3, 2)
    data_range = (-1, 1)
    # Plot histogram for Group A in the first subplot
    axs[0, 0].hist(data[0][0], bins=20, alpha=0.5, range=data_range, color="blue")
    axs[0, 0].set_title('X_Positive')
    axs[0, 0].set_xlabel('Value')
    axs[0, 0].set_ylabel('Frequency')

    # Plot histogram for Group B in the second subplot
    axs[0, 1].hist(data[0][1], bins=20, alpha=0.5, range=data_range, color="orange")
    axs[0, 1].set_title('X_Negative')
    axs[0, 1].set_xlabel('Value')
    axs[0, 1].set_ylabel('Frequency')

    # Plot histogram for Group A in the first subplot
    axs[1, 0].hist(data[1][0], bins=20, alpha=0.5, range=data_range, color="blue")
    axs[1, 0].set_title('Y_Positive')
    axs[1, 0].set_xlabel('Value')
    axs[1, 0].set_ylabel('Frequency')

    # Plot histogram for Group B in the second subplot
    axs[1, 1].hist(data[1][1], bins=20, alpha=0.5, range=data_range, color="orange")
    axs[1, 1].set_title('Y_Negative')
    axs[1, 1].set_xlabel('Value')
    axs[1, 1].set_ylabel('Frequency')

    # Plot histogram for Group A in the first subplot
    axs[2, 0].hist(data[2][0], bins=20, alpha=0.5, range=data_range, color="blue")
    axs[2, 0].set_title('Dir_Positive')
    axs[2, 0].set_xlabel('Value')
    axs[2, 0].set_ylabel('Frequency')

    # Plot histogram for Group B in the second subplot
    axs[2, 1].hist(data[2][1], bins=20, alpha=0.5, range=data_range, color="orange")
    axs[2, 1].set_title('Dir_Negative')
    axs[2, 1].set_xlabel('Value')
    axs[2, 1].set_ylabel('Frequency')

    # Adjust layout for better spacing
    plt.tight_layout()

    # Add labels and a legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    filename = str(Path(path) / f"histogram_{name}.png")
    plt.savefig(filename)

    plot_array = plot_to_np_array()

    if cla_leg:
        plt.cla()

    matplotlib.pyplot.close()
    return plot_array


def plot_decision_boundary(x_tensor, y_tensor, model, path, name, log_x=False, log_y=False, cla_leg=True):
    model.eval()
    with torch.no_grad():
        x_min, x_max = x_tensor[:, 0].min() - 1, x_tensor[:, 0].max() + 1
        y_min, y_max = x_tensor[:, 1].min() - 1, x_tensor[:, 1].max() + 1
        z_min, z_max = x_tensor[:, 2].min() - 1, x_tensor[:, 2].max() + 1

        unit = max((y_max - y_min) / 100, (x_max - x_min) / 100, (z_max - z_min) / 100)
        xx, yy, zz = torch.meshgrid(
            torch.arange(x_min, x_max, unit),
            torch.arange(y_min, y_max, unit),
            torch.arange(z_min, z_max, unit),
            indexing='ij')
        grid_tensor = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=1)
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

    file_name = str(Path(path) / f"{name}.png")
    plt.savefig(file_name)
    # print(f'- plot saved as {file_name}')
    plot_array = plot_to_np_array()

    if cla_leg:
        plt.cla()

    matplotlib.pyplot.close()
    return plot_array


def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    try:
        (h, w) = image.shape[:2]
    except AttributeError:
        print("")

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    if r > 1:

        resized = cv.resize(image, dim, interpolation=inter)
    else:
        resized = cv.resize(image, dim)

    # return the resized image
    return resized


def addText(img, text, pos='upper_left', font_size=1.6, color=(255, 255, 255), thickness=1):
    h, w = img.shape[:2]
    if pos == 'upper_left':
        position = [350, 140]
    elif pos == 'upper_right':
        position = [w - 350, 180]
    elif pos == 'lower_right':
        position = [h - 200, w - 20]
    elif pos == 'lower_left':
        position = [10, w - 20]
    else:
        position = pos

    text_y_shift = 40
    lines = text.split("\n")
    for line in lines:
        addCustomText(img, f"{line}", position, font_size=font_size, color=color, thickness=thickness)
        position[1] += text_y_shift
    # cv.putText(img, text=text, org=position,
    #            fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=color,
    #            thickness=thickness, lineType=cv.LINE_AA)


def addCustomText(img, text, pos, font_size=1.6, color=(255, 255, 255), thickness=1, shift=None):
    h, w = img.shape[:2]
    if pos[0] > w or pos[0] < 0 or pos[1] > h or pos[1] < 0:
        pos[0] = 10
        pos[1] = 10
        # raise ValueError('unsupported position to put text in the image.')
    if shift is not None:
        pos[0] = pos[0] + shift[0]
        pos[1] = pos[1] + shift[1]
    cv.putText(img, text=text, org=pos,
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=color,
               thickness=thickness, lineType=cv.LINE_AA)


# Function to draw arrow on an image
def draw_arrow(image, position, acceleration, scale=50, color=(0, 0, 255), thickness=2, shift=None):
    x, y = position
    ax, ay = acceleration
    if shift is not None:
        x, y = x + shift[0], y + shift[1]

    # Calculate the arrow endpoint based on acceleration
    arrow_end = (int(x + scale * ax), int(y + scale * ay))

    # Draw arrow on the image
    image_with_arrow = cv.arrowedLine(image.copy(), (int(x), int(y)), arrow_end, color, thickness)

    return image_with_arrow


def visual_info(data, height, width, font_size, text_pos):
    info_image = np.zeros((height, width, 3), dtype=np.uint8)
    # predicates info

    text_y_shift = 40
    lines = data.split("\n")[:9]
    for line in lines:
        addCustomText(info_image, f"{line}", text_pos, font_size=font_size)
        text_pos[1] += text_y_shift
    return info_image


def vconcat_resize(img_list, interpolation=cv.INTER_CUBIC):
    w_min = min(img.shape[1] for img in img_list)
    im_list_resize = [cv.resize(img,
                                (w_min, int(img.shape[0] * w_min / img.shape[1])), interpolation=interpolation)
                      for img in img_list]
    return cv.vconcat(im_list_resize)


def hconcat_resize(img_list, interpolation=cv.INTER_CUBIC):
    h_min = min(img.shape[0] for img in img_list)
    im_list_resize = [cv.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min))
                      for img in img_list]

    return cv.hconcat(im_list_resize)


def three_to_four_channel(img):
    alpha_plot = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8) * 255
    four_channel_image = np.concatenate([img, alpha_plot], axis=2)
    return four_channel_image


def show_images(array, title):
    cv.imshow(title, array)
    cv.waitKey(0)
    cv.destroyAllWindows()


def create_video_out(width, height):
    fps = 30
    fourcc = cv.VideoWriter_fourcc(*'XVID')  # Specify the video codec (XVID is just an example)
    # Create a VideoWriter object
    out = cv.VideoWriter('output_video.avi', fourcc, fps, (width, height))
    return out


def write_video_frame(video, frame):
    # Display or process the image as needed (replace this with your actual processing logic)
    cv.imshow('Frame', frame)
    cv.waitKey(1)  # Adjust the waitKey delay as needed

    # Write the 4-channel image to the video file
    video.write(frame)
    return video


def release_video(video):
    video.release()
    cv.destroyAllWindows()


def rgb_to_bgr(rgb_img):
    bgr_img = cv.cvtColor(rgb_img, cv.COLOR_RGB2BGR)
    return bgr_img


def save_np_as_img(np_array, file_name):
    # Convert the NumPy array to a Pillow Image
    cv.imwrite(str(file_name), np_array)


def load_img(image_path):
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Error: Unable to load the image from {image_path}")
    return img


def plot_heat_map(data, path=None, name=None, cla_leg=True, figsize=(10, 10), key_col=None, row_names=None,
                  col_names=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    # Plot the heatmap
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add colorbar to show values
    # Adjust layout for better spacing

    if key_col is not None:
        plt.axvline(x=key_col, color='red', linestyle='--', linewidth=2)
        plt.text(key_col, 5, 'key', color='red', fontsize=12, ha='center', va='center')
    if row_names is not None:
        plt.yticks(range(len(row_names)), row_names)
    if col_names is not None:
        plt.xticks(range(len(col_names)), col_names)
    filename = str(Path(path) / f"heatmap_{name}.png")
    plt.savefig(filename)

    plot_array = plot_to_np_array()

    if cla_leg:
        plt.cla()

    matplotlib.pyplot.close()
    return plot_array


def plot_compare_line_chart(data, path, name, figsize, row_names=None, cla_leg=True, key_name=None, neg_name=None,
                            key_rows=None, key_cols=None, neg_rows=None, neg_cols=None,
                            pos_color=None, neg_color=None):
    # Create a figure and three subplots vertically
    fig, axs = plt.subplots(len(data), 1, figsize=figsize, sharex='col')

    # Plot the first line chart
    for i in range(len(data)):
        if row_names is not None:
            label = row_names[i]
        else:
            label = "None"
        axs[i].plot(data[i], label=label)
        axs[i].set_title(label)
        axs[i].legend()

    if key_cols is not None:
        for ax in axs:
            for key_col in key_cols:
                ax.axvline(x=key_col, color=pos_color, linestyle='--', linewidth=1)
                ax.text(key_col, 5, "", color=pos_color, fontsize=8, ha='center', va='center')
            for key_col in neg_cols:
                ax.axvline(x=key_col, color=neg_color, linestyle='--', linewidth=1)
                ax.text(key_col, 5, "", color=neg_color, fontsize=8, ha='center', va='center')
    if key_rows is not None:
        for plot_i in range(len(key_rows)):
            pos_rows = key_rows[plot_i]
            for r_i, row in enumerate(pos_rows):
                axs[plot_i].axhline(y=row, color=pos_color, linestyle='--', linewidth=1)
                axs[plot_i].text(5, row, key_name[plot_i][r_i], color=pos_color, fontsize=10, ha='center', va='center')
            plot_neg_rows = neg_rows[plot_i]
            for r_i, row in enumerate(plot_neg_rows):
                axs[plot_i].axhline(y=row, color=neg_color, linestyle='--', linewidth=1)
                axs[plot_i].text(5, row, neg_name[plot_i][r_i], color=neg_color, fontsize=10, ha='center', va='center')
    filename = str(Path(path) / f"lines_{name}.png")
    plt.savefig(filename)

    plot_array = plot_to_np_array()

    if cla_leg:
        plt.cla()

    matplotlib.pyplot.close()
    return plot_array
