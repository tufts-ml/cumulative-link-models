#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_model(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, export_path=None):
    """Function to plot the decision boundaries given a model.

    Parameters
    ----------
    model : tf.keras.Model
        Tensorflow like model that outputs decision scores f(x) for any input x
    X : np.ndarray-like
        Feature data, must be 2D
    y : np.ndarray-like
        Data labels
    export_path : pathlib.Path-like
        Path of desired export

    TODO
    ----
    * Generalize so that the function works with any number of ordinal labels--DONE (up until 5)
        * create a list of colors that are guaranteed to have the right color
          maps--DONE
        * Probably limited to the number of colormaps available. Assert that
          the function is not supported for more than N color maps
    * Assert that this is not supported for more than 2 dimensions
    * Figure out how to create own colormap so that the label limitation can be removed
    * Plot decision boundaries using contours
    """
    # Obtain Training data
    x0 = X[:, 0]
    x1 = X[:, 1]

    # Set the limits on x/y-axis close to the data
    pad = 1
    x0_lims = [x0.min()-pad, x0.max()+pad]
    x1_lims = [x1.min()-pad, x1.max()+pad]

    # Set up bounds and resolution for probability shading
    grid_resolution = 1000
    eps = 1.0
    x0_min, x0_max = x0_lims[0] - eps, x0_lims[1] + eps
    x1_min, x1_max = x1_lims[0] - eps, x1_lims[1] + eps
    left, right = x0_min, x0_max
    bottom, top = x1_min, x1_max

    # Create a grid of 2-d points to plot the decision scores
    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min, x0_max, grid_resolution),
        np.linspace(x1_min, x1_max, grid_resolution),
    )

    # Flatten the grid
    X_grid = np.c_[xx0.ravel(), xx1.ravel()]

    # Predict the scores on the grid of points
    p_grid = model(X_grid).categorical_probs().numpy().squeeze(axis=1)

    # Set up the shade and marker colors
    shade_colors = [plt.cm.Blues, plt.cm.Oranges,
                    plt.cm.Greens, plt.cm.Reds, plt.cm.Purples]
    marker_colors = ['darkblue', 'darkorange',
                     'darkgreen', 'darkred', 'darkmagenta']

    # Set the Greys colormap (for colorbar)
    grey_colors = plt.cm.Greys(np.linspace(0, 1, 201))
    grey_colors[:, 3] = 0.4
    grey_cmap = matplotlib.colors.ListedColormap(grey_colors)

    # Set up the figure, axis, and colorbar location
    f, axs = plt.subplots(1, 1, figsize=(5, 5))
    divider = make_axes_locatable(axs)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    # cax0 = divider.append_axes('right', size='5%', pad=0.2)
    # cax1 = divider.append_axes('right', size='5%', pad=0.3)
    # cax2 = divider.append_axes('right', size='5%', pad=0.4)
    # cax3 = divider.append_axes('right', size='5%', pad=0.5)
    axs.set_xlim(x0_lims)
    axs.set_ylim(x1_lims)
    axs.grid(False)

    # Iterate through labels and plot data and probability maps
    for label in range(np.max(y)+1):
        # Generate RGB-A values
        rgba_values = shade_colors[label](np.linspace(0, 1, 201))
        # Make opacity values increasingly transparent as colors become lighter
        rgba_values[:, 3] = np.linspace(0, 1, 201)
        # Create colormap object
        shade_cmap = matplotlib.colors.ListedColormap(rgba_values)

        # Plot training data markers and label colors
        axs.scatter(x0[y == label], x1[y == label],
                    marker='x',
                    linewidths=1,
                    color=marker_colors[label],
                    alpha=0.9,
                    label=f'y={label}',
                    )
        # Plot image of probability values for respective label
        axs.imshow(
            p_grid[:, [label]].reshape(xx0.shape),
            alpha=0.4, cmap=shade_cmap,
            interpolation='nearest',
            origin='lower',  # this is crucial
            extent=(left, right, bottom, top),
            vmin=0.0, vmax=1.0)

    # Create the color bar -- Generalized to be grey
    cbar = plt.colorbar(plt.cm.ScalarMappable(
        cmap=grey_cmap), cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar0 = plt.colorbar(im0, cax=cax0, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar1 = plt.colorbar(im1, cax=cax1, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar2 = plt.colorbar(im2, cax=cax2, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar3 = plt.colorbar(im3, cax=cax3, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # cbar0.draw_all()
    # cbar1.draw_all()
    # cbar2.draw_all()
    # cbar3.draw_all()

    # Include legend
    # handles, labels = axs.get_legend_handles_labels()
    # axs.legend(handles[::-1], labels[::-1],
    #            bbox_to_anchor=(1.04, 1), borderaxespad=0)
    # axs.legend()
    if export_path is not None:
        plt.savefig(export_path,
                    bbox_inches='tight', pad_inches=0)
        plt.close(f)
    else:
        plt.show()
