#!/usr/bin/env python3
#
# PROGRAMMER: Christiaan Lombard
# DATE CREATED: 2020-11-21
# REVISED DATE: 2020-11-21
# PURPOSE: Helper functions for plotting images
#
#


import matplotlib.pyplot as plt
import numpy as np


def plot_image_grid(images, ncols=4, nrows=1):
    """Plot images (as tensors) in a grid

    Args:
        images (Tensor[]): List of Tensors
        ncols (int, optional): Defaults to 4.
        nrows (int, optional): Defaults to 1.

    Returns:
        Axis: Plotted Axis
    """
    fig, axes = plt.subplots(nrows, ncols)

    for r in range(nrows):
        for c in range(ncols):
            image = images[(r*nrows)+c]
            plot_image(image, axes[r][c])
    return axes


def plot_image(image, ax=None):
    """Plot image from Tensor

    Args:
        image (Tensor): Image tensor as DataLoader provides
        ax (Axis, optional): Axis to plot to. Defaults to None.

    Returns:
        Axis: The plotted Axis
    """

    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.axis('off')
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.imshow(image)

    return ax


def plot_image_prediction(image, ps_values, ps_classes, ps_labels, title='Image'):
    """Plot Image and Prediction Chart

    Args:
        image (Tensor): Image Tensor
        ps_values (list): Predicted values
        ps_classes (list): Predicted class values
        ps_labels (list): Predicted class names
        figure ([type], optional): [description]. Defaults to None.
        title (str, optional): Image title. Defaults to 'Image'.
    """

    fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)

    plot_image(image, ax1)
    ax1.set_title(title)
    ax2.barh(np.arange(len(ps_values)), ps_values)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(ps_labels, size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1)

    plt.tight_layout()
