import matplotlib.pyplot as plt
import numpy as np

def plot_image_grid(images, ncols = 4, nrows = 1):
    """Plot images (as tensors) in a grid"""
    fig, axes = plt.subplots(nrows, ncols)

    for r in range(nrows):
        for c in range(ncols):
            image = images[(r*nrows)+c]
            plot_image(image, axes[r][c])
    return axes

def plot_image(image, ax=None):
    """Imshow for Tensor."""
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
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.imshow(image)

    return ax
