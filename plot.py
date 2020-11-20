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
    ax.axis('off')
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    ax.imshow(image)

    return ax

def plot_image_prediction(image, ps_values, ps_classes, ps_labels):
    ''' Function for viewing an image and it's predicted classes.
    '''

    fig, (ax1, ax2) = plt.subplots(figsize=(6,3), ncols=2)
    plot_image(image, ax1)
    ax2.barh(np.arange(len(ps_values)), ps_values)
    # ax2.set_aspect(0.1)
    # ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels([ps_labels], size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1)

    plt.tight_layout()