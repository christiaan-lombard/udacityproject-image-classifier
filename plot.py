import matplotlib.pyplot as plt

def plot_image_grid(images, ncols = 4, nrows = 1):
    """Plot images (as tensors) in a grid"""
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows)

    normal_images = []

    for r in range(nrows):
        for c in range(ncols):
            image = images[(r*nrows)+c].numpy().transpose((1, 2, 0))
            axes[r][c].imshow(image)
            axes[r][c].axis('off')
            # axes[r][c].set_adjustable('box-forced')
