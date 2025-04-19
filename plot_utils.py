import matplotlib.pyplot as plt
import torch 
import numpy as np 
from log_utils import make_dir
import os 

def plot_patch_heatmap(patch_values, path, name, grid_size=5, ):
    """
    Plot a heatmap from patch values in bottom-right → top-left order.
    Grid is shown directly; no image coordinates involved.
    """
    assert len(patch_values) == grid_size ** 2, f"Expected {grid_size**2} patch values"

    # Create empty grid
    heatmap = np.zeros((grid_size, grid_size))

    idx = 0
    for row in range(grid_size):  # from bottom (0) to top (4)
        for col in range(grid_size-1, -1, -1):  # right to left
            heatmap[grid_size - 1 - row, col] = patch_values[idx]
            idx += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heatmap, cmap='viridis', interpolation='nearest')

    # Annotate with values
    for i in range(grid_size):
        for j in range(grid_size):
            ax.text(j, i, f'{heatmap[i, j]:.2f}', ha='center', va='center', color='white', fontsize=8)

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels([str(i) for i in range(grid_size)])
    ax.set_yticklabels([str(i) for i in range(grid_size)])
    ax.set_title("Patch Value Heatmap (Bottom-Right → Top-Left)")
    plt.colorbar(im, ax=ax)
    plt.savefig(f'{path}/{name}.png')

def display_image_grid(images1, images2, titles1=None, titles2=None, save_path=None, name_to_save=None, dpi=150):
    """
    images1 and images2: lists of 8 grayscale images (numpy arrays)
    titles1 and titles2: optional lists of 8 subtitles for images1 and images2
    save_path: optional string to save the figure (e.g., 'output/grid.png')
    """
    assert images1.shape[0] == 8 and images2.shape[0] == 8, "Each input should contain 8 images"

    fig, axes = plt.subplots(4, 4, figsize=(10, 10), dpi=dpi)
    
    if titles1 is None:
        titles1 = [f"Image1_{i+1}" for i in range(8)]
    if titles2 is None:
        titles2 = [f"Image2_{i+1}" for i in range(8)]
    
    for row in range(4):
        # Column 1: first 4 images from images1
        axes[row, 0].imshow(images1[row, 0], cmap='gray')
        axes[row, 0].set_title(titles1[row], fontsize=8)
        axes[row, 0].axis('off')

        # Column 2: first 4 images from images2
        axes[row, 1].imshow(images2[row, 0], cmap='gray')
        axes[row, 1].set_title(titles2[row], fontsize=8)
        axes[row, 1].axis('off')

        # Column 3: last 4 images from images1
        axes[row, 2].imshow(images1[row + 4, 0], cmap='gray')
        axes[row, 2].set_title(titles1[row + 4], fontsize=8)
        axes[row, 2].axis('off')

        # Column 4: last 4 images from images2
        axes[row, 3].imshow(images2[row + 4, 0], cmap='gray')
        axes[row, 3].set_title(titles2[row + 4], fontsize=8)
        axes[row, 3].axis('off')
    
    plt.tight_layout()
    
    make_dir(save_path)
    plt.savefig(f'{save_path}/{name_to_save}', bbox_inches='tight')
    
    plt.close(fig)  # Close the figure to free memory if saving
    
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_and_save_confusion_matrix(y_true, y_pred, save_dir, name):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create the display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # Plot and save the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
    
    # Save the plot
    save_path = os.path.join(save_dir, f"{name}_confusion_matrix.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to: {save_path}")