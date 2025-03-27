import matplotlib.pyplot as plt
import torch 
import numpy as np 
from log_utils import make_dir
import os 

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