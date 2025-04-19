# from margins.utils import generate_subspace_list
# SUBSPACE_DIM = 8
# DIM = 28
# SUBSPACE_STEP = 1

# subspace_list = generate_subspace_list(SUBSPACE_DIM, DIM, SUBSPACE_STEP, channels=1)
# print(len(subspace_list))

import numpy as np
import matplotlib.pyplot as plt

def generate_patch_masks_numpy(mask_size: int, input_dim: int, step_size: int, channels: int):
    masks = []
    for y in range(input_dim - mask_size, -1, -step_size):
        for x in range(input_dim - mask_size, -1, -step_size):
            mask = np.zeros((input_dim, input_dim), dtype=np.float32)
            mask[y:y+mask_size, x:x+mask_size] = 1.0
            # print(y,y+mask_size, ' | ', x,x+mask_size)
            mask = np.repeat(mask[np.newaxis, :, :], channels, axis=0)
            masks.append(mask)
    return np.stack(masks)

# Generate masks with numpy
masks_np = generate_patch_masks_numpy(mask_size=5, input_dim=28, step_size=5, channels=1)


# Visualize
num_masks = masks_np.shape[0]
num_masks = 2
print(num_masks)
fig, axes = plt.subplots(1, num_masks, figsize=(num_masks * 2, 2))
for i in range(num_masks):
    ax = axes[i]
    ax.imshow(masks_np[i][0], cmap='gray')
    ax.set_xticks([27, 26, 25, 24, 23, 22, 21])
    ax.set_yticks([27, 26, 25, 24, 23, 22, 21])
    # ax.set_xlim((0, 28))
    # ax.set_ylim((0, 28))
    # ax.axis('off')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_patch_heatmap(patch_values, grid_size=5):
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
    plt.show()

patch_values = np.arange(25)   # Random values from 0 to 10
plot_patch_heatmap(patch_values)
