import numpy as np
import matplotlib.pyplot as plt

def dct_basis(N=8):
    basis = np.zeros((N, N, N, N))
    for u in range(N):
        for v in range(N):
            for x in range(N):
                for y in range(N):
                    alpha_u = np.sqrt(1/N) if u == 0 else np.sqrt(2/N)
                    alpha_v = np.sqrt(1/N) if v == 0 else np.sqrt(2/N)
                    basis[u, v, x, y] = (
                        alpha_u * alpha_v *
                        np.cos(((2*x + 1) * u * np.pi) / (2 * N)) *
                        np.cos(((2*y + 1) * v * np.pi) / (2 * N))
                    )
    return basis

def plot_dct_basis(basis):
    N = basis.shape[0]
    fig, axs = plt.subplots(N, N, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    for u in range(N):
        for v in range(N):
            axs[u, v].imshow(basis[u, v], cmap='gray', extent=[0, N, 0, N])
            axs[u, v].set_title(f'u={u}, v={v}', fontsize=8)
            axs[u, v].axis('off')

    plt.suptitle('2D DCT Basis Functions (8x8)', fontsize=16)
    plt.show()

# Run it
basis = dct_basis(N=8)
plot_dct_basis(basis)
exit()

import numpy as np 
import pandas as pd
import math
import matplotlib.pyplot as plt 

def plot_grayscale_images(images):
    if not isinstance(images, np.ndarray) or images.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array")
    
    N, H, W = images.shape
    if (H, W) not in [(4, 4), (8, 8)]:
        raise ValueError("Each image must be 4x4 or 8x8 in size")

    fig, axes = plt.subplots(1, N, figsize=(2 * N, 2))
    if N == 1:
        axes = [axes]  # Make iterable if only one image

    for i in range(N):
        ax = axes[i]
        ax.imshow(images[i], cmap='gray',)
        ax.axis('off')
        ax.set_title(f'Image {i+1}', fontsize=10)

    plt.tight_layout()
    plt.show()

D = 4
def fi(i, j, d): 
    return np.cos((math.pi/d)*(i+0.5)*j)

# rand_img = np.abs(np.random.randn(D, D))
# plt.imshow(rand_img, cmap='gray')
# plt.show()
# plt.clf()

fis = []
for d in range(D): 
    fi_mat = np.empty((D, D))

    for i in range(0, D):
        for j in range(0, D): 
            fi_mat[i, j] = fi(i=i, j=j, d=d+1)

    fis.append(fi_mat)

fis = np.array(fis)
print(fis.shape)
plot_grayscale_images(fis)
exit()

# fi_mat = fi_mat[1:, :]
# fi_mat = fi_mat[:, 1:]

plt.imshow(fi_mat, cmap='gray', interpolation='none')
# plt.plot(fi_mat)
plt.show()

v_mat = np.empty((D+1, D+1))


img = plt.imread('D:\Backdoor\code\DCT\lena.jpg')
print(img.shape)