import pdb

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage import io

# Load the data
x, y = torch.load("data/facemap_test_224.pt")
h, w = x[0].shape[1], x[0].shape[2]  # Ensure dimensions are taken from the same tensor
sigma = 3
softlabels = torch.zeros(x.shape)

for i in range(len(y)):
    mask = np.zeros((h, w))
    y_i = y[i][
        ~torch.isnan(y[i])
    ]  # Use torch.isnan instead of np.isnan for PyTorch tensors

    # Convert to integer indices using .to(torch.int)
    idx_x, idx_y = y_i[1::2].to(torch.int).numpy(), y_i[::2].to(torch.int).numpy()

    mask[idx_x, idx_y] = 1
    label = gaussian_filter(mask, sigma=sigma)

    # Store the result in the first channel of the softlabels tensor
    softlabels[i, 0] = torch.Tensor(label)

# Save the data
torch.save((x, y, softlabels), "data/facemap_softlabels_test_RGB.pt")
print("Data saved with labels having shape:", softlabels.shape)
