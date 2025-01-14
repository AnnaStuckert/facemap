import pdb

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage import io

# Load the data
x, y = torch.load("data/facemap_test_224_RGB.pt")
h, w = x[0].shape[1], x[0].shape[2]
sigma = 3

# Initialize softlabels with the same shape as x
softlabels = torch.zeros((x.shape[0], 1, h, w))

for i in range(len(y)):
    mask = np.zeros((h, w))

    # Use torch.isnan and ensure boolean indexing compatibility
    y_i = y[i][~torch.isnan(y[i])]  # Filter out NaN values

    # Convert to integer indices using .to(torch.int) and then .numpy() for numpy indexing
    idx_x, idx_y = y_i[1::2].to(torch.int).numpy(), y_i[::2].to(torch.int).numpy()

    # Set mask locations to 1
    mask[idx_x, idx_y] = 1

    # Apply Gaussian filter to the mask
    label = gaussian_filter(mask, sigma=sigma)

    # Store the result in the first channel of the softlabels tensor
    softlabels[i, 0] = torch.tensor(label, dtype=torch.float32)

# Save the processed data
torch.save((x, y, softlabels), "data/facemap_softlabels_test_RGB.pt")
print("Shapes - x:", x.shape, "| y:", y.shape, "| softlabels:", softlabels.shape)
