import pdb

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from skimage import io

x, y = torch.load("data/facemap_test_224_RGB.pt")
h, w = x[0].shape[1], x[1].shape[2]
sigma = 3
softlabels = torch.zeros(x.shape)
for i in range(len(y)):
    mask = np.zeros((h, w))
    y_i = y[i][~np.isnan(y[i])]
    idx_x, idx_y = y_i[1::2].astype(int), y_i[::2].astype(int)
    mask[idx_x, idx_y] = 1
    label = gaussian_filter(mask, sigma=sigma)
    softlabels[i, 0] = torch.Tensor(label)

torch.save((x, y, softlabels), "data/facemap_softlabels_test_RGB.pt")
