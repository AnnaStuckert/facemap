
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

# Load the data from the .pt file
data = torch.load("data/processed_images_labels/schroeder_test.pt")
x = data['images']  # Tensor of images
y = data['labels']  # Tensor of keypoints (shape: N x 2*K)
filenames = data['filenames']  # List of filenames

# Select the first image and its keypoints for visualization
img = x[0].permute(1, 2, 0).numpy()  # Convert image tensor to numpy (H, W, C)
keypoints = y[0].numpy()  # Assuming keypoints are stored as a 1D array of x, y values

# Plot the image with keypoints
plt.figure(figsize=(12, 6))

# Left: Image with keypoints
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Image with Keypoints")
plt.scatter(keypoints[::2], keypoints[1::2], c='r', label='Keypoints')  # Keypoints are in (x, y) pairs
plt.legend()

# Right: Soft labels (we will use the same method as before to generate soft labels)
h, w = img.shape[0], img.shape[1]  # Height and Width of the image
mask = np.zeros((h, w))

# Remove NaN values from keypoints and create a mask
keypoints_filtered = keypoints[~np.isnan(keypoints)]
idx_x, idx_y = keypoints_filtered[1::2].astype(int), keypoints_filtered[::2].astype(int)
mask[idx_x, idx_y] = 1

# Apply Gaussian filter to generate soft labels
sigma = 5  # Increase from original 3 to 5 for smoother heatmap
soft_label = gaussian_filter(mask, sigma=sigma)

torch.save({
    'images': x,
    'softlabels': soft_label,
}, "data/schroeder_softlabels_new.pt")

torch.save({
    'images': x,
    'labels': y,
}, "data/schroeder_KP_new.pt")
