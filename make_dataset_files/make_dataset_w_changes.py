import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io, transform

# Base image location
IMG_LOC = r"C:\Users\avs20\Documents\GitHub\facemap\data\facemap"

# Ensure "low_res" folder exists
low_res_folder = os.path.join(IMG_LOC, "low_res")
if os.path.isdir(low_res_folder):
    print("Folder exists!")
else:
    os.makedirs(low_res_folder)

# Get list of image files
img_files = sorted(glob.glob(os.path.join(IMG_LOC, "*.png")))

# Load labels CSV
labels_path = os.path.join(IMG_LOC, "labels.csv")
labels = pd.read_csv(labels_path)

h = w = 224  # Target resolution

# Load the first image to get original dimensions
img = plt.imread(img_files[0])
h_org, w_org = img.shape[:2]  # Original image height and width

# Calculate offset for cropping
x_off = (h / h_org * w_org - w) // 2

# Adjust labels for low-res images
labels = labels.iloc[2:, 2:]  # Remove the first 3 rows and columns
target = labels.iloc[:, 1:].values.astype(np.float32)

# Rescale and adjust labels
target = target * h / h_org  # Rescale markers
target[:, ::2] -= x_off  # Adjust x-coordinates
target = torch.Tensor(target)

# Save updated labels
labels.iloc[:, 1:] = target
labels.to_csv(os.path.join(low_res_folder, "labels.csv"), index=False)

# Prepare to save resized images and tensor data
data = torch.zeros((len(img_files), h, w))
print("Resizing images... \nSaving in torch format")

for i, img_file in enumerate(img_files):
    im = plt.imread(img_file)[:, :, 0]  # Load image and convert to grayscale

    # Crop the width to match original height, centered
    x_start = (w_org - h_org) // 2
    im_cropped = im[:, x_start:x_start + h_org]

    # Resize image to target dimensions
    im_r = (transform.resize(im_cropped, (h, w), anti_aliasing=True) * 255).astype("uint8")
    
    # Save resized image
    save_path = os.path.join(low_res_folder, os.path.basename(img_file))
    io.imsave(save_path, im_r)

    # Normalize and add to tensor
    data[i] = torch.Tensor(im_r / 255.0)

# Save data and labels in PyTorch format
torch_save_path = os.path.join(low_res_folder, "TEST.pt")
torch.save((data, target), torch_save_path)

print(f"Done! Saved resized images and data in {low_res_folder}")

# Load and transform the saved data
x, y = torch.load(torch_save_path)


# Transformations for `x` and `y`
x = x.unsqueeze(1)  # Add channel dimension to `x`
y = y.numpy()  # Convert `y` to numpy if needed

print(x.shape)
print(y.shape)

# Save final transformed data
#final_save_path = "data/schroeder_test_224_new.pt"
final_save_path = "data/TEST.pt"
torch.save((x, y), final_save_path)

print(f"Data and labels are loaded and transformed.\nSaved in {final_save_path}")
print("Final x shape:", x.shape)
