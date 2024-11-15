import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io, transform

IMG_LOC = "/Users/annastuckert/Documents/GitHub/facemap/data/facemap/"

if os.path.isdir(IMG_LOC + "low_res"):
    print("Folder exists!")
else:
    os.makedirs(IMG_LOC + "low_res")

img_files = sorted(glob.glob(IMG_LOC + "*.png"))
labels = pd.read_csv(IMG_LOC + "labels.csv")
h = w = 224

# Read the first image to get original dimensions
img = plt.imread(img_files[0])
h_org, w_org, _ = img.shape  # Use all 3 channels

### Make new labels for low-res
x_off = (h / h_org * w_org - w) // 2

# Remove the first 3 rows and the first 3 columns from `labels`
labels = labels.iloc[2:, 2:]
target = labels.iloc[:, 1:].values
target = np.array(target, dtype=np.float32)
target = target * h / h_org  # rescale markers
target[:, ::2] = target[:, ::2] - x_off
target = torch.Tensor(target)

labels.iloc[:, 1:] = target
labels.to_csv(IMG_LOC + "low_res/labels.csv", index=False)

# Update data tensor shape to include 3 color channels
data = torch.zeros(
    (len(img_files), 3, h, w)
)  # Shape: (num_images, channels, height, width)

print("Resizing images... \nSaving in torch format")

for i in range(len(img_files)):
    im = plt.imread(img_files[i])  # Read the entire image with all channels

    ### Crop width to match the aspect ratio if necessary
    x_start = (w_org - h_org) // 2
    im_cropped = im[:, x_start : x_start + h_org, :]  # Keep all 3 channels

    # Resize to (h, w) and normalize pixel values to [0, 1]
    im_r = (transform.resize(im_cropped, (h, w), anti_aliasing=True) * 255).astype(
        "uint8"
    )

    # Store in data tensor (normalize values by dividing by 255.0)
    data[i] = (
        torch.Tensor(im_r).permute(2, 0, 1) / 255.0
    )  # Permute to (channels, height, width)

    # Save the resized image for reference
    io.imsave(IMG_LOC + "low_res/" + img_files[i].split("/")[-1], im_r)

# Save the final dataset
torch.save((data, target), IMG_LOC + "low_res/schroeder_224.pt")

print("Done! Saved in " + IMG_LOC + "low_res/")

# Load the data and transform `x` and `y` in the correct place
x, y = torch.load(IMG_LOC + "low_res/schroeder_224.pt")
print(x.shape)

print("Data and labels are loaded and transformed.")

# Save the transformed data for the next steps
torch.save((x, y), "data/facemap_test_224_RGB.pt")
