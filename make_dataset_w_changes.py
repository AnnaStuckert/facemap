import glob
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io, transform

IMG_LOC = "/Users/annastuckert/Documents/GitHub/facemap/data/facemap/"
# IMG_LOC = r"C:\Users\avs20\Documents\GitHub\facemap\data\schroeder"
if os.path.isdir(IMG_LOC + "low_res"):
    print("Folder exists!")
else:
    os.makedirs(IMG_LOC + "low_res")

img_files = sorted(glob.glob(IMG_LOC + "*.png"))
labels = pd.read_csv(IMG_LOC + "labels.csv")
h = w = 224

# print(img_files.shape)

img = plt.imread(img_files[0])

h_org = img.shape[0]
w_org = img.shape[1]

### Make new labels for low-res

# x_off = (300 - 224) // 2  # (h/h_org*w_org - w) // 2
x_off = (h / h_org * w_org - w) // 2

# Remove the first 3 rows and the first 3 columns from `labels`
labels = labels.iloc[2:, 2:]

target = labels.iloc[:, 1:].values
# print(target)
# print(target)
# print(target.dtype)
# print("h type:", type(h), "h_org type:", type(h_org))
target = np.array(target, dtype=np.float32)

target = target * h / h_org  # rescale markers

target[:, ::2] = target[:, ::2] - x_off
target = torch.Tensor(target)

labels.iloc[:, 1:] = target
labels.to_csv(IMG_LOC + "low_res/labels.csv", index=False)

data = torch.zeros((len(img_files), h, w))
print("Resizing images... \nSaving in torch format")

for i in range(len(img_files)):
    im = plt.imread(img_files[i])[:, :, 0]

    ### nyt fra Søreno ###
    x_start = (w_org - h_org) // 2
    im_cropped = im[:, x_start : x_start + h_org]  # Crop width to h_org, centered
    ### nyt fra Søreno ###

    im_r = (transform.resize(im_cropped, (h, w), anti_aliasing=True) * 255).astype(
        "uint8"
    )
    data[i] = torch.Tensor(im_r / 255.0)
    io.imsave(IMG_LOC + "low_res/" + img_files[i].split("/")[-1], im_r)

torch.save((data, target), IMG_LOC + "low_res/schroeder_224.pt")

print("Done! Saved in " + IMG_LOC + "low_res/")

# Load the data and transform `x` and `y` in the correct place
x, y = torch.load(IMG_LOC + "low_res/schroeder_224.pt")

# Transformations for `x` and `y`
x = x.unsqueeze(1)  # Add channel dimension to x
y = y.numpy()  # Convert y to numpy if needed

print("Data and labels are loaded and transformed.")

torch.save((x, y), "data/facemap_test_224.pt")
print(x.shape)
