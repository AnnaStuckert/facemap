import gc
import glob
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from numpy import random
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, TensorDataset
from torchvision.utils import save_image

from models import Unet

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceMapDataset(Dataset):
    def __init__(
        self,
        data_file="data/schroeder_test_softlabels_224.pt",
        transform=None,
        rotation_degrees=15,
        blur_radius=(1, 2),  # Tuple for Gaussian blur radius range
    ):
        super().__init__()
        self.transform = transform
        self.rotation_degrees = rotation_degrees
        self.blur_radius = blur_radius
        self.data, _, self.targets = torch.load(data_file)

    def __len__(self):
        # Return the total count, multiplied by 5 for five versions per image
        return len(self.targets) * 5

    def __getitem__(self, index):
        # Get the base index (original image index) and augmentation type
        base_index = index // 5  # Original image index
        aug_type = (
            index % 5
        )  # 0: original, 1: flipped, 2: rotated, 3: zoomed, 4: blurred

        # Load the original image and label
        image, label = self.data[base_index].clone(), self.targets[base_index].clone()

        # Apply the augmentation based on the `aug_type`
        if self.transform is not None:
            if aug_type == 1:  # Flipping
                image = image.flip([2])
                label = label.flip([2])
            elif aug_type == 2:  # Rotation
                angle = (torch.rand(1).item() * 2 - 1) * self.rotation_degrees
                image = TF.rotate(image, angle)
                label = TF.rotate(label, angle)
            elif aug_type == 3:  # Zooming
                scale_factor = 1.1 if torch.rand(1).item() < 0.5 else 0.9
                image = self.zoom(image, scale_factor)
                label = self.zoom(label, scale_factor)
            elif aug_type == 4:  # Gaussian Blur
                # Random radius within the specified range
                radius = (
                    torch.rand(1).item() * (self.blur_radius[1] - self.blur_radius[0])
                    + self.blur_radius[0]
                )
                image = TF.gaussian_blur(image, kernel_size=int(radius))
                label = TF.gaussian_blur(label, kernel_size=int(radius))

        return image, label

    def zoom(self, img, scale_factor):
        # Calculate new dimensions
        _, h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # Resize and center-crop back to the original size
        img = TF.resize(img, [new_h, new_w])
        img = TF.center_crop(img, [h, w])
        return img


### Make dataset
dataset = FaceMapDataset(transform=None)

x = dataset[0][0]
dim = x.shape[-1]
print("Using %d size of images" % dim)
N = len(dataset)
train_sampler = SubsetRandomSampler(np.arange(int(0.6 * N)))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6 * N), int(0.8 * N)))
test_sampler = SubsetRandomSampler(np.arange(int(0.8 * N), N))
batch_size = 4
# Initialize loss and metrics
loss_fun = torch.nn.MSELoss(reduction="sum")

# Initiliaze input dimensions
num_train = len(train_sampler)
num_valid = len(valid_sampler)
num_test = len(test_sampler)
print(
    "Num. train = %d, Num. val = %d, Num. test = %d" % (num_train, num_valid, num_test)
)

# Initialize dataloaders
loader_train = DataLoader(
    dataset=dataset,
    drop_last=False,
    num_workers=0,
    batch_size=batch_size,
    pin_memory=True,
    sampler=train_sampler,
)
loader_valid = DataLoader(
    dataset=dataset,
    drop_last=True,
    num_workers=0,
    batch_size=batch_size,
    pin_memory=True,
    sampler=valid_sampler,
)
loader_test = DataLoader(
    dataset=dataset,
    drop_last=True,
    num_workers=0,
    batch_size=1,
    pin_memory=True,
    sampler=test_sampler,
)

nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

### hyperparam
lr = 5e-4
num_epochs = 300

model = Unet()
# timm.create_model('vit_base_patch8_224',
#        pretrained=True,in_chans=1,num_classes=num_output_classes)

model = model.to(device)
nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:%2f M" % (nParam / 1e6))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
minLoss = 1e6
convIter = 0
patience = 1000
train_loss = []
valid_loss = []

for epoch in range(num_epochs):
    tr_loss = 0
    for i, (inputs, labels) in enumerate(loader_train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, _ = model(inputs)
        loss = loss_fun((scores), ((labels)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, i + 1, nTrain, loss.item()
            )
        )
        tr_loss += loss.item()
    train_loss.append(tr_loss / (i + 1))

    with torch.no_grad():
        val_loss = 0
        for i, (inputs, labels) in enumerate(loader_valid):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores, fmap = model(inputs)
            loss = loss_fun((scores), ((labels)))
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

        valid_loss.append(val_loss)

        print("Val. loss :%.4f" % val_loss)

        labels = labels.squeeze().detach().cpu().numpy()
        scores = scores.squeeze().detach().cpu().numpy()
        img = inputs.squeeze().detach().cpu().numpy()
        fmap = inputs.mean(1).squeeze().detach().cpu().numpy()

        plt.clf()
        plt.figure(figsize=(16, 12))
        for i in range(batch_size):
            plt.subplot(batch_size, 3, 3 * i + 1)
            plt.imshow(labels[i])
            plt.subplot(batch_size, 3, 3 * i + 2)
            plt.imshow(scores[i] * img[i])
            plt.subplot(batch_size, 3, 3 * i + 3)
            plt.imshow(fmap[i])

        plt.tight_layout()

        plt.savefig("logs/epoch_%03d.jpg" % epoch)
        plt.close()  # prevent 'fail to allocate bitmap' error at epoch 262
        gc.collect()

        if minLoss > val_loss:
            convEpoch = epoch
            minLoss = val_loss
            convIter = 0
            torch.save(
                model.state_dict(), "models/best_model.pt"
            )  # why is this hashed out?
        else:
            convIter += 1

        if convIter == patience:
            print(
                "Converged at epoch %d with val. loss %.4f" % (convEpoch + 1, minLoss)
            )
            break
plt.clf()
plt.plot(train_loss, label="Training")
plt.plot(valid_loss, label="Valid")
plt.plot(convEpoch, valid_loss[convEpoch], "x", label="Final Model")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.pdf")

import gc
import glob

### Load best model for inference
import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from numpy import random
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, TensorDataset
from torchvision.utils import save_image

from models import Unet

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceMapDataset(Dataset):
    def __init__(
        self,
        data_file="data/schroeder_test_softlabels_224.pt",
        transform=None,
        rotation_degrees=15,
        blur_radius=(1, 2),  # Tuple for Gaussian blur radius range
    ):
        super().__init__()
        self.transform = transform
        self.rotation_degrees = rotation_degrees
        self.blur_radius = blur_radius
        self.data, _, self.targets = torch.load(data_file)

    def __len__(self):
        # Return the total count, multiplied by 5 for five versions per image
        return len(self.targets) * 5

    def __getitem__(self, index):
        # Get the base index (original image index) and augmentation type
        base_index = index // 5  # Original image index
        aug_type = (
            index % 5
        )  # 0: original, 1: flipped, 2: rotated, 3: zoomed, 4: blurred

        # Load the original image and label
        image, label = self.data[base_index].clone(), self.targets[base_index].clone()

        # Apply the augmentation based on the `aug_type`
        if self.transform=="transform":
            if aug_type == 1:  # Flipping
                image = image.flip([2])
                label = label.flip([2])
            elif aug_type == 2:  # Rotation
                angle = (torch.rand(1).item() * 2 - 1) * self.rotation_degrees
                image = TF.rotate(image, angle)
                label = TF.rotate(label, angle)
            elif aug_type == 3:  # Zooming
                scale_factor = 1.1 if torch.rand(1).item() < 0.5 else 0.9
                image = self.zoom(image, scale_factor)
                label = self.zoom(label, scale_factor)
            elif aug_type == 4:  # Gaussian Blur
                # Random radius within the specified range
                radius = (
                    torch.rand(1).item() * (self.blur_radius[1] - self.blur_radius[0])
                    + self.blur_radius[0]
                )
                image = TF.gaussian_blur(image, kernel_size=int(radius))
                label = TF.gaussian_blur(label, kernel_size=int(radius))

        return image, label

    def zoom(self, img, scale_factor):
        # Calculate new dimensions
        _, h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # Resize and center-crop back to the original size
        img = TF.resize(img, [new_h, new_w])
        img = TF.center_crop(img, [h, w])
        return img


### Make dataset
dataset = FaceMapDataset(transform="transform")

x = dataset[0][0]
dim = x.shape[-1]
print("Using %d size of images" % dim)
N = len(dataset)
train_sampler = SubsetRandomSampler(np.arange(int(0.6 * N)))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6 * N), int(0.8 * N)))
test_sampler = SubsetRandomSampler(np.arange(int(0.8 * N), N))
batch_size = 4
# Initialize loss and metrics
loss_fun = torch.nn.MSELoss(reduction="sum")

# Initiliaze input dimensions
num_train = len(train_sampler)
num_valid = len(valid_sampler)
num_test = len(test_sampler)
print(
    "Num. train = %d, Num. val = %d, Num. test = %d" % (num_train, num_valid, num_test)
)

# Initialize dataloaders
loader_train = DataLoader(
    dataset=dataset,
    drop_last=False,
    num_workers=0,
    batch_size=batch_size,
    pin_memory=True,
    sampler=train_sampler,
)
loader_valid = DataLoader(
    dataset=dataset,
    drop_last=True,
    num_workers=0,
    batch_size=batch_size,
    pin_memory=True,
    sampler=valid_sampler,
)
loader_test = DataLoader(
    dataset=dataset,
    drop_last=True,
    num_workers=0,
    batch_size=1,
    pin_memory=True,
    sampler=test_sampler,
)

nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

### hyperparam
lr = 5e-4
num_epochs = 300

model = Unet()
# timm.create_model('vit_base_patch8_224',
#        pretrained=True,in_chans=1,num_classes=num_output_classes)

model = model.to(device)
nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:%2f M" % (nParam / 1e6))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
minLoss = 1e6
convIter = 0
patience = 10
train_loss = []
valid_loss = []

for epoch in range(num_epochs):
    tr_loss = 0
    for i, (inputs, labels) in enumerate(loader_train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, _ = model(inputs)
        loss = loss_fun((scores), ((labels)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                epoch + 1, num_epochs, i + 1, nTrain, loss.item()
            )
        )
        tr_loss += loss.item()
    train_loss.append(tr_loss / (i + 1))

    with torch.no_grad():
        val_loss = 0
        for i, (inputs, labels) in enumerate(loader_valid):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores, fmap = model(inputs)
            loss = loss_fun((scores), ((labels)))
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

        valid_loss.append(val_loss)

        print("Val. loss :%.4f" % val_loss)

        labels = labels.squeeze().detach().cpu().numpy()
        scores = scores.squeeze().detach().cpu().numpy()
        img = inputs.squeeze().detach().cpu().numpy()
        fmap = inputs.mean(1).squeeze().detach().cpu().numpy()

        plt.clf()
        plt.figure(figsize=(16, 12))
        for i in range(batch_size):
            plt.subplot(batch_size, 3, 3 * i + 1)
            plt.imshow(labels[i])
            plt.subplot(batch_size, 3, 3 * i + 2)
            plt.imshow(scores[i] * img[i])
            plt.subplot(batch_size, 3, 3 * i + 3)
            plt.imshow(fmap[i])

        plt.tight_layout()

        plt.savefig("logs/epoch_%03d.jpg" % epoch)
        plt.close()  # prevent 'fail to allocate bitmap' error at epoch 262
        gc.collect()

        if minLoss > val_loss:
            convEpoch = epoch
            minLoss = val_loss
            convIter = 0
            torch.save(
                model.state_dict(), "models/best_model.pt"
            )  # why is this hashed out?
        else:
            convIter += 1

        if convIter == patience:
            print(
                "Converged at epoch %d with val. loss %.4f" % (convEpoch + 1, minLoss)
            )
            break
plt.clf()
plt.plot(train_loss, label="Training")
plt.plot(valid_loss, label="Valid")
plt.plot(convEpoch, valid_loss[convEpoch], "x", label="Final Model")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.pdf")

### Load best model for inference
### Load best model for inference
with torch.no_grad():

    # Second pass: compute loss and plot images with normalized feature maps
    val_loss = 0
    for i, (inputs, labels) in enumerate(loader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, fmap = model(inputs)
        loss = loss_fun(scores, labels)
        val_loss += loss.item()

        img = inputs.squeeze().detach().cpu().numpy()
        pred = scores.squeeze().detach().cpu().numpy()
        labels_np = labels.squeeze().cpu().numpy()
        fmap_mean = fmap.mean(1).squeeze().cpu().numpy()
        fmap_each = fmap.squeeze().cpu().numpy()

        # Extract individual normalized feature maps
        fmap_1 = fmap_each[0]
        fmap_2 = fmap_each[1]
        fmap_3 = fmap_each[2]
        fmap_4 = fmap_each[3]
        fmap_5 = fmap_each[4]
        fmap_6 = fmap_each[5]
        fmap_7 = fmap_each[6]
        fmap_8 = fmap_each[7]

        # Plotting code
        plt.clf()
        plt.figure(figsize=(12, 9))

        # Display the main images
        plt.subplot(3, 4, 1)
        plt.imshow(img)
        plt.title("Input Image")

        plt.subplot(3, 4, 2)
        plt.imshow(labels_np)
        plt.title("Ground Truth")

        plt.subplot(3, 4, 3)
        plt.imshow(pred)
        plt.title("Prediction")

        plt.subplot(3, 4, 4)
        plt.imshow(fmap_mean)
        plt.title("Normalized Feature Map Mean")

        # Display each individual normalized feature map
        plt.subplot(3, 4, 5)
        plt.imshow(fmap_1)
        plt.title("Feature Map 1")

        plt.subplot(3, 4, 6)
        plt.imshow(fmap_2)
        plt.title("Feature Map 2")

        plt.subplot(3, 4, 7)
        plt.imshow(fmap_3)
        plt.title("Feature Map 3")

        plt.subplot(3, 4, 8)
        plt.imshow(fmap_4)
        plt.title("Feature Map 4")

        plt.subplot(3, 4, 9)
        plt.imshow(fmap_5)
        plt.title("Feature Map 5")

        plt.subplot(3, 4, 10)
        plt.imshow(fmap_6)
        plt.title("Feature Map 6")

        plt.subplot(3, 4, 11)
        plt.imshow(fmap_7)
        plt.title("Feature Map 7")

        plt.subplot(3, 4, 12)
        plt.imshow(fmap_8)
        plt.title("Feature Map 8")

        plt.tight_layout()
        plt.savefig("preds/test_{:03d}.jpg".format(i))
        plt.close()
        gc.collect()

    val_loss = val_loss / (i + 1)
    print("Test loss: {:.4f}".format(val_loss))
