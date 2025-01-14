import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from models import Unet

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FaceMapDataset(Dataset):
    def __init__(
        self,
        data_file="data/dolensek_facemap_softlabels_224.pt",
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
        aug_type = index % 5  # 0: original, 1: flipped, 2: rotated, 3: zoomed, 4: blurred

        # Load the original image and label
        image, label = self.data[base_index].clone(), self.targets[base_index].clone()

        # Apply the augmentation based on the `aug_type`
        if self.transform:
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


# Dataset creation with transformation
dataset = FaceMapDataset(transform="test")

# Checking the size of an image
x = dataset[0][0]
dim = x.shape[-1]
print(f"Using {dim} size of images")

# Data Splitting
N = len(dataset)
indices = np.random.permutation(N)
train_indices = indices[:int(0.6*N)]
valid_indices = indices[int(0.6*N):int(0.8*N)]
test_indices = indices[int(0.8*N):]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(valid_indices)
test_sampler = SubsetRandomSampler(test_indices)

batch_size = 4
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

# Model initialization
model = Unet()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fun = torch.nn.MSELoss(reduction="sum")

# Training loop
for epoch in range(300):
    tr_loss = 0
    for i, (inputs, labels) in enumerate(loader_train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, _ = model(inputs)
        loss = loss_fun(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()

    print(f"Epoch [{epoch + 1}/300], Train Loss: {tr_loss / (i + 1):.4f}")

    # Validation loop
    with torch.no_grad():
        val_loss = 0
        for i, (inputs, labels) in enumerate(loader_valid):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores, _ = model(inputs)
            loss = loss_fun(scores, labels)
            val_loss += loss.item()
        val_loss = val_loss / (i + 1)
        print(f"Validation Loss: {val_loss:.4f}")

# Test the model after training
with torch.no_grad():
    val_loss = 0
    for i, (inputs, labels) in enumerate(loader_test):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, _ = model(inputs)
        loss = loss_fun(scores, labels)
        val_loss += loss.item()

    print(f"Test Loss: {val_loss / (i + 1):.4f}")

