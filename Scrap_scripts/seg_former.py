import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import timm
import numpy as np
import matplotlib.pyplot as plt
import gc

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceMapDataset(Dataset):
    def __init__(self, data_file="data/facemap_softlabels_test.pt", transform=None):
        super().__init__()
        self.transform = transform
        self.data, _, self.targets = torch.load(data_file)

    def __len__(self):
        return len(self.targets) * 5  # for five versions per image

    def __getitem__(self, index):
        base_index = index // 5
        aug_type = index % 5
        image, label = self.data[base_index].clone(), self.targets[base_index].clone()

        if self.transform is not None:
            if aug_type == 1:  # Flipping
                image = image.flip([2])
                label = label.flip([2])
            elif aug_type == 2:  # Rotation
                angle = (torch.rand(1).item() * 2 - 1) * 15
                image = TF.rotate(image, angle)
                label = TF.rotate(label, angle)
            elif aug_type == 3:  # Zooming
                scale_factor = 1.1 if torch.rand(1).item() < 0.5 else 0.9
                image = self.zoom(image, scale_factor)
                label = self.zoom(label, scale_factor)
            elif aug_type == 4:  # Gaussian Blur
                radius = torch.rand(1).item() + 1
                image = TF.gaussian_blur(image, kernel_size=int(radius))
                label = TF.gaussian_blur(label, kernel_size=int(radius))

        return image, label

    def zoom(self, img, scale_factor):
        _, h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        img = TF.resize(img, [new_h, new_w])
        img = TF.center_crop(img, [h, w])
        return img

# Initialize dataset and data loaders
dataset = FaceMapDataset()
batch_size = 4
train_sampler = SubsetRandomSampler(np.arange(int(0.6 * len(dataset))))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6 * len(dataset)), int(0.8 * len(dataset))))
test_sampler = SubsetRandomSampler(np.arange(int(0.8 * len(dataset)), len(dataset)))

loader_train = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
loader_valid = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)
loader_test = DataLoader(dataset, batch_size=1, sampler=test_sampler)

# Define the segmentation model using a Vision Transformer
class ViTSegmentation(nn.Module):
    def __init__(self, num_classes=1):
        super(ViTSegmentation, self).__init__()
        # Load a ViT model and modify it for segmentation
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=1, num_classes=1)
        self.head = nn.Conv2d(768, num_classes, kernel_size=1)  # Convert transformer outputs to pixel labels

    def forward(self, x):
        # Pass input through the ViT model
        x = self.backbone.patch_embed(x)
        b, _, h, w = x.shape
        x = self.backbone(x)
        
        # Reshape and apply convolution for segmentation output
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2)
        return self.head(x), x

# Initialize model, optimizer, and loss
model = ViTSegmentation(num_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fun = nn.MSELoss(reduction='sum')
num_epochs = 1000
patience = 100

train_loss, valid_loss = [], []
minLoss, convIter = float('inf'), 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    tr_loss = 0
    for inputs, labels in loader_train:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        scores, _ = model(inputs)
        loss = loss_fun(scores, labels)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
    train_loss.append(tr_loss / len(loader_train))

    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for inputs, labels in loader_valid:
            inputs, labels = inputs.to(device), labels.to(device)
            scores, fmap = model(inputs)
            loss = loss_fun(scores, labels)
            val_loss += loss.item()
        val_loss /= len(loader_valid)
        valid_loss.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss[-1]:.4f}, Validation Loss: {val_loss:.4f}")

    if val_loss < minLoss:
        minLoss = val_loss
        convIter = 0
        torch.save(model.state_dict(), 'models/best_model.pt')
    else:
        convIter += 1

    if convIter == patience:
        print(f"Converged at epoch {epoch+1} with validation loss {minLoss:.4f}")
        break

# Testing with best model
model.load_state_dict(torch.load('models/best_model.pt'))
model.eval()
test_loss = 0
for inputs, labels in loader_test:
    inputs, labels = inputs.to(device), labels.to(device)
    with torch.no_grad():
        scores, fmap = model(inputs)
        loss = loss_fun(scores, labels)
        test_loss += loss.item()

    # Visualize results
    img = inputs.squeeze().cpu().numpy()
    pred = scores.squeeze().cpu().numpy()
    labels = labels.squeeze().cpu().numpy()

    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title("Input Image")
    plt.subplot(132), plt.imshow(labels), plt.title("Ground Truth")
    plt.subplot(133), plt.imshow(pred), plt.title("Prediction")
    plt.tight_layout()
    plt.show()

test_loss /= len(loader_test)
print(f"Test Loss: {test_loss:.4f}")
