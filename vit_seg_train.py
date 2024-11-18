import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the custom dataset
class FaceMapDataset(Dataset):
    def __init__(
        self,
        data_file="data/facemap_softlabels_test.pt",
        transform=None,
        rotation_degrees=15,
        blur_radius=(1, 2),
    ):
        super().__init__()
        self.transform = transform
        self.rotation_degrees = rotation_degrees
        self.blur_radius = blur_radius
        self.data, _, self.targets = torch.load(data_file)

    def __len__(self):
        return len(self.targets) * 5

    def __getitem__(self, index):
        base_index = index // 5
        aug_type = index % 5
        image, label = self.data[base_index].clone(), self.targets[base_index].clone()

        if self.transform is not None:
            if aug_type == 1:
                image = image.flip([2])
                label = label.flip([2])
            elif aug_type == 2:
                angle = (torch.rand(1).item() * 2 - 1) * self.rotation_degrees
                image = TF.rotate(image, angle)
                label = TF.rotate(label, angle)
            elif aug_type == 3:
                scale_factor = 1.1 if torch.rand(1).item() < 0.5 else 0.9
                image = self.zoom(image, scale_factor)
                label = self.zoom(label, scale_factor)
            elif aug_type == 4:
                radius = (
                    torch.rand(1).item() * (self.blur_radius[1] - self.blur_radius[0])
                    + self.blur_radius[0]
                )
                image = TF.gaussian_blur(image, kernel_size=int(radius))
                label = TF.gaussian_blur(label, kernel_size=int(radius))

        return image, label

    def zoom(self, img, scale_factor):
        _, h, w = img.shape
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        img = TF.resize(img, [new_h, new_w])
        img = TF.center_crop(img, [h, w])
        return img


# Load the dataset and create data loaders
dataset = FaceMapDataset(transform=None)
N = len(dataset)
train_sampler = SubsetRandomSampler(np.arange(int(0.6 * N)))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6 * N), int(0.8 * N)))
test_sampler = SubsetRandomSampler(np.arange(int(0.8 * N), N))
batch_size = 4

loader_train = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=0,
    pin_memory=True,
)
loader_valid = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=0,
    pin_memory=True,
)
loader_test = DataLoader(
    dataset=dataset, batch_size=1, sampler=test_sampler, num_workers=0, pin_memory=True
)


# Define the ViT-based segmentation model
class ViT_Segmentation(nn.Module):
    def __init__(self, input_size=224, num_classes=24):
        super(ViT_Segmentation, self).__init__()

        # Convert 1-channel input to 3 channels
        self.input_conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)

        # Initialize the Vision Transformer
        self.encoder = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )

        # Projection layer to reduce the dimensionality
        self.conv_proj = nn.Conv2d(
            in_channels=768, out_channels=256, kernel_size=3, padding=1
        )

        # Segmentation head
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Preprocess the input to match ViT's input dimensions
        x = self.input_conv(x)  # [batch_size, 3, 224, 224]
        b, c, h, w = x.shape

        # Get encoder features
        features = self.encoder.forward_features(x)  # [batch_size, 197, 768]
        print(
            "Feature shape from ViT encoder before removing class token:",
            features.shape,
        )

        # Remove the class token
        features = features[:, 1:]  # Now features should be [batch_size, 196, 768]
        print("Feature shape after removing class token:", features.shape)

        # Reshape features to match expected dimensions
        features = features.permute(0, 2, 1).reshape(b, 768, 14, 14)

        # Apply conv projection to reduce dimensionality
        features = self.conv_proj(features)  # [batch_size, 256, 14, 14]

        # Upsample back to input resolution
        out = F.interpolate(
            features, scale_factor=16, mode="bilinear", align_corners=True
        )
        out = self.segmentation_head(out)  # [batch_size, num_classes, 224, 224]
        return out


# Initialize the model, loss function, and optimizer
num_classes = 24
model = ViT_Segmentation(num_classes=num_classes).to(device)
loss_fun = nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# Training loop
num_epochs = 5
train_loss, valid_loss = [], []
minLoss = float("inf")
patience, convIter = 5, 0

for epoch in range(num_epochs):
    model.train()
    tr_loss = 0
    for i, (inputs, labels) in enumerate(loader_train):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fun(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        tr_loss += loss.item()

    train_loss.append(tr_loss / len(loader_train))

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for inputs, labels in loader_valid:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)
            val_loss += loss.item()

        val_loss /= len(loader_valid)
        valid_loss.append(val_loss)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss[-1]:.4f}, Validation Loss: {valid_loss[-1]:.4f}"
        )

        # Save the model if it improves
        if val_loss < minLoss:
            minLoss = val_loss
            convIter = 0
            torch.save(model.state_dict(), "best_vit_segmentation_model.pth")
        else:
            convIter += 1
            if convIter == patience:
                print("Early stopping at epoch:", epoch + 1)
                break

# Plot training and validation loss
plt.plot(train_loss, label="Training Loss")
plt.plot(valid_loss, label="Validation Loss")
plt.legend()
plt.show()

# Inference on the test set
model.load_state_dict(torch.load("best_vit_segmentation_model.pth"))
model.eval()
with torch.no_grad():
    val_loss = 0
    for i, (inputs, labels) in enumerate(loader_test):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = loss_fun(outputs, labels)
        val_loss += loss.item()

        # Visualization of results
        img = inputs.squeeze().cpu().numpy()
        pred = outputs.squeeze().cpu().numpy()
        labels = labels.squeeze().cpu().numpy()

        plt.clf()
        plt.figure(figsize=(12, 4))
        plt.subplot(141)
        plt.imshow(img[0], cmap="gray")
        plt.subplot(142)
        plt.imshow(labels[0])
        plt.subplot(143)
        plt.imshow(pred[0])
        plt.subplot(144)
        plt.imshow(pred.mean(0), cmap="viridis")

        plt.tight_layout()
        plt.savefig(f"test_preds/test_{i:03d}.jpg")

    print(f"Average Test Loss: {val_loss / len(loader_test):.4f}")
