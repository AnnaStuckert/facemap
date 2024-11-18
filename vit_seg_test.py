import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import timm
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FaceMapDataset(Dataset):
    def __init__(self, data_file='data/facemap_softlabels.pt', transform=None):
        super().__init__()
        self.transform = transform
        self.data, _, self.targets = torch.load(data_file)
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image, label = self.data[index].clone(), self.targets[index].clone()
        if self.transform and torch.rand(1) > 0.5:
            image = image.flip([2])
            label = label.flip([2])
        return image, label

### Dataset and DataLoader Setup
dataset = FaceMapDataset()
batch_size = 4
train_sampler = SubsetRandomSampler(np.arange(int(0.6 * len(dataset))))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6 * len(dataset)), int(0.8 * len(dataset))))
test_sampler = SubsetRandomSampler(np.arange(int(0.8 * len(dataset)), len(dataset)))

loader_train = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True)
loader_valid = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=True)
loader_test = DataLoader(dataset, batch_size=1, sampler=test_sampler, pin_memory=True)

### Model Setup
model = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=1, num_classes=224*224)
model = model.to(device)

### Adjust Model Output Shape
class SegmentationHead(nn.Module):
    def __init__(self, in_features, out_channels=1, img_size=224):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv2d(in_features, out_channels, kernel_size=1)
        self.img_size = img_size
    
    def forward(self, x):
        x = x.view(-1, self.img_size, self.img_size)
        return x.unsqueeze(1)  # Add channel dimension for segmentation map

segmentation_head = SegmentationHead(224*224)
model.head = segmentation_head

### Loss and Optimizer
loss_fn = nn.BCEWithLogitsLoss()  # Use BCE for binary segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

### Training Loop
num_epochs = 100
min_loss = float('inf')
patience = 10
train_losses, valid_losses = [], []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, labels in loader_train:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = segmentation_head(outputs)
        loss = loss_fn(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(loader_train)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for inputs, labels in loader_valid:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            outputs = segmentation_head(outputs)
            loss = loss_fn(outputs, labels)
            valid_loss += loss.item()
    
    valid_loss /= len(loader_valid)
    valid_losses.append(valid_loss)

    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
    
    # Early stopping
    if valid_loss < min_loss:
        min_loss = valid_loss
        patience = 10
        torch.save(model.state_dict(), 'best_segmentation_model.pt')
    else:
        patience -= 1
        if patience == 0:
            print("Early stopping")
            break

### Plot Training and Validation Loss
plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.legend()
plt.savefig('segmentation_loss_curve.pdf')

### Testing and Visualization
model.load_state_dict(torch.load('best_segmentation_model.pt'))
model.eval()
with torch.no_grad():
    for i, (inputs, labels) in enumerate(loader_test):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        outputs = segmentation_head(outputs).squeeze(1).cpu().numpy()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(inputs[0].squeeze().cpu(), cmap='gray')
        plt.title("Input Image")
        
        plt.subplot(1, 3, 2)
        plt.imshow(labels[0].squeeze().cpu(), cmap='gray')
        plt.title("Ground Truth")
        
        plt.subplot(1, 3, 3)
        plt.imshow(outputs[0], cmap='hot')
        plt.title("Predicted Heatmap")
        
        plt.savefig(f'predictions/test_{i}.jpg')
        plt.close()
