import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import TensorDataset, DataLoader, Dataset, SubsetRandomSampler
import pdb
import glob
import pandas as pd
from numpy import random
import numpy as np
import timm
import matplotlib.pyplot as plt
from models import Unet

torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class facemapdataset(Dataset):
     def __init__(self, 
             data_file = 'data/facemap_softlabels.pt',
             transform=None):
          super().__init__()

          self.transform = transform
          self.data, _, self.targets = torch.load(data_file)

     def __len__(self):
          return len(self.targets)

     def __getitem__(self, index):
          image, label = self.data[index].clone(), self.targets[index].clone()
          if (self.transform is not None) and (torch.rand(1) > 0.5):
               image = image.flip([2])
               label = label.flip([2])
          return image, label

### Make dataset
dataset  = facemapdataset(transform='flip')

x = dataset[0][0]
dim = x.shape[-1]
print('Using %d size of images'%dim)
N = len(dataset)
train_sampler = SubsetRandomSampler(np.arange(int(0.6*N)))
valid_sampler = SubsetRandomSampler(np.arange(int(0.6*N),int(0.8*N)))
test_sampler = SubsetRandomSampler(np.arange(int(0.8*N),N))
batch_size = 4
# Initialize loss and metrics
loss_fun = torch.nn.MSELoss(reduction='sum')

# Initiliaze input dimensions
num_train = len(train_sampler)
num_valid = len(valid_sampler)
num_test = len(test_sampler)
print("Num. train = %d, Num. val = %d, Num. test = %d"%(num_train,num_valid,num_test))

# Initialize dataloaders
loader_train = DataLoader(dataset = dataset, drop_last=False,num_workers=0,
        batch_size=batch_size, pin_memory=True,sampler=train_sampler)
loader_valid = DataLoader(dataset = dataset, drop_last=True,num_workers=0,
        batch_size=batch_size, pin_memory=True,sampler=valid_sampler)
loader_test = DataLoader(dataset = dataset, drop_last=True,num_workers=0,
        batch_size=1, pin_memory=True,sampler=test_sampler)

nValid = len(loader_valid)
nTrain = len(loader_train)
nTest = len(loader_test)

### hyperparam
lr = 5e-4
num_epochs = 500

num_input_channels = 1  # Change this to the desired number of input channels
num_output_classes = 24  # Change this to the desired number of output classes


model = Unet()
#timm.create_model('vit_base_patch8_224',
#        pretrained=True,in_chans=1,num_classes=num_output_classes)

model = model.to(device)
nParam = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters:%2f M"%(nParam/1e6))

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
minLoss = 1e6
convIter = 0
patience = 50
train_loss = []
valid_loss = []

for epoch in range(num_epochs):
    tr_loss = 0
    for i, (inputs,labels) in enumerate(loader_train):
        inputs = inputs.to(device)
        labels = labels.to(device)
        scores, _ = (model(inputs))

        loss = loss_fun((scores),((labels)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, nTrain, loss.item()))
        tr_loss += loss.item()
    train_loss.append(tr_loss/(i+1))

    with torch.no_grad():
        val_loss = 0
        for i, (inputs,labels) in enumerate(loader_valid):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores, fmap = (model(inputs))
            loss = loss_fun((scores),((labels)))
            val_loss += loss.item()
        val_loss = val_loss/(i+1)
        
        valid_loss.append(val_loss)

        print('Val. loss :%.4f'%val_loss)
        
        labels = labels.squeeze().detach().cpu().numpy()
        scores = scores.squeeze().detach().cpu().numpy()
        img = inputs.squeeze().detach().cpu().numpy()
        fmap = inputs.mean(1).squeeze().detach().cpu().numpy()

        plt.clf()
        plt.figure(figsize=(16,12))
        for i in range(batch_size):
            plt.subplot(batch_size,3,3*i+1)
            plt.imshow(labels[i])
            plt.subplot(batch_size,3,3*i+2)
            plt.imshow(scores[i]*img[i])
            plt.subplot(batch_size,3,3*i+3)
            plt.imshow(fmap[i])

        plt.tight_layout()

        plt.savefig('logs/epoch_%03d.jpg'%epoch)
            
        if minLoss > val_loss:
            convEpoch = epoch
            minLoss = val_loss
            convIter = 0
            #torch.save(model.state_dict(),'models/best_model.pt')
        else:
            convIter += 1

        if convIter == patience:
            print('Converged at epoch %d with val. loss %.4f'%(convEpoch+1,minLoss))
            break
plt.clf()
plt.plot(train_loss,label='Training')
plt.plot(valid_loss,label='Valid')
plt.plot(convEpoch,valid_loss[convEpoch],'x',label='Final Model')
plt.legend()
plt.tight_layout()
plt.savefig('loss_curve.pdf')

if 0:
    ### Load best model for inference
    with torch.no_grad():
        val_loss = 0

        for i, (inputs,labels) in enumerate(loader_test):
            inputs = inputs.to(device)
            labels = labels.to(device)
            scores,_ = (model(inputs))
            loss = loss_fun((scores),((labels)))
            val_loss += loss.item()

            img = inputs.squeeze().detach().cpu().numpy()
            pred = scores.squeeze().detach().cpu().numpy()
            labels = labels.squeeze().cpu().numpy()
            plt.clf()
            plt.imshow(img,cmap='gray')
            plt.plot(pred[::2],pred[1::2],'x',c='tab:red')
            plt.plot(labels[::2],labels[1::2],'o',c='tab:green')
            plt.tight_layout()
            plt.savefig('preds/test_%03d.jpg'%i)

        val_loss = val_loss/(i+1)
        

        print('Test. loss :%.4f'%val_loss)
        

