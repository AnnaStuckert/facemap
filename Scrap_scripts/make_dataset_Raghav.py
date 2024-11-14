import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from skimage import transform,io
import os
import pdb

IMG_LOC=r'C:\Users\avs20\Documents\GitHub\facemap\data\schroeder'

if os.path.isdir(IMG_LOC+'low_res'):
    print('Folder exists!')
else:
    os.makedirs(IMG_LOC+'low_res')

img_files = sorted(glob.glob(os.path.join(IMG_LOC, '*.png')))
labels = pd.read_csv(os.path.join(IMG_LOC, 'labels.csv'))
h = w = 224

img = plt.imread(img_files[0])

h_org = img.shape[0]
w_org = img.shape[1]

### Make new labels for low-res

x_off = (300-224)//2 # (h/h_org*w_org - w) // 2
target = labels.iloc[:,1:].values

target = target*h/h_org # rescale markers 

target[:,::2] = target[:,::2] - x_off
target = torch.Tensor(target)

labels.iloc[:,1:] = target
labels.to_csv(IMG_LOC+'low_res/labels.csv',index=False)

data = torch.zeros((len(img_files),h,w))
print('Resizing images... \nSaving in torch format')

for i in range(len(img_files)):
    im = plt.imread(img_files[i])[:,:,0]
    im_r = (transform.resize(im,(h,w),anti_aliasing=True)*255).astype('uint8')
    data[i] = torch.Tensor(im_r/255.0)
    io.imsave(IMG_LOC+'low_res/'+img_files[i].split('/')[-1],im_r)

torch.save((data,target),IMG_LOC+'low_res/facemap_224.pt')

print('Done! Saved in '+IMG_LOC+'low_res/')

