import torch
from torchvision.datasets import CIFAR10, CIFAR100,STL10
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import os
import numpy as np


class TinyImagenet(Dataset):
    def __init__(self,root = None,withLabel=False,labelSubSet=None,labelTrans=None,transform=None,**kwds):
        self.transform = transform
        self.data = datasets.ImageFolder(root)
        self.withLabel = withLabel
        self.labelTrans = labelTrans
        if labelSubSet is not None:
            self.data = self.data[labelSubSet]
    def __getitem__(self, idx):
        # if not self.train:
        #     return super().__getitem__(idx)

        img = self.data[idx][0]
        label = self.data[idx][1]
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            imgLabelTrans = self.labelTrans(img)
            return torch.stack(imgs), imgLabelTrans, label
        
    def __len__(self,):
        return len(self.data.imgs)

class CustomSTL10(STL10):
    def __init__(self, root, withLabel=False, labelSubSet=None, labelTrans=None, transform=None, **kwds):
        root = os.path.join(root, 'stl10','train')
        super().__init__(root=root, split='unlabeled', download=True, transform=None, **kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans
        self.transform = transform
        
        if labelSubSet is not None:
            self.data = self.data[labelSubSet]
            if self.labels is not None:
                self.labels = self.labels[labelSubSet]
    
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert('RGB')
        # img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        
    

class CustomImagenet100(Dataset):
    def __init__(self, data_path,withLabel=False, labelSubSet=None, labelTrans=None, transform=None, **kwds):
        self.transform = transform
        self.data = datasets.ImageFolder(os.path.join(data_path,"train"))
        self.withLabel = withLabel
        self.labelTrans = labelTrans
        if labelSubSet is not None:
            self.data = self.data[labelSubSet]

    def __getitem__(self, idx):
        img, label = self.data[idx]
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[idx]
            return torch.stack(imgs), imgLabelTrans, label

    def __len__(self):
        return len(self.data.imgs)

class CustomCIFAR10(CIFAR10):
    def __init__(self, withLabel=False, labelSubSet=None, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans
        if labelSubSet is not None:
            self.data = self.data[labelSubSet]

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[idx]
            return torch.stack(imgs), imgLabelTrans, label


class CustomCIFAR100(CIFAR100):
    def __init__(self, withLabel=False, labelTrans=None, **kwds):
        super().__init__(**kwds)
        self.withLabel = withLabel
        self.labelTrans = labelTrans

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        if not self.withLabel:
            return torch.stack(imgs)
        else:
            imgLabelTrans = self.labelTrans(img)
            label = self.targets[idx]
            return torch.stack(imgs), imgLabelTrans, label

