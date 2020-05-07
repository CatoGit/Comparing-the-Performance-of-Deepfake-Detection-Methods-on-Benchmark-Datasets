import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from albumentations import (
    Compose, FancyPCA, GaussianBlur, GaussNoise, HorizontalFlip,
    HueSaturationValue, ImageCompression, OneOf, PadIfNeeded,
    RandomBrightnessContrast, Resize, ShiftScaleRotate, ToGray)

class UADFVDataset(Dataset):
    """
       UADFV Dataset from Yuezun Li, Ming-Ching Chang, Siwei Lyu 
       (https://arxiv.org/abs/1806.02877)
       
       Implementation: Christopher Otto
    """
    def __init__(self, data,img_size,normalization, augmentations):
        """Dataset constructor."""
        self.data = data
        self.img_size = img_size
        self.augmentations = augmentations
        self.normalization = normalization
        
    def __getitem__(self, idx):
        """Load and return item and label by index."""
        image_row = self.data.iloc[idx]
        image = image_row.loc['video']
        label = image_row.loc['label']
        if label == 1:
            img_path = os.path.join(image)
        else:
            img_path = os.path.join(image)
       
        #load image from path
        try:
            img = cv2.imread(img_path)
        except:
            print(img_path)
        #turn img to rgb color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #apply augmentations to image
        if self.augmentations:
            img = self.augmentations(image=img)['image']
        else:
            #no augmentation during validation or test, just resize to fit DNN input
            augmentations = Resize(width=self.img_size,height=self.img_size)
            img = augmentations(image=img)['image']
        # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
        img = torch.tensor(img).permute(2, 0, 1)
        # turn dtype from uint8 to float and normalize to [0,1] range
        img = img.float() / 255.0
        # normalize 
        if self.normalization == "xception":
            # normalize by xception stats
            transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        elif self.normalization == "imagenet":
            # normalize by imagenet stats
            transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = transform(img)
        return img,label
        
    def __len__(self):
        """Length of dataset."""
        return len(self.data)


class CelebDFDataset(Dataset):
    """
       Celeb-DF Dataset from Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, Siwei Lyu
       (https://arxiv.org/abs/1909.12962)
       
       Implementation: Christopher Otto
    """
    def __init__(self, data,img_size,normalization, augmentations):
        """Dataset constructor."""
        self.data = data
        self.img_size = img_size
        self.augmentations = augmentations
        self.normalization = normalization
        
    def __getitem__(self, idx):
        """Load and return item and label by index."""
        image_row = self.data.iloc[idx]
        image = image_row.loc['video']
        label = image_row.loc['label']
        if label == 1:
            img_path = os.path.join(image)
        else:
            img_path = os.path.join(image)
       
        #load image from path
        try:
            img = cv2.imread(img_path)
        except:
            print(img_path)
        #turn img to rgb color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #apply augmentations to image
        if self.augmentations:
            img = self.augmentations(image=img)['image']
        else:
            #no augmentation during validation or test, just resize to fit DNN input
            augmentations = Resize(width=self.img_size,height=self.img_size)
            img = augmentations(image=img)['image']
        # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
        img = torch.tensor(img).permute(2, 0, 1)
        # turn dtype from uint8 to float and normalize to [0,1] range
        img = img.float() / 255.0
        # normalize 
        if self.normalization == "xception":
            # normalize by xception stats
            transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        elif self.normalization == "imagenet":
            # normalize by imagenet stats
            transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        img = transform(img)
        return img,label
        
    def __len__(self):
        """Length of dataset."""
        return len(self.data)