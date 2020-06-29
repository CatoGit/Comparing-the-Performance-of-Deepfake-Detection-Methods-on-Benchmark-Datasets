import os
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

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

    def __init__(self, data, img_size, method, normalization, augmentations):
        """Dataset constructor."""
        self.data = data
        self.img_size = img_size
        self.method = method
        self.normalization = normalization
        self.augmentations = augmentations

    def __getitem__(self, idx):
        """Load and return item and label by index."""
        image_row = self.data.iloc[idx]
        # get label
        label = image_row.loc['label']
        if self.method == 'resnet_lstm' or self.method == 'efficientnetb1_lstm':
            imgs = []
            # get name of video that sequence is from
            image = image_row.loc['original']
            # load 20 frames from video in correct order
            for i in range(20):
                # load image from path by position in sequence
                if label == 1:
                    img_path = os.path.join(image + '_' + str(i) + '.jpg')
                else:
                    img_path = os.path.join(image + '_' + str(i) + '.jpg')
                try:
                    img = cv2.imread(img_path)
                except:
                    print(img_path)
                # turn img to rgb color
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # apply augmentations to image
                if self.augmentations:
                    img = self.augmentations(image=img)['image']
                else:
                    # no augmentation during validation or test, just resize to fit DNN input
                    augmentations = Resize(
                        width=self.img_size, height=self.img_size)
                    img = augmentations(image=img)['image']
                # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
                img = torch.tensor(img).permute(2, 0, 1)
                # turn dtype from uint8 to float and normalize to [0,1] range
                img = img.float() / 255.0
                # normalize
                if self.normalization == "xception":
                    # normalize by xception stats
                    transform = transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                elif self.normalization == "imagenet":
                    # normalize by imagenet stats
                    transform = transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img = transform(img)
                imgs.append(img.numpy())
            # return image sequence and corresponding label
            return np.array(imgs), label
        else:
            # load image from video
            image = image_row.loc['video']
            if label == 1:
                img_path = os.path.join(image)
            else:
                img_path = os.path.join(image)
            # load image from path
            try:
                img = cv2.imread(img_path)
            except:
                print(img_path)
            # turn img to rgb color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # apply augmentations to image
            if self.augmentations:
                img = self.augmentations(image=img)['image']
            else:
                # no augmentation during validation or test, just resize to fit DNN input
                augmentations = Resize(
                    width=self.img_size, height=self.img_size)
                img = augmentations(image=img)['image']
            # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
            img = torch.tensor(img).permute(2, 0, 1)
            # turn dtype from uint8 to float and normalize to [0,1] range
            img = img.float() / 255.0
            # normalize
            if self.normalization == "xception":
                # normalize by xception stats
                transform = transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            elif self.normalization == "imagenet":
                # normalize by imagenet stats
                transform = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img = transform(img)
            # return image and label
            return img, label

    def __len__(self):
        """Length of dataset."""
        return len(self.data)


class CelebDFDataset(Dataset):
    """
       Celeb-DF Dataset from Yuezun Li, Xin Yang, Pu Sun, Honggang Qi, Siwei Lyu
       (https://arxiv.org/abs/1909.12962)

       Implementation: Christopher Otto
    """

    def __init__(self, data, img_size, method, normalization, augmentations):
        """Dataset constructor."""
        self.data = data
        self.img_size = img_size
        self.method = method
        self.augmentations = augmentations
        self.normalization = normalization

    def __getitem__(self, idx):
        """Load and return item and label by index."""
        image_row = self.data.iloc[idx]

        label = image_row.loc['label']
        if self.method == 'resnet_lstm' or self.method == 'efficientnetb1_lstm':
            imgs = []
            # get name of video that sequence is from
            image = image_row.loc['original']
            # load 20 frames from video in correct order
            for i in range(20):
                # load image from path by position in sequence
                if label == 1:
                    img_path = os.path.join(image + '_' + str(i) + '.jpg')
                else:
                    img_path = os.path.join(image + '_' + str(i) + '.jpg')
                try:
                    img = cv2.imread(img_path)
                except:
                    print(img_path)
                # turn img to rgb color
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # apply augmentations to image
                if self.augmentations:
                    img = self.augmentations(image=img)['image']
                else:
                    # no augmentation during validation or test, just resize to fit DNN input
                    augmentations = Resize(
                        width=self.img_size, height=self.img_size)
                    img = augmentations(image=img)['image']
                # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
                img = torch.tensor(img).permute(2, 0, 1)
                # turn dtype from uint8 to float and normalize to [0,1] range
                img = img.float() / 255.0
                # normalize
                if self.normalization == "xception":
                    # normalize by xception stats
                    transform = transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                elif self.normalization == "imagenet":
                    # normalize by imagenet stats
                    transform = transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img = transform(img)
                imgs.append(img.numpy())
            # return image sequence and corresponding label
            return np.array(imgs), label
        else:
            # load image from video
            image = image_row.loc['video']
            if label == 1:
                img_path = os.path.join(image)
            else:
                img_path = os.path.join(image)
            # load image from path
            try:
                img = cv2.imread(img_path)
            except:
                print(img_path)
            # turn img to rgb color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # apply augmentations to image
            if self.augmentations:
                img = self.augmentations(image=img)['image']
            else:
                # no augmentation during validation or test, just resize to fit DNN input
                augmentations = Resize(
                    width=self.img_size, height=self.img_size)
                img = augmentations(image=img)['image']
            # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
            img = torch.tensor(img).permute(2, 0, 1)
            # turn dtype from uint8 to float and normalize to [0,1] range
            img = img.float() / 255.0
            # normalize
            if self.normalization == "xception":
                # normalize by xception stats
                transform = transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            elif self.normalization == "imagenet":
                # normalize by imagenet stats
                transform = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img = transform(img)
            return img, label

    def __len__(self):
        """Length of dataset."""
        return len(self.data)


class DFTIMITHQDataset(Dataset):
    """
       Deepfake Timit HQ dataset from P. Korshunov and S. Marcel,
       DeepFakes: a New Threat to Face Recognition? Assessment and Detection.
       (https://arxiv.org/abs/1812.08685)

       Implementation: Christopher Otto
    """

    def __init__(self, data, img_size, method, normalization, augmentations):
        """Dataset constructor."""
        self.data = data
        self.img_size = img_size
        self.method = method
        self.augmentations = augmentations
        self.normalization = normalization

    def __getitem__(self, idx):
        """Load and return item and label by index."""
        image_row = self.data.iloc[idx]

        label = image_row.loc['label']
        if self.method == 'resnet_lstm' or self.method == 'efficientnetb1_lstm':
            imgs = []
            # get name of video that sequence is from
            image = image_row.loc['original']
            # load 20 frames from video in correct order
            for i in range(20):
                # load image from path by position in sequence
                if label == 1:
                    img_path = os.path.join(image + '_' + str(i) + '.jpg')
                else:
                    img_path = os.path.join(image + '_' + str(i) + '.jpg')
                try:
                    img = cv2.imread(img_path)
                except:
                    print(img_path)
                # turn img to rgb color
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # apply augmentations to image
                if self.augmentations:
                    img = self.augmentations(image=img)['image']
                else:
                    # no augmentation during validation or test, just resize to fit DNN input
                    augmentations = Resize(
                        width=self.img_size, height=self.img_size)
                    img = augmentations(image=img)['image']
                # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
                img = torch.tensor(img).permute(2, 0, 1)
                # turn dtype from uint8 to float and normalize to [0,1] range
                img = img.float() / 255.0
                # normalize
                if self.normalization == "xception":
                    # normalize by xception stats
                    transform = transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                elif self.normalization == "imagenet":
                    # normalize by imagenet stats
                    transform = transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img = transform(img)
                imgs.append(img.numpy())
            # return image sequence and corresponding label
            return np.array(imgs), label
        else:
            # load image from video
            image = image_row.loc['video']
            if label == 1:
                img_path = os.path.join(image)
            else:
                img_path = os.path.join(image)
            # load image from path
            try:
                img = cv2.imread(img_path)
            except:
                print(img_path)
            # turn img to rgb color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # apply augmentations to image
            if self.augmentations:
                img = self.augmentations(image=img)['image']
            else:
                # no augmentation during validation or test, just resize to fit DNN input
                augmentations = Resize(
                    width=self.img_size, height=self.img_size)
                img = augmentations(image=img)['image']
            # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
            img = torch.tensor(img).permute(2, 0, 1)
            # turn dtype from uint8 to float and normalize to [0,1] range
            img = img.float() / 255.0
            # normalize
            if self.normalization == "xception":
                # normalize by xception stats
                transform = transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            elif self.normalization == "imagenet":
                # normalize by imagenet stats
                transform = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img = transform(img)
            return img, label

    def __len__(self):
        """Length of dataset."""
        return len(self.data)


class DFTIMITLQDataset(Dataset):
    """
       Deepfake Timit LQ dataset from P. Korshunov and S. Marcel,
       DeepFakes: a New Threat to Face Recognition? Assessment and Detection.
       (https://arxiv.org/abs/1812.08685)

       Implementation: Christopher Otto
    """

    def __init__(self, data, img_size, method, normalization, augmentations):
        """Dataset constructor."""
        self.data = data
        self.img_size = img_size
        self.method = method
        self.augmentations = augmentations
        self.normalization = normalization

    def __getitem__(self, idx):
        """Load and return item and label by index."""
        image_row = self.data.iloc[idx]

        label = image_row.loc['label']
        if self.method == 'resnet_lstm' or self.method == 'efficientnetb1_lstm':
            imgs = []
            # get name of video that sequence is from
            image = image_row.loc['original']
            # load 20 frames from video in correct order
            for i in range(20):
                # load image from path by position in sequence
                if label == 1:
                    img_path = os.path.join(image + '_' + str(i) + '.jpg')
                else:
                    img_path = os.path.join(image + '_' + str(i) + '.jpg')
                try:
                    img = cv2.imread(img_path)
                except:
                    print(img_path)
                # turn img to rgb color
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # apply augmentations to image
                if self.augmentations:
                    img = self.augmentations(image=img)['image']
                else:
                    # no augmentation during validation or test, just resize to fit DNN input
                    augmentations = Resize(
                        width=self.img_size, height=self.img_size)
                    img = augmentations(image=img)['image']
                # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
                img = torch.tensor(img).permute(2, 0, 1)
                # turn dtype from uint8 to float and normalize to [0,1] range
                img = img.float() / 255.0
                # normalize
                if self.normalization == "xception":
                    # normalize by xception stats
                    transform = transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                elif self.normalization == "imagenet":
                    # normalize by imagenet stats
                    transform = transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img = transform(img)
                imgs.append(img.numpy())
            # return image sequence and corresponding label
            return np.array(imgs), label
        else:
            # load image from video
            image = image_row.loc['video']
            if label == 1:
                img_path = os.path.join(image)
            else:
                img_path = os.path.join(image)
            # load image from path
            try:
                img = cv2.imread(img_path)
            except:
                print(img_path)
            # turn img to rgb color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # apply augmentations to image
            if self.augmentations:
                img = self.augmentations(image=img)['image']
            else:
                # no augmentation during validation or test, just resize to fit DNN input
                augmentations = Resize(
                    width=self.img_size, height=self.img_size)
                img = augmentations(image=img)['image']
            # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
            img = torch.tensor(img).permute(2, 0, 1)
            # turn dtype from uint8 to float and normalize to [0,1] range
            img = img.float() / 255.0
            # normalize
            if self.normalization == "xception":
                # normalize by xception stats
                transform = transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            elif self.normalization == "imagenet":
                # normalize by imagenet stats
                transform = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img = transform(img)
            return img, label

    def __len__(self):
        """Length of dataset."""
        return len(self.data)
    

class DFDCDataset(Dataset):
    """
       Deepfake Detection Challenge Dataset (DFDC) from Brian Dolhansky, Russ Howes, Ben Pflaum, Nicole Baram, Cristian Canton Ferrer
       The Deepfake Detection Challenge (DFDC) Preview Dataset
       (https://arxiv.org/abs/1910.08854)

       Implementation: Christopher Otto
    """

    def __init__(self, data, img_size, method, normalization, augmentations):
        """Dataset constructor."""
        self.data = data
        self.img_size = img_size
        self.method = method
        self.augmentations = augmentations
        self.normalization = normalization

    def __getitem__(self, idx):
        """Load and return item and label by index."""
        image_row = self.data.iloc[idx]
        label = image_row.loc['label']
        if self.method == 'resnet_lstm' or self.method == 'efficientnetb1_lstm':
            imgs = []
            # get name of video that sequence is from
            image = image_row.loc['original']
            # load 5 frames from video in correct order
            for i in range(5):
                # load image from path by position in sequence
                if label == 1:
                    img_path = os.path.join(image + '_' + str(i) + '.jpg')
                else:
                    img_path = os.path.join(image + '_' + str(i) + '.jpg')
                try:
                    img = cv2.imread(img_path)
                except:
                    print(img_path)
                # turn img to rgb color
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # apply augmentations to image
                if self.augmentations:
                    img = self.augmentations(image=img)['image']
                else:
                    # no augmentation during validation or test, just resize to fit DNN input
                    augmentations = Resize(
                        width=self.img_size, height=self.img_size)
                    img = augmentations(image=img)['image']
                # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
                img = torch.tensor(img).permute(2, 0, 1)
                # turn dtype from uint8 to float and normalize to [0,1] range
                img = img.float() / 255.0
                # normalize
                if self.normalization == "xception":
                    # normalize by xception stats
                    transform = transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                elif self.normalization == "imagenet":
                    # normalize by imagenet stats
                    transform = transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                img = transform(img)
                imgs.append(img.numpy())
            # return image sequence and corresponding label
            return np.array(imgs), label
        else:
            # load image from video
            image = image_row.loc['video']
            if label == 1:
                img_path = os.path.join(image)
            else:
                img_path = os.path.join(image)
            # load image from path
            try:
                img = cv2.imread(img_path)
            except:
                print(img_path)
            # turn img to rgb color
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # apply augmentations to image
            if self.augmentations:
                img = self.augmentations(image=img)['image']
            else:
                # no augmentation during validation or test, just resize to fit DNN input
                augmentations = Resize(
                    width=self.img_size, height=self.img_size)
                img = augmentations(image=img)['image']
            # turn into tensor and switch to channels first, i.e. (3,img_size,img_size)
            img = torch.tensor(img).permute(2, 0, 1)
            # turn dtype from uint8 to float and normalize to [0,1] range
            img = img.float() / 255.0
            # normalize
            if self.normalization == "xception":
                # normalize by xception stats
                transform = transforms.Normalize(
                    [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            elif self.normalization == "imagenet":
                # normalize by imagenet stats
                transform = transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            img = transform(img)
            return img, label

    def __len__(self):
        """Length of dataset."""
        return len(self.data)

