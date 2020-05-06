# result = dfdetector.detect(video, method, optional->facedetector->default retinafce) --> detects whether file is real/fake
# result = dfdetector.benchmark(dataset,method) ->benchmarks method on datasets test data
import argparse
import copy
import os
import shutil
import test
import time
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import ShuffleSplit
from torch.optim import Adam, lr_scheduler

import cv2
import datasets
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from albumentations import Compose, HorizontalFlip, Resize
from pretrained_mods import xception
from tqdm import tqdm


class DFDetector():
    def __init__(self, facedetector="retinaface_resnet", visuals=False):
        self.facedetector = facedetector
        self.visuals = visuals

    @classmethod
    def detect(cls, video=None,  method="xception"):
        return result

    @classmethod
    def benchmark(cls, dataset=None, data_path=None, method="xception"):
        """Benchmark deepfake detection methods against popular deepfake datasets.
           The methods are already pretrained on the datasets and benchmarked against a
           test set that is distinct from the training data.
        # Arguments:
            dataset: The dataset that the method is tested against.
            data_path: The path to the test videos.
            method: The deepfake detection method that is used.
        # Implementation: Christopher Otto
        """
        if method not in ['xception']:
            raise ValueError("Method is not available for benchmarking.")
        else:
            cls.method = method
            cls.data_path = data_path
            if cls.method == "xception":
                img_size = 299
                # no augs because of testing
                augmentations = None
                normalization = "xception"
            else:
                img_size = None
        if dataset == 'uadfv':
            if cls.data_path is None:
                raise ValueError("""Please go to https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi
                                and scroll down to the UADFV section.
                                Click on the link \"here\" and download the dataset. 
                                Extract the files and organize the folders follwing this folder structure:
                                ./fake_videos/
                                            fake/
                                            real/
                                """)
            if cls.data_path.endswith("fake_videos.zip"):
                raise ValueError("Please make sure to extract the zipfile.")
            if cls.data_path.endswith("fake_videos"):
                print(f"Benchmarking \033[1m{cls.method}\033[0m on the UADFV dataset with ...")
                # create test directories if they don't exist
                if not os.path.exists(data_path + '/test/'):
                    structure_uadfv_files(
                        files_needed_csv="uadfv_test.csv", path_to_data=data_path)
                else:
                    # check if path exists but files are not complete (from https://stackoverflow.com/a/2632251)
                    num_files = len([f for f in os.listdir(
                        data_path + '/test/') if os.path.isfile(os.path.join(data_path + '/test/real/', f))])
                    num_files += len([f for f in os.listdir(data_path + '/test/')
                                      if os.path.isfile(os.path.join(data_path + '/test/fake/', f))])
                    # check whether all 28 test videos are in directories
                    if num_files != 28:
                        # recreate all 28 files
                        shutil.rmtree(data_path + '/test/')
                        structure_uadfv_files(
                            files_needed_csv="uadfv_test.csv", path_to_data=data_path)

                # get test labels for metric evaluation
                df = label_data(dataset_path=cls.data_path, test_data=True)
                if img_size is None:
                    print("Please specify the DNN input image size.")
                # uadfv test dataset
                #cls.dataset = datasets.UADFVDataset(df,img_size,normalization,augmentations)
                model = xception.imagenet_pretrained_xception()
                # load the xception model that was pretrained on the uadfv training data
                model_params = torch.load(
                    './deepfake_detector/pretrained_mods/weights/xception_best_fulltrain_UADFV.pth')
                model.load_state_dict(model_params)
                print(f"Detect deepfakes with \033[1m{cls.method}\033[0m ...")
                auc, ap, loss, acc = test.inference(model, df, img_size)
            else:
                raise ValueError("""Please organize the dataset directory in this way:
                                    ./fake_videos/
                                                fake/
                                                real/
                                """)

        else:
            raise ValueError(f"{dataset} does not exist.")
        return [auc, ap, loss, acc]


def label_data(dataset_path=None, test_data=False):
    """
    Label the data.
    # Arguments:
        dataset_path: path to data
        test_data: binary choice that indicates whether data is for testing or not.
    # Implementation: Christopher Otto
    """
    # structure data from folder in data frame for loading
    if dataset_path is None:
        # TEST
        raise ValueError("Please specify a dataset path.")
    # TEST
    if not test_data:
        # prepare training data
        video_path_real = os.path.join(dataset_path + "/real/")
        video_path_fake = os.path.join(dataset_path + "/fake/")

        # add labels to videos
        data_list = []
        for _, _, videos in os.walk(video_path_real):
            for video in tqdm(videos):
                # label 0 for real video
                data_list.append(
                    {'label': 0, 'image': video_path_real + video})

        for _, _, videos in os.walk(video_path_fake):
            for video in tqdm(videos):
                # label 1 for deepfake video
                data_list.append(
                    {'label': 1, 'image': video_path_fake + video})
    else:
        # prepare test data
        video_path_test_real = os.path.join(dataset_path + "/test/real/")
        video_path_test_fake = os.path.join(dataset_path + "/test/fake/")
        data_list = []
        for _, _, videos in os.walk(video_path_test_real):
            for video in tqdm(videos):
                # append test video
                data_list.append(
                    {'label': 0, 'video': video_path_test_real + video})

        for _, _, videos in os.walk(video_path_test_fake):
            for video in tqdm(videos):
                # label 1 for deepfake image
                data_list.append(
                    {'label': 1, 'video': video_path_test_fake + video})
    # put data into dataframe
    df = pd.DataFrame(data=data_list)
    print(f"{len(df)} test videos.")
    print()
    return df


def df_augmentations(strengh="weak"):
    """
    Augmentations with the albumentations package.
    # Arguments:
        strength: strong or weak augmentations

    # Implementation: Christopher Otto
    """
    if strength == "weak":
        augs = Compose([
            # hflip with prob 0.5
            HorizontalFlip(p=0.5),
            # adjust image to DNN input size
            Resize(width=img_size, height=img_size)
        ])
        return augs
    else:
        # augmentations via albumentations package
        # augmentations similar to 3rd place private leaderboard solution of
        # https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721
        augs = Compose([
            # hflip with prob 0.5
            HorizontalFlip(p=0.5),
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            # IsotropicResize(max_side=size),
            PadIfNeeded(min_height=img_size, min_width=img_size,
                        border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(),
                   HueSaturationValue()], p=0.7),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                             rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            # adjust image to DNN input size
            Resize(width=img_size, height=img_size)
        ])
        return augs


def structure_uadfv_files(files_needed_csv, path_to_data):
    """Creates test folders and moves test videos there."""
    os.mkdir(path_to_data + '/test/')
    os.mkdir(path_to_data + '/test/fake/')
    os.mkdir(path_to_data + '/test/real/')
    test_data = pd.read_csv(
        "./deepfake_detector/data/uadfv_test.csv", names=['video'], header=None)
    for idx, row in test_data.iterrows():
        if len(str(row.loc['video'])) > 8:
            # video is fake, therefore copy it into fake test folder
            shutil.copy(path_to_data + '/fake/' +
                        row['video'], path_to_data + '/test/fake/')
        else:
            # video is real, therefore move it into real test folder
            shutil.copy(path_to_data + '/real/' +
                        row['video'], path_to_data + '/test/real/')
#result = DFDetector.detect(video=video_file, method="xception", heatmap=False)

#benchmark_result = DFDetector.benchmark(dataset="uadfv", method="xception", compare=False)

# ->compare: whether to compare with the results of other methods that were precomputed by myself