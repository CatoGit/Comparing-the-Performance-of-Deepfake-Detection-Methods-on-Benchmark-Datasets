import argparse
import copy
import os
import shutil
import test
import time
import zipfile
import timm
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import metrics
import matplotlib.pyplot as plt
import cv2
import datasets
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import train
import utils
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from albumentations import (
    Compose, FancyPCA, GaussianBlur, GaussNoise, HorizontalFlip,
    HueSaturationValue, ImageCompression, OneOf, PadIfNeeded,
    RandomBrightnessContrast, Resize, ShiftScaleRotate, ToGray)
from pretrained_mods import xception
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from facedetector.retinaface import df_retinaface
from pretrained_mods import efficientnetb1lstm
from pretrained_mods import mesonet
from pretrained_mods import resnetlstm
from utils import vidtimit_setup_real_videos



parser = argparse.ArgumentParser(
    description='Start deepfake detection.')
parser.add_argument('--detect_single', default=False,
                    type=bool, help='Choose for single prediction.')
parser.add_argument('--benchmark', default=False, type=bool,
                    help='Choose for benchmarking.')
parser.add_argument('--train', default=False, type=bool,
                    help='Choose for training.')
parser.add_argument('--path_to_vid', default=None,
                    type=str, help='Choose video path.')
parser.add_argument('--path_to_img', default=None,
                    type=str, help='Choose image path.')
parser.add_argument('--detection_method', default="xception_uadfv",
                    type=str, help='Choose detection method.')
parser.add_argument('--data_path', default=None, type=str,
                    help='Specify path to dataset.')
parser.add_argument('--dataset', default="uadfv", type=str,
                    help='Specify the name of the dataset.')
parser.add_argument('--cmd', default="True", type=str,
                    help='True if executed via command line.')
parser.add_argument('--model_type', default="xception",
                    type=str, help='Choose detection model type for training.')
parser.add_argument('--epochs', default=1,
                    type=int, help='Choose number of training epochs.')
parser.add_argument('--batch_size', default=32,
                    type=int, help='Choose the minibatch size.')
parser.add_argument('--lr', default=0.0001,
                    type=int, help='Choose the minibatch size.')
parser.add_argument('--folds', default=1,
                    type=int, help='Choose validation folds.')
parser.add_argument('--augs', default="weak",
                    type=str, help='Choose augmentation strength.')
parser.add_argument('--fulltrain', default=False,
                    type=bool, help='Choose whether to train with the full dataset and no validation set.')
parser.add_argument('--facecrops_available', default=False,
                    type=bool, help='Choose whether videos are already preprocessed.')
parser.add_argument('--face_margin', default=0.3,
                    type=float, help='Choose the face margin.')
parser.add_argument('--seed', default=24,
                    type=int, help='Choose the random seed.')
parser.add_argument('--save_path', default=None,
                    type=str, help='Choose the path where face crops shall be saved.')                       
                                                             
                                      


class DFDetector():
    """
    The Deepfake Detector. 
    It can detect on a single video, 
    benchmark several methods on benchmark datasets
    and train detectors on several datasets.
    """

    def __init__(self):
        pass

    @classmethod
    def detect_single(cls, video_path=None, image_path=None, label=None, method="xception_uadfv", cmd=False):
        """Perform deepfake detection on a single video with a chosen method."""
        # prepare the method of choice
        sequence_model = False
        if method == "xception_uadfv":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "Xception_UADFV"
        elif method == "xception_celebdf":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "Xception_CELEB-DF"
        elif method == "xception_dfdc":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "Xception_DFDC"
        elif method == "xception_dftimit_hq":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "Xception_DF-TIMIT-HQ"
        elif method == "xception_dftimit_lq":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "Xception_DF-TIMIT-LQ"
        elif method == "efficientnetb7_uadfv":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B7_UADFV"
        elif method == "efficientnetb7_celebdf":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B7_CELEB-DF"
        elif method == "efficientnetb7_dfdc":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B7_DFDC"
        elif method == "efficientnetb7_dftimit_hq":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B7_DF-TIMIT-HQ"
        elif method == "efficientnetb7_dftimit_lq":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B7_DF-TIMIT-LQ"
        elif method == "mesonet_uadfv":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "MesoNet_UADFV"
        elif method == "mesonet_celebdf":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "MesoNet_CELEB-DF"
        elif method == "mesonet_dfdc":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "MesoNet_DFDC"
        elif method == "mesonet_dftimit_hq":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "MesoNet_DF-TIMIT-HQ"
        elif method == "mesonet_dftimit_lq":
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "MesoNet_DF-TIMIT-LQ"
        elif method == "resnet_lstm_uadfv":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "ResNet+LSTM_UADFV"
        elif method == "resnet_lstm_celebdf":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "ResNet+LSTM_CELEB-DF"
        elif method == "resnet_lstm_dfdc":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "ResNet+LSTM_DFDC"
        elif method == "resnet_lstm_dftimit_hq":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "ResNet+LSTM_DF-TIMIT-HQ"
        elif method == "resnet_lstm_dftimit_lq":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "ResNet+LSTM_DF-TIMIT-LQ"
        elif method == "efficientnetb1_lstm_uadfv":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B1+LSTM_UADFV"
        elif method == "efficientnetb1_lstm_celebdf":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B1+LSTM_CELEB-DF"
        elif method == "efficientnetb1_lstm_dfdc":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B1+LSTM_DFDC"
        elif method == "efficientnetb1_lstm_dftimit_hq":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B1+LSTM_DF-TIMIT-HQ"
        elif method == "efficientnetb1_lstm_dftimit_lq":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B1+LSTM_DF-TIMIT-LQ"
        elif method == "efficientnetb1_lstm_dftimit_lq":
            sequence_model = True
            model, img_size, normalization = prepare_method(
                method=method, dataset=None, mode='test')
            used = "EfficientNet-B1+LSTM_DF-TIMIT-LQ"
        elif method == "dfdcrank90_uadfv" or method == 'dfdcrank90_celebdf' or method == 'dfdcrank90_dftimit_lq' or method == 'dfdcrank90_dftimit_hq' or method == 'dfdcrank90_dfdc':
            ds = None
            if video_path:
                data = [[1, video_path]]
            df = pd.DataFrame(data, columns=['label', 'video'])
            loss = prepare_dfdc_rank90(
                method, ds, df, face_margin=0.3, num_frames=20, single=True, cmd=cmd)
            if method == 'dfdcrank90_uadfv':
                used = "DFDC-Rank-90_UADFV"
            elif method == 'dfdcrank90_celebdf':
                used = "DFDC-Rank-90_CELEB-DF"
            elif method == 'dfdcrank90_dftimit_lq':
                used = "DFDC-Rank-90_DF-TIMIT-LQ"
            elif method == 'dfdcrank90_dftimit_hq':
                used = "DFDC-Rank-90_DF-TIMIT-HQ"
            elif method == 'dfdcrank90_dfdc':
                used = "DFDC-Rank-90_DFDC"
        elif method == "six_method_ensemble_uadfv":
            method_uadfv = "xception_uadfv"
            loss1 = six_method_app(
                method_uadfv, video_path, sequence_model=False, cmd=cmd)
            method_effb7 = "efficientnetb7_uadfv"
            loss2 = six_method_app(
                method_effb7, video_path, sequence_model=False, cmd=cmd)
            method_meso = "mesonet_uadfv"
            loss3 = six_method_app(
                method_meso, video_path, sequence_model=False, cmd=cmd)
            method_resnet = "resnet_lstm_uadfv"
            loss4 = six_method_app(
                method_resnet, video_path, sequence_model=True, cmd=cmd)
            method_effb1 = "efficientnetb1_lstm_uadfv"
            loss5 = six_method_app(
                method_effb1, video_path, sequence_model=True, cmd=cmd)
            method_uadfv = "dfdcrank90_uadfv"
            loss6 = six_method_app(
                method_uadfv, video_path, sequence_model=False, cmd=cmd)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6
            used = "Six-Method-Ensemble_UADFV"
        elif method == "six_method_ensemble_celebdf":
            method_uadfv = "xception_celebdf"
            loss1 = six_method_app(
                method_uadfv, video_path, sequence_model=False, cmd=cmd)
            method_effb7 = "efficientnetb7_celebdf"
            loss2 = six_method_app(
                method_effb7, video_path, sequence_model=False, cmd=cmd)
            method_meso = "mesonet_celebdf"
            loss3 = six_method_app(
                method_meso, video_path, sequence_model=False, cmd=cmd)
            method_resnet = "resnet_lstm_celebdf"
            loss4 = six_method_app(
                method_resnet, video_path, sequence_model=True, cmd=cmd)
            method_effb1 = "efficientnetb1_lstm_celebdf"
            loss5 = six_method_app(
                method_effb1, video_path, sequence_model=True, cmd=cmd)
            method_uadfv = "dfdcrank90_celebdf"
            loss6 = six_method_app(
                method_uadfv, video_path, sequence_model=False, cmd=cmd)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6
            used = "Six-Method-Ensemble_CELEB-DF"
        elif method == "six_method_ensemble_dftimit_lq":
            method_uadfv = "xception_dftimit_lq"
            loss1 = six_method_app(
                method_uadfv, video_path, sequence_model=False, cmd=cmd)
            method_effb7 = "efficientnetb7_dftimit_lq"
            loss2 = six_method_app(
                method_effb7, video_path, sequence_model=False, cmd=cmd)
            method_meso = "mesonet_dftimit_lq"
            loss3 = six_method_app(
                method_meso, video_path, sequence_model=False, cmd=cmd)
            method_resnet = "resnet_lstm_dftimit_lq"
            loss4 = six_method_app(
                method_resnet, video_path, sequence_model=True, cmd=cmd)
            method_effb1 = "efficientnetb1_lstm_dftimit_lq"
            loss5 = six_method_app(
                method_effb1, video_path, sequence_model=True, cmd=cmd)
            method_uadfv = "dfdcrank90_dftimit_lq"
            loss6 = six_method_app(
                method_uadfv, video_path, sequence_model=False, cmd=cmd)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6
            used = "Six-Method-Ensemble_DF-TIMIT-LQ"
        elif method == "six_method_ensemble_dftimit_hq":
            method_uadfv = "xception_dftimit_hq"
            loss1 = six_method_app(
                method_uadfv, video_path, sequence_model=False, cmd=cmd)
            method_effb7 = "efficientnetb7_dftimit_hq"
            loss2 = six_method_app(
                method_effb7, video_path, sequence_model=False, cmd=cmd)
            method_meso = "mesonet_dftimit_hq"
            loss3 = six_method_app(
                method_meso, video_path, sequence_model=False, cmd=cmd)
            method_resnet = "resnet_lstm_dftimit_hq"
            loss4 = six_method_app(
                method_resnet, video_path, sequence_model=True, cmd=cmd)
            method_effb1 = "efficientnetb1_lstm_dftimit_hq"
            loss5 = six_method_app(
                method_effb1, video_path, sequence_model=True, cmd=cmd)
            method_uadfv = "dfdcrank90_dftimit_hq"
            loss6 = six_method_app(
                method_uadfv, video_path, sequence_model=False, cmd=cmd)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6
            used = "Six-Method-Ensemble_DF-TIMIT-HQ"
        elif method == "six_method_ensemble_dfdc":
            method_uadfv = "xception_dfdc"
            loss1 = six_method_app(
                method_uadfv, video_path, sequence_model=False, cmd=cmd)
            method_effb7 = "efficientnetb7_dfdc"
            loss2 = six_method_app(
                method_effb7, video_path, sequence_model=False, cmd=cmd)
            method_meso = "mesonet_dfdc"
            loss3 = six_method_app(
                method_meso, video_path, sequence_model=False, cmd=cmd)
            method_resnet = "resnet_lstm_dfdc"
            loss4 = six_method_app(
                method_resnet, video_path, sequence_model=True, cmd=cmd)
            method_effb1 = "efficientnetb1_lstm_dfdc"
            loss5 = six_method_app(
                method_effb1, video_path, sequence_model=True, cmd=cmd)
            method_uadfv = "dfdcrank90_dfdc"
            loss6 = six_method_app(
                method_uadfv, video_path, sequence_model=False, cmd=cmd)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6)/6
            used = "Six-Method-Ensemble_DFDC"

        if video_path:
            if not method == "dfdcrank90_uadfv" and not method == 'dfdcrank90_celebdf' and not method == 'dfdcrank90_dfdc' and not method == 'dfdcrank90_dftimit_lq' and not method == 'dfdcrank90_dftimit_hq' and not method == "six_method_ensemble_uadfv" and not method == "six_method_ensemble_celebdf" and not method == "six_method_ensemble_dftimit_lq" and not method == "six_method_ensemble_dftimit_hq" and not method == "six_method_ensemble_dfdc":
                data = [[1, video_path]]
                df = pd.DataFrame(data, columns=['label', 'video'])
                loss = test.inference(
                    model, df, img_size, normalization, dataset=None, method=method, face_margin=0.3, sequence_model=sequence_model, num_frames=20, single=True, cmd=cmd)

            if round(loss) == 1:
                result = "Deepfake detected."
                print("Deepfake detected.")
                return used, result
            else:
                result = "This is a real video."
                print("This is a real video.")
                return used, result

    @classmethod
    def benchmark(cls, dataset=None, data_path=None, method="xception_celebdf", seed=24):
        """Benchmark deepfake detection methods against popular deepfake datasets.
           The methods are already pretrained on the datasets. 
           Methods get benchmarked against a test set that is distinct from the training data.
        # Arguments:
            dataset: The dataset that the method is tested against.
            data_path: The path to the test videos.
            method: The deepfake detection method that is used.
        # Implementation: Christopher Otto
        """
        # seed numpy and pytorch for reproducibility
        reproducibility_seed(seed)
        if method not in ['xception_uadfv', 'xception_celebdf', 'xception_dftimit_hq', 'xception_dftimit_lq', 'xception_dfdc', 'efficientnetb7_uadfv', 'efficientnetb7_celebdf', 'efficientnetb7_dftimit_hq', 'efficientnetb7_dftimit_lq', 'efficientnetb7_dfdc', 'mesonet_uadfv', 'mesonet_celebdf', 'mesonet_dftimit_hq', 'mesonet_dftimit_lq', 'mesonet_dfdc', 'resnet_lstm_uadfv', 'resnet_lstm_celebdf', 'resnet_lstm_dftimit_hq', 'resnet_lstm_dftimit_lq', 'resnet_lstm_dfdc', 'efficientnetb1_lstm_uadfv', 'efficientnetb1_lstm_celebdf', 'efficientnetb1_lstm_dftimit_hq', 'efficientnetb1_lstm_dftimit_lq', 'efficientnetb1_lstm_dfdc', 'dfdcrank90_uadfv', 'dfdcrank90_celebdf', 'dfdcrank90_dftimit_hq', 'dfdcrank90_dftimit_lq', 'dfdcrank90_dfdc', 'six_method_ensemble_uadfv', 'six_method_ensemble_celebdf', 'six_method_ensemble_dftimit_hq', 'six_method_ensemble_dftimit_lq', 'six_method_ensemble_dfdc']:
            raise ValueError("Method is not available for benchmarking.")
        else:
            # method exists
            cls.dataset = dataset
            cls.data_path = data_path
            cls.method = method
            if method in []:
                face_margin = 0.0
            else:
                face_margin = 0.3
        if cls.dataset == 'uadfv':
            num_frames = 20
            # setup the dataset folders
            setup_uadfv_benchmark(cls.data_path, cls.method)
        elif cls.dataset == 'celebdf':
            num_frames = 20
            setup_celebdf_benchmark(cls.data_path, cls.method)
        elif cls.dataset == 'dftimit_hq':
            num_frames = 20
            setup_dftimit_hq_benchmark(cls.data_path, cls.method)
        elif cls.dataset == 'dftimit_lq':
            num_frames = 20
            setup_dftimit_lq_benchmark(cls.data_path, cls.method)
        elif cls.dataset == 'dfdc':
            # benchmark on only 5 frames per video, because of dataset size
            num_frames = 5
        else:
            raise ValueError(f"{cls.dataset} does not exist.")
        # get test labels for metric evaluation
        df = label_data(dataset_path=cls.data_path,
                        dataset=cls.dataset, test_data=True)
        # prepare the method of choice
        if cls.method == "xception_uadfv" or cls.method == 'xception_celebdf' or cls.method == 'xception_dftimit_hq' or cls.method == 'xception_dftimit_lq' or cls.method == 'xception_dfdc':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')
        elif cls.method == "efficientnetb7_uadfv" or cls.method == 'efficientnetb7_celebdf' or cls.method == 'efficientnetb7_dftimit_hq' or cls.method == 'efficientnetb7_dftimit_lq' or cls.method == 'efficientnetb7_dfdc':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')
        elif cls.method == 'mesonet_uadfv' or cls.method == 'mesonet_celebdf' or cls.method == 'mesonet_dftimit_hq' or cls.method == 'mesonet_dftimit_lq' or cls.method == 'mesonet_dfdc':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')
        elif cls.method == 'resnet_lstm_uadfv' or cls.method == 'resnet_lstm_celebdf' or cls.method == 'resnet_lstm_dftimit_hq' or cls.method == 'resnet_lstm_dftimit_lq' or cls.method == 'resnet_lstm_dfdc':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')
        elif cls.method == 'efficientnetb1_lstm_uadfv' or cls.method == 'efficientnetb1_lstm_celebdf' or cls.method == 'efficientnetb1_lstm_dftimit_hq'or cls.method == 'efficientnetb1_lstm_dftimit_lq' or cls.method == 'efficientnetb1_lstm_dfdc':
            model, img_size, normalization = prepare_method(
                method=cls.method, dataset=cls.dataset, mode='test')
        elif cls.method == 'dfdcrank90_uadfv' or cls.method == 'dfdcrank90_celebdf' or cls.method == 'dfdcrank90_dftimit_hq' or cls.method == 'dfdcrank90_dftimit_lq' or cls.method == 'dfdcrank90_dfdc':
            # evaluate dfdcrank90 ensemble
            auc, ap, loss, acc = prepare_dfdc_rank90(
                method, cls.dataset, df, face_margin, num_frames)
            return [auc, ap, loss, acc]
        elif cls.method == 'six_method_ensemble_uadfv' or cls.method == 'six_method_ensemble_celebdf' or cls.method == 'six_method_ensemble_dftimit_hq' or cls.method == 'six_method_ensemble_dftimit_lq'or cls.method == 'six_method_ensemble_dfdc':
            # evaluate six method ensemble
            auc, ap, loss, acc = prepare_six_method_ensemble(
                method, cls.dataset, df)
            return [auc, ap, loss, acc]

        print(f"Detecting deepfakes with \033[1m{cls.method}\033[0m ...")
        # benchmarking
        if cls.method == 'resnet_lstm_uadfv' or cls.method == 'efficientnetb1_lstm_uadfv' or cls.method == 'resnet_lstm_celebdf' or cls.method == 'resnet_lstm_dfdc' or cls.method == 'efficientnetb1_lstm_celebdf' or cls.method == 'resnet_lstm_dftimit_hq' or cls.method == 'resnet_lstm_dftimit_lq' or cls.method == 'efficientnetb1_lstm_dftimit_hq' or cls.method == 'efficientnetb1_lstm_dftimit_lq' or cls.method == 'efficientnetb1_lstm_dfdc':
            # inference for sequence models
            auc, ap, loss, acc = test.inference(
                model, df, img_size, normalization, dataset=cls.dataset, method=cls.method, face_margin=face_margin, sequence_model=True, num_frames=num_frames)
        else:
            auc, ap, loss, acc = test.inference(
                model, df, img_size, normalization, dataset=cls.dataset, method=cls.method, face_margin=face_margin, num_frames=num_frames)

        return [auc, ap, loss, acc]

    @classmethod
    def train_method(cls, dataset=None, data_path=None, method="xception", img_save_path=None, epochs=1, batch_size=32,
                     lr=0.001, folds=1, augmentation_strength='weak', fulltrain=False, faces_available=False, face_margin=0, seed=24):
        """Train a deepfake detection method on a dataset."""
        if img_save_path is None:
            raise ValueError(
                "Need a path to save extracted images for training.")
        cls.dataset = dataset
        print(f"Training on {cls.dataset} dataset.")
        cls.data_path = data_path
        cls.method = method
        cls.epochs = epochs
        cls.batch_size = batch_size
        cls.lr = lr
        cls.augmentations = augmentation_strength
        # no k-fold cross val if folds == 1
        cls.folds = folds
        # whether to train on the entire training data (without val sets)
        cls.fulltrain = fulltrain
        cls.faces_available = faces_available
        cls.face_margin = face_margin
        print(f"Training on {cls.dataset} dataset with {cls.method}.")
        # seed numpy and pytorch for reproducibility
        reproducibility_seed(seed)
        #folder_count = 35
        _, img_size, normalization = prepare_method(
            cls.method, dataset=cls.dataset, mode='train')
        # # get video train data and labels
        df = label_data(dataset_path=cls.data_path,
                        dataset=cls.dataset, test_data=False, fulltrain=cls.fulltrain)
        # detect and extract faces if they are not available already
        if not cls.faces_available:
            if cls.dataset == 'uadfv':
                addon_path = '/train_imgs/'
                if not os.path.exists(img_save_path + '/train_imgs/'):
                    # create directory in save path for images
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/train_imgs/real/')
                    os.mkdir(img_save_path + '/train_imgs/fake/')
            elif cls.dataset == 'celebdf':
                addon_path = '/facecrops/'
                # check if all folders are available
                if not os.path.exists(img_save_path + '/Celeb-real/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"Celeb-real\" folder is missing.")
                if not os.path.exists(img_save_path + '/Celeb-synthesis/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"Celeb-synthesis\" folder is missing.")
                if not os.path.exists(img_save_path + '/YouTube-real/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"YouTube-real\" folder is missing.")
                if not os.path.exists(img_save_path + '/List_of_testing_videos.txt'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"List_of_testing_videos.txt\" file is missing.")
                if not os.path.exists(img_save_path + '/facecrops/'):
                    # create directory in save path for face crops
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops/real/')
                    os.mkdir(img_save_path + '/facecrops/fake/')
                else:
                    # delete create again if it already exists with old files
                    shutil.rmtree(img_save_path + '/facecrops/')
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops/real/')
                    os.mkdir(img_save_path + '/facecrops/fake/')
            elif cls.dataset == 'dftimit_hq':
                addon_path = '/facecrops_hq/'
                # check if all folders are available
                if not os.path.exists(img_save_path + '/higher_quality/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"higher_quality\" folder is missing.")
                if not os.path.exists(img_save_path + '/dftimitreal/'):
                    raise ValueError(
                        """Please put the real videos into the \'/dftimitreal\' folder. Please organize the folder as follows:
                        ./DeepfakeTIMIT
                            /higher_quality/
                            /dftimitreal/ """
                    )
                if not os.path.exists(img_save_path + '/facecrops_hq/'):
                    # create directory in save path for face crops
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops_hq/real/')
                    os.mkdir(img_save_path + '/facecrops_hq/fake/')
                else:
                    # delete create again if it already exists with old files
                    shutil.rmtree(img_save_path + '/facecrops_hq/')
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops_hq/real/')
                    os.mkdir(img_save_path + '/facecrops_hq/fake/')
            elif cls.dataset == 'dftimit_lq':
                addon_path = '/facecrops_lq/'
                # check if all folders are available
                if not os.path.exists(img_save_path + '/lower_quality/'):
                    raise ValueError(
                        "Please unpack the dataset again. The \"lower_quality\" folder is missing.")
                if not os.path.exists(img_save_path + '/dftimitreal/'):
                    raise ValueError(
                        """Please put the real videos into the \'/dftimitreal\' folder. Please organize the folder as follows:
                        ./DeepfakeTIMIT
                            /lower_quality/
                            /dftimitreal/"""
                    )
                if not os.path.exists(img_save_path + '/facecrops_lq/'):
                    # create directory in save path for face crops
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops_lq/real/')
                    os.mkdir(img_save_path + '/facecrops_lq/fake/')
                else:
                    # delete create again if it already exists with old files
                    shutil.rmtree(img_save_path + '/facecrops_lq/')
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops_lq/real/')
                    os.mkdir(img_save_path + '/facecrops_lq/fake/')
            elif cls.dataset == 'dfdc':
                addon_path = '/facecrops/'
                val_path = '/val/'
                # check if all folders are available
                if not os.path.exists(img_save_path + '/train/'):
                    raise ValueError(
                        "Please prepare the dataset again. The \"train\" folder is missing.")
                if not os.path.exists(img_save_path + '/test/') or not os.path.exists(img_save_path + '/val/'):
                    raise ValueError(
                        """Please organize the folder as follows:
                        ./dfdcdataset
                            /train/
                            /test/ 
                            /val/
                    """
                    )
                if not os.path.exists(img_save_path + addon_path):
                    # create directory in save path for face crops
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops/real/')
                    os.mkdir(img_save_path + '/facecrops/fake/')
                    for i in range(50):
                        os.mkdir(img_save_path + f'/facecrops/real/{i}/')
                        os.mkdir(img_save_path + f'/facecrops/fake/{i}/')
                    os.mkdir(img_save_path + val_path)
                    os.mkdir(img_save_path + '/val/facecrops/')
                    os.mkdir(img_save_path + '/val/facecrops/real')
                    os.mkdir(img_save_path + '/val/facecrops/fake/')

                else:
                    # delete create again if it already exists with old files
                    shutil.rmtree(img_save_path + addon_path)
                    os.mkdir(img_save_path + addon_path)
                    os.mkdir(img_save_path + '/facecrops/real/')
                    os.mkdir(img_save_path + '/facecrops/fake/')
                    shutil.rmtree(img_save_path + val_path)
                    for i in range(50):
                        os.mkdir(img_save_path + f'/facecrops/real/{i}/')
                        os.mkdir(img_save_path + f'/facecrops/fake/{i}/')
                    os.mkdir(img_save_path + val_path)
                    os.mkdir(img_save_path + '/val/facecrops/')
                    os.mkdir(img_save_path + '/val/facecrops/real')
                    os.mkdir(img_save_path + '/val/facecrops/fake/')

            if cls.dataset == 'dfdc':
                num_frames = 5
            else:
                num_frames = 20
            print(
                f"Detect and save {num_frames} faces from each video for training.")
            if cls.face_margin > 0.0:
                print(
                    f"Apply {cls.face_margin*100}% margin to each side of the face crop.")
            else:
                print("Apply no margin to the face crop.")
            # load retinaface face detector
            net, cfg = df_retinaface.load_face_detector()
            for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
                video = row.loc['video']
                label = row.loc['label']
                vid = os.path.join(video)
                if cls.dataset == 'uadfv':
                    if label == 1:
                        video = video[-14:]
                        save_dir = os.path.join(
                            img_save_path + '/train_imgs/fake/')
                    else:
                        video = video[-9:]
                        save_dir = os.path.join(
                            img_save_path + '/train_imgs/real/')
                elif cls.dataset == 'celebdf':
                    vid_name = row.loc['video_name']
                    if label == 1:
                        video = vid_name
                        save_dir = os.path.join(
                            img_save_path + '/facecrops/fake/')
                    else:
                        video = vid_name
                        save_dir = os.path.join(
                            img_save_path + '/facecrops/real/')
                elif cls.dataset == 'dftimit_hq':
                    vid_name = row.loc['videoname']
                    video = vid_name
                    if label == 1:
                        save_dir = os.path.join(
                            img_save_path + '/facecrops_hq/fake/')
                    else:
                        save_dir = os.path.join(
                            img_save_path + '/facecrops_hq/real/')
                elif cls.dataset == 'dftimit_lq':
                    vid_name = row.loc['videoname']
                    video = vid_name
                    if label == 1:
                        save_dir = os.path.join(
                            img_save_path + '/facecrops_lq/fake/')
                    else:
                        save_dir = os.path.join(
                            img_save_path + '/facecrops_lq/real/')
                elif cls.dataset == 'dfdc':
                    # extract only 5 frames because of dataset size
                    vid_name = row.loc['videoname']
                    video = vid_name
                    folder = row.loc['folder']
                    if cls.fulltrain:
                        if label == 1:
                            save_dir = os.path.join(
                                img_save_path + f'/facecrops/fake/{folder}/')
                        else:
                            save_dir = os.path.join(
                                img_save_path + f'/facecrops/real/{folder}/')
                    else:
                        if label == 1:
                            save_dir = os.path.join(
                                img_save_path + '/val/facecrops/fake/')
                        else:
                            save_dir = os.path.join(
                                img_save_path + '/val/facecrops/real/')

                # detect faces, add margin, crop, upsample to same size, save to images
                faces = df_retinaface.detect_faces(
                    net, vid, cfg, num_frames=num_frames)
                # save frames to directory
                vid_frames = df_retinaface.extract_frames(
                    faces, video, save_to=save_dir, face_margin=cls.face_margin, num_frames=num_frames, test=False)

        # put all face images in dataframe
        df_faces = label_data(dataset_path=cls.data_path,
                              dataset=cls.dataset, method=cls.method, face_crops=True, test_data=False, fulltrain=cls.fulltrain)
        # choose augmentation strength
        augs = df_augmentations(img_size, strength=cls.augmentations)
        # start method training

        model, average_auc, average_ap, average_acc, average_loss = train.train(dataset=cls.dataset, data=df_faces,
                                                                                method=cls.method, img_size=img_size, normalization=normalization, augmentations=augs,
                                                                                folds=cls.folds, epochs=cls.epochs, batch_size=cls.batch_size, lr=cls.lr, fulltrain=cls.fulltrain
                                                                                )
        return model, average_auc, average_ap, average_acc, average_loss


def prepare_method(method, dataset, mode='train'):
    """Prepares the method that will be used for training or benchmarking."""
    if method == 'xception' or method == 'xception_uadfv' or method == 'xception_celebdf' or method == 'xception_dftimit_hq' or method == 'xception_dftimit_lq' or method == 'xception_dfdc':
        img_size = 299
        normalization = 'xception'
        if mode == 'test':
            model = xception.imagenet_pretrained_xception()
            # load the xception model that was pretrained on the respective datasets training data
            if method == 'xception_uadfv' or method == 'xception_celebdf' or method == 'xception_dftimit_hq' or method == 'xception_dftimit_lq' or method == 'xception_dfdc':
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
            return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'efficientnetb7' or method == 'efficientnetb7_uadfv' or method == 'efficientnetb7_celebdf' or method == 'efficientnetb7_dftimit_hq' or method == 'efficientnetb7_dftimit_lq' or method == 'efficientnetb7_dfdc':
        # 380 image size as introduced here https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721
        img_size = 380
        normalization = 'imagenet'
        if mode == 'test':
            if method == 'efficientnetb7_uadfv' or method == 'efficientnetb7_celebdf' or method == 'efficientnetb7_dftimit_hq' or method == 'efficientnetb7_dftimit_lq' or method == 'efficientnetb7_dfdc':
                # successfully used by https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721 (noisy student weights)
                model = timm.create_model(
                    'tf_efficientnet_b7_ns', pretrained=True)
                model.classifier = nn.Linear(2560, 1)
                # load the efficientnet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
            return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'mesonet' or method == 'mesonet_uadfv' or method == 'mesonet_celebdf' or method == 'mesonet_dftimit_hq' or method == 'mesonet_dftimit_lq' or method == 'mesonet_dfdc':
        # 256 image size as proposed in the MesoNet paper (https://arxiv.org/abs/1809.00888)
        img_size = 256
        # use [0.5,0.5,0.5] normalization scheme, because no imagenet pretraining
        normalization = 'xception'
        if mode == 'test':
            if method == 'mesonet_uadfv' or method == 'mesonet_celebdf' or method == 'mesonet_dftimit_hq' or method == 'mesonet_dftimit_lq' or method == 'mesonet_dfdc':
                # load MesoInception4 model
                model = mesonet.MesoInception4()
                # load the mesonet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
                return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'resnet_lstm' or method == 'resnet_lstm_uadfv' or method == 'resnet_lstm_celebdf' or method == 'resnet_lstm_dftimit_hq' or method == 'resnet_lstm_dftimit_lq' or method == 'resnet_lstm_dfdc':
        img_size = 224
        normalization = 'imagenet'
        if mode == 'test':
            if method == 'resnet_lstm_uadfv' or method == 'resnet_lstm_celebdf' or method == 'resnet_lstm_dftimit_hq' or method == 'resnet_lstm_dftimit_lq' or method == 'resnet_lstm_dfdc':
                # load MesoInception4 model
                model = resnetlstm.ResNetLSTM()
                # load the mesonet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
                return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization
    elif method == 'efficientnetb1_lstm' or method == 'efficientnetb1_lstm_uadfv' or method == 'efficientnetb1_lstm_celebdf' or method == 'efficientnetb1_lstm_dftimit_hq' or method == 'efficientnetb1_lstm_dftimit_lq' or method == 'efficientnetb1_lstm_dfdc':
        img_size = 240
        normalization = 'imagenet'
        if mode == 'test':
            if method == 'efficientnetb1_lstm_uadfv' or method == 'efficientnetb1_lstm_celebdf' or method == 'efficientnetb1_lstm_dftimit_hq' or method == 'efficientnetb1_lstm_dftimit_lq' or method == 'efficientnetb1_lstm_dfdc':
                # load EfficientNetB1+LSTM
                model = efficientnetb1lstm.EfficientNetB1LSTM()
                # load the mesonet model that was pretrained on the uadfv training data
                model_params = torch.load(
                    os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                print(os.getcwd(
                ) + f'/deepfake_detector/pretrained_mods/weights/{method}.pth')
                model.load_state_dict(model_params)
                return model, img_size, normalization
        elif mode == 'train':
            # model is loaded in the train loop, because easier in case of k-fold cross val
            model = None
            return model, img_size, normalization

    else:
        raise ValueError(
            f"{method} is not available. Please use one of the available methods.")


def prepare_dfdc_rank90(method, dataset, df, face_margin, num_frames, single=False, cmd=False):
    """Prepares the DFDC rank 90 ensemble."""
    img_size_xception = 299
    img_size_b1 = 240
    normalization_xception = 'xception'
    normalization_b1 = 'imagenet'
    inference_time = time.time()
    if method == 'dfdcrank90_uadfv':
        mod1 = 'efficientnetb1_lstm_uadfv'
        mod2 = 'xception_uadfv'
        mod3 = 'xception_uadfv_seed25'
    elif method == 'dfdcrank90_celebdf':
        mod1 = 'efficientnetb1_lstm_celebdf'
        mod2 = 'xception_celebdf'
        mod3 = 'xception_celebdf_seed25'
    elif method == 'dfdcrank90_dftimit_hq':
        mod1 = 'efficientnetb1_lstm_dftimit_hq'
        mod2 = 'xception_dftimit_hq'
        mod3 = 'xception_dftimit_hq_seed25'
    elif method == 'dfdcrank90_dftimit_lq':
        mod1 = 'efficientnetb1_lstm_dftimit_lq'
        mod2 = 'xception_dftimit_lq'
        mod3 = 'xception_dftimit_lq_seed25'
    elif method == 'dfdcrank90_dfdc':
        mod1 = 'efficientnetb1_lstm_dfdc'
        mod2 = 'xception_dfdc'
        mod3 = 'xception_dfdc'
    model3 = efficientnetb1lstm.EfficientNetB1LSTM()
    # load the xception model that was pretrained on the uadfv training data
    model_params3 = torch.load(
        os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{mod1}.pth')
    model3.load_state_dict(model_params3)
    print("Inference EfficientNetB1 + LSTM")
    df3 = test.inference(
        model3, df, img_size_b1, normalization_b1, dataset=dataset, method=method, face_margin=face_margin, sequence_model=True, ensemble=True, num_frames=num_frames, single=single, cmd=cmd)

    model1 = xception.imagenet_pretrained_xception()
    # load the xception model that was pretrained on the uadfv training data
    model_params1 = torch.load(
        os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{mod2}.pth')
    model1.load_state_dict(model_params1)

    print("Inference Xception One")
    df1 = test.inference(
        model1, df, img_size_xception, normalization_xception, dataset=dataset, method=method, face_margin=face_margin, ensemble=True, num_frames=num_frames, single=single, cmd=cmd)

    model2 = xception.imagenet_pretrained_xception()
    # load the xception model that was pretrained on the uadfv training data
    model_params2 = torch.load(
        os.getcwd() + f'/deepfake_detector/pretrained_mods/weights/{mod3}.pth')
    model2.load_state_dict(model_params2)

    print("Inference Xception Two")
    df2 = test.inference(
        model2, df, img_size_xception, normalization_xception, dataset=dataset, method=method, face_margin=face_margin, ensemble=True, num_frames=num_frames, single=single, cmd=cmd)
    # average predictions of all three models

    if single:
        prds = (df1 + df2 + df3)/3
        return prds

    df1['Prediction'] = (df1['Prediction'] +
                         df2['Prediction'] + df3['Prediction'])/3
    labs = list(df1['Label'])
    prds = list(df1['Prediction'])
    df1.to_csv(f'{method}_predictions_on_{dataset}.csv', index=False)
    running_corrects = 0
    running_false = 0
    running_corrects += np.sum(np.round(prds) == labs)
    running_false += np.sum(np.round(prds) != labs)

    loss_func = nn.BCEWithLogitsLoss()
    loss = loss_func(torch.Tensor(prds), torch.Tensor(labs))

    # calculate metrics
    one_rec, five_rec, nine_rec = metrics.prec_rec(
        labs, prds, method, alpha=100, plot=False)
    auc = round(roc_auc_score(labs, prds), 5)
    ap = round(average_precision_score(labs, prds), 5)
    loss = round(loss.numpy().tolist(), 5)
    acc = round(running_corrects / len(labs), 5)
    print("Benchmark results:")
    print("Confusion matrix:")
    print(confusion_matrix(labs, np.round(prds)))
    tn, fp, fn, tp = confusion_matrix(labs, np.round(prds)).ravel()
    print(f"Loss: {loss}")
    print(f"Acc: {acc}")
    print(f"AUC: {auc}")
    print(f"AP: {auc}")
    print()
    print("Cost (best possible cost is 0.0):")
    print(f"{one_rec} cost for 0.1 recall.")
    print(f"{five_rec} cost for 0.5 recall.")
    print(f"{nine_rec} cost for 0.9 recall.")
    print(
        f"Duration: {(time.time() - inference_time) // 60} min and {(time.time() - inference_time) % 60} sec.")
    print()
    print(
        f"Detected \033[1m {tp}\033[0m true deepfake videos and correctly classified \033[1m {tn}\033[0m real videos.")
    print(
        f"Mistook \033[1m {fp}\033[0m real videos for deepfakes and \033[1m {fn}\033[0m deepfakes went by undetected by the method.")
    if fn == 0 and fp == 0:
        print("Wow! A perfect classifier!")

    return auc, ap, loss, acc


def label_data(dataset_path=None, dataset='uadfv', method='xception', face_crops=False, test_data=False, fulltrain=False):
    """
    Label the data.
    # Arguments:
        dataset_path: path to data
        test_data: binary choice that indicates whether data is for testing or not.
    # Implementation: Christopher Otto
    """
    # structure data from folder in data frame for loading
    if dataset_path is None:
        raise ValueError("Please specify a dataset path.")
    if not test_data:
        if dataset == 'uadfv':
            # prepare training data
            video_path_real = os.path.join(dataset_path + "/real/")
            video_path_fake = os.path.join(dataset_path + "/fake/")
            # if no face crops available yet, read csv for videos
            if not face_crops:
                # read csv for videos
                test_dat = pd.read_csv(os.getcwd(
                ) + "/deepfake_detector/data/uadfv_test.csv", names=['video'], header=None)
                test_list = test_dat['video'].tolist()

                full_list = []
                for _, _, videos in os.walk(video_path_real):
                    for video in videos:
                        # label 0 for real video
                        full_list.append(video)

                for _, _, videos in os.walk(video_path_fake):
                    for video in videos:
                        # label 1 for deepfake video
                        full_list.append(video)

                # training data (not used for testing)
                new_list = [
                    entry for entry in full_list if entry not in test_list]

                # add labels to videos
                data_list = []
                for _, _, videos in os.walk(video_path_real):
                    for video in tqdm(videos):
                        # append if video in training data
                        if video in new_list:
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video_path_real + video})

                for _, _, videos in os.walk(video_path_fake):
                    for video in tqdm(videos):
                        # append if video in training data
                        if video in new_list:
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video_path_fake + video})

                # put data into dataframe
                df = pd.DataFrame(data=data_list)

            else:
                # if sequence, prepare sequence dataframe
                if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
                    # prepare dataframe for sequence model
                    video_path_real = os.path.join(
                        dataset_path + "/train_imgs/real/")
                    video_path_fake = os.path.join(
                        dataset_path + "/train_imgs/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video})

                    for _, _, videos in os.walk(video_path_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    df = prepare_sequence_data(dataset, df)
                    # add path to data
                    for idx, row in df.iterrows():
                        if row['label'] == 0:
                            df.loc[idx, 'original'] = str(
                                video_path_real) + str(row['original'])
                        elif row['label'] == 1:
                            df.loc[idx, 'original'] = str(
                                video_path_fake) + str(row['original'])

                else:
                    # if face crops available go to path with face crops
                    # add labels to videos

                    video_path_real = os.path.join(
                        dataset_path + "/train_imgs/real/")
                    video_path_fake = os.path.join(
                        dataset_path + "/train_imgs/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video_path_real + video})

                    for _, _, videos in os.walk(video_path_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video_path_fake + video})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)

        elif dataset == 'celebdf':
            # prepare celebdf training data by
            # reading in the testing data first
            df_test = pd.read_csv(
                dataset_path + '/List_of_testing_videos.txt', sep=" ", header=None)
            df_test.columns = ["label", "video"]
            # switch labels so that fake label is 1
            df_test['label'] = df_test['label'].apply(switch_one_zero)
            df_test['video'] = dataset_path + '/' + df_test['video']
            # structure data from folder in data frame for loading
            if not face_crops:
                video_path_real = os.path.join(dataset_path + "/Celeb-real/")
                video_path_fake = os.path.join(
                    dataset_path + "/Celeb-synthesis/")
                real_list = []
                for _, _, videos in os.walk(video_path_real):
                    for video in tqdm(videos):
                        # label 0 for real image
                        real_list.append({'label': 0, 'video': video})

                fake_list = []
                for _, _, videos in os.walk(video_path_fake):
                    for video in tqdm(videos):
                        # label 1 for deepfake image
                        fake_list.append({'label': 1, 'video': video})

                # put data into dataframe
                df_real = pd.DataFrame(data=real_list)
                df_fake = pd.DataFrame(data=fake_list)
                # add real and fake path to video file name
                df_real['video_name'] = df_real['video']
                df_fake['video_name'] = df_fake['video']
                df_real['video'] = video_path_real + df_real['video']
                df_fake['video'] = video_path_fake + df_fake['video']
                # put testing vids in list
                testing_vids = list(df_test['video'])
                # remove testing videos from training videos
                df_real = df_real[~df_real['video'].isin(testing_vids)]
                df_fake = df_fake[~df_fake['video'].isin(testing_vids)]
                # undersampling strategy to ensure class balance of 50/50
                df_fake_sample = df_fake.sample(
                    n=len(df_real), random_state=24).reset_index(drop=True)
                # concatenate both dataframes to get full training data (964 training videos with 50/50 class balance)
                df = pd.concat([df_real, df_fake_sample], ignore_index=True)
            else:
                # if sequence, prepare sequence dataframe
                if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
                    # prepare dataframe for sequence model
                    video_path_crops_real = os.path.join(
                        dataset_path + "/facecrops/real/")
                    video_path_crops_fake = os.path.join(
                        dataset_path + "/facecrops/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    df = prepare_sequence_data(dataset, df)
                    # add path to data
                    for idx, row in df.iterrows():
                        if row['label'] == 0:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_real) + str(row['original'])
                        elif row['label'] == 1:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_fake) + str(row['original'])
                else:
                    # if face crops available go to path with face crops
                    video_path_crops_real = os.path.join(
                        dataset_path + "/facecrops/real/")
                    video_path_crops_fake = os.path.join(
                        dataset_path + "/facecrops/fake/")
                    # add labels to videos
                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video_path_crops_real + video})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video_path_crops_fake + video})
                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    if len(df) == 0:
                        raise ValueError(
                            "No faces available. Please set faces_available=False.")
        elif dataset == 'dftimit_hq' or dataset == 'dftimit_lq':
            # prepare dftimit_lq training data by
            # structure data from folder in data frame for loading
            test_df_real = pd.read_csv(
                os.getcwd() + "/deepfake_detector/data/dftimit_test_real.csv")
            test_df_real['testlist'] = test_df_real['path'].str[:5] + \
                test_df_real['videoname'].apply(str)
            testing_vids_real = test_df_real['testlist'].tolist()
            test_df_fake = pd.read_csv(
                os.getcwd() + "/deepfake_detector/data/dftimit_test_fake.csv")
            test_df_fake['testlist'] = test_df_fake['videoname'].apply(str)
            testing_vids_fake = test_df_fake['testlist'].tolist()
            # join test vids in list
            test_vids = testing_vids_real + testing_vids_fake
            if not face_crops:
                # read in the reals
                reals = pd.read_csv(
                    os.getcwd() + "/deepfake_detector/data/dftimit_reals.csv")
                reals['testlist'] = reals['path'].str[:5] + \
                    reals['videoname'].apply(str)
                reals['path'] = reals['path'] + reals['videofolder'] + \
                    '/' + reals['videoname'].apply(str) + '.avi'
                # remove testing videos from training videos
                reals = reals[~reals['testlist'].isin(test_vids)]
                reals['videoname'] = reals['videoname'].apply(str) + '.avi'
                del reals['videofolder']
                reals['label'] = 0
                reals['path'] = dataset_path + '/dftimitreal/' + reals['path']
                if dataset == 'dftimit_hq':
                    fake_path = os.path.join(dataset_path, 'higher_quality')
                elif dataset == 'dftimit_lq':
                    fake_path = os.path.join(dataset_path, 'lower_quality')
                # get list of fakes
                data_list = []
                data_list_name = []
                for path, dirs, files in os.walk(fake_path):
                    for filename in files:
                        if filename.endswith(".avi"):
                            data_list.append(os.path.join(path, filename))
                            data_list_name.append(filename)
                fakes = pd.DataFrame(list(zip(data_list, data_list_name)), columns=[
                                     'path', 'videoname'])

                fakes['testlist'] = fakes['videoname'].str[:-4]
                fakes = fakes[~fakes['testlist'].isin(test_vids)]
                fakes['label'] = 1
                # put fakes and reals in one dataframe
                df = pd.concat([reals, fakes])
                df = df.rename(columns={"path": "video"})

            else:
                # if sequence and if face crops available go to path with face crops and prepare sequence data
                if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
                    # prepare dataframe for sequence model
                    if dataset == 'dftimit_hq':
                        video_path_crops_real = os.path.join(
                            dataset_path + "/facecrops_hq/real/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/facecrops_hq/fake/")
                    elif dataset == 'dftimit_lq':
                        video_path_crops_real = os.path.join(
                            dataset_path + "/facecrops_lq/real/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/facecrops_lq/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    df = prepare_sequence_data(dataset, df)
                    # add path to data
                    for idx, row in df.iterrows():
                        if row['label'] == 0:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_real) + str(row['original'])
                        elif row['label'] == 1:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_fake) + str(row['original'])
                else:
                    # if face crops available go to path with face crops
                    if dataset == 'dftimit_hq':
                        video_path_crops_real = os.path.join(
                            dataset_path + "/facecrops_hq/real/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/facecrops_hq/fake/")
                    elif dataset == 'dftimit_lq':
                        video_path_crops_real = os.path.join(
                            dataset_path + "/facecrops_lq/real/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/facecrops_lq/fake/")
                    # add labels to videos
                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': video_path_crops_real + video})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video': video_path_crops_fake + video})
                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    if len(df) == 0:
                        raise ValueError(
                            "No faces available. Please set faces_available=False.")
        elif dataset == 'dfdc':
            # prepare dfdc training data
            # structure data from folder in data frame for loading
            all_meta_train, all_meta_test, full_margin_aug_val = utils.dfdc_metadata_setup()
            if not face_crops:
                # read in the reals
                if fulltrain:
                    all_meta_train['videoname'] = all_meta_train['video']
                    all_meta_train['video'] = dataset_path + \
                        '/train/' + all_meta_train['videoname']
                    all_meta_train = all_meta_train.sort_values(
                        'folder').reset_index(drop=True)
                    df = all_meta_train[all_meta_train['folder'] > 35]
                    print(df)
                else:
                    print("Validation DFDC data.")
                    full_margin_aug_val['videoname'] = full_margin_aug_val['video']
                    full_margin_aug_val['video'] = dataset_path + \
                        '/train/' + full_margin_aug_val['videoname']
                    df = full_margin_aug_val
            else:
                # if face crops available
                # if sequence and if face crops available go to path with face crops and prepare sequence data
                if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
                    # prepare dataframe for sequence model
                    if fulltrain:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/facecrops/real/all/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/facecrops/fake/all/")
                    else:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/val/facecrops/real/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/val/facecrops/fake/")

                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video':  os.path.join(video_path_crops_real, video)})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video':  os.path.join(video_path_crops_fake, video)})

                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    df = prepare_sequence_data(dataset, df)
                    # add path to data
                    for idx, row in df.iterrows():
                        if row['label'] == 0:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_real) + str(row['original'])
                        elif row['label'] == 1:
                            df.loc[idx, 'original'] = str(
                                video_path_crops_fake) + str(row['original'])
                else:
                    # if face crops available and not a sequence model go to path with face crops
                    if fulltrain:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/facecrops/real/all/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/facecrops/fake/all/")
                    else:
                        video_path_crops_real = os.path.join(
                            dataset_path + "/val/facecrops/real/")
                        video_path_crops_fake = os.path.join(
                            dataset_path + "/val/facecrops/fake/")
                    # add labels to videos
                    data_list = []
                    for _, _, videos in os.walk(video_path_crops_real):
                        for video in tqdm(videos):
                            # label 0 for real video
                            data_list.append(
                                {'label': 0, 'video': os.path.join(video_path_crops_real, video)})

                    for _, _, videos in os.walk(video_path_crops_fake):
                        for video in tqdm(videos):
                            # label 1 for deepfake video
                            data_list.append(
                                {'label': 1, 'video':  os.path.join(video_path_crops_fake, video)})
                    # put data into dataframe
                    df = pd.DataFrame(data=data_list)
                    print(df)
                    if len(df) == 0:
                        raise ValueError(
                            "No faces available. Please set faces_available=False.")

    else:
        # prepare test data
        if dataset == 'uadfv':
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
        elif dataset == 'celebdf':
            # reading in the celebdf testing data
            df_test = pd.read_csv(
                dataset_path + '/List_of_testing_videos.txt', sep=" ", header=None)
            df_test.columns = ["label", "video"]
            # switch labels so that fake label is 1
            df_test['label'] = df_test['label'].apply(switch_one_zero)
            df_test['video'] = dataset_path + '/' + df_test['video']
            print(f"{len(df_test)} test videos.")
            return df_test
        elif dataset == 'dftimit_hq' or dataset == 'dftimit_lq':
            test_df_real = pd.read_csv(
                os.getcwd() + "/deepfake_detector/data/dftimit_test_real.csv")
            test_df_real['testlist'] = test_df_real['path'].str[:5] + \
                test_df_real['videoname'].apply(str)
            testing_vids_real = test_df_real['testlist'].tolist()
            test_df_fake = pd.read_csv(
                os.getcwd() + "/deepfake_detector/data/dftimit_test_fake.csv")
            test_df_fake['testlist'] = test_df_fake['videoname'].apply(str)
            testing_vids_fake = test_df_fake['testlist'].tolist()
            # join test vids in list
            test_vids = testing_vids_real + testing_vids_fake
            # read in the reals
            reals = pd.read_csv(
                os.getcwd() + "/deepfake_detector/data/dftimit_reals.csv")
            reals['testlist'] = reals['path'].str[:5] + \
                reals['videoname'].apply(str)
            reals['path'] = reals['path'] + reals['videofolder'] + \
                '/' + reals['videoname'].apply(str) + '.avi'
            # remove testing videos from training videos
            reals = reals[reals['testlist'].isin(test_vids)]
            reals['videoname'] = reals['videoname'].apply(str) + '.avi'
            del reals['videofolder']
            reals['label'] = 0
            reals['path'] = dataset_path + '/dftimitreal/' + reals['path']
            if dataset == 'dftimit_hq':
                fake_path = os.path.join(dataset_path, 'higher_quality')
            elif dataset == 'dftimit_lq':
                fake_path = os.path.join(dataset_path, 'lower_quality')
            # get list of fakes
            data_list = []
            data_list_name = []
            for path, dirs, files in os.walk(fake_path):
                for filename in files:
                    if filename.endswith(".avi"):
                        data_list.append(os.path.join(path, filename))
                        data_list_name.append(filename)
            fakes = pd.DataFrame(list(zip(data_list, data_list_name)), columns=[
                                 'path', 'videoname'])
            fakes['testlist'] = fakes['videoname'].str[:-4]
            fakes = fakes[fakes['testlist'].isin(test_vids)]
            fakes['label'] = 1
            # put fakes and reals in one dataframe
            df_test = pd.concat([reals, fakes], ignore_index=True)
            del df_test['testlist']
            del df_test['videoname']
            df_test = df_test.rename(columns={'path': 'video'})
            return df_test
        elif dataset == 'dfdc':
            # prepare dfdc training data
            # structure data from folder in data frame for loading
            all_meta_train, all_meta_test, full_margin_aug_val = utils.dfdc_metadata_setup()
            all_meta_test['videoname'] = all_meta_test['video']
            all_meta_test['video'] = dataset_path + \
                '/test/' + all_meta_test['videoname']
            # randomly sample 1000 test videos
            df_test_reals = all_meta_test[all_meta_test['label'] == 0]
            df_test_fakes = all_meta_test[all_meta_test['label'] == 1]
            df_test_reals = df_test_reals.sample(
                n=500, replace=False, random_state=24)
            df_test_fakes = df_test_fakes.sample(
                n=500, replace=False, random_state=24)
            df_test = pd.concat(
                [df_test_reals, df_test_fakes], ignore_index=True)
            print(df_test)
            return df_test
        # put data into dataframe
        df = pd.DataFrame(data=data_list)

    if test_data:
        print(f"{len(df)} test videos.")
    else:
        if face_crops:
            print(f"Lead to: {len(df)} face crops.")
        else:
            print(f"{len(df)} train videos.")
    print()
    return df


def df_augmentations(img_size, strength="weak"):
    """
    Augmentations with the albumentations package.
    # Arguments:
        strength: strong or weak augmentations

    # Implementation: Christopher Otto
    """
    if strength == "weak":
        print("Weak augmentations.")
        augs = Compose([
            # hflip with prob 0.5
            HorizontalFlip(p=0.5),
            # adjust image to DNN input size
            Resize(width=img_size, height=img_size)
        ])
        return augs
    elif strength == "strong":
        print("Strong augmentations.")
        # augmentations via albumentations package
        # augmentations adapted from Selim Seferbekov's 3rd place private leaderboard solution from
        # https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721
        augs = Compose([
            # hflip with prob 0.5
            HorizontalFlip(p=0.5),
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
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
    else:
        raise ValueError(
            "This augmentation option does not exist. Choose \"weak\" or \"strong\".")


def structure_uadfv_files(path_to_data):
    """Creates test folders and moves test videos there."""
    os.mkdir(path_to_data + '/test/')
    os.mkdir(path_to_data + '/test/fake/')
    os.mkdir(path_to_data + '/test/real/')
    test_data = pd.read_csv(
        os.getcwd() + "/deepfake_detector/data/uadfv_test.csv", names=['video'], header=None)
    for idx, row in test_data.iterrows():
        if len(str(row.loc['video'])) > 8:
            # video is fake, therefore copy it into fake test folder
            shutil.copy(path_to_data + '/fake/' +
                        row['video'], path_to_data + '/test/fake/')
        else:
            # video is real, therefore move it into real test folder
            shutil.copy(path_to_data + '/real/' +
                        row['video'], path_to_data + '/test/real/')


def reproducibility_seed(seed):
    print(f"The random seed is set to {seed}.")
    # set numpy random seed
    np.random.seed(seed)
    # set pytorch random seed for cpu and gpu
    torch.manual_seed(seed)
    # get deterministic behavior
    torch.backends.cudnn.deterministic = True


def switch_one_zero(num):
    """Switch label 1 to 0 and 0 to 1
        so that fake videos have label 1.
    """
    if num == 1:
        num = 0
    else:
        num = 1
    return num


def prepare_sequence_data(dataset, df):
    """
    Prepares the dataframe for sequence models.
    """
    print(df)
    df = df.sort_values(by=['video']).reset_index(drop=True)
    # add original column
    df['original'] = ""
    if dataset == 'uadfv':
        # label data
        print("Preparing sequence data.")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if row.loc['label'] == 0:
                df.loc[idx, 'original'] = row.loc['video'][:4]
            elif row.loc['label'] == 1:
                df.loc[idx, 'original'] = row.loc['video'][:9]
    elif dataset == 'celebdf' or dataset == 'dftimit_hq' or dataset == 'dftimit_lq':
        print("Preparing sequence data.")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # remove everything after last underscore
            df.loc[idx, 'original'] = row.loc['video'].rpartition("_")[0]
    elif dataset == 'dfdc':
        print("Preparing sequence data.")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            # remove everything after last underscore
            df.loc[idx, 'original'] = row.loc['video'].rpartition("_")[0][-10:]
    # count frames per video
    df1 = df.groupby(['original']).size().reset_index(name='count')
    df = pd.merge(df, df1, on='original')
    # remove videos that don't where less than 20 frames
    # were detected to ensure equal frame size of 20 for sequence
    # for dfdc only 5 frames because dataset is so large
    if dataset == 'dfdc':
        df = df[df['count'] == 5]
    else:
        df = df[df['count'] == 20]
    df = df[['label', 'original']]
    # ensure that dataframe includes each video with 20 frames once
    df = df.groupby(['label', 'original']).size().reset_index(name='count')
    df = df[['label', 'original']]
    return df


def setup_uadfv_benchmark(data_path, method):
    """
    Setup the folder structure of the UADFV Dataset.
    """
    if data_path is None:
        raise ValueError("""Please go to https://github.com/danmohaha/WIFS2018_In_Ictu_Oculi
                                and scroll down to the UADFV section.
                                Click on the link \"here\" and download the dataset. 
                                Extract the files and organize the folders follwing this folder structure:
                                ./fake_videos/
                                            fake/
                                            real/
                                """)
    if data_path.endswith("fake_videos.zip"):
        raise ValueError("Please make sure to extract the zipfile.")
    if data_path.endswith("fake_videos"):
        print(
            f"Benchmarking \033[1m{method}\033[0m on the \033[1m UADFV \033[0m dataset with ...")
        # create test directories if they don't exist
        if not os.path.exists(data_path + '/test/'):
            structure_uadfv_files(path_to_data=data_path)
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
                structure_uadfv_files(path_to_data=data_path)
    else:
        raise ValueError("""Please organize the dataset directory in this way:
                            ./fake_videos/
                                        fake/
                                        real/
                        """)


def setup_celebdf_benchmark(data_path, method):
    """
    Setup the folder structure of the Celeb-DF Dataset.
    """
    if data_path is None:
        raise ValueError("""Please go to https://github.com/danmohaha/celeb-deepfakeforensics
                                and scroll down to the dataset section.
                                Click on the link \"this form\" and download the dataset. 
                                Extract the files and organize the folders follwing this folder structure:
                                ./celebdf/
                                        Celeb-real/
                                        Celeb-synthesis/
                                        YouTube-real/
                                        List_of_testing_videos.txt
                                """)
    if data_path.endswith("celebdf"):
        print(
            f"Benchmarking \033[1m{method}\033[0m on the \033[1m Celeb-DF \033[0m dataset with ...")
    else:
        raise ValueError("""Please organize the dataset directory in this way:
                            ./celebdf/
                                    Celeb-real/
                                    Celeb-synthesis/
                                    YouTube-real/
                                    List_of_testing_videos.txt
                        """)


def setup_dftimit_hq_benchmark(data_path, method):
    """
    Setup the folder structure of the DFTIMIT HQ Dataset.
    """
    if data_path is None:
        raise ValueError("""Please go to http://conradsanderson.id.au/vidtimit/ to download the real videos and to
                                https://www.idiap.ch/dataset/deepfaketimit to download the deepfake videos.
                                Extract the files and organize the folders follwing this folder structure:
                                ./DeepfakeTIMIT
                                    /lower_quality/
                                    /higher_quality/
                                    /dftimitreal/
                                """)
    if data_path.endswith("DeepfakeTIMIT"):
        print(
            f"Benchmarking \033[1m{method}\033[0m on the \033[1m DF-TIMIT-HQ \033[0m dataset with ...")
    else:
        raise ValueError("""Make sure your data_path argument ends with \"DeepfakeTIMIT\" and organize the dataset directory in this way:
                            ./DeepfakeTIMIT
                                    /lower_quality/
                                    /higher_quality/
                                    /dftimitreal/
                        """)


def setup_dftimit_lq_benchmark(data_path, method):
    """
    Setup the folder structure of the DFTIMIT HQ Dataset.
    """
    if data_path is None:
        raise ValueError("""Please go to http://conradsanderson.id.au/vidtimit/ to download the real videos and to
                                https://www.idiap.ch/dataset/deepfaketimit to download the deepfake videos.
                                Extract the files and organize the folders follwing this folder structure:
                                ./DeepfakeTIMIT
                                    /lower_quality/
                                    /higher_quality/
                                    /dftimitreal/
                                """)
    if data_path.endswith("DeepfakeTIMIT"):
        print(
            f"Benchmarking \033[1m{method}\033[0m on the \033[1m DF-TIMIT-LQ \033[0m dataset with ...")
    else:
        raise ValueError("""Make sure your data_path argument ends with \"DeepfakeTIMIT\" and organize the dataset directory in this way:
                            ./DeepfakeTIMIT
                                    /lower_quality/
                                    /higher_quality/
                                    /dftimitreal/
                        """)


def prepare_six_method_ensemble(method, dataset, df):
    """Calculates the metrics for the six method ensemble."""

    if method == 'six_method_ensemble_uadfv':
        ens = 'uadfv'
    elif method == 'six_method_ensemble_celebdf':
        ens = 'celebdf'
    elif method == 'six_method_ensemble_dftimit_hq':
        ens = 'dftimit_hq'
    elif method == 'six_method_ensemble_dftimit_lq':
        ens = 'dftimit_lq'
    elif method == 'six_method_ensemble_dfdc':
        ens = 'dfdc'
    six_method_ens = pd.read_csv(
        f"efficientnetb1_lstm_{ens}_predictions_on_{dataset}.csv")
    six_method_ens['Prediction'] = 0
    # read predictions of all six methods
    effb1lstm = pd.read_csv(
        f"efficientnetb1_lstm_{ens}_predictions_on_{dataset}.csv")
    resnetlstm = pd.read_csv(
        f"resnet_lstm_{ens}_predictions_on_{dataset}.csv")
    meso = pd.read_csv(f"mesonet_{ens}_predictions_on_{dataset}.csv")
    effb7 = pd.read_csv(
        f"efficientnetb7_{ens}_predictions_on_{dataset}.csv")
    xcep = pd.read_csv(f"xception_{ens}_predictions_on_{dataset}.csv")
    rank90ens = pd.read_csv(
        f"dfdcrank90_{ens}_predictions_on_{dataset}.csv")
    # calculate the average of the prediction
    six_method_ens['Prediction'] = (effb1lstm['Prediction'] + resnetlstm['Prediction'] +
                                    meso['Prediction'] + effb7['Prediction'] + xcep['Prediction'] + rank90ens['Prediction'])/6
    # calculate metrics for ensemble
    labs = list(six_method_ens['Label'])
    prds = list(six_method_ens['Prediction'])
    running_corrects = 0
    running_false = 0
    running_corrects += np.sum(np.round(prds) == labs)
    running_false += np.sum(np.round(prds) != labs)

    loss_func = nn.BCEWithLogitsLoss()
    loss = loss_func(torch.Tensor(prds), torch.Tensor(labs))
    # calculate metrics
    one_rec, five_rec, nine_rec = metrics.prec_rec(
        labs, prds, method, alpha=100, plot=False)
    auc = round(roc_auc_score(labs, prds), 5)
    ap = round(average_precision_score(labs, prds), 5)
    loss = round(loss.numpy().tolist(), 5)
    acc = round(running_corrects / len(labs), 5)
    print("Benchmark results:")
    print("Confusion matrix:")
    print(confusion_matrix(labs, np.round(prds)))
    tn, fp, fn, tp = confusion_matrix(labs, np.round(prds)).ravel()
    print(f"Loss: {loss}")
    print(f"Acc: {acc}")
    print(f"AUC: {auc}")
    print(f"AP: {auc}")
    print()
    print("Cost (best possible cost is 0.0):")
    print(f"{one_rec} cost for 0.1 recall.")
    print(f"{five_rec} cost for 0.5 recall.")
    print(f"{nine_rec} cost for 0.9 recall.")
    print()
    print(
        f"Detected \033[1m {tp}\033[0m true deepfake videos and correctly classified \033[1m {tn}\033[0m real videos.")
    print(
        f"Mistook \033[1m {fp}\033[0m real videos for deepfakes and \033[1m {fn}\033[0m deepfakes went by undetected by the method.")
    if fn == 0 and fp == 0:
        print("Wow! A perfect classifier!")

    return auc, ap, loss, acc


def six_method_app(method, video_path, sequence_model, cmd=False):
    if method.startswith("dfdcrank90"):
        ds = None
        if video_path:
            data = [[1, video_path]]
            df = pd.DataFrame(data, columns=['label', 'video'])
            loss = prepare_dfdc_rank90(
                method, ds, df, face_margin=0.3, num_frames=20, single=True, cmd=cmd)
            return loss
    model, img_size, normalization = prepare_method(
        method=method, dataset=None, mode='test')
    if video_path:
        data = [[1, video_path]]
    df = pd.DataFrame(data, columns=['label', 'video'])
    loss = test.inference(
        model, df, img_size, normalization, dataset=None, method=method, face_margin=0.3, sequence_model=sequence_model, num_frames=20, single=True, cmd=cmd)
    return loss


def main():
    # parse arguments
    args = parser.parse_args()
    # initialize the deepfake detector with the desired task
    if args.detect_single:
        print(f"Detecting with {args.detection_method}.")
        DFDetector.detect_single(
            video_path=args.path_to_vid, image_path=args.path_to_img, method=args.detection_method, cmd=args.cmd)
    elif args.benchmark:
        DFDetector.benchmark(
            dataset=args.dataset, data_path=args.data_path, method=args.detection_method)
    elif args.train:
        print(args)
        print(args.facecrops_available)
        DFDetector.train_method(dataset=args.dataset, data_path=args.data_path, method=args.model_type, img_save_path=args.save_path, epochs=args.epochs, batch_size=args.batch_size,
                     lr=args.lr, folds=args.folds, augmentation_strength=args.augs, fulltrain=args.fulltrain,  face_margin=args.face_margin, faces_available=args.facecrops_available, seed=args.seed)
    else:
        print("Please choose one of the three modes: detect_single, benchmark, or train.")


if __name__ == '__main__':
    main()
