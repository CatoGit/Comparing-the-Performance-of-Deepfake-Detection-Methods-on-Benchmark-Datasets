import argparse
import copy
import os
import time
import gc
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, train_test_split
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader, Dataset


import datasets
import timm
import metrics
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from facedetector.retinaface import df_retinaface
from pretrained_mods import xception
from pretrained_mods import mesonet
from pretrained_mods import resnetlstm
from pretrained_mods import efficientnetb1lstm
from tqdm import tqdm


def train(dataset, data, method, normalization, augmentations, img_size,
          folds=1, epochs=1, batch_size=32, lr=0.001, fulltrain=False, load_model_path=None, return_best=False
          ):
    """
    Train a DNN for a number of epochs.

    # parts from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # adapted by: Christopher Otto
    """
    training_time = time.time()
    # use gpu for calculations if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    average_auc = []
    average_loss = []
    average_acc = []
    average_ap = []
    average_one_rec = []
    average_five_rec = []
    average_nine_rec = []

    # k-fold cross-val if folds > 1
    for fold in range(folds):
        if folds > 1:
            # doing k-fold cross-validation
            print(f"Fold: {fold}")

        best_acc = 0.0
        best_loss = 100
        current_acc = 0.0
        current_loss = 100.0
        best_auc = 0.0
        best_ap = 0.0

        # get train and val indices
        if fulltrain == False:
            if folds > 1:
                # cross validation
                train_idx, val_idx = kfold_cross_val(method, fold, data)
            else:
                _, _, _, _, train_idx, val_idx = holdout_val(
                    method, fold, data)
        # prepare training and validation data
        if fulltrain == True:
            train_dataset, train_loader = prepare_fulltrain_datasets(
                dataset, method, data, img_size, normalization, augmentations, batch_size)
        else:
            train_dataset, train_loader, val_dataset, val_loader = prepare_train_val(
                dataset, method, data, img_size, normalization, augmentations, batch_size, train_idx, val_idx)
        if load_model_path is None:
            # train model from pretrained imagenet or mesonet or noisy student weights
            if method == 'xception':
                # load the xception model
                model = xception.imagenet_pretrained_xception()
            elif method == 'efficientnetb7':
                model = timm.create_model(
                    'tf_efficientnet_b7_ns', pretrained=True)
                # binary classification output
                model.classifier = nn.Linear(2560, 1)
            elif method == 'mesonet':
                # load MesoInception4 model
                model = mesonet.MesoInception4()
                # load mesonet weights that were pretrained on the mesonet dataset from https://github.com/DariusAf/MesoNet
                model.load_state_dict(torch.load(
                    "./deepfake_detector/pretrained_mods/weights/mesonet_pretrain.pth"))
            elif method == 'resnet_lstm':
                model = resnetlstm.ResNetLSTM()
            elif method == 'efficientnetb1_lstm':
                model = efficientnetb1lstm.EfficientNetB1LSTM()

        else:
            # continue to train model from custom checkpoint
            model = torch.load(load_model_path)

        if return_best:
            best_model_state = copy.deepcopy(model.state_dict())

        # put model on gpu
        model = model.cuda()
        # binary cross-entropy loss
        loss_func = nn.BCEWithLogitsLoss()
        lr = lr
        # adam optimizer
        optimizer = Adam(model.parameters(), lr=lr)
        # cosine annealing scheduler
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, epochs, eta_min=0.000001, last_epoch=-1)

        for e in range(epochs):
            print('#' * 20)
            print(f"Epoch {e}/{epochs}")
            print('#' * 20)
            # training and validation loop
            for phase in ["train", "val"]:
                if phase == "train":
                    # put layers in training mode
                    model.train()
                else:
                    # turn batchnorm and dropout layers to eval mode
                    model.eval()
                running_loss = 0.0
                running_corrects = 0.0
                running_auc_labels = []
                running_ap_labels = []
                running_auc_preds = []
                running_ap_preds = []
                if phase == "train":
                    # then load training data
                    for imgs, labels in tqdm(train_loader):
                        # put calculations on gpu
                        imgs = imgs.to(device)
                        labels = labels.to(device)
                        # set accumulated gradients to zero
                        optimizer.zero_grad()

                        # forward pass of inputs and turn on gradient computation during train
                        with torch.set_grad_enabled(phase == "train"):
                            predictions = model(imgs)
                            sig = torch.sigmoid(predictions)
                            # predictions for acc calculation; classification thresh 0.5
                            thresh_preds = torch.round(
                                torch.sigmoid(predictions))
                            loss = loss_func(
                                predictions.squeeze(-1), labels.type_as(predictions))

                            if phase == "train":
                                # backpropagate gradients
                                loss.backward()
                                # update parameters
                                optimizer.step()

                        running_loss += loss.item() * imgs.size(0)
                        # calc accuracy
                        running_corrects += torch.sum(thresh_preds ==
                                                      labels.unsqueeze(1))
                        running_auc_labels.extend(
                            labels.detach().cpu().numpy())
                        running_auc_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())
                        running_ap_labels.extend(labels.detach().cpu().numpy())
                        running_ap_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())
                    if phase == 'train':
                        # update lr
                        scheduler.step()
                    epoch_loss = running_loss / len(train_dataset)
                    epoch_acc = running_corrects / len(train_dataset)
                    epoch_auc = roc_auc_score(
                        running_auc_labels, running_auc_preds)
                    epoch_ap = average_precision_score(
                        running_ap_labels, running_ap_preds)
                    print(
                        f"{phase} Loss: {epoch_loss}, Acc: {epoch_acc}, AUC: {epoch_auc}, AP: {epoch_ap}")
                    if fulltrain == True and e+1 == epochs:
                        # save model if epochs reached
                        print("Save fulltrain model.")
                        torch.save(
                            model.state_dict(), os.getcwd() + f'/{method}_{dataset}.pth')

                else:
                    if fulltrain == True:
                        continue
                    # get valitation data
                    for imgs, labels in tqdm(val_loader):
                        # put calculations on gpu
                        imgs = imgs.to(device)
                        labels = labels.to(device)
                        # set accumulated gradients to zero
                        optimizer.zero_grad()

                        # forward pass of inputs and turn on gradient computation during train
                        with torch.set_grad_enabled(phase == "train"):
                            predictions = model(imgs)
                            sig = torch.sigmoid(predictions)
                            # predictions for acc calculation; classification thresh 0.5
                            thresh_preds = torch.round(
                                torch.sigmoid(predictions))
                            loss = loss_func(
                                predictions.squeeze(-1), labels.type_as(predictions))

                        running_loss += loss.item() * imgs.size(0)
                        # calc accuracy
                        running_corrects += torch.sum(thresh_preds ==
                                                      labels.unsqueeze(1))

                        running_auc_labels.extend(
                            labels.detach().cpu().numpy())
                        running_auc_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())
                        running_ap_labels.extend(labels.detach().cpu().numpy())
                        running_ap_preds.extend(
                            sig.detach().cpu().numpy().flatten().tolist())

                    epoch_loss = running_loss / len(val_dataset)
                    epoch_acc = running_corrects / len(val_dataset)
                    epoch_auc = roc_auc_score(
                        running_auc_labels, running_auc_preds)
                    epoch_ap = average_precision_score(
                        running_ap_labels, running_ap_preds)
                    print(
                        f"{phase} Loss: {epoch_loss}, Acc: {epoch_acc}, AUC: {epoch_auc}, AP: {epoch_ap}")

                    # save model if acc better than best acc
                    if epoch_acc > best_acc:
                        print("Found a better model.")
                        one_rec, five_rec, nine_rec = metrics.prec_rec(
                            running_auc_labels, running_auc_preds, method, alpha=100, plot=False)
                        best_acc = epoch_acc
                        best_auc = epoch_auc
                        best_loss = epoch_loss
                        best_ap = epoch_ap
                        if folds > 1:
                            torch.save(
                                model.state_dict(), os.getcwd() + f'/{method}_best_acc_model_fold{fold}.pth')
                            if return_best:
                                best_model_state = copy.deepcopy(
                                    model.state_dict())
                        else:
                            print(best_acc)
                            print(best_auc)
                            print(best_loss)
                            print(best_ap)
                            with open("Output.txt", "w") as text_file:
                                print(f"Acc: {best_acc}", file=text_file)
                                print(f"AUC: {best_auc}", file=text_file)
                                print(f"Loss: {best_loss}", file=text_file)
                                print(f"AP: {best_ap}", file=text_file)
                                print(f"Epoch: {e+1}", file=text_file)
#                             torch.save(
#                                 model.state_dict(), os.getcwd() + f'/{method}_ep{e}_{best_acc}_{best_auc}_{best_ap}_{best_loss}.pth')
                            if return_best:
                                best_model_state = copy.deepcopy(
                                    model.state_dict())
                    # if loss is lower, but accuracy equal, take that model as best new model
                    # e.g. used with small datasets when accuracy goes to 1.0 quickly
                    elif epoch_acc == best_acc and epoch_loss < best_loss:
                        print("Found a better model.")
                        one_rec, five_rec, nine_rec = metrics.prec_rec(
                            running_auc_labels, running_auc_preds, method, alpha=100, plot=False)
                        best_acc = epoch_acc
                        best_auc = epoch_auc
                        best_loss = epoch_loss
                        best_ap = epoch_ap
                        if folds > 1:
                            torch.save(
                                model.state_dict(), os.getcwd() + f'/{method}_best_acc_model_fold{fold}.pth')
                            if return_best:
                                best_model_state = copy.deepcopy(
                                    model.state_dict())
                        else:
                            print(best_acc)
                            print(best_auc)
                            print(best_loss)
                            print(best_ap)
#                             torch.save(
#                                 model.state_dict(), os.getcwd() + f'/{method}_ep{e}_{best_acc}_{best_auc}_{best_ap}_{best_loss}.pth')
                            if return_best:
                                best_model_state = copy.deepcopy(
                                    model.state_dict())

            gc.collect()
            torch.cuda.empty_cache()
        average_auc.append(best_auc)
        average_ap.append(best_ap)
        average_acc.append(best_acc)
        average_loss.append(best_loss)
        if not fulltrain:
            average_one_rec.append(one_rec)
            average_five_rec.append(five_rec)
            average_nine_rec.append(nine_rec)
        else:
            # only saved model is returned
            return model, 0, 0, 0, 0
    # average the best results of all folds
    average_auc = np.array(average_auc).mean()
    average_ap = np.array(average_ap).mean()
    average_acc = np.mean(np.asarray(
        [entry.cpu().numpy() for entry in average_acc]))
    average_loss = np.mean(np.asarray([entry for entry in average_loss]))
    average_one_rec = np.array(average_one_rec).mean()
    average_five_rec = np.array(average_five_rec).mean()
    average_nine_rec = np.array(average_nine_rec).mean()

    if folds > 1:
        print(f"Average AUC: {average_auc}")
        print(f"Average AP: {average_ap}")
        print(f"Average Acc: {average_acc}")
        print(f"Average Loss: {average_loss}")
        print()
        print("Average Cost (best possible cost is 0.0):")
        print(f"{average_one_rec} cost for 0.1 recall.")
        print(f"{average_five_rec} cost for 0.5 recall.")
        print(f"{average_nine_rec} cost for 0.9 recall.")
        print(
            f"Duration: {(time.time() - training_time) // 60} min and {(time.time() - training_time) % 60} sec.")
    else:
        print()
        print("Best models metrics:")
        print(f"Acc: {average_acc}")
        print(f"AUC: {average_auc}")
        print(f"AP: {average_ap}")
        print(f"Loss: {average_loss}")
        print()
        print("Cost (best possible cost is 0.0):")
        print(f"{average_one_rec} cost for 0.1 recall.")
        print(f"{average_five_rec} cost for 0.5 recall.")
        print(f"{average_nine_rec} cost for 0.9 recall.")
        print(
            f"Duration: {(time.time() - training_time) // 60} min and {(time.time() - training_time) % 60} sec.")

    if return_best:
        # load best model params
        model.load_state_dict(best_model_state)
    return model, average_auc, average_ap, average_acc, average_loss


def kfold_cross_val(method, fold, df):
    """
    Return train and val indices for 5 folds of 5-fold cross validation.
    """
    if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
        X = df['original'].values
        y = df['label'].values
    else:
        X = df['video'].values
        y = df['label'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=24)
    train = []
    val = []
    for train_index, val_index in kf.split(X):
        train.append(train_index)
        val.append(val_index)

    # return indices for fold
    return list(train[fold]), list(val[fold])


def holdout_val(method, fold, df):
    """
        Return training and validation data in a holdout split.
    """
    if method == 'resnet_lstm' or method == 'efficientnetb1_lstm':
        X = df['original'].values
        y = df['label'].values
        indices = df.index.values.tolist()
    else:
        X = df['video'].values
        y = df['label'].values
        indices = df.index.values.tolist()

    X_train, X_test, y_train, y_test, train_idx, val_idx = train_test_split(
        X, y, indices, test_size=0.2, random_state=24)
    return X_train, X_test, y_train, y_test, train_idx, val_idx


def prepare_fulltrain_datasets(dataset, method, data, img_size, normalization, augmentations, batch_size):
    """
    Prepare datasets for training with all data.
    """
    if dataset == 'uadfv':
        train_dataset = datasets.UADFVDataset(
            data, img_size, method=method,  normalization=normalization, augmentations=augmentations)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
    elif dataset == 'celebdf':
        train_dataset = datasets.CelebDFDataset(
            data, img_size, method=method,  normalization=normalization, augmentations=augmentations)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
    elif dataset == 'dftimit_hq':
        train_dataset = datasets.DFTIMITHQDataset(
            data.iloc[train_idx], img_size, method=method, normalization=normalization, augmentations=augmentations)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_dataset, train_loader
    


def prepare_train_val(dataset, method, data, img_size, normalization, augmentations, batch_size, train_idx, val_idx):
    """
    Prepare training and validation dataset.
    """
    if dataset == 'uadfv':
        train_dataset = datasets.UADFVDataset(
            data.iloc[train_idx], img_size, method=method, normalization=normalization, augmentations=augmentations)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = datasets.UADFVDataset(
            data.iloc[val_idx], img_size, method=method, normalization=normalization, augmentations=None)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

    elif dataset == 'celebdf':
        train_dataset = datasets.CelebDFDataset(
            data.iloc[train_idx], img_size, method=method, normalization=normalization, augmentations=augmentations)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dataset = datasets.CelebDFDataset(
            data.iloc[val_idx], img_size, method=method, normalization=normalization, augmentations=None)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
    elif dataset == 'dftimit_hq':
        train_dataset = datasets.DFTIMITHQDataset(
            data.iloc[train_idx], img_size, method=method, normalization=normalization, augmentations=augmentations)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dataset = datasets.DFTIMITHQDataset(
            data.iloc[val_idx], img_size, method=method, normalization=normalization, augmentations=None)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_dataset, train_loader, val_dataset, val_loader
