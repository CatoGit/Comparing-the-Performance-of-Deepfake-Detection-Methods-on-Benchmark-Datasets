from albumentations import Compose, HorizontalFlip, Resize, ImageCompression, GaussNoise, GaussianBlur
from albumentations import PadIfNeeded, OneOf, RandomBrightnessContrast, FancyPCA, HueSaturationValue
from albumentations import ToGray, ShiftScaleRotate


def label_data(dataset_path="uadfv/", test_data=False):
    """
    Label the data.
    # Arguments:
        dataset_path: path to data
        test_data: binary choice that indicates whether data is for testing or not.
    # Implementation: Christopher Otto
    """
    # structure data from folder in data frame for loading
    video_path_real = os.path.join(dataset_path + "real/")
    video_path_fake = os.path.join(dataset_path + "fake/")

    # add labels to videos
    data_list = []
    for _, _, videos in os.walk(video_path_real):
        for video in tqdm(videos):
            # label 0 for real video
            data_list.append({'label': 0, 'image': video})

    for _, _, videos in os.walk(video_path_fake):
        for video in tqdm(videos):
            # label 1 for deepfake video
            data_list.append({'label': 1, 'image': video})

    # put data into dataframe
    df = pd.DataFrame(data=data_list)
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


def train(folds=5, epochs=0, fulltrain=False):
    """
    Train DNN for a number of epochs.

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

    for fold in range(folds):
        print(f"Fold: {fold}")

        best_acc = 0.0
        best_loss = 10
        current_acc = 0.0
        current_loss = 10.0
        best_auc = 0.0
        best_ap = 0.0
        # get train and val indices
        if fulltrain == False:
            train_idx, val_idx = shuffeled_cross_val(fold)

        # prepare training and validation data
        if fulltrain == True:
            train_dataset = UADFVDataset(
                img_dir, df, img_size, augmentations_weak)
            train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=True)
        else:
            val_dataset = UADFVDataset(
                img_dir, df.iloc[val_idx], img_size, augmentations=None)
            train_dataset = UADFVDataset(
                img_dir, df.iloc[train_idx], img_size, augmentations_weak)
            train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # load the xception model
        model = imagenet_pretrained_xception()

        if fold == 0:
            best_model = copy.deepcopy(model.state_dict())

        # put model on gpu
        model = model.cuda()
        # binary cross-entropy loss
        loss_func = nn.BCEWithLogitsLoss()
        lr = 0.001
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
                running_auc = []
                running_ap = []
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
                                predictions.squeeze(), labels.type_as(predictions))

                            if phase == "train":
                                # backpropagate gradients
                                loss.backward()
                                # update parameters
                                optimizer.step()

                        running_loss += loss.item() * imgs.size(0)
                        # calc accuracy
                        running_corrects += torch.sum(thresh_preds ==
                                                      labels.unsqueeze(1))
                        running_auc.append(roc_auc_score(
                            labels.detach().cpu().numpy(), sig.detach().cpu().numpy()))
                        running_ap.append(average_precision_score(
                            labels.detach().cpu().numpy(), sig.detach().cpu().numpy()))
                    if phase == 'train':
                        # update lr
                        scheduler.step()
                    epoch_loss = running_loss / len(train_dataset)
                    epoch_acc = running_corrects / len(train_dataset)
                    epoch_auc = np.mean(running_auc)
                    epoch_ap = np.mean(running_ap)
                    print(
                        f"{phase} Loss: {epoch_loss}, Acc: {epoch_acc}, AUC: {epoch_auc}, AP: {epoch_ap}")
                    print(e)
                    if fulltrain == True and e+1 == epochs:
                        # save model if epochs reached
                        torch.save(
                            model.state_dict(), f'/home/jupyter/xception_best_fulltrain_UADFV.pth')

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
                                predictions.squeeze(), labels.type_as(predictions))

                        running_loss += loss.item() * imgs.size(0)
                        # calc accuracy
                        running_corrects += torch.sum(thresh_preds ==
                                                      labels.unsqueeze(1))

                        running_auc.append(roc_auc_score(
                            labels.detach().cpu().numpy(), sig.detach().cpu().numpy()))
                        running_ap.append(average_precision_score(
                            labels.detach().cpu().numpy(), sig.detach().cpu().numpy()))

                    epoch_loss = running_loss / len(val_dataset)
                    epoch_acc = running_corrects / len(val_dataset)
                    epoch_auc = np.mean(running_auc)
                    epoch_ap = np.mean(running_ap)
                    print(
                        f"{phase} Loss: {epoch_loss}, Acc: {epoch_acc}, AUC: {epoch_auc}, AP: {epoch_ap}")

                    # save model if loss better than best loss
                    if epoch_auc > best_auc:
                        best_auc = epoch_auc
                        best_model = copy.deepcopy(model.state_dict())
                        # save best model
                        torch.save(
                            model.state_dict(), f'/home/jupyter/xception_best_auc_model_fold{fold}.pth')
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                    if epoch_ap > best_ap:
                        best_ap = epoch_ap

        average_auc.append(best_auc)
        average_ap.append(best_ap)
        average_acc.append(best_acc)
        average_loss.append(best_loss)
    # load best model params
    # model.load_state_dict(best_model)
    return model, average_auc, average_ap, average_acc, average_loss

from facedetector import retinaface

if __name__ == "__main__":
    retinaface.detect_face("C:/Users/Chris/Desktop/fake_videos/", saveimgs_path="C:/Users/Chris/Desktop/fake_videos/saved/")