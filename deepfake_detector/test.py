# author: Christopher Otto
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from facedetector.retinaface import df_retinaface

from sklearn.metrics import confusion_matrix
from albumentations import Resize
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def vid_inference(model, video_frames, label, img_size, normalization):
    # model evaluation mode
    model.cuda()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_func = nn.BCEWithLogitsLoss()
    #label = torch.from_numpy(label).to(device)
    # get prediction for each frame from vid
    avg_vid_loss = []
    avg_preds = []
    avg_loss = []
    for frame in video_frames:   
        # turn image to rgb color
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # resize to DNN input size
        resize = Resize(width=img_size,height=img_size)
        frame = resize(image=frame)['image']
        frame = torch.tensor(frame).to(device)
        #forward pass of inputs and turn on gradient computation during train
        with torch.no_grad():
            # predict for frame
            # channels first
            frame = frame.permute(2,0,1)
            # turn dtype from uint8 to float and normalize to [0,1] range
            frame = frame.float() / 255.0
            # normalize by imagenet stats
            if normalization == 'xception':
                transform = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            elif normalization == "imagenet":
                transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            frame = transform(frame)
            # add batch dimension and input into model to get logits
            predictions = model(frame.unsqueeze(0))
            
            # get probabilitiy for frame from logits
            preds = torch.sigmoid(predictions)
            avg_preds.append(preds.cpu().numpy())
            # calculate loss from logits
            loss = loss_func(predictions.squeeze(1), torch.tensor(label).unsqueeze(0).type_as(predictions))
            avg_loss.append(loss.cpu().numpy())
    # return the prediction for the video as average of the predictions over all frames
    return np.mean(avg_preds), np.mean(avg_loss)


def inference(model, test_df, img_size, normalization):
    # path to save extracted faces
    #video_path = os.path.join("/home/jupyter/fake_videos/real/test/")
    #save_path = os.path.join("/home/jupyter/0margin/test/real/")
    # detect faces, add margin, crop, upsample to same size, save to images
    running_loss = 0.0
    running_corrects = 0.0
    running_false = 0.0
    running_auc = []
    running_ap = []
    labs = []
    prds = []
    # load retinaface face detector
    net, cfg = df_retinaface.detect()
    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        video = row.loc['video']
        label = row.loc['label']
        vid = os.path.join(video)
        #inference (no saving of images inbetween to make it faster)
        # detect faces, add margin, crop, upsample to same size, save to images
        faces = df_retinaface.detect_faces(net, vid, cfg, num_frames=20)
        # save frames to images
        #try:
        vid_frames = df_retinaface.extract_frames(faces, video,save_to=None, face_margin=0,num_frames=20, test=True)
        #except:
            #print("Error: Video frames.")
        # inference for each frame
        vid_pred, vid_loss = vid_inference(model,vid_frames, label, img_size, normalization)
        labs.append(label)
        prds.append(vid_pred)
        running_loss += vid_loss
        #calc accuracy; thresh 0.5
        running_corrects += np.sum(np.round(vid_pred) == label) 
        running_false += np.sum(np.round(vid_pred) != label) 
        
    auc = round(roc_auc_score(labs, prds),5)
    ap = round(average_precision_score(labs, prds),5)
    loss = round(running_loss / len(test_df), 5)
    acc = round(running_corrects / len(test_df),5)
    print("Benchmark results:")
    print("Confusion matrix:")
    print(confusion_matrix(labs,np.round(prds)))
    tn, fp, fn, tp = confusion_matrix(labs,np.round(prds)).ravel()
    print(f"Loss: {loss}")
    print(f"Acc: {acc}")
    print(f"AUC: {auc}")
    print(f"AP: {auc}")
    print(f"Detected \033[1m {tp}\033[0m true deepfake videos and correctly classified \033[1m {tn}\033[0m real videos.")
    print(f"Mistook \033[1m {fp}\033[0m real videos for deepfakes and \033[1m {fn}\033[0m deepfakes went by undetected by the method.")
    if fn == 0 and fp == 0:
        print("Wow! A perfect classifier!")
    return auc,ap,loss,acc