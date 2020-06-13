import argparse
import os
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

import cv2
from facedetector.retinaface.data import cfg_mnet, cfg_re50
from facedetector.retinaface.layers.functions.prior_box import PriorBox
from facedetector.retinaface.models.retinaface import RetinaFace
from facedetector.retinaface.utils.box_utils import decode, decode_landm
from facedetector.retinaface.utils.nms.py_cpu_nms import py_cpu_nms
from tqdm import tqdm

# from https://github.com/biubug6/Pytorch_Retinaface
# MIT License


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # print('Loading pretrained face detector model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def my_detector(cfg_mnet, cfg_re50, inp, model_path, cpu=False):
    """Create the RetinaFace Detector. """
    torch.set_grad_enabled(False)
    cfg = None
    cfg_mnet['pretrain'] = False
    if inp == "mobile0.25":
        cfg = cfg_mnet
    elif inp == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, model_path, cpu)
    net.eval()
    # print('Finished loading RetinaFace face detector!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if cpu else "cuda")
    net = net.to(device)
    return net, cfg


def detect_face_from_frame(frame, net, cfg):
    """Face detection in single frame."""
    resize = 1
    top_k = 1
    keep_top_k = 1
    confidence_threshold = 0.9
    nms_threshold = 0.4
    save_image = True
    vis_thres = 0.6
    # run on gpu
    device = "cuda"
    img_raw = frame
    img = np.float32(img_raw)
    im_height, im_width, _ = img.shape
    scale = torch.Tensor(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    tic = time.time()
    loc, conf, landms = net(img)  # forward pass

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]

    landms = landms[order]
    scores = scores[order]
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]

    dets = np.concatenate((dets, landms), axis=1)
    # image, box,score,landmarks pairs
    return [img_raw, dets]


def detect_faces(net, video, cfg, num_frames):
    """
    Detect faces in video frames.

    # parts from https://www.kaggle.com/unkownhihi/dfdc-lrcn-inference
    # APACHE LICENSE, VERSION 2.0
    # adapted by Christopher Otto

    """
    cap = cv2.VideoCapture(video)
    # get frames in video
    frame_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # choose num frames (20) equally spaced from video
    search_frames = np.linspace(
        0, frame_len, num_frames, endpoint=False, dtype=np.int64)
    frames = []
    count = 0
    for idx in range(frame_len):
        # grab frame
        success = cap.grab()
        if not success:
            continue
        # retrieve frame if in search frames
        if idx >= search_frames[count]:
            # if successful retrieve the frame
            _, frame = cap.retrieve()
            if frame is None:
                continue
            frame = detect_face_from_frame(frame, net, cfg)
            frames.append(frame)
            count += 1
            # break if num_frames extracted
            if count == num_frames:
                break
    return frames


def extract_frames(faces, video, save_to, face_margin, num_frames, test=False):
    """
    Extract frames from video and save image with frames.

    # parts from https://github.com/biubug6/Pytorch_Retinaface

    # adapted by Christopher Otto
    """
    # threshold for confidence in face that is required to confirm it as face
    thresh = 0.6
    imgs_result = []
    max_height = 0
    max_width = 0
    for idx, face in enumerate(faces):
        img_raw = face[0]
        for b in face[1]:
            if b[4] < thresh:
                continue
            b = list(map(int, b))
            # add 100*margin% around the face as recommended here:
            # https://www.kaggle.com/c/deepfake-detection-challenge/discussion/140236
            # and here https://www.kaggle.com/c/deepfake-detection-challenge/discussion/145721
            if face_margin > 0.0:
                old_length_height = b[3] - b[1]
                old_length_width = b[2] - b[0]
                new_length_height = (b[3] - b[1])*(1+face_margin)
                new_length_width = (b[2] - b[0])*(1+face_margin)
                pixel_add_sub_height = int(
                    (new_length_height - old_length_height) / 2)
                pixel_add_sub_width = int(
                    (new_length_width - old_length_width) / 2)

                b = [b[0]-pixel_add_sub_width, b[1]-pixel_add_sub_height,
                     b[2]+pixel_add_sub_width, b[3]+pixel_add_sub_height]
            else:
                b = [b[0], b[1],
                     b[2], b[3]]
        try:
            img_raw = img_raw[b[1]:b[3], b[0]:b[2]]
        except:
            print(f"No face detected in frame {idx + 1} from video: {video}.")
            continue
        imgs_result.append(img_raw)
        # to resize images of same video to max height/width
        if img_raw.shape[0] > max_height:
            max_height = img_raw.shape[0]
        if img_raw.shape[1] > max_width:
            max_width = img_raw.shape[1]

    # resize images of same video to max height/width of img from vid as recommended here:
    # https://www.kaggle.com/c/deepfake-detection-challenge/discussion/140236
    imgs_same_size = []
    for img in imgs_result:
        # bilinear interpolation for upsampling
        try:
            img = cv2.resize(img, (max_height, max_width))
            imgs_same_size.append(img)
        except:
            print("Zero-sized image.")
    if test:
        return imgs_same_size
    # only save if specified number of frames available
    if len(imgs_same_size) <= num_frames:
        for idx, i in enumerate(imgs_same_size):
            name = save_to + video[:-4] + '_' + str(idx) + ".jpg"
            cv2.imwrite(name, i)
    # return sequence length for metadata
    return len(imgs_same_size)


def load_face_detector(backbone="resnet50", backbone_path=os.getcwd() + "/deepfake_detector/facedetector/retinaface/Resnet50_Final.pth"):
    """
    Detect faces from video frames.
    # Arguments:
        backbone: Backbone of the face detector.
        backbone_path: Weights for face detector model.

    # Implementation: Christopher Otto
    """
    detector, config = my_detector(
        cfg_mnet, cfg_re50, inp=backbone, model_path=backbone_path, cpu=False)
    return detector, config
