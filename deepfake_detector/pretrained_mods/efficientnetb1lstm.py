# has to be timm==0.1.26
import timm
import torch
import torch.nn as nn
import torchvision.models as models

class EfficientNetB1LSTM(nn.Module):
    """
    Implementation of a EfficientNetB1 + LSTM that was one part of the DeepfakeDetection Challenge
    Rank 90 private leaderboard solution https://www.kaggle.com/c/deepfake-detection-challenge/leaderboard 

    # Architecture inspired by https://www.kaggle.com/unkownhihi/dfdc-lrcn-inference
    
    To make it comparable with ResNet50 + LSTM it uses the same fully connected layers and 
    also uses 512 hidden units as it was described in the paper
    DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection (https://arxiv.org/abs/2001.03024)

    # parts from https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/2
    # and from https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/6 

    # adapted by: Christopher Otto

    Arguments:
        hidden_size = 512  # as described in the Deeperforensics-1.0 paper
    """
    def __init__(self, input_size=128, hidden_size=512, num_layers=2, num_classes=1):
        super(EfficientNetB1LSTM, self).__init__()
        self.b1 =timm.create_model('efficientnet_b1', pretrained=True)
        # delete b1 fc layer
        self.b1 = nn.Sequential(*list(self.b1.children())[:-2],
                   nn.Conv2d(1280, 128, 1, bias=False),
                   nn.BatchNorm2d(128),
                   timm.models.efficientnet.Swish(),
                   nn.AdaptiveAvgPool2d((1, 1)))
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        # another fc layer, because it seems to improve performance
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # [32, 20, 3, 224, 224]
        batch_size, num_frames, channels, height, width = x.size()
        # combine batch and frame dimensions for 2d cnn
        c_in = x.reshape(batch_size * num_frames, channels, height, width)
        c_out = self.b1(c_in)
        # separate batch and frame dimensions for lstm 
        c_out = c_out.view(batch_size, num_frames, -1)
        r_out, _ = self.lstm(c_out)
        # get last hidden state
        out = self.relu(self.fc1(r_out[:, -1, :]))
        result = self.fc2(out)
        return result