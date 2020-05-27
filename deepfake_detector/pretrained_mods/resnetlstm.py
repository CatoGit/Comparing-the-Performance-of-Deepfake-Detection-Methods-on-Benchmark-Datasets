import torch
import torch.nn as nn
import torchvision.models as models

class ResNetLSTM(nn.Module):
    """
    Implementation of a Resnet50 + LSTM with 512 hidden units as it was described in the paper
    DeeperForensics-1.0: A Large-Scale Dataset for Real-World Face Forgery Detection (https://arxiv.org/abs/2001.03024)

    # parts from https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/2
    # and from https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/6 

    # adapted by: Christopher Otto

    Arguments:
        hidden_size = 512  # as described in the Deeperforensics-1.0 paper
    """

    def __init__(self, input_size, num_layers, num_classes, hidden_size=512):
        super(ResNetLSTM, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # delete resnet fc layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2],
                                    # add another conv layer to keep input size to LSTM smaller
                                    nn.Conv2d(2048, 128, 1, bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool2d((1, 1)))
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()
        # another fc layer, because it seems to improve performance
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # [32, 20, 3, 224, 224]
        batch_size, num_frames, channels, height, width = x.size()
        c_in = x.reshape(batch_size * num_frames, channels, height, width)
        c_out = self.resnet(c_in)
        c_out = c_out.view(batch_size, num_frames, -1)
        r_out, _ = self.lstm(c_out)
        # get last hidden state
        out = self.relu(self.fc1(r_out[:, -1, :]))
        result = self.fc2(out)
        return result
