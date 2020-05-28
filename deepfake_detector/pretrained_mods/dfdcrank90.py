import timm
import torch
import torch.nn as nn
import torchvision.models as models
import resnetlstm
import efficientnetb1lstm
import xception

class Rank90DFDC(nn.Module):
    """
    Implementation of the Rank 90 private leaderboard solution of the DeepfakeDetection Challenge:

    https://www.kaggle.com/catochris

    https://www.kaggle.com/c/deepfake-detection-challenge/leaderboard 

    Ensemble:
    1xEfficientNetB1
    2xXception (different seeds)
    
    Can only be used for inference, as models were trained individually.
    
    # Implementation by: Christopher Otto

    """
    def __init__(self):
        super(Rank90DFDC, self).__init__()
        