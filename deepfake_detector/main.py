from dfdetector import DFDetector
import os
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import _ranking
from sklearn.utils import multiclass
from sklearn.metrics._plot import precision_recall_curve
import metrics
import matplotlib.pyplot as plt
from pretrained_mods.mesonet import MesoInception4
from utils import vidtimit_setup_real_videos

if __name__ == "__main__":
    used, result = DFDetector.detect_single(video_path="C:/Users/Chris/Desktop/fake_videos/fake/0007_fake.mp4", image_path=None, label=None, method="xception_uadfv")
    #vidtimit_setup_real_videos("C:/Users/Chris/Desktop/vidtimitreal/vidtimitreal")
    #DFDetector.detect_single(video_path=None,image_path="C:/Users/Chris/Desktop/fake_videos/train_imgs/fake/0000_fake_0.jpg", label=None, method="xception_uadfv")
    #benchmark_result = DFDetector.benchmark(dataset="uadfv",data_path="C:/Users/Chris/Desktop/fake_videos", method="xception")
    #_ = DFDetector.detect_single(video_path=None,image_path="C:/Users/Chris/Desktop/fake_videos/train_imgs/real/0000_0.jpg", label=0, method="xception_uadfv")
    
    # result = metrics.weighted_precision(y_true, y_pred)
    # print(result)
    # model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
    #     dataset="dftimit_hq", data_path="C:/Users/Chris/Desktop/DeepfakeTIMIT", method="xception",
    #     img_save_path="C:/Users/Chris/Desktop/DeepfakeTIMIT", epochs=1, batch_size=32, lr=0.0001, 
    #     folds=1, augmentation_strength="weak", fulltrain=False, faces_available=True, face_margin=0.0, seed=24)

    # model, average_auc, average_ap, average_acc, average_loss = DFDetector.train_method(
    # dataset="celebdf", data_path='C:/Users/Chris/Desktop/Celeb-DF-v2', method="xception",
    # img_save_path="C:/Users/Chris/Desktop/Celeb-DF-v2",epochs=1, batch_size=32, lr=0.001, faces_available=False, face_margin=0)


# img save path and data path are the same -> redundancy can be removed!


#result = DFDetector.detect(video=video_file, method="xception", heatmap=False)

   

# ->compare: whether to compare with the results of other methods that were precomputed by myself
