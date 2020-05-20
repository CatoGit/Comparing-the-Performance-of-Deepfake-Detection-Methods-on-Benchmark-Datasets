from sklearn.metrics import multilabel_confusion_matrix
import math
import numpy as np


def weighted_precision(y_true, y_pred, alpha=100, base='e'):
    """
    Calculates the weighted precision metric at recall levels 0.1, 0.5 and 0.9 as proposed in:
    The Deepfake Detection Challenge (DFDC) Preview Dataset (https://arxiv.org/abs/1910.08854)
    """
    # compute confusion matrix
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    # get true positives
    tp = np.sum(mcm[:, 1, 1])
    # get false positives
    fp = np.sum(mcm[:, 0, 1])
    # get false negatives for recall
    fn = np.sum(mcm[:, 1, 0])
    recall = tp / (tp + fn)
    print(recall)
    # weight false positives to approximate organic traffic precision
    fp = fp*alpha
    weighted_precision = tp / (tp + fp)
    if base == 'e':
        weighted_precision = round(math.log(weighted_precision), 5)
    elif base == 10:
        weighted_precision = round(math.log10(weighted_precision), 5)
    elif base == 2:
        weighted_precision = round(math.log2(weighted_precision), 5)
    return weighted_precision
