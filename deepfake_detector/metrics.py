from sklearn.metrics import multilabel_confusion_matrix
import math
import numpy as np
from sklearn.metrics import _ranking
from sklearn.utils import multiclass
from sklearn.metrics._plot import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

def prec_rec(y_true, y_pred, method, alpha=100, plot = False):
    """
    Calculates the weighted precision metric at recall levels 0.1, 0.5 and 0.9 as proposed in:
    The Deepfake Detection Challenge (DFDC) Preview Dataset (https://arxiv.org/abs/1910.08854)

    Parts from sklearn.metrics precision_recall_curve adapted by: Christopher Otto

    alpha = 100 as suggested in the paper.
    """
    fps, tps, thresholds = _ranking._binary_clf_curve(
        y_true, y_pred, pos_label=None, sample_weight=None)

    weighted_precision = tps / (tps + alpha*fps)
    weighted_precision[np.isnan(weighted_precision)] = 0
    # take log of weighted precision similar to The Deepfake Detection Challenge (DFDC) Preview Dataset (https://arxiv.org/abs/1910.08854)
    weighted_precision = [math.log(entry) if entry > 0 else 0 for entry in weighted_precision]
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    prec, rec, thresh = np.r_[weighted_precision[sl],
                              1], np.r_[recall[sl], 0], thresholds[sl]

    # first precision entry for recall level at 0.9
    threshold_index_point_nine = len(
        [entry for entry in rec if entry >= 0.9])-1
    weighted_precision_at_point_nine_rec = prec[threshold_index_point_nine]
    # first precision entry for recall level at 0.5
    threshold_index_point_five = len(
        [entry for entry in rec if entry >= 0.5])-1
    weighted_precision_at_point_five_rec = prec[threshold_index_point_five]
    # first precision entry for recall level at 0.1
    threshold_index_point_one = len([entry for entry in rec if entry >= 0.1])-1
    weigthed_precision_at_point_one_rec = prec[threshold_index_point_one] 
    
    if plot:
        average_precision = average_precision_score(y_true, y_pred)
        viz = precision_recall_curve.PrecisionRecallDisplay(
            precision=prec, recall=rec,
            average_precision=average_precision, estimator_name=f"{method}"
        )
        disp = viz.plot(ax=None, name=f"Method: {method}")
        disp.ax_.set_title('Weighted Precision-Recall curve')
        plt.xlabel('Weighted Precision (Cost)')
        plt.ylabel('Recall')
        plt.savefig('w_prec_recall_curve.png')
        plt.show()

    return weigthed_precision_at_point_one_rec,weighted_precision_at_point_five_rec,weighted_precision_at_point_nine_rec