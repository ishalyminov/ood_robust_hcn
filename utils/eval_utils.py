import numpy as np
from sklearn.metrics import roc_curve


def eer(y_true, y_scores):
    fpr, tpr, threshold = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer
