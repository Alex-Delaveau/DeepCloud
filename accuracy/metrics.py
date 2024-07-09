import numpy as np
import tensorflow as tf

def clean_mask(mask):
    """Replace NaNs and Infs in the mask with zeros."""
    mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    return mask.astype(np.float64)

def dice_coefficient(gt_mask, pred_mask):
    gt_mask, pred_mask = clean_mask(gt_mask), clean_mask(pred_mask)
    intersection = np.logical_and(gt_mask, pred_mask)
    gt_sum = gt_mask.sum()
    pred_sum = pred_mask.sum()
    
    if gt_sum == 0 and pred_sum == 0:
        return 1.0  # Both masks are empty, consider it a perfect match
    elif gt_sum == 0 or pred_sum == 0:
        return 0.0  # One mask is empty while the other isn't
    
    return 2. * intersection.sum() / (gt_sum + pred_sum)



def jaccard_coefficient(y_true, y_pred):
    
    smooth = 0.0000001
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    
    jacc = (intersection + smooth) / (union + smooth)
    return 1 - jacc