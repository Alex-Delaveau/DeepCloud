import numpy as np
from scipy.spatial.distance import directed_hausdorff
from skimage import measure
import os
from PIL import Image
import pandas as pd

def boundary_f1_score(gt_mask, pred_mask, tolerance=2):
    def find_boundaries(mask, tolerance):
        boundaries = np.zeros_like(mask, dtype=bool)
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            for coord in contour:
                y, x = coord
                boundaries[int(y), int(x)] = True
        return boundaries

    gt_boundaries = find_boundaries(gt_mask, tolerance)
    pred_boundaries = find_boundaries(pred_mask, tolerance)
    
    true_positive = np.sum(gt_boundaries & pred_boundaries)
    false_positive = np.sum(~gt_boundaries & pred_boundaries)
    false_negative = np.sum(gt_boundaries & ~pred_boundaries)
    
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def hausdorff_distance(gt_mask, pred_mask):
    gt_points = np.column_stack(np.where(gt_mask > 0))
    pred_points = np.column_stack(np.where(pred_mask > 0))
    
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.inf  # Infinite distance if one of the masks is empty
    
    forward_hausdorff = directed_hausdorff(gt_points, pred_points)[0]
    backward_hausdorff = directed_hausdorff(pred_points, gt_points)[0]
    
    return max(forward_hausdorff, backward_hausdorff)

def relaxed_iou_score(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    
    return iou_score

def normalized_cross_correlation(gt_mask, pred_mask):
    gt_mean = np.mean(gt_mask)
    pred_mean = np.mean(pred_mask)
    
    numerator = np.sum((gt_mask - gt_mean) * (pred_mask - pred_mean))
    denominator = np.sqrt(np.sum((gt_mask - gt_mean) ** 2) * np.sum((pred_mask - pred_mean) ** 2))
    
    return numerator / denominator

def run_accuracy_x(gt_mask, pred_mask_binary):
    # Calculate metrics
    # f1 = boundary_f1_score(gt_mask, pred_mask_binary)
    # hausdorff = hausdorff_distance(gt_mask, pred_mask_binary)
    iou = relaxed_iou_score(gt_mask, pred_mask_binary)
    # ncc = normalized_cross_correlation(gt_mask, pred_mask_binary)

    return iou

def run_accuracy(gt_path, prediction_path):
    # for each file in gt_path and prediction_path (same file name)
    # calculate the accuracy metrics
    # save the results in a csv file and compute the mean and std of each metric
    results = []

    for filename in os.listdir(gt_path):
        gt_file = os.path.join(gt_path, filename)
        pred_file = os.path.join(prediction_path, filename)
        if not os.path.exists(pred_file):
            print(f"Prediction file {pred_file} does not exist. Skipping.")
            continue

        gt_mask = np.array(Image.open(gt_file))
        pred_mask = np.array(Image.open(pred_file))

        metrics = run_accuracy_x(gt_mask, pred_mask)

        # metrics_data = {
        #     "filename": filename,
        #     "f1": metrics[0],
        #     "hausdorff": metrics[1],
        #     "iou": metrics[2],
        #     "ncc": metrics[3]
        # }
        print(f"Metrics for {filename}: {metrics}")
        results.append(metrics)

    df = pd.DataFrame(results)
    df.to_csv("accuracy_results.csv", index=False)
    mean_metrics = df.mean()
    std_metrics = df.std()

    print(f"Mean Metrics:\n{mean_metrics}")
    print(f"\nStandard Deviation of Metrics:\n{std_metrics}")

    return df, mean_metrics, std_metrics

